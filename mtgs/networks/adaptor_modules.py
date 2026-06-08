# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, use_kv_bias=False, use_q_bias=False):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=use_kv_bias)
        self.q = nn.Linear(dim, dim, bias=use_q_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, gaze_token, img_tokens):
        B, N, C = img_tokens.shape
        _, NP, _ = gaze_token.shape

        kv = (
            self.kv(img_tokens)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # (b, nh, n, dh)

        # (b, np, d) >> (b, nh, np, dh) where np=num of people
        q = (
            self.q(gaze_token)
            .reshape(B, NP, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b, nh, np, n)
        attn = attn.softmax(dim=-1)

        o = (attn @ v).transpose(1, 2).reshape(B, NP, C)  # (b, np, d)
        o = self.proj(o)
        return o


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop_rate=0.0
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = MLP(
                in_features=dim, hidden_features=int(dim * cffn_ratio), drop_rate=drop
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, feat):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query)))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
        with_cp=False,
        same_norm=False,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        if same_norm:
            self.feature_norm = self.query_norm
        else:
            self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            num_heads=num_heads,
            init_values=init_values,
            norm_layer=norm_layer,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            num_heads=num_heads,
            norm_layer=norm_layer,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )

        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, num_valid_people):
        x = self.injector(query=x, feat=c)
        for idx, blk in enumerate(blocks):
            x = blk(x)
        c = self.extractor(query=c, feat=x)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, feat=x)

        return x, c

    def forward_extract_only(self, x, c):
        """scene→people cross-attn only (extractor step, no injection or ViT)."""
        c = self.extractor(query=c, feat=x)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, feat=x)
        return x, c

    def forward_inject_vit(self, x, c, blocks):
        """people→scene cross-attn + ViT blocks only (no extraction)."""
        x = self.injector(query=x, feat=c)
        for blk in blocks:
            x = blk(x)
        return x, c


class SocialGraphBlock(nn.Module):
    """
    Social interaction graph block replacing I^b_pp (Social Encoder).

    Supports three aggregation modes (controlled by `aggr`):

      "outgoing" (default):
          msg_i = Σ_{i→j} α[i→j] · W_msg(h_j)  +  α_null[i] · W_msg(null_node)
          softmax over destinations j (dim=-1).
          i collects info from nodes it is looking at.
          Null node: "i looks at no person" → null feature absorbed into msg_i.

      "ingoing":
          msg_i = Σ_{j→i} α[j→i] · W_msg(h_j)
          softmax over sources j (dim=1).
          i collects info from nodes looking at it.
          Null node disabled (null has no meaningful gaze direction as a source).

      "both":
          msg_i = W_out(msg_out_i) + W_in(msg_in_i)
          outgoing part includes null; ingoing part does not.

    Geometric LAH cosine prior is injected into attention weights on iteration 0 only.
    Social prediction (LAH/SA) is handled by the shared pair-wise decoder downstream.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_channels: int = 96,
        heads: int = 8,           # unused; kept for API compatibility
        num_layers: int = 2,      # internal message-passing iterations
        use_null_node: bool = True,
        use_gaze_prior: bool = True,
        prior_weight: float = 0.5,
        layer_idx: int = 0,       # unused; kept for API compatibility
        aggr: str = "outgoing",   # "outgoing" | "ingoing" | "both"
        use_sa_prior: bool = True,
        sa_prior_weight: float = 0.5,
    ):
        super().__init__()
        assert aggr in ("outgoing", "ingoing", "both"), f"Unknown aggr: {aggr!r}"
        self.num_layers     = num_layers
        self.use_gaze_prior = use_gaze_prior
        self.aggr           = aggr

        # Null node is only meaningful in outgoing direction.
        self.use_null_node = use_null_node and (aggr in ("outgoing", "both"))

        # Learnable prior weight for attention routing only (single scalar).
        self.prior_w_attn = nn.Parameter(torch.tensor(prior_weight))
        # Per-iteration decay weights for gaze prior injection (softmax over num_layers)
        self.prior_decay_logits = nn.Parameter(torch.zeros(num_layers))

        # SA prior: gaze direction cosine similarity (symmetric).
        self.use_sa_prior = use_sa_prior and use_gaze_prior
        if self.use_sa_prior:
            self.prior_w_sa = nn.Parameter(torch.tensor(sa_prior_weight))

        # ── Attention scoring MLP (directed edge i→j) ───────────────────────
        self.mlp_dir = MLP(token_dim * 2, hidden_channels, 1)
        if self.use_null_node:
            # Null_in : in-frame non-person targets (looks at scene object)
            # Null_out: out-of-frame gaze (supervised by inout label)
            self.null_in_node  = nn.Parameter(torch.zeros(token_dim))
            self.null_out_node = nn.Parameter(torch.zeros(token_dim))
            self.mlp_null_in   = MLP(token_dim * 2, hidden_channels, 1)
            self.mlp_null_out  = MLP(token_dim * 2, hidden_channels, 1)

        # ── Message passing & node update ────────────────────────────────────
        self.W_msg       = nn.Linear(token_dim, token_dim, bias=False)
        if aggr == "both":
            # separate projections to combine outgoing and ingoing messages
            self.W_combine_out = nn.Linear(token_dim, token_dim, bias=False)
            self.W_combine_in  = nn.Linear(token_dim, token_dim, bias=False)
        self.update_proj = nn.Linear(token_dim * 2, token_dim)
        self.W_gate      = nn.Linear(token_dim, token_dim)
        self.norm        = nn.LayerNorm(token_dim)

        self._edge_cache: dict = {}

    @staticmethod
    def _build_edges(nv: int, device: torch.device):
        """Directed edges in GT label order: [(s,d) for s in range(nv) for d in range(nv) if s!=d]."""
        src = torch.tensor(
            [s for s in range(nv) for d in range(nv) if s != d],
            dtype=torch.long, device=device,
        )
        dst = torch.tensor(
            [d for s in range(nv) for d in range(nv) if s != d],
            dtype=torch.long, device=device,
        )
        return src, dst

    def _get_edge_cache(self, N: int, device: torch.device) -> dict:
        if N not in self._edge_cache:
            src_N, dst_N = self._build_edges(N, device)
            self._edge_cache[N] = {"src_N": src_N, "dst_N": dst_N}
        return self._edge_cache[N]

    def forward(
        self,
        person_tokens,
        num_valid_people,
        gaze_vecs=None,
        head_bboxes=None,
        readout=False,   # unused; kept for call-site compatibility
    ):
        """
        Args:
            person_tokens:    (B, N, D)
            num_valid_people: (B,) int
            gaze_vecs:        (B, N, 2) unit gaze direction
            head_bboxes:      (B, N, 4) normalized [x1,y1,x2,y2]

        Returns:
            tokens_out:     (B, N, D) updated node features.
            alpha_null_in:  (B, N) or None — attention weight to Null_in (last iter).
            alpha_null_out: (B, N) or None — attention weight to Null_out (last iter).
        """
        B, N, D = person_tokens.shape
        device  = person_tokens.device
        dtype   = person_tokens.dtype

        cache  = self._get_edge_cache(N, device)
        src_N  = cache["src_N"]   # (E,)  E = N*(N-1)
        dst_N  = cache["dst_N"]   # (E,)

        # Valid nodes occupy the BACK slots [N-nv .. N-1]; front slots are padding.
        node_valid = (
            torch.arange(N, device=device).unsqueeze(0) >= (N - num_valid_people.unsqueeze(1))
        )  # (B, N)
        pair_valid = node_valid.unsqueeze(2) & node_valid.unsqueeze(1)   # (B, N, N)
        diag_mask  = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)

        # ── Pre-compute LAH cosine prior (iteration-independent) ─────────────
        lah_prior = None
        if self.use_gaze_prior and gaze_vecs is not None and head_bboxes is not None:
            centers = (head_bboxes[..., :2] + head_bboxes[..., 2:]) / 2   # (B, N, 2)
            dir_ij    = F.normalize(centers[:, dst_N] - centers[:, src_N], dim=-1)
            lah_prior = (gaze_vecs[:, src_N] * dir_ij).sum(-1)            # (B, E)

        # ── Pre-compute SA gaze cosine prior (gaze_i · gaze_j, symmetric) ──
        sa_prior = None
        if self.use_sa_prior and gaze_vecs is not None:
            sa_prior = (gaze_vecs[:, src_N] * gaze_vecs[:, dst_N]).sum(-1)  # (B, E)

        # Pre-build prior matrices (iteration-independent) and per-iteration decay weights
        lah_prior_mat = None
        if self.use_gaze_prior and lah_prior is not None:
            lah_prior_mat = torch.zeros(B, N, N, device=device, dtype=dtype)
            lah_prior_mat[:, src_N, dst_N] = lah_prior.to(dtype)
        sa_prior_mat = None
        if self.use_sa_prior and sa_prior is not None:
            sa_prior_mat = torch.zeros(B, N, N, device=device, dtype=dtype)
            sa_prior_mat[:, src_N, dst_N] = sa_prior.to(dtype)
        decay_w = torch.softmax(self.prior_decay_logits, dim=0)  # (num_layers,)

        h = person_tokens.clone()
        _alpha_null_in  = None
        _alpha_null_out = None

        for iter_idx in range(self.num_layers):
            # ── Directed edge attention scores ───────────────────────────────
            h_i = h.unsqueeze(2).expand(B, N, N, D)
            h_j = h.unsqueeze(1).expand(B, N, N, D)

            e_dir_mat = self.mlp_dir(
                torch.cat([h_i, h_j], dim=-1).reshape(B * N * N, 2 * D)
            ).reshape(B, N, N)
            e_dir_mat = e_dir_mat.masked_fill(diag_mask,   float("-inf"))
            e_dir_mat = e_dir_mat.masked_fill(~pair_valid, float("-inf"))

            if lah_prior_mat is not None:
                e_dir_mat = e_dir_mat + self.prior_w_attn * decay_w[iter_idx] * lah_prior_mat
            if sa_prior_mat is not None:
                e_dir_mat = e_dir_mat + self.prior_w_sa * decay_w[iter_idx] * sa_prior_mat

            W_msg_h = self.W_msg(h)   # (B, N, D)

            # ── Outgoing aggregation: i collects from nodes it looks at ──────
            # α_out[i,j]: softmax over destinations j (dim=-1)
            # msg_out_i = Σ_j α_out[i→j] · W_msg(h_j)  +  α_in[i] · W_msg(null_in)  +  α_out[i] · W_msg(null_out)
            if self.aggr in ("outgoing", "both"):
                if self.use_null_node:
                    # Node-dependent null scores: e_{i->null} = MLP_null([h_i; v_null])
                    v_in  = self.null_in_node.expand(B, N, -1)   # (B, N, D)
                    v_out = self.null_out_node.expand(B, N, -1)  # (B, N, D)
                    e_null_in  = self.mlp_null_in(
                        torch.cat([h, v_in],  dim=-1).reshape(B * N, 2 * D)
                    ).reshape(B, N)
                    e_null_out = self.mlp_null_out(
                        torch.cat([h, v_out], dim=-1).reshape(B * N, 2 * D)
                    ).reshape(B, N)
                    e_null_in  = e_null_in.masked_fill(~node_valid, float("-inf"))
                    e_null_out = e_null_out.masked_fill(~node_valid, float("-inf"))
                    # (B, N, N+2): last two cols are null_in and null_out scores
                    e_aug_out = torch.cat(
                        [e_dir_mat, e_null_in.unsqueeze(-1), e_null_out.unsqueeze(-1)], dim=-1
                    )
                else:
                    e_aug_out = e_dir_mat
                all_inf_out = e_aug_out.isinf().all(dim=-1, keepdim=True)
                e_aug_out   = e_aug_out.masked_fill(all_inf_out, 0.0)
                alpha_out   = torch.softmax(e_aug_out, dim=-1)   # (B, N, N[+2])
                msg_out = torch.einsum("bij,bjd->bid", alpha_out[:, :, :N], W_msg_h)
                if self.use_null_node:
                    _alpha_null_in  = alpha_out[:, :, N]      # (B, N)
                    _alpha_null_out = alpha_out[:, :, N + 1]  # (B, N)
                    msg_out = (
                        msg_out
                        + _alpha_null_in.unsqueeze(-1)  * self.W_msg(self.null_in_node).to(dtype)
                        + _alpha_null_out.unsqueeze(-1) * self.W_msg(self.null_out_node).to(dtype)
                    )

            # ── Ingoing aggregation: i collects from nodes looking at it ─────
            # α_in[i,j]: softmax over sources j (dim=1), treating e_dir_mat[j,i] as score of j→i
            # msg_in_i = Σ_j α_in[j→i] · W_msg(h_j)
            if self.aggr in ("ingoing", "both"):
                # mask invalid pairs before softmax (same pair_valid, diag already -inf)
                e_in = e_dir_mat.masked_fill(e_dir_mat.isinf().all(dim=1, keepdim=True), 0.0)
                alpha_in = torch.softmax(e_in, dim=1)   # (B, N, N): softmax over source dim
                # alpha_in[b, j, i] = how much j contributes to i
                msg_in = torch.einsum("bji,bjd->bid", alpha_in, W_msg_h)

            # ── Combine ───────────────────────────────────────────────────────
            if self.aggr == "outgoing":
                msg = msg_out
            elif self.aggr == "ingoing":
                msg = msg_in
            else:  # both
                msg = self.W_combine_out(msg_out) + self.W_combine_in(msg_in)

            # ── Node update ──────────────────────────────────────────────────
            gate  = torch.sigmoid(self.W_gate(h))  # (B, N, D)
            delta = self.update_proj(torch.cat([h, msg], dim=-1))
            h_new = self.norm(h + gate * delta).to(dtype)
            h = torch.where(node_valid.unsqueeze(-1), h_new, h)

        return h.float(), _alpha_null_in, _alpha_null_out


class UndirectedSocialGraphBlock(nn.Module):
    """
    Undirected social graph for Shared-Attention (SA / co-attention).

    SA is symmetric ("do i and j look at the same thing?") and non-exclusive
    (a person can co-attend with several others at once). So, unlike the
    directed LAH block — which uses softmax+null aggregation to encode an
    "attend to ~one target" prior — this block uses INDEPENDENT sigmoid edge
    gates and aggregates a gated MEAN over valid neighbours. Mean (rather than
    sum) normalisation keeps the message scale invariant to the number of
    people N.

    Edge scores are order-invariant by construction (built from the symmetric
    pair feature [h_i+h_j ; |h_i-h_j|]), so gate(i,j) == gate(j,i): a true
    undirected graph. The SA gaze prior (gaze_i · gaze_j) is symmetric too and
    injected with a learnable per-iteration decay, mirroring SocialGraphBlock.
    """

    def __init__(
        self,
        token_dim: int,
        num_layers: int = 2,
        hidden_channels: int = 96,
        use_sa_prior: bool = True,
        sa_prior_weight: float = 0.5,
    ):
        super().__init__()
        self.num_layers   = num_layers
        self.use_sa_prior = use_sa_prior

        # Symmetric edge scorer: [h_i+h_j ; |h_i-h_j|] -> scalar logit
        self.mlp_edge    = MLP(token_dim * 2, hidden_channels, 1)
        self.W_msg       = nn.Linear(token_dim, token_dim, bias=False)
        self.update_proj = nn.Linear(token_dim * 2, token_dim)
        self.W_gate      = nn.Linear(token_dim, token_dim)
        self.norm        = nn.LayerNorm(token_dim)

        if use_sa_prior:
            self.prior_w_sa = nn.Parameter(torch.tensor(sa_prior_weight))
        self.prior_decay_logits = nn.Parameter(torch.zeros(num_layers))

        self._edge_cache: dict = {}

    @staticmethod
    def _build_edges(N: int, device: torch.device):
        src = torch.tensor([s for s in range(N) for d in range(N) if s != d],
                           dtype=torch.long, device=device)
        dst = torch.tensor([d for s in range(N) for d in range(N) if s != d],
                           dtype=torch.long, device=device)
        return src, dst

    def _get_edge_cache(self, N: int, device: torch.device):
        if N not in self._edge_cache:
            self._edge_cache[N] = dict(zip(("src", "dst"), self._build_edges(N, device)))
        return self._edge_cache[N]

    def forward(self, person_tokens, num_valid_people, gaze_vecs=None):
        """
        Args:
            person_tokens:    (B, N, D)
            num_valid_people: (B,) int — valid nodes occupy back slots [N-nv .. N-1]
            gaze_vecs:        (B, N, 2) unit gaze direction (optional, for SA prior)

        Returns:
            tokens_out: (B, N, D) SA-tailored node features.
        """
        B, N, D = person_tokens.shape
        device  = person_tokens.device
        dtype   = person_tokens.dtype

        node_valid = (
            torch.arange(N, device=device).unsqueeze(0) >= (N - num_valid_people.unsqueeze(1))
        )  # (B, N)
        pair_valid = node_valid.unsqueeze(2) & node_valid.unsqueeze(1)              # (B, N, N)
        diag       = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        edge_ok    = pair_valid & ~diag                                            # (B, N, N)
        # valid neighbour count per node (>=1) for N-invariant mean aggregation
        deg = edge_ok.sum(-1).clamp(min=1).to(dtype)                               # (B, N)

        # ── SA gaze prior (symmetric, iteration-independent) ──────────────────
        sa_prior_mat = None
        if self.use_sa_prior and gaze_vecs is not None:
            cache        = self._get_edge_cache(N, device)
            src_N, dst_N = cache["src"], cache["dst"]
            sa_prior = (gaze_vecs[:, src_N] * gaze_vecs[:, dst_N]).sum(-1)          # (B, E)
            sa_prior_mat = torch.zeros(B, N, N, device=device, dtype=dtype)
            sa_prior_mat[:, src_N, dst_N] = sa_prior.to(dtype)

        decay_w = torch.softmax(self.prior_decay_logits, dim=0)                     # (num_layers,)

        h = person_tokens.clone()
        for it in range(self.num_layers):
            h_i = h.unsqueeze(2).expand(B, N, N, D)
            h_j = h.unsqueeze(1).expand(B, N, N, D)
            # symmetric pair feature -> symmetric edge logit (gate(i,j)==gate(j,i))
            sym_feat = torch.cat([h_i + h_j, (h_i - h_j).abs()], dim=-1).reshape(B * N * N, 2 * D)
            e = self.mlp_edge(sym_feat).reshape(B, N, N)
            if sa_prior_mat is not None:
                e = e + self.prior_w_sa * decay_w[it] * sa_prior_mat

            # independent sigmoid gates, zeroed on invalid / self edges
            gate = torch.sigmoid(e) * edge_ok.to(dtype)                            # (B, N, N)

            # gated mean aggregation (N-invariant)
            msg = torch.einsum("bij,bjd->bid", gate, self.W_msg(h)) / deg.unsqueeze(-1)

            g     = torch.sigmoid(self.W_gate(h))
            delta = self.update_proj(torch.cat([h, msg], dim=-1))
            h_new = self.norm(h + g * delta).to(dtype)
            h     = torch.where(node_valid.unsqueeze(-1), h_new, h)

        return h.float()


class TemporalGraphBlock(nn.Module):
    """
    Temporal attention over t frames per person using nn.MultiheadAttention.

    Expects input (M, t, D) where M = B * N (caller reshapes before calling).
    For a fully-connected temporal graph, MHA is equivalent to GAT and avoids
    PyG sparse kernel overhead for small t (e.g. t=3).
    """

    def __init__(self, token_dim: int, hidden_channels: int = 96, heads: int = 8):
        super().__init__()
        # hidden_channels kept for API compatibility but unused; MHA uses token_dim directly.
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = MLP(token_dim, token_dim, token_dim)

    def forward(self, person_tokens_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            person_tokens_t: (M, t, D)  where M = B * N

        Returns:
            (M, t, D)
        """
        if person_tokens_t.shape[1] <= 1:
            return person_tokens_t

        attn_out, _ = self.attn(
            person_tokens_t, person_tokens_t, person_tokens_t,
            need_weights=False,
        )
        dtype = person_tokens_t.dtype
        h = self.norm1(person_tokens_t + attn_out).to(dtype)
        h = self.norm2(h + self.ffn(h)).to(dtype)
        return h


class _FusedGazeRefiner(nn.Module):
    """Asymmetric dual-role refinement with a SHARED person node (×L).

    Per iteration:
      • LAH path: row-wise OUTGOING attention over [P_1..P_N, O] targets → msg_out_i
      • SA  path: column-wise INCOMING attention over sources → msg_in_k (region k)
      • shared person node fused from BOTH directions via cross-attention:
            v_i ← LN( v_i + CrossAttn(q=v_i, kv=MLP([msg_out_i ‖ sg(msg_in_i)])) )
        (msg_in stop-gradient'd into the fusion, per request)
      • null / region target nodes updated from their incoming pools
      • LAH / SA edges refreshed from the updated nodes
    LAH logits read the outgoing (person/null) edges, SA logits the incoming
    (region) edges; the person node is the single point where the two roles meet.

    NOTE: only the fusion's msg_in is detached. The SA edge refresh still reads the
    shared node, so SA gradient can still reach LAH through it — intentional (full
    isolation deferred; revisit if LAEO degrades).
    """

    def __init__(self, edge_dim: int, num_layers: int, heads: int):
        super().__init__()
        De = edge_dim
        self.De = De
        self.num_layers = num_layers
        enc = lambda: nn.TransformerEncoderLayer(
            d_model=De, nhead=heads, dim_feedforward=2 * De,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.row = enc()   # LAH: outgoing (same-source targets attend)
        self.col = enc()   # SA : incoming (same-target sources attend)

        # shared person-node fusion: cross-attn(q=v, kv=MLP([msg_out ‖ sg(msg_in)]))
        self.fuse_kv    = MLP(2 * De, De, De)
        self.fuse_xattn = CrossAttention(De, num_heads=heads)
        self.norm_v     = nn.LayerNorm(De)

        # target-side node updates (null, region) from their incoming pools
        self.upd_null = MLP(2 * De, De, De)
        self.norm_null = nn.LayerNorm(De)
        self.upd_reg  = MLP(2 * De, De, De)
        self.norm_reg = nn.LayerNorm(De)

        # per-path edge refresh from updated incident nodes
        self.refresh_lah = MLP(3 * De, De, De)
        self.norm_lah    = nn.LayerNorm(De)
        self.refresh_sa  = MLP(3 * De, De, De)
        self.norm_sa     = nn.LayerNorm(De)

    def forward(self, E_lah, E_sa, v, v_null, v_reg,
                ev_lah, ev_sa, row_kpm, col_kpm, deg_out, deg_in, deg_null):
        # E_lah: (B, N, N+1, De)  targets [P_1..P_N, O];  E_sa: (B, N, N, De)  targets [R_k]
        B, N, Tl, De = E_lah.shape
        for _ in range(self.num_layers):
            # LAH: row-wise outgoing (each source's targets attend)
            E_lah = self.row(
                E_lah.reshape(B * N, Tl, De), src_key_padding_mask=row_kpm
            ).reshape(B, N, Tl, De) * ev_lah
            # SA: column-wise incoming (each region's sources attend)
            E_sa = self.col(
                E_sa.permute(0, 2, 1, 3).reshape(B * N, N, De), src_key_padding_mask=col_kpm
            ).reshape(B, N, N, De).permute(0, 2, 1, 3) * ev_sa

            # directional messages
            msg_out = E_lah[:, :, :N].sum(2) / deg_out       # (B, N, De)  source i over person targets
            msg_in  = E_sa.sum(1) / deg_in                   # (B, N, De)  region k over sources

            # shared person node fusion: cross-attn(q=v, kv=MLP([msg_out ‖ sg(msg_in)]))
            kv = self.fuse_kv(torch.cat([msg_out, msg_in.detach()], -1))   # (B, N, De)
            fused = self.fuse_xattn(
                v.reshape(B * N, 1, De), kv.reshape(B * N, 1, De)
            ).reshape(B, N, De)
            v = self.norm_v(v + fused)

            # target-side nodes: incoming pools
            null_pool = E_lah[:, :, N].sum(1, keepdim=True) / deg_null     # (B, 1, De)
            v_null = self.norm_null(v_null + self.upd_null(torch.cat([v_null, null_pool], -1)))
            v_reg = self.norm_reg(v_reg + self.upd_reg(torch.cat([v_reg, msg_in], -1)))

            # edge refresh from updated incident nodes
            tgt_lah = torch.cat([v, v_null], dim=1)                        # (B, N+1, De)
            E_lah = self.norm_lah(E_lah + self.refresh_lah(torch.cat(
                [E_lah,
                 v.unsqueeze(2).expand(B, N, Tl, De),
                 tgt_lah.unsqueeze(1).expand(B, N, Tl, De)], -1))) * ev_lah
            E_sa = self.norm_sa(E_sa + self.refresh_sa(torch.cat(
                [E_sa,
                 v.unsqueeze(2).expand(B, N, N, De),
                 v_reg.unsqueeze(1).expand(B, N, N, De)], -1))) * ev_sa
        return E_lah, E_sa


class _SocialReadoutHead(nn.Module):
    """ResidualLinearBlock(scale=4) + fc — mirrors LinearDecoderSocialGraph in mtgs_net.py."""
    def __init__(self, dim: int, scale: int = 4):
        super().__init__()
        self.fc1    = nn.Linear(dim, dim // scale, bias=False)
        self.bn1    = nn.BatchNorm1d(dim // scale)
        self.fc2    = nn.Linear(dim // scale, dim // scale ** 2, bias=False)
        self.bn2    = nn.BatchNorm1d(dim // scale ** 2)
        self.res_fc = nn.Linear(dim, dim // scale ** 2)
        self.fc_out = nn.Linear(dim // scale ** 2, 1)

    def forward(self, x):
        z = torch.relu(self.bn1(self.fc1(x)))
        h = torch.relu(self.bn2(self.fc2(z)) + self.res_fc(x))
        return self.fc_out(h)


class GazeGraphBlock(nn.Module):
    """
    Standalone directed gaze graph + dual-role edge refinement
    (interaction.type="gaze_graph").

    Runs ONCE after the ViT-Adaptor has produced per-person evidence tokens, so
    it does NOT feed the heatmap / inout decoders (those stay on the trunk). It
    exists only to produce socially-contextualised gaze-relation edge evidence.

    Nodes (per frame):
      Person  P_1..P_N  — source AND target (features = projected person tokens, dim D)
      Region  R_1..R_N  — target only       (R_j = "where person j looks"; feature =
                                              positional embedding of j's gaze target)
      Null    O         — target only       (single learnable node; out-of-frame gaze)

    Edge-state matrix E[i, t]  (source person i → target t ∈ {P_*, R_*, O}):
      e^0_{i→t} = MLP_init([ p_i ; node_t ; w·[s_hm, align]_{i→t} ])
        align = gaze-direction · dir(center_i → target-location) cosine
        s_hm  = source i's gaze heatmap H_i sampled (bilinear) at the target
                location (heatmap–target overlap)
        target location = head center for person targets, gaze anchor for region
        targets; both evidence channels are 0 for the null target.

    Asymmetric dual-role refinement (_FusedGazeRefiner), repeated L (= num_layers)
    times, with a SHARED person node where the two roles meet:
      LAH = outgoing role : persons [0:N] + null [2N]  → (B, N, N+1)
      SA  = incoming role : regions [N:2N]             → (B, N, N)
    Per iteration: LAH row-wise (outgoing) → msg_out; SA column-wise (incoming) →
    msg_in; person node fused via cross-attn(q=v, kv=MLP([msg_out ‖ sg(msg_in)]));
    null/region target nodes updated from incoming pools; LAH/SA edges refreshed
    from the updated nodes. LAEO is derived downstream as min(LAH_ij, LAH_ji).

    Read-out — PER-TYPE heads map each refined edge to a logit
    ℓ_{i→t} = head_*(ê_{i→t}) ("does i look at t?", 1/0 BCE):
      lah_mat[i,j] = head_lah (ê_{i→P_j})   (LAH: i looks at j's head)
      sa_mat[i,j]  = head_sa  (ê_{i→R_j})   (SA / co-attention: i looks where j looks)
      null_vec[i]  = head_null(ê_{i→O})     (out-of-frame)
    Diagonal person/region targets (i→P_i, i→R_i) are masked everywhere; valid
    people occupy the BACK slots [N-nv .. N-1] (padding convention).
    """

    def __init__(
        self,
        token_dim: int,
        edge_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        use_prior: bool = True,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        D, De = token_dim, edge_dim
        self.De         = De
        self.num_layers = num_layers
        self.use_prior  = use_prior

        # Region node = positional embedding of a gaze-target (x, y) in [0, 1],
        # then fused with the corresponding person j via cross-attention so R_j
        # carries BOTH "where j looks" (position) and "who j is" (appearance) —
        # restores the R_j ↔ P_j link the position-only node was missing.
        self.region_pos_mlp  = MLP(2, De, D)
        self.region_xattn    = CrossAttention(D, num_heads=heads)
        self.region_xattn_norm = nn.LayerNorm(D)
        self.null_node       = nn.Parameter(torch.zeros(D))

        # Heatmap encoding: pool to fixed grid, project to D.
        hm_grid = 8
        self.hm_grid = hm_grid
        self.hm_pool = nn.AdaptiveAvgPool2d(hm_grid)
        self.hm_proj = nn.Linear(1, D)
        self.hm_pos_emb = nn.Parameter(torch.randn(hm_grid * hm_grid, D) * 0.02)

        # Cross-attention for edge init: Pi'=xattn(Pi, hm_i), Pj'=xattn(Pj, hm_j)
        self.src_xattn = CrossAttention(D, num_heads=heads)
        self.tgt_xattn = CrossAttention(D, num_heads=heads)
        self.src_xattn_norm = nn.LayerNorm(D)
        self.tgt_xattn_norm = nn.LayerNorm(D)

        # Edge init: [Pi' ; Pj' ; s_hm ; align] -> De
        #   cross-attn appearance features + geometric prior (heatmap-target
        #   overlap, gaze-target alignment) per PDF.
        self.mlp_init = MLP(2 * D + 2, De, De)
        if use_prior:
            self.prior_w = nn.Parameter(torch.tensor(prior_weight))

        # Node states live in edge space (De).
        self.node_src_proj = nn.Linear(D, De)
        self.node_tgt_proj = nn.Linear(D, De)

        # Asymmetric refinement with a SHARED person node:
        #   LAH = outgoing role (person + null targets), SA = incoming role (regions),
        #   fused at the person node via cross-attn(q=v, kv=MLP([msg_out ‖ sg(msg_in)])).
        self.refiner = _FusedGazeRefiner(De, num_layers, heads)

        # Per-type read-out heads (one logit per refined edge, 1/0 BCE). Separate
        # heads keep SA's gradient off the LAH/null mapping; the shared single head
        # (previous version) let SA dominate and eroded LAH precision.
        self.head_lah  = _SocialReadoutHead(De)   # ê_{i→P_j}  → LAH
        self.head_null = _SocialReadoutHead(De)   # ê_{i→O}    → out-of-frame
        self.head_sa   = _SocialReadoutHead(De)   # ê_{i→R_j}  → SA / co-attention

    @staticmethod
    def _safe_kpm(kpm: torch.Tensor) -> torch.Tensor:
        # nn.TransformerEncoderLayer emits NaN for fully-masked sequences; unmask
        # those rows (their outputs are discarded downstream via edge_valid).
        return kpm & ~kpm.all(dim=1, keepdim=True)

    def forward(self, person_tokens, num_valid_people, region_anchors,
                gaze_vecs, head_bboxes, gaze_heatmaps):
        """
        Args:
            person_tokens:    (B, N, D)      B = b*t
            num_valid_people: (B,)           valid people at BACK slots [N-nv .. N-1]
            region_anchors:   (B, N, 2)      normalized (x, y) gaze-target of each person
            gaze_vecs:        (B, N, 2)      unit gaze direction
            head_bboxes:      (B, N, 4)      normalized [x1, y1, x2, y2]
            gaze_heatmaps:    (B, N, Hh, Ww) per-person gaze heatmap

        Targets (T = 2N+1):
            P_1..P_N  person nodes  → LAH edges e_{i→P_j}
            R_1..R_N  region nodes  → SA edges  e_{i→R_j}
            O         null node     → out-of-frame edge e_{i→O}

        Returns:
            lah_mat:   (B, N, N)  logits e_{i→P_j}
            sa_mat:    (B, N, N)  logits e_{i→R_j}
            null_vec:  (B, N)     logits e_{i→O}
            edge_valid:(B, N, T)  validity mask
        """
        B, N, D = person_tokens.shape
        device, dtype = person_tokens.device, person_tokens.dtype
        De = self.De
        T  = 2 * N + 1

        node_valid = (
            torch.arange(N, device=device).unsqueeze(0)
            >= (N - num_valid_people.unsqueeze(1))
        )  # (B, N)

        # ── Target node features: [persons (updated below) | regions | null] ───
        region_nodes = self.region_pos_mlp(region_anchors.to(dtype))           # (B, N, D)
        # Inject person j's content into region node R_j (1:1 index-aligned xattn):
        # R_j' = LN(R_j + CrossAttn(R_j, P_j)).
        r_q  = region_nodes.reshape(B * N, 1, D)
        p_kv = person_tokens.reshape(B * N, 1, D)
        region_nodes = self.region_xattn_norm(
            r_q + self.region_xattn(r_q, p_kv)
        ).reshape(B, N, D)                                                      # (B, N, D)
        null_node    = self.null_node.to(dtype).view(1, 1, D).expand(B, 1, D)

        # ── Validity / self-edge masks over targets ──────────────────────────
        tgt_valid = torch.cat(
            [node_valid, node_valid,
             torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1,
        )                                                                      # (B, T)
        eye       = torch.eye(N, device=device, dtype=torch.bool)
        self_mask = torch.zeros(N, T, dtype=torch.bool, device=device)        # (N, T)
        self_mask[:, :N]      = eye    # i → P_i
        self_mask[:, N:2 * N] = eye    # i → R_i
        edge_valid = (
            node_valid.unsqueeze(2) & tgt_valid.unsqueeze(1) & ~self_mask.unsqueeze(0)
        )                                                                      # (B, N, T)
        ev = edge_valid.unsqueeze(-1).to(dtype)                                # (B, N, T, 1)

        # ── Heatmap features: pool to P×P grid, project to D ────────────────
        Hh, Ww = gaze_heatmaps.shape[-2:]
        P = self.hm_grid ** 2
        # detach: graph reads the heatmap as a fixed evidence signal so social
        # loss gradients don't corrupt heatmap decoder training.
        hm_small = self.hm_pool(gaze_heatmaps.reshape(B * N, 1, Hh, Ww).detach())  # (B*N, 1, g, g)
        hm_feat  = (
            self.hm_proj(hm_small.reshape(B * N, P, 1).to(dtype))
            + self.hm_pos_emb.to(dtype)
        )                                                                      # (B*N, P, D)

        # ── Source cross-attention: Pi' = LN(Pi + CrossAttn(Pi, hm_i)) ──────
        src_q   = person_tokens.reshape(B * N, 1, D)                          # (B*N, 1, D)
        src_prime = self.src_xattn_norm(
            src_q + self.src_xattn(src_q, hm_feat)
        ).reshape(B, N, D)                                                    # (B, N, D)

        # ── Target cross-attention for person nodes: Pj' = LN(Pj + CrossAttn(Pj, hm_j))
        # Region/null targets keep their existing features unchanged.
        tgt_q  = person_tokens.reshape(B * N, 1, D)                           # (B*N, 1, D)
        tgt_prime_persons = self.tgt_xattn_norm(
            tgt_q + self.tgt_xattn(tgt_q, hm_feat)
        ).reshape(B, N, D)                                                    # (B, N, D)

        tgt_nodes = torch.cat([tgt_prime_persons, region_nodes, null_node], 1)  # (B, T, D)

        # ── Edge evidence [s_hm, align] (geometric prior, per PDF) ────────────
        evidence = torch.zeros(B, N, T, 2, device=device, dtype=dtype)
        if self.use_prior:
            centers = (head_bboxes[..., :2] + head_bboxes[..., 2:]) / 2        # (B, N, 2)
            # target locations: person → head center, region → gaze anchor (2N pts)
            tgt_loc = torch.cat([centers, region_anchors], dim=1)             # (B, 2N, 2)

            # ① gaze-target alignment: gaze_i · dir(center_i → loc_t)
            dir_t = F.normalize(tgt_loc.unsqueeze(1) - centers.unsqueeze(2), dim=-1)
            align = (gaze_vecs.unsqueeze(2) * dir_t).sum(-1)                   # (B, N, 2N)

            # ② heatmap-target overlap: H_i (detached) sampled at loc_t (bilinear)
            heat = gaze_heatmaps.reshape(B * N, 1, Hh, Ww).detach().float()
            grid = (2.0 * tgt_loc - 1.0).unsqueeze(1).expand(B, N, 2 * N, 2)
            grid = grid.reshape(B * N, 2 * N, 1, 2).float()
            s_hm = F.grid_sample(
                heat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            ).reshape(B, N, 2 * N).to(dtype)                                   # (B, N, 2N)

            evidence[:, :, :2 * N, 0] = s_hm
            evidence[:, :, :2 * N, 1] = align
            evidence = self.prior_w * evidence

        # ── Edge initialisation: e^0 = MLP([Pi' ; Pj' ; s_hm ; align]) ────────
        src_e = src_prime.unsqueeze(2).expand(B, N, T, D)
        tgt_e = tgt_nodes.unsqueeze(1).expand(B, N, T, D)
        E = self.mlp_init(
            torch.cat([src_e, tgt_e, evidence], dim=-1).reshape(B * N * T, -1)
        ).reshape(B, N, T, De) * ev

        node_src = self.node_src_proj(src_prime)   # (B, N, De)  shared person node v init
        node_tgt = self.node_tgt_proj(tgt_nodes)   # (B, T, De)  target-node init

        # ── Asymmetric paths over disjoint target subsets, fused at v ────────
        #   LAH (outgoing): persons [0:N] + null [2N]   → (B, N, N+1)
        #   SA  (incoming): regions [N:2N]              → (B, N, N)
        null_col = slice(2 * N, 2 * N + 1)
        gather_lah = lambda x: torch.cat([x[:, :, :N], x[:, :, null_col]], dim=2)

        E_lah  = gather_lah(E)                      # (B, N, N+1, De)
        E_sa   = E[:, :, N:2 * N]                   # (B, N, N,   De)
        ev_lah = gather_lah(ev)                     # (B, N, N+1, 1)
        ev_sa  = ev[:, :, N:2 * N]                  # (B, N, N,   1)

        v      = node_src                           # (B, N,   De)  shared person node
        v_null = node_tgt[:, null_col]              # (B, 1,   De)
        v_reg  = node_tgt[:, N:2 * N]               # (B, N,   De)

        valid_lah = gather_lah(edge_valid.unsqueeze(-1)).squeeze(-1)           # (B, N, N+1)
        valid_sa  = edge_valid[:, :, N:2 * N]                                  # (B, N, N)

        row_kpm = self._safe_kpm((~valid_lah).reshape(B * N, N + 1))           # LAH: over targets
        col_kpm = self._safe_kpm((~valid_sa).permute(0, 2, 1).reshape(B * N, N))  # SA: over sources
        deg_out = valid_lah[:, :, :N].sum(2, keepdim=True).clamp(min=1).to(dtype)  # (B, N, 1)
        deg_in  = valid_sa.sum(1).clamp(min=1).to(dtype).unsqueeze(-1)         # (B, N, 1) per region
        deg_null = valid_lah[:, :, N].sum(1, keepdim=True).clamp(min=1).to(dtype).unsqueeze(-1)  # (B,1,1)

        E_lah, E_sa = self.refiner(
            E_lah, E_sa, v, v_null, v_reg,
            ev_lah, ev_sa, row_kpm, col_kpm, deg_out, deg_in, deg_null,
        )

        # ── Per-type read-out (one logit per refined edge, 1/0 BCE) ──────────
        lah_mat  = self.head_lah(E_lah[:, :, :N].reshape(B * N * N, De)).reshape(B, N, N)
        null_vec = self.head_null(E_lah[:, :, N].reshape(B * N, De)).reshape(B, N)
        sa_mat   = self.head_sa(E_sa.reshape(B * N * N, De)).reshape(B, N, N)
        return lah_mat.float(), sa_mat.float(), null_vec.float(), edge_valid
