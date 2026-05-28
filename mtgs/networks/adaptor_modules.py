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

            # Inject LAH cosine prior into attention on iteration 0 only
            if self.use_gaze_prior and lah_prior is not None and iter_idx == 0:
                lah_prior_mat = torch.zeros(B, N, N, device=device, dtype=dtype)
                lah_prior_mat[:, src_N, dst_N] = lah_prior.to(dtype)
                e_dir_mat = e_dir_mat + self.prior_w_attn * lah_prior_mat

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
