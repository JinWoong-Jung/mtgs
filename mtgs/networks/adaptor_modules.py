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

    def forward(self, x, c, blocks, num_valid_people, position_embeddings=None):
        x = self.injector(query=x, feat=c)
        for idx, blk in enumerate(blocks):
            # HuggingFace transformer blocks return a tuple; take the hidden states
            if position_embeddings is not None:
                out = blk(x, position_embeddings=position_embeddings)
            else:
                out = blk(x)
            x = out[0] if isinstance(out, (tuple, list)) else out
        c = self.extractor(query=c, feat=x)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, feat=x)

        return x, c


class SocialGraphBlock(nn.Module):
    """
    Social interaction block with unified edge-based social predictions.

    Edge logits computed during message passing serve dual roles:
    attention weights for aggregation AND direct social predictions (LAH, SA).
    No separate pair-reconstruction decoder is needed after this block.

    LAH (directed):   e_dir(i→j) = MLP_dir(cat(h_i, h_j))
    SA  (undirected): e_sa(i,j)  = MLP_sa(h_i + h_j)   (symmetric by construction)
    Null:             e_null(i)  = MLP_null(h_i)

    Geometric priors:
      - LAH cosine prior injected into the softmax on iteration 0 only (attention routing)
      - Both LAH and SA priors added to the final output logits (prediction bias)

    Message aggregation uses OUTGOING attention — aggregating by whom i looks at:
        msg_i = Σ_j α[i→j] · W_msg(h_j)  +  α_null[i] · W_msg(null_node)

    Output edge logits come from the final internal iteration, with priors applied.
    Edge ordering matches GT label ordering:
        [(s, d) for s in range(N) for d in range(N) if s != d]
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
    ):
        super().__init__()
        self.num_layers     = num_layers
        self.use_null_node  = use_null_node
        self.use_gaze_prior = use_gaze_prior

        # Learnable prior weights — initialized to prior_weight, unconstrained.
        # Three separate scalars: attention routing, LAH output, SA output.
        self.prior_w_attn = nn.Parameter(torch.tensor(prior_weight))
        self.prior_w_lah  = nn.Parameter(torch.tensor(prior_weight))
        self.prior_w_sa   = nn.Parameter(torch.tensor(prior_weight))

        # ── Edge scoring MLPs ────────────────────────────────────────────────
        self.mlp_dir = MLP(token_dim * 2, hidden_channels, 1)
        self.mlp_sa  = MLP(token_dim,     hidden_channels, 1)
        if use_null_node:
            self.null_node = nn.Parameter(torch.zeros(token_dim))
            self.mlp_null  = MLP(token_dim, hidden_channels, 1)

        # ── Message passing & node update ────────────────────────────────────
        self.W_msg       = nn.Linear(token_dim, token_dim, bias=False)
        self.update_proj = nn.Linear(token_dim * 2, token_dim)
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
            tokens_out:  (B, N, D)       updated node features
            lah_logits:  (B, N*(N-1))    LAH edge logits (final iter + priors)
            sa_logits:   (B, N*(N-1))    SA  edge logits (final iter + prior)
            null_logits: (B, N)          null-node logits from final h; zeros if disabled
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

        # ── Pre-compute geometric priors (data-dependent, iteration-independent) ──
        lah_prior = sa_prior = None
        if self.use_gaze_prior and gaze_vecs is not None and head_bboxes is not None:
            centers = (head_bboxes[..., :2] + head_bboxes[..., 2:]) / 2   # (B, N, 2)

            # LAH prior: cosine(gaze_i, dir_{i→j}), edge-list form (B, E)
            dir_ij    = F.normalize(centers[:, dst_N] - centers[:, src_N], dim=-1)
            lah_prior = (gaze_vecs[:, src_N] * dir_ij).sum(-1)            # (B, E)

            # SA prior: gaze ray convergence, (B, E) ∈ {-1, 0, +1}
            c_src = centers[:, src_N];   c_dst = centers[:, dst_N]
            g_src = gaze_vecs[:, src_N]; g_dst = gaze_vecs[:, dst_N]
            dc    = c_dst - c_src
            det   = g_src[..., 0] * g_dst[..., 1] - g_src[..., 1] * g_dst[..., 0]
            t_i   = (dc[..., 0] * g_dst[..., 1] - dc[..., 1] * g_dst[..., 0]) / (det + 1e-6)
            t_j   = (dc[..., 0] * g_src[..., 1] - dc[..., 1] * g_src[..., 0]) / (det + 1e-6)
            sa_prior = (t_i > 0).float() + (t_j > 0).float() - 1.0       # (B, E)

        h = person_tokens.clone()
        e_dir_last = e_sa_last = None

        for iter_idx in range(self.num_layers):
            # ── Edge logit computation ───────────────────────────────────────
            h_i = h.unsqueeze(2).expand(B, N, N, D)   # (B, N, N, D)
            h_j = h.unsqueeze(1).expand(B, N, N, D)   # (B, N, N, D)

            # LAH: directed, cat(h_i, h_j) → scalar
            e_dir_mat = self.mlp_dir(
                torch.cat([h_i, h_j], dim=-1).reshape(B * N * N, 2 * D)
            ).reshape(B, N, N)
            e_dir_mat = e_dir_mat.masked_fill(diag_mask,   float("-inf"))
            e_dir_mat = e_dir_mat.masked_fill(~pair_valid, float("-inf"))

            # SA: symmetric, h_i + h_j → scalar
            e_sa_mat = self.mlp_sa(
                (h_i + h_j).reshape(B * N * N, D)
            ).reshape(B, N, N)

            # Inject LAH prior into attention softmax on iter 0 only
            if self.use_gaze_prior and lah_prior is not None and iter_idx == 0:
                lah_prior_mat = torch.zeros(B, N, N, device=device, dtype=dtype)
                lah_prior_mat[:, src_N, dst_N] = lah_prior.to(dtype)
                e_dir_mat = e_dir_mat + self.prior_w_attn * lah_prior_mat

            # ── Softmax over outgoing edges per source node (+ null) ─────────
            if self.use_null_node:
                e_null = self.mlp_null(h.reshape(B * N, D)).reshape(B, N)
                e_null = e_null.masked_fill(~node_valid, float("-inf"))
                e_aug  = torch.cat([e_dir_mat, e_null.unsqueeze(-1)], dim=-1)  # (B, N, N+1)
            else:
                e_aug = e_dir_mat

            all_inf = e_aug.isinf().all(dim=-1, keepdim=True)
            e_aug   = e_aug.masked_fill(all_inf, 0.0)
            alpha   = torch.softmax(e_aug, dim=-1)   # (B, N, N[+1])

            # ── Message aggregation — outgoing: whom i is looking at ────────────
            # msg_i = Σ_j α[i→j] · W_msg(h_j)
            W_msg_h = self.W_msg(h)                              # (B, N, D)
            msg = torch.einsum(
                "bij,bjd->bid", alpha[:, :, :N], W_msg_h
            )
            if self.use_null_node:
                alpha_null = alpha[:, :, N]                      # (B, N): i→null weight
                msg = msg + alpha_null.unsqueeze(-1) * self.W_msg(self.null_node).to(dtype)

            # ── Node update ─────────────────────────────────────────────────
            h_new = self.update_proj(torch.cat([h, msg], dim=-1))
            h_new = self.norm(h + h_new).to(dtype)
            h = torch.where(node_valid.unsqueeze(-1), h_new, h)

            e_dir_last = e_dir_mat
            e_sa_last  = e_sa_mat

        # ── Extract edge-list logits in GT ordering, add priors ──────────────
        lah_logits = e_dir_last[:, src_N, dst_N]   # (B, E)
        sa_logits  = e_sa_last[:, src_N, dst_N]    # (B, E)

        if self.use_gaze_prior and lah_prior is not None:
            lah_logits = lah_logits + self.prior_w_lah * lah_prior.to(dtype)
        if self.use_gaze_prior and sa_prior is not None:
            sa_logits  = sa_logits  + self.prior_w_sa  * sa_prior.to(dtype)

        # Null logits from fully-updated h
        if self.use_null_node:
            null_logits = self.mlp_null(h.reshape(B * N, D)).reshape(B, N) * node_valid.float()
        else:
            null_logits = torch.zeros(B, N, device=device)

        return h.float(), lah_logits.float(), sa_logits.float(), null_logits.float()


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
