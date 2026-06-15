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


def _compute_bbox_overlap(
    hm_norm: torch.Tensor, bboxes: torch.Tensor
) -> torch.Tensor:
    """Integral of hm_i inside bbox_j for all (source i, target j) pairs.

    Args:
        hm_norm: (BT, N, H, W)  relu-normalised heatmap (pixels sum to 1 per person)
        bboxes:  (BT, N, 4)     [x1, y1, x2, y2] in normalised [0, 1] image coords

    Returns:
        overlap: (BT, N, N)     overlap[b, i, j] = fraction of hm_i mass inside bbox_j
    """
    BT, N, H, W = hm_norm.shape
    device = hm_norm.device
    dtype  = hm_norm.dtype

    # Pixel-centre coordinates normalised to [0, 1]
    yy = (torch.arange(H, device=device).float() + 0.5) / H   # (H,)
    xx = (torch.arange(W, device=device).float() + 0.5) / W   # (W,)

    x1 = bboxes[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # (BT, N, 1, 1)
    y1 = bboxes[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    x2 = bboxes[:, :, 2].unsqueeze(-1).unsqueeze(-1)
    y2 = bboxes[:, :, 3].unsqueeze(-1).unsqueeze(-1)

    # Binary spatial mask per target bbox: (BT, N_tgt, H, W)
    mask = (
        (xx.view(1, 1, 1, W) >= x1) & (xx.view(1, 1, 1, W) <= x2) &
        (yy.view(1, 1, H, 1) >= y1) & (yy.view(1, 1, H, 1) <= y2)
    ).to(dtype)
    area = mask.sum((-2, -1)).clamp(min=1.0)  # (BT, N_tgt)

    # overlap[b, i, j] = Σ_{hw} hm[b,i,h,w] * mask[b,j,h,w]
    #   hm_norm  (BT, N_src, H, W) → unsqueeze N_tgt → (BT, N_src, 1, H, W)
    #   mask     (BT, N_tgt, H, W) → unsqueeze N_src → (BT, 1, N_tgt, H, W)
    overlap = (hm_norm.unsqueeze(2) * mask.unsqueeze(1)).sum((-2, -1))  # (BT, N_src, N_tgt)
    return overlap.to(dtype)  # already in [0,1]: hm_norm sums to 1


class _UnifiedRefiner(nn.Module):
    """Per-frame spatial edge refinement with row + col attention.

    E shape throughout: (B, T, N, Tl, De)   where Tl = N + 2.
    Column attention covers only the first N+1 targets (null_out excluded).
    Frames are processed independently; temporal context is already carried by
    the upstream person tokens.
    """

    def __init__(self, edge_dim: int, num_layers: int, heads: int):
        super().__init__()
        De = self.De = edge_dim
        self.num_layers = num_layers

        _enc = lambda: nn.TransformerEncoderLayer(
            d_model=De, nhead=heads, dim_feedforward=2 * De,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.row      = _enc()
        self.col      = _enc()

        self.refresh  = MLP(3 * De, De, De)
        self.norm_e   = nn.LayerNorm(De)
        self.pool_out = nn.Linear(De, 1)
        self.pool_in  = nn.Linear(De, 1)

        # Source and target roles are updated from their own edge direction.
        self.upd_src     = MLP(2 * De, De, De)
        self.norm_src    = nn.LayerNorm(De)
        self.upd_tgt     = MLP(2 * De, De, De)
        self.norm_tgt    = nn.LayerNorm(De)
        # null_in: only incoming (no outgoing edges)
        self.upd_nullin  = MLP(2 * De, De, De)
        self.norm_nullin = nn.LayerNorm(De)
        # null_out: no update (idea.md §3)

        self.inject   = MLP(3 * De, De, De)
        self.norm_inj = nn.LayerNorm(De)

    @staticmethod
    def _safe_kpm(kpm: torch.Tensor) -> torch.Tensor:
        # Fully-masked sequences cause NaN in TransformerEncoderLayer.
        # Unmask them; their output is discarded by multiplying with ev.
        return kpm & ~kpm.all(dim=1, keepdim=True)

    def forward(self, E, ev, row_kpm, col_kpm, v_src, v_tgt, deg_out, deg_in):
        """
        E:       (B, T, N, Tl, De)
        ev:      (B, T, N, Tl, 1)    float 0/1 validity mask
        row_kpm: (B*T*N, Tl)
        col_kpm: (B*T*(N+1), N)      null_out column excluded
        v_src:   (B, T, N, De)
        v_tgt:   (B, T, Tl, De)
        deg_out: (B, T, N, 1)
        deg_in:  (B, T, Tl, 1)
        """
        B, T, N, Tl, De = E.shape
        ev_sq = ev.squeeze(-1)  # (B, T, N, Tl) float; constant across layers

        for _ in range(self.num_layers):
            # ── ① Row: source i attends over all Tl targets ───────────────────
            row_context = self.row(
                E.reshape(B * T * N, Tl, De), src_key_padding_mask=row_kpm
            ).reshape(B, T, N, Tl, De) * ev
            E = row_context

            # ── ② Col: target k (person + null_in) attends over N sources ─────
            #    null_out excluded; its col_context slot retains row_context so the
            #    refresh always has a well-defined 3-way concat for every edge.
            E_col_in = E[:, :, :, :N + 1, :]
            E_col_out_N1 = self.col(
                E_col_in.permute(0, 1, 3, 2, 4).reshape(B * T * (N + 1), N, De),
                src_key_padding_mask=col_kpm,
            ).reshape(B, T, N + 1, N, De).permute(0, 1, 3, 2, 4)  # (B, T, N, N+1, De)
            col_context = torch.cat(
                [E_col_out_N1, row_context[:, :, :, N + 1:, :]], dim=3
            ) * ev
            E = col_context

            # ── ③ Edge refresh: refine each frame's edge graph independently ──
            #    edge = LN(edge + MLP(concat(edge, row_context, col_context)))
            E = self.norm_e(
                E + self.refresh(torch.cat([E, row_context, col_context], dim=-1))
            ) * ev

            # ── ④ Node update: learned attention pooling (idea.md §7) ────────
            # out_agg: source i attends over all Tl outgoing edges
            scores_out = self.pool_out(E).squeeze(-1)              # (B, T, N, Tl)
            scores_out = scores_out.masked_fill(ev_sq == 0, float('-inf'))
            safe_out   = scores_out.isinf().all(dim=-1, keepdim=True)
            scores_out = scores_out.masked_fill(safe_out, 0.0)
            alpha_out  = F.softmax(scores_out, dim=-1) * ev_sq    # (B, T, N, Tl)
            out_agg    = (alpha_out.unsqueeze(-1) * E).sum(3)     # (B, T, N, De)

            # in_agg: target k (person/null_in) attends over N incoming sources
            scores_in   = self.pool_in(E[:, :, :, :N + 1, :]).squeeze(-1)  # (B, T, N, N+1)
            scores_in_t = scores_in.permute(0, 1, 3, 2)                    # (B, T, N+1, N)
            ev_in_t     = ev_sq[:, :, :, :N + 1].permute(0, 1, 3, 2)      # (B, T, N+1, N)
            scores_in_t = scores_in_t.masked_fill(ev_in_t == 0, float('-inf'))
            safe_in     = scores_in_t.isinf().all(dim=-1, keepdim=True)
            scores_in_t = scores_in_t.masked_fill(safe_in, 0.0)
            alpha_in    = F.softmax(scores_in_t, dim=-1) * ev_in_t         # (B, T, N+1, N)
            E_src_col   = E[:, :, :, :N + 1, :].permute(0, 1, 3, 2, 4)   # (B, T, N+1, N, De)
            in_agg_full = (alpha_in.unsqueeze(-1) * E_src_col).sum(3)      # (B, T, N+1, De)
            in_agg_p    = in_agg_full[:, :, :N, :]                         # (B, T, N, De)
            in_agg_ni   = in_agg_full[:, :, N:N + 1, :]                    # (B, T, 1, De)

            v_src = self.norm_src(
                v_src + self.upd_src(torch.cat([v_src, out_agg], dim=-1))
            )
            v_tgt_p = self.norm_tgt(
                v_tgt[:, :, :N, :] + self.upd_tgt(
                    torch.cat([v_tgt[:, :, :N, :], in_agg_p], dim=-1)
                )
            )  # (B, T, N, De)

            v_ni = self.norm_nullin(
                v_tgt[:, :, N:N + 1, :] + self.upd_nullin(
                    torch.cat([v_tgt[:, :, N:N + 1, :], in_agg_ni], dim=-1)
                )
            )  # (B, T, 1, De)

            v_tgt = torch.cat([v_tgt_p, v_ni, v_tgt[:, :, N + 1:, :]], dim=2)

            # ── ⑤ Re-inject updated nodes into edges ──────────────────────────
            src_exp = v_src.unsqueeze(3).expand(B, T, N, Tl, De)
            tgt_exp = v_tgt.unsqueeze(2).expand(B, T, N, Tl, De)
            E = self.norm_inj(
                E + self.inject(torch.cat([E, src_exp, tgt_exp], dim=-1))
            ) * ev

        return E, v_src, v_tgt


class _SocialReadoutHead(nn.Module):
    """Small residual MLP head for per-edge logit.
    LayerNorm instead of BatchNorm1d: independent of batch size and zero-padding ratio."""
    def __init__(self, dim: int, scale: int = 4):
        super().__init__()
        self.fc1    = nn.Linear(dim, dim // scale, bias=False)
        self.ln1    = nn.LayerNorm(dim // scale)
        self.fc2    = nn.Linear(dim // scale, dim // scale ** 2, bias=False)
        self.ln2    = nn.LayerNorm(dim // scale ** 2)
        self.res_fc = nn.Linear(dim, dim // scale ** 2)
        self.fc_out = nn.Linear(dim // scale ** 2, 1)

    def forward(self, x):
        z = torch.relu(self.ln1(self.fc1(x)))
        h = torch.relu(self.ln2(self.fc2(z)) + self.res_fc(x))
        return self.fc_out(h)


class GazeGraphBlock(nn.Module):
    """Unified directed gaze graph following idea.md design.

    Edge tensor E: (B, T, N, Tl, De)   Tl = N+2
      [0..N-1]  person targets
      [N]       null_in  — in-frame, looking at scene object
      [N+1]     null_out — out-of-frame

    Per refinement layer (×L):
      row-attn → col-attn (null_out excluded) → edge-refresh → node-update

    Readout (all T frames → (B,T,N,N) / (B,T,N) outputs):
      LAH    : head_lah(E[:,:,:,:N])
      LAEO   : head_laeo(cat(E[i,j], E[j,i]))       — learned MLP
      SA     : head_sa(cat(E[i→null_in], E[j→null_in], |diff|, E[i→j], E[j→i]))
      null_in : head_null_in(E[:,:,:,N])
      null_out: head_null_out(E[:,:,:,N+1])
    """

    def __init__(
        self,
        token_dim: int,
        edge_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        use_prior: bool = True,
        prior_weight: float = 0.5,
        use_node_xattn: bool = True,
    ):
        super().__init__()
        D, De = token_dim, edge_dim
        self.De             = De
        self.use_prior      = use_prior
        self.use_node_xattn = use_node_xattn

        # ── Null node parameters (one per null type) ──────────────────────────
        self.null_in_node  = nn.Parameter(torch.zeros(D))
        self.null_out_node = nn.Parameter(torch.zeros(D))

        # ── Target type embedding: person=0, null_in=1, null_out=2 ────────────
        self.type_emb = nn.Embedding(3, De)

        # ── Source node: heatmap cross-attention (kept from prior design) ─────
        hm_grid = 8
        self.hm_grid = hm_grid
        self.hm_pool = nn.AdaptiveAvgPool2d(hm_grid)
        if use_node_xattn:
            self.hm_proj        = nn.Linear(1, D)
            self.hm_pos_emb     = nn.Parameter(torch.randn(hm_grid * hm_grid, D) * 0.02)
            self.src_xattn      = CrossAttention(D, num_heads=heads)
            self.src_xattn_norm = nn.LayerNorm(D)
        else:
            self.hm_proj = self.hm_pos_emb = self.src_xattn = self.src_xattn_norm = None

        # ── Node projections ──────────────────────────────────────────────────
        self.node_src_proj = nn.Linear(D, De)   # source (XAttn-enriched)
        self.node_tgt_proj = nn.Linear(D, De)   # all targets (persons + nulls)

        # ── Target node: incoming gaze message from people looking at this bbox ─
        self.tgt_msg_mlp  = MLP(2 * D, D, D)
        self.tgt_msg_norm = nn.LayerNorm(D)
        nn.init.zeros_(self.tgt_msg_mlp.fc2.weight)
        nn.init.zeros_(self.tgt_msg_mlp.fc2.bias)

        # ── Edge scalar prior projection (1-D → De) ─────────────────────────
        self.linear_edge = nn.Linear(1, De)
        if use_prior:
            self.prior_w = nn.Parameter(torch.tensor(prior_weight))

        # ── Edge init MLP: cat(src_De, tgt_De, edge_De, type_De) → De ────────
        self.mlp_init = MLP(4 * De, De, De)

        # ── Unified refiner ───────────────────────────────────────────────────
        self.refiner = _UnifiedRefiner(De, num_layers, heads)

        # ── Readout heads ─────────────────────────────────────────────────────
        # SA:   ni_i   || ni_j          || |ni_i - ni_j|   (pure scene-gaze comparison)
        self.head_lah      = _SocialReadoutHead(De)
        self.head_laeo     = _SocialReadoutHead(2 * De)   # cat(E[i→j], E[j→i])
        self.head_sa       = _SocialReadoutHead(5 * De)   # cat(ni_i, no_j, |ni_i-ni_j|, E[i→j], E[j→i])
        self.head_null_in  = _SocialReadoutHead(De)
        self.head_null_out = _SocialReadoutHead(De)

    @staticmethod
    def _safe_kpm(kpm: torch.Tensor) -> torch.Tensor:
        return kpm & ~kpm.all(dim=1, keepdim=True)

    def forward(
        self,
        person_tokens,    # (B, T, N, D)
        num_valid_people, # (B,)
        gaze_vecs,        # (B, T, N, 2)
        head_bboxes,      # (B, T, N, 4)
        gaze_heatmaps,    # (B, T, N, Hh, Ww)
        inout_logits,     # (B, T, N)   raw logit from inout_decoder
    ):
        """
        Returns:
            lah_mat:    (B, T, N, N)   LAH logits for all T frames
            laeo_mat:   (B, T, N, N)   LAEO logits (learned MLP)
            sa_mat:     (B, T, N, N)   SA logits  (symmetric, all T frames)
            null_in:    (B, T, N)
            null_out:   (B, T, N)
            edge_valid: (B, N, 2N+2)   [0:N]=LAH, [N:2N]=SA-proxy,
                                        [2N]=null_in, [2N+1]=null_out
        """
        B, T, N, D = person_tokens.shape
        Hh, Ww = gaze_heatmaps.shape[-2:]
        device, dtype = person_tokens.device, person_tokens.dtype
        De = self.De
        Tl = N + 2

        # ── Validity mask (frame-independent) ────────────────────────────────
        node_valid = (
            torch.arange(N, device=device).view(1, N)
            >= (N - num_valid_people.view(B, 1))
        )  # (B, N)
        eye = torch.eye(N, device=device, dtype=torch.bool)

        # person-to-person: valid src AND valid tgt AND no self-loop
        p2p_valid = node_valid.unsqueeze(2) & node_valid.unsqueeze(1) & ~eye.unsqueeze(0)
        # null_in / null_out: any valid source
        null_valid = node_valid.unsqueeze(2)  # (B, N, 1)

        ev_bool = torch.cat([
            p2p_valid,                           # (B, N, N)   person targets
            null_valid.expand(B, N, 1),          # null_in
            null_valid.expand(B, N, 1),          # null_out
        ], dim=2)  # (B, N, Tl)

        # expand over T
        ev_bool_T = ev_bool.unsqueeze(1).expand(B, T, N, Tl)  # (B, T, N, Tl)
        ev = ev_bool_T.unsqueeze(-1).to(dtype)                 # (B, T, N, Tl, 1)

        # ── Source node init: Pi' = LN(Pi + XAttn(Pi, hm_i)) ─────────────────
        hm_flat  = gaze_heatmaps.reshape(B * T * N, 1, Hh, Ww).detach()
        P_hm     = self.hm_grid ** 2
        hm_small = self.hm_pool(hm_flat)                        # (B*T*N, 1, g, g)
        if self.use_node_xattn:
            hm_feat = (
                self.hm_proj(hm_small.reshape(B * T * N, P_hm, 1).to(dtype))
                + self.hm_pos_emb.to(dtype)
            )                                                    # (B*T*N, P_hm, D)
            src_q     = person_tokens.reshape(B * T * N, 1, D)
            src_prime = self.src_xattn_norm(
                src_q + self.src_xattn(src_q, hm_feat)
            ).reshape(B, T, N, D)
        else:
            src_prime = person_tokens   # (B, T, N, D)
            hm_feat = None

        # ── Heatmap normalisation for overlap prior ───────────────────────────
        hm_norm = hm_flat.squeeze(1).reshape(B * T, N, Hh, Ww).float()
        hm_norm = torch.relu(hm_norm)
        hm_norm = hm_norm / (hm_norm.sum((-2, -1), keepdim=True) + 1e-6)

        # ── Geometric edge features ───────────────────────────────────────────
        bboxes_bt = head_bboxes.reshape(B * T, N, 4).float()
        overlap   = _compute_bbox_overlap(hm_norm, bboxes_bt)           # (BT, N, N)
        overlap   = overlap.to(dtype).reshape(B, T, N, N)

        in_prob          = torch.sigmoid(inout_logits)                  # (B, T, N)
        person_bbox_mass = overlap.sum(-1).clamp(max=1.0)              # (B, T, N)
        null_in_prior    = 1.0 - person_bbox_mass
        null_out_prior   = 1.0 - in_prob

        # p2p prior: cos(gaze_vec[i], normalize(center[j] - center[i]))
        centers  = (head_bboxes[..., :2] + head_bboxes[..., 2:]) * 0.5  # (B, T, N, 2)
        dir_ij   = F.normalize(
            centers.unsqueeze(3) - centers.unsqueeze(2), dim=-1
        )                                                                 # (B, T, N, N, 2)
        align    = (gaze_vecs.unsqueeze(3) * dir_ij).sum(-1)             # (B, T, N, N) ∈ [-1,1]

        feat_p2p = align.unsqueeze(-1)                                   # (B, T, N, N, 1)
        feat_ni  = null_in_prior.unsqueeze(3).unsqueeze(-1)
        feat_no  = null_out_prior.unsqueeze(3).unsqueeze(-1)
        feat_all = torch.cat([feat_p2p, feat_ni, feat_no], dim=3)       # (B, T, N, Tl, 1)

        # ── Node projections ──────────────────────────────────────────────────
        v_src = self.node_src_proj(src_prime)   # (B, T, N, De)

        # Target person update: for target j, aggregate source tokens whose
        # heatmaps overlap bbox_j, then inject the incoming-looking message.
        tgt_scores = overlap.masked_fill(~p2p_valid.unsqueeze(1), float("-inf"))
        no_incoming = torch.isinf(tgt_scores).all(dim=2, keepdim=True)
        tgt_scores = tgt_scores.masked_fill(no_incoming, 0.0)
        tgt_w = F.softmax(tgt_scores, dim=2).masked_fill(no_incoming, 0.0)
        tgt_msg = torch.einsum("btij,btid->btjd", tgt_w, person_tokens)
        tgt_gate = overlap.masked_fill(~p2p_valid.unsqueeze(1), 0.0).amax(dim=2)
        tgt_delta = self.tgt_msg_norm(
            self.tgt_msg_mlp(torch.cat([person_tokens, tgt_msg], dim=-1))
        )
        tgt_person_tokens = (
            person_tokens + tgt_gate.unsqueeze(-1).to(dtype) * tgt_delta
        ).to(dtype)

        null_in_t  = self.null_in_node.to(dtype).view(1, 1, 1, D).expand(B, T, 1, D)
        null_out_t = self.null_out_node.to(dtype).view(1, 1, 1, D).expand(B, T, 1, D)
        tgt_tokens = torch.cat([tgt_person_tokens, null_in_t, null_out_t], dim=2)  # (B, T, Tl, D)
        v_tgt = self.node_tgt_proj(tgt_tokens)  # (B, T, Tl, De)

        # ── Type embeddings ───────────────────────────────────────────────────
        type_ids = torch.cat([
            torch.zeros(N, dtype=torch.long, device=device),
            torch.ones(1,  dtype=torch.long, device=device),
            torch.full((1,), 2, dtype=torch.long, device=device),
        ])  # (Tl,)
        type_e = self.type_emb(type_ids).view(1, 1, 1, Tl, De)  # broadcast

        # ── Edge initialisation ───────────────────────────────────────────────
        src_proj     = v_src.unsqueeze(3).expand(B, T, N, Tl, De)
        tgt_proj     = v_tgt.unsqueeze(2).expand(B, T, N, Tl, De)
        if self.use_prior:
            edge_feat_e = self.prior_w * self.linear_edge(
                feat_all.reshape(B * T * N * Tl, 1)
            ).reshape(B, T, N, Tl, De)
        else:
            edge_feat_e = torch.zeros(B, T, N, Tl, De, device=device, dtype=dtype)
        type_exp = type_e.expand(B, T, N, Tl, De)

        E = self.mlp_init(
            torch.cat([src_proj, tgt_proj, edge_feat_e, type_exp], dim=-1)
            .reshape(B * T * N * Tl, 4 * De)
        ).reshape(B, T, N, Tl, De) * ev

        # ── KPMs ─────────────────────────────────────────────────────────────
        row_kpm = self._safe_kpm((~ev_bool_T).reshape(B * T * N, Tl))
        col_ev  = ev_bool_T[:, :, :, :N + 1]
        col_kpm = self._safe_kpm(
            (~col_ev).permute(0, 1, 3, 2).reshape(B * T * (N + 1), N)
        )

        deg_out = ev_bool_T[:, :, :, :N].sum(3).clamp(min=1).to(dtype).unsqueeze(-1)
        deg_in  = ev_bool_T.sum(2).clamp(min=1).to(dtype).unsqueeze(-1)

        # ── Refinement ────────────────────────────────────────────────────────
        E, v_src, v_tgt = self.refiner(E, ev, row_kpm, col_kpm, v_src, v_tgt, deg_out, deg_in)

        # ── Readout (all T frames) ────────────────────────────────────────────
        E_pp = E[:, :, :, :N, :]   # person-to-person edges (B, T, N, N, De)

        lah_mat = self.head_lah(
            E_pp.reshape(B * T * N * N, De)
        ).reshape(B, T, N, N)

        # LAEO: MLP(cat(E[i→j], E[j→i])) — average both orderings for exact symmetry
        laeo_mat = self.head_laeo(
            torch.cat([E_pp, E_pp.transpose(2, 3)], dim=-1)
            .reshape(B * T * N * N, 2 * De)
        ).reshape(B, T, N, N)
        laeo_mat = (laeo_mat + laeo_mat.transpose(2, 3)) * 0.5

        # SA: ni_i || no_j || |ni_i - ni_j| || E[i→j] || E[j→i]
        ni     = E[:, :, :, N,     :]   # E[i→null_in]  (B, T, N, De)
        no     = E[:, :, :, N + 1, :]   # E[i→null_out] (B, T, N, De)
        ni_i   = ni.unsqueeze(3).expand(B, T, N, N, De)   # i's null_in, broadcast over j
        no_j   = no.unsqueeze(2).expand(B, T, N, N, De)   # j's null_out, broadcast over i
        ni_j   = ni.unsqueeze(2).expand(B, T, N, N, De)   # j's null_in (for diff only)
        ni_dif = (ni_i - ni_j).abs()
        sa_mat = self.head_sa(
            torch.cat([ni_i, no_j, ni_dif, E_pp, E_pp.transpose(2, 3)], dim=-1)
            .reshape(B * T * N * N, 5 * De)
        ).reshape(B, T, N, N)
        sa_mat = (sa_mat + sa_mat.transpose(2, 3)) * 0.5

        # null_in / null_out
        null_in_out  = self.head_null_in(
            E[:, :, :, N, :].reshape(B * T * N, De)
        ).reshape(B, T, N)
        null_out_out = self.head_null_out(
            E[:, :, :, N + 1, :].reshape(B * T * N, De)
        ).reshape(B, T, N)

        # edge_valid: (B, N, 2N+2) — frame-independent
        edge_valid = torch.cat([
            ev_bool[:, :, :N],   # [0:N]   LAH person targets
            ev_bool[:, :, :N],   # [N:2N]  SA proxy (same p2p mask)
            ev_bool[:, :, N:],   # [2N:2N+2] null_in + null_out
        ], dim=2)

        return (
            lah_mat.float(), laeo_mat.float(), sa_mat.float(),
            null_in_out.float(), null_out_out.float(), edge_valid,
        )
