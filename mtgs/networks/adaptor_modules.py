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

    # overlap[b, i, j] = Σ_{hw} hm_norm[b,i,h,w] * mask[b,j,h,w]
    #   Contract H,W directly with einsum — avoids materialising the
    #   (BT, N_src, N_tgt, H, W) outer product (huge for large N at test time).
    overlap = torch.einsum("bihw,bjhw->bij", hm_norm, mask)  # (BT, N_src, N_tgt)
    return overlap.to(dtype)  # already in [0,1]: hm_norm sums to 1


class _RefinerLayer(nn.Module):
    """One dual-role edge-refinement layer with its OWN weights (no cross-layer
    sharing). Steps: row-attn → col-attn → edge-refresh → node-update →
    re-inject → temporal-edge-attn (only when T > 1).

    E shape throughout: (B, T, N, Tl, De)   where Tl = N + 2.
    Column attention covers only the first N+1 targets (null_out excluded).
    """

    def __init__(self, edge_dim: int, heads: int):
        super().__init__()
        De = edge_dim

        _enc = lambda: nn.TransformerEncoderLayer(
            d_model=De, nhead=heads, dim_feedforward=2 * De,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.row      = _enc()
        self.col      = _enc()
        self.temporal = _enc()   # edge temporal consistency (2-D)

        self.refresh  = MLP(2 * De, De, De)
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

    def forward(self, E, ev, ev_sq, row_kpm, col_kpm, v_src, v_tgt):
        B, T, N, Tl, De = E.shape
        E_in = E   # edge state before this layer's row/col attention

        # ── ① Row: source i attends over all Tl targets ───────────────────
        row_context = self.row(
            E_in.reshape(B * T * N, Tl, De), src_key_padding_mask=row_kpm
        ).reshape(B, T, N, Tl, De) * ev

        # ── ② Col: target k (person + null_in) attends over N sources ─────
        #    Parallel to row: col also reads from E_in (not row_context).
        #    null_out excluded; its slot is filled with E_in so refresh has
        #    a well-defined 2-way concat for every edge position.
        E_col_in = E_in[:, :, :, :N + 1, :]
        E_col_out_N1 = self.col(
            E_col_in.permute(0, 1, 3, 2, 4).reshape(B * T * (N + 1), N, De),
            src_key_padding_mask=col_kpm,
        ).reshape(B, T, N + 1, N, De).permute(0, 1, 3, 2, 4)  # (B, T, N, N+1, De)
        col_context = torch.cat(
            [E_col_out_N1, E_in[:, :, :, N + 1:, :]], dim=3
        ) * ev

        # ── ③ Edge refresh: parallel row+col context, E_in as residual base ──
        #    edge = LN(E_in + MLP(concat(row_context, col_context)))
        E = self.norm_e(
            E_in + self.refresh(torch.cat([row_context, col_context], dim=-1))
        ) * ev

        # ── ④ Node update: learned attention pooling ──────────────────────
        # out_agg: source i attends over all Tl outgoing edges
        scores_out = self.pool_out(E).squeeze(-1)                      # (B, T, N, Tl)
        scores_out = scores_out.masked_fill(ev_sq == 0, float('-inf'))
        safe_out   = scores_out.isinf().all(dim=-1, keepdim=True)
        scores_out = scores_out.masked_fill(safe_out, 0.0)
        alpha_out  = F.softmax(scores_out, dim=-1) * ev_sq            # (B, T, N, Tl)
        out_agg    = (alpha_out.unsqueeze(-1) * E).sum(3)             # (B, T, N, De)

        # in_agg: target k (person/null_in) attends over N incoming sources
        E_col       = E[:, :, :, :N + 1, :].permute(0, 1, 3, 2, 4)  # (B, T, N+1, N, De)
        scores_in_t = self.pool_in(E_col).squeeze(-1)                  # (B, T, N+1, N)
        ev_in_t     = ev_sq[:, :, :, :N + 1].permute(0, 1, 3, 2)        # (B, T, N+1, N)
        scores_in_t = scores_in_t.masked_fill(ev_in_t == 0, float('-inf'))
        safe_in     = scores_in_t.isinf().all(dim=-1, keepdim=True)
        scores_in_t = scores_in_t.masked_fill(safe_in, 0.0)
        alpha_in    = F.softmax(scores_in_t, dim=-1) * ev_in_t          # (B, T, N+1, N)
        in_agg_full = (alpha_in.unsqueeze(-1) * E_col).sum(3)           # (B, T, N+1, De)
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

        # ── ⑥ Temporal edge attention: each edge attends over its T frames ──
        #    Edge validity is frame-independent, so no temporal kpm is needed;
        #    globally-invalid edges are re-zeroed by * ev afterwards.
        if T > 1:
            E_t = E.permute(0, 2, 3, 1, 4).reshape(B * N * Tl, T, De)
            E_t = self.temporal(E_t)
            E = E_t.reshape(B, N, Tl, T, De).permute(0, 3, 1, 2, 4) * ev

        return E, v_src, v_tgt


class _UnifiedRefiner(nn.Module):
    """Stack of dual-role edge-refinement layers, each with independent weights
    (2-E) and per-layer temporal edge attention (2-D).

    E shape throughout: (B, T, N, Tl, De)   where Tl = N + 2.
    """

    def __init__(self, edge_dim: int, num_layers: int, heads: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_RefinerLayer(edge_dim, heads) for _ in range(num_layers)]
        )

    @staticmethod
    def _safe_kpm(kpm: torch.Tensor) -> torch.Tensor:
        # Fully-masked sequences cause NaN in TransformerEncoderLayer.
        # Unmask them; their output is discarded by multiplying with ev.
        return kpm & ~kpm.all(dim=1, keepdim=True)

    def forward(self, E, ev, row_kpm, col_kpm, v_src, v_tgt):
        """
        E:       (B, T, N, Tl, De)
        ev:      (B, T, N, Tl, 1)    float 0/1 validity mask
        row_kpm: (B*T*N, Tl)
        col_kpm: (B*T*(N+1), N)      null_out column excluded
        v_src:   (B, T, N, De)
        v_tgt:   (B, T, Tl, De)
        """
        ev_sq = ev.squeeze(-1)  # (B, T, N, Tl) float; constant across layers
        for layer in self.layers:
            E, v_src, v_tgt = layer(E, ev, ev_sq, row_kpm, col_kpm, v_src, v_tgt)
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
      SA     : head_sa(cat(ni_i, ni_j, |diff|, E[i→j], E[j→i]))  — edge+null_in based
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
        use_node_xattn: bool = True,   # deprecated (V14): node init no longer uses heatmap XAttn
        face_dim: int = 768,           # raw GazeEncoder token dim (pre-adaptor facial feature)
        laeo_derive: str = "lah_min",  # "decoder": use head_laeo | "lah_min": derived downstream
    ):
        super().__init__()
        D, De = token_dim, edge_dim
        self.De             = De
        self.use_prior      = use_prior
        self.laeo_derive    = laeo_derive

        # ── Null node parameters (one per null type) ──────────────────────────
        self.null_in_node  = nn.Parameter(torch.zeros(D))
        self.null_out_node = nn.Parameter(torch.zeros(D))

        # ── Target type embedding: person=0, null_in=1, null_out=2 ────────────
        self.type_emb = nn.Embedding(3, De)

        # ── Unified node init (V14): node = LN(person_token + Linear_face(gaze)) + geom
        #    scene context is already carried by person_token (ViT-Adaptor mixed);
        #    facial detail is re-injected from detached raw GazeEncoder tokens.
        #    src and tgt persons share this single init (roles diverge only in refiner).
        self.face_proj = nn.Linear(face_dim, D)
        nn.init.zeros_(self.face_proj.weight)   # start as no-op (safe A/B)
        nn.init.zeros_(self.face_proj.bias)
        self.node_in_norm = nn.LayerNorm(D)

        # ── Node geometry encoding (2-C): [cx, cy, w, h, gaze_vec] → D ────────
        #    zero-init last layer so it starts as a no-op.
        self.node_geom_mlp = MLP(6, D, D)
        nn.init.zeros_(self.node_geom_mlp.fc2.weight)
        nn.init.zeros_(self.node_geom_mlp.fc2.bias)

        # ── Single node projection shared by source & target roles (V14) ──────
        self.node_proj = nn.Linear(D, De)

        # ── Edge scalar prior projection (4 channels → De) ───────────────────
        #    channel 0: primary prior (p2p=cosine, null_in/out=routing prior)
        #    channel 1: heatmap overlap[i→j] (LAH grounding, 2-B); 0 for nulls
        #    channel 2-3: rel_pos = normalize(center_j - center_i) (V13); 0 for nulls
        self.linear_edge = nn.Linear(4, De)
        if use_prior:
            self.prior_w = nn.Parameter(torch.tensor(prior_weight))

        # ── Edge init MLP: cat(src_De, tgt_De, edge_De, type_De) → De ────────
        self.mlp_init = MLP(4 * De, De, De)

        # ── Unified refiner ───────────────────────────────────────────────────
        self.refiner = _UnifiedRefiner(De, num_layers, heads)

        # ── Readout heads ─────────────────────────────────────────────────────
        self.head_lah      = _SocialReadoutHead(De)
        self.head_laeo     = _SocialReadoutHead(2 * De)   # cat(E[i→j], E[j→i])
        # SA: edge-based, cat(ni_i, ni_j, |diff|, E[i→j], E[j→i])
        self.head_sa       = _SocialReadoutHead(5 * De)
        self.head_null_in  = _SocialReadoutHead(De)
        self.head_null_out = _SocialReadoutHead(De)

    @staticmethod
    def _safe_kpm(kpm: torch.Tensor) -> torch.Tensor:
        return kpm & ~kpm.all(dim=1, keepdim=True)

    def forward(
        self,
        person_tokens,    # (B, T, N, D)   scene-contextualised (ViT-Adaptor output)
        num_valid_people, # (B,)
        gaze_vecs,        # (B, T, N, 2)
        head_bboxes,      # (B, T, N, 4)
        gaze_heatmaps,    # (B, T, N, Hh, Ww)
        inout_logits,     # (B, T, N)   raw logit from inout_decoder
        gaze_feat,        # (B, T, N, face_dim)  raw GazeEncoder tokens (pre-adaptor face)
    ):
        """
        Returns:
            lah_mat:    (B, T, N, N)   LAH logits for all T frames
            laeo_mat:   (B, T, N, N)   LAEO logits (learned MLP)
            sa_mat:     (B, T, N, N)   SA logits  (node-based, asymmetric)
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

        # ── Heatmap (detached) — used only for the edge overlap prior now (V14) ─
        hm_flat  = gaze_heatmaps.reshape(B * T * N, 1, Hh, Ww).detach()

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

        # Detach gaze_vecs: the graph consumes predicted gaze as fixed geometric
        # evidence (like the detached heatmap and the data-only bboxes). Social-graph
        # gradients must not flow back into the gaze-regression head, keeping the
        # front-end origin-faithful.  Used by both the align prior and the geom node feat.
        gaze_vecs = gaze_vecs.detach()

        # p2p prior: cos(gaze_vec[i], normalize(center[j] - center[i]))
        centers  = (head_bboxes[..., :2] + head_bboxes[..., 2:]) * 0.5  # (B, T, N, 2)
        wh       = (head_bboxes[..., 2:] - head_bboxes[..., :2])        # (B, T, N, 2)
        dir_ij   = F.normalize(
            centers.unsqueeze(3) - centers.unsqueeze(2), dim=-1
        )                                                                 # (B, T, N, N, 2)
        align    = (gaze_vecs.unsqueeze(3) * dir_ij).sum(-1)             # (B, T, N, N) ∈ [-1,1]

        # Edge prior, 4 channels: [primary, heatmap-overlap, rel_pos_dx, rel_pos_dy].
        #   (2-B) overlap + (V13) rel_pos = dir_ij = normalize(center_j - center_i).
        zeros_ch = torch.zeros_like(null_in_prior)                      # (B, T, N)
        rel_pos  = dir_ij.to(dtype)                                     # (B, T, N, N, 2)
        zeros2   = torch.zeros(B, T, N, 1, 2, device=device, dtype=dtype)
        feat_p2p = torch.cat(
            [align.unsqueeze(-1), overlap.unsqueeze(-1), rel_pos], dim=-1
        )                                                               # (B, T, N, N, 4)
        feat_ni  = torch.cat(
            [torch.stack([null_in_prior, zeros_ch], dim=-1).unsqueeze(3), zeros2],
            dim=-1,
        )                                                               # (B, T, N, 1, 4)
        feat_no  = torch.cat(
            [torch.stack([null_out_prior, zeros_ch], dim=-1).unsqueeze(3), zeros2],
            dim=-1,
        )                                                               # (B, T, N, 1, 4)
        feat_all = torch.cat([feat_p2p, feat_ni, feat_no], dim=3)       # (B, T, N, Tl, 4)

        # ── Unified node init (V14): node = LN(person_token + face) + geom ─────
        #    person_token = scene context; face = detached raw GazeEncoder token.
        geom     = torch.cat([centers, wh, gaze_vecs], dim=-1).to(dtype)  # (B, T, N, 6)
        geom_emb = self.node_geom_mlp(geom)                              # (B, T, N, D)
        face     = self.face_proj(gaze_feat.detach().to(dtype))         # (B, T, N, D)
        node     = self.node_in_norm(person_tokens + face) + geom_emb   # (B, T, N, D)

        # src and tgt persons share the same node feature; nulls are target-only.
        null_in_t  = self.null_in_node.to(dtype).view(1, 1, 1, D).expand(B, T, 1, D)
        null_out_t = self.null_out_node.to(dtype).view(1, 1, 1, D).expand(B, T, 1, D)
        tgt_tokens = torch.cat([node, null_in_t, null_out_t], dim=2)    # (B, T, Tl, D)
        v_tgt = self.node_proj(tgt_tokens)                             # (B, T, Tl, De)
        v_src = v_tgt[:, :, :N, :]                                     # persons as sources

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
                feat_all.reshape(B * T * N * Tl, 4)
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

        # ── Refinement ────────────────────────────────────────────────────────
        E, v_src, v_tgt = self.refiner(E, ev, row_kpm, col_kpm, v_src, v_tgt)

        # ── Readout (all T frames) ────────────────────────────────────────────
        E_pp = E[:, :, :, :N, :]   # person-to-person edges (B, T, N, N, De)

        lah_mat = self.head_lah(
            E_pp.reshape(B * T * N * N, De)
        ).reshape(B, T, N, N)

        # LAEO: only the "decoder" mode uses head_laeo. "lah_min" derives LAEO from
        # LAH downstream (mtgs_net), so skip the head_laeo forward entirely here to
        # avoid wasted compute on a discarded output.
        if self.laeo_derive == "decoder":
            # MLP(cat(E[i→j], E[j→i])) — average both orderings for exact symmetry
            laeo_mat = self.head_laeo(
                torch.cat([E_pp, E_pp.transpose(2, 3)], dim=-1)
                .reshape(B * T * N * N, 2 * De)
            ).reshape(B, T, N, N)
            laeo_mat = (laeo_mat + laeo_mat.transpose(2, 3)) * 0.5
        else:
            laeo_mat = None

        # SA: edge-based, cat(ni_i, ni_j, |diff|, E[i→j], E[j→i])
        ni     = E[:, :, :, N, :]                                          # (B,T,N,De) null_in edge per person
        ni_i   = ni.unsqueeze(3).expand(B, T, N, N, De)
        ni_j   = ni.unsqueeze(2).expand(B, T, N, N, De)
        E_ji   = E_pp.transpose(2, 3)
        sa_mat = self.head_sa(
            torch.cat([ni_i, ni_j, (ni_i - ni_j).abs(), E_pp, E_ji], dim=-1)
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
            lah_mat.float(),
            laeo_mat.float() if laeo_mat is not None else None,
            sa_mat.float(),
            null_in_out.float(), null_out_out.float(), edge_valid,
        )
