# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import itertools
import os
from typing import List, Tuple, Union

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mtgs.utils import pair
from mtgs.networks.adaptor_modules import InteractionBlock, GazeGraphBlock

import logging

logger = logging.getLogger(__name__)


# ==================================================================================================================== #
#                                                INTERACT-NET ARCHITECTURE                                                #
# ==================================================================================================================== #
class MTGS(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        token_dim: int = 768,
        image_size: int = 224,
        gaze_feature_dim: int = 512,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_num_global_tokens: int = 1,
        encoder_mlp_ratio: float = 4.0,
        encoder_use_qkv_bias: bool = True,
        encoder_drop_rate: float = 0.0,
        encoder_attn_drop_rate: float = 0.0,
        encoder_drop_path_rate: float = 0.0,
        decoder_feature_dim: int = 256,
        decoder_hooks: list = [2, 5, 8, 11],
        decoder_hidden_dims: list = [96, 192, 384, 768],
        decoder_use_bn: bool = False,
        proj_feature_dim: int = 128,
        temporal_context: int = 2,
        hm_size=(64, 64),  # width, height
        output="heatmap",
        gaze_graph_num_layers: int = 2,
        gaze_graph_edge_dim: int = 128,
        gaze_graph_use_prior: bool = True,
        gaze_graph_prior_weight: float = 0.5,
        gaze_graph_laeo_derive: str = "decoder",
        gaze_graph_use: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.token_dim = token_dim
        self.image_size = pair(image_size)
        self.hm_size = hm_size
        self.gaze_feature_dim = gaze_feature_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_num_global_tokens = encoder_num_global_tokens
        self.encoder_mlp_ratio = encoder_mlp_ratio
        self.encoder_use_qkv_bias = encoder_use_qkv_bias
        self.encoder_drop_rate = encoder_drop_rate
        self.encoder_attn_drop_rate = encoder_attn_drop_rate
        self.encoder_drop_path_rate = encoder_drop_path_rate
        self.decoder_feature_dim = decoder_feature_dim
        self.decoder_hooks = decoder_hooks
        self.decoder_hidden_dims = decoder_hidden_dims
        self.decoder_use_bn = decoder_use_bn
        window_size = 2 * temporal_context + 1

        # gaze encoder
        self.gaze_encoder = GazeEncoder(
            token_dim=token_dim, feature_dim=gaze_feature_dim
        )

        self.gaze_encoder_temporal = TransformerBlock(
            dim=gaze_feature_dim, num_heads=8, mlp_ratio=0.25, drop_path_rate=0.3
        )

        # scene encoder: DINOv2
        _dinov2_cache = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if os.path.isdir(_dinov2_cache):
            self.encoder = torch.hub.load(_dinov2_cache, "dinov2_vitb14", source="local")
        else:
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        # scene, person interaction
        self.interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
        self.vit_adaptor = nn.Sequential(
            *[
                InteractionBlock(
                    dim=token_dim,
                    num_heads=encoder_num_heads,
                    drop_path=0.3,
                    cffn_ratio=0.25,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )

        # ── Interaction module (fixed): ViT-Adaptor → people_interaction (spatial)
        # + people_temporal. The social-prediction head is a standalone
        # GazeGraphBlock attached after the loop. ──────────────────────────────
        self.people_interaction = nn.Sequential(
            *[
                TransformerBlock(
                    dim=token_dim,
                    num_heads=encoder_num_heads,
                    mlp_ratio=0.25,
                    drop_path_rate=0.3,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )
        # people temporal
        self.people_temporal = nn.Sequential(
            *[
                TransformerBlock(
                    dim=token_dim,
                    num_heads=encoder_num_heads,
                    mlp_ratio=0.25,
                    drop_path_rate=0.3,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )
        # Social head: either the unified GazeGraphBlock (use=True) or the original
        # per-pair social decoder operating directly on person token pairs (use=False).
        self.gaze_graph_use = gaze_graph_use
        if gaze_graph_use:
            self.gaze_graph_block = GazeGraphBlock(
                token_dim=proj_feature_dim * len(self.interaction_indexes),
                edge_dim=gaze_graph_edge_dim,
                num_layers=gaze_graph_num_layers,
                use_prior=gaze_graph_use_prior,
                prior_weight=gaze_graph_prior_weight,
                face_dim=token_dim,   # raw GazeEncoder token dim (pre-adaptor face)
            )
        else:
            # Original social decoders: LAH (directed [h_i‖h_j]) + SA (symmetric
            # [s_i+s_j‖|s_i−s_j|]). LAEO derived as min(LAH_ij, LAH_ji).
            self.decoder_lah = LinearDecoderSocialGraph(
                proj_feature_dim * len(self.interaction_indexes)
            )
            self.decoder_sa = LinearDecoderSocialGraph(
                proj_feature_dim * len(self.interaction_indexes)
            )
        self.gaze_graph_laeo_derive = gaze_graph_laeo_derive

        # pair indices cache keyed by n — avoids rebuilding itertools.permutations every forward
        self._pair_indices_cache: dict = {}
        # reverse-pair index cache keyed by n — maps pair (s,d) -> index of (d,s).
        # Used to derive LAEO as logit-space AND (min) of both LAH directions.
        self._rev_pair_cache: dict = {}

        # temporal position embedding
        if window_size > 1:
            self.temp_emb = nn.Parameter(
                torch.zeros(window_size, gaze_feature_dim)
            )  # final used

        # gaze point decoder
        self.output = output
        if output == "heatmap":
            self.gaze_hm_decoder_new = ConditionalDPTDecoder(
                token_dim=token_dim,
                feature_dim=decoder_feature_dim,
                patch_size=patch_size,
                hooks=decoder_hooks,
                hidden_dims=decoder_hidden_dims,
                use_bn=decoder_use_bn,
            )

        # projection layers for social gaze prediction
        self.gaze_projs = nn.Sequential(
            *[
                nn.Linear(token_dim, proj_feature_dim, bias=True)
                for i in range(len(self.interaction_indexes))
            ]
        )
        self.inout_decoder = InOutDecoder(
            proj_feature_dim * len(self.interaction_indexes)
        )
        # Social gaze (LAH/LAEO/SA) is predicted by GazeGraphBlock's own heads.

    def forward(self, x):
        # Expected x = {"image": image, "heads": heads, "head_bboxes": head_bboxes, "coatt_ids": coatt_ids}

        # n = total nb of people, t = temporal window
        b, t, n, c, h, w = x["heads"].shape

        # Encode Gaze Tokens ===================================================
        gaze_emb = self.gaze_encoder.forward_backbone(
            x["heads"].view(b * t, n, c, h, w)
        )
        gaze_emb = gaze_emb.view(b, t, n, -1)  # (b, t, n, 512)

        # Apply temporal attention
        if t > 1:
            # add temporal position embedding
            temp_emb = self.temp_emb.unsqueeze(0).tile(b, 1, 1).unsqueeze(2)
            gaze_emb = gaze_emb + temp_emb
            # perform self-attention
            gaze_emb = self.gaze_encoder_temporal(
                gaze_emb.permute([0, 2, 1, 3]).reshape(b * n, t, -1), print_att=False
            )
            gaze_emb = gaze_emb.view(b, n, t, -1).permute([0, 2, 1, 3])

        # Predict gaze vector
        gaze_tokens, gaze_vec = self.gaze_encoder.forward_head(
            gaze_emb.reshape(b * t, n, -1), x["head_bboxes"].view(b * t, n, -1)
        )  # (b*t, n, 768), (b*t, n, 2)
        gaze_vec = gaze_vec.view(b, t, n, -1)

        # Tokenize Inputs ===================================================
        b, t, c, h_img, w_img = x["image"].shape
        image_tokens = self.encoder.prepare_tokens_with_masks(
            x["image"].view(b * t, c, h_img, w_img)
        )

        person_tokens = gaze_tokens.view(b * t, n, -1).clone()
        face_feat     = gaze_tokens.view(b, t, n, -1)   # raw GazeEncoder tokens (pre-adaptor face) for gaze_graph
        num_valid_b   = x["num_valid_people"].view(b, t).max(dim=1).values.clamp(min=1)  # (b,) for gaze_graph

        # Apply ViT Adaptor =================================================
        img_layers = []
        gaze_layers = []
        for i, layer in enumerate(self.vit_adaptor):
            indexes = self.interaction_indexes[i]
            vit_blocks = self.encoder.blocks[indexes[0] : indexes[-1] + 1]

            image_tokens, person_tokens = layer(
                image_tokens, person_tokens, vit_blocks, x["num_valid_people"],
            )

            # spatio-temporal social interaction: original spatial + temporal
            person_tokens = self.people_interaction[i](person_tokens)
            if t > 1:
                person_tokens = self.people_temporal[i](
                    person_tokens.view(b, t, n, -1)
                    .permute([0, 2, 1, 3])
                    .reshape(b * n, t, -1)
                )
                person_tokens = (
                    person_tokens.view(b, n, t, -1)
                    .permute([0, 2, 1, 3])
                    .reshape(b * t, n, -1)
                )

            # save intermediate outputs
            # for DinoV2, remove class token
            img_layers.append(image_tokens[:, 1:])
            gaze_layers.append(person_tokens)

        if self.output == "heatmap":
            # conditional DPT
            gaze_hm = self.gaze_hm_decoder_new(img_layers, gaze_layers, (h_img, w_img))
            _, _, hm_height, hm_width = gaze_hm.shape
            gaze_hm = gaze_hm.view(b, t, n, hm_height, hm_width)

        # project and concat person tokens from each gaze layer
        person_tokens = [
            self.gaze_projs[i](gaze_layer) for i, gaze_layer in enumerate(gaze_layers)
        ]
        person_tokens = torch.cat(person_tokens, axis=-1)

        # Classify inout ====================================================
        inout = self.inout_decoder(person_tokens.view(b * t * n, -1))  # (b*t*n, 1)

        # make person pairs — indices cached per n to avoid rebuilding every forward
        if n not in self._pair_indices_cache:
            self._pair_indices_cache[n] = torch.tensor(
                list(itertools.permutations(range(n), 2)), dtype=torch.long
            ).T  # (2, num_pairs)
        indices = self._pair_indices_cache[n]
        src_idx, dst_idx = indices[0], indices[1]
        num_pairs = src_idx.shape[0]

        # reverse-pair lookup (s,d) -> index of (d,s); used to derive LAEO.
        if n not in self._rev_pair_cache:
            pos = torch.full((n, n), -1, dtype=torch.long, device=indices.device)
            pos[src_idx, dst_idx] = torch.arange(num_pairs, device=indices.device)
            self._rev_pair_cache[n] = pos[dst_idx, src_idx]   # (num_pairs,)
        rev_idx = self._rev_pair_cache[n]

        if not self.gaze_graph_use:
            # ── Original social decoders (MTGS_origin) ─────────────────────────
            # Both LAH and SA read the SAME asymmetric pair concat [h_i ‖ h_j];
            # SA prediction is per-direction (not symmetrized). person_tokens is
            # indexed by (src_idx, dst_idx) permutations, matching label order.
            h_src = person_tokens[:, src_idx]   # (b*t, P, D)
            h_dst = person_tokens[:, dst_idx]
            person_token_pairs = torch.cat([h_src, h_dst], dim=-1).reshape(
                b * t * num_pairs, -1
            )                                   # (b*t*P, 2*D)
            lah   = self.decoder_lah(person_token_pairs).view(b * t, num_pairs)
            coatt = self.decoder_sa(person_token_pairs).view(b * t, num_pairs)
            # LAEO ⟺ mutual looking = logit-space AND (min) of both LAH directions
            laeo  = torch.minimum(lah, lah[:, rev_idx])
            return (
                None,
                gaze_vec,
                gaze_hm,
                inout.view(b, t, n),
                lah.view(b, t, num_pairs),
                laeo.view(b, t, num_pairs),
                coatt.view(b, t, num_pairs),
                None,   # null_in (gaze_graph 전용)
                None,   # null_out (gaze_graph 전용)
            )

        # ── gaze_graph: unified directed graph (persons + null_in + null_out) ──
        lah_mat, laeo_mat, sa_mat, null_in_mat, null_out_mat, edge_valid = (
            self.gaze_graph_block(
                person_tokens.view(b, t, n, -1),   # (B, T, N, D)
                num_valid_b,                        # (B,)
                gaze_vec.view(b, t, n, -1),        # (B, T, N, 2)
                x["head_bboxes"].view(b, t, n, -1),# (B, T, N, 4)
                gaze_hm,                           # (B, T, N, Hh, Ww)
                inout.view(b, t, n),               # (B, T, N) — raw logit
                face_feat,                         # (B, T, N, token_dim) — raw face for node init
            )
        )
        # lah_mat, laeo_mat, sa_mat: (B, T, N, N)
        # null_in_mat, null_out_mat: (B, T, N)
        # edge_valid: (B, N, 2N+2)

        # Mask invalid edges to large negative so they can't win per-target max
        ev_lah = edge_valid[:, :, :n]                   # (B, N, N)
        ev_sa  = edge_valid[:, :, n:2 * n]
        lah_mat  = lah_mat.masked_fill(
            ~ev_lah.unsqueeze(1).expand(b, t, n, n), -1e4
        )
        laeo_mat = laeo_mat.masked_fill(
            ~ev_lah.unsqueeze(1).expand(b, t, n, n), -1e4
        )
        sa_mat   = sa_mat.masked_fill(
            ~ev_sa.unsqueeze(1).expand(b, t, n, n), -1e4
        )

        # Gather directed pairs (b*t, P)
        # Dataset convention: pair (a,b) label = "b looks at a" (TARGET, LOOKER).
        # E[i→j] encodes "i looks at j", so lah_mat[b,a] = E[b→a] = "b looks at a" ✓
        lah   = lah_mat.view(b * t, n, n)[:, dst_idx, src_idx]
        if self.gaze_graph_laeo_derive == "lah_min":
            laeo = torch.minimum(lah, lah[:, rev_idx])
        else:
            laeo = laeo_mat.view(b * t, n, n)[:, dst_idx, src_idx]
        coatt = sa_mat.view(b * t, n, n)[:, src_idx, dst_idx]

        return (
            None,
            gaze_vec,
            gaze_hm,
            inout.view(b, t, n),
            lah.view(b, t, num_pairs),
            laeo.view(b, t, num_pairs),
            coatt.view(b, t, num_pairs),
            torch.sigmoid(null_in_mat),   # (B, T, N)  — null_in probs
            torch.sigmoid(null_out_mat),  # (B, T, N)  — null_out probs
        )


# ==================================================================================================================== #
#                                                   SHARINGAN BLOCKS                                                   #
# ==================================================================================================================== #


# ****************************************************** #
#                      GAZE ENCODER                      #
# ****************************************************** #
class GazeEncoder(nn.Module):
    def __init__(self, token_dim=768, feature_dim=512):
        super().__init__()

        self.feature_dim = feature_dim
        self.token_dim = token_dim

        base = models.resnet18(weights=None)  # type: ignore
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.backbone(dummy_head)
        embed_dim = dummy_head.size(1)

        self.gaze_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.pos_proj = nn.Linear(4, token_dim)

        self.gaze_predictor = nn.Sequential(  # self.gaze_predictor
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(inplace=True),
            # 2 = number of outputs (x, y) unit vector
            nn.Linear(feature_dim, 2),
            nn.Tanh(),
        )

    def forward_backbone(self, head):
        b, n, c, h, w = head.shape

        gaze_emb = self.backbone(head.view(-1, c, h, w)).flatten(
            1, -1
        )  # (b*n, embed_dim)

        return gaze_emb

    def forward_head(self, gaze_emb, head_bbox):
        b, n, p = head_bbox.shape
        head_bbox_emb = self.pos_proj(head_bbox.view(-1, p))  # (b*n, token_dim)

        gaze_token = (
            self.gaze_proj(gaze_emb.view(b * n, -1)) + head_bbox_emb
        )  # (b*n, token_dim)
        gaze_token = gaze_token.view(b, n, -1)  # (b, n, token_dim)

        gaze_vec = self.gaze_predictor(gaze_emb.view(b * n, -1))  # (b*n, 2)
        # normalize to unit vector
        gaze_vec = F.normalize(gaze_vec, p=2, dim=1)
        gaze_vec = gaze_vec.view(b, n, -1)  # (b, n, 2)

        return gaze_token, gaze_vec


# ****************************************************** #
#                     LINEAR DECODER                     #
# ****************************************************** #


class ResidualLinearBlock(nn.Module):
    def __init__(self, feature_dim, scale=2):
        super().__init__()
        self.scale = scale
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(feature_dim, feature_dim // scale, bias=False)
        self.bn1 = nn.BatchNorm1d(feature_dim // scale)
        self.fc2 = nn.Linear(feature_dim // scale, feature_dim // scale**2, bias=False)
        self.bn2 = nn.BatchNorm1d(feature_dim // scale**2)
        self.res_fc = nn.Linear(feature_dim, feature_dim // scale**2)

    def forward(self, x):
        z = torch.relu(self.bn1(self.fc1(x)))
        o = torch.relu(self.bn2(self.fc2(z)) + self.res_fc(x))
        return o


class InOutDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.block1 = ResidualLinearBlock(dim)
        self.block2 = ResidualLinearBlock(dim // 4)
        self.fc = nn.Linear(dim // 16, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        return x


class LinearDecoderSocialGraph(nn.Module):
    """Per-pair social decoder (LAH / SA) used when gaze_graph.use=False.

    Operates directly on concatenated person-token pair features (2*token_dim)
    and reads out a single logit.
    """

    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = token_dim

        scale = 4
        self.block1 = ResidualLinearBlock(2 * token_dim, scale=scale)
        self.fc = nn.Linear(token_dim * 2 // scale**2, 1)

    def forward(self, edge_tokens):
        x = self.block1(edge_tokens)
        x = self.fc(x)
        return x


# ****************************************************** #
#                      HEATMAP DECODERS                  #
# ****************************************************** #
class ConditionalDPTDecoder(nn.Module):
    """
    Adapted re-assemble stage of standard DPT for person-conditioned gaze heatmap prediction
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        hooks: List[int] = [2, 5, 8, 11],
        hidden_dims: List[int] = [96, 192, 384, 768],
        token_dim: int = 768,
        feature_dim: int = 128,
        use_bn: bool = True,
    ):
        super().__init__()

        self.patch_size = pair(patch_size)
        self.hooks = hooks
        self.token_dim = token_dim
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim
        self.use_bn = use_bn

        self.patch_h = self.patch_size[0]
        self.patch_w = self.patch_size[1]

        assert len(hooks) <= 4, "The argument hooks can't have more than 4 elements."
        self.factors = [4, 8, 16, 32][-len(hooks) :]
        self.reassemble_blocks = nn.ModuleDict(
            {
                f"r{factor}": Reassemble(
                    factor,
                    hidden_dims[idx],
                    feature_dim=feature_dim,
                    token_dim=token_dim,
                )
                for idx, factor in enumerate(self.factors)
            }
        )

        self.fusion_blocks = nn.ModuleDict(
            {
                f"f{factor}": FusionBlock(feature_dim, use_bn=use_bn)
                for idx, factor in enumerate(self.factors)
            }
        )

        self.gaze_projs = nn.ModuleDict(
            {
                f"g{factor}": nn.Linear(token_dim, feature_dim, bias=True)
                for idx, factor in enumerate(self.factors)
            }
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1
            ),
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReLU(True),
            nn.Conv2d(feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, img_layers, gaze_layers, img_size):
        img_h, img_w = img_size
        feat_h = img_h // self.patch_h
        feat_w = img_w // self.patch_w

        b, n, _ = gaze_layers[-1].shape

        # Reshape tokens into spatial representation
        img_layers = [
            einops.rearrange(l, "b (fh fw) d -> b d fh fw", fh=feat_h, fw=feat_w)
            for l in img_layers
        ]

        # Apply reassemble and fusion blocks
        for idx, (factor, img_layer, gaze_layer) in enumerate(
            zip(self.factors[::-1], img_layers[::-1], gaze_layers[::-1])
        ):
            f = self.reassemble_blocks[f"r{factor}"](img_layer)
            _, d, h, w = f.shape
            g = self.gaze_projs[f"g{factor}"](gaze_layer)  # (b, n, d) > # (b, n, d')
            # (b, n, d', H/32, W/32) > (b*n, d', H/32, W/32)
            f = torch.einsum("bdhw,bnd->bndhw", f, g).view(-1, self.feature_dim, h, w)
            if idx == 0:
                z = self.fusion_blocks[f"f{factor}"](f)  # (b*n, d', H/16, W/16)
            else:
                z = self.fusion_blocks[f"f{factor}"](f, z)  # (b*n, d', H/16, W/16)

        # Apply prediction head and downscale (224 > 64)
        z = self.head(z)  # (b*n, d', H/2, W/2) > (b*n, 1, H/2, W/2)
        # (b*n, 1, H, W) > (b*n, 1, 64, 64)
        z = F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False)
        z = z.view(b, n, 64, 64)  # (b*n, 1, 64, 64) > (b, n, 64, 64)
        return z


# ****************************************************** #
#                      VIT ENCODER                       #
# ****************************************************** #
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        use_qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            use_qkv_bias=use_qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim, hidden_features=int(dim * mlp_ratio), drop_rate=drop_rate
        )

    def forward(self, x, print_att=False):
        x = x + self.drop_path(self.attn(self.norm1(x), print_att))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        use_qkv_bias=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=use_qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x, print_att):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if print_att:
            logger.info(attn.shape)
            logger.info(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# ****************************************************** #
#                SPATIAL INPUT TOKENIZER                 #
# ****************************************************** #
class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = x.contiguous()
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

    def __repr__(self):
        return f"Interpolate(scale_factor={self.scale_factor}, mode={self.mode}, align_corners={self.align_corners})"


class Reassemble(nn.Module):
    def __init__(self, factor, hidden_dim, feature_dim=256, token_dim=768):
        super().__init__()

        assert factor in [4, 8, 16, 32], (
            "Argument `factor` not supported. Choose from [0.5, 4, 8, 16, 32]."
        )
        self.factor = factor
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.token_dim = token_dim

        if factor == 4:
            self.resample = nn.Sequential(
                nn.Conv2d(
                    token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True
                ),
                nn.ConvTranspose2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True,
                ),
            )
            self.proj = nn.Conv2d(
                hidden_dim,
                feature_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
        elif factor == 8:
            self.resample = nn.Sequential(
                nn.Conv2d(
                    token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True
                ),
                nn.ConvTranspose2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
            )
            self.proj = nn.Conv2d(
                hidden_dim,
                feature_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
        elif factor == 16:
            self.resample = nn.Sequential(
                nn.Conv2d(
                    token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True
                ),
            )
            self.proj = nn.Conv2d(
                hidden_dim,
                feature_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
        elif factor == 32:
            self.resample = nn.Sequential(
                nn.Conv2d(
                    token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True
                ),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
            )
            self.proj = nn.Conv2d(
                hidden_dim,
                feature_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )

    def forward(self, x):
        x = self.resample(x)
        x = self.proj(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, feature_dim, use_bn=False):
        """Init.
        Args:
            features (int): dimension of feature maps
            use_bn (bool): whether to use batch normalization in the Residual Conv Units.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.use_bn = use_bn

        modules = nn.ModuleList(
            [
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(not self.use_bn),
                ),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(not self.use_bn),
                ),
            ]
        )
        if self.use_bn:
            modules.insert(2, nn.BatchNorm2d(feature_dim))
            modules.insert(5, nn.BatchNorm2d(feature_dim))
        self.residual_module = nn.Sequential(*modules)

    def forward(self, x):
        z = self.residual_module(x)
        return z + x


class FusionBlock(nn.Module):
    def __init__(self, feature_dim, use_bn=False):
        super().__init__()

        self.feature_dim = feature_dim
        self.use_bn = use_bn

        self.rcu1 = ResidualConvUnit(feature_dim, use_bn=use_bn)
        self.rcu2 = ResidualConvUnit(feature_dim, use_bn=use_bn)
        self.resample = Interpolate(2, "bilinear", align_corners=True)
        self.proj = nn.Conv2d(
            feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, *xs):
        assert 1 <= len(xs) <= 2, (
            f"Can only accept inputs of length <= 2. Received len(xs)={len(xs)}"
        )

        z = self.rcu1(xs[0])
        if len(xs) == 2:
            z = z + xs[1]
        z = self.rcu2(z)
        z = self.resample(z)
        z = self.proj(z)

        return z
