# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
from collections import OrderedDict

import math
import pickle
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm
import lightning.pytorch as pl
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    MultiStepLR,
)

from mtgs.train.losses import (
    compute_sharingan_loss,
    compute_interact_loss,
    compute_social_loss,
    compute_dual_null_loss,
    compute_inout_loss,
    social_loss,
)

from mtgs.performance.metrics import (
    GFTestDistance,
    GFTestAUC,
    Distance,
    AUC,
)
from mtgs.networks import MTGS
from mtgs.utils import spatial_argmax2d

import logging

logger = logging.getLogger(__name__)


class MTGSModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Initialize model
        self.model = MTGS(
            patch_size=cfg.model.patch_size,
            token_dim=cfg.model.token_dim,
            image_size=cfg.model.image_size,
            gaze_feature_dim=cfg.model.gaze_feature_dim,
            encoder_depth=cfg.model.encoder_depth,
            encoder_num_heads=cfg.model.encoder_num_heads,
            encoder_num_global_tokens=cfg.model.encoder_num_global_tokens,
            encoder_mlp_ratio=cfg.model.encoder_mlp_ratio,
            encoder_use_qkv_bias=cfg.model.encoder_use_qkv_bias,
            encoder_drop_rate=cfg.model.encoder_drop_rate,
            encoder_attn_drop_rate=cfg.model.encoder_attn_drop_rate,
            encoder_drop_path_rate=cfg.model.encoder_drop_path_rate,
            decoder_feature_dim=cfg.model.decoder_feature_dim,
            decoder_hooks=cfg.model.decoder_hooks,
            decoder_hidden_dims=cfg.model.decoder_hidden_dims,
            decoder_use_bn=cfg.model.decoder_use_bn,
            temporal_context=cfg.data.temporal_context,
            output=cfg.model.output,
            gaze_graph_num_layers=cfg.gaze_graph.num_layers,
            gaze_graph_edge_dim=cfg.gaze_graph.edge_dim,
            gaze_graph_use_prior=cfg.gaze_graph.use_prior,
            gaze_graph_prior_weight=cfg.gaze_graph.prior_weight,
            gaze_graph_laeo_derive=cfg.gaze_graph.laeo_derive,
            gaze_graph_use=cfg.gaze_graph.use,
        )

        self.cfg = cfg
        self.output = cfg.model.output
        self.num_tranining_samples = cfg.data.num_samples
        self.num_steps_in_epoch = math.ceil(
            self.num_tranining_samples / cfg.train.batch_size
        )
        self._pred_file = None
        self._pred_write_count = 0
        self._upper_tri_cache: dict = {}  # cache upper-triangle SA pair mask per n

        # Model weights paths
        self.model_weights = cfg.model.weights
        self.gaze_weights = cfg.model.gaze_weights

        # Define Metrics
        if cfg.experiment.dataset == "gazefollow":
            self.metrics = nn.ModuleDict(
                {
                    "val_dist": Distance(),
                    "test_dist": GFTestDistance(),
                    "test_auc": GFTestAUC(),
                }
            )
        else:
            self.metrics = nn.ModuleDict(
                {"val_dist": Distance(), "test_dist": Distance(), "test_auc": AUC()}
            )

        # Define Social Gaze Metrics
        self.val_coatt_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.val_coatt_ap = tm.AveragePrecision(task="binary", ignore_index=-1)

        self.val_laeo_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.val_laeo_ap = tm.AveragePrecision(task="binary", ignore_index=-1)

        self.val_lah_auc = tm.AUROC(task="binary", ignore_index=-1)
        self.val_lah_ap = tm.AveragePrecision(task="binary", ignore_index=-1)

        # Define Loss Function
        self.compute_hm_loss = compute_interact_loss
        self.compute_dist_loss = compute_sharingan_loss
        self.compute_social_loss = compute_social_loss
        self.compute_speaking_loss = compute_inout_loss

        # Initialize Weights
        self._init_weights()

        # Freeze Weights
        self._freeze()

    @staticmethod
    def _remap_gaze_graph_weights(weights):
        """Back-compat warm-start for the two-path GazeGraphBlock.

        A pre-split gaze_graph checkpoint has a single shared refinement
        (`row_layer/col_layer/upd_*/norm_*/refresh/norm_e`) + one `edge_head`.
        The new layout has two isolated refiners (`trunk.*`, `sa.*`) + per-type
        heads (`head_lah/head_sa/head_null`). Copy the old shared weights onto
        BOTH paths (and the old head onto all three heads) so LAH refinement is
        preserved instead of cold-starting. No-op for new-layout checkpoints.
        """
        prefix = "gaze_graph_block."
        has_old = any(k.startswith(prefix + "edge_head.") for k in weights)
        has_new = any(k.startswith(prefix + "trunk.") for k in weights)
        if not has_old or has_new:
            return weights

        refine_map = {
            "row_layer": "row", "col_layer": "col",
            "upd_src": "upd_src", "upd_tgt": "upd_tgt",
            "norm_src": "norm_src", "norm_tgt": "norm_tgt",
            "refresh": "refresh", "norm_e": "norm_e",
        }
        out = OrderedDict()
        for k, v in weights.items():
            if not k.startswith(prefix):
                out[k] = v
                continue
            sub = k[len(prefix):]                       # e.g. "row_layer.self_attn..."
            top = sub.split(".", 1)[0]
            rest = sub[len(top):]                       # ".<param...>"
            if top in refine_map:
                new = refine_map[top] + rest
                out[prefix + "trunk." + new] = v
                out[prefix + "sa." + new] = v.clone()
            elif top == "edge_head":
                out[prefix + "head_lah" + rest] = v
                out[prefix + "head_null" + rest] = v.clone()
                out[prefix + "head_sa" + rest] = v.clone()
            else:                                       # mlp_init, *xattn*, node projs, region/hm, etc.
                out[k] = v
        logger.info("Remapped pre-split GazeGraphBlock weights onto trunk/sa two-path layout")
        return out

    def _init_weights(self):
        # Load pre-trained weights
        if self.model_weights:
            model_ckpt = torch.load(self.model_weights, map_location="cpu", weights_only=False)
            model_weights = OrderedDict(
                [
                    (name.replace("model.", ""), value)
                    for name, value in model_ckpt["state_dict"].items()
                ]
            )
            model_weights = self._remap_gaze_graph_weights(model_weights)
            model_state = self.model.state_dict()
            skipped = []
            filtered_weights = OrderedDict()
            for name, value in model_weights.items():
                if name in model_state and model_state[name].shape != value.shape:
                    skipped.append((name, tuple(value.shape), tuple(model_state[name].shape)))
                    continue
                filtered_weights[name] = value
            if skipped:
                logger.info(
                    "Skipped %d checkpoint tensors with incompatible shapes: %s",
                    len(skipped),
                    skipped[:8],
                )
            self.model.load_state_dict(filtered_weights, strict=False)
            logger.info(
                f"Successfully loaded pre-trained weights from {self.model_weights}"
            )
            del model_ckpt
        else:
            # Load Gaze Encoder Gaze360 Pre-trained Weights
            gaze360_ckpt = torch.load(self.gaze_weights, map_location="cpu", weights_only=False)
            gaze360_weights = OrderedDict(
                [
                    (name.replace("base_head.", ""), value)
                    for name, value in gaze360_ckpt["model_state_dict"].items()
                    if "base_head" in name
                ]
            )
            self.model.gaze_encoder.backbone.load_state_dict(
                gaze360_weights, strict=True
            )
            logger.info(
                f"Successfully loaded weights for the gaze backbone from {self.gaze_weights}"
            )

            # Delete checkpoints
            del gaze360_ckpt, gaze360_weights

    def _set_batchnorm_eval(self, model):
        for module in model.modules():
            module.eval()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _social_head_modules(self):
        # Active social head: GazeGraphBlock (use=True) or the original per-pair
        # decoders (use=False). Used by the frozen-trunk post-training recipe.
        if self.cfg.gaze_graph.use:
            return [self.model.gaze_graph_block]
        return [self.model.decoder_lah, self.model.decoder_sa]

    def _freeze(self):
        # Post-training recipe: frozen transformer trunk as a visual extractor,
        # train ONLY the social head. Overrides the per-module flags below.
        _freeze_gaze_graph = getattr(self.cfg.train.freeze, "all_but_gaze_graph", False) or (
            getattr(self.cfg.gaze_graph, "frozen", False)
        )
        if _freeze_gaze_graph:
            self.freeze_module(self.model)
            for m in self._social_head_modules():
                for p in m.parameters():
                    p.requires_grad = True
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            head_name = "gaze_graph_block" if self.cfg.gaze_graph.use else "decoder_lah/decoder_sa"
            logger.info(
                f"Frozen trunk (extractor); training ONLY {head_name} "
                f"({n_train:,} / {n_total:,} params trainable)."
            )
            return

        if self.cfg.train.freeze.gaze_encoder_backbone:
            logger.info("Freezing the Gaze Encoder backbone layers.")
            self.freeze_module(self.model.gaze_encoder.backbone)
        if self.cfg.train.freeze.gaze_encoder:
            logger.info("Freezing the Gaze Encoder layers.")
            self.freeze_module(self.model.gaze_encoder)
        if self.cfg.train.freeze.vit_encoder:
            logger.info("Freezing the ViT Encoder layers.")
            self.freeze_module(self.model.encoder)
        if self.cfg.train.freeze.vit_adaptor:
            logger.info("Freezing the ViT Adaptor layers.")
            self.freeze_module(self.model.vit_adaptor)
        if self.cfg.train.freeze.inout_decoder:
            logger.info("Freezing the InOut Decoder layers.")
            self.freeze_module(self.model.inout_decoder)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        # Fixed interaction trunk (people_interaction + people_temporal); the social
        # head is gaze_graph_block (highest LR). 4 param groups.
        base_lr = self.cfg.optimizer.lr
        # Social head param group: GazeGraphBlock (use=True) or the original
        # per-pair decoders (use=False). ×3 keeps the same 4-group structure/LR.
        if self.cfg.gaze_graph.use:
            social_head_group = {
                "params": self.model.gaze_graph_block.parameters(),
                "name": "gaze-graph-block",
                "lr": base_lr * 3,
                "init_lr": base_lr * 3,
            }
            social_head_prefixes = {"gaze_graph_block"}
        else:
            social_head_group = {
                "params": list(self.model.decoder_lah.parameters())
                + list(self.model.decoder_sa.parameters()),
                "name": "social-decoder",
                "lr": base_lr * 3,
                "init_lr": base_lr * 3,
            }
            social_head_prefixes = {"decoder_lah", "decoder_sa"}
        high_lr_params = [
            {
                "params": self.model.gaze_encoder_temporal.parameters(),
                "name": "gaze-encoder-temporal",
                "lr": base_lr * 3,
                "init_lr": base_lr * 3,
            },
            {
                "params": self.model.people_temporal.parameters(),
                "name": "people-temporal",
                "lr": base_lr * 3,
                "init_lr": base_lr * 3,
            },
            social_head_group,
        ]
        high_lr_prefixes = {
            "gaze_encoder_temporal",
            "people_temporal",
        } | social_head_prefixes
        other_params = [
            v for k, v in self.model.named_parameters()
            if not any(k.startswith(prefix) for prefix in high_lr_prefixes)
        ]
        other_params = [
            {
                "params": other_params,
                "name": "base",
                "lr": base_lr,
                "init_lr": base_lr,
            }
        ]
        params = high_lr_params + other_params

        optimizer = optim.AdamW(params, weight_decay=self.cfg.optimizer.weight_decay)

        # Per-step linear warmup → cosine annealing. Step counts are derived from the
        # REAL dataloader length (trainer.estimated_stepping_batches), so the schedule
        # is smooth per-step yet independent of the (possibly stale) data.num_samples.
        if self.cfg.scheduler.type == "CosineAnnealingLR":
            warmup_epochs = self.cfg.scheduler.warmup_epochs
            total_epochs = self.cfg.train.epochs
            eta_min = self.cfg.scheduler.eta_min
            total_steps = int(self.trainer.estimated_stepping_batches)
            warmup_steps = max(1, round(total_steps * warmup_epochs / total_epochs))
            cosine_steps = max(1, total_steps - warmup_steps)
            if warmup_epochs > 0:
                warmup = LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine = CosineAnnealingLR(
                    optimizer, T_max=cosine_steps, eta_min=eta_min
                )
                lr_scheduler = SequentialLR(
                    optimizer, schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
            else:
                lr_scheduler = CosineAnnealingLR(
                    optimizer, T_max=total_steps, eta_min=eta_min
                )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.cfg.scheduler.type == "StepLR":
            lr_scheduler = MultiStepLR(
                optimizer, milestones=[10, 11, 12, 13, 14, 15, 16], gamma=0.5
            )
            #             lr_scheduler = StepLR(optimizer, step_size=self.cfg.scheduler.t_0_epochs, gamma=0.1)
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            logger.info("Invalid scheduler selected...")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_train_epoch_start(self):
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

        # Post-training: keep the ENTIRE frozen trunk in eval mode so its BatchNorm
        # running stats stay fixed — otherwise heatmap/in-out drift from the original
        # checkpoint even with weights frozen. Only gaze_graph_block stays in train.
        _freeze_gaze_graph = getattr(self.cfg.train.freeze, "all_but_gaze_graph", False) or (
            getattr(self.cfg.gaze_graph, "frozen", False)
        )
        if _freeze_gaze_graph:
            self._set_batchnorm_eval(self.model)   # whole trunk → eval (BN stats fixed)
            for m in self._social_head_modules():  # trained head stays in train mode
                m.train()
            return

        # Set BN layers to eval mode for frozen modules
        if self.cfg.train.freeze.gaze_encoder:
            self.model.gaze_encoder.apply(self._set_batchnorm_eval)
        if self.cfg.train.freeze.vit_encoder:
            self.model.encoder.apply(self._set_batchnorm_eval)
        if self.cfg.train.freeze.inout_decoder:
            self.model.inout_decoder.apply(self._set_batchnorm_eval)

    def training_step(self, batch, batch_idx):
        nv = int((batch["speaking"] != -1).sum().item())
        ni = int((batch["inout"] == 1).sum().item())

        # Forward pass
        if self.output == "heatmap":
            (
                _,
                gaze_vec_pred,
                gaze_hm_pred,
                inout_pred,
                lah_pred,
                laeo_pred,
                coatt_pred,
                alpha_null_in,
                alpha_null_out,
            ) = self(batch)
            batch_size, t, n, hm_h, hm_w = gaze_hm_pred.shape
            gaze_hm_pred = gaze_hm_pred.view(batch_size * t, n, hm_h, hm_w)
        else:
            gaze_vec_pred, gaze_pt_pred, inout_pred, lah_pred, laeo_pred, coatt_pred = (
                self(batch)
            )
            alpha_null_in = alpha_null_out = None
            batch_size, t, n = gaze_pt_pred.shape[:-1]
            gaze_pt_pred = gaze_pt_pred.view(batch_size * t, n, -1)
        gaze_vec_pred = gaze_vec_pred.view(batch_size * t, n, -1)
        inout_pred = inout_pred.view(batch_size * t, -1)
        lah_pred = lah_pred.view(batch_size * t, -1)
        laeo_pred = laeo_pred.view(batch_size * t, -1)
        coatt_pred = coatt_pred.view(batch_size * t, -1)

        # Compute distance, inout loss
        if self.output == "heatmap":
            loss_dist, logs_dist = self.compute_hm_loss(
                batch["gaze_vecs"].view(batch_size * t, n, -1),
                batch["gaze_heatmaps"].view(batch_size * t, n, hm_h, hm_w),
                batch["inout"].view(batch_size * t, -1),
                gaze_vec_pred,
                gaze_hm_pred,
                inout_pred,
            )  # 2d gaze angle loss
        else:
            loss_dist, logs_dist = self.compute_dist_loss(
                batch["gaze_vecs"],
                batch["gaze_pts"],
                batch["inout"].view(batch_size * t, -1),
                gaze_vec_pred,
                gaze_pt_pred,
                inout_pred,
            )

        loss = loss_dist
        # Compute social gaze loss
        coatt_gt = batch["coatt_labels"].view(batch_size * t, -1)
        coatt_mask = coatt_gt != -1
        laeo_gt = batch["laeo_labels"].view(batch_size * t, -1)
        laeo_mask = laeo_gt != -1
        lah_gt = batch["lah_labels"].view(batch_size * t, -1)
        lah_mask = lah_gt != -1
        # gaze_graph mode: SA & LAEO are symmetric (mat[i,j]=mat[j,i]), so both (i,j)
        # and (j,i) hit the same edge. Restrict to upper-triangle pairs (src < dst)
        # so each undirected pair is counted once. LAH is directed → keep both.
        # (use=False decoder predicts each direction independently → supervise all.)
        if self.cfg.gaze_graph.use:
            if n not in self._upper_tri_cache:
                pairs = list(itertools.permutations(range(n), 2))
                self._upper_tri_cache[n] = torch.tensor(
                    [s < d for s, d in pairs], dtype=torch.bool
                )
            utri = self._upper_tri_cache[n].to(coatt_mask.device)
            coatt_mask = coatt_mask & utri
            laeo_mask = laeo_mask & utri

        loss_social, logs_social = self.compute_social_loss(
            lah_pred,
            lah_gt,
            lah_mask,
            laeo_pred,
            laeo_gt,
            laeo_mask,
            coatt_pred,
            coatt_gt,
            coatt_mask,
            coatt_is_prob=False,
            lah_pos_weight=self.cfg.loss.lah_pos_weight,
            laeo_pos_weight=self.cfg.loss.laeo_pos_weight,
            coatt_pos_weight=self.cfg.loss.sa_pos_weight,
        )
        loss += loss_social

        # Dual-null routing loss (gaze_graph null_in/null_out edge supervision)
        if alpha_null_in is not None and alpha_null_out is not None:
            lam_null = getattr(self.cfg.gaze_graph, "lambda_null", 0.5)
            inout_gt_bt  = batch["inout"].view(batch_size * t, n)
            num_valid_bt = batch["num_valid_people"].view(batch_size * t)
            loss_null_out, loss_null_in = compute_dual_null_loss(
                alpha_null_out.view(batch_size * t, n),
                alpha_null_in.view(batch_size * t, n),
                inout_gt_bt,
                lah_gt,
                num_valid_bt,
            )
            loss_null = lam_null * (loss_null_out + loss_null_in)
            loss = loss + loss_null
            self.log("loss/train/null_out", loss_null_out.item(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
            self.log("loss/train/null_in",  loss_null_in.item(),  batch_size=n, prog_bar=False, on_step=True, on_epoch=True)

        # Log Social Gaze Losses
        self.log(
            "loss/train/lah",
            logs_social["lah_loss"],
            batch_size=lah_mask.sum(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "loss/train/laeo",
            logs_social["laeo_loss"],
            batch_size=laeo_mask.sum(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "loss/train/coatt",
            logs_social["coatt_loss"],
            batch_size=coatt_mask.sum(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        # Logging Distance, InOut losses
        self.log(
            "loss/train/heatmap",
            logs_dist["heatmap_loss"],
            batch_size=ni,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "loss/train/dist",
            logs_dist["dist_loss"],
            batch_size=ni,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "loss/train/angular",
            logs_dist["angular_loss"],
            batch_size=ni,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "loss/train/inout",
            logs_dist["inout_loss"],
            batch_size=n,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "loss/train",
            loss.item(),
            batch_size=n,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        nv = int((batch["speaking"] != -1).sum().item())
        ni = int((batch["inout"] == 1).sum().item())

        # Forward pass
        if self.output == "heatmap":
            (
                _,
                gaze_vec_pred,
                gaze_hm_pred,
                inout_pred,
                lah_pred,
                laeo_pred,
                coatt_pred,
                *_,
            ) = self(batch)
            # only take outputs of central frame
            batch_size, t, n, hm_h, hm_w = gaze_hm_pred.shape
            middle_frame_idx = int(t / 2)
            gaze_hm_pred = gaze_hm_pred[:, middle_frame_idx, :, :, :]
            # perform argmax for gaze point
            gaze_pt_pred = spatial_argmax2d(
                gaze_hm_pred.reshape(batch_size * n, hm_h, hm_w), normalize=True
            ).view(batch_size, n, -1)
        else:
            gaze_vec_pred, gaze_pt_pred, inout_pred, lah_pred, laeo_pred, coatt_pred = (
                self(batch)
            )
            batch_size, t, n = gaze_pt_pred.shape[:-1]
            middle_frame_idx = int(t / 2)
            gaze_pt_pred = gaze_pt_pred[:, middle_frame_idx, :, :]
        gaze_vec_pred = gaze_vec_pred[:, middle_frame_idx, :, :]
        inout_pred = inout_pred[:, middle_frame_idx, :]
        lah_pred = lah_pred[:, middle_frame_idx, :]
        laeo_pred = laeo_pred[:, middle_frame_idx, :]
        coatt_pred = coatt_pred[:, middle_frame_idx, :]

        # Compute dist, inout loss
        if self.output == "heatmap":
            loss_dist, logs_dist = self.compute_hm_loss(
                batch["gaze_vecs"][:, middle_frame_idx, :, :],
                batch["gaze_heatmaps"][:, middle_frame_idx, :, :, :],
                batch["inout"][:, middle_frame_idx, :],
                gaze_vec_pred,
                gaze_hm_pred,
                inout_pred,
            )  # 2d gaze vector loss
        else:
            loss_dist, logs_dist = self.compute_dist_loss(
                batch["gaze_vecs"],
                batch["gaze_pts"],
                batch["inout"][:, middle_frame_idx, :],
                gaze_vec_pred,
                gaze_pt_pred,
                inout_pred,
            )

        loss = loss_dist
        # Logging losses
        self.log(
            "loss/val/heatmap",
            logs_dist["heatmap_loss"],
            batch_size=ni,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/val/dist",
            logs_dist["dist_loss"],
            batch_size=ni,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/val/angular",
            logs_dist["angular_loss"],
            batch_size=ni,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/val/inout",
            logs_dist["inout_loss"],
            batch_size=n,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/val",
            loss.item(),
            batch_size=n,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Update dist metrics
        # self.metrics["val_auc"].update(gaze_heatmap_pred, gaze_heatmap, inout)
        self.metrics["val_dist"].update(
            gaze_pt_pred,
            batch["gaze_pts"][:, middle_frame_idx, :, :],
            batch["inout"][:, middle_frame_idx, :],
        )
        # self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "metric/val/dist",
            self.metrics["val_dist"],
            batch_size=ni,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Compute social gaze loss
        coatt_gt = batch["coatt_labels"][:, middle_frame_idx, :]
        coatt_mask = coatt_gt != -1
        laeo_gt = batch["laeo_labels"][:, middle_frame_idx, :]
        laeo_mask = laeo_gt != -1
        lah_gt = batch["lah_labels"][:, middle_frame_idx, :]
        lah_mask = lah_gt != -1
        # gaze_graph mode: SA & LAEO are symmetric → restrict to upper-triangle pairs
        # (src < dst) so each undirected pair is counted once in loss AND metrics.
        # LAH is directed → keep both. (use=False decoder → supervise all.)
        if self.cfg.gaze_graph.use:
            if n not in self._upper_tri_cache:
                pairs = list(itertools.permutations(range(n), 2))
                self._upper_tri_cache[n] = torch.tensor(
                    [s < d for s, d in pairs], dtype=torch.bool
                )
            utri = self._upper_tri_cache[n].to(coatt_mask.device)
            coatt_mask = coatt_mask & utri
            laeo_mask = laeo_mask & utri

        loss_social, logs_social = self.compute_social_loss(
            lah_pred,
            lah_gt,
            lah_mask,
            laeo_pred,
            laeo_gt,
            laeo_mask,
            coatt_pred,
            coatt_gt,
            coatt_mask,
            coatt_is_prob=False,
            lah_pos_weight=self.cfg.loss.lah_pos_weight,
            laeo_pos_weight=self.cfg.loss.laeo_pos_weight,
            coatt_pos_weight=self.cfg.loss.sa_pos_weight,
        )
        loss += loss_social

        # Log Social Gaze Losses
        self.log(
            "loss/val/lah",
            logs_social["lah_loss"],
            batch_size=lah_mask.sum(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/val/laeo",
            logs_social["laeo_loss"],
            batch_size=laeo_mask.sum(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/val/coatt",
            logs_social["coatt_loss"],
            batch_size=coatt_mask.sum(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/val/social",
            loss_social,
            batch_size=n,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Update CoAtt Metrics
        if coatt_pred.sum() != 0:
            coatt_pred = torch.sigmoid(coatt_pred)
            # -1 outside the (upper-tri ∩ valid) mask → ignore_index skips them,
            # so each symmetric SA pair is scored once (consistent with the loss).
            coatt_gt = coatt_gt.long().masked_fill(~coatt_mask, -1)
            if coatt_mask.sum() > 0:
                self.val_coatt_auc(coatt_pred, coatt_gt)
                self.val_coatt_ap(coatt_pred, coatt_gt)

                self.log(
                    "metric/val/coatt_auc",
                    self.val_coatt_auc,
                    batch_size=coatt_mask.sum(),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "metric/val/coatt_ap",
                    self.val_coatt_ap,
                    batch_size=coatt_mask.sum(),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Update LAEO metrics
        if laeo_pred.sum() != 0:
            laeo_pred = torch.sigmoid(laeo_pred)
            # -1 outside the (upper-tri ∩ valid) mask → score each symmetric LAEO pair once.
            laeo_gt = laeo_gt.long().masked_fill(~laeo_mask, -1)
            if laeo_mask.sum() > 0:
                self.val_laeo_auc(laeo_pred, laeo_gt)
                self.val_laeo_ap(laeo_pred, laeo_gt)

                self.log(
                    "metric/val/laeo_auc",
                    self.val_laeo_auc,
                    batch_size=laeo_mask.sum(),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "metric/val/laeo_ap",
                    self.val_laeo_ap,
                    batch_size=laeo_mask.sum(),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Update LAH metrics
        if lah_pred.sum() != 0:
            lah_pred = torch.sigmoid(lah_pred)
            lah_gt = lah_gt.long()
            if lah_mask.sum() > 0:
                self.val_lah_auc(lah_pred, lah_gt)
                self.val_lah_ap(lah_pred, lah_gt)

                self.log(
                    "metric/val/lah_auc",
                    self.val_lah_auc,
                    batch_size=lah_mask.sum(),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "metric/val/lah_ap",
                    self.val_lah_ap,
                    batch_size=lah_mask.sum(),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def on_validation_epoch_end(self):
        aps = []
        for m in (self.val_lah_ap, self.val_laeo_ap, self.val_coatt_ap):
            try:
                aps.append(m.compute())
            except Exception:
                pass
        if aps:
            self.log(
                "metric/val/social_ap",
                torch.stack(aps).mean(),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )

        aucs = []
        for m in (self.val_lah_auc, self.val_laeo_auc, self.val_coatt_auc):
            try:
                aucs.append(m.compute())
            except Exception:
                pass
        if aucs:
            self.log(
                "metric/val/social_auc",
                torch.stack(aucs).mean(),
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )

    def on_test_start(self):
        output_file = os.path.join(
            self.cfg.experiment.output_folder, "test_predictions.p"
        )
        os.makedirs(self.cfg.experiment.output_folder, exist_ok=True)
        self._pred_file = open(output_file, "wb")
        self._pred_file_path = output_file
        self._pred_write_count = 0

    def test_step(self, batch, batch_idx):
        ni = int((batch["inout"] == 1).sum().item())
        #         assert n == ni, f"Expected all test samples to be looking inside. Got {n} samples, {ni} of which are looking inside."

        # Forward pass
        if self.output == "heatmap":
            (
                _,
                gaze_vec_pred,
                gaze_hm_pred,
                inout_pred,
                lah_pred,
                laeo_pred,
                coatt_pred,
                *_,
            ) = self(batch)
            batch_size, t, num_people, hm_h, hm_w = gaze_hm_pred.shape
            # only take outputs of central frame
            middle_frame_idx = int(t / 2)
            gaze_hm_pred = gaze_hm_pred[:, middle_frame_idx, :, :, :]
            # perform arg max for gaze point
            gaze_pt_pred = spatial_argmax2d(
                gaze_hm_pred.reshape(batch_size * num_people, hm_h, hm_w),
                normalize=True,
            ).view(batch_size, num_people, -1)
        else:
            gaze_vec_pred, gaze_pt_pred, inout_pred, lah_pred, laeo_pred, coatt_pred = (
                self(batch)
            )
            batch_size, t, num_people = gaze_pt_pred.shape[:-1]
            middle_frame_idx = int(t / 2)
            gaze_pt_pred = gaze_pt_pred[:, middle_frame_idx, :, :]
        gaze_vec_pred = gaze_vec_pred[:, middle_frame_idx, :, :]
        inout_pred = inout_pred[:, middle_frame_idx, :]
        lah_pred = lah_pred[:, middle_frame_idx, :]
        laeo_pred = laeo_pred[:, middle_frame_idx, :]
        coatt_pred = coatt_pred[:, middle_frame_idx, :]

        # Update distance metrics
        if self.cfg.experiment.dataset == "gazefollow":
            gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
            gaze_pt_pred = gaze_pt_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
            inout_pred = inout_pred[:, -1]  # (b, n) >> (b,)
            inout_gt = batch["inout"][:, middle_frame_idx]
            gaze_hm_pred = gaze_hm_pred[:, -1, :, :]

            test_auc = self.metrics["test_auc"](
                gaze_hm_pred, batch["gaze_pts"][:, middle_frame_idx, :, :]
            )
            test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](
                gaze_pt_pred, batch["gaze_pts"][:, middle_frame_idx, :, :]
            )
            # Log metrics
            self.log(
                "metric/test/auc",
                test_auc,
                batch_size=ni,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "metric/test/dist_to_avg",
                test_dist_to_avg,
                batch_size=ni,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "metric/test/avg_dist",
                test_avg_dist,
                batch_size=ni,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "metric/test/min_dist",
                test_min_dist,
                batch_size=ni,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        else:
            inout_gt = batch["inout"][:, middle_frame_idx, :]
            # Log metrics
            if self.output == "heatmap":
                test_auc = self.metrics["test_auc"](
                    gaze_hm_pred.reshape(batch_size * num_people, hm_h, hm_w),
                    batch["gaze_heatmaps"][:, middle_frame_idx, :, :, :].reshape(
                        batch_size * num_people, hm_h, hm_w
                    ),
                    inout_gt.reshape(batch_size * num_people, -1),
                )
                self.log(
                    "metric/test/auc",
                    test_auc,
                    batch_size=ni,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            self.metrics["test_dist"].update(
                gaze_pt_pred, batch["gaze_pts"][:, middle_frame_idx, :, :], inout_gt
            )
            self.log(
                "metric/test/dist",
                self.metrics["test_dist"],
                batch_size=ni,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        coatt_gt = batch["coatt_labels"][:, middle_frame_idx, :]
        coatt_mask = coatt_gt != -1
        laeo_gt = batch["laeo_labels"][:, middle_frame_idx, :]
        laeo_mask = laeo_gt != -1
        lah_gt = batch["lah_labels"][:, middle_frame_idx, :]
        lah_mask = lah_gt != -1

        _coatt_pred_m = _coatt_gt_m = None
        _laeo_pred_m = _laeo_gt_m = None
        _lah_pred_m = _lah_gt_m = None

        # CoAtt — collect for epoch-end metric computation
        if coatt_pred.sum() != 0:
            coatt_pred = torch.sigmoid(coatt_pred)
            coatt_gt = coatt_gt.long()
            if coatt_mask.sum() > 0:
                _coatt_pred_m = coatt_pred.cpu()
                _coatt_gt_m = coatt_gt.cpu()

        pair_indices = torch.tensor(
            list(itertools.permutations(torch.arange(num_people), 2))
        )
        # LAEO — collect for epoch-end metric computation
        if laeo_pred.sum() != 0:
            laeo_pred = torch.sigmoid(laeo_pred)
            laeo_gt = laeo_gt.long()
            laeo_pred_argmax = torch.zeros_like(laeo_pred)
            for bi in range(batch_size):
                for pi in range(num_people):
                    valid_indices = torch.where(
                        (pair_indices[:, 1] == pi).int()
                        * (pair_indices[:, 0] != 0).int()
                    )[0]
                    if valid_indices.shape[0] > 0:
                        max_val, max_idx = torch.max(laeo_pred[bi][valid_indices], 0)
                        laeo_pred_argmax[bi][valid_indices[max_idx]] = max_val
            if laeo_mask.sum() > 0:
                _laeo_pred_m = laeo_pred_argmax.cpu()
                _laeo_gt_m = laeo_gt.cpu()

        # LAH — collect for epoch-end metric computation
        if lah_pred.sum() != 0:
            lah_pred = torch.sigmoid(lah_pred)
            lah_gt = lah_gt.long()
            lah_pred_argmax = torch.zeros_like(lah_pred)
            lah_gt_metric = torch.zeros(batch_size, num_people).long() - 1
            lah_pred_metric = torch.zeros(batch_size, num_people)
            for bi in range(batch_size):
                for pi in range(num_people):
                    if self.cfg.experiment.dataset == "gazefollow":
                        io = 1
                    else:
                        io = batch["inout"][bi][middle_frame_idx][pi] == 1
                    if io == 1:
                        valid_indices = torch.where((pair_indices[:, 1] == pi).int())[0]
                        if valid_indices.shape[0] > 0:
                            if (lah_gt[bi][valid_indices] != -1).sum() == 0:
                                continue

                            max_val, max_idx = torch.max(lah_pred[bi][valid_indices], 0)
                            lah_pred_argmax[bi][valid_indices[max_idx]] = max_val

                            lah_gt_metric[bi][pi] = min(
                                lah_gt[bi][valid_indices][
                                    lah_gt[bi][valid_indices] != -1
                                ].sum(),
                                1,
                            )
                            gt_idx = torch.where(lah_gt[bi][valid_indices] == 1)[0]
                            if len(gt_idx) > 0:
                                if len(gt_idx) > 1:
                                    gt_idx = gt_idx[0]
                                lah_pred_metric[bi][pi] = lah_pred_argmax[bi][
                                    valid_indices
                                ][gt_idx]
                            else:
                                lah_pred_metric[bi][pi] = max_val
            if (lah_gt_metric != -1).sum() > 0:
                _lah_pred_m = lah_pred_metric.cpu()
                _lah_gt_m = lah_gt_metric.cpu()

        # Build output dict — move tensors to CPU to prevent GPU memory accumulation
        output = {
            "head_bboxes": batch["head_bboxes"][:, middle_frame_idx, :, :].cpu(),
            "gp_pred": gaze_pt_pred.cpu(),
            "gp_gt": batch["gaze_pts"][:, middle_frame_idx, :, :].cpu(),
            "gv_pred": gaze_vec_pred.cpu(),
            "gv_gt": batch["gaze_vecs"][:, middle_frame_idx, :, :].cpu(),
            #   "hm_pred": gaze_hm_pred.cpu(),
            #   "hm_gt": batch["gaze_heatmaps"][:,middle_frame_idx,:,:,:].cpu(),
            "inout_gt": inout_gt.cpu(),
            "path": batch["path"],
            "inout_pred": inout_pred.cpu(),
            "coatt_pred": coatt_pred.cpu(),
            "laeo_pred": laeo_pred.cpu(),
            "lah_pred": lah_pred.cpu(),
            "coatt_gt": coatt_gt.cpu(),
            "laeo_gt": laeo_gt.cpu(),
            "lah_gt": lah_gt.cpu(),
            "dataset": batch["dataset"],
            "num_valid_people": batch["num_valid_people"].cpu(),
            "coatt_pred_metric": _coatt_pred_m,
            "coatt_gt_metric": _coatt_gt_m,
            "laeo_pred_metric": _laeo_pred_m,
            "laeo_gt_metric": _laeo_gt_m,
            "lah_pred_metric": _lah_pred_m,
            "lah_gt_metric": _lah_gt_m,
        }
        if self._pred_file is not None:
            pickle.dump(output, self._pred_file)
            self._pred_write_count += 1
            if self._pred_write_count % 500 == 0:
                self._pred_file.flush()

    def on_test_epoch_end(self):
        self.metrics["test_dist"].reset()

        if self._pred_file is not None:
            self._pred_file.close()
            self._pred_file = None

        # Read per-batch metric tensors from pickle and compute social metrics once
        coatt_preds, coatt_gts = [], []
        laeo_preds, laeo_gts = [], []
        lah_preds, lah_gts = [], []

        with open(self._pred_file_path, "rb") as f:
            while True:
                try:
                    b = pickle.load(f)
                except EOFError:
                    break
                if b.get("coatt_pred_metric") is not None:
                    coatt_preds.append(b["coatt_pred_metric"].reshape(-1))
                    coatt_gts.append(b["coatt_gt_metric"].reshape(-1))
                if b.get("laeo_pred_metric") is not None:
                    laeo_preds.append(b["laeo_pred_metric"].reshape(-1))
                    laeo_gts.append(b["laeo_gt_metric"].reshape(-1))
                if b.get("lah_pred_metric") is not None:
                    lah_preds.append(b["lah_pred_metric"].reshape(-1))
                    lah_gts.append(b["lah_gt_metric"].reshape(-1))

        def _log_social(preds_list, gts_list, prefix):
            if not preds_list:
                return
            preds = torch.cat(preds_list)
            gts = torch.cat(gts_list)
            auc = tm.AUROC(task="binary", ignore_index=-1)(preds, gts)
            ap = tm.AveragePrecision(task="binary", ignore_index=-1)(preds, gts)
            self.log(f"metric/test/{prefix}_auc", auc, prog_bar=True, sync_dist=True)
            self.log(f"metric/test/{prefix}_ap", ap, prog_bar=True, sync_dist=True)

        _log_social(coatt_preds, coatt_gts, "coatt")
        _log_social(laeo_preds, laeo_gts, "laeo")
        _log_social(lah_preds, lah_gts, "lah")
