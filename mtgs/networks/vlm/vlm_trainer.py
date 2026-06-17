# mtgs/networks/vlm/vlm_trainer.py
import io
import os
import pickle
from PIL import Image

import torch
import torchmetrics as tm
import lightning.pytorch as pl
from omegaconf import DictConfig

from mtgs.networks.vlm.reasoner import EvidenceAugmentedVLM
from mtgs.networks.vlm.mtgs_builder import build_mtgs, attach_graph_state_hooks
from mtgs.datasets.gaze_qa import GazeQACollator
from mtgs.performance.compute_metrics import compute as compute_metrics, CPU_Unpickler


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class VLMReasonerModel(pl.LightningModule):
    """Stage B: trains EvidenceAugmentedVLM on top of frozen gaze_graph.

    Graph-feature source is decided per batch:
      - cached batch (has "E_c"): precomputed center-frame features are read
        directly — used for train/val when vlm.feature_cache.use=true.
      - online batch (raw inputs): the frozen MTGS backbone is run and hooked.

    When the cache is enabled, train/val/test all read precomputed features and
    the frozen MTGS backbone is not built. Online mode still builds MTGS and runs
    it for each batch.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.use_cache = bool(
            cfg.get("vlm", {}).get("feature_cache", {}).get("use", False)
        )
        self._graph_states: dict = {}

        # ── Frozen MTGS (online mode only; cache mode uses precomputed h5) ────
        self.frozen_mtgs = None
        if not self.use_cache:
            self.frozen_mtgs = build_mtgs(cfg)
            for p in self.frozen_mtgs.parameters():
                p.requires_grad_(False)
            self.frozen_mtgs.eval()
            attach_graph_state_hooks(self.frozen_mtgs, self._graph_states)

        # ── Trainable VLM module ──────────────────────────────────────────────
        self.vlm_model = EvidenceAugmentedVLM(cfg)

        # ── QA pair generator ─────────────────────────────────────────────────
        self.qa_collator = GazeQACollator()

        # ── Validation metrics ────────────────────────────────────────────────
        self.val_lah_ap    = tm.AveragePrecision(task="binary", ignore_index=-1)
        self.val_laeo_ap   = tm.AveragePrecision(task="binary", ignore_index=-1)
        self.val_coatt_ap  = tm.AveragePrecision(task="binary", ignore_index=-1)

        # Test metrics are computed via compute_metrics (same protocol as gaze_graph).
        # No torchmetrics needed for test — results come from the streamed pickle.

        # ── test_predictions.p handle (opened in on_test_epoch_start) ─────────
        self._pred_file = None
        self._pred_path = None

    def load_stage_a_weights(self, ckpt_path: str):
        """Load Stage A gaze_graph checkpoint into frozen_mtgs for online mode."""
        if self.frozen_mtgs is None:
            return
        from mtgs.networks.vlm.mtgs_builder import load_stage_a_into
        load_stage_a_into(self.frozen_mtgs, ckpt_path)
        self.frozen_mtgs.eval()
        for p in self.frozen_mtgs.parameters():
            p.requires_grad_(False)

    # ── Center-frame graph states (cached batch → read; online batch → MTGS) ──
    def _graph_features(self, batch):
        if "E_c" in batch:                      # cached train/val batch
            return (batch["E_c"], batch["edge_valid"],
                    batch["v_src_c"], batch["v_tgt_c"])

        if self.frozen_mtgs is None:
            raise RuntimeError(
                "Received a raw batch while vlm.feature_cache.use=true. "
                "Run scripts/extract_vlm_features.py to create train/val/test.h5 "
                "or disable vlm.feature_cache.use."
            )

        with torch.no_grad():                   # online train/val/test batch
            self.frozen_mtgs.eval()
            self.frozen_mtgs(batch)
        E = self._graph_states["E"]
        v_src = self._graph_states["v_src"]
        v_tgt = self._graph_states["v_tgt"]
        edge_valid = self._graph_states["edge_valid"]
        t_c = E.shape[1] // 2
        return E[:, t_c], edge_valid, v_src[:, t_c], v_tgt[:, t_c]

    def _build_vis_prefix(self, batch) -> "dict[int, torch.Tensor] | None":
        """Return {b: (1, L_vis, d_llm) bf16} for every batch item, or None.

        Dispatch order:
          1. vis_tokens in batch  → pre-extracted cache (zero cost, fast path)
          2. image_jpeg in batch  → decode JPEG bytes → PIL → VLM vision tower
          3. image in batch       → MTGS-normalized tensor → VLM vision tower (legacy)
          4. none of the above    → visual_encoder disabled / cache is graph-only
        """
        if not _as_bool(self.cfg.vlm.get("visual_encoder", False)):
            return None
        if "vis_tokens" in batch:
            vis = batch["vis_tokens"].to(self.device)   # (B, L_vis, d_llm) bf16
            return {b: vis[b:b+1] for b in range(vis.shape[0])}
        if "image_jpeg" in batch:
            # Decode JPEG bytes → PIL → fed directly to VLM processor (no MTGS denorm)
            return {
                b: self.vlm_model._encode_scene_pil(
                    Image.open(io.BytesIO(batch["image_jpeg"][b])).convert("RGB")
                )
                for b in range(len(batch["image_jpeg"]))
            }
        images = batch.get("image")
        if images is None:
            return None
        t_c = images.shape[1] // 2
        return {b: self.vlm_model._encode_scene(images[b, t_c]) for b in range(images.shape[0])}

    def training_step(self, batch, batch_idx):
        E_c, edge_valid, v_src_c, v_tgt_c = self._graph_features(batch)

        qa_pairs = self.qa_collator(batch)
        if not qa_pairs:
            return None

        vis_prefix = self._build_vis_prefix(batch)
        loss = self.vlm_model(E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs,
                              vis_prefix=vis_prefix)
        self.log("train/loss_vlm", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            E_c, edge_valid, v_src_c, v_tgt_c = self._graph_features(batch)
            qa_pairs = self.qa_collator(batch)
            if not qa_pairs:
                return
            vis_prefix = self._build_vis_prefix(batch)
            loss, scores = self.vlm_model.loss_and_scores(
                E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs,
                vis_prefix=vis_prefix,
            )
            probs = torch.sigmoid(scores.float())
            labels = torch.tensor([qa.label for qa in qa_pairs], device=probs.device)
            for task, metric in (("lah", self.val_lah_ap),
                                 ("laeo", self.val_laeo_ap),
                                 ("sa", self.val_coatt_ap)):
                sel = [i for i, qa in enumerate(qa_pairs) if qa.task == task]
                if sel:
                    metric.update(probs[sel], labels[sel])
        self.log("val/loss_vlm", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        results = {}
        for name, metric in (("lah", self.val_lah_ap),
                             ("laeo", self.val_laeo_ap),
                             ("coatt", self.val_coatt_ap)):
            try:
                results[name] = metric.compute()
            except Exception:
                results[name] = None
        for name, val in results.items():
            if val is not None:
                self.log(f"metric/val/{name}_ap", val, sync_dist=True)
        aps = [v for v in results.values() if v is not None]
        if aps:
            social_ap = torch.stack([torch.as_tensor(a) for a in aps]).float().mean()
            self.log("metric/val/social_ap", social_ap, prog_bar=True, sync_dist=True)
        for metric in (self.val_lah_ap, self.val_laeo_ap, self.val_coatt_ap):
            metric.reset()

    # ── Test: LP/LAEO/SA scores → pred matrix → test_predictions.p ────────────

    def on_test_epoch_start(self):
        out_dir = str(self.cfg.experiment.output_folder)
        os.makedirs(out_dir, exist_ok=True)
        self._pred_path = os.path.join(out_dir, "test_predictions.p")
        self._pred_file = open(self._pred_path, "wb")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        E_c, edge_valid, v_src_c, v_tgt_c = self._graph_features(batch)
        qa_pairs = self.qa_collator(batch)

        B, N = E_c.shape[0], E_c.shape[1]
        P = N * (N - 1)

        # Initialise pred matrices to 0 (invalid pairs stay 0; GT stays -1 → masked)
        lah_pred   = torch.zeros(B, P)
        laeo_pred  = torch.zeros(B, P)
        coatt_pred = torch.zeros(B, P)

        if qa_pairs:
            vis_prefix = self._build_vis_prefix(batch)
            scores = self.vlm_model.score_pairs(
                E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs, vis_prefix=vis_prefix
            )
            probs = torch.sigmoid(scores.float())

            # ── Reconstruct (B, N*(N-1)) pred matrices ────────────────────────
            # Pair index in permutation ordering: src*(N-1) + (dst if dst<src else dst-1)
            # LAEO and SA are symmetric: fill both (src,dst) and (dst,src) so that
            # compute_metrics (which iterates permutations) sees identical scores in
            # both directions — matching the gaze_graph evaluation protocol.
            for qa, p in zip(qa_pairs, probs.cpu()):
                src, dst, b = qa.src_idx, qa.dst_idx, qa.batch_idx
                pidx     = src * (N - 1) + (dst if dst < src else dst - 1)
                pidx_rev = dst * (N - 1) + (src if src < dst else src - 1)
                if qa.task == "lah":
                    lah_pred[b, pidx] = p
                elif qa.task == "laeo":
                    laeo_pred[b, pidx]     = p
                    laeo_pred[b, pidx_rev] = p   # symmetric
                elif qa.task == "sa":
                    coatt_pred[b, pidx]     = p
                    coatt_pred[b, pidx_rev] = p  # symmetric

        # ── Write one record per batch to streaming pickle ────────────────────
        if self._pred_file is not None:
            t_c = batch["lah_labels"].shape[1] // 2
            rec = {
                "lah_pred":   lah_pred,
                "laeo_pred":  laeo_pred,
                "coatt_pred": coatt_pred,
                "lah_gt":     batch["lah_labels"][:, t_c, :].cpu(),
                "laeo_gt":    batch["laeo_labels"][:, t_c, :].cpu(),
                "coatt_gt":   batch["coatt_labels"][:, t_c, :].cpu(),
                "head_bboxes": batch["head_bboxes"][:, t_c, :].cpu(),
                "num_valid_people": batch["num_valid_people"][:, t_c].cpu(),
                # Treat all N people as "in" so LAH loop in compute_metrics runs.
                # VLM pipeline does not predict inout; distance/AP_IO are N/A.
                "inout_gt": torch.ones(B, N, dtype=torch.float32),
                "dataset": ["vsgaze"] * B,
            }
            pickle.dump(rec, self._pred_file)

    def on_test_epoch_end(self):
        # ── Close streaming pickle ────────────────────────────────────────────
        if self._pred_file is not None:
            self._pred_file.close()
            self._pred_file = None

        if self._pred_path is None or not os.path.exists(self._pred_path):
            return

        # ── Reload pickle and run compute_metrics (same protocol as metric.sh) ─
        results = []
        with open(self._pred_path, "rb") as f:
            while True:
                try:
                    results.append(CPU_Unpickler(f).load())
                except EOFError:
                    break

        metrics = compute_metrics(results, shuffle=False, thr=0.5)

        # ── Log to wandb ──────────────────────────────────────────────────────
        for key, val in metrics.items():
            if val is not None:
                self.log(f"metric/test/{key}", float(val), sync_dist=True,
                         prog_bar=(key == "social_ap"))

        print(f"\n  test_predictions.p → {self._pred_path}")
        print(f"  Run:  sbatch scripts/metric.sh \"{self._pred_path}\"")

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        opt_cfg = self.cfg.vlm.optimizer
        sched_cfg = self.cfg.vlm.scheduler

        params = [p for p in self.vlm_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )

        sched_type = sched_cfg.get("type", None)
        if sched_type is None:
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(sched_cfg.warmup_epochs * total_steps / self.trainer.max_epochs))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup_sched = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0,
                                total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_steps])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
