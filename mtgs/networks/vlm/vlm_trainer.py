# mtgs/networks/vlm/vlm_trainer.py
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig

from mtgs.networks.vlm.reasoner import EvidenceAugmentedVLM
from mtgs.networks.vlm.mtgs_builder import build_mtgs, attach_graph_state_hooks
from mtgs.datasets.gaze_qa import GazeQACollator


class VLMReasonerModel(pl.LightningModule):
    """Stage B: trains EvidenceAugmentedVLM on top of frozen gaze_graph.

    Two modes:
      - online  (default): runs the frozen MTGS backbone every step and hooks
        _UnifiedRefiner to capture E, v_src, v_tgt.
      - cached  (vlm.feature_cache.use=true): reads precomputed center-frame
        graph features directly from the batch — the MTGS backbone is NOT
        instantiated, saving GPU memory and a full forward per step.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.use_cache = bool(
            cfg.get("vlm", {}).get("feature_cache", {}).get("use", False)
        )
        self._graph_states: dict = {}

        if not self.use_cache:
            # ── Frozen MTGS (online extraction) ───────────────────────────────
            self.frozen_mtgs = build_mtgs(cfg)
            for p in self.frozen_mtgs.parameters():
                p.requires_grad_(False)
            self.frozen_mtgs.eval()
            attach_graph_state_hooks(self.frozen_mtgs, self._graph_states)
        else:
            self.frozen_mtgs = None

        # ── Trainable VLM module ──────────────────────────────────────────────
        self.vlm_model = EvidenceAugmentedVLM(cfg)

        # ── QA pair generator ─────────────────────────────────────────────────
        self.qa_collator = GazeQACollator()

    def load_stage_a_weights(self, ckpt_path: str):
        """Load Stage A gaze_graph checkpoint into frozen_mtgs (online mode only)."""
        if self.use_cache:
            return  # cached features already encode Stage A; nothing to load
        from mtgs.networks.vlm.mtgs_builder import load_stage_a_into
        load_stage_a_into(self.frozen_mtgs, ckpt_path)
        self.frozen_mtgs.eval()
        for p in self.frozen_mtgs.parameters():
            p.requires_grad_(False)

    # ── Center-frame graph states (online: run MTGS; cached: read batch) ──────
    def _graph_features(self, batch):
        if self.use_cache:
            return (batch["E_c"], batch["edge_valid"],
                    batch["v_src_c"], batch["v_tgt_c"])

        with torch.no_grad():
            self.frozen_mtgs.eval()
            self.frozen_mtgs(batch)
        E = self._graph_states["E"]
        v_src = self._graph_states["v_src"]
        v_tgt = self._graph_states["v_tgt"]
        edge_valid = self._graph_states["edge_valid"]
        t_c = E.shape[1] // 2
        return E[:, t_c], edge_valid, v_src[:, t_c], v_tgt[:, t_c]

    def training_step(self, batch, batch_idx):
        E_c, edge_valid, v_src_c, v_tgt_c = self._graph_features(batch)

        qa_pairs = self.qa_collator(batch)
        if not qa_pairs:
            return None

        loss = self.vlm_model(E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs)
        self.log("train/loss_vlm", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            E_c, edge_valid, v_src_c, v_tgt_c = self._graph_features(batch)
            qa_pairs = self.qa_collator(batch)
            if not qa_pairs:
                return
            loss = self.vlm_model(E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs)
        self.log("val/loss_vlm", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
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
        warmup_steps = int(sched_cfg.warmup_epochs * total_steps / self.trainer.max_epochs)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
