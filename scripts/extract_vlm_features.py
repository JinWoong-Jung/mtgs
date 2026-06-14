# scripts/extract_vlm_features.py
"""Offline extraction of center-frame gaze-graph features for Stage-B (VLM).

Runs the frozen MTGS backbone once over the VSGaze train/val splits and writes,
per usable sample, the center-frame edge tensor + node features + the labels the
QA collator needs. Samples that yield zero annotated QA pairs are skipped so the
cache contains only useful clips (contiguous indices).

Usage:
    python extract_vlm_features.py \
        vlm.stage_a_ckpt=/path/to/stage_a.ckpt \
        vlm.feature_cache.dir=/path/to/cache_dir \
        data.num_people=11

The extractor forces batch_size=1 so every sample is stored at its native N
(no cross-sample padding), keeping cached_collate_fn trivial. Output is plain
torch files (no h5py dependency): <cache_dir>/<split>/<i>.pt + meta.pt.
"""
import os

import hydra
import torch
from omegaconf import DictConfig
import lightning.pytorch as pl

from mtgs.config import ConfigManager
from mtgs.networks.vlm.mtgs_builder import (
    build_mtgs, load_stage_a_into, attach_graph_state_hooks,
)
from mtgs.datasets.vlm_datamodule import VLMDataModule
from mtgs.datasets.gaze_qa import GazeQACollator


@torch.no_grad()
def _extract_split(loader, mtgs, graph_states, qa_collator, out_dir, meta):
    """Run MTGS over a split and write usable samples as per-sample torch files."""
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    skipped_empty = 0

    for batch in loader:
        # Move tensors to device for the MTGS forward (lists pass through).
        dev_batch = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        mtgs(dev_batch)

        E = graph_states["E"]               # (1, T, N, Tl, De)
        v_src = graph_states["v_src"]       # (1, T, N, De)
        v_tgt = graph_states["v_tgt"]       # (1, T, Tl, De)
        edge_valid = graph_states["edge_valid"]  # (1, N, 2N+2)
        t_c = E.shape[1] // 2

        # Skip samples with no annotated QA pairs (saves space + train time).
        qa_pairs = qa_collator(batch)
        if not qa_pairs:
            skipped_empty += 1
            continue

        sample = {
            "E_c":        E[0, t_c].to(torch.float16).cpu(),
            "v_src_c":    v_src[0, t_c].to(torch.float16).cpu(),
            "v_tgt_c":    v_tgt[0, t_c].to(torch.float16).cpu(),
            "edge_valid": edge_valid[0].bool().cpu(),
            "lah_labels":   batch["lah_labels"][0, t_c].long().cpu(),
            "laeo_labels":  batch["laeo_labels"][0, t_c].long().cpu(),
            "coatt_labels": batch["coatt_labels"][0, t_c].long().cpu(),
            "head_bboxes":  batch["head_bboxes"][0, t_c].float().cpu(),
            "num_valid_people": batch["num_valid_people"][0, t_c].long().cpu(),
        }
        torch.save(sample, os.path.join(out_dir, f"{written}.pt"))
        written += 1

        if written % 500 == 0:
            print(f"  [{out_dir}] wrote {written} samples "
                  f"(skipped {skipped_empty} empty)")

    meta = dict(meta)
    meta["num_samples"] = written
    torch.save(meta, os.path.join(out_dir, "meta.pt"))
    print(f"  [{out_dir}] DONE: {written} samples written, "
          f"{skipped_empty} skipped (no QA pairs)")
    return written


@hydra.main(
    config_path="./../mtgs/config/",
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    ConfigManager.set_config(cfg)
    pl.seed_everything(cfg.train.get("seed", 42))
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)

    stage_a_ckpt = cfg.vlm.get("stage_a_ckpt", None)
    if not stage_a_ckpt:
        raise ValueError("vlm.stage_a_ckpt must be set for feature extraction")
    cache_dir = cfg.vlm.feature_cache.dir

    # ── Frozen MTGS + hooks ───────────────────────────────────────────────────
    mtgs = build_mtgs(cfg).cuda().eval()
    load_stage_a_into(mtgs, stage_a_ckpt)
    for p in mtgs.parameters():
        p.requires_grad_(False)
    graph_states: dict = {}
    attach_graph_state_hooks(mtgs, graph_states)

    qa_collator = GazeQACollator()

    # ── Data (force batch_size=1, no shuffle for stable native-N storage) ─────
    dm = VLMDataModule(cfg)
    dm.setup()

    from torch.utils.data import DataLoader
    from mtgs.train.collate import pad_collate_fn
    nw = getattr(cfg.train, "num_workers", 4)

    def _loader(ds):
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=nw,
                          collate_fn=pad_collate_fn, pin_memory=True)

    meta = {
        "edge_dim": int(cfg.gaze_graph.edge_dim),
        "num_people": str(cfg.data.num_people),
        "stage_a_ckpt": str(stage_a_ckpt),
    }

    print(f"Extracting TRAIN split → {cache_dir}/train")
    _extract_split(_loader(dm.train_ds), mtgs, graph_states, qa_collator,
                   os.path.join(cache_dir, "train"), meta)

    print(f"Extracting VAL split → {cache_dir}/val")
    _extract_split(_loader(dm.val_ds), mtgs, graph_states, qa_collator,
                   os.path.join(cache_dir, "val"), meta)

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
