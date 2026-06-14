# mtgs/datasets/cached_vlm_dataset.py
"""Precomputed center-frame graph features for Stage-B (VLM) training.

The offline extractor (scripts/extract_vlm_features.py) runs the frozen MTGS
backbone once and stores, per usable sample, the center-frame graph evidence
plus the labels/bboxes the QA collator needs. Stage-B training then skips the
MTGS forward entirely.

On-disk layout (no h5py dependency — plain torch files, lazily loaded):
    <cache_dir>/<split>/meta.pt      {"num_samples", "edge_dim", "num_people",
                                       "stage_a_ckpt"}
    <cache_dir>/<split>/<i>.pt       one dict per usable sample, i = 0..M-1:
        E_c              (N, N+2, De)   float16
        v_src_c          (N, De)        float16
        v_tgt_c          (N+2, De)      float16
        edge_valid       (N, 2N+2)      bool
        lah_labels       (P,)           int64   P = N*(N-1), center frame
        laeo_labels      (P,)           int64
        coatt_labels     (P,)           int64
        head_bboxes      (N, 4)         float32  center frame
        num_valid_people scalar         int64
"""
import os

import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader


class CachedVLMDataset(Dataset):
    """Reads one split's per-sample torch cache directory."""

    def __init__(self, split_dir: str):
        self.split_dir = split_dir
        meta = torch.load(os.path.join(split_dir, "meta.pt"), weights_only=False)
        self._len = int(meta["num_samples"])
        self.meta = meta

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        s = torch.load(os.path.join(self.split_dir, f"{idx}.pt"),
                       weights_only=False)
        return {
            "E_c":        s["E_c"].float(),
            "v_src_c":    s["v_src_c"].float(),
            "v_tgt_c":    s["v_tgt_c"].float(),
            "edge_valid": s["edge_valid"].bool(),
            "lah_labels":   s["lah_labels"].long(),
            "laeo_labels":  s["laeo_labels"].long(),
            "coatt_labels": s["coatt_labels"].long(),
            "head_bboxes":  s["head_bboxes"].float(),
            "num_valid_people": s["num_valid_people"].long(),
        }


def cached_collate_fn(batch):
    """Collate cached samples into the batch dict the VLM trainer/QA expect.

    Adds a singleton temporal axis to label/bbox/nv keys so GazeQACollator
    (which indexes [b, t_c]) works with t_c = 0.

    All samples in a batch must share the same N (person count). The extractor
    stores native-N samples, so cache training uses batch_size=1 — under which
    this is trivially satisfied. For batch_size>1, only uniform-N batches are
    supported (a clear error is raised otherwise).
    """
    ns = {s["E_c"].shape[0] for s in batch}
    if len(ns) != 1:
        raise ValueError(
            f"cached_collate_fn requires uniform N across the batch, got {sorted(ns)}. "
            "Use batch_size=1 for cached VLM training."
        )

    def stack(key):
        return torch.stack([s[key] for s in batch], dim=0)

    return {
        "E_c":        stack("E_c"),         # (B, N, Tl, De)
        "v_src_c":    stack("v_src_c"),     # (B, N, De)
        "v_tgt_c":    stack("v_tgt_c"),     # (B, Tl, De)
        "edge_valid": stack("edge_valid"),  # (B, N, 2N+2)
        # T=1 singleton temporal axis for the QA collator:
        "lah_labels":   stack("lah_labels").unsqueeze(1),    # (B, 1, P)
        "laeo_labels":  stack("laeo_labels").unsqueeze(1),
        "coatt_labels": stack("coatt_labels").unsqueeze(1),
        "head_bboxes":  stack("head_bboxes").unsqueeze(1),   # (B, 1, N, 4)
        "num_valid_people": stack("num_valid_people").unsqueeze(1),  # (B, 1)
    }


class CachedVLMDataModule(pl.LightningDataModule):
    """Serves precomputed feature caches for train/val splits."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cache_dir = cfg.vlm.feature_cache.dir
        self.train_dir = os.path.join(cache_dir, "train")
        self.val_dir   = os.path.join(cache_dir, "val")

    def setup(self, stage=None):
        self.train_ds = CachedVLMDataset(self.train_dir)
        self.val_ds   = CachedVLMDataset(self.val_dir)

    def train_dataloader(self):
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=cached_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cached_collate_fn,
            pin_memory=True,
        )
