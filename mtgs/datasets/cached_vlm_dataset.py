# mtgs/datasets/cached_vlm_dataset.py
"""Precomputed all-frame graph features for Stage-B (VLM) train/val/test.

The offline extractor (scripts/extract_vlm_features.py) runs the frozen MTGS
backbone once and stores, per usable sample, the full-temporal graph evidence
plus the labels/bboxes the QA collator needs. Stage-B train/val/test then skip
the MTGS forward entirely when the cache is enabled.

On-disk layout (single HDF5 file per split, one group per usable sample):
    <cache_dir>/<split>.h5
        .attrs: num_samples, edge_dim, num_people, stage_a_ckpt,
                has_vis_tokens, has_image, has_image_jpeg
        /<i>/   one group per usable sample, i = 0..M-1:
            E                (T, N, N+2, De)  float16   all T frames
            v_src            (T, N, De)       float16
            v_tgt            (T, N+2, De)     float16
            edge_valid       (N, 2N+2)        bool      (not temporal)
            lah_labels       (T, P)           int64   P = N*(N-1)
            laeo_labels      (T, P)           int64
            coatt_labels     (T, P)           int64
            head_bboxes      (T, N, 4)        float32
            num_valid_people (T,)             int64
            vis_tokens       (L_vis, d_llm)   float16  (if has_vis_tokens=true)
            image            (C, H, W)        float16  (if has_image=true, legacy)
            image_jpeg       (variable uint8[]) JPEG bytes (if has_image_jpeg=true)
"""
import io
import os
from functools import partial

import h5py
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader


class CachedVLMDataset(Dataset):
    """Reads one split's HDF5 cache file (one group per sample).

    The HDF5 handle is opened lazily inside each worker process (never in the
    parent), so the dataset stays picklable across DataLoader fork.

    Args:
        load_vis: If True, load visual data when present in the cache:
            - ``vis_tokens`` (L_vis, d_llm) float16 — pre-encoded visual prefix tokens.
            - ``image_jpeg`` bytes — JPEG-encoded center frame (new format); decoded
              to PIL and fed directly to the VLM vision tower at train time.
            - ``image`` (C, H, W) float16 — MTGS-normalized frame (legacy format).
            Set to False when visual_encoder=false to skip unnecessary disk I/O.
    """

    def __init__(self, h5_path: str, load_vis: bool = True):
        self.h5_path = h5_path
        self.load_vis = load_vis
        with h5py.File(h5_path, "r") as f:
            self._len = int(f.attrs["num_samples"])
            self.has_vis_tokens  = bool(f.attrs.get("has_vis_tokens",  False))
            self.has_image       = bool(f.attrs.get("has_image",       False))
            self.has_image_jpeg  = bool(f.attrs.get("has_image_jpeg",  False))
        self._h5 = None  # opened per-worker on first access

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        g = self._file()[str(idx)]
        out = {
            "E_c":        torch.from_numpy(g["E"][()]).float(),       # (T, N, N+2, De)
            "v_src_c":    torch.from_numpy(g["v_src"][()]).float(),   # (T, N, De)
            "v_tgt_c":    torch.from_numpy(g["v_tgt"][()]).float(),   # (T, N+2, De)
            "edge_valid": torch.from_numpy(g["edge_valid"][()]).bool(),
            "lah_labels":   torch.from_numpy(g["lah_labels"][()]).long(),    # (T, P)
            "laeo_labels":  torch.from_numpy(g["laeo_labels"][()]).long(),
            "coatt_labels": torch.from_numpy(g["coatt_labels"][()]).long(),
            "head_bboxes":  torch.from_numpy(g["head_bboxes"][()]).float(),  # (T, N, 4)
            "num_valid_people": torch.from_numpy(np.asarray(g["num_valid_people"][()])).long(),
        }
        if self.load_vis:
            if "vis_tokens" in g:
                # New format: pre-encoded (L_vis, d_llm) float16 → load as bf16
                out["vis_tokens"] = torch.from_numpy(g["vis_tokens"][()]).to(torch.bfloat16)
            elif "image_jpeg" in g:
                # JPEG bytes: decoded to PIL at train time → VLM processor directly
                out["image_jpeg"] = g["image_jpeg"][()].tobytes()
            elif "image" in g:
                # Legacy format: raw MTGS-normalized (C, H, W) frame
                out["image"] = torch.from_numpy(g["image"][()]).float()
        return out

    def __getstate__(self):
        # Drop any open handle before pickling to a worker.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


# ── Collate factory ───────────────────────────────────────────────────────────

def cached_collate_fn(batch, *, use_all_frames: bool = False):
    """Collate cached samples into the batch dict the VLM trainer/QA expect.

    E/v_src/v_tgt are all-frame tensors (T, N, ...); when *use_all_frames* is
    False (default) this fn picks t_c = T//2 (center frame); when True it
    mean-pools over T so the full temporal context feeds the graph tokenizer
    while the downstream trainer interface (B, N, Tl, De) remains unchanged.

    Labels, bboxes, and num_valid_people retain the full T dimension so
    GazeQACollator can index [b, t_c] correctly with t_c = T // 2.

    All samples in a batch must share the same N (person count). Train/val
    caches are uniform because extraction uses a fixed num_people. Test uses
    N=all and is evaluated with batch_size=1.
    """
    ns = {s["E_c"].shape[1] for s in batch}   # shape: (T, N, Tl, De) → N is dim 1
    if len(ns) != 1:
        raise ValueError(
            f"cached_collate_fn requires uniform N across the batch, got {sorted(ns)}. "
            "Use batch_size=1 for cached VLM training."
        )

    def stack(key):
        return torch.stack([s[key] for s in batch], dim=0)

    E_all     = stack("E_c")      # (B, T, N, Tl, De)
    v_src_all = stack("v_src_c")  # (B, T, N, De)
    v_tgt_all = stack("v_tgt_c")  # (B, T, Tl, De)

    if use_all_frames:
        # Temporal mean: incorporate all T frames into graph evidence
        E_center    = E_all.mean(dim=1)      # (B, N, Tl, De)
        v_src_center = v_src_all.mean(dim=1) # (B, N, De)
        v_tgt_center = v_tgt_all.mean(dim=1) # (B, Tl, De)
    else:
        t_c = E_all.shape[1] // 2
        E_center    = E_all[:, t_c]          # (B, N, Tl, De)
        v_src_center = v_src_all[:, t_c]     # (B, N, De)
        v_tgt_center = v_tgt_all[:, t_c]     # (B, Tl, De)

    out = {
        "E_c":        E_center,
        "v_src_c":    v_src_center,
        "v_tgt_c":    v_tgt_center,
        "edge_valid": stack("edge_valid"),  # (B, N, 2N+2)
        # Full T dimension so GazeQACollator uses t_c = T // 2 correctly:
        "lah_labels":   stack("lah_labels"),    # (B, T, P)
        "laeo_labels":  stack("laeo_labels"),
        "coatt_labels": stack("coatt_labels"),
        "head_bboxes":  stack("head_bboxes"),   # (B, T, N, 4)
        "num_valid_people": stack("num_valid_people"),  # (B, T)
    }
    if "vis_tokens" in batch[0]:
        out["vis_tokens"] = torch.stack([s["vis_tokens"] for s in batch], dim=0)
    elif "image_jpeg" in batch[0]:
        # List of bytes objects — not stackable; decoded to PIL in vlm_trainer
        out["image_jpeg"] = [s["image_jpeg"] for s in batch]
    elif "image" in batch[0]:
        out["image"] = stack("image").unsqueeze(1)
    return out


def make_cached_collate_fn(use_all_frames: bool = False):
    """Return a picklable collate function with the given temporal mode baked in."""
    return partial(cached_collate_fn, use_all_frames=use_all_frames)


class CachedVLMDataModule(pl.LightningDataModule):
    """Serves precomputed feature caches for train/val/test splits."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cache_dir = cfg.vlm.feature_cache.dir
        self.train_path = os.path.join(cache_dir, "train.h5")
        self.val_path   = os.path.join(cache_dir, "val.h5")
        self.test_path  = os.path.join(cache_dir, "test.h5")

    def setup(self, stage=None):
        use_vis = bool(self.cfg.vlm.get("visual_encoder", False))
        use_all_frames = bool(self.cfg.vlm.get("use_all_frames", False))
        self.train_ds = CachedVLMDataset(self.train_path, load_vis=use_vis)
        self.val_ds   = CachedVLMDataset(self.val_path,   load_vis=use_vis)
        self.test_ds  = CachedVLMDataset(self.test_path,  load_vis=use_vis)
        self._collate_fn = make_cached_collate_fn(use_all_frames)

    def train_dataloader(self):
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.val.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.make_test_loader()

    def make_test_loader(self):
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
