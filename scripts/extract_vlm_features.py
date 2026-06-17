# scripts/extract_vlm_features.py
"""Offline extraction of all-frame gaze-graph features for Stage-B (VLM).

Runs the frozen MTGS backbone once over VSGaze train/val/test splits and writes,
per usable sample, the full-temporal graph evidence plus labels/bboxes.

HDF5 schema per sample group
-----------------------------
    E                (T, N, N+2, De)  float16   edge tensor, all T frames
    v_src            (T, N, De)       float16   source node features, all frames
    v_tgt            (T, N+2, De)     float16   target node features, all frames
    edge_valid       (N, 2N+2)        bool      valid-edge mask (not temporal)
    lah_labels       (T, N*(N-1))     int64
    laeo_labels      (T, N*(N-1))     int64
    coatt_labels     (T, N*(N-1))     int64
    head_bboxes      (T, N, 4)        float32
    num_valid_people (T,)             int64
    image_jpeg       (variable uint8[]) JPEG bytes iff visual_encoder=true (has_image_jpeg)
"""
import io
import os
import sys

import numpy as np
from PIL import Image

import hydra
import h5py
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import lightning.pytorch as pl
from tqdm import tqdm

from mtgs.utils.image import IMG_MEAN, IMG_STD

from mtgs.config import ConfigManager
from mtgs.networks.vlm.mtgs_builder import (
    build_mtgs, load_stage_a_into, attach_graph_state_hooks,
)
from mtgs.datasets.vlm_datamodule import VLMDataModule
from mtgs.datasets.gaze_qa import GazeQACollator
from mtgs.train.collate import pad_collate_fn


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _identity_collate(samples):
    """Return samples as-is; pad_collate_fn is applied inside the loop."""
    return samples


@torch.no_grad()
def _extract_split(loader, mtgs, graph_states, qa_collator, h5_path, meta,
                   store_image=False, profile_batches=3, profile_only=False):
    """Extract and write one split to HDF5.

    For the first ``profile_batches`` iterations, prints a per-stage wall-time
    breakdown (DataLoader wait / MTGS fwd / h5 write) so the bottleneck is
    measured, not guessed.
    """
    import time

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    written = 0
    skipped_empty = 0

    split_name = os.path.splitext(os.path.basename(h5_path))[0]
    pbar = tqdm(loader, total=len(loader), desc=f"extract[{split_name}]", unit="batch",
                file=sys.stdout)

    with h5py.File(h5_path, "w") as h5:
        _t_prev = time.perf_counter()
        for it, sample_list in enumerate(pbar):
            t_fetch = time.perf_counter()
            B = len(sample_list)

            # ── MTGS forward ────────────────────────────────────────────────
            padded = pad_collate_fn(sample_list)
            dev_padded = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in padded.items()
            }
            mtgs(dev_padded)
            _sync()
            t_mtgs = time.perf_counter()

            E          = graph_states["E"]           # (B, T, N_max, N_max+2, De)
            v_src      = graph_states["v_src"]       # (B, T, N_max, De)
            v_tgt      = graph_states["v_tgt"]       # (B, T, N_max+2, De)
            edge_valid = graph_states["edge_valid"]  # (B, N_max, 2*N_max+2)
            t_c = E.shape[1] // 2                   # center frame index for QA filtering

            # ── QA filtering ─────────────────────────────────────────────────
            qa_pairs    = qa_collator(padded)
            valid_items = {qa.batch_idx for qa in qa_pairs}

            # ── Write per-clip h5 groups ─────────────────────────────────────
            for b in range(B):
                if b not in valid_items:
                    skipped_empty += 1
                    continue

                N_b = sample_list[b]["heads"].shape[1]

                rec = {
                    "E":      E[b, :, :N_b, :(N_b + 2), :].to(torch.float16).cpu().numpy(),
                    "v_src":  v_src[b, :, :N_b, :].to(torch.float16).cpu().numpy(),
                    "v_tgt":  v_tgt[b, :, :(N_b + 2), :].to(torch.float16).cpu().numpy(),
                    "edge_valid": edge_valid[b, :N_b, :(2 * N_b + 2)].bool().cpu().numpy(),
                    "lah_labels":   sample_list[b]["lah_labels"].long().numpy(),
                    "laeo_labels":  sample_list[b]["laeo_labels"].long().numpy(),
                    "coatt_labels": sample_list[b]["coatt_labels"].long().numpy(),
                    "head_bboxes":  sample_list[b]["head_bboxes"].float().numpy(),
                    "num_valid_people": sample_list[b]["num_valid_people"].long().numpy(),
                }
                if store_image:
                    # Reverse MTGS normalisation → JPEG bytes.  At train time
                    # the bytes are decoded to PIL and fed directly to the
                    # Qwen3-VL image_processor (_encode_scene_pil), bypassing
                    # any MTGS denorm step.  JPEG quality=95 is ~10-20 KB vs
                    # ~1 MB for float16, keeping the cache compact.
                    img_t = sample_list[b]["image"][t_c].float()  # (C, H, W)
                    img_std  = torch.tensor(IMG_STD).view(3, 1, 1)
                    img_mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
                    raw = (img_t * img_std + img_mean).clamp(0.0, 1.0)
                    pil = Image.fromarray(
                        (raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    )
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=95)
                    rec["image_jpeg"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)

                g = h5.create_group(str(written))
                for key, arr in rec.items():
                    g.create_dataset(key, data=arr)
                written += 1

            t_write = time.perf_counter()

            if it < profile_batches:
                dl    = t_fetch - _t_prev
                mtgs_dt = t_mtgs - t_fetch
                wr    = t_write - t_mtgs
                total = t_write - _t_prev
                print(f"  [profile it={it}] total={total:6.2f}s | "
                      f"dataloader_wait={dl:6.2f}s  mtgs_fwd={mtgs_dt:6.2f}s  "
                      f"h5_write={wr:5.2f}s  (B={B})", flush=True)
                if profile_only and it + 1 >= profile_batches:
                    print(f"  [profile] profile_only=true → stopping after "
                          f"{profile_batches} batches.", flush=True)
                    break

            _t_prev = time.perf_counter()
            pbar.set_postfix(written=written, skipped=skipped_empty)

        h5.attrs["num_samples"]     = written
        h5.attrs["edge_dim"]        = int(meta["edge_dim"])
        h5.attrs["num_people"]      = str(meta["num_people"])
        h5.attrs["stage_a_ckpt"]   = str(meta["stage_a_ckpt"])
        h5.attrs["has_vis_tokens"]  = False
        h5.attrs["has_image"]       = False           # legacy float16 format not written
        h5.attrs["has_image_jpeg"]  = bool(store_image)

    print(f"  [{h5_path}] {written} samples written, {skipped_empty} skipped")
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
    cache_dir    = cfg.vlm.feature_cache.dir
    store_image  = _as_bool(cfg.vlm.get("visual_encoder", False))
    batch_size   = int(cfg.vlm.get("extract_batch_size", 32))
    nw           = int(getattr(cfg.train, "num_workers", 8))
    profile_only = _as_bool(cfg.vlm.get("profile_only", False))

    print(f"Extraction  batch_size={batch_size}  num_workers={nw}  "
          f"store_image={store_image}  profile_only={profile_only}")

    # ── DataModule + loaders (created BEFORE CUDA init → fork-safe workers) ───
    dm = VLMDataModule(cfg)
    dm.setup()

    def _loader(ds):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, collate_fn=_identity_collate,
            pin_memory=False, persistent_workers=(nw > 0),
        )

    os.makedirs(cache_dir, exist_ok=True)
    train_loader = _loader(dm.train_ds)
    val_loader   = _loader(dm.val_ds)
    # test uses num_people=all (N up to 39) → batch_size=1 to avoid CUDA OOM
    # in the gaze heatmap decoder (activation scales as B×N×T×H×W).
    test_loader  = DataLoader(
        dm._make_dataset("test", num_people="all"),
        batch_size=1, shuffle=False,
        num_workers=nw, collate_fn=_identity_collate,
        pin_memory=False, persistent_workers=(nw > 0),
    )

    # ── Frozen MTGS backbone + graph-state hooks (CUDA after loaders) ────────
    mtgs = build_mtgs(cfg).cuda().eval()
    load_stage_a_into(mtgs, stage_a_ckpt)
    for p in mtgs.parameters():
        p.requires_grad_(False)
    graph_states: dict = {}
    attach_graph_state_hooks(mtgs, graph_states)

    qa_collator = GazeQACollator()

    if store_image:
        print("visual_encoder=true → center-frame JPEG cached (has_image_jpeg); "
              "PIL decoded directly into Qwen3-VL vision tower at train time.")
    else:
        print("visual_encoder=false → graph-only cache")

    base_meta     = {"edge_dim": int(cfg.gaze_graph.edge_dim),
                     "stage_a_ckpt": str(stage_a_ckpt)}
    trainval_meta = {**base_meta, "num_people": str(cfg.data.num_people)}
    test_meta     = {**base_meta, "num_people": "all"}

    skip_done = _as_bool(cfg.vlm.get("skip_done", False))

    train_h5 = os.path.join(cache_dir, "train.h5")
    val_h5   = os.path.join(cache_dir, "val.h5")
    test_h5  = os.path.join(cache_dir, "test.h5")

    if skip_done and os.path.exists(train_h5):
        print(f"Skipping TRAIN (already exists): {train_h5}")
    else:
        print(f"Extracting TRAIN → {train_h5}")
        _extract_split(train_loader, mtgs, graph_states, qa_collator,
                       train_h5, trainval_meta,
                       store_image=store_image, profile_only=profile_only)
    if profile_only:
        print("profile_only=true → done (stage breakdown printed above).")
        return

    if skip_done and os.path.exists(val_h5):
        print(f"Skipping VAL (already exists): {val_h5}")
    else:
        print(f"Extracting VAL   → {val_h5}")
        _extract_split(val_loader, mtgs, graph_states, qa_collator,
                       val_h5, trainval_meta, store_image=store_image)

    print(f"Extracting TEST  → {test_h5}")
    _extract_split(test_loader, mtgs, graph_states, qa_collator,
                   test_h5, test_meta, store_image=store_image)

    print("Extraction complete.")


if __name__ == "__main__":
    main()
