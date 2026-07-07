from __future__ import annotations
"""Graph-agnostic per-pair overlay + manifest + gtmeta generation for VSGaze splits.

Ported from peer sgg/data_prep.py (_cmd_render_overlays, lines 179-368).
sid convention: sample{global_frame_idx:06d} — must match graph_export.py (Task 3).

CLI:
  python -m vlm.data_prep overlays --split <s> --out <dir> \
      --manifest <jsonl> --gtmeta <pt> [--limit N] [--workers W]
"""

import argparse
import collections
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtgs.train.dataset import build_dataset
from mtgs.utils.image import IMG_MEAN, IMG_STD
from vlm.cfg import make_cfg
from vlm.overlay import denormalize_to_pil, build_overlay_pair, display_labels


STAGE = {"train": "fit", "val": "validate", "test": "test"}
ATTR  = {"train": "train_dataset", "val": "val_dataset", "test": "test_dataset"}


def _one(b):
    return b[0]


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", required=True, help="root dir for overlay PNGs")
    ap.add_argument("--manifest", default="", help="write manifest JSONL here")
    ap.add_argument("--gtmeta", default="", help="write per-sid GT/bbox/inout cache (.pt) here")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0=all; >0 cap first N frames (smoke)")
    ap.add_argument("--start", type=int, default=0, help="shard: global frame start idx")
    ap.add_argument("--end", type=int, default=0, help="shard: global frame end idx (0=all)")
    ap.add_argument("--shard", type=int, default=0, help="strided shard id (idx %% nshards == shard)")
    ap.add_argument("--nshards", type=int, default=1, help="strided: number of shards")
    args = ap.parse_args()

    # The dataset enumeration order is RNG-dependent at construction; seed so that
    # sid = enum-index is reproducible across processes (render vs eval must agree,
    # and this must match graph_export.py / Task 3).
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    cfg = make_cfg(args.split)
    cfg.device = "cpu"  # CPU-only render pass — no model inference

    data = build_dataset(**cfg)
    data.setup(STAGE[args.split])

    if args.split == "train":
        # The train split applies STOCHASTIC augmentation (RandomCropSafeGaze +
        # ColorJitter + horizontal_flip + random people-subset), all gated on
        # split=="train" — which washes out frames and makes overlays
        # non-deterministic. Reuse the eval (val) transform and flip the gate to
        # "val" on each train sub-dataset for a clean, reproducible overlay cache.
        eval_tf = data.val_dataset.datasets[0].transform
        for sub in data.train_dataset.datasets:
            sub.split = "val"
            sub.transform = eval_tf
        print("[render] train: augmentation disabled (eval transform, split gate -> val)", flush=True)

    ds = getattr(data, ATTR[args.split])

    # Determine which GLOBAL frame indices this process handles.
    # sid stays the global index so shards merge cleanly.
    # STRIDED (idx % nshards == shard) spreads dense videocoatt blocks.
    if args.nshards > 1:
        indices = list(range(args.shard, len(ds), args.nshards))
    else:
        start = args.start
        end = args.end if args.end else len(ds)
        if args.limit:
            end = min(start + args.limit, end)
        indices = list(range(start, min(end, len(ds))))

    sub = torch.utils.data.Subset(ds, indices)
    outroot = Path(args.out) / args.split
    print(f"[render] split={args.split} shard {args.shard}/{args.nshards} "
          f"frames={len(indices)} -> {outroot}", flush=True)

    # num_workers=0 is REQUIRED for sid alignment with graph_export.py: VSGaze
    # __getitem__ consumes numpy/python `random` (people-subset selection) that
    # PyTorch does NOT re-seed per worker, so num_workers>0 desyncs which sample
    # each sid maps to (verified: nw>0 → 0.588 bbox drift vs the canonical nw=0
    # sample). ds[idx] is deterministic at nw=0, so parallelise via --nshards
    # (separate processes over disjoint index ranges), never DataLoader workers.
    dl = DataLoader(sub, batch_size=1, num_workers=0, collate_fn=_one)
    records = []   # manifest, emitted in the SAME pass as overlays
    gtmeta  = {}   # per-sid GT/bbox/inout — eval reads THIS, never re-iterates
    cnt = collections.Counter()
    frames = imgs = skipped = 0

    for local_idx, s in enumerate(tqdm(dl, desc=f"render:{args.split}")):
        idx = indices[local_idx]       # GLOBAL frame index → determines sid
        sid = f"sample{idx:06d}"
        cidx = s["image"].shape[0] // 2
        bb   = s["head_bboxes"][cidx].float()
        n    = bb.shape[0]
        valid = {k for k in range(n)
                 if (bb[k, 2] - bb[k, 0]) > 1e-4 and (bb[k, 3] - bb[k, 1]) > 1e-4}
        if len(valid) < 2:
            continue

        # GT vectors are per ordered pair in permutations(n, 2) order (== compute_metrics).
        pairs = list(itertools.permutations(range(n), 2))
        lah   = s["lah_labels"][cidx]
        laeo  = s["laeo_labels"][cidx]
        coatt = s["coatt_labels"][cidx]
        if not (len(lah) == len(laeo) == len(coatt) == len(pairs)):
            print(f"[render] WARN idx={idx}: label/pair length mismatch "
                  f"({len(lah)}/{len(laeo)}/{len(coatt)} vs {len(pairs)}), skipping", flush=True)
            continue

        vis = torch.zeros(n, dtype=torch.bool)
        vis[list(valid)] = True
        _, lab = display_labels(vis)

        # Build records (GT != -1 only) AND the set of overlays they need —
        # one pass, so manifest and overlays can never diverge.
        frame_recs = []
        needed: set[tuple[int, int]] = set()
        for q, (i, j) in enumerate(pairs):
            if i not in valid or j not in valid:
                continue
            v = float(lah[q])                                 # LAH directed
            if v != -1:
                a = "yes" if v == 1.0 else "no"
                frame_recs.append({"sid": sid, "task": "lah",
                                   "i": i, "j": j,
                                   "li": lab[i], "lj": lab[j], "ans": a})
                needed.add((i, j)); cnt["lah"] += 1
            if i < j:                                         # LAEO / SA undirected
                for task, arr in (("laeo", laeo), ("sa", coatt)):
                    v = float(arr[q])
                    if v != -1:
                        a = "yes" if v == 1.0 else "no"
                        frame_recs.append({"sid": sid, "task": task,
                                           "i": i, "j": j,
                                           "li": lab[i], "lj": lab[j], "ans": a})
                        needed.add((i, j)); cnt[task] += 1
        if not frame_recs:
            continue

        records.extend(frame_recs)
        gtmeta[sid] = {
            "head_bboxes":       bb.clone(),                       # (n, 4)
            "lah_gt":            lah.long().clone(),
            "laeo_gt":           laeo.long().clone(),
            "coatt_gt":          coatt.long().clone(),
            "inout":             s["inout"][cidx].float().clone(), # (n,)
            "num_valid_people":  int(s["num_valid_people"][cidx]),
            "dataset":           s["dataset"],
        }

        boxes = bb.tolist()
        sdir = outroot / f"sample{idx:06d}"
        sdir.mkdir(parents=True, exist_ok=True)
        pil = None
        for (i, j) in needed:
            fp = sdir / f"{i}_{j}.png"
            if fp.exists():
                skipped += 1
                continue
            if pil is None:
                pil = denormalize_to_pil(s["image"][cidx], IMG_MEAN, IMG_STD)
            build_overlay_pair(pil, i, j, boxes, lab).save(fp)
            imgs += 1

        frames += 1
        if frames % 500 == 0:
            print(f"[render] {args.split} {local_idx + 1}/{len(indices)} | "
                  f"{imgs} new, {skipped} skipped, {len(records)} records", flush=True)

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[render] manifest {len(records)} records -> {args.manifest} "
              f"({'  '.join(f'{k}={v}' for k, v in cnt.items())})", flush=True)

    if args.gtmeta:
        Path(args.gtmeta).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gtmeta, args.gtmeta)
        print(f"[render] gtmeta {len(gtmeta)} samples -> {args.gtmeta}", flush=True)

    print(f"[render] DONE split={args.split} frames={frames} new_imgs={imgs} skipped={skipped}",
          flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] != "overlays":
        sys.exit("usage: python -m vlm.data_prep overlays --split <s> --out <dir> --manifest <jsonl> --gtmeta <pt>")
    sys.argv.pop(1); main()
