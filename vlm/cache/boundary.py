from __future__ import annotations
"""Gaze-event boundary detection (Eyes on Gaze / EyeVLM-style transition-frame filtering).

Continuous LAH/LAEO/SA behaviours are annotated as temporal *events*; the few
frames right at an event's onset/offset are where the physical gaze shift is
still in flight, so the instantaneous binary label there is the least reliable
one in the sequence. This module flags, per (sid, task, i, j), whether the
center frame's GT label agrees with every other GT-valid frame inside the
model's local temporal context window (``temporal_context``/``temporal_stride``
in config.yaml — the same window MTGS itself attends over). A pair flagged
True is a local label transition and is a *candidate* for exclusion from a
manifest profile.

This intentionally mirrors vlm/cache/render.py's dataset setup exactly (same
seed, CPU device, selection plan, train-split transform gating) so sid and
person indices line up one-to-one with the canonical manifest/gtmeta caches.
It reads the same per-window ``lah_labels``/``laeo_labels``/``coatt_labels``
render.py already has in memory before collapsing them to the center frame —
no model/checkpoint is involved, and no existing artifact is modified.

Caveat: this only sees the local window (default ±2 steps of stride 3 = ±6
raw frames), not the full annotated event. A pair with no GT-valid neighbor in
that window (e.g. near a clip boundary) cannot be judged and is left
unflagged (kept), which is reported separately from genuine local transitions.

CLI:
  python -m vlm.cache.boundary --split <s> --out <pt> \
      [--num_people N --selection_plan PLAN] [--limit N]
"""

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtgs.train.dataset import build_dataset
from vlm.cache.config import make_cfg
from vlm.cache.selection import apply_plan, load_plan


STAGE = {"train": "fit", "val": "validate", "test": "test"}
ATTR = {"train": "train_dataset", "val": "val_dataset", "test": "test_dataset"}
TASK_FIELDS = (("lah", "lah_labels"), ("laeo", "laeo_labels"), ("sa", "coatt_labels"))


def _one(batch):
    return batch[0]


def pair_boundary_flags(labels: torch.Tensor, cidx: int, n: int) -> torch.Tensor:
    """Per-pair local transition flags for one task.

    ``labels``: (T, num_pairs) GT vector stack in ``itertools.permutations(range(n), 2)``
    order, -1 = invalid/unannotated (matches render.py's convention exactly).
    Returns an (n, n) bool matrix; entry [i, j] is True iff the center frame's
    label for pair (i, j) disagrees with at least one other GT-valid frame in
    the window. Diagonal and GT-invalid-at-center entries are always False —
    render.py already drops those before they reach the manifest, so this
    module leaves the responsibility for that filtering where it already is.
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be (T, num_pairs), got shape {tuple(labels.shape)}")
    t_steps, num_pairs = labels.shape
    if not 0 <= cidx < t_steps:
        raise ValueError(f"cidx={cidx} outside window [0,{t_steps})")
    pairs = list(itertools.permutations(range(n), 2))
    if len(pairs) != num_pairs:
        raise ValueError(f"expected {len(pairs)} pairs for n={n}, got {num_pairs}")

    center = labels[cidx]
    boundary = torch.zeros((n, n), dtype=torch.bool)
    valid_neighbor = torch.zeros((num_pairs,), dtype=torch.bool)
    for t in range(t_steps):
        if t == cidx:
            continue
        neighbor = labels[t]
        seen = neighbor != -1
        valid_neighbor |= seen & (center != -1)
        disagree = seen & (center != -1) & (neighbor != center)
        if bool(disagree.any()):
            for q in disagree.nonzero(as_tuple=True)[0].tolist():
                i, j = pairs[q]
                boundary[i, j] = True
    return boundary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", required=True, help="write per-sid boundary flags (.pt) here")
    ap.add_argument("--num_people", default="all", help="all (default) or a fixed positive cap")
    ap.add_argument(
        "--selection_plan", default="",
        help="frozen vlm.cache.selection JSON; must match the canonical manifest/gtmeta build",
    )
    ap.add_argument("--limit", type=int, default=0, help="0=all; >0 cap first N frames (smoke)")
    args = ap.parse_args()

    torch.manual_seed(101)
    np.random.seed(101)
    random.seed(101)

    num_people = "all" if args.num_people == "all" else int(args.num_people)
    if args.selection_plan and num_people == "all":
        raise ValueError("--selection_plan requires a finite --num_people cap")
    cfg = make_cfg(args.split, num_people=num_people)
    cfg.device = "cpu"

    data = build_dataset(**cfg)
    data.setup(STAGE[args.split])
    ds = getattr(data, ATTR[args.split])
    if args.selection_plan:
        apply_plan(ds, load_plan(args.selection_plan, split=args.split, num_people=num_people))

    if args.split == "train":
        eval_tf = data.val_dataset.datasets[0].transform
        for sub in data.train_dataset.datasets:
            sub.split = "val"
            sub.transform = eval_tf
        print("[boundary] train: augmentation disabled (eval transform, split gate -> val)", flush=True)

    end = len(ds) if not args.limit else min(args.limit, len(ds))
    indices = list(range(end))
    dl = DataLoader(
        torch.utils.data.Subset(ds, indices), batch_size=1, num_workers=0, collate_fn=_one
    )

    cache: dict[str, dict[str, torch.Tensor]] = {}
    # [boundary_pairs, judged_pairs] per task, over GT-valid pairs at the center frame.
    stats = {task: [0, 0] for task, _ in TASK_FIELDS}
    unjudged = {task: 0 for task, _ in TASK_FIELDS}  # valid-at-center but no GT-valid neighbor

    for local_idx, s in enumerate(tqdm(dl, desc=f"boundary:{args.split}")):
        idx = indices[local_idx]
        sid = f"sample{idx:06d}"
        cidx = s["image"].shape[0] // 2
        bb = s["head_bboxes"][cidx].float()
        n = bb.shape[0]
        valid = {
            k for k in range(n)
            if (bb[k, 2] - bb[k, 0]) > 1e-4 and (bb[k, 3] - bb[k, 1]) > 1e-4
        }
        if len(valid) < 2:
            continue

        entry = {}
        for task, field in TASK_FIELDS:
            labels = s[field]
            flags = pair_boundary_flags(labels, cidx, n)
            entry[f"{task}_boundary"] = flags

            center = labels[cidx]
            pairs = list(itertools.permutations(range(n), 2))
            for q, (i, j) in enumerate(pairs):
                if i not in valid or j not in valid:
                    continue
                if task in ("laeo", "sa") and i > j:
                    continue  # undirected: count once, matching render.py
                if float(center[q]) == -1:
                    continue
                stats[task][1] += 1
                if bool(flags[i, j]):
                    stats[task][0] += 1
                else:
                    # Distinguish "confirmed stable" from "no valid neighbor to judge with".
                    any_neighbor = False
                    for t in range(labels.shape[0]):
                        if t != cidx and float(labels[t, q]) != -1:
                            any_neighbor = True
                            break
                    if not any_neighbor:
                        unjudged[task] += 1
        cache[sid] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    print(f"[boundary] saved {len(cache)} sids -> {args.out}", flush=True)
    for task, (boundary_n, judged) in stats.items():
        pct = 100.0 * boundary_n / judged if judged else 0.0
        print(
            f"[boundary] {task}: {boundary_n}/{judged} pairs ({pct:.2f}%) flagged as a local "
            f"transition; {unjudged[task]} pairs had no GT-valid neighbor to judge with",
            flush=True,
        )


if __name__ == "__main__":
    main()
