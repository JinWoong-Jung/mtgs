"""Build a ``sid -> center-frame path`` sidecar for a VSGaze split.

The graph cache (gtmeta / vlmgraph) keys frames by ``sid = sample{global_index}``
-- the deterministic dataset-iteration index render.py used -- but stores no
path. It therefore cannot be joined to the *shuffled* standalone
``test_predictions.p`` by frame content alone: static-camera video frames share
rounded bbox + social-GT signatures (~3k colliding groups on the test split,
whose graph predictions differ by up to 0.9), so a content hash is ambiguous.

The center-frame path IS unique per frame. For the test split each temporal
sub-dataset resolves ``__getitem__(index) -> self.paths[index]`` with no image
decode and no RNG, so this sidecar is a cheap, exact, order-independent join
key that ties render.py's sid space to the native ``test_predictions.p`` records.

CLI:
  python -m vlm.cache.legacy_test_index --split test --out <sid_path.json>
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch

from vlm.cache.config import make_cfg
from mtgs.train.dataset import build_dataset


_STAGE = {"train": "fit", "val": "validate", "test": "test"}
_ATTR = {"train": "train_dataset", "val": "val_dataset", "test": "test_dataset"}


def build_sid_path_index(split: str = "test", num_people="all") -> dict[str, str]:
    """Return ``{sid: center_frame_path}`` for every frame of ``split``.

    Mirrors render.py's dataset construction (same seed / cfg) so the sid space
    is identical. Reads ``.paths`` directly from each ConcatDataset member -- no
    ``__getitem__``, so no images are decoded.
    """
    torch.manual_seed(101)
    np.random.seed(101)
    random.seed(101)
    cfg = make_cfg(split, num_people=num_people)
    cfg.device = "cpu"
    data = build_dataset(**cfg)
    data.setup(_STAGE[split])
    ds = getattr(data, _ATTR[split])

    if not hasattr(ds, "datasets") or not hasattr(ds, "cumulative_sizes"):
        raise TypeError(f"{split} dataset is not a ConcatDataset: {type(ds).__name__}")
    cumulative = ds.cumulative_sizes
    for member in ds.datasets:
        if not hasattr(member, "paths"):
            raise TypeError(f"sub-dataset {type(member).__name__} has no .paths")

    index: dict[str, str] = {}
    for idx in range(len(ds)):
        member_idx = bisect.bisect_right(cumulative, idx)
        local = idx - (cumulative[member_idx - 1] if member_idx > 0 else 0)
        index[f"sample{idx:06d}"] = str(ds.datasets[member_idx].paths[local])
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = build_sid_path_index(args.split)
    unique = len(set(index.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(index))
    print(f"[legacy-test-index] {len(index)} sids, {unique} unique paths -> {args.out}", flush=True)
    if unique != len(index):
        raise SystemExit("non-unique center paths in sidecar -- join key is not safe")


if __name__ == "__main__":
    main()
