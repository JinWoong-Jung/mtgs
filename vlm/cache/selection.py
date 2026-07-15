"""Frozen, auditable VSGaze person-selection plans for offline VLM caches.

MTGS chooses a random 2..N subset every dataset access for train/validation.
That is correct for online augmentation but cannot keep the VLM's overlays,
gtmeta, and graph export aligned across separate processes.  This module freezes
one seed-controlled draw per canonical VSGaze sample and applies it by original
person ID to each offline extractor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mtgs.train.dataset import build_dataset
from mtgs.utils.social_gaze import get_shuffle_idx
from vlm.cache.config import make_cfg


PLAN_VERSION = 1
STAGE = {"train": "fit", "val": "validate"}
ATTR = {"train": "train_dataset", "val": "val_dataset"}
DATASET_NAMES = {
    "ChildPlayDataset_temporal": "childplay",
    "VideoAttentionTargetDataset_temporal": "videoattentiontarget",
    "VideoLAEODataset_temporal": "laeo",
    "VideoCoAttDataset_temporal": "coatt",
}


def dataset_name(dataset: object) -> str:
    try:
        return DATASET_NAMES[type(dataset).__name__]
    except KeyError as exc:
        raise TypeError(f"unsupported VSGaze dataset type {type(dataset).__name__}") from exc


def _select_person_ids(
    person_ids: np.ndarray,
    inout: np.ndarray,
    *,
    split: str,
    num_people: int,
) -> np.ndarray:
    """Mirror the duplicated MTGS dataset subset code exactly for one access."""
    if split == "train":
        person_ids = person_ids[get_shuffle_idx(inout)]
    num_heads = len(person_ids)
    num_keep = num_heads
    if num_heads > 1:
        num_keep = np.random.randint(2, min(num_heads, num_people) + 1)
    return np.asarray(person_ids[-num_keep:]).copy()


def build_plan(*, split: str, num_people: int = 4, seed: int = 101) -> dict[str, Any]:
    """Build one canonical, seed-controlled MTGS N<=``num_people`` draw per sid."""
    if split not in STAGE:
        raise ValueError(f"split must be one of {tuple(STAGE)}, got {split!r}")
    if num_people < 2:
        raise ValueError(f"num_people must be >= 2, got {num_people}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = make_cfg(split, num_people=num_people)
    cfg.device = "cpu"
    data = build_dataset(**cfg)
    data.setup(STAGE[split])
    datasets = getattr(data, ATTR[split]).datasets

    records: list[dict[str, Any]] = []
    for dataset in datasets:
        name = dataset_name(dataset)
        for path in dataset.paths:
            row = dataset.annotations.get_group(path).iloc[0]
            selected = _select_person_ids(
                np.asarray(row["person_ids"]),
                np.asarray(row["inout"]),
                split=split,
                num_people=num_people,
            )
            records.append(
                {
                    "sid": f"sample{len(records):06d}",
                    "dataset": name,
                    "path": str(path),
                    "person_ids": [int(person_id) for person_id in selected],
                }
            )

    return {
        "version": PLAN_VERSION,
        "split": split,
        "num_people": num_people,
        "seed": seed,
        "selection": "mtgs_train_shuffle_then_random_tail" if split == "train" else "mtgs_val_random_tail",
        "records": records,
    }


def write_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_plan(path: str | Path, *, split: str, num_people: int) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("version") != PLAN_VERSION:
        raise ValueError(f"unsupported selection-plan version in {source}")
    if payload.get("split") != split:
        raise ValueError(f"selection plan split={payload.get('split')!r}, expected {split!r}")
    if payload.get("num_people") != num_people:
        raise ValueError(
            f"selection plan num_people={payload.get('num_people')!r}, expected {num_people}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"selection plan {source} is missing a records list")
    return payload


def apply_plan(dataset: object, plan: Mapping[str, Any]) -> None:
    """Attach a frozen selection mapping to each VSGaze subdataset.

    The temporal datasets consult ``vlm_person_ids_by_path`` before their normal
    stochastic selection.  Verifying sid/dataset/path here prevents silently
    mixing a plan with a differently enumerated dataset version.
    """
    records = plan["records"]
    cursor = 0
    for subdataset in dataset.datasets:
        name = dataset_name(subdataset)
        mapping: dict[str, np.ndarray] = {}
        for path in subdataset.paths:
            if cursor >= len(records):
                raise ValueError("selection plan ended before dataset enumeration")
            record = records[cursor]
            expected_sid = f"sample{cursor:06d}"
            if (
                record.get("sid") != expected_sid
                or record.get("dataset") != name
                or record.get("path") != str(path)
            ):
                raise ValueError(
                    "selection plan/dataset mismatch at "
                    f"{expected_sid}: plan=({record.get('dataset')!r}, {record.get('path')!r}) "
                    f"dataset=({name!r}, {str(path)!r})"
                )
            person_ids = record.get("person_ids")
            if not isinstance(person_ids, list) or not person_ids:
                raise ValueError(f"selection plan {expected_sid} has no person IDs")
            if len(person_ids) > int(plan["num_people"]):
                raise ValueError(f"selection plan {expected_sid} exceeds N={plan['num_people']}")
            mapping[str(path)] = np.asarray(person_ids, dtype=np.int64)
            cursor += 1
        subdataset.vlm_person_ids_by_path = mapping
    if cursor != len(records):
        raise ValueError(f"selection plan has {len(records) - cursor} trailing records")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-people", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    plan = build_plan(split=args.split, num_people=args.num_people, seed=args.seed)
    write_plan(args.out, plan)
    print(
        f"[selection-plan] split={args.split} N={args.num_people} seed={args.seed} "
        f"records={len(plan['records'])} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    _main()
