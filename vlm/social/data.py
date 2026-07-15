"""Canonical pair-annotation dataset for social-relation VLM fine-tuning.

The VSGaze manifest is already annotation-centric: every JSONL row is one labelled
LAH/LAEO/SA query. This module keeps that one-row-to-one-sample contract. It does not
expand unlabelled person pairs and it does not collapse multiple task annotations from
the same frame.

The only dataset-to-model direction conversion happens through
``mtgs.social_vlm.conventions.manifest_record_to_indices``:

* LAH: ``person_i`` is the looker/source and ``person_j`` is the target.
* LAEO/SA: ``person_i`` and ``person_j`` retain the manifest order.

The original manifest indices are retained separately for the locked evaluation
harness, whose keys still use the raw manifest convention.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from torch.utils.data import Dataset

from mtgs.social_vlm.conventions import manifest_record_to_indices


SOCIAL_TASKS = ("lah", "laeo", "sa")
SOCIAL_TASK_ID = {task: index for index, task in enumerate(SOCIAL_TASKS)}


def _binary_label(value: Any) -> int:
    """Convert a manifest answer to an exact binary target, rejecting ambiguity."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "yes":
            return 1
        if value == "no":
            return 0
    # bool is intentionally accepted as a binary value.
    if value is True or value == 1 or value == 1.0:
        return 1
    if value is False or value == 0 or value == 0.0:
        return 0
    raise ValueError(f"answer must be yes/no or 1/0, got {value!r}")


def _person_index(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}") from exc
    if index < 0 or index != value:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return index


@dataclass(frozen=True)
class SocialSample:
    """One supervised relation query in the canonical internal orientation.

    ``person_i`` and ``person_j`` are the two people named A and B downstream.
    For LAH this always means ``person_i -> person_j``. ``raw_i/raw_j`` are never
    used for feature indexing; they exist only to reconstruct manifest/eval keys.
    """

    sid: str
    task: str
    person_i: int
    person_j: int
    label: int
    raw_i: int
    raw_j: int
    person_i_name: str | None = None
    person_j_name: str | None = None

    @property
    def eval_key(self) -> tuple[str, str, int, int]:
        """Key in the existing prediction/evaluation manifest convention."""
        return self.sid, self.task, self.raw_i, self.raw_j

    @property
    def canonical_key(self) -> tuple[str, str, int, int]:
        """Key in model-internal Person-i/Person-j orientation."""
        return self.sid, self.task, self.person_i, self.person_j

    @property
    def answer(self) -> str:
        return "yes" if self.label else "no"

    @classmethod
    def from_manifest_record(cls, record: Mapping[str, Any]) -> "SocialSample":
        missing = {"sid", "task", "i", "j", "ans"}.difference(record)
        if missing:
            raise ValueError(f"manifest record is missing fields: {sorted(missing)}")

        sid = record["sid"]
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError(f"sid must be a non-empty string, got {sid!r}")

        task = record["task"]
        if task not in SOCIAL_TASKS:
            raise ValueError(f"task must be one of {SOCIAL_TASKS}, got {task!r}")

        raw_i = _person_index(record["i"], "i")
        raw_j = _person_index(record["j"], "j")
        if raw_i == raw_j:
            raise ValueError(f"self-pair is not a social relation sample: i=j={raw_i}")

        # The single source of truth for the raw VSGaze -> internal conversion.
        person_i, person_j = manifest_record_to_indices(
            {"task": task, "i": raw_i, "j": raw_j}
        )

        # Preserve display names in the same canonical orientation when available.
        raw_names = {raw_i: record.get("li"), raw_j: record.get("lj")}
        name_i = raw_names.get(person_i)
        name_j = raw_names.get(person_j)
        if name_i is not None:
            name_i = str(name_i)
        if name_j is not None:
            name_j = str(name_j)

        return cls(
            sid=sid,
            task=task,
            person_i=person_i,
            person_j=person_j,
            label=_binary_label(record["ans"]),
            raw_i=raw_i,
            raw_j=raw_j,
            person_i_name=name_i,
            person_j_name=name_j,
        )


class SocialAnnotationDataset(Dataset):
    """Load every labelled manifest row as exactly one :class:`SocialSample`.

    The dataset performs no class balancing or task filtering. Those are sampler
    concerns and must not silently change which annotations constitute the dataset.
    Duplicate ``(sid, task, raw_i, raw_j)`` rows are rejected by default because
    they would train the same ground-truth annotation more than once.
    """

    def __init__(self, manifest: str | Path, *, reject_duplicates: bool = True):
        self.manifest = Path(manifest)
        self.samples: list[SocialSample] = []
        seen: dict[tuple[str, str, int, int], int] = {}

        with self.manifest.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    sample = SocialSample.from_manifest_record(record)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid pair annotation at {self.manifest}:{line_number}: {exc}"
                    ) from exc

                if reject_duplicates and sample.eval_key in seen:
                    first = seen[sample.eval_key]
                    raise ValueError(
                        f"duplicate pair annotation at {self.manifest}:{line_number}; "
                        f"first seen on line {first}: {sample.eval_key}"
                    )
                seen[sample.eval_key] = line_number
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SocialSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[SocialSample]:
        return iter(self.samples)

    @property
    def task_counts(self) -> dict[str, int]:
        counts = Counter(sample.task for sample in self.samples)
        return {task: counts[task] for task in SOCIAL_TASKS}

    @property
    def class_counts(self) -> dict[str, dict[int, int]]:
        counts = Counter((sample.task, sample.label) for sample in self.samples)
        return {
            task: {0: counts[(task, 0)], 1: counts[(task, 1)]}
            for task in SOCIAL_TASKS
        }

    @property
    def frame_count(self) -> int:
        return len({sample.sid for sample in self.samples})


def frame_path(frame_root: str | Path, sample: SocialSample) -> Path:
    """Return the shared raw-frame cache path without loading or copying the image."""
    return Path(frame_root) / sample.sid / "frame.png"
