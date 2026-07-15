"""Reproducibly derive compact VLM pair manifests from canonical JSONL files.

The canonical manifests are never modified.  This utility writes a separate
view that can be used for an EyeVLM-style pilot: restrict sources, take a
deterministic frame-level stride, then sample task-wise negatives without
replacement to match the available positives.

``sid`` is the only frame identifier retained by the existing VLM cache.  It
is contiguous within each source but does *not* retain original video/frame
indices or gaze-event boundaries.  Consequently ``--frame-stride`` is clearly
reported as a *sid-order approximation*, not as an exact reproduction of
EyeVLM's per-video temporal filtering.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from vlm.social.data import SocialSample, SOCIAL_TASKS


Record = dict[str, Any]
DEFAULT_EYEVLM_SOURCES = ("childplay", "videoattentiontarget")


@dataclass(frozen=True)
class ManifestReport:
    """Auditable summary of one derived manifest."""

    input_records: int
    output_records: int
    input_frames: int
    output_frames: int
    sources: tuple[str, ...]
    frame_stride: int
    frame_offset: int
    approximate_sid_stride: bool
    counts: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_records": self.input_records,
            "output_records": self.output_records,
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "sources": list(self.sources),
            "frame_stride": self.frame_stride,
            "frame_offset": self.frame_offset,
            "approximate_sid_stride": self.approximate_sid_stride,
            "counts": self.counts,
        }


def read_manifest(path: str | Path) -> list[Record]:
    """Read and validate a canonical manifest without changing its records."""

    source = Path(path)
    records: list[Record] = []
    seen: set[tuple[str, str, int, int]] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample = SocialSample.from_manifest_record(record)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid manifest record at {source}:{line_number}: {exc}") from exc
            if sample.eval_key in seen:
                raise ValueError(f"duplicate pair annotation at {source}:{line_number}: {sample.eval_key}")
            seen.add(sample.eval_key)
            # A shallow copy prevents an in-memory caller from observing mutations.
            records.append(dict(record))
    return records


def load_sid_sources(gtmeta_path: str | Path) -> dict[str, str]:
    """Read the source dataset name retained in the canonical gtmeta cache."""

    raw = torch.load(gtmeta_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError(f"gtmeta must be a mapping keyed by sid, got {type(raw).__name__}")
    sources: dict[str, str] = {}
    for sid, meta in raw.items():
        if not isinstance(sid, str) or not isinstance(meta, Mapping):
            raise ValueError("gtmeta must map string sid values to metadata mappings")
        dataset = meta.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"gtmeta[{sid!r}] is missing a non-empty dataset name")
        sources[sid] = dataset
    return sources


def _sid_index(sid: str) -> int:
    """Extract the canonical numerical sid suffix for deterministic ordering."""

    if not sid.startswith("sample") or not sid[6:].isdigit():
        raise ValueError(f"frame stride requires canonical sid 'sampleNNNNNN', got {sid!r}")
    return int(sid[6:])


def filter_records(
    records: Sequence[Record],
    *,
    sid_sources: Mapping[str, str] | None = None,
    allowed_sources: Iterable[str] | None = None,
    frame_stride: int = 1,
    frame_offset: int = 0,
) -> list[Record]:
    """Apply source and frame-closed sid-order filters.

    All rows belonging to a retained frame are kept, which prevents a task from
    silently seeing a different set of images than another task.  Striding is
    performed separately per retained source to avoid source-boundary effects.
    """

    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if not 0 <= frame_offset < frame_stride:
        raise ValueError(
            f"frame_offset must be in [0, {frame_stride}), got {frame_offset}"
        )
    allowed = None if allowed_sources is None else set(allowed_sources)
    if allowed is not None and not allowed:
        raise ValueError("allowed_sources must be non-empty when provided")
    if allowed is not None and sid_sources is None:
        raise ValueError("source filtering requires sid_sources from --gtmeta")

    frame_sources: dict[str, str] = {}
    for record in records:
        sid = str(record["sid"])
        if sid_sources is not None:
            try:
                frame_sources[sid] = sid_sources[sid]
            except KeyError as exc:
                raise ValueError(f"manifest sid {sid!r} is absent from gtmeta") from exc
        elif allowed is not None:
            raise AssertionError("source validation above should have failed")
        else:
            frame_sources[sid] = "__all__"

    candidate_sids = [
        sid for sid in frame_sources
        if allowed is None or frame_sources[sid] in allowed
    ]
    source_sids: dict[str, list[str]] = defaultdict(list)
    for sid in candidate_sids:
        source_sids[frame_sources[sid]].append(sid)
    kept_sids: set[str] = set()
    for sids in source_sids.values():
        # Numerical sid order is deterministic even when input JSONL order changes.
        for index, sid in enumerate(sorted(sids, key=_sid_index)):
            if index % frame_stride == frame_offset:
                kept_sids.add(sid)
    return [dict(record) for record in records if str(record["sid"]) in kept_sids]


def balance_task_labels(records: Sequence[Record], *, seed: int) -> list[Record]:
    """Keep all positives and sample exactly as many task-matched negatives.

    This is the EyeVLM table's documented Pos:Neg=1:1 operation.  It is
    intentionally *not* a task-frequency equalizer: LAH can retain more rows
    than LAEO/SA, as in the paper.  Sampling is without replacement.
    """

    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        sample = SocialSample.from_manifest_record(record)
        buckets[(sample.task, sample.label)].append(index)

    rng = random.Random(seed)
    selected: set[int] = set()
    for task in SOCIAL_TASKS:
        positives = buckets[(task, 1)]
        negatives = buckets[(task, 0)]
        if not positives:
            raise ValueError(f"cannot balance {task}: no positive annotations after filtering")
        if len(negatives) < len(positives):
            raise ValueError(
                f"cannot balance {task}: {len(positives)} positives but only "
                f"{len(negatives)} negatives after filtering"
            )
        selected.update(positives)
        chosen_negatives = rng.sample(negatives, k=len(positives))
        selected.update(chosen_negatives)
    # Preserve the original row order.  The train sampler controls random order;
    # stable output is substantially easier to diff and audit.
    return [dict(record) for index, record in enumerate(records) if index in selected]


def summarize(
    input_records: Sequence[Record],
    output_records: Sequence[Record],
    *,
    sid_sources: Mapping[str, str] | None,
    sources: Iterable[str] | None,
    frame_stride: int,
    frame_offset: int,
) -> ManifestReport:
    counts = {task: {"yes": 0, "no": 0} for task in SOCIAL_TASKS}
    for record in output_records:
        sample = SocialSample.from_manifest_record(record)
        counts[sample.task][sample.answer] += 1
    return ManifestReport(
        input_records=len(input_records),
        output_records=len(output_records),
        input_frames=len({str(record["sid"]) for record in input_records}),
        output_frames=len({str(record["sid"]) for record in output_records}),
        sources=tuple(sorted(sources or ())),
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        approximate_sid_stride=frame_stride > 1,
        counts=counts,
    )


def build_manifest(
    records: Sequence[Record],
    *,
    sid_sources: Mapping[str, str] | None = None,
    allowed_sources: Iterable[str] | None = None,
    frame_stride: int = 1,
    frame_offset: int = 0,
    balance_labels: bool = True,
    seed: int = 101,
) -> tuple[list[Record], ManifestReport]:
    """Derive one auditable manifest view from canonical records."""

    filtered = filter_records(
        records,
        sid_sources=sid_sources,
        allowed_sources=allowed_sources,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
    )
    output = balance_task_labels(filtered, seed=seed) if balance_labels else filtered
    return output, summarize(
        records,
        output,
        sid_sources=sid_sources,
        sources=allowed_sources,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
    )


def write_manifest(path: str | Path, records: Sequence[Record]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="canonical input JSONL (never modified)")
    parser.add_argument("--output", required=True, help="derived output JSONL")
    parser.add_argument("--report", default="", help="optional JSON audit report")
    parser.add_argument("--gtmeta", default="", help="required when --sources is used")
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="keep only these gtmeta dataset names (e.g. childplay videoattentiontarget)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="keep one complete frame every N sid-ordered frames within each source; approximation only",
    )
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--no-balance-labels", action="store_true",
        help="do not apply task-wise Pos:Neg=1:1 negative sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = read_manifest(args.manifest)
    sid_sources = load_sid_sources(args.gtmeta) if args.gtmeta else None
    output, report = build_manifest(
        records,
        sid_sources=sid_sources,
        allowed_sources=args.sources,
        frame_stride=args.frame_stride,
        frame_offset=args.frame_offset,
        balance_labels=not args.no_balance_labels,
        seed=args.seed,
    )
    write_manifest(args.output, output)
    payload = report.as_dict()
    payload.update(
        {
            "input_manifest": str(Path(args.manifest).resolve()),
            "output_manifest": str(Path(args.output).resolve()),
            "gtmeta": str(Path(args.gtmeta).resolve()) if args.gtmeta else None,
            "seed": args.seed,
            "balance_labels": not args.no_balance_labels,
        }
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
