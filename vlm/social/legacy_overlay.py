"""Legacy-compatible VLM overlay onto the MTGS+Graph ``test_predictions.p``.

Goal: add a ``MTGS+Graph + VLM refinement`` row to the *exact* legacy metric
table (``metric_calculation_*.out``) WITHOUT modifying ``mtgs/`` or any
MTGS+Graph training/test code.

Mechanism
---------
The standalone ``test_predictions.p`` is treated as the single source of truth
for the test-frame universe, the GT (``lah_gt``/``laeo_gt``/``coatt_gt``), the
bbox / in-out / dataset structure, and the legacy pooling candidate set (ghost
and ``GT=-1`` pairs included). For the VLM row we copy that stream and overwrite
ONLY the prediction slots the VLM actually queried (labelled manifest pairs).
Every other slot -- padding / ghost / ``GT=-1`` / unqueried valid pair / frame
the VLM never rendered -- keeps the graph's own prediction (graph fallback).
The unmodified :func:`mtgs.performance.compute_metrics.compute` then scores both
streams identically.

Direction (verified against ``mtgs/social_vlm/conventions.py`` and
``mtgs/networks/mtgs_net.py``)::

    native pair vector slot (a, b) == "b looks at a"   (a = target, b = looker)
    graph cache matrix            == [looker, target]
    VLM predictions.pt EvalKey (LAH) == (sid, "lah", raw_i = target, raw_j = looker)  [raw]

So the native flat-vector slot for a VLM LAH key ``(target, looker)`` is::

    q = target * (N - 1) + (looker if looker < target else looker - 1)

which is exactly ``itertools.permutations(range(N), 2)`` index of the ordered
pair ``(first = target, second = looker)``. LAEO/SA are symmetric: the single
``i < j`` VLM probability is written to BOTH directed slots ``(i, j)`` and
``(j, i)``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


# task -> native prediction field. VLM task "sa" maps to the legacy "coatt" name.
_TASK_PRED_FIELD = {"lah": "lah_pred", "laeo": "laeo_pred", "sa": "coatt_pred"}
_SYMMETRIC_TASKS = ("laeo", "sa")


# ── low-level pickle loader (torch storages -> CPU) ───────────────────────────

class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_prediction_stream(path: str | Path) -> list[dict]:
    """Load every record of a legacy ``test_predictions.p`` into a list."""
    records: list[dict] = []
    with open(path, "rb") as handle:
        unpickler = _CPUUnpickler(handle)
        while True:
            try:
                records.append(unpickler.load())
            except EOFError:
                break
    if not records:
        raise ValueError(f"no records loaded from {path}")
    return records


# ── permutation / slot arithmetic ─────────────────────────────────────────────

def permutation_slot(first: int, second: int, n: int) -> int:
    """Index of the ordered pair ``(first, second)`` in ``permutations(range(n), 2)``."""
    if not (0 <= first < n and 0 <= second < n):
        raise ValueError(f"indices out of range for n={n}: ({first}, {second})")
    if first == second:
        raise ValueError(f"pair indices must differ: ({first}, {second})")
    return first * (n - 1) + (second if second < first else second - 1)


def lah_native_slot(target: int, looker: int, n: int) -> int:
    """Native LAH flat-vector slot for ``looker -> target`` (pair (target, looker))."""
    return permutation_slot(target, looker, n)


# ── order-sensitive frame signature ───────────────────────────────────────────

def _round_bbox_bytes(bboxes: torch.Tensor, decimals: int) -> bytes:
    # Order-sensitive: never sort/set. Round to absorb GPU/CPU float epsilon while
    # staying discriminative (head bboxes are continuous, near-unique per frame).
    arr = torch.as_tensor(bboxes, dtype=torch.float64).reshape(-1)
    scaled = torch.round(arr * (10 ** decimals)).to(torch.int64)
    return scaled.numpy().tobytes()


def _gt_bytes(*vectors: torch.Tensor) -> bytes:
    parts = []
    for vec in vectors:
        parts.append(torch.as_tensor(vec, dtype=torch.int64).reshape(-1).numpy().tobytes())
    return b"|".join(parts)


def frame_signature(
    dataset: str,
    head_bboxes: torch.Tensor,
    lah_gt: torch.Tensor,
    laeo_gt: torch.Tensor,
    coatt_gt: torch.Tensor,
    *,
    decimals: int = 3,
) -> str:
    """Deterministic, order-sensitive hash identifying one test frame.

    Uses fields that exist verbatim on BOTH sides (native record and gtmeta
    entry): dataset string, rounded head bboxes (people order preserved), and
    the three flat GT vectors (int, exact). Person ordering is encoded because
    bboxes/GT are hashed in their stored order.
    """
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(str(dataset).encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(_round_bbox_bytes(head_bboxes, decimals))
    hasher.update(b"\x00")
    hasher.update(_gt_bytes(lah_gt, laeo_gt, coatt_gt))
    return hasher.hexdigest()


def _native_signature(record: Mapping[str, Any], decimals: int) -> str:
    return frame_signature(
        record["dataset"][0] if isinstance(record["dataset"], (list, tuple)) else record["dataset"],
        record["head_bboxes"].reshape(-1, record["head_bboxes"].shape[-1]),
        record["lah_gt"].reshape(-1),
        record["laeo_gt"].reshape(-1),
        record["coatt_gt"].reshape(-1),
        decimals=decimals,
    )


# ── sid <-> native frame join (by unique center-frame path) ───────────────────

def native_center_path(record: Mapping[str, Any]) -> str:
    """Center-frame path of a native ``test_predictions.p`` record.

    ``record["path"]`` is the temporal window (a list of per-step paths, each
    possibly wrapped in a length-1 list by the batch collate); the middle entry
    is the anchor frame render.py keyed the sid on.
    """
    paths = record["path"]
    center = paths[len(paths) // 2]
    while isinstance(center, (list, tuple)):
        center = center[0]
    return str(center)


@dataclass
class JoinAudit:
    native_records: int = 0
    sidecar_sids: int = 0
    gtmeta_sids: int = 0
    matched: int = 0
    unmatched_sids: list[str] = field(default_factory=list)
    duplicate_native_paths: int = 0
    gt_mismatch: int = 0
    gt_mismatch_examples: list[str] = field(default_factory=list)
    native_only_frames: int = 0  # native frames no sid maps to -> pure graph fallback

    def to_dict(self) -> dict:
        out = self.__dict__.copy()
        out["unmatched_sids"] = self.unmatched_sids[:20]
        out["unmatched_count"] = len(self.unmatched_sids)
        return out


def build_join(
    native_records: Sequence[Mapping[str, Any]],
    sid_to_path: Mapping[str, str],
    *,
    gtmeta: Mapping[str, Mapping[str, Any]] | None = None,
    restrict_sids: Iterable[str] | None = None,
) -> tuple[dict[str, int], JoinAudit]:
    """Map each sid to its native record index via the unique center-frame path.

    ``sid_to_path`` comes from :mod:`vlm.cache.legacy_test_index`. When ``gtmeta``
    is given, GT vectors of every matched frame are checked for exact agreement
    (``gt_mismatch`` must be 0). ``restrict_sids`` limits the join to a subset
    (e.g. only the sids the VLM actually predicted).
    """
    audit = JoinAudit(native_records=len(native_records), sidecar_sids=len(sid_to_path))

    native_by_path: dict[str, int] = {}
    dup_native = 0
    for idx, record in enumerate(native_records):
        path = native_center_path(record)
        if path in native_by_path:
            dup_native += 1
        else:
            native_by_path[path] = idx
    audit.duplicate_native_paths = dup_native

    sids = set(restrict_sids) if restrict_sids is not None else set(sid_to_path)
    if gtmeta is not None:
        audit.gtmeta_sids = len(gtmeta)

    sid_to_native: dict[str, int] = {}
    matched_native_idx: set[int] = set()
    for sid in sids:
        path = sid_to_path.get(sid)
        idx = None if path is None else native_by_path.get(path)
        if idx is None:
            audit.unmatched_sids.append(sid)
            continue
        sid_to_native[sid] = idx
        matched_native_idx.add(idx)
        if gtmeta is not None and sid in gtmeta:
            entry = gtmeta[sid]
            record = native_records[idx]
            ok = all(
                torch.equal(
                    torch.as_tensor(entry[gk]).reshape(-1).long(),
                    record[gk].reshape(-1).long(),
                )
                for gk in ("lah_gt", "laeo_gt", "coatt_gt")
            )
            if not ok:
                audit.gt_mismatch += 1
                if len(audit.gt_mismatch_examples) < 10:
                    audit.gt_mismatch_examples.append(sid)

    audit.matched = len(sid_to_native)
    audit.native_only_frames = len(native_records) - len(matched_native_idx)
    return sid_to_native, audit


# ── overlay ───────────────────────────────────────────────────────────────────

@dataclass
class OverlayAudit:
    overlay_pairs: dict[str, int] = field(default_factory=lambda: {"lah": 0, "laeo": 0, "sa": 0})
    overlay_gt: dict[str, dict[str, int]] = field(
        default_factory=lambda: {t: {"0": 0, "1": 0, "-1": 0} for t in ("lah", "laeo", "sa")}
    )
    vlm_keys_total: int = 0
    vlm_keys_applied: int = 0
    vlm_keys_unmatched_sid: int = 0
    vlm_keys_index_oob: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def overlay_predictions(
    native_records: Sequence[Mapping[str, Any]],
    sid_to_native: Mapping[str, int],
    vlm_probabilities: Mapping[tuple, float],
) -> tuple[list[dict], OverlayAudit]:
    """Return a new record stream with VLM probabilities written into queried slots.

    Only the ``lah_pred`` / ``laeo_pred`` / ``coatt_pred`` tensors of touched
    frames are cloned; the original stream is never mutated. All other fields
    (GT, bboxes, in-out, gaze) are shared by reference and left unchanged.
    """
    audit = OverlayAudit()
    # Shallow-copy every record; deep-clone only the three prediction tensors of
    # frames that receive an overlay (done lazily below).
    new_records: list[dict] = [dict(rec) for rec in native_records]
    cloned: set[int] = set()

    def ensure_cloned(idx: int) -> dict:
        rec = new_records[idx]
        if idx not in cloned:
            for fld in _TASK_PRED_FIELD.values():
                rec[fld] = rec[fld].clone()
            cloned.add(idx)
        return rec

    for key, prob in vlm_probabilities.items():
        audit.vlm_keys_total += 1
        sid, task, raw_i, raw_j = key
        idx = sid_to_native.get(sid)
        if idx is None:
            audit.vlm_keys_unmatched_sid += 1
            continue
        record = new_records[idx]
        n = record["head_bboxes"].shape[1]
        if not (0 <= raw_i < n and 0 <= raw_j < n and raw_i != raw_j):
            audit.vlm_keys_index_oob += 1
            continue

        field_name = _TASK_PRED_FIELD[task]
        if task == "lah":
            slots = [lah_native_slot(target=raw_i, looker=raw_j, n=n)]
        else:  # symmetric: write both directed slots
            slots = [permutation_slot(raw_i, raw_j, n), permutation_slot(raw_j, raw_i, n)]

        rec = ensure_cloned(idx)
        pred = rec[field_name]
        for q in slots:
            pred[0, q] = float(prob)

        # GT bucket for this queried pair (use one representative directed slot).
        gt_field = {"lah": "lah_gt", "laeo": "laeo_gt", "sa": "coatt_gt"}[task]
        gt_val = int(record[gt_field].reshape(-1)[slots[0]].item())
        audit.overlay_pairs[task] += 1
        audit.overlay_gt[task][str(gt_val) if gt_val in (0, 1, -1) else "-1"] += 1
        audit.vlm_keys_applied += 1

    return new_records, audit


# ── direction cross-check (independent verification of join + slot) ───────────

def direction_crosscheck(
    native_records: Sequence[Mapping[str, Any]],
    sid_to_native: Mapping[str, int],
    vlmgraph: Mapping[str, Mapping[str, Any]],
    *,
    samples: int = 500,
    tol: float = 0.05,
) -> dict:
    """Confirm native ``lah_pred[q]`` matches ``sigmoid(graph lah_logits[looker,target])``.

    Both come from the same checkpoint, so agreement validates the sid<->native
    join AND the LAH slot direction simultaneously. Padding/masked slots
    (graph prob ~0) are skipped so we compare only meaningful predictions.
    """
    checked = 0
    mismatches: list[dict] = []
    max_abs = 0.0
    for sid, idx in sid_to_native.items():
        if checked >= samples:
            break
        if sid not in vlmgraph:
            continue
        record = native_records[idx]
        n = record["head_bboxes"].shape[1]
        entry = vlmgraph[sid]
        logits = torch.as_tensor(entry["lah_logits"], dtype=torch.float32)
        gn = logits.shape[0]
        # Restrict to VALID people: the cache stores raw unmasked logits (padding
        # edges get real-ish values), while native masks invalid edges to 0. The
        # VLM only ever queries valid real-real pairs, so compare only those.
        vis = entry.get("vis_mask")
        if vis is None:
            vis = entry.get("person_mask")
        vis = None if vis is None else torch.as_tensor(vis).bool()
        for target in range(min(n, gn)):
            if vis is not None and not bool(vis[target]):
                continue
            for looker in range(min(n, gn)):
                if target == looker:
                    continue
                if vis is not None and not bool(vis[looker]):
                    continue
                graph_p = float(torch.sigmoid(logits[looker, target]))
                q = lah_native_slot(target=target, looker=looker, n=n)
                native_p = float(record["lah_pred"].reshape(-1)[q])
                diff = abs(native_p - graph_p)
                max_abs = max(max_abs, diff)
                if diff > tol:
                    mismatches.append(
                        {"sid": sid, "target": target, "looker": looker,
                         "q": q, "native_p": round(native_p, 4), "graph_p": round(graph_p, 4)}
                    )
                checked += 1
                if checked >= samples:
                    break
            if checked >= samples:
                break
    return {
        "checked": checked,
        "max_abs_diff": round(max_abs, 5),
        "tol": tol,
        "mismatches": len(mismatches),
        "examples": mismatches[:10],
        "pass": len(mismatches) == 0 and checked > 0,
    }


# ── compute() invocation with captured table ──────────────────────────────────

def run_compute(records: Sequence[Mapping[str, Any]], *, thr: float = 0.5) -> tuple[dict, str]:
    """Run the UNMODIFIED legacy evaluator, returning (scalar dict, log table)."""
    from mtgs.performance.compute_metrics import compute

    logger = logging.getLogger("mtgs.performance.compute_metrics")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(message)s"))
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        result = compute(list(records), shuffle=False, thr=thr)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    return result, buffer.getvalue()


def _parse_out_table(path: str | Path) -> dict[str, float]:
    """Parse the key AP/AUC values from a metric_calculation_*.out file."""
    section = None
    values: dict[str, float] = {}
    import re
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if text.startswith("----- LAEO"):
            section = "laeo"
        elif text.startswith("----- LAH"):
            section = "lah"
        elif text.startswith("----- CoAtt"):
            section = "coatt"
        elif section and text.startswith("AP") and ":" in text and "AP_" not in text:
            m = re.search(r"-?\d+\.\d+", text)
            if m and f"{section}_ap" not in values:
                values[f"{section}_ap"] = float(m.group())
        elif section and text.startswith("AUC") and ":" in text:
            m = re.search(r"-?\d+\.\d+", text)
            if m and f"{section}_auc" not in values:
                values[f"{section}_auc"] = float(m.group())
    return values


def baseline_gate(
    records: Sequence[Mapping[str, Any]],
    base_out: str | Path | None,
    *,
    expected_records: int = 43581,
    tol: float = 1e-3,
) -> dict:
    """Reproduce the historical .out with the unmodified compute() before overlay."""
    result, table = run_compute(records)
    report: dict[str, Any] = {
        "record_count": len(records),
        "record_count_ok": len(records) == expected_records,
        "recomputed": {k: round(float(v), 4) for k, v in result.items()
                       if isinstance(v, (int, float))},
    }
    if base_out is not None and Path(base_out).exists():
        expected = _parse_out_table(base_out)
        diffs = {}
        ok = True
        for key, exp in expected.items():
            got = result.get(key)
            if got is None:
                diffs[key] = {"expected": exp, "got": None}
                ok = False
                continue
            d = abs(float(got) - exp)
            diffs[key] = {"expected": exp, "got": round(float(got), 4), "diff": round(d, 5)}
            if d > tol:
                ok = False
        report["expected_from_out"] = expected
        report["diffs"] = diffs
        report["reproduces_out"] = ok
    report["pass"] = report["record_count_ok"] and report.get("reproduces_out", True)
    report["_table"] = table
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def _load_vlm_probabilities(path: str | Path) -> dict[tuple, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    probs = state["probabilities"] if isinstance(state, dict) and "probabilities" in state else state
    out: dict[tuple, float] = {}
    for key, value in probs.items():
        sid, task, raw_i, raw_j = key
        out[(str(sid), str(task), int(raw_i), int(raw_j))] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_pred", required=True, help="MTGS+Graph test_predictions.p")
    ap.add_argument("--base_out", default="", help="metric_calculation_*.out for the baseline gate")
    ap.add_argument("--sid_path", required=True,
                    help="sid->path sidecar JSON from vlm.cache.legacy_test_index")
    ap.add_argument("--gtmeta", default="", help="graph_cache/gtmeta_test.pt for GT-agreement check")
    ap.add_argument("--vlm_pred", default="", help="VLM predictions.pt to overlay (skip = audit only)")
    ap.add_argument("--vlmgraph", default="", help="graph_cache/vlmgraph_test.pt for direction cross-check")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--expected_records", type=int, default=43581)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)

    print(f"[legacy-overlay] loading base predictions: {args.base_pred}", flush=True)
    records = load_prediction_stream(args.base_pred)

    # Gate 1: baseline reproduction.
    print("[legacy-overlay] gate 1: baseline reproduction", flush=True)
    base_report = baseline_gate(records, args.base_out or None,
                                expected_records=args.expected_records)
    base_table = base_report.pop("_table")
    (out_dir / "baseline_reproduction.json").write_text(json.dumps(base_report, indent=2))
    (out_dir / "metric_calculation_baseline_recompute.out").write_text(base_table)
    print(f"  record_count={base_report['record_count']} "
          f"reproduces_out={base_report.get('reproduces_out')} pass={base_report['pass']}", flush=True)
    if not base_report["pass"]:
        raise SystemExit("baseline gate FAILED -- refusing to overlay")

    # Join (by unique center-frame path).
    print("[legacy-overlay] building sid <-> native join (path sidecar)", flush=True)
    sid_to_path = {str(k): str(v) for k, v in json.loads(Path(args.sid_path).read_text()).items()}
    gtmeta = torch.load(args.gtmeta, map_location="cpu", weights_only=False) if args.gtmeta else None
    restrict = set(gtmeta) if gtmeta is not None else None
    sid_to_native, join_audit = build_join(records, sid_to_path, gtmeta=gtmeta, restrict_sids=restrict)
    (out_dir / "join_audit.json").write_text(json.dumps(join_audit.to_dict(), indent=2))
    print(f"  matched={join_audit.matched} "
          f"dup_native_paths={join_audit.duplicate_native_paths} "
          f"unmatched={len(join_audit.unmatched_sids)} "
          f"gt_mismatch={join_audit.gt_mismatch} "
          f"native_only={join_audit.native_only_frames}", flush=True)
    join_ok = (
        join_audit.duplicate_native_paths == 0
        and len(join_audit.unmatched_sids) == 0
        and join_audit.gt_mismatch == 0
        and join_audit.matched > 0
    )
    if not join_ok:
        raise SystemExit("join gate FAILED -- path join not 1:1 or GT disagrees")

    # Gate 2: direction cross-check.
    if args.vlmgraph:
        print("[legacy-overlay] gate 2: LAH direction cross-check", flush=True)
        vlmgraph = torch.load(args.vlmgraph, map_location="cpu", weights_only=False)
        cross = direction_crosscheck(records, sid_to_native, vlmgraph)
        (out_dir / "direction_crosscheck.json").write_text(json.dumps(cross, indent=2))
        print(f"  checked={cross['checked']} max_abs_diff={cross['max_abs_diff']} "
              f"mismatches={cross['mismatches']} pass={cross['pass']}", flush=True)
        if not cross["pass"]:
            raise SystemExit("direction cross-check FAILED")

    if not args.vlm_pred:
        print("[legacy-overlay] no --vlm_pred: audit-only run complete", flush=True)
        return

    # Overlay + score.
    print(f"[legacy-overlay] overlaying VLM predictions: {args.vlm_pred}", flush=True)
    vlm_probs = _load_vlm_probabilities(args.vlm_pred)
    new_records, overlay_audit = overlay_predictions(records, sid_to_native, vlm_probs)
    (out_dir / "overlay_coverage.json").write_text(json.dumps(overlay_audit.to_dict(), indent=2))
    print(f"  overlay_pairs={overlay_audit.overlay_pairs} "
          f"applied={overlay_audit.vlm_keys_applied}/{overlay_audit.vlm_keys_total} "
          f"unmatched_sid={overlay_audit.vlm_keys_unmatched_sid} "
          f"oob={overlay_audit.vlm_keys_index_oob}", flush=True)

    print("[legacy-overlay] scoring VLM-refined stream with unmodified compute()", flush=True)
    result, table = run_compute(new_records)
    (out_dir / "metric_calculation_vlm_refinement_legacy.out").write_text(table)
    (out_dir / "legacy_overlay_result.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in result.items() if k != "detail"}, indent=2)
    )

    # Provenance.
    (out_dir / "legacy_overlay_provenance.json").write_text(json.dumps({
        "base_pred": str(args.base_pred),
        "base_out": str(args.base_out),
        "gtmeta": str(args.gtmeta),
        "vlm_pred": str(args.vlm_pred),
        "vlmgraph": str(args.vlmgraph),
        "sid_path": str(args.sid_path),
        "baseline": base_report.get("recomputed"),
        "vlm_refinement": {k: round(float(v), 4) for k, v in result.items()
                           if isinstance(v, (int, float))},
    }, indent=2))
    print("[legacy-overlay] done. Baseline vs VLM-refinement:", flush=True)
    for key in ("lah_ap", "laeo_ap", "coatt_ap", "social_ap"):
        b = base_report["recomputed"].get(key)
        v = result.get(key)
        if b is not None and v is not None:
            print(f"    {key}: {b:.4f} -> {float(v):.4f} ({float(v)-b:+.4f})", flush=True)


if __name__ == "__main__":
    main()
