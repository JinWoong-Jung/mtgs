"""Unit tests for vlm.social.legacy_overlay.

Pins the LAH slot direction (the failure mode that silently collapses LAH AUC),
the symmetric double-write, the graph-fallback invariant (non-overlaid slots
never change), and signature determinism / order-sensitivity.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from vlm.social.legacy_overlay import (
    build_join,
    frame_signature,
    lah_native_slot,
    overlay_predictions,
    permutation_slot,
)


def test_permutation_slot_matches_itertools():
    for n in range(2, 8):
        pairs = list(itertools.permutations(range(n), 2))
        for q, (a, b) in enumerate(pairs):
            assert permutation_slot(a, b, n) == q


def test_lah_direction_codex_example():
    # N=3, raw_i=2 (target), raw_j=1 (looker): "Person 1 looks at Person 2".
    # Native pair (a,b) = "b looks at a" => slot must be pair (2,1) = index 5.
    n = 3
    q = lah_native_slot(target=2, looker=1, n=n)
    assert q == 5
    assert list(itertools.permutations(range(n), 2))[q] == (2, 1)
    # The reversed direction must be a DIFFERENT slot (index 3 = pair (1,2)).
    assert lah_native_slot(target=1, looker=2, n=n) == 3


def test_lah_overlay_writes_only_correct_slot():
    n = 3
    p = n * (n - 1)
    record = {
        "head_bboxes": torch.zeros(1, n, 4),
        "lah_pred": torch.zeros(1, p),
        "laeo_pred": torch.zeros(1, p),
        "coatt_pred": torch.zeros(1, p),
        "lah_gt": torch.full((1, p), -1, dtype=torch.long),
        "laeo_gt": torch.full((1, p), -1, dtype=torch.long),
        "coatt_gt": torch.full((1, p), -1, dtype=torch.long),
        "dataset": ["childplay"],
    }
    record["lah_gt"][0, 5] = 1  # mark the queried pair positive
    sid_to_native = {"s0": 0}
    vlm = {("s0", "lah", 2, 1): 0.9}  # target=2, looker=1
    new, audit = overlay_predictions([record], sid_to_native, vlm)

    assert float(new[0]["lah_pred"][0, 5]) == pytest.approx(0.9, abs=1e-6)  # correct slot set
    assert float(new[0]["lah_pred"][0, 3]) == 0.0  # reversed slot untouched
    # original stream not mutated
    assert float(record["lah_pred"][0, 5]) == 0.0
    assert audit.vlm_keys_applied == 1
    assert audit.overlay_gt["lah"]["1"] == 1


def test_symmetric_overlay_writes_both_slots():
    n = 3
    p = n * (n - 1)
    record = {
        "head_bboxes": torch.zeros(1, n, 4),
        "lah_pred": torch.zeros(1, p),
        "laeo_pred": torch.zeros(1, p),
        "coatt_pred": torch.zeros(1, p),
        "lah_gt": torch.full((1, p), -1, dtype=torch.long),
        "laeo_gt": torch.full((1, p), -1, dtype=torch.long),
        "coatt_gt": torch.full((1, p), -1, dtype=torch.long),
        "dataset": ["childplay"],
    }
    sid_to_native = {"s0": 0}
    q_ij = permutation_slot(0, 1, n)
    q_ji = permutation_slot(1, 0, n)
    vlm = {("s0", "laeo", 0, 1): 0.7}
    new, _ = overlay_predictions([record], sid_to_native, vlm)
    assert float(new[0]["laeo_pred"][0, q_ij]) == pytest.approx(0.7, abs=1e-6)
    assert float(new[0]["laeo_pred"][0, q_ji]) == pytest.approx(0.7, abs=1e-6)
    # other tasks untouched
    assert float(new[0]["coatt_pred"][0, q_ij]) == 0.0


def test_non_overlaid_slots_are_graph_fallback():
    n = 4
    p = n * (n - 1)
    base = torch.arange(p, dtype=torch.float32).unsqueeze(0) / 100.0
    record = {
        "head_bboxes": torch.zeros(1, n, 4),
        "lah_pred": base.clone(),
        "laeo_pred": base.clone(),
        "coatt_pred": base.clone(),
        "lah_gt": torch.full((1, p), -1, dtype=torch.long),
        "laeo_gt": torch.full((1, p), -1, dtype=torch.long),
        "coatt_gt": torch.full((1, p), -1, dtype=torch.long),
        "dataset": ["vat"],
    }
    sid_to_native = {"s0": 0}
    q = lah_native_slot(target=1, looker=2, n=n)
    vlm = {("s0", "lah", 1, 2): 0.123}
    new, _ = overlay_predictions([record], sid_to_native, vlm)
    for slot in range(p):
        if slot == q:
            assert float(new[0]["lah_pred"][0, slot]) == pytest.approx(0.123, abs=1e-6)
        else:
            assert float(new[0]["lah_pred"][0, slot]) == float(base[0, slot])


def test_signature_deterministic_and_order_sensitive():
    bboxes = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lah = torch.tensor([1, 0])
    laeo = torch.tensor([0, 0])
    coatt = torch.tensor([1, 1])
    s1 = frame_signature("childplay", bboxes, lah, laeo, coatt)
    s2 = frame_signature("childplay", bboxes.clone(), lah.clone(), laeo.clone(), coatt.clone())
    assert s1 == s2  # deterministic
    # swapping people order changes the signature (order-sensitive)
    s3 = frame_signature("childplay", bboxes.flip(0), lah, laeo, coatt)
    assert s1 != s3
    # different dataset changes it
    assert frame_signature("vat", bboxes, lah, laeo, coatt) != s1


def test_build_join_by_path_crosses_shuffled_order():
    # native stream is shuffled relative to sid order; join must follow center path.
    def rec(path, lah):
        p = len(lah)
        return {
            "path": [[path]],  # temporal window wrapped like the real collate
            "head_bboxes": torch.zeros(1, 3, 4),
            "lah_gt": torch.tensor(lah).unsqueeze(0),
            "laeo_gt": torch.full((1, p), -1, dtype=torch.long),
            "coatt_gt": torch.full((1, p), -1, dtype=torch.long),
        }

    native = [rec("clip/b.jpg", [0, 1, -1, 0, 1, 0]), rec("clip/a.jpg", [1, 0, 0, -1, 0, 1])]
    sid_to_path = {"sample000000": "clip/a.jpg", "sample000001": "clip/b.jpg"}
    gtmeta = {
        "sample000000": {"lah_gt": torch.tensor([1, 0, 0, -1, 0, 1]),
                         "laeo_gt": torch.full((6,), -1), "coatt_gt": torch.full((6,), -1)},
        "sample000001": {"lah_gt": torch.tensor([0, 1, -1, 0, 1, 0]),
                         "laeo_gt": torch.full((6,), -1), "coatt_gt": torch.full((6,), -1)},
    }
    sid_to_native, audit = build_join(native, sid_to_path, gtmeta=gtmeta)
    assert sid_to_native == {"sample000000": 1, "sample000001": 0}  # crosses shuffle
    assert audit.matched == 2
    assert audit.duplicate_native_paths == 0
    assert audit.gt_mismatch == 0
    assert len(audit.unmatched_sids) == 0


def test_build_join_flags_gt_mismatch():
    def rec(path, lah):
        p = len(lah)
        return {
            "path": [[path]],
            "head_bboxes": torch.zeros(1, 3, 4),
            "lah_gt": torch.tensor(lah).unsqueeze(0),
            "laeo_gt": torch.full((1, p), -1, dtype=torch.long),
            "coatt_gt": torch.full((1, p), -1, dtype=torch.long),
        }

    native = [rec("clip/a.jpg", [1, 0, 0, -1, 0, 1])]
    sid_to_path = {"sample000000": "clip/a.jpg"}
    gtmeta = {"sample000000": {"lah_gt": torch.tensor([0, 0, 0, 0, 0, 0]),  # disagrees
                               "laeo_gt": torch.full((6,), -1), "coatt_gt": torch.full((6,), -1)}}
    _, audit = build_join(native, sid_to_path, gtmeta=gtmeta)
    assert audit.gt_mismatch == 1
