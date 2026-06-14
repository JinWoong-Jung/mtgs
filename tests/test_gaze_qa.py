# tests/test_gaze_qa.py
import importlib.util
import pathlib
import sys
import torch
import itertools
import pytest

# gaze_qa.py has no imports from mtgs.datasets (no circular dependency).
# We load it as a standalone module to avoid triggering the pre-existing circular
# import inside mtgs/datasets/__init__.py.
_root = pathlib.Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "mtgs.datasets.gaze_qa",
    _root / "mtgs" / "datasets" / "gaze_qa.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mtgs.datasets.gaze_qa"] = _mod
_spec.loader.exec_module(_mod)
GazeQACollator = _mod.GazeQACollator
QAPair = _mod.QAPair


def _make_batch(B=2, T=5, N=4):
    """Synthetic batch with the same key structure as pad_collate_fn output."""
    P = N * (N - 1)
    batch = {
        "lah_labels":   torch.randint(-1, 2, (B, T, P)).float(),
        "laeo_labels":  torch.full((B, T, P), -1.0),  # no LAEO annotation
        "coatt_labels": torch.randint(-1, 2, (B, T, P)).float(),
        "num_valid_people": torch.full((B, 1), N, dtype=torch.long),
        "head_bboxes":  torch.rand(B, T, N, 4),  # normalized [x1,y1,x2,y2]
    }
    # Force at least one valid label per batch item
    batch["lah_labels"][:, T // 2, 0] = 1.0
    batch["lah_labels"][:, T // 2, 1] = 0.0
    return batch


def test_returns_qa_pairs():
    collator = GazeQACollator()
    batch = _make_batch(B=2, N=4)
    pairs = collator(batch)
    assert len(pairs) > 0
    for p in pairs:
        assert isinstance(p, QAPair)
        assert p.task in ("lah", "laeo", "sa")
        assert p.label in (0, 1)
        assert p.batch_idx < 2
        assert len(p.src_bbox) == 4
        assert len(p.dst_bbox) == 4


def test_skips_minus_one_labels():
    collator = GazeQACollator()
    batch = _make_batch(B=1, N=4)
    batch["lah_labels"][:] = -1.0
    batch["coatt_labels"][:] = -1.0
    pairs = collator(batch)
    assert len(pairs) == 0


def test_laeo_skipped_when_all_minus_one():
    collator = GazeQACollator()
    batch = _make_batch(B=1, N=4)
    laeo_pairs = [p for p in collator(batch) if p.task == "laeo"]
    assert len(laeo_pairs) == 0


def test_all_annotated_pairs_included():
    """All pairs with label 0 or 1 must appear — no subsampling."""
    collator = GazeQACollator()
    B, T, N = 1, 5, 4
    P = N * (N - 1)
    batch = {
        "lah_labels":   torch.zeros(B, T, P),   # all 0 → all N*(N-1) pairs
        "laeo_labels":  torch.full((B, T, P), -1.0),
        "coatt_labels": torch.full((B, T, P), -1.0),
        "num_valid_people": torch.full((B, 1), N, dtype=torch.long),
        "head_bboxes":  torch.rand(B, T, N, 4),
    }
    batch["lah_labels"][0, T // 2, :3] = 1.0
    pairs = collator(batch)
    lah_pairs = [p for p in pairs if p.task == "lah"]
    assert len(lah_pairs) == P, f"Expected {P} LAH pairs, got {len(lah_pairs)}"


def test_laeo_sa_no_duplicate_pairs():
    """LAEO/SA must not contain both (i,j) and (j,i) for the same pair."""
    collator = GazeQACollator()
    B, T, N = 1, 5, 4
    P = N * (N - 1)
    batch = {
        "lah_labels":   torch.full((B, T, P), -1.0),
        "laeo_labels":  torch.zeros(B, T, P),
        "coatt_labels": torch.zeros(B, T, P),
        "num_valid_people": torch.full((B, 1), N, dtype=torch.long),
        "head_bboxes":  torch.rand(B, T, N, 4),
    }
    pairs = collator(batch)
    for task in ("laeo", "sa"):
        task_pairs = [(p.src_idx, p.dst_idx) for p in pairs if p.task == task]
        seen = set()
        for i, j in task_pairs:
            key = (min(i, j), max(i, j))
            assert key not in seen, f"{task}: duplicate pair {(i,j)}"
            seen.add(key)


def test_bbox_values_match_head_bboxes():
    """src_bbox/dst_bbox must match head_bboxes[b, t_c, idx]."""
    collator = GazeQACollator()
    batch = _make_batch(B=1, N=4)
    T = batch["lah_labels"].shape[1]
    t_c = T // 2
    pairs = collator(batch)
    for p in pairs:
        expected_src = tuple(batch["head_bboxes"][p.batch_idx, t_c, p.src_idx].tolist())
        expected_dst = tuple(batch["head_bboxes"][p.batch_idx, t_c, p.dst_idx].tolist())
        assert p.src_bbox == pytest.approx(expected_src, abs=1e-5)
        assert p.dst_bbox == pytest.approx(expected_dst, abs=1e-5)
