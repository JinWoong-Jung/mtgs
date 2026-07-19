import itertools

import pytest
import torch

from vlm.cache.boundary import pair_boundary_flags


def _labels(n, per_frame):
    """Build a (T, num_pairs) tensor from per-frame {(i, j): value} dicts."""
    pairs = list(itertools.permutations(range(n), 2))
    t_steps = len(per_frame)
    out = torch.full((t_steps, len(pairs)), -1.0)
    for t, frame in enumerate(per_frame):
        for q, (i, j) in enumerate(pairs):
            if (i, j) in frame:
                out[t, q] = frame[(i, j)]
    return out


def test_stable_label_across_window_is_not_flagged():
    # 5-frame window, pair (0,1) is "yes" the whole time.
    labels = _labels(2, [{(0, 1): 1.0}] * 5)
    flags = pair_boundary_flags(labels, cidx=2, n=2)
    assert flags[0, 1].item() is False


def test_label_change_in_window_is_flagged():
    # Center says "yes" but an earlier frame says "no" -> in-flight transition.
    labels = _labels(2, [{(0, 1): 0.0}, {(0, 1): 0.0}, {(0, 1): 1.0}, {(0, 1): 1.0}, {(0, 1): 1.0}])
    flags = pair_boundary_flags(labels, cidx=2, n=2)
    assert flags[0, 1].item() is True


def test_invalid_neighbor_frames_do_not_trigger_a_flag():
    # Every non-center frame is -1 (e.g. near a clip boundary) -> unjudged, not flagged.
    labels = _labels(2, [{}, {}, {(0, 1): 1.0}, {}, {}])
    flags = pair_boundary_flags(labels, cidx=2, n=2)
    assert flags[0, 1].item() is False


def test_invalid_center_frame_is_never_flagged_regardless_of_neighbors():
    labels = _labels(2, [{(0, 1): 1.0}, {(0, 1): 1.0}, {}, {(0, 1): 0.0}, {(0, 1): 0.0}])
    flags = pair_boundary_flags(labels, cidx=2, n=2)
    assert flags[0, 1].item() is False


def test_flags_are_independent_per_pair():
    n = 3
    per_frame = [
        {(0, 1): 0.0, (1, 2): 1.0},
        {(0, 1): 0.0, (1, 2): 1.0},
        {(0, 1): 1.0, (1, 2): 1.0},  # center
        {(0, 1): 1.0, (1, 2): 1.0},
        {(0, 1): 1.0, (1, 2): 1.0},
    ]
    labels = _labels(n, per_frame)
    flags = pair_boundary_flags(labels, cidx=2, n=n)
    assert flags[0, 1].item() is True     # (0,1) transitions 0 -> 1
    assert flags[1, 2].item() is False    # (1,2) stable the whole window
    assert flags[2, 0].item() is False    # untouched pair stays False


def test_rejects_pair_count_mismatch():
    labels = torch.zeros((5, 3))  # 3 pairs cannot come from n=3 (needs 6)
    with pytest.raises(ValueError, match="expected 6 pairs"):
        pair_boundary_flags(labels, cidx=2, n=3)


def test_rejects_out_of_range_center_index():
    labels = _labels(2, [{(0, 1): 1.0}] * 5)
    with pytest.raises(ValueError, match="outside window"):
        pair_boundary_flags(labels, cidx=5, n=2)
