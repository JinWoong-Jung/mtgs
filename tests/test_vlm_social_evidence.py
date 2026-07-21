import torch
import pytest

from vlm.social.data import SocialSample
from vlm.social.evidence import (
    PersonGazeText,
    TextGraphEvidence,
    _direction_bin,
    assemble_text_graph_evidence,
)


def _text_cache(n=4):
    lah = torch.full((n, n), -3.0)
    lah[0, 1] = 2.0
    lah[1, 0] = -1.0
    lah[0, 2] = 1.0
    lah[1, 3] = 0.5
    laeo = torch.zeros((n, n))
    laeo[0, 1], laeo[1, 0] = 0.4, 0.8
    sa = torch.zeros((n, n))
    sa[0, 1], sa[1, 0] = -0.2, 0.2
    return {
        "lah_logits": lah,
        "laeo_logits": laeo,
        "sa_logits": sa,
        "null_in_logits": torch.tensor([0.0, 0.8, -0.5, 0.2]),
        "head_bboxes": torch.tensor([[0., 0., 1., 1.],
                                     [2., 2., 3., 3.],
                                     [4., 4., 5., 5.],
                                     [6., 6., 7., 7.]]),
        "gaze_point": torch.tensor([[1.5, 1.5], [3.5, 3.5], [0.5, 0.5], [6.5, 6.5]]),
        # (dx, dy) in normalized, y-down image coordinates: person 0 -> right,
        # person 1 -> up (dy<0), person 2 -> left, person 3 -> down (dy>0).
        "gaze_vecs": torch.tensor([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]]),
        "vis_mask": torch.ones(n, dtype=torch.bool),
    }


def test_direction_bin_covers_all_eight_compass_labels():
    # dx, dy in normalized y-down coordinates; expected label is the visual direction
    # (dy negated internally so "up" means visually up, i.e. dy<0).
    cases = [
        ((1.0, 0.0), "right"),
        ((1.0, -1.0), "upper-right"),
        ((0.0, -1.0), "up"),
        ((-1.0, -1.0), "upper-left"),
        ((-1.0, 0.0), "left"),
        ((-1.0, 1.0), "lower-left"),
        ((0.0, 1.0), "down"),
        ((1.0, 1.0), "lower-right"),
    ]
    for (dx, dy), expected in cases:
        assert _direction_bin(dx, dy) == expected, (dx, dy, expected)


def test_direction_bin_boundary_rounds_to_nearest_bin():
    # 22.5 degrees (visual, dy negated) is exactly the midpoint between right(0) and
    # upper-right(45); round-half-to-even/away ties should land on one deterministic side.
    dx, dy = 1.0, -torch.tan(torch.tensor(22.4 * 3.14159265 / 180)).item()
    assert _direction_bin(dx, dy) == "right"
    dx, dy = 1.0, -torch.tan(torch.tensor(22.6 * 3.14159265 / 180)).item()
    assert _direction_bin(dx, dy) == "upper-right"


def test_text_evidence_lah_is_directional_and_includes_lookers_gaze_point():
    sample = SocialSample(
        sid="x", task="lah", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, _text_cache())
    assert isinstance(evidence, TextGraphEvidence)
    assert abs(evidence.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    assert evidence.gaze_a_xy == (1.5, 1.5)
    assert evidence.gaze_a_dir == "right"
    assert evidence.gaze_b_dir is None  # LAH surfaces only the looker's direction
    assert evidence.p_ba is None and evidence.person_a is None and evidence.person_b is None
    assert not hasattr(evidence, "temporal_probs")


def test_text_evidence_laeo_has_both_directions_and_both_gaze_points():
    sample = SocialSample(
        sid="x", task="laeo", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, _text_cache())
    assert abs(evidence.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    assert abs(evidence.p_ba - torch.sigmoid(torch.tensor(-1.0)).item()) < 1e-5
    assert abs(evidence.task_prob - torch.sigmoid(torch.tensor(0.6)).item()) < 1e-5
    assert evidence.gaze_a_xy == (1.5, 1.5)
    assert evidence.gaze_b_xy == (3.5, 3.5)
    assert evidence.gaze_a_dir == "right"
    assert evidence.gaze_b_dir == "up"
    assert evidence.person_a is None


def test_text_evidence_sa_includes_person_targets_nonperson_prob_and_gaze_points():
    sample = SocialSample(
        sid="x", task="sa", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, _text_cache())
    assert isinstance(evidence.person_a, PersonGazeText)
    assert evidence.person_a.third_bbox == (4.0, 4.0, 5.0, 5.0)
    assert evidence.person_a.third_person_index == 2
    assert abs(evidence.person_a.third_prob - torch.sigmoid(torch.tensor(1.0)).item()) < 1e-5
    assert evidence.person_a.gaze_xy == (1.5, 1.5)
    assert evidence.person_a.gaze_dir == "right"
    assert abs(evidence.person_a.nonperson_prob - 0.5) < 1e-5
    assert evidence.person_b.third_person_index == 3
    assert evidence.person_b.gaze_xy == (3.5, 3.5)
    assert evidence.person_b.gaze_dir == "up"
    assert abs(evidence.person_b.nonperson_prob - torch.sigmoid(torch.tensor(0.8)).item()) < 1e-5
    assert abs(evidence.task_prob - 0.5) < 1e-5
    assert evidence.p_ab is None


def test_text_evidence_sa_without_third_person_still_gives_gaze_point_and_nonperson_prob():
    cache = _text_cache()
    cache["vis_mask"] = torch.tensor([True, True, False, False])
    sample = SocialSample(
        sid="x", task="sa", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, cache)
    assert evidence.person_a.third_bbox is None and evidence.person_a.third_prob is None
    assert evidence.person_a.gaze_xy == (1.5, 1.5)
    assert evidence.person_a.nonperson_prob is not None


def test_temporal_cache_fields_are_ignored():
    cache = _text_cache()
    cache["lah_logits_frames"] = torch.full((5, 4, 4), float("nan"))
    sample = SocialSample(
        sid="x", task="lah", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, cache)
    assert evidence.gaze_a_xy == (1.5, 1.5)

@pytest.mark.parametrize(
    ("bad_vector", "message"),
    [
        (torch.tensor([float("nan"), 0.0]), "finite"),
        (torch.tensor([0.0, 0.0]), "L2 norm"),
    ],
)
def test_text_evidence_rejects_non_directional_gaze_vectors(bad_vector, message):
    cache = _text_cache()
    cache["gaze_vecs"][0] = bad_vector
    sample = SocialSample(
        sid="x", task="lah", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1
    )
    with pytest.raises(ValueError, match=message):
        assemble_text_graph_evidence(sample, cache)
