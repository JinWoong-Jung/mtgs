import pytest
import torch

from vlm.social.data import SocialSample
from vlm.social.evidence import (
    PersonGazeText,
    TextGraphEvidence,
    assemble_text_graph_evidence,
)


def _text_cache(n=4):
    torch.manual_seed(0)
    lah = torch.full((n, n), -3.0)
    lah[0, 1] = 2.0      # P(A->B) high
    lah[1, 0] = -1.0     # P(B->A) low-ish
    lah[0, 2] = 1.0      # A's best third person is 2
    lah[1, 3] = 0.5      # B's best third person is 3
    laeo = torch.zeros((n, n))
    laeo[0, 1], laeo[1, 0] = 0.4, 0.8
    sa = torch.zeros((n, n))
    sa[0, 1], sa[1, 0] = -0.2, 0.2
    return {
        "lah_logits": lah,
        "laeo_logits": laeo,
        "sa_logits": sa,
        # Five cached positions. The prompt must select the center-adjacent three,
        # not the deliberately distant outer two.
        "lah_logits_frames": torch.stack((lah - 4, lah - 1, lah, lah + 1, lah + 4)),
        "laeo_logits_frames": torch.stack((
            laeo - 4, laeo - 0.4, laeo, laeo + 0.4, laeo + 4,
        )),
        "sa_logits_frames": torch.stack((sa - 4, sa - 0.5, sa, sa + 0.5, sa + 4)),
        "null_in_logits": torch.tensor([0.0, 0.8, -0.5, 0.2]),
        "head_bboxes": torch.tensor([[0., 0., 1., 1.],
                                     [2., 2., 3., 3.],
                                     [4., 4., 5., 5.],
                                     [6., 6., 7., 7.]]),
        # No gaze_point field: prompt evidence must no longer depend on coordinates.
        "vis_mask": torch.ones(n, dtype=torch.bool),
    }


def _assert_probs(actual, expected):
    torch.testing.assert_close(torch.tensor(actual), torch.sigmoid(torch.tensor(expected)))


def test_text_evidence_lah_is_directional_ab_only_and_temporal():
    sample = SocialSample(
        sid="x", task="lah", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, _text_cache())
    assert isinstance(evidence, TextGraphEvidence)
    assert abs(evidence.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    _assert_probs(evidence.temporal_probs, [1.0, 2.0, 3.0])
    assert evidence.p_ba is None and evidence.person_a is None and evidence.person_b is None
    assert evidence.gaze_a_xy is None


def test_text_evidence_laeo_has_both_directions_and_symmetric_temporal_decoder():
    sample = SocialSample(
        sid="x", task="laeo", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, _text_cache())
    assert abs(evidence.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    assert abs(evidence.p_ba - torch.sigmoid(torch.tensor(-1.0)).item()) < 1e-5
    assert abs(evidence.task_prob - torch.sigmoid(torch.tensor(0.6)).item()) < 1e-5
    _assert_probs(evidence.temporal_probs, [0.2, 0.6, 1.0])
    assert evidence.person_a is None
    assert evidence.gaze_a_xy is None and evidence.gaze_b_xy is None


def test_text_evidence_sa_gives_person_target_nonperson_prob_and_temporal_decoder():
    sample = SocialSample(
        sid="x", task="sa", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, _text_cache())
    # A's best third (k not in {0,1}) is person 2, bbox [4,4,5,5], logit 1.0.
    assert isinstance(evidence.person_a, PersonGazeText)
    assert evidence.person_a.third_bbox == (4.0, 4.0, 5.0, 5.0)
    assert evidence.person_a.third_person_index == 2
    assert abs(
        evidence.person_a.third_prob - torch.sigmoid(torch.tensor(1.0)).item()
    ) < 1e-5
    assert evidence.person_a.gaze_xy is None
    assert abs(
        evidence.person_a.nonperson_prob - torch.sigmoid(torch.tensor(0.0)).item()
    ) < 1e-5
    assert evidence.person_b.third_person_index == 3
    assert evidence.person_b.gaze_xy is None
    assert abs(
        evidence.person_b.nonperson_prob - torch.sigmoid(torch.tensor(0.8)).item()
    ) < 1e-5
    assert abs(evidence.task_prob - 0.5) < 1e-5
    _assert_probs(evidence.temporal_probs, [-0.5, 0.0, 0.5])
    assert evidence.p_ab is None


def test_text_evidence_sa_no_third_person_still_gives_nonperson_prob():
    cache = _text_cache()
    cache["vis_mask"] = torch.tensor([True, True, False, False])
    sample = SocialSample(
        sid="x", task="sa", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1
    )
    evidence = assemble_text_graph_evidence(sample, cache)
    assert evidence.person_a.third_bbox is None and evidence.person_a.third_prob is None
    assert evidence.person_a.gaze_xy is None
    assert evidence.person_a.nonperson_prob is not None


def test_temporal_center_must_match_exported_center_probability():
    cache = _text_cache()
    cache["lah_logits_frames"][2, 0, 1] += 1.0
    sample = SocialSample(
        sid="x", task="lah", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1
    )
    with pytest.raises(ValueError, match="does not match"):
        assemble_text_graph_evidence(sample, cache)
