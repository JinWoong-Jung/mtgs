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
        "null_in_logits": torch.tensor([0.0, 0.8, -0.5, 0.2]),  # sigmoid -> .5,.69,.38,.55
        "head_bboxes": torch.tensor([[0., 0., 1., 1.],
                                     [2., 2., 3., 3.],
                                     [4., 4., 5., 5.],
                                     [6., 6., 7., 7.]]),
        # A's point (1.5,1.5) and B's (3.5,3.5) both fall outside every head bbox.
        "gaze_point": torch.tensor([[1.5, 1.5], [3.5, 3.5], [0.5, 0.5], [6.5, 6.5]]),
        "vis_mask": torch.ones(n, dtype=torch.bool),
    }


def test_text_evidence_lah_is_directional_ab_only():
    s = SocialSample(sid="x", task="lah", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, _text_cache())
    assert isinstance(ev, TextGraphEvidence)
    assert abs(ev.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    assert ev.p_ba is None and ev.person_a is None and ev.person_b is None


def test_text_evidence_laeo_has_both_directions():
    s = SocialSample(sid="x", task="laeo", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, _text_cache())
    assert abs(ev.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    assert abs(ev.p_ba - torch.sigmoid(torch.tensor(-1.0)).item()) < 1e-5
    assert abs(ev.task_prob - torch.sigmoid(torch.tensor(0.6)).item()) < 1e-5
    assert ev.person_a is None


def test_text_evidence_sa_gives_person_target_gaze_point_and_nonperson_prob():
    s = SocialSample(sid="x", task="sa", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, _text_cache())
    # A's best third (k not in {0,1}) is person 2, bbox [4,4,5,5], logit 1.0.
    assert isinstance(ev.person_a, PersonGazeText)
    assert ev.person_a.third_bbox == (4.0, 4.0, 5.0, 5.0)
    assert ev.person_a.third_person_index == 2
    assert abs(ev.person_a.third_prob - torch.sigmoid(torch.tensor(1.0)).item()) < 1e-5
    # Gaze point + P(non-person) surfaced unconditionally (no gate).
    assert ev.person_a.gaze_xy == (1.5, 1.5)
    assert abs(ev.person_a.nonperson_prob - torch.sigmoid(torch.tensor(0.0)).item()) < 1e-5
    # B: person 3 target, its own gaze point and null_in probability.
    assert ev.person_b.third_person_index == 3
    assert ev.person_b.gaze_xy == (3.5, 3.5)
    assert abs(ev.person_b.nonperson_prob - torch.sigmoid(torch.tensor(0.8)).item()) < 1e-5
    assert abs(ev.task_prob - 0.5) < 1e-5
    assert ev.p_ab is None


def test_text_evidence_sa_no_third_person_still_gives_gaze_point_and_prob():
    cache = _text_cache()
    cache["vis_mask"] = torch.tensor([True, True, False, False])  # only A,B visible
    s = SocialSample(sid="x", task="sa", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, cache)
    assert ev.person_a.third_bbox is None and ev.person_a.third_prob is None
    assert ev.person_a.gaze_xy == (1.5, 1.5)
    assert ev.person_a.nonperson_prob is not None
