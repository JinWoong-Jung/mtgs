import pytest
import torch

from vlm.pair_dataset import PairSample
from vlm.pair_features import (
    PERSON_CHANNEL,
    SLOT_NAMES,
    assemble_pair_graph_evidence,
    stack_pair_graph_evidence,
    PersonGazeText,
    TextGraphEvidence,
    assemble_text_graph_evidence,
)


def _vector(value, dim=4):
    return torch.full((dim,), float(value))


def _fake_cache(num_people=4, dim=4, height=3, width=5):
    v_src = torch.stack([_vector(100 + i, dim) for i in range(num_people)])
    v_tgt = torch.stack([_vector(200 + i, dim) for i in range(num_people + 2)])
    edge_pp = torch.empty(num_people, num_people, dim)
    for i in range(num_people):
        for j in range(num_people):
            edge_pp[i, j] = _vector(1000 + 10 * i + j, dim)
    edge_null_in = torch.stack([_vector(300 + i, dim) for i in range(num_people)])
    heatmaps = torch.stack([
        torch.full((height, width), float(400 + i)) for i in range(num_people)
    ])
    base = torch.arange(num_people * num_people, dtype=torch.float32).reshape(
        num_people, num_people
    )
    return {
        "v_src": v_src,
        "v_tgt": v_tgt,
        "edge_pp": edge_pp,
        "edge_null_in": edge_null_in,
        "gaze_heatmap": heatmaps,
        "lah_logits": base,
        "laeo_logits": base + 100,
        "sa_logits": base + 200,
        "vis_mask": torch.ones(num_people, dtype=torch.bool),
    }


def _sample(task, i, j, answer="yes"):
    return PairSample.from_manifest_record({
        "sid": "sample000000", "task": task, "i": i, "j": j, "ans": answer,
    })


def test_lah_uses_canonical_looker_to_target_and_na_masks():
    cache = _fake_cache()
    # Raw LAH (i=target=1, j=looker=3) -> canonical A=3, B=1.
    evidence = assemble_pair_graph_evidence(_sample("lah", 1, 3), cache)

    assert SLOT_NAMES == (
        "person_a", "person_b", "relation_ab", "relation_ba", "heatmap_a", "heatmap_b"
    )
    assert evidence.person_channel_present.tolist() == [
        [True, False, False], [False, True, False]
    ]
    assert torch.equal(evidence.person_features[0, PERSON_CHANNEL["src"]], cache["v_src"][3])
    assert torch.equal(evidence.person_features[1, PERSON_CHANNEL["tgt"]], cache["v_tgt"][1])
    assert torch.equal(evidence.relation_features[0], cache["edge_pp"][3, 1])
    assert evidence.relation_present.tolist() == [True, False]
    assert torch.count_nonzero(evidence.relation_features[1]) == 0
    assert torch.equal(evidence.heatmap_features[0], cache["gaze_heatmap"][3])
    assert evidence.heatmap_present.tolist() == [True, False]
    assert evidence.slot_presence.tolist() == [True, True, True, False, True, False]
    assert evidence.graph_logit.item() == cache["lah_logits"][3, 1].item()


def test_laeo_fills_src_and_tgt_for_both_people_and_both_directions():
    cache = _fake_cache()
    evidence = assemble_pair_graph_evidence(_sample("laeo", 1, 3), cache)

    assert evidence.person_channel_present.tolist() == [
        [True, True, False], [True, True, False]
    ]
    assert torch.equal(evidence.person_features[0, PERSON_CHANNEL["src"]], cache["v_src"][1])
    assert torch.equal(evidence.person_features[0, PERSON_CHANNEL["tgt"]], cache["v_tgt"][1])
    assert torch.equal(evidence.person_features[1, PERSON_CHANNEL["src"]], cache["v_src"][3])
    assert torch.equal(evidence.person_features[1, PERSON_CHANNEL["tgt"]], cache["v_tgt"][3])
    assert torch.equal(evidence.relation_features[0], cache["edge_pp"][1, 3])
    assert torch.equal(evidence.relation_features[1], cache["edge_pp"][3, 1])
    assert evidence.slot_presence.tolist() == [True] * 6
    expected = 0.5 * (cache["laeo_logits"][1, 3] + cache["laeo_logits"][3, 1])
    assert torch.equal(evidence.graph_logit, expected)


def test_sa_fills_src_null_in_pair_edges_and_both_heatmaps():
    cache = _fake_cache()
    evidence = assemble_pair_graph_evidence(_sample("sa", 0, 2), cache)

    assert evidence.person_channel_present.tolist() == [
        [True, False, True], [True, False, True]
    ]
    assert torch.equal(
        evidence.person_features[0, PERSON_CHANNEL["null_in"]], cache["edge_null_in"][0]
    )
    assert torch.equal(
        evidence.person_features[1, PERSON_CHANNEL["null_in"]], cache["edge_null_in"][2]
    )
    # SA deliberately retains both p2p directions in addition to Null_in evidence.
    assert torch.equal(evidence.relation_features[0], cache["edge_pp"][0, 2])
    assert torch.equal(evidence.relation_features[1], cache["edge_pp"][2, 0])
    assert torch.equal(evidence.heatmap_features[0], cache["gaze_heatmap"][0])
    assert torch.equal(evidence.heatmap_features[1], cache["gaze_heatmap"][2])
    assert evidence.slot_presence.tolist() == [True] * 6


def test_stack_preserves_fixed_shapes_and_task_masks():
    cache = _fake_cache()
    items = [
        assemble_pair_graph_evidence(_sample("lah", 1, 3), cache),
        assemble_pair_graph_evidence(_sample("laeo", 1, 3), cache),
        assemble_pair_graph_evidence(_sample("sa", 0, 2), cache),
    ]
    batch = stack_pair_graph_evidence(items)

    assert batch.tasks == ("lah", "laeo", "sa")
    assert batch.person_features.shape == (3, 2, 3, 4)
    assert batch.relation_features.shape == (3, 2, 4)
    assert batch.heatmap_features.shape == (3, 2, 3, 5)
    assert batch.graph_logits.shape == (3,)
    assert batch.slot_presence.tolist() == [
        [True, True, True, False, True, False],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
    ]


def test_non_visible_or_out_of_range_people_are_rejected():
    cache = _fake_cache()
    cache["vis_mask"][3] = False
    with pytest.raises(ValueError, match="non-visible"):
        assemble_pair_graph_evidence(_sample("laeo", 1, 3), cache)

    out_of_range = PairSample(
        sid="s", task="sa", person_i=0, person_j=8, label=1, raw_i=0, raw_j=8
    )
    with pytest.raises(IndexError, match="outside graph cache"):
        assemble_pair_graph_evidence(out_of_range, _fake_cache())


def test_cache_schema_mismatch_fails_early():
    cache = _fake_cache()
    cache["edge_pp"] = torch.zeros(4, 4, 7)
    with pytest.raises(ValueError, match="edge_pp axis 2"):
        assemble_pair_graph_evidence(_sample("sa", 0, 2), cache)


def test_half_cache_casts_only_fixed_outputs_to_float32():
    cache = _fake_cache()
    for name, value in list(cache.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            cache[name] = value.half()

    evidence = assemble_pair_graph_evidence(_sample("sa", 0, 2), cache)

    assert evidence.person_features.dtype == torch.float32
    assert evidence.relation_features.dtype == torch.float32
    assert evidence.heatmap_features.dtype == torch.float32
    assert evidence.graph_logit.dtype == torch.float32


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        stack_pair_graph_evidence([])


def _text_cache(n=4):
    torch.manual_seed(0)
    lah = torch.full((n, n), -3.0)
    lah[0, 1] = 2.0      # P(A->B) high
    lah[1, 0] = -1.0     # P(B->A) low-ish
    lah[0, 2] = 1.0      # A's best third person is 2
    lah[1, 3] = 0.5      # B's best third person is 3
    return {
        "lah_logits": lah,
        "null_in_logits": torch.tensor([0.0, 0.8, -0.5, 0.2]),  # sigmoid -> .5,.69,.38,.55
        "head_bboxes": torch.tensor([[0., 0., 1., 1.],
                                     [2., 2., 3., 3.],
                                     [4., 4., 5., 5.],
                                     [6., 6., 7., 7.]]),
        "vis_mask": torch.ones(n, dtype=torch.bool),
    }


def test_text_evidence_lah_is_directional_ab_only():
    s = PairSample(sid="x", task="lah", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, _text_cache())
    assert abs(ev.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    assert ev.p_ba is None and ev.person_a is None and ev.person_b is None


def test_text_evidence_laeo_has_both_directions():
    s = PairSample(sid="x", task="laeo", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, _text_cache())
    assert abs(ev.p_ab - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-5
    assert abs(ev.p_ba - torch.sigmoid(torch.tensor(-1.0)).item()) < 1e-5
    assert ev.person_a is None


def test_text_evidence_sa_picks_best_third_person_and_nonperson_prob():
    s = PairSample(sid="x", task="sa", person_i=0, person_j=1, label=1, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, _text_cache())
    # A's best third (k not in {0,1}) is person 2, bbox [4,4,5,5]
    assert ev.person_a.third_bbox == (4.0, 4.0, 5.0, 5.0)
    assert abs(ev.person_a.third_prob - torch.sigmoid(torch.tensor(1.0)).item()) < 1e-5
    assert abs(ev.person_a.nonperson_prob - torch.sigmoid(torch.tensor(0.0)).item()) < 1e-5
    # B's best third (k not in {0,1}) is person 3, bbox [6,6,7,7]
    assert ev.person_b.third_bbox == (6.0, 6.0, 7.0, 7.0)
    assert ev.p_ab is None


def test_text_evidence_sa_no_third_person_when_none_visible():
    cache = _text_cache()
    cache["vis_mask"] = torch.tensor([True, True, False, False])  # only A,B visible
    s = PairSample(sid="x", task="sa", person_i=0, person_j=1, label=0, raw_i=0, raw_j=1)
    ev = assemble_text_graph_evidence(s, cache)
    assert ev.person_a.third_bbox is None and ev.person_a.third_prob is None
    assert ev.person_a.nonperson_prob is not None
