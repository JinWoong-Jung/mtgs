import json

import pytest
import torch
from PIL import Image

from vlm.overlay import build_canonical_pair_overlay
from vlm.pair_features import TextGraphEvidence
from vlm.pair_input import (
    GraphControlDataset,
    GraphFeatureControlDataset,
    PairInputDataset,
    RawFrameCache,
    pair_control_collate,
    pair_feature_control_collate,
    pair_pos_weights,
    pair_sample_weights,
    partition_by_graph_confidence,
)
from vlm.pair_prompt import TASK_DEFINITIONS, task_conditioned_pair_prompt


def _fake_graph_cache():
    num_people, dim = 2, 4
    edge_pp = torch.zeros(num_people, num_people, dim)
    edge_pp[0, 1] = 1
    edge_pp[1, 0] = 2
    logits = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    return {
        "v_src": torch.stack((torch.full((dim,), 10.0), torch.full((dim,), 11.0))),
        "v_tgt": torch.stack([
            torch.full((dim,), 20.0 + index) for index in range(num_people + 2)
        ]),
        "edge_pp": edge_pp,
        "edge_null_in": torch.stack((torch.full((dim,), 30.0), torch.full((dim,), 31.0))),
        "gaze_heatmap": torch.stack((torch.full((8, 8), 40.0), torch.full((8, 8), 41.0))),
        "lah_logits": logits,
        "laeo_logits": logits + 10,
        "sa_logits": logits + 20,
        "head_bboxes": torch.tensor([
            [0.10, 0.10, 0.35, 0.35],
            [0.60, 0.60, 0.85, 0.85],
        ]),
        "vis_mask": torch.ones(num_people, dtype=torch.bool),
    }


def _write_frame(root, sid, color="black"):
    path = root / sid
    path.mkdir(parents=True)
    Image.new("RGB", (100, 100), color).save(path / "frame.png")


def _write_manifest(path):
    records = [
        # Raw LAH: person 1 looks at person 0 -> canonical A=1(red), B=0(blue).
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "no"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _make_generative_dataset(tmp_path, *, graph_evidence="gtoken", draw_bboxes=True):
    frame_root = tmp_path / "frames"
    _write_frame(frame_root, "s0")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    graph = {"s0": _fake_graph_cache()}
    return PairInputDataset(
        manifest,
        frame_root,
        graph,
        output_mode="generative",
        graph_evidence=graph_evidence,
        draw_bboxes=draw_bboxes,
    )


def test_text_mode_builds_text_prompt_and_text_evidence(tmp_path):
    ds = _make_generative_dataset(tmp_path, graph_evidence="text", draw_bboxes=True)
    item = ds[0]
    assert isinstance(item.evidence, TextGraphEvidence)
    assert "Person A" in item.prompt and "Person B" in item.prompt
    assert item.prompt.rstrip().endswith('Answer with a single word, "yes" or "no".')
    assert item.draw_bboxes is True     # overlay enabled in text mode


def test_text_mode_default_config_uses_plain_unmarked_images(tmp_path):
    from vlm.pair_prompt import TEXT_MARKED_IDENTITY

    ds = _make_generative_dataset(tmp_path, graph_evidence="text", draw_bboxes=False)
    item = ds[0]

    assert item.draw_bboxes is False
    assert TEXT_MARKED_IDENTITY not in item.prompt
    assert item.image.getpixel((70, 85)) == (0, 0, 0)


def test_canonical_overlay_does_not_mutate_raw_image():
    raw = Image.new("RGB", (100, 100), "black")
    overlay = build_canonical_pair_overlay(
        raw, [0.60, 0.60, 0.85, 0.85], [0.10, 0.10, 0.35, 0.35]
    )
    assert raw.getpixel((70, 85)) == (0, 0, 0)
    assert overlay.getpixel((70, 85)) == (255, 0, 0)
    assert overlay.getpixel((35, 30)) == (0, 0, 255)


def test_pair_dataset_uses_task_text_canonical_boxes_and_raw_lru(tmp_path):
    frame_root = tmp_path / "frames"
    _write_frame(frame_root, "s0")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    graph = {"s0": _fake_graph_cache()}
    dataset = PairInputDataset(manifest, frame_root, graph, raw_image_cache_size=2)

    lah = dataset[0]
    sa = dataset[1]

    assert (lah.annotation.person_i, lah.annotation.person_j) == (1, 0)
    assert lah.prompt == task_conditioned_pair_prompt("lah")
    assert sa.prompt == task_conditioned_pair_prompt("sa")
    assert lah.prompt != sa.prompt
    assert TASK_DEFINITIONS["lah"] in lah.prompt
    assert TASK_DEFINITIONS["sa"] in sa.prompt
    assert lah.image.getpixel((70, 85)) == (255, 0, 0)  # LAH looker 1 = A/red
    assert lah.image.getpixel((35, 30)) == (0, 0, 255)  # LAH target 0 = B/blue
    assert lah.evidence.task == "lah" and sa.evidence.task == "sa"
    assert lah.image is not sa.image
    assert dataset.frames.cache_info() == (1, 1, 2, 1)
    assert dataset.raw_frame_path(0) == frame_root / "s0" / "frame.png"
    with Image.open(dataset.raw_frame_path(0)) as raw:
        assert raw.convert("RGB").getpixel((70, 85)) == (0, 0, 0)


def test_unmarked_pair_dataset_reuses_the_raw_frame_object(tmp_path):
    frame_root = tmp_path / "frames"
    _write_frame(frame_root, "s0")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    dataset = PairInputDataset(
        manifest,
        frame_root,
        {"s0": _fake_graph_cache()},
        raw_image_cache_size=2,
        draw_bboxes=False,
    )
    lah = dataset[0]
    sa = dataset[1]
    assert not lah.draw_bboxes and not sa.draw_bboxes
    assert lah.image is sa.image
    assert lah.image.getpixel((70, 85)) == (0, 0, 0)
    assert "image is unmodified" in lah.prompt
    assert lah.vision_cache_key == sa.vision_cache_key


def test_raw_frame_lru_is_bounded_and_can_be_disabled(tmp_path):
    _write_frame(tmp_path, "s0", "red")
    _write_frame(tmp_path, "s1", "blue")
    cache = RawFrameCache(tmp_path, max_items=1)
    cache.get("s0")
    cache.get("s1")
    cache.get("s1")
    assert cache.cache_info() == (1, 2, 1, 1)
    cache.get("s0")  # s0 was evicted by s1
    assert cache.cache_info() == (1, 3, 1, 1)

    disabled = RawFrameCache(tmp_path, max_items=0)
    disabled.get("s0")
    disabled.get("s0")
    assert disabled.cache_info() == (0, 2, 0, 0)


def test_missing_graph_frame_is_rejected_at_construction(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    with pytest.raises(ValueError, match="missing 1 manifest frames"):
        PairInputDataset(manifest, tmp_path, {})


def test_missing_raw_frame_has_contextual_error(tmp_path):
    cache = RawFrameCache(tmp_path)
    with pytest.raises(FileNotFoundError, match="s404"):
        cache.get("s404")


def test_graph_control_reuses_rows_without_loading_images_and_collates(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    dataset = GraphControlDataset(manifest, {"s0": _fake_graph_cache()})
    batch = pair_control_collate([dataset[0], dataset[1]])
    assert batch["task_ids"].tolist() == [0, 2]
    assert batch["pair_labels"].tolist() == [1.0, 0.0]
    # Raw LAH is reversed once: canonical graph logit is [looker=1,target=0].
    assert batch["graph_logits"][0].item() == 2.0
    assert batch["graph_logits"][1].item() == 21.5


def test_graph_feature_control_uses_six_slot_evidence_without_images(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    dataset = GraphFeatureControlDataset(manifest, {"s0": _fake_graph_cache()})
    batch = pair_feature_control_collate([dataset[0], dataset[1]])
    graph = batch["pair_graph"]

    assert dataset.feature_dim == 4
    assert batch["task_ids"].tolist() == [0, 2]
    assert graph.tasks == ("lah", "sa")
    # LAH raw indices are transposed once by PairAnnotationDataset: A=1 -> B=0.
    assert graph.graph_logits.tolist() == [2.0, 21.5]
    assert graph.person_channel_present[0].tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    assert graph.relation_present[0].tolist() == [True, False]
    assert graph.heatmap_present[0].tolist() == [True, False]
    assert graph.person_channel_present[1].tolist() == [
        [True, False, True],
        [True, False, True],
    ]
    # Constructing and indexing the dataset never needs a frame_root.
    assert not hasattr(dataset, "frames")


def test_sampler_balance_hardness_and_pos_weights(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "lah", "i": 1, "j": 0, "ans": "no"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "laeo", "i": 1, "j": 0, "ans": "no"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "sa", "i": 1, "j": 0, "ans": "no"},
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    dataset = GraphControlDataset(manifest, {"s0": _fake_graph_cache()})
    balanced = dataset.sample_weights(balance_mode="task_label")
    torch.testing.assert_close(balanced, torch.ones(6, dtype=torch.double))
    hard = dataset.sample_weights(balance_mode="task_label", hard_floor=0.25)
    assert torch.all(hard >= 0.25)
    assert torch.all(hard <= 1.25)
    assert pair_pos_weights(dataset.annotations) == {
        "lah": 1.0, "laeo": 1.0, "sa": 1.0,
    }


def test_pair_sample_weights_task_label_balance_unaffected_by_route_threshold_none(
    tmp_path,
):
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "lah", "i": 1, "j": 0, "ans": "no"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "laeo", "i": 1, "j": 0, "ans": "no"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "sa", "i": 1, "j": 0, "ans": "no"},
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    dataset = GraphControlDataset(manifest, {"s0": _fake_graph_cache()})

    default_weights = pair_sample_weights(
        dataset.annotations,
        dataset.graph_cache,
        balance_mode="task_label",
    )
    explicit_none_weights = pair_sample_weights(
        dataset.annotations,
        dataset.graph_cache,
        balance_mode="task_label",
        route_threshold=None,
    )
    torch.testing.assert_close(default_weights, explicit_none_weights)


def test_confidence_routing_partitions_and_zeroes_high_confidence_weights(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    # lah(0,1) logit=1.0 -> conf 0.73 (low). sa/laeo symmetrize to logit>=11.5 -> conf ~1 (high).
    records = [
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "no"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "yes"},
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    dataset = GraphControlDataset(manifest, {"s0": _fake_graph_cache()})

    high, low = partition_by_graph_confidence(dataset.annotations, dataset.graph_cache, 0.9)
    assert low == [0]                       # only the low-confidence lah pair reaches the VLM
    assert high == [1, 2]

    weights = pair_sample_weights(
        dataset.annotations, dataset.graph_cache,
        balance_mode="task", route_threshold=0.9,
    )
    assert weights[0] > 0
    assert weights[1] == 0 and weights[2] == 0

    with pytest.raises(ValueError, match="no low-confidence training pairs"):
        pair_sample_weights(
            dataset.annotations, dataset.graph_cache,
            balance_mode="task", route_threshold=0.5,   # everything is high-confidence
        )
    with pytest.raises(ValueError, match="threshold must be in"):
        partition_by_graph_confidence(dataset.annotations, dataset.graph_cache, 1.5)


def test_confidence_routing_sends_exact_threshold_to_vlm(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{\"sid\":\"s0\",\"task\":\"lah\",\"i\":0,\"j\":1,\"ans\":\"yes\"}\n")
    cache = _fake_graph_cache()
    dataset = GraphControlDataset(manifest, {"s0": cache})
    cache["lah_logits"].zero_()  # conf = 0.5 exactly, so this pair goes to the VLM.

    high, low = partition_by_graph_confidence(
        dataset.annotations, dataset.graph_cache, 0.5
    )
    assert high == []
    assert low == [0]
