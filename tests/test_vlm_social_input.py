import json

import pytest
import torch
from PIL import Image

from vlm.social.data import SocialAnnotationDataset
from vlm.social.input import (
    SocialInputDataset,
    RawFrameCache,
    boost_low_confidence_weights,
    task_pos_weights,
    sample_weights,
    partition_by_graph_confidence,
)
from vlm.social.graph_tokens import extract_graph_token_payload


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
        "lah_logits_frames": torch.stack((logits - 1, logits, logits + 1)),
        "laeo_logits_frames": torch.stack((
            logits + 9, logits + 10, logits + 11,
        )),
        "sa_logits_frames": torch.stack((logits + 19, logits + 20, logits + 21)),
        "null_in_logits": torch.tensor([0.0, 0.5]),
        "head_bboxes": torch.tensor([
            [0.10, 0.10, 0.35, 0.35],
            [0.60, 0.60, 0.85, 0.85],
        ]),
        "gaze_point": torch.tensor([[0.50, 0.50], [0.45, 0.45]]),
        "vis_mask": torch.ones(num_people, dtype=torch.bool),
    }


def _write_frame(root, sid, color="black"):
    path = root / sid
    path.mkdir(parents=True)
    Image.new("RGB", (100, 100), color).save(path / "frame.png")


def _write_manifest(path):
    records = [
        # Raw LAH: person 1 looks at person 0 -> canonical A=1, B=0.
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "no"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _make_dataset(
    tmp_path,
    *,
    include_graph_evidence=True,
    routing_threshold=None,
    graph_evidence_mode="text",
    graph_token_features=None,
    draw_pair_bboxes=False,
):
    frame_root = tmp_path / "frames"
    _write_frame(frame_root, "s0")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    graph = {"s0": _fake_graph_cache()}
    return SocialInputDataset(
        manifest,
        frame_root,
        graph,
        include_graph_evidence=include_graph_evidence,
        routing_threshold=routing_threshold,
        graph_evidence_mode=graph_evidence_mode,
        graph_token_features=graph_token_features,
        draw_pair_bboxes=draw_pair_bboxes,
    )


def test_draw_pair_bboxes_off_keeps_plain_frame_and_frame_cache_key(tmp_path):
    ds = _make_dataset(tmp_path, draw_pair_bboxes=False)
    item = ds[0]
    colors = {rgb for _, rgb in item.image.convert("RGB").getcolors(maxcolors=1 << 20)}
    assert colors == {(0, 0, 0)}  # untouched all-black frame
    assert item.vision_cache_key.endswith("frame.png")


def test_draw_pair_bboxes_on_draws_ab_boxes_and_pair_unique_cache_key(tmp_path):
    ds = _make_dataset(tmp_path, draw_pair_bboxes=True)
    item = ds[0]  # raw (i=0, j=1) -> canonical A=person 1, B=person 0
    colors = {rgb for _, rgb in item.image.convert("RGB").getcolors(maxcolors=1 << 20)}
    # A=red source box and B=blue target box are both drawn on the frame.
    assert (255, 0, 0) in colors
    assert (0, 0, 255) in colors
    # Each pair gets a distinct vision-reuse key so the collator cannot reuse one pair's
    # boxed frame for the rest of the frame. A=person_i=1, B=person_j=0.
    assert item.vision_cache_key.endswith("frame.png::A1B0")


def test_text_mode_builds_text_prompt_on_plain_image(tmp_path):
    ds = _make_dataset(tmp_path)
    item = ds[0]

    assert "Person A" in item.prompt and "Person B" in item.prompt
    assert item.prompt.rstrip().endswith(
        'Answer with a single word, "yes" or "no".'
    )
    assert "graph" in item.prompt.lower()          # evidence written into the prompt as text
    assert item.graph_token_payload is None          # default text path adds no dense payload
    assert item.image.getpixel((70, 85)) == (0, 0, 0)


def test_text_tokens_extract_canonical_lah_and_sa_features(tmp_path):
    dataset = _make_dataset(tmp_path, graph_evidence_mode="text_tokens")

    # Raw LAH (i=0,j=1) canonicalizes to Person A=1 -> Person B=0. The payload
    # must follow that canonical direction, never the raw manifest field order.
    lah = dataset[0]
    assert "<|graph_heatmap_a|>" in lah.prompt
    assert "<|graph_edge_ab|>" in lah.prompt
    assert lah.graph_token_payload is not None
    torch.testing.assert_close(
        lah.graph_token_payload.values["heatmap_a"], torch.full((8, 8), 41.0)
    )
    torch.testing.assert_close(
        lah.graph_token_payload.values["edge_ab"], torch.full((4,), 2.0)
    )

    sa = dataset[1]
    assert sa.graph_token_payload is not None
    assert set(sa.graph_token_payload.values) == {
        "heatmap_a", "heatmap_b", "edge_ab", "edge_ba"
    }
    torch.testing.assert_close(
        sa.graph_token_payload.values["heatmap_a"], torch.full((8, 8), 40.0)
    )
    torch.testing.assert_close(
        sa.graph_token_payload.values["heatmap_b"], torch.full((8, 8), 41.0)
    )
    torch.testing.assert_close(
        sa.graph_token_payload.values["edge_ab"], torch.full((4,), 1.0)
    )
    torch.testing.assert_close(
        sa.graph_token_payload.values["edge_ba"], torch.full((4,), 2.0)
    )


def test_graph_token_payload_whitelist_excludes_gt_and_metadata():
    cache = _fake_graph_cache()
    # These fields deliberately resemble evaluation labels/metadata. The extractor
    # accepts only gaze_heatmap and edge_pp, so adding them cannot alter the payload.
    cache.update(
        {
            "lah_gt": torch.ones(2, 2),
            "laeo_gt": torch.zeros(2, 2),
            "sa_gt": torch.full((2, 2), -1),
            "inout_gt": torch.tensor([1, 0]),
            "head_bboxes": torch.zeros(2, 4),
        }
    )
    payload = extract_graph_token_payload(
        task="lah", person_a=1, person_b=0, cache=cache,
        features=("gaze_heatmap", "edge_pp"),
    )
    assert set(payload.values) == {"heatmap_a", "edge_ab"}
    assert all("gt" not in name for name in payload.values)
    torch.testing.assert_close(payload.values["edge_ab"], torch.full((4,), 2.0))


def test_text_tokens_require_graph_evidence(tmp_path):
    with pytest.raises(ValueError, match="requires include_graph_evidence=True"):
        _make_dataset(
            tmp_path,
            include_graph_evidence=False,
            graph_evidence_mode="text_tokens",
        )


def test_text_mode_ablation_flag_drops_graph_from_prompt(tmp_path):
    ds = _make_dataset(tmp_path, include_graph_evidence=False)
    item = ds[0]

    assert "Person A" in item.prompt and "Person B" in item.prompt
    assert "graph" not in item.prompt.lower()


def test_routing_prompt_cue_matches_the_same_low_confidence_partition(tmp_path):
    dataset = _make_dataset(tmp_path, routing_threshold=0.9)

    # LAH logit=1.0 -> confidence 0.73, so this pair is VLM-routed.
    assert "not confident enough about this relation" in dataset[0].prompt
    # SA logits are very high-confidence in the fixture, so this pair retains the
    # standard prompt even though the same dataset object has routing enabled.
    assert "not confident enough about this relation" not in dataset[1].prompt


def test_routing_prompt_cue_includes_the_confidence_boundary(tmp_path):
    frame_root = tmp_path / "frames"
    _write_frame(frame_root, "s0")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    graph = {"s0": _fake_graph_cache()}
    # The raw LAH record is canonicalized to Person A=1 -> Person B=0.
    graph["s0"]["lah_logits"][1, 0] = torch.logit(torch.tensor(0.9))
    graph["s0"]["lah_logits_frames"][1, 1, 0] = torch.logit(torch.tensor(0.9))
    boundary = SocialInputDataset(manifest, frame_root, graph, routing_threshold=0.9)
    assert "not confident enough about this relation" in boundary[0].prompt

    graph["s0"]["lah_logits"][1, 0] = torch.logit(torch.tensor(0.91))
    graph["s0"]["lah_logits_frames"][1, 1, 0] = torch.logit(torch.tensor(0.91))
    high = SocialInputDataset(manifest, frame_root, graph, routing_threshold=0.9)
    assert "not confident enough about this relation" not in high[0].prompt


def test_dataset_canonicalizes_lah_direction_and_reuses_raw_lru(tmp_path):
    frame_root = tmp_path / "frames"
    _write_frame(frame_root, "s0")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    graph = {"s0": _fake_graph_cache()}
    dataset = SocialInputDataset(manifest, frame_root, graph, raw_image_cache_size=2)

    lah = dataset[0]
    sa = dataset[1]

    assert (lah.annotation.person_i, lah.annotation.person_j) == (1, 0)
    assert lah.prompt != sa.prompt
    assert lah.image is sa.image                    # same frame reused from the LRU
    assert dataset.frames.cache_info() == (1, 1, 2, 1)
    assert dataset.raw_frame_path(0) == frame_root / "s0" / "frame.png"


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
        SocialInputDataset(manifest, tmp_path, {})


def test_missing_raw_frame_has_contextual_error(tmp_path):
    cache = RawFrameCache(tmp_path)
    with pytest.raises(FileNotFoundError, match="s404"):
        cache.get("s404")


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
    annotations = SocialAnnotationDataset(manifest)
    cache = {"s0": _fake_graph_cache()}
    balanced = sample_weights(annotations, cache, balance_mode="task_label")
    torch.testing.assert_close(balanced, torch.ones(6, dtype=torch.double))
    hard = sample_weights(annotations, cache, balance_mode="task_label", hard_floor=0.25)
    assert torch.all(hard >= 0.25)
    assert torch.all(hard <= 1.25)
    assert task_pos_weights(annotations) == {"lah": 1.0, "laeo": 1.0, "sa": 1.0}


def test_confidence_routing_partitions_and_zeroes_high_confidence_weights(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    # lah(0,1) logit=1.0 -> conf 0.73 (low). sa/laeo symmetrize to logit>=11.5 -> conf ~1 (high).
    records = [
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "no"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "yes"},
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    annotations = SocialAnnotationDataset(manifest)
    cache = {"s0": _fake_graph_cache()}

    high, low = partition_by_graph_confidence(annotations, cache, 0.9)
    assert low == [0]                       # only the low-confidence lah pair reaches the VLM
    assert high == [1, 2]

    weights = sample_weights(annotations, cache, balance_mode="task", route_threshold=0.9)
    assert weights[0] > 0
    assert weights[1] == 0 and weights[2] == 0

    with pytest.raises(ValueError, match="no low-confidence training pairs"):
        sample_weights(annotations, cache, balance_mode="task", route_threshold=0.5)
    with pytest.raises(ValueError, match="threshold must be in"):
        partition_by_graph_confidence(annotations, cache, 1.5)


def test_low_confidence_boost_keeps_full_population_and_only_reweights_routed_pairs(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "no"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "yes"},
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    annotations = SocialAnnotationDataset(manifest)
    cache = {"s0": _fake_graph_cache()}

    boosted, low = boost_low_confidence_weights(
        torch.ones(3, dtype=torch.double), annotations, cache,
        threshold=0.9, multiplier=4.0,
    )
    assert low == [0]
    torch.testing.assert_close(
        boosted, torch.tensor([4.0, 1.0, 1.0], dtype=torch.double)
    )

    with pytest.raises(ValueError, match=">= 1"):
        boost_low_confidence_weights(
            torch.ones(3), annotations, cache, threshold=0.9, multiplier=0.5
        )
