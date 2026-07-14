import json

import pytest

from vlm.pair_dataset import PairAnnotationDataset, PairSample, frame_path


def _write_manifest(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_pair_sample_canonicalizes_lah_direction_once():
    sample = PairSample.from_manifest_record({
        "sid": "sample000001", "task": "lah",
        "i": 2, "j": 5, "li": "P2", "lj": "P5", "ans": "yes",
    })

    # Raw VSGaze/manifest: j looks at i. Internal contract: person_i -> person_j.
    assert (sample.person_i, sample.person_j) == (5, 2)
    assert (sample.person_i_name, sample.person_j_name) == ("P5", "P2")
    assert sample.label == 1
    assert sample.eval_key == ("sample000001", "lah", 2, 5)
    assert sample.canonical_key == ("sample000001", "lah", 5, 2)


@pytest.mark.parametrize("task", ["laeo", "sa"])
def test_symmetric_tasks_keep_manifest_order(task):
    sample = PairSample.from_manifest_record({
        "sid": "sample000002", "task": task,
        "i": 1, "j": 4, "li": "A", "lj": "B", "ans": 0,
    })
    assert (sample.person_i, sample.person_j) == (1, 4)
    assert (sample.person_i_name, sample.person_j_name) == ("A", "B")
    assert sample.label == 0


def test_every_manifest_annotation_becomes_one_sample(tmp_path):
    records = [
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "no"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": 1},
        {"sid": "s1", "task": "lah", "i": 3, "j": 2, "ans": 0},
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, records)

    dataset = PairAnnotationDataset(manifest)

    assert len(dataset) == len(records)
    assert dataset.frame_count == 2
    assert dataset.task_counts == {"lah": 2, "laeo": 1, "sa": 1}
    assert dataset.class_counts == {
        "lah": {0: 1, 1: 1},
        "laeo": {0: 1, 1: 0},
        "sa": {0: 0, 1: 1},
    }
    assert frame_path(tmp_path / "frames", dataset[0]) == tmp_path / "frames" / "s0" / "frame.png"


@pytest.mark.parametrize(
    "record, message",
    [
        ({"sid": "s", "task": "other", "i": 0, "j": 1, "ans": 1}, "task must"),
        ({"sid": "s", "task": "lah", "i": 0, "j": 0, "ans": 1}, "self-pair"),
        ({"sid": "s", "task": "lah", "i": 0, "j": 1, "ans": -1}, "answer must"),
        ({"sid": "s", "task": "lah", "i": 0, "j": 1, "ans": "maybe"}, "answer must"),
    ],
)
def test_invalid_annotations_are_rejected(record, message):
    with pytest.raises(ValueError, match=message):
        PairSample.from_manifest_record(record)


def test_duplicate_annotation_is_not_silently_oversampled(tmp_path):
    record = {"sid": "s", "task": "lah", "i": 0, "j": 1, "ans": "yes"}
    manifest = tmp_path / "duplicate.jsonl"
    _write_manifest(manifest, [record, record])

    with pytest.raises(ValueError, match="duplicate pair annotation"):
        PairAnnotationDataset(manifest)


def test_manifest_error_reports_source_line(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(
        json.dumps({"sid": "s", "task": "lah", "i": 0, "j": 1, "ans": "yes"})
        + "\n"
        + "not-json\n"
    )

    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        PairAnnotationDataset(manifest)
