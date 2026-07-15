import pytest

from vlm.cache.manifest import (
    balance_task_labels,
    build_manifest,
    filter_records,
    summarize,
)


def _record(sid, task, answer, pair=0):
    return {
        "sid": sid,
        "task": task,
        "i": pair,
        "j": pair + 1,
        "li": f"P{pair}",
        "lj": f"P{pair + 1}",
        "ans": answer,
    }


def _balanced_records():
    records = []
    for task in ("lah", "laeo", "sa"):
        records.append(_record("sample000000", task, "yes", pair=0))
        records.extend(_record(f"sample00000{i}", task, "no", pair=i) for i in range(1, 4))
    return records


def test_balance_keeps_all_positives_and_task_matched_negative_count():
    records = _balanced_records()
    result = balance_task_labels(records, seed=7)
    for task in ("lah", "laeo", "sa"):
        selected = [record for record in result if record["task"] == task]
        assert sum(record["ans"] == "yes" for record in selected) == 1
        assert sum(record["ans"] == "no" for record in selected) == 1
    assert all("li" in record and "lj" in record for record in result)


def test_balance_is_seeded_and_without_replacement():
    records = []
    for task in ("lah", "laeo", "sa"):
        records.extend(_record(f"sample{index:06d}", task, "yes", pair=index * 2) for index in range(2))
        records.extend(_record(f"sample{index + 10:06d}", task, "no", pair=index * 2 + 1) for index in range(6))
    first = balance_task_labels(records, seed=101)
    assert first == balance_task_labels(records, seed=101)
    assert len({(r["sid"], r["task"], r["i"], r["j"]) for r in first}) == len(first)


def test_balance_rejects_insufficient_negatives():
    records = _balanced_records()
    records = [record for record in records if not (record["task"] == "sa" and record["ans"] == "no")]
    with pytest.raises(ValueError, match="only 0 negatives"):
        balance_task_labels(records, seed=1)


def test_source_filter_and_stride_are_frame_closed():
    records = [
        _record("sample000000", "lah", "yes"),
        _record("sample000000", "laeo", "yes"),
        _record("sample000001", "lah", "no"),
        _record("sample000001", "laeo", "no"),
        _record("sample000002", "lah", "yes"),
        _record("sample000003", "lah", "no"),
    ]
    sources = {
        "sample000000": "childplay",
        "sample000001": "childplay",
        "sample000002": "videoattentiontarget",
        "sample000003": "laeo",
    }
    result = filter_records(
        records,
        sid_sources=sources,
        allowed_sources=("childplay", "videoattentiontarget"),
        frame_stride=2,
    )
    # stride restarts per source: childplay keeps first sid (both rows), VAT keeps its first sid.
    assert [(record["sid"], record["task"]) for record in result] == [
        ("sample000000", "lah"),
        ("sample000000", "laeo"),
        ("sample000002", "lah"),
    ]


def test_build_report_marks_sid_stride_as_approximate():
    records = _balanced_records()
    # Add a second positive/negative pair per task to make stride retain all tasks.
    records.extend(
        [
            _record("sample000010", task, "yes", pair=10)
            for task in ("lah", "laeo", "sa")
        ]
    )
    records.extend(
        [
            _record("sample000011", task, "no", pair=11)
            for task in ("lah", "laeo", "sa")
        ]
    )
    sources = {record["sid"]: "childplay" for record in records}
    output, report = build_manifest(
        records, sid_sources=sources, allowed_sources=("childplay",), frame_stride=1, seed=3
    )
    assert report.approximate_sid_stride is False
    assert report.output_records == len(output)
    assert report.counts == {task: {"yes": 2, "no": 2} for task in ("lah", "laeo", "sa")}
