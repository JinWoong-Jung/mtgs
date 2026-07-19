"""Regression test for _dataset_frame_ids under confidence-gated routing.

Routing wraps the low-confidence remainder in a torch.utils.data.Subset
(collect_generative_predictions), which has no .annotations of its own. This was never
exercised before route_threshold was actually wired up, and _dataset_frame_ids raised
ValueError the first time a real routed run reached validation with group_by_frame=True.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import Subset

from vlm.social.training import _dataset_frame_ids


class _FakeDataset:
    """Minimal stand-in for SocialInputDataset: only .annotations.samples and __len__."""

    def __init__(self, sids):
        samples = [SimpleNamespace(sid=sid) for sid in sids]
        self.annotations = SimpleNamespace(samples=samples)

    def __len__(self):
        return len(self.annotations.samples)


def test_dataset_frame_ids_plain_dataset():
    ds = _FakeDataset(["s0", "s1", "s2"])
    assert _dataset_frame_ids(ds) == ["s0", "s1", "s2"]


def test_dataset_frame_ids_subset_matches_indices():
    # Mirrors collect_generative_predictions: eval_dataset = Subset(dataset, low).
    ds = _FakeDataset(["s0", "s1", "s2", "s3", "s4"])
    low = [4, 1, 3]  # arbitrary, unordered, like partition_by_graph_confidence's output
    subset = Subset(ds, low)
    assert _dataset_frame_ids(subset) == ["s4", "s1", "s3"]
    assert len(_dataset_frame_ids(subset)) == len(subset)


def test_dataset_frame_ids_rejects_non_pair_dataset():
    class _NoAnnotations:
        def __len__(self):
            return 3

    with pytest.raises(ValueError, match="frame grouping requires a pair dataset"):
        _dataset_frame_ids(_NoAnnotations())
