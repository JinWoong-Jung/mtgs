import numpy as np
import pytest

from mtgs.utils.social_gaze import get_shuffle_idx
from vlm.cache.selection import _select_person_ids, apply_plan


def test_train_selection_matches_mtgs_shuffle_then_random_tail():
    person_ids = np.asarray([-1, 10, 11, 12, 13])
    inout = np.asarray([-1, -1, 1, 1, -1])
    np.random.seed(37)
    shuffled = person_ids[get_shuffle_idx(inout)]
    keep = np.random.randint(2, min(len(shuffled), 4) + 1)
    expected = shuffled[-keep:]

    np.random.seed(37)
    actual = _select_person_ids(person_ids, inout, split="train", num_people=4)

    np.testing.assert_array_equal(actual, expected)


def test_validation_selection_keeps_original_tail_order():
    person_ids = np.asarray([-1, 10, 11, 12, 13])
    inout = np.asarray([-1, -1, 1, 1, -1])
    np.random.seed(41)
    keep = np.random.randint(2, 5)

    np.random.seed(41)
    actual = _select_person_ids(person_ids, inout, split="val", num_people=4)

    np.testing.assert_array_equal(actual, person_ids[-keep:])


def test_apply_plan_attaches_path_keyed_person_ids():
    class ChildPlayDataset_temporal:
        paths = np.asarray(["a", "b"])

    class Concat:
        datasets = [ChildPlayDataset_temporal()]

    dataset = Concat()
    plan = {
        "num_people": 4,
        "records": [
            {"sid": "sample000000", "dataset": "childplay", "path": "a", "person_ids": [-1, 1]},
            {"sid": "sample000001", "dataset": "childplay", "path": "b", "person_ids": [2, 3]},
        ],
    }

    apply_plan(dataset, plan)

    np.testing.assert_array_equal(dataset.datasets[0].vlm_person_ids_by_path["a"], [-1, 1])
    np.testing.assert_array_equal(dataset.datasets[0].vlm_person_ids_by_path["b"], [2, 3])


def test_apply_plan_rejects_dataset_enumeration_mismatch():
    class ChildPlayDataset_temporal:
        paths = np.asarray(["a"])

    class Concat:
        datasets = [ChildPlayDataset_temporal()]

    with pytest.raises(ValueError, match="selection plan/dataset mismatch"):
        apply_plan(
            Concat(),
            {"num_people": 4, "records": [
                {"sid": "sample000000", "dataset": "childplay", "path": "other", "person_ids": [1, 2]}
            ]},
        )
