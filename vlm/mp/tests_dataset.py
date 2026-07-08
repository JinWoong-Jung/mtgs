import itertools
import torch
from vlm.mp.dataset import gt_matrices, person_feats, bucket_collate, LengthBucketSampler


def _synthetic_gtmeta(n):
    pairs = list(itertools.permutations(range(n), 2))
    lah = torch.tensor([1 if (i == 0 and j == 1) else (-1 if i == j else 0) for (i, j) in pairs])
    laeo = torch.zeros(len(pairs), dtype=torch.long)
    sa = torch.zeros(len(pairs), dtype=torch.long)
    return {"lah_gt": lah, "laeo_gt": laeo, "coatt_gt": sa}


def test_gt_matrices_reshape_and_diag():
    n = 3
    m = _synthetic_gtmeta(n)
    lah, laeo, sa = gt_matrices(m, n)
    assert lah.shape == (n, n)
    assert lah[0, 1].item() == 1
    assert (torch.diag(lah) == -1).all()
    assert lah[2, 0].item() == 0


def test_person_feats_concat_1024():
    n = 5
    gf = {"v_src": torch.randn(n, 256), "v_tgt": torch.randn(n + 2, 256),
          "edge_null_in": torch.randn(n, 256), "edge_null_out": torch.randn(n, 256)}
    f = person_feats(gf, [0, 2, 4])
    assert f.shape == (3, 1024)
    # first 256 dims of row 0 == v_src[0]
    assert torch.equal(f[0, :256], gf["v_src"][0])
    assert torch.equal(f[0, 256:512], gf["v_tgt"][0])


def test_bucket_collate_keeps_variable_n_as_lists():
    b0 = {"sid": "s0", "pil": None, "labels": ["P1", "P2"],
          "bboxes": torch.zeros(2, 4), "feats": torch.zeros(2, 1024),
          "edge_pp": torch.zeros(2, 2, 256), "lah": torch.zeros(2, 2, dtype=torch.long),
          "laeo": torch.zeros(2, 2, dtype=torch.long), "sa": torch.zeros(2, 2, dtype=torch.long)}
    b1 = {"sid": "s1", "pil": None, "labels": ["P1", "P2", "P3"],
          "bboxes": torch.zeros(3, 4), "feats": torch.zeros(3, 1024),
          "edge_pp": torch.zeros(3, 3, 256), "lah": torch.zeros(3, 3, dtype=torch.long),
          "laeo": torch.zeros(3, 3, dtype=torch.long), "sa": torch.zeros(3, 3, dtype=torch.long)}
    out = bucket_collate([b0, b1])
    assert out["sid"] == ["s0", "s1"]
    assert out["feats"][0].shape == (2, 1024) and out["feats"][1].shape == (3, 1024)
    assert out["lah"][0].shape == (2, 2) and out["lah"][1].shape == (3, 3)


def test_length_bucket_sampler_batches_have_small_length_spread():
    # continuous lengths -> sorted chunking pairs adjacent N (spread <= 1 for bs=2)
    lengths = [2, 3, 4, 5, 6, 7, 8, 9]
    s = LengthBucketSampler(lengths, batch_size=2, shuffle=True, seed=0)
    batches = list(iter(s))
    assert len(batches) == 4
    assert sorted(i for b in batches for i in b) == list(range(8))   # covers all, no dupes
    for b in batches:
        ns = [lengths[i] for i in b]
        assert max(ns) - min(ns) <= 1        # minimal padding within a batch


def test_length_bucket_sampler_covers_all_indices_uneven():
    lengths = [2, 2, 2, 5, 5]           # 5 items, bs=2 -> 3 batches, last partial
    s = LengthBucketSampler(lengths, batch_size=2, shuffle=False, seed=1)
    batches = list(iter(s))
    assert len(batches) == 3
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3, 4]
