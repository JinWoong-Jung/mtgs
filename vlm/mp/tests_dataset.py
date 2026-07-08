import itertools
import torch
from vlm.mp.dataset import gt_matrices, person_feats


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
