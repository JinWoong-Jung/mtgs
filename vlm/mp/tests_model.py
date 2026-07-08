import torch
from vlm.mp.model import PersonTokenProjector, SocialHead, symmetrize, read_person_hidden


def test_person_projector_shape():
    proj = PersonTokenProjector(out_dim=32)
    feats = torch.randn(5, 1024)
    out = proj(feats)
    assert out.shape == (5, 32)


def test_social_head_shape_and_uses_reverse_edge():
    N, D = 4, 32
    head = SocialHead(d_model=D, edge_dim=8, hidden=16)
    h = torch.randn(N, D)
    edge = torch.randn(N, N, 8)
    logits = head(h, edge)
    assert logits.shape == (N, N, 3)


def test_symmetrize():
    x = torch.tensor([[[1., 3.], [5., 7.]]])   # (1,2,2)
    s = symmetrize(x)
    assert torch.allclose(s, torch.tensor([[[1., 4.], [4., 7.]]]))


def test_read_person_hidden_selects_masked_positions_in_order():
    D = 3
    # batch of 1, seq len 5, ptok at positions 1 and 3
    last_hidden = torch.arange(15.).reshape(1, 5, D)
    mask = torch.tensor([[False, True, False, True, False]])
    out = read_person_hidden(last_hidden, mask)
    assert len(out) == 1
    assert out[0].shape == (2, D)
    assert torch.equal(out[0][0], last_hidden[0, 1])
    assert torch.equal(out[0][1], last_hidden[0, 3])
