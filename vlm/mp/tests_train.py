import torch
from vlm.mp.train import social_bce


def test_social_bce_masks_invalid_and_is_finite():
    B, N = 1, 3
    logits = torch.zeros(B, N, N, 3, requires_grad=True)
    lah = torch.full((B, N, N), -1)      # all masked
    laeo = torch.full((B, N, N), -1)
    sa = torch.full((B, N, N), -1)
    lah[0, 0, 1] = 1                     # one valid positive
    loss = social_bce(logits, lah, laeo, sa)
    assert torch.isfinite(loss)
    loss.backward()
    # gradient only flows through the LAH channel of the valid pair
    assert logits.grad[0, 0, 1, 0].abs() > 0
    assert logits.grad[0, 2, 2, 0].abs() == 0    # diagonal masked


def test_social_bce_all_masked_returns_zero():
    B, N = 1, 2
    logits = torch.zeros(B, N, N, 3, requires_grad=True)
    g = torch.full((B, N, N), -1)
    loss = social_bce(logits, g, g.clone(), g.clone())
    assert loss.item() == 0.0
