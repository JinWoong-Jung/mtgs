import torch
from vlm.mp.eval import logits_to_preds


def test_logits_to_preds_keys_and_symmetry():
    # 2 selected people whose ORIGINAL indices are 3 and 5
    idxs = [3, 5]
    logits = torch.zeros(2, 2, 3)
    logits[0, 1, 0] = 10.0     # lah 3->5 strong yes
    logits[1, 0, 0] = -10.0    # lah 5->3 strong no
    logits[0, 1, 1] = 10.0
    logits[1, 0, 1] = 0.0      # laeo asymmetric -> averaged
    preds = logits_to_preds("s1", logits, idxs)
    assert preds[("s1", "lah", 3, 5)] > 0.99
    assert preds[("s1", "lah", 5, 3)] < 0.01
    # laeo keyed canonically (lo<hi) and symmetric (avg of 10 and 0 -> logit 5 -> ~0.993)
    assert ("s1", "laeo", 3, 5) in preds and ("s1", "laeo", 5, 3) not in preds
    assert abs(preds[("s1", "laeo", 3, 5)] - torch.sigmoid(torch.tensor(5.0)).item()) < 1e-5
    # no diagonal / self keys
    assert ("s1", "lah", 3, 3) not in preds
