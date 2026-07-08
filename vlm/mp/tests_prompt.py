import torch
from vlm.mp.prompt import frame_prompt, PTOK


def test_frame_prompt_emits_one_ptok_per_person():
    bb = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.1, 0.2, 0.3]])
    labels = ["P1", "P2", "P3"]
    p = frame_prompt(labels, bb)
    assert p.count(PTOK) == 3
    # person order preserved: P1 appears before P2 before P3
    assert p.index("P1") < p.index("P2") < p.index("P3")
    # each person's bbox rendered with 2-decimal coords
    assert "[0.10,0.20,0.30,0.40]" in p


def test_frame_prompt_single_person():
    bb = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    p = frame_prompt(["P1"], bb)
    assert p.count(PTOK) == 1
