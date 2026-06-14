# tests/test_vlm_trainer_hook.py
"""Verifies that the _UnifiedRefiner forward hook captures correct shapes.
Does NOT load Qwen3-VL-8B. Tests only the hook mechanism."""
import torch
import pytest
from mtgs.networks.adaptor_modules import GazeGraphBlock


def test_refiner_hook_captures_shapes():
    B, T, N, D = 1, 5, 3, 64  # small for CPU test
    De = 64   # edge_dim: must be >= 64 so _SocialReadoutHead(De) stays valid
    heads = 4  # D % heads == 0, De % heads == 0
    Hh, Ww = 16, 16  # small heatmap spatial dims

    # Instantiate GazeGraphBlock with minimal valid args
    block = GazeGraphBlock(
        token_dim=D,
        edge_dim=De,
        num_layers=1,      # 1 refiner layer is enough for the hook test
        heads=heads,
        use_prior=True,
        use_node_xattn=True,
    )
    block.eval()

    captured = {}

    def hook(module, inp, output):
        E, v_src, v_tgt = output
        captured["E"] = E
        captured["v_src"] = v_src
        captured["v_tgt"] = v_tgt

    block.refiner.register_forward_hook(hook)

    # Build minimal inputs and call forward
    # All N people are valid (num_valid_people = N)
    person_tokens = torch.randn(B, T, N, D)
    num_valid_people = torch.tensor([N] * B)
    gaze_vecs = torch.nn.functional.normalize(torch.randn(B, T, N, 2), dim=-1)
    head_bboxes = torch.rand(B, T, N, 4)
    # Ensure x1 < x2 and y1 < y2 for valid bboxes
    head_bboxes[..., 2:] = head_bboxes[..., :2] + 0.1
    head_bboxes = head_bboxes.clamp(0.0, 1.0)
    gaze_heatmaps = torch.relu(torch.randn(B, T, N, Hh, Ww))
    inout_logits = torch.randn(B, T, N)

    with torch.no_grad():
        block(person_tokens, num_valid_people, gaze_vecs,
              head_bboxes, gaze_heatmaps, inout_logits)

    De_out = De
    Tl = N + 2
    assert "E" in captured, "Hook did not fire — refiner was not called"
    assert captured["E"].shape == (B, T, N, Tl, De_out), \
        f"E shape mismatch: got {captured['E'].shape}"
    assert captured["v_src"].shape == (B, T, N, De_out), \
        f"v_src shape mismatch: got {captured['v_src'].shape}"
    assert captured["v_tgt"].shape == (B, T, Tl, De_out), \
        f"v_tgt shape mismatch: got {captured['v_tgt'].shape}"
