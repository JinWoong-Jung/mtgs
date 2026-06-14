# tests/test_memory_attn.py
import torch
import pytest
from mtgs.networks.vlm.memory_attn import MemoryCrossAttn, MemoryAugmentedLayer


def test_gate_zero_init_is_identity():
    """With gate=0, output must equal input exactly."""
    ca = MemoryCrossAttn(d_model=32, num_heads=4)
    assert ca.gate.item() == 0.0
    B, S, M = 2, 10, 8
    h = torch.randn(B, S, 32)
    G = torch.randn(B, M, 32)
    out = ca(h, G)
    assert torch.allclose(out, h), "gate=0 should be identity"


def test_cross_attn_output_shape():
    ca = MemoryCrossAttn(d_model=64, num_heads=4)
    h = torch.randn(3, 15, 64)
    G = torch.randn(3, 32, 64)
    out = ca(h, G)
    assert out.shape == h.shape


class _DummyLayer(torch.nn.Module):
    """Simulates a transformer layer that returns (hidden_states, *extras)."""
    def forward(self, h, **kwargs):
        return (h * 2.0, torch.tensor(1.0))  # two-element output


def test_memory_augmented_layer_no_G():
    """Without G_LLM set, wrapper is transparent (passes through to original)."""
    orig = _DummyLayer()
    ca = MemoryCrossAttn(d_model=8, num_heads=2)
    layer = MemoryAugmentedLayer(orig, ca)
    h = torch.randn(1, 5, 8)
    out = layer(h)
    assert torch.allclose(out[0], h * 2.0)


def test_memory_augmented_layer_with_G():
    """With G_LLM set and gate trained, output differs from input*2."""
    orig = _DummyLayer()
    ca = MemoryCrossAttn(d_model=8, num_heads=2)
    # Force gate != 0
    with torch.no_grad():
        ca.gate.fill_(1.0)
    layer = MemoryAugmentedLayer(orig, ca)
    h = torch.randn(1, 5, 8)
    G = torch.randn(1, 4, 8)
    layer._G_LLM = G
    out = layer(h)
    # Should NOT equal h*2.0 because cross-attn adds something
    assert not torch.allclose(out[0], h * 2.0)
