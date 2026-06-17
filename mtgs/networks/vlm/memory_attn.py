import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MemoryCrossAttn(nn.Module):
    """Cross-attention from LLM hidden states to G_LLM evidence tokens.

    h_out = h + gate * LN(CrossAttn(q=h, kv=G_LLM))
    gate is initialized to 0 so the layer starts as identity.
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: Tensor, G: Tensor) -> Tensor:
        # h: (B, seq_len, d_model)
        # G: (B, M, d_model)
        delta, _ = self.attn(h, G, G)
        update = (self.gate * self.norm(delta)).to(h.dtype)  # preserve hidden-state dtype
        return h + update


class MemoryAugmentedLayer(nn.Module):
    """Wraps one transformer layer, injecting cross-attn when _G_LLM is set.

    Usage:
        layer._G_LLM = G_LLM   # before forward
        output = layer(...)
        layer._G_LLM = None    # after forward
    """

    def __init__(self, original_layer: nn.Module, cross_attn: MemoryCrossAttn):
        super().__init__()
        self.layer = original_layer
        self.cross_attn = cross_attn
        self._G_LLM: Optional[Tensor] = None

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if self._G_LLM is None:
            return out
        # Qwen3-VL decoder layers return a bare hidden-state tensor; older HF
        # decoder layers return a tuple (hidden_states, ...). Support both so the
        # wrapper never changes the layer's output contract.
        if isinstance(out, tuple):
            h = self.cross_attn(out[0], self._G_LLM)
            return (h,) + out[1:]
        return self.cross_attn(out, self._G_LLM)
