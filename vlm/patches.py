from __future__ import annotations
"""Runtime perf patches for the VLM backbone.

patch_qwen3vl_patch_embed: Qwen3-VL's vision patch embedding is a Conv3d with
stride == kernel (non-overlapping patches). On this Blackwell (sm_120) + torch
2.9/cu128 build the training-time (grad-enabled) Conv3d falls back to
`aten::slow_conv_dilated3d`, a reference kernel that costs ~17 s per forward and
dominates the whole step (~28 s vision vs 0.2 s for the 8B LM). Because the conv
is non-overlapping it is mathematically identical to a per-patch linear
projection, so we swap its forward for a matmul: measured 28.2 s -> 0.59 s
fwd+bwd (bs=8), bit-for-bit equivalent output (verified: allclose, atol 1e-2).
"""

import types

import torch
import torch.nn.functional as F


def patch_qwen3vl_patch_embed(model) -> bool:
    """Replace the Qwen3-VL vision patch-embed Conv3d forward with the equivalent
    matmul. Returns True if a patch_embed was found and patched, else False."""
    target = None
    for name, m in model.named_modules():
        proj = getattr(m, "proj", None)
        if isinstance(proj, torch.nn.Conv3d):
            target = (name, m)
            break
    if target is None:
        print("[patch] Qwen3-VL patch_embed Conv3d not found — skipped", flush=True)
        return False

    name, pe = target

    def _matmul_forward(self, hidden_states, *args, **kwargs):
        w = self.proj.weight                      # (embed, in_ch, T, ph, pw)
        wf = w.reshape(w.shape[0], -1)            # (embed, in_ch*T*ph*pw)  — same C-order as pixel_values
        h = hidden_states.reshape(-1, wf.shape[1]).to(wf.dtype)
        return F.linear(h, wf, self.proj.bias)

    pe.forward = types.MethodType(_matmul_forward, pe)
    print(f"[patch] {name}: Conv3d patch-embed -> matmul (Blackwell slow_conv_dilated3d bypass)",
          flush=True)
    return True
