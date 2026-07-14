"""WP3/WP6 acceptance tests for PairResidualDecoder (CPU, synthetic tensors).
Run: python -m mtgs.social_vlm.test_residual

  * init: final == graph_logit EXACTLY (delta zero-init)
  * LAH directed; LAEO/SA EXACTLY symmetric (even with random weights)
  * gradient flows to decoder AND to fused memory
  * tiny-subset overfit: final can be driven toward arbitrary targets
"""

from __future__ import annotations

import torch

from mtgs.social_vlm.residual_decoder import PairResidualDecoder

De, DM, K = 256, 512, 16


def _fake_center(B=2, N=4):
    g = lambda *s: torch.randn(*s)
    lah = g(B, N, N)
    laeo = 0.5 * (g(B, N, N) + g(B, N, N).transpose(1, 2)); laeo = 0.5 * (laeo + laeo.transpose(1, 2))
    sa = g(B, N, N); sa = 0.5 * (sa + sa.transpose(1, 2))
    return {
        "v_src": g(B, N, De), "v_tgt": g(B, N + 2, De), "edge_states": g(B, N, N + 2, De),
        "lah_logits": lah, "laeo_logits": laeo, "sa_logits": sa,
        "null_in_logits": g(B, N), "null_out_logits": g(B, N), "inout_logits": g(B, N),
        "alignment": g(B, N, N), "overlap": g(B, N, N),
        "valid_person_mask": torch.ones(B, N, dtype=torch.bool),
        "pair_mask": torch.ones(B, N, N, dtype=torch.bool),
    }


def test_init_graph_equivalent():
    dec = PairResidualDecoder(d_edge=De, d_mem=DM).eval()
    c = _fake_center(); mem = torch.randn(2, K, DM)
    out = dec(c, mem)
    for t in ("lah", "laeo", "sa"):
        assert torch.allclose(out[t]["final"], c[f"{t}_logits"], atol=1e-6), f"{t} not graph at init"
        assert out[t]["delta"].abs().max() < 1e-6, f"{t} delta nonzero at init"
    print("[1] init: final == graph_logit (all tasks), delta=0  OK")


def test_symmetry_and_direction():
    torch.manual_seed(0)
    dec = PairResidualDecoder(d_edge=De, d_mem=DM).eval()
    # perturb final layers so delta != 0 (init is zero); use random memory
    for m in dec.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data += 0.05 * torch.randn_like(m.weight)
    c = _fake_center(); mem = torch.randn(2, K, DM)
    out = dec(c, mem)
    sym_laeo = (out["laeo"]["delta"] - out["laeo"]["delta"].transpose(1, 2)).abs().max()
    sym_sa = (out["sa"]["delta"] - out["sa"]["delta"].transpose(1, 2)).abs().max()
    dir_lah = (out["lah"]["delta"] - out["lah"]["delta"].transpose(1, 2)).abs().max()
    print(f"[2] symmetry: laeo asym={sym_laeo:.2e} sa asym={sym_sa:.2e} | lah dir-asym={dir_lah:.3f}")
    assert sym_laeo < 1e-5 and sym_sa < 1e-5, "LAEO/SA must be exactly symmetric"
    assert dir_lah > 1e-3, "LAH must be directed (asymmetric)"


def test_gradient_flow():
    # delta head is zero-init (LoRA-B pattern): at step 0, gradient to upstream/memory is
    # 0 by design (final=graph). After the delta head moves once, gradient reaches memory.
    # Simulate post-warmup by perturbing the zero-init final layers, then check flow.
    torch.manual_seed(0)
    dec = PairResidualDecoder(d_edge=De, d_mem=DM).train()
    for name, p in dec.named_parameters():
        if name.endswith("delta.2.weight") or name.endswith("delta.2.bias"):
            p.data += 0.1 * torch.randn_like(p)          # delta head no longer zero
    c = _fake_center(); mem = torch.randn(2, K, DM, requires_grad=True)
    out = dec(c, mem)
    loss = sum(out[t]["final"].pow(2).mean() for t in ("lah", "laeo", "sa"))
    loss.backward()
    g_dec = sum(p.grad.abs().sum() for p in dec.parameters() if p.grad is not None)
    print(f"[3] grad (post-warmup): decoder={float(g_dec):.3f}  "
          f"memory(VLM)={float(mem.grad.abs().sum()):.4f}")
    assert g_dec > 0 and mem.grad.abs().sum() > 0, "gradient must flow to decoder AND memory"


def test_tiny_overfit():
    torch.manual_seed(0)
    N = 3
    dec = PairResidualDecoder(d_edge=De, d_mem=DM).train()
    c = _fake_center(B=1, N=N); mem = torch.randn(1, K, DM)
    off = ~torch.eye(N, dtype=torch.bool)              # ignore self-pairs (like the real mask)
    tgt = {}
    for t in ("lah", "laeo", "sa"):
        m = torch.rand(1, N, N).round()
        if t != "lah":
            m = ((m + m.transpose(1, 2)) >= 1).float()  # symmetric target for laeo/sa
        tgt[t] = m
    # NOTE: the learned-gate residual is lr-sensitive — too high an lr collapses the
    # gate to 0 (sigmoid(gate)->0, correction vanishes). 3e-3 opens the gate and fits
    # perfectly; this informs the real Stage-A decoder/gate lr (keep it moderate).
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(600):
        out = dec(c, mem)
        loss = sum(bce(out[t]["final"][:, off], tgt[t][:, off]) for t in ("lah", "laeo", "sa"))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[4] tiny-overfit final loss={float(loss):.4f} (should be ~0)")
    assert float(loss) < 0.15, "should overfit a tiny batch"


if __name__ == "__main__":
    test_init_graph_equivalent()
    test_symmetry_and_direction()
    test_gradient_flow()
    test_tiny_overfit()
    print("\nWP3/WP6 residual-decoder tests PASSED")
