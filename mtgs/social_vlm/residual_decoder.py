"""WP3/WP6 — PairResidualDecoder: graph-residual social-gaze correction.

For every pair the decoder builds a task query from the frozen graph evidence (center
frame), cross-attends to a fused scene memory (K tokens: frozen Qwen video features in
WP3, graph-conditioned Qwen hidden states in WP5), and emits a per-pair (delta, gate):

    final = graph_logit + sigmoid(gate) * delta                    (learned-gate residual)

Init (spec): delta final layer zero, gate final bias = -2. So at step 0 delta=0 =>
final == graph_logit EXACTLY (graph-equivalent), and once delta grows the initial gate
is sigmoid(-2)≈0.12 (small, graph-dominant). LAH is directed; LAEO/SA use symmetric
query construction so delta is EXACTLY symmetric by build (no upper-tri bookkeeping).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _delta_head(d_model, hidden=256):
    last = nn.Linear(hidden, 1)
    nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)          # delta=0 at init
    return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), last)


def _gate_head(d_model, hidden=256):
    last = nn.Linear(hidden, 1)
    nn.init.zeros_(last.weight); nn.init.constant_(last.bias, -2.0)  # sigmoid(-2)≈0.12
    return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), last)


class _PairXAttn(nn.Module):
    """pair query -> cross-attention to fused memory -> FFN -> (delta, gate) per pair."""

    def __init__(self, q_in, d_model, d_mem, heads=8, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(q_in, d_model)
        self.kv_proj = nn.Linear(d_mem, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.delta = _delta_head(d_model)
        self.gate = _gate_head(d_model)

    def forward(self, q, memory, mem_mask=None):
        # q: (B, P, q_in)   memory: (B, K, d_mem)   mem_mask: (B,K) bool True=valid
        q = self.q_proj(q)
        kv = self.kv_proj(memory)
        kpm = ~mem_mask if mem_mask is not None else None      # key_padding_mask: True=ignore
        a, _ = self.attn(q, kv, kv, key_padding_mask=kpm)
        x = self.norm(q + a)
        x = self.norm2(x + self.ffn(x))
        return self.delta(x).squeeze(-1), self.gate(x).squeeze(-1)   # (B,P),(B,P)


class PairResidualDecoder(nn.Module):
    """Builds per-task pair queries from bundle.center() + cross-attends to fused memory.
    d_mem = fused memory feature dim (Qwen video feat in WP3 / Qwen hidden 4096 in WP5)."""

    def __init__(self, d_edge=256, d_model=256, d_mem=512, heads=8, dropout=0.1):
        super().__init__()
        De = d_edge
        # query input dims (see _q_* builders)
        self.lah  = _PairXAttn(2 * De + De + 3,           d_model, d_mem, heads, dropout)
        self.laeo = _PairXAttn(2 * De + 2 * De + 2,       d_model, d_mem, heads, dropout)
        self.sa   = _PairXAttn(2 * De + 2 * De + 1,       d_model, d_mem, heads, dropout)

    # ── per-task pair-query construction (from center-frame bundle dict) ──────────

    @staticmethod
    def _q_lah(c):
        vs, vt, E = c["v_src"], c["v_tgt"], c["edge_states"]     # [B,N,De],[B,N+2,De],[B,N,N+2,De]
        B, N, De = vs.shape
        looker = vs.unsqueeze(2).expand(B, N, N, De)            # [B,looker,target,De]
        target = vt[:, :N].unsqueeze(1).expand(B, N, N, De)     # person target nodes
        e = E[:, :, :N, :]                                      # E[looker,target]
        geom = torch.stack([c["alignment"], c["overlap"], c["lah_logits"]], -1)   # [B,N,N,3]
        return torch.cat([looker, target, e, geom], dim=-1)    # [B,N,N,2De+De+3]

    @staticmethod
    def _q_laeo(c):
        vs, E, lah = c["v_src"], c["edge_states"], c["lah_logits"]
        B, N, De = vs.shape
        va = vs.unsqueeze(2).expand(B, N, N, De)
        vb = vs.unsqueeze(1).expand(B, N, N, De)
        e_ab = E[:, :, :N, :]                                   # E[a,b]
        e_ba = e_ab.transpose(1, 2)                             # E[b,a]
        node = torch.cat([va + vb, (va - vb).abs()], -1)                       # symmetric
        edge = torch.cat([e_ab + e_ba, (e_ab - e_ba).abs()], -1)              # symmetric
        rel = torch.stack([lah + lah.transpose(1, 2), (lah - lah.transpose(1, 2)).abs()], -1)
        return torch.cat([node, edge, rel], dim=-1)            # [B,N,N,4De+2]

    @staticmethod
    def _q_sa(c):
        vs, E, sa = c["v_src"], c["edge_states"], c["sa_logits"]
        B, N, De = vs.shape
        va = vs.unsqueeze(2).expand(B, N, N, De)
        vb = vs.unsqueeze(1).expand(B, N, N, De)
        nin = E[:, :, N, :]                                     # null_in edge per person [B,N,De]
        na = nin.unsqueeze(2).expand(B, N, N, De)
        nb = nin.unsqueeze(1).expand(B, N, N, De)
        node = torch.cat([va + vb, (va - vb).abs()], -1)
        scene = torch.cat([na + nb, (na - nb).abs()], -1)      # symmetric scene-gaze
        return torch.cat([node, scene, sa.unsqueeze(-1)], dim=-1)   # [B,N,N,4De+1]

    def _run_task(self, head, q, graph_logit, memory, mem_mask):
        B, N, _, Dq = q.shape
        delta, gate = head(q.reshape(B, N * N, Dq), memory, mem_mask)
        delta = delta.reshape(B, N, N); gate = gate.reshape(B, N, N)
        final = graph_logit + torch.sigmoid(gate) * delta
        return final, delta, gate

    def forward(self, center, memory, mem_mask=None):
        """center = GraphFeatureBundle.center() dict; memory = (B,K,d_mem).
        Returns dict per task: {"final","delta","gate"} each [B,N,N] (lah directed;
        laeo/sa exactly symmetric by construction)."""
        out = {}
        out["lah"] = dict(zip(("final", "delta", "gate"),
                              self._run_task(self.lah, self._q_lah(center),
                                          center["lah_logits"], memory, mem_mask)))
        out["laeo"] = dict(zip(("final", "delta", "gate"),
                               self._run_task(self.laeo, self._q_laeo(center),
                                           center["laeo_logits"], memory, mem_mask)))
        out["sa"] = dict(zip(("final", "delta", "gate"),
                             self._run_task(self.sa, self._q_sa(center),
                                         center["sa_logits"], memory, mem_mask)))
        return out
