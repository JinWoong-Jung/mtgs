from __future__ import annotations
"""Frame-level pairwise social-gaze readout head (VLM Stage-2, frame pipeline).

After ONE Qwen3-VL forward over a whole frame, each person k has an anchor hidden
state h_k (read from the <panc> position). For every queried pair the head predicts a
STANDALONE VLM logit and fuses it with the frozen graph logit by a FIXED convex blend:

    logit_final  = blend_w · graph_logit + (1-blend_w) · vlm_logit(h_a,h_b,E…)

blend_w is a CONSTANT (default 0.5), NOT learned. On test, the v4 learned per-pair
router overfit (its α collapsed toward VLM-only on LAH/LAEO) and LOST to a plain fixed
blend; the fixed blend (esp. graph-dominant ~0.75) beat graph on all tasks. So we drop
the router and fix the weight.

vlm_logit is separately supervised (aux BCE), so it stays a CALIBRATED logit on the same
scale as graph_logit — making the fixed logit-space average sensible — and gets a live
gradient regardless of blend_w (no deadlock/stall). The vlm MLP's final layer is zero-
init so vlm_logit=0 at step 0 (logit_final = blend_w·graph → graph-equivalent ranking).

Role of each input (why graph features enter the readout, not just the prompt):
  h_a, h_b : VLM scene/appearance semantics for the two people (frame-level context).
  lah      : + E[a→b]                       — directed "a looks at b" edge evidence.
  laeo     : + E[a→b], E[b→a]  (symmetrised) — both directions of the mutual look.
  sa       : + nin_a, nin_b, |nin_a−nin_b|  (symmetrised) — scene-gaze (null_in) channels;
             SA is judged from WHERE each person looks in the scene, not the p2p edge.
"""

import torch
import torch.nn as nn

from vlm.prompt import TASK_ID


def _mlp(in_dim, hidden=512):
    """3-layer VLM-logit MLP -> scalar, FINAL layer zero-init so vlm_logit starts at 0
    (=> logit_final = α·graph at init: graph-equivalent ranking/decisions)."""
    last = nn.Linear(hidden, 1)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        last,
    )


class PairwiseSocialHead(nn.Module):
    """Per-task VLM-logit heads (lah/laeo/sa) fused with the frozen graph logit by a
    FIXED (non-learned) convex blend: final = blend_w·graph + (1-blend_w)·vlm_logit.

    blend_w is a constant (default 0.5 = 50/50), NOT learned — the learned per-pair
    router (v4) overfit and lost to a plain fixed blend on test, so we fix the weight.
    vlm_logit is separately supervised (aux BCE) so it stays a calibrated logit on the
    same scale as graph_logit, making the fixed logit-space average meaningful.

    forward returns (final, vlm_logit, alpha) where alpha is the constant blend_w
    broadcast over the batch (kept for logging/diagnostic compatibility)."""

    def __init__(self, d_lm, d_edge=256, d_proj=512, hidden=512, blend_w=0.5):
        super().__init__()
        self.d_proj = d_proj
        # Shared person projection: LM hidden (D≈4096) -> d_proj, keeps head MLPs small.
        self.p_proj = nn.Sequential(nn.Linear(d_lm, d_proj), nn.GELU())
        self.head_lah  = _mlp(2 * d_proj + d_edge, hidden)                 # +E_fwd
        self.head_laeo = _mlp(2 * d_proj + 2 * d_edge, hidden)             # +E_fwd,E_bwd
        self.head_sa   = _mlp(2 * d_proj + 3 * d_edge, hidden)             # +nin_a,nin_b,|Δ|
        # Fixed graph blend weight (buffer -> saved in ckpt, restored at eval).
        self.register_buffer("blend_w", torch.tensor(float(blend_w)))

    # ── per-task standalone VLM logit (directed for lah; symmetrised for laeo/sa) ──

    def _vlm_lah(self, pa, pb, e_fwd):
        return self.head_lah(torch.cat([pa, pb, e_fwd], dim=-1)).squeeze(-1)

    def _vlm_laeo(self, pa, pb, e_fwd, e_bwd):
        d_ab = self.head_laeo(torch.cat([pa, pb, e_fwd, e_bwd], dim=-1)).squeeze(-1)
        d_ba = self.head_laeo(torch.cat([pb, pa, e_bwd, e_fwd], dim=-1)).squeeze(-1)
        return 0.5 * (d_ab + d_ba)

    def _vlm_sa(self, pa, pb, nin_a, nin_b):
        diff = (nin_a - nin_b).abs()
        d_ab = self.head_sa(torch.cat([pa, pb, nin_a, nin_b, diff], dim=-1)).squeeze(-1)
        d_ba = self.head_sa(torch.cat([pb, pa, nin_b, nin_a, diff], dim=-1)).squeeze(-1)
        return 0.5 * (d_ab + d_ba)

    def forward(self, task, h_a, h_b, edges, graph_logit):
        """Batched over the records of ONE task.

        task        : "lah" | "laeo" | "sa"
        h_a, h_b    : (R, d_lm) anchor hidden states for the pair's two people
        edges       : dict of (R, d_edge) tensors —
                        lah : {"e_fwd"}
                        laeo: {"e_fwd", "e_bwd"}
                        sa  : {"nin_a", "nin_b"}
        graph_logit : (R,) frozen graph logit for the pair (blend base)
        Returns     : (final_logit (R,), vlm_logit (R,), alpha (R,))
                        final_logit = blend_w·graph_logit + (1-blend_w)·vlm_logit
                        vlm_logit   = standalone VLM prediction (for the aux BCE)
                        alpha       = constant blend_w broadcast (logging compatibility)
        """
        pa, pb = self.p_proj(h_a), self.p_proj(h_b)
        if task == "lah":
            vlm_logit = self._vlm_lah(pa, pb, edges["e_fwd"])
        elif task == "laeo":
            vlm_logit = self._vlm_laeo(pa, pb, edges["e_fwd"], edges["e_bwd"])
        elif task == "sa":
            vlm_logit = self._vlm_sa(pa, pb, edges["nin_a"], edges["nin_b"])
        else:
            raise ValueError(f"unknown task {task!r}")
        aw = self.blend_w
        final = aw * graph_logit + (1.0 - aw) * vlm_logit
        alpha = aw.expand_as(final)
        return final, vlm_logit, alpha
