from __future__ import annotations
"""Model pieces for experiment F: per-person graph-token projector, N×N social head,
the <ptok> injection hook (same mechanism as vlm.injection.install_hook), and a helper to
read per-person hidden states out of the final layer."""

import torch
import torch.nn as nn

from vlm.mp.prompt import PTOK   # noqa: F401  (re-export)


class PersonTokenProjector(nn.Module):
    """[v_src ‖ v_tgt ‖ null_in ‖ null_out] (4×256=1024) -> one soft token in VLM hidden."""
    def __init__(self, out_dim, in_dim=1024, hidden=1024):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, feats):          # feats (M, in_dim)
        return self.mlp(feats)


class SocialHead(nn.Module):
    """Per ordered pair (i,j): [h_i ‖ h_j ‖ edge_pp[j,i]] -> (LAH, LAEO, SA) logits.
    edge_pp[j,i] is the i->j directed edge (matches vlm.injection.gather_feats)."""
    def __init__(self, d_model, edge_dim=256, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3))

    def forward(self, h, edge_pp):     # h (N,D), edge_pp (N,N,E) -> (N,N,3)
        N = h.shape[0]
        hi = h.unsqueeze(1).expand(N, N, -1)        # row i
        hj = h.unsqueeze(0).expand(N, N, -1)        # col j
        rev = edge_pp.transpose(0, 1)               # rev[i,j] = edge_pp[j,i] = i->j
        x = torch.cat([hi, hj, rev], dim=-1)        # (N,N, 2D+E)
        return self.mlp(x)


def symmetrize(x):
    """(...,N,N) -> average with its transpose over the last two dims."""
    return (x + x.transpose(-2, -1)) / 2


def install_ptok_hook(lang_model):
    """forward_pre_hook: if lang_model._ptok = {'tokens': (K,D), 'mask': (B,L) bool} set,
    overwrite inputs_embeds at masked positions (row-major) with tokens."""
    def hook(module, args, kwargs):
        data = getattr(module, "_ptok", None)
        emb = kwargs.get("inputs_embeds")
        if data is not None and emb is not None:
            emb = emb.clone()
            emb[data["mask"]] = data["tokens"].to(emb.dtype).reshape(-1, emb.shape[-1])
            kwargs["inputs_embeds"] = emb
        return args, kwargs
    lang_model.register_forward_pre_hook(hook, with_kwargs=True)


def read_person_hidden(last_hidden, ptok_mask):
    """last_hidden (B,L,D), ptok_mask (B,L) bool -> list of (n_b, D), person order preserved."""
    out = []
    for b in range(last_hidden.shape[0]):
        out.append(last_hidden[b][ptok_mask[b]])
    return out
