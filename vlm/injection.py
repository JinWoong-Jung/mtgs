from __future__ import annotations
"""Graph soft-token injection for Qwen3-VL (latent-space fusion, à la GraphVLM).

Injects the frozen Stage-1 graph's node/edge EMBEDDINGS as soft tokens at <gtok>
placeholders in the prompt, via a forward_pre_hook on the language model.
"""

import torch
import torch.nn as nn


"""Graph soft-token injection for Qwen3-VL (latent-space fusion, à la GraphVLM).

Injects the V14.5 graph's node/edge EMBEDDINGS as soft tokens into the VLM, instead of
serialising them to text (which the VLM parrots). Per query pair (i,j), N_TOK=10 vectors,
covering each person's emitter/receiver duality + the directed relation + both null channels:
  v_src[i], v_src[j]            -> i/j as LOOKERS (emitter / source role)
  v_tgt[i], v_tgt[j]            -> i/j as TARGETS (receiver / target role)
  edge_pp[i,j], edge_pp[j,i]    -> directed pair embeddings (both directions)
  edge_null_out[i], edge_null_out[j]  -> 'looks out-of-frame' channel (negative cue)
  edge_null_in[i], edge_null_in[j]    -> 'is gazed-at-by-nobody' channel (focal/in-degree)

Mechanism: the prompt carries N_TOK "<gtok>" placeholder tokens. A forward_pre_hook on the
TEXT model (model.model.language_model, which receives inputs_embeds AFTER the image merge)
overwrites the placeholder embeddings with the projected graph vectors. mrope/positions are
untouched (gtok are ordinary text positions).
"""


GTOK = "<gtok>"

N_TOK = 10   # DEAD after Task 4 — kept here only so still-old TokenDS/train/eval imports resolve; removed in Task 4.

# Role ids (INVARIANT — shared by gather_feats, the prompt builder, and the projector).
ROLE = {"SRC": 0, "TGT": 1, "EDGE_FWD": 2, "EDGE_BWD": 3, "NULL_IN": 4}
N_ROLES = 5

# Tokens injected per task (must equal the number of <gtok> the prompt emits).
TOK_COUNT = {"lah": 3, "laeo": 4, "sa": 6}


def gather_feats(d, task, i, j):
    """Task-specific role-aware graph embeddings for query pair (i, j).

    Returns (feats (K,256) float, role_ids (K,) long), K == TOK_COUNT[task], in the
    SAME order the prompt emits its <gtok> placeholders.

    Orientation (see module note / injection history): the directed edge whose readout
    means "i looks at j" is the slice edge_pp[j, i]; "j looks at i" is edge_pp[i, j].
      EDGE_FWD := edge_pp[j, i]   (i -> j)
      EDGE_BWD := edge_pp[i, j]   (j -> i)
    """
    v_src = d["v_src"].float()
    v_tgt = d["v_tgt"].float()
    epp = d["edge_pp"].float()
    e_fwd = epp[j, i]           # i -> j
    e_bwd = epp[i, j]           # j -> i
    if task == "lah":
        feats = [v_src[i], v_tgt[j], e_fwd]
        roles = [ROLE["SRC"], ROLE["TGT"], ROLE["EDGE_FWD"]]
    elif task == "laeo":
        feats = [v_src[i], v_src[j], e_fwd, e_bwd]
        roles = [ROLE["SRC"], ROLE["SRC"], ROLE["EDGE_FWD"], ROLE["EDGE_BWD"]]
    elif task == "sa":
        nin = d["edge_null_in"].float()
        feats = [v_src[i], v_src[j], nin[i], nin[j], e_fwd, e_bwd]
        roles = [ROLE["SRC"], ROLE["SRC"], ROLE["NULL_IN"], ROLE["NULL_IN"],
                 ROLE["EDGE_FWD"], ROLE["EDGE_BWD"]]
    else:
        raise ValueError(f"unknown task {task!r}")
    return torch.stack(feats), torch.tensor(roles, dtype=torch.long)


class GraphTokenProjector(nn.Module):
    """K 256-d graph vectors -> K soft tokens in the VLM hidden space, with a per-ROLE
    (not per-slot-index) type embedding so a given role means the same thing across the
    variable-length LAH/LAEO/SA token sets."""
    def __init__(self, out_dim, in_dim=256, n_roles=N_ROLES, hidden=1024):
        super().__init__()
        self.role_emb = nn.Parameter(torch.zeros(n_roles, in_dim))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, feats, role_ids):    # feats (M, in_dim), role_ids (M,)
        return self.mlp(feats + self.role_emb[role_ids])


def install_hook(lang_model):
    """Register the injection hook on the text model. Per batch, set
    lang_model._gtok = {'tokens': (B,K,D), 'mask': (B,L) bool} before the forward."""
    def hook(module, args, kwargs):
        data = getattr(module, "_gtok", None)
        emb = kwargs.get("inputs_embeds")
        if data is not None and emb is not None:
            emb = emb.clone()
            emb[data["mask"]] = data["tokens"].to(emb.dtype).reshape(-1, emb.shape[-1])
            kwargs["inputs_embeds"] = emb
        return args, kwargs
    lang_model.register_forward_pre_hook(hook, with_kwargs=True)
