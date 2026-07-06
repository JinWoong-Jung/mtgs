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


N_TOK = 10


def gather_feats(d, i, j):
    """(N_TOK, 256) graph embedding stack for query pair (i, j) from a v14graph[sid] dict."""
    return torch.stack([
        d["v_src"][i].float(), d["v_src"][j].float(),
        d["v_tgt"][i].float(), d["v_tgt"][j].float(),
        d["edge_pp"][i, j].float(), d["edge_pp"][j, i].float(),
        d["edge_null_out"][i].float(), d["edge_null_out"][j].float(),
        d["edge_null_in"][i].float(), d["edge_null_in"][j].float(),
    ])


class GraphTokenProjector(nn.Module):
    """N_TOK 256-d graph vectors -> N_TOK soft tokens in the VLM hidden space."""
    def __init__(self, out_dim, in_dim=256, n_tok=N_TOK, hidden=1024):
        super().__init__()
        self.type_emb = nn.Parameter(torch.zeros(n_tok, in_dim))  # per-slot type embedding
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, feats):  # (B, n_tok, in_dim) -> (B, n_tok, out_dim)
        return self.mlp(feats + self.type_emb)


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
