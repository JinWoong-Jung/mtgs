from __future__ import annotations
"""Graph + heatmap soft-token injection for Qwen3-VL (latent-space fusion).

Two latent modalities are injected via a forward_pre_hook on the language model:
  <gtok>  — frozen Stage-1 graph node/edge EMBEDDINGS (256-d), role-keyed
            (TOK_COUNT[task]: lah=5, laeo=6, sa=6). SA uses only the scene-gaze
            (null_in) channel, not the person↔person edge.
  <hmtok> — the graph stage's predicted gaze HEATMAP per queried person
            (HM_COUNT[task]: lah=1 [A], laeo/sa=2 [A,B]), encoded by HeatmapEncoder.
"""

import torch
import torch.nn as nn

from mtgs.social_vlm.conventions import manifest_record_to_indices


GTOK = "<gtok>"
HMTOK = "<hmtok>"
PANC = "<panc>"   # per-person anchor: hidden state here is read as the person token (frame pipeline)

# Role ids (INVARIANT — shared by gather_feats, the prompt builder, and the projector).
ROLE = {"SRC": 0, "TGT": 1, "EDGE_FWD": 2, "EDGE_BWD": 3, "NULL_IN": 4}
N_ROLES = 5

# Graph soft-tokens per task (must equal the number of <gtok> the prompt emits).
TOK_COUNT = {"lah": 5, "laeo": 6, "sa": 6}
# Heatmap soft-tokens per task (must equal the number of <hmtok> the prompt emits).
HM_COUNT = {"lah": 1, "laeo": 2, "sa": 2}


def query_slots(rec):
    """Map a manifest record to (a, b, label_a, label_b) query slots.

    ORIENTATION INVARIANT (empirically verified on val, 20k LAH records):
    the manifest stores pair (i, j) in the DATASET pair convention where the
    answer means "j looks at i" (TARGET, LOOKER) — P(gaze_point[j] in bbox[i])
    is 0.77 for yes vs 0.07 for no, and graph lah_logits[j, i] separates the
    answer at AUC 0.94 (vs 0.57 for [i, j]).

    The prompt asks the natural question "Is A looking at B?", so for LAH the
    first-named person A must be the LOOKER = slot j, and B = slot i.
    LAEO/SA are symmetric; (i, j) order is kept as-is.
    """
    a, b = manifest_record_to_indices(rec)
    names = {rec["i"]: rec["li"], rec["j"]: rec["lj"]}
    return a, b, names[a], names[b]


def gather_feats(d, task, a, b):
    """Task-specific role-aware graph embeddings for query slots (a, b) from
    query_slots(): a = first-named person A (the LOOKER for lah), b = person B.

    Returns (feats (K,256) float, role_ids (K,) long), K == TOK_COUNT[task], in the
    SAME order the prompt emits its <gtok> placeholders.

    Orientation: edge_pp[x, y] = E[x→y] = "x looks at y" (readout lah_logits[x, y];
    verified AUC 0.94 against the manifest answers via query_slots' mapping).
      EDGE_FWD := edge_pp[a, b]   (A looks at B — the asked direction)
      EDGE_BWD := edge_pp[b, a]   (B looks at A)
    """
    v_src = d["v_src"].float()
    v_tgt = d["v_tgt"].float()
    epp = d["edge_pp"].float()
    e_fwd = epp[a, b]           # A -> B (asked direction)
    e_bwd = epp[b, a]           # B -> A
    # Trailing SRC(A)/{TGT|SRC}(B) re-inject each person's node embedding at the point
    # their label is referenced in the question ("Is {A} <gtok> looking at {B} <gtok>?").
    if task == "lah":
        feats = [v_src[a], v_tgt[b], e_fwd, v_src[a], v_tgt[b]]
        roles = [ROLE["SRC"], ROLE["TGT"], ROLE["EDGE_FWD"], ROLE["SRC"], ROLE["TGT"]]
    elif task == "laeo":
        feats = [v_src[a], v_src[b], e_fwd, e_bwd, v_src[a], v_src[b]]
        roles = [ROLE["SRC"], ROLE["SRC"], ROLE["EDGE_FWD"], ROLE["EDGE_BWD"],
                 ROLE["SRC"], ROLE["SRC"]]
    elif task == "sa":
        # Shared attention is judged from each person's SCENE-gaze channel
        # (E[·→null_in]) — where their gaze lands in the scene — not the direct
        # person↔person edge, so the p2p edge tokens are intentionally omitted.
        nin = d["edge_null_in"].float()
        feats = [v_src[a], v_src[b], nin[a], nin[b], v_src[a], v_src[b]]
        roles = [ROLE["SRC"], ROLE["SRC"], ROLE["NULL_IN"], ROLE["NULL_IN"],
                 ROLE["SRC"], ROLE["SRC"]]
    else:
        raise ValueError(f"unknown task {task!r}")
    return torch.stack(feats), torch.tensor(roles, dtype=torch.long)


def gather_heatmaps(d, task, a, b):
    """Predicted gaze heatmaps for the queried persons, in the SAME order the
    prompt emits its <hmtok> placeholders. K == HM_COUNT[task].
      lah      : [hm[A]]        (only the looker's gaze target matters)
      laeo/sa  : [hm[A], hm[B]] (both people's gaze needed for mutual / shared)
    Returns (M, Hh, Ww) raw heatmap logits (encoded downstream by HeatmapEncoder)."""
    hm = d["gaze_heatmap"].float()   # (N, Hh, Ww) — graph-stage predicted heatmap
    if task == "lah":
        return hm[a:a + 1]
    return torch.stack([hm[a], hm[b]])


def gather_frame_feats(d, slots):
    """Per-person graph soft-tokens for the frame pipeline (one <gtok> per person).

    Each listed person k contributes their source node embedding v_src[k] (role SRC);
    scene-gaze / edge context enters at the READOUT head, not the prompt, so the prompt
    injection stays a compact one-token-per-person summary.

      slots : ordered valid slot indices (== prompt person order P1..PK)
    Returns (feats (K,256) float, role_ids (K,) long)."""
    v_src = d["v_src"].float()
    feats = torch.stack([v_src[k] for k in slots])
    roles = torch.full((len(slots),), ROLE["SRC"], dtype=torch.long)
    return feats, roles


def gather_frame_heatmaps(d, slots):
    """Per-person predicted gaze heatmap for the frame pipeline (one <hmtok> per person),
    in the SAME order as gather_frame_feats. Returns (K, Hh, Ww) raw heatmap logits."""
    hm = d["gaze_heatmap"].float()   # (N, Hh, Ww)
    return torch.stack([hm[k] for k in slots])


def gather_pair_edges(d, task, a, b):
    """Head-side edge features for a queried pair (a, b) in query_slots() orientation.
    Mirrors gather_feats' edge semantics but returns ONLY the readout-head edge tensors:
      lah : {"e_fwd": E[a→b]}
      laeo: {"e_fwd": E[a→b], "e_bwd": E[b→a]}
      sa  : {"nin_a": E[a→null_in], "nin_b": E[b→null_in]}
    """
    if task == "lah":
        epp = d["edge_pp"].float()
        return {"e_fwd": epp[a, b]}
    if task == "laeo":
        epp = d["edge_pp"].float()
        return {"e_fwd": epp[a, b], "e_bwd": epp[b, a]}
    if task == "sa":
        nin = d["edge_null_in"].float()
        return {"nin_a": nin[a], "nin_b": nin[b]}
    raise ValueError(f"unknown task {task!r}")


def graph_pair_logit(d, task, a, b):
    """Frozen graph logit for pair (a, b) in query_slots() orientation — the residual
    base whose sigmoid equals pair_belief's p (verified same direction as gather_feats)."""
    key = {"lah": "lah_logits", "laeo": "laeo_logits", "sa": "sa_logits"}[task]
    lg = d[key].float()
    if task == "lah":
        return float(lg[a, b])
    return float(0.5 * (lg[a, b] + lg[b, a]))   # laeo/sa already symmetric; avg is a no-op safety


def pair_belief(d, task, a, b):
    """Graph prior for query slots (a, b) — same orientation as gather_feats.

    Returns a dict of calibrated scalars for the prompt's belief sentence:
      lah : {"p": P(A looks at B), "ov": heatmap-mass of A inside B's box}
      laeo: {"p": symmetrised P(mutual gaze)}
      sa  : {"p": symmetrised P(shared attention)}
    """
    if task == "lah":
        p = torch.sigmoid(d["lah_logits"][a, b].float()).item()
        ov = float(d["overlap"][a, b]) if "overlap" in d else None
        return {"p": p, "ov": ov}
    key = "laeo_logits" if task == "laeo" else "sa_logits"
    lg = d[key].float()
    p = 0.5 * (torch.sigmoid(lg[a, b]) + torch.sigmoid(lg[b, a])).item()
    return {"p": p}


class GraphTokenProjector(nn.Module):
    """K 256-d graph vectors -> K soft tokens in the VLM hidden space, with a per-ROLE
    (not per-slot-index) type embedding so a given role means the same thing across the
    variable-length LAH/LAEO/SA token sets.

    Output = gain * LayerNorm(mlp(...)): LayerNorm pins each soft token to unit RMS and
    the learnable scalar gain (initialised at train start to the LM's text-embedding
    RMS) puts it on the same scale as real token embeddings in the residual stream —
    without this a randomly-initialised MLP's output norm is arbitrary and the injected
    tokens can be ignored or dominate."""
    def __init__(self, out_dim, in_dim=256, n_roles=N_ROLES, hidden=1024):
        super().__init__()
        self.role_emb = nn.Parameter(torch.zeros(n_roles, in_dim))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        self.out_norm = nn.LayerNorm(out_dim)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, feats, role_ids):    # feats (M, in_dim), role_ids (M,)
        return self.gain * self.out_norm(self.mlp(feats + self.role_emb[role_ids]))


class HeatmapEncoder(nn.Module):
    """Encode a predicted 64x64 gaze heatmap into ONE soft token in the VLM hidden
    space. A spatial-softmax normalises away magnitude (raw DPT logits) so the token
    encodes WHERE the gaze lands, scale-invariantly; a small conv stack + linear then
    pools it to out_dim. Output scale handled like GraphTokenProjector:
    gain * LayerNorm(...), gain initialised to the LM's text-embedding RMS."""
    def __init__(self, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),   # 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),  # 16x16
            nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),  # 8x8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),   # (M, hidden)
        )
        self.proj = nn.Linear(hidden, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, hm):    # hm (M, Hh, Ww) raw logits
        m, Hh, Ww = hm.shape
        x = torch.softmax(hm.reshape(m, -1).float(), dim=-1).reshape(m, 1, Hh, Ww)
        x = self.net(x.to(self.proj.weight.dtype))
        return self.gain * self.out_norm(self.proj(x))


def install_hook(lang_model):
    """Register the injection hook on the text model. Per batch, set (before forward)
      lang_model._gtok  = {'tokens': (ΣK, D), 'mask': (B, L) bool}   graph soft-tokens
      lang_model._hmtok = {'tokens': (ΣM, D), 'mask': (B, L) bool}   heatmap soft-tokens
    Either may be absent/None. Masks select disjoint placeholder token ids, filled
    row-major to match the sample-major flat concat from the collate."""
    def hook(module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None:
            return args, kwargs
        slots = [getattr(module, a, None) for a in ("_gtok", "_hmtok")]
        if any(s is not None for s in slots):
            emb = emb.clone()
            for data in slots:
                if data is not None:
                    emb[data["mask"]] = data["tokens"].to(emb.dtype).reshape(-1, emb.shape[-1])
            kwargs["inputs_embeds"] = emb
        return args, kwargs
    lang_model.register_forward_pre_hook(hook, with_kwargs=True)
