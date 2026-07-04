from __future__ import annotations
"""Auto-consolidated module (see ../README.md). Merged from: graph_text, graph_token."""

import math
import torch
import torch.nn as nn


"""Graph-text block for the graph-augmented VLM (CONCRETE NUMERIC style).

Turns the aligned V14.5 graph features (v14graph_{split}.pt[sid]) into a short text
block for the query pair (red=i, blue=j), giving the **useful GAZE / HEATMAP info the
graph provides, as concrete numbers** (no left/right words, no abstract phrasing).

Positions are deliberately NOT described — the overlay already shows them (red/blue
boxes). The text covers only what the image cannot convey, all as values:
  - red's predicted gaze direction as "right/left, ~N deg below/above horizontal" (the
    angle is the concrete value; a raw unit vector is hard for the VLM to ground, an
    angle+word is digestible). gaze_vecs is a unit vector, image coords, +x=right +y=down.
  - aim = cos(gaze, red->blue) computed here (NOT `align`/`dir_ij`: `dir_ij` is stored
    as blue->red, so the raw `align` is sign-flipped); +1 = aimed straight at the partner
  - gaze-heatmap landing on the partner's head (`overlap`); 0 = miss, 1 = direct hit
  - graph edge scores (sigmoid of lah/laeo/sa/null logits), incl. per-frame lists so the
    VLM (which sees ONE frame) gets the graph's temporal view
  - SA: where red's and blue's forward gaze rays intersect (gaze-derived point), if at all
  - focal in-degree: how many OTHERS the graph predicts look at blue (LAH)
The 256-d embeddings stay out (separate soft-token channel).
"""


def _sig(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def _center(bb):
    return (0.5 * (float(bb[0]) + float(bb[2])), 0.5 * (float(bb[1]) + float(bb[3])))


def _box(bb):
    return f"[{float(bb[0]):.2f},{float(bb[1]):.2f},{float(bb[2]):.2f},{float(bb[3]):.2f}]"


def _gaze_dir(g):
    """Unit gaze vector -> 'right/left, ~N deg below/above horizontal' (angle is the
    concrete value; the word grounds it). Image coords: +x=right, +y=DOWN."""
    dx, dy = float(g[0]), float(g[1])
    if abs(dx) < 0.15 and abs(dy) > 0.15:
        return "straight down" if dy > 0 else "straight up"
    horiz = "right" if dx >= 0 else "left"
    ang = math.degrees(math.atan2(dy, abs(dx)))     # >0 below horizontal, <0 above
    if abs(ang) < 7:
        return f"{horiz}, near horizontal"
    return f"{horiz}, ~{abs(ang):.0f} deg {'below' if ang > 0 else 'above'} horizontal"


def _valid_slots(gf):
    bb = gf["head_bboxes"].float()
    return [k for k in range(bb.shape[0])
            if (bb[k, 2] - bb[k, 0]) > 1e-4 and (bb[k, 3] - bb[k, 1]) > 1e-4]


def _toward(gv, ci, cj):
    """cos(red gaze, red->blue): +1 aimed straight at blue, -1 straight away."""
    rx, ry = cj[0] - ci[0], cj[1] - ci[1]
    n = math.hypot(rx, ry)
    if n < 1e-6:
        return 0.0
    return float(gv[0]) * rx / n + float(gv[1]) * ry / n


def _frames(gf, key, i, j):
    """Per-frame sigmoid edge scores as a comma list, or '' if unavailable."""
    fr = gf.get(key)
    if fr is None or fr.shape[0] < 2:
        return ""
    vals = [f"{_sig(fr[t, i, j]):.2f}" for t in range(fr.shape[0])]
    return ", ".join(vals)


def _focal_indegree(gf, j, valid, exclude):
    # NOTE orientation: lah_logits[a,b] = "b looks at a" (export comment is wrong; verified
    # by eval — orient [j,i] gives F1_LAH 0.82, [i,j] gives 0.42). So "p looks at j" = lah[j,p].
    lah = gf["lah_logits"]
    return sum(1 for p in valid if p not in exclude and float(lah[j, p]) > 0.0)


def _ray_meet(ci, gi, cj, gj):
    """Where red's and blue's forward gaze rays meet (gaze-derived point). Returns
    (x,y) if they cross in front of both and inside the frame, else None."""
    det = (-float(gi[0])) * float(gj[1]) - (-float(gi[1])) * float(gj[0])
    if abs(det) < 1e-6:
        return None
    bx, by = cj[0] - ci[0], cj[1] - ci[1]
    t = (bx * (-float(gj[1])) - by * (-float(gj[0]))) / det
    s = (float(gi[0]) * by - float(gi[1]) * bx) / det
    if t <= 0 or s <= 0:
        return None
    x, y = ci[0] + t * float(gi[0]), ci[1] + t * float(gi[1])
    if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
    return None


def graph_text_block(task, i, j, gf, li, lj, answer_blind=False):
    """One concrete-numeric graph-text block for query pair. li/lj = the overlay
    person labels (e.g. "P1","P2"); red=i=li, blue=j=lj. gf=v14graph[sid].
    answer_blind=True (VERITAS de-risk): expose only the graph's gaze EVIDENCE
    (gaze direction, gaze point, head boxes, in-frame-object/out-of-frame existence)
    and DROP every i->j relational verdict (edge score, per-frame, aim, overlap, focal),
    so the VLM cannot copy the graph's answer."""
    bb = gf["head_bboxes"].float()
    gv = gf["gaze_vecs"].float()
    overlap = gf["overlap"]
    lah, laeo, sa = gf["lah_logits"], gf["laeo_logits"], gf["sa_logits"]
    nout = gf["null_out_logits"]; nin = gf["null_in_logits"]
    valid = _valid_slots(gf)
    ci, cj = _center(bb[i]), _center(bb[j])  # used only for gaze aim
    gp = gf.get("gaze_point")                # (N,2) heatmap argmax, normalized xy
    src, tgt = f"{li} (red)", f"{lj} (blue)"

    def _pt(k):  # "predicted gaze point" phrase for slot k, '' if unavailable
        return f" ({float(gp[k, 0]):.2f},{float(gp[k, 1]):.2f})" if gp is not None else ""

    if answer_blind:
        # Structured EVIDENCE-only block (VERITAS): per-person gaze field + geometry + the i->j
        # alignment/overlap evidence. DROPPED verdicts: edge score, per-frame, null logits, focal.
        # (no head-yaw in v14graph -> gaze direction is the orientation cue we have.)
        head = f"[Scene] {len(valid)} people. "
        pi = (f"{src}: head box {_box(bb[i])}, gaze points {_gaze_dir(gv[i])}, "
              f"gaze landing at{_pt(i) or ' n/a'}. ")
        if task == "lah":
            pj = f"{tgt}: head box {_box(bb[j])}. "
            ev = (f"Gaze alignment ({src} toward {tgt}): {_toward(gv[i], ci, cj):+.2f}. "
                  f"Gaze-heatmap overlap with {tgt}'s head: {float(overlap[i, j]):.2f}.")
            return head + pi + pj + ev
        pj = (f"{tgt}: head box {_box(bb[j])}, gaze points {_gaze_dir(gv[j])}, "
              f"gaze landing at{_pt(j) or ' n/a'}. ")
        if task == "laeo":
            ev = (f"{src} gaze alignment toward {tgt}: {_toward(gv[i], ci, cj):+.2f}; "
                  f"{tgt} gaze alignment toward {src}: {_toward(gv[j], cj, ci):+.2f}. "
                  f"Heatmap overlap {li}->{lj}: {float(overlap[i, j]):.2f}; "
                  f"{lj}->{li}: {float(overlap[j, i]):.2f}.")
            return head + pi + pj + ev
        d = (f"Distance between the two gaze landing points: "
             f"{math.hypot(float(gp[i,0])-float(gp[j,0]), float(gp[i,1])-float(gp[j,1])):.2f}."
             if gp is not None else "")
        return head + pi + pj + d

    # head boxes in the SAME normalized [x1,y1,x2,y2] space as the gaze points, so the
    # VLM can numerically check whether a gaze point falls inside a head box.
    boxes = f"{src} head box {_box(bb[i])}; {tgt} head box {_box(bb[j])}. "

    if task == "lah":
        t = _toward(gv[i], ci, cj)
        focal = _focal_indegree(gf, j, valid, exclude={i, j})
        fr = _frames(gf, "lah_logits_frames", j, i)   # lah[a,b]="b looks at a" -> i->j is [j,i]
        fr_s = f" Per-frame {li}->{lj} scores: {fr}." if fr else ""
        return (boxes +
                f"{src} gaze points {_gaze_dir(gv[i])}; predicted gaze point{_pt(i)}. "
                f"Aim at {tgt} (cos of gaze vs {li}->{lj}): {t:+.2f} (+1=straight at {lj}, -1=opposite). "
                f"Gaze-heatmap landing on {tgt}'s head: {float(overlap[i, j]):.2f} (0=miss, 1=hit). "
                f"Graph edge score {li}->{lj}: {_sig(lah[j, i]):.2f}; "
                f"{li}->nobody/out-of-frame: {_sig(nout[i]):.2f}.{fr_s} "
                f"Other people predicted to look at {lj}: {focal}.")

    if task == "laeo":
        ti, tj = _toward(gv[i], ci, cj), _toward(gv[j], cj, ci)
        fr = _frames(gf, "laeo_logits_frames", i, j)
        fr_s = f" Per-frame mutual scores: {fr}." if fr else ""
        return (boxes +
                f"{src} gaze points {_gaze_dir(gv[i])} (gaze point{_pt(i)}); aim at {lj}: {ti:+.2f}; "
                f"gaze-heatmap landing on {lj}: {float(overlap[i, j]):.2f}. "
                f"{tgt} gaze points {_gaze_dir(gv[j])} (gaze point{_pt(j)}); aim at {li}: {tj:+.2f}; "
                f"gaze-heatmap landing on {li}: {float(overlap[j, i]):.2f}. "
                f"Graph edge scores: {li}->{lj} {_sig(lah[j, i]):.2f}, "
                f"{lj}->{li} {_sig(lah[i, j]):.2f}, "
                f"mutual {_sig(0.5 * (float(laeo[i, j]) + float(laeo[j, i]))):.2f}.{fr_s}")

    # SA — do the two predicted gaze POINTS (heatmap argmax) land in the same place?
    if gp is not None:
        d = math.hypot(float(gp[i, 0]) - float(gp[j, 0]), float(gp[i, 1]) - float(gp[j, 1]))
        pts = (f"{li}'s gaze point{_pt(i)}, {lj}'s gaze point{_pt(j)}; "
               f"distance between the two gaze points: {d:.2f} (small = same target). ")
    else:
        meet = _ray_meet(ci, gv[i], cj, gv[j])
        pts = (f"forward gaze rays intersect at ({meet[0]:.2f},{meet[1]:.2f}). " if meet
               else "forward gaze rays do not intersect in the frame. ")
    fr = _frames(gf, "sa_logits_frames", i, j)
    fr_s = f" Per-frame shared-attention scores: {fr}." if fr else ""
    return (boxes +
            f"{src} gaze points {_gaze_dir(gv[i])}; {tgt} gaze points {_gaze_dir(gv[j])}. "
            f"{pts}"
            f"Graph edge scores: shared-attention "
            f"{_sig(0.5 * (float(sa[i, j]) + float(sa[j, i]))):.2f}, "
            f"{li}->nobody {_sig(nout[i]):.2f}, {lj}->nobody {_sig(nout[j]):.2f}.{fr_s}")


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
