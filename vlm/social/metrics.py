from __future__ import annotations
"""Locked social-gaze metric harness for the pair VLM pipeline.

Extracted verbatim from the legacy ``vlm/eval.py`` (which mixed this live harness
with the now-removed soft-token / blend evaluation code). The pair pipeline reaches
the repository's authoritative ``mtgs.performance.compute_metrics.compute`` axis
(F1_LAH / F1_LAEO / AP_SA via per-target argmax, thr=0.5) exclusively through these
two functions, called from :func:`vlm.social.evaluation.evaluate_predictions`.

Public API:
  build_mtgs_dicts(gtmeta_path, preds, restrict_sids=None) -> list[dict]
  evaluate(samples, thr=0.5)                               -> dict
"""

import io as _io
import itertools
import logging
import re

import torch

from mtgs.performance.compute_metrics import compute


# ── LOCKED evaluate() harness (verbatim from the peer sgg/eval.py lines 215-257) ──

def evaluate(samples, thr=0.5):
    """Run compute() and return a metrics dict.

    Uses compute()'s RETURN dict for the authoritative per-task AP/AUC + PP/gaze
    values (lah_ap/lah_auc/laeo_ap/laeo_auc/coatt_ap/coatt_auc/dist/ap_io/...),
    and parses the per-target F1 (LAH/LAEO, thr=0.5) from the log (compute() does
    not return those). Keys:
      MAIN:   F1_LAH, F1_LAEO, AP_SA
      PERTASK AP/AUC: {LAH,LAEO,SA}_{AP,AUC}
      EXTRA:  F1_LAH_PP, F1_LAEO_PP, Dist, AP_IO
      detail: full compute() text breakdown
    """
    buf = _io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    lg = logging.getLogger("mtgs.performance.compute_metrics")
    old = (lg.handlers, lg.level, lg.propagate)
    lg.handlers, lg.level, lg.propagate = [handler], logging.INFO, False
    try:
        ret = compute(samples, shuffle=False, thr=thr) or {}
    finally:
        lg.handlers, lg.level, lg.propagate = old

    # per-target F1 (LAH/LAEO) is only logged, not returned -> parse it (require the
    # "thr=" tag so we don't pick up the "F1 PP (geometric)" line in the same section)
    def num(s):
        mm = re.search(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", s)
        return float(mm.group()) if mm else None
    f1 = {"LAH": None, "LAEO": None}
    section = None
    for line in buf.getvalue().splitlines():
        s = line.strip()
        if s.startswith("----- LAEO"):
            section = "LAEO"
        elif s.startswith("----- LAH"):
            section = "LAH"
        elif s.startswith("----- CoAtt"):
            section = "SA"
        elif s.startswith("F1 ") and "thr=" in s and section in ("LAH", "LAEO"):
            f1[section] = num(s)

    out = {
        "F1_LAH": f1["LAH"], "F1_LAEO": f1["LAEO"], "AP_SA": ret.get("coatt_ap"),
        "LAH_AP": ret.get("lah_ap"),   "LAH_AUC": ret.get("lah_auc"),
        "LAEO_AP": ret.get("laeo_ap"), "LAEO_AUC": ret.get("laeo_auc"),
        "SA_AP": ret.get("coatt_ap"),  "SA_AUC": ret.get("coatt_auc"),
        "F1_LAH_PP": ret.get("f1_lah_pp"), "F1_LAEO_PP": ret.get("f1_laeo_pp"),
        "Dist": ret.get("dist"), "AP_IO": ret.get("ap_io"),
        "detail": buf.getvalue().rstrip(),
    }
    return out


# ── build_mtgs_dicts ───────────────────────────────────────────────────────────

def build_mtgs_dicts(gtmeta_path, preds, restrict_sids=None):
    """Phase 2: per-sample MTGS dicts from the render-pass gtmeta (GT/bbox/inout)
    + the VLM P(yes). Reads gtmeta (authoritative, written in the same pass as the
    overlays) — never re-iterates the dataset (whose __getitem__ is RNG/worker
    dependent and would diverge from what the VLM was evaluated on).

    preds: {(sid, task, i, j): P(yes)}  LAEO/SA keys with i<j (canonical).
    restrict_sids: optional sid set — evaluate only these frames (used by the
    in-training val subset; frames without preds would otherwise score 0 everywhere).
    """
    gtmeta = torch.load(gtmeta_path, weights_only=False)
    out = []
    for sid, m in gtmeta.items():
        if restrict_sids is not None and sid not in restrict_sids:
            continue
        bb = m["head_bboxes"].float()
        n = bb.shape[0]
        pairs = list(itertools.permutations(range(n), 2))
        L = len(pairs)
        lah_pred = torch.zeros(L)
        laeo_pred = torch.zeros(L)
        coatt_pred = torch.zeros(L)
        for q, (i, j) in enumerate(pairs):
            p = preds.get((sid, "lah", i, j))
            if p is not None:
                lah_pred[q] = p
            lo, hi = (i, j) if i < j else (j, i)
            p = preds.get((sid, "laeo", lo, hi))
            if p is not None:
                laeo_pred[q] = p
            p = preds.get((sid, "sa", lo, hi))
            if p is not None:
                coatt_pred[q] = p
        out.append({
            "head_bboxes":  bb.unsqueeze(0),
            "lah_pred":     lah_pred.unsqueeze(0),
            "lah_gt":       m["lah_gt"].long().unsqueeze(0),
            "laeo_pred":    laeo_pred.unsqueeze(0),
            "laeo_gt":      m["laeo_gt"].long().unsqueeze(0),
            "coatt_pred":   coatt_pred.unsqueeze(0),
            "coatt_gt":     m["coatt_gt"].long().unsqueeze(0),
            "inout_gt":     m["inout"].float().unsqueeze(0),
            "dataset":      [m["dataset"]],
        })
    return out
