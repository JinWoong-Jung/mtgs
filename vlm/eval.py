from __future__ import annotations
"""VLM Stage-2 eval harness — locked to the same compute_metrics axis as the
graph baseline (F1_LAH / F1_LAEO / AP_SA via per-target-argmax, thr=0.5).

Two CLI subcommands:
  token    — LoRA + GraphTokenProjector soft-token inference -> compute_metrics
  blend    — soft-blend alpha-sweep using a cached feat + optional pvlm .pt

Public API (imported by vlm.train):
  build_mtgs_dicts(gtmeta_path, preds) -> list[dict]
  evaluate(samples, thr=0.5)           -> dict with F1_LAH/F1_LAEO/AP_SA etc.
  build_results(feat, pvlm, alpha)     -> list[dict]  (for blend scoring)
  score(res)                           -> {"F1_LAH", "F1_LAEO", "AP_SA"}
  add_vis_mask(feat)                   -> feat (in-place)
"""

import argparse
import glob
import io as _io
import itertools
import json
import logging
import re
import sys
from pathlib import Path

import torch
import torch.multiprocessing as _mp
from omegaconf import OmegaConf
from PIL import Image
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mtgs.performance.compute_metrics import CPU_Unpickler, compute
from vlm.cfg import QWEN
from vlm.injection import (
    GTOK,
    GraphTokenProjector,
    gather_feats,
    install_hook,
)
from vlm.patches import patch_qwen3vl_patch_embed
from vlm.overlay import display_labels
from vlm.prompt import TASKS, token_prompt

# ── Local constants (replaces sgg.hgr.train_phase1 / sgg.hgr.eval_softblend) ─

LOGIT = {"lah": "lah_logits", "laeo": "laeo_logits", "sa": "sa_logits"}
GT = {"lah": "lah_gt", "laeo": "laeo_gt", "sa": "sa_gt"}


# ── vis_mask helper ────────────────────────────────────────────────────────────

def add_vis_mask(feat):
    """vis_mask = head-box area > 0 (== v6_feat.vis_mask, verified 100%)."""
    for s, d in feat.items():
        bb = d["head_bboxes"].float()
        d["vis_mask"] = ((bb[:, 2] - bb[:, 0]) > 1e-4) & ((bb[:, 3] - bb[:, 1]) > 1e-4)
        if "sa_gt" not in d and "coatt_gt" in d:
            d["sa_gt"] = d["coatt_gt"]
    return feat


# ── blend helpers ──────────────────────────────────────────────────────────────

def orient(pvlm):
    """build_results scores pair (i,j) with graph sigmoid(lah[i,j]) = '_j_ looks at _i_'
    (verified [a,b]='b looks at a'). Our VLM stores P(i looks at j) at key (i,j), so for
    LAH we TRANSPOSE keys into the harness convention. laeo/sa symmetric -> unchanged."""
    out = {}
    for (sid, tk, i, j), pv in pvlm.items():
        out[(sid, tk, j, i) if tk == "lah" else (sid, tk, i, j)] = pv
    return out


def oracle_pvlm(feat, pvlm):
    """Per-pair ORACLE (perfect selective router): keep the VLM value ONLY where the
    graph is wrong AND the VLM is right; else graph stays. `pvlm` must already be in
    harness convention (LAH transposed via orient()). At cell (i,j): graph=sigmoid(
    lah[i,j]) for lah / symmetric avg for laeo,sa; GT=GT[tk][i,j]."""
    by_sid = {}
    for k, pv in pvlm.items():
        by_sid.setdefault(k[0], []).append((k, pv))
    out = {}
    for sid, items in by_sid.items():
        if sid not in feat:
            continue
        s = feat[sid]
        lg = {t: torch.sigmoid(s[LOGIT[t]].float()) for t in TASKS}
        gt = {t: s[GT[t]] for t in TASKS}
        for (sd, tk, i, j), pv in items:
            y = float(gt[tk][i, j])
            if y < 0:
                continue
            pg = float(lg[tk][i, j]) if tk == "lah" else float(0.5 * (lg[tk][i, j] + lg[tk][j, i]))
            if round(pg) != y and round(pv) == y:
                out[(sd, tk, i, j)] = pv
    return out


def dsn(i):
    """Dataset name from sample index (val-split hardcoded boundaries).
    Known limitation: only affects gazefollow/inout branches; headline F1/AP unchanged."""
    return ("childplay" if i < 1714
            else "videoattentiontarget" if i < 2380
            else "laeo" if i < 2657
            else "coatt")


def build_results(feat, pvlm, alpha):
    """p_final = (1-alpha)*p_graph + alpha*p_vlm on routed pairs (in pvlm), else p_graph.
    Extract contiguous k×k submatrices over visible slots vs (matches ab_eval).

    feat  : {sid: graph_cache_dict}  (already has vis_mask; call add_vis_mask first)
    pvlm  : {(sid, task, i, j): P(yes)}  already in harness convention (LAH transposed)
    alpha : blend weight for VLM (0 = graph-only, 1 = VLM-only)
    """
    import numpy as np

    # group pvlm by sid for speed
    by_sid = {}
    for (sd, tk, i, j), pv in pvlm.items():
        by_sid.setdefault(sd, []).append((tk, i, j, pv))

    res = []
    for sid, s in feat.items():
        vis = s["vis_mask"].bool()
        vs, _ = display_labels(vis)
        k = len(vs)
        if k < 2:
            continue
        idx = int(sid[6:])
        # Layout: slot 0 = DUMMY padding, slots 1..k = real visible people.
        # compute()'s LAEO `source!=0` exclusion drops the harmless dummy.
        K = k + 1
        vmap = {slot: a + 1 for a, slot in enumerate(vs)}   # real person -> 1..k
        P = {}
        for t in TASKS:
            m = np.zeros((K, K), dtype=np.float32)
            m[1:, 1:] = torch.sigmoid(s[LOGIT[t]][vs][:, vs].float()).numpy()
            P[t] = m
        for tk, i, j, pv in by_sid.get(sid, []):
            if i in vmap and j in vmap:
                a, b = vmap[i], vmap[j]
                P[tk][a, b] = (1 - alpha) * P[tk][a, b] + alpha * pv
        GTm = {}
        for t in TASKS:
            g = torch.full((K, K), -1.0)
            g[1:, 1:] = s[GT[t]][vs][:, vs].float()
            GTm[t] = g
        io = torch.full((K,), -1, dtype=torch.long)
        io[1:] = s["inout_gt"][vs].long()
        hb = torch.zeros((K, 4))
        hb[1:] = s["head_bboxes"][vs].float()
        pairs = list(itertools.permutations(range(K), 2))

        def vec(m, tr):
            return torch.tensor([m[b, a] if tr else m[a, b]
                                  for a, b in pairs]).float().unsqueeze(0)

        res.append({
            "dataset": [dsn(idx)],
            "gp_pred": torch.zeros(1, K, 2),
            "gp_gt": torch.zeros(1, K, 2),
            "inout_gt": io.unsqueeze(0),
            "inout_pred": torch.zeros(1, K),
            "head_bboxes": hb.unsqueeze(0),
            "lah_pred": vec(P["lah"], True),   "lah_gt": vec(GTm["lah"], True),
            "laeo_pred": vec(P["laeo"], False), "laeo_gt": vec(GTm["laeo"], False),
            "coatt_pred": vec(P["sa"], False),  "coatt_gt": vec(GTm["sa"], False),
        })
    return res


def score(res):
    """Run compute() and return {F1_LAH, F1_LAEO, AP_SA} from the locked harness."""
    buf = _io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(message)s"))
    lg = logging.getLogger("mtgs.performance.compute_metrics")
    old_handlers, old_level, old_prop = lg.handlers[:], lg.level, lg.propagate
    lg.handlers = [h]
    lg.setLevel(logging.INFO)
    lg.propagate = False
    try:
        compute(res, shuffle=False, thr=0.5)
    finally:
        lg.handlers, lg.level, lg.propagate = old_handlers, old_level, old_prop

    d = {}
    cur = None
    for ln in buf.getvalue().splitlines():
        s = ln.strip()
        if "LAEO" in s:
            cur = "LAEO"
        elif "LAH" in s:
            cur = "LAH"
        elif "CoAtt" in s:
            cur = "SA"
        elif s.startswith("F1") and "thr=" in s and cur in ("LAH", "LAEO"):
            d[f"F1_{cur}"] = float(s.split(":", 1)[1].split()[0])
        elif s.startswith("AP ") and ":" in s and cur == "SA":
            d["AP_SA"] = float(s.split(":", 1)[1].split()[0])
    return d


# ── stream / sample helpers ────────────────────────────────────────────────────

def load_stream(path):
    """Load the stream of per-sample dicts the MTGS test path writes."""
    out = []
    with open(path, "rb") as f:
        u = CPU_Unpickler(f)
        while True:
            try:
                out.append(u.load())
            except EOFError:
                break
    return out


def sample_key(sample):
    """Stable id for a sample: (dataset, center-frame path).

    `path` is the 5-frame temporal window; index 2 is the center (the frame the
    social labels are defined on)."""
    ds = sample["dataset"][0]
    center = sample["path"][2]
    center = center[0] if isinstance(center, (list, tuple)) else center
    return (ds, center)


def pair_order(num_people):
    """The ordered-pair layout the pred/gt vectors follow."""
    return list(itertools.permutations(range(num_people), 2))


def num_people_of(sample):
    return sample["head_bboxes"].shape[1]


def inject_vlm_scores(samples, preds_by_key, tasks=("lah", "laeo", "coatt")):
    """Overwrite {task}_pred in each sample with the VLM's per-pair P(yes).

    preds_by_key: {sample_key: {"lah": seq[L], "laeo": seq[L], "coatt": seq[L]}}
        L = num_people*(num_people-1), in pair_order().
    Returns the number of samples with no matching prediction (left as-is).
    """
    missing = 0
    for s in samples:
        k = sample_key(s)
        pr = preds_by_key.get(k)
        if pr is None:
            missing += 1
            continue
        L = num_people_of(s) * (num_people_of(s) - 1)
        for t in tasks:
            if t not in pr:
                continue
            vec = torch.as_tensor(pr[t], dtype=torch.float32).view(-1)
            assert vec.numel() == L, f"{t} for {k}: expected {L} pairs, got {vec.numel()}"
            s[f"{t}_pred"] = vec.view(1, L)
    return missing


# ── LOCKED evaluate() harness (verbatim from peer sgg/eval.py lines 215-257) ──

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


def format_metrics(m, title=""):
    """Human-readable Main + Detail block for a metrics dict from evaluate()."""
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    lines = []
    if title:
        lines.append(f"===== {title} =====")
    lines.append(f"[MAIN]   F1_LAH={f(m.get('F1_LAH'))}  F1_LAEO={f(m.get('F1_LAEO'))}  "
                 f"AP_SA={f(m.get('AP_SA'))}")
    lines.append(f"[DETAIL] LAH  AP={f(m.get('LAH_AP'))} AUC={f(m.get('LAH_AUC'))} | "
                 f"LAEO AP={f(m.get('LAEO_AP'))} AUC={f(m.get('LAEO_AUC'))} | "
                 f"SA AP={f(m.get('SA_AP'))} AUC={f(m.get('SA_AUC'))}")
    lines.append(f"[DETAIL] Dist={f(m.get('Dist'))}  AP_IO={f(m.get('AP_IO'))}  "
                 f"F1_LAH(PP)={f(m.get('F1_LAH_PP'))}  F1_LAEO(PP)={f(m.get('F1_LAEO_PP'))}")
    if m.get("detail"):
        lines.append("----- compute() full breakdown -----")
        lines.append(m["detail"])
    return "\n".join(lines)


# ── build_mtgs_dicts ───────────────────────────────────────────────────────────

def build_mtgs_dicts(gtmeta_path, preds):
    """Phase 2: per-sample MTGS dicts from the render-pass gtmeta (GT/bbox/inout)
    + the VLM P(yes). Reads gtmeta (authoritative, written in the same pass as the
    overlays) — never re-iterates the dataset (whose __getitem__ is RNG/worker
    dependent and would diverge from what the VLM was evaluated on).

    preds: {(sid, task, i, j): P(yes)}  LAEO/SA keys with i<j (canonical).
    """
    gtmeta = torch.load(gtmeta_path, weights_only=False)
    out = []
    for sid, m in gtmeta.items():
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


# ── token eval dataset / collate ──────────────────────────────────────────────

_mp.set_sharing_strategy("file_system")


class _TokenRecDS(Dataset):
    def __init__(self, recs, overlay_dir, gf):
        self.recs = recs
        self.dir = Path(overlay_dir)
        self.gf = gf

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, k):
        r = self.recs[k]
        gfd = self.gf[r["sid"]]
        bb = gfd["head_bboxes"]
        pil = Image.open(self.dir / r["sid"] / "frame.png").convert("RGB")
        prompt = token_prompt(r["task"], r["li"], r["lj"], bb[r["i"]], bb[r["j"]])
        feats, roles = gather_feats(gfd, r["task"], r["i"], r["j"])
        return (r["sid"], r["task"], r["i"], r["j"]), pil, prompt, feats, roles


def _coll(b):
    keys, pils, prompts, feats, roles = zip(*b)
    return (list(keys), list(pils), list(prompts),
            torch.cat(feats, dim=0), torch.cat(roles, dim=0))


# ── token CLI main ─────────────────────────────────────────────────────────────

def _main_eval_lora_token():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="",
                    help="checkpoint dir; empty -> derive from config experiment + --which")
    ap.add_argument("--which", default="best", choices=["best", "last"],
                    help="which trained checkpoint to eval when --ckpt is not given")
    ap.add_argument("--manifest", required=True, help="manifest_nograph_<split>.jsonl")
    ap.add_argument("--overlay_dir", required=True, help="vlm_overlays/<split>")
    ap.add_argument("--graph_feats", required=True, help="v14graph_<split>.pt")
    ap.add_argument("--gtmeta", required=True, help="gtmeta_<split>.pt")
    ap.add_argument("--config", default="mtgs/config/config_vlm.yaml",
                    help="config YAML; supplies eval.vlm_bs / eval.num_workers + experiment path")
    ap.add_argument("--vlm_bs", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--preds_out", default="")
    ap.add_argument("--compute_from", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Config supplies test batch/workers (CLI overrides win) and, when --ckpt is empty,
    # the trained checkpoint location: <out_root>/<name>/train/checkpoints/<which>.
    full_cfg = OmegaConf.load(args.config) if Path(args.config).exists() else None
    ecfg = full_cfg.get("eval") if full_cfg is not None else None
    if args.vlm_bs is None:
        args.vlm_bs = int(ecfg.vlm_bs) if ecfg is not None else 64
    if args.num_workers is None:
        args.num_workers = int(ecfg.num_workers) if ecfg is not None else 10
    if not args.ckpt:
        assert full_cfg is not None and "experiment" in full_cfg, \
            "no --ckpt and no experiment section in config"
        xc = full_cfg.experiment
        args.ckpt = str(Path(str(xc.out_root)) / str(xc.name) / "train" / "checkpoints" / args.which)
        print(f"[eval] ckpt (from config, which={args.which}): {args.ckpt}", flush=True)

    # Fixed-shape (448x448) inputs -> let cuDNN pick fast kernels; TF32 on residual fp32
    # matmuls. Free speedups, no effect on the bf16 model's outputs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.compute_from:
        preds = {}
        for f in sorted(glob.glob(args.compute_from)):
            preds.update(torch.load(f, weights_only=False))
        m = evaluate(build_mtgs_dicts(args.gtmeta, preds))
        print(f"\n===== compute_from {args.compute_from} ({len(preds)} preds) =====", flush=True)
        print(f"[RESULT] F1_LAH={m['F1_LAH']:.4f}  F1_LAEO={m['F1_LAEO']:.4f}  "
              f"AP_SA={m['AP_SA']:.4f}", flush=True)
        return

    proc = AutoProcessor.from_pretrained(QWEN)
    proc.tokenizer.padding_side = "left"
    proc.tokenizer.add_special_tokens({"additional_special_tokens": [GTOK]})
    gtok_id = proc.tokenizer.convert_tokens_to_ids(GTOK)
    yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = proc.tokenizer.encode("no", add_special_tokens=False)[0]
    base = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16,
                                                           device_map="cuda")
    base.resize_token_embeddings(len(proc.tokenizer))
    patch_qwen3vl_patch_embed(base)   # Blackwell slow_conv_dilated3d bypass (~48x fwd speedup)
    D = base.config.text_config.hidden_size
    model = PeftModel.from_pretrained(base, args.ckpt).merge_and_unload().eval()
    proj = GraphTokenProjector(out_dim=D).to("cuda", torch.bfloat16)
    proj.load_state_dict(torch.load(Path(args.ckpt) / "projector.pt", weights_only=True))
    proj.eval()
    lm = model.model.language_model
    install_hook(lm)

    recs = [json.loads(l) for l in open(args.manifest)]
    if args.nshards > 1:
        recs = recs[args.shard::args.nshards]
    gf = torch.load(args.graph_feats, weights_only=False)
    from vlm.vision_cache import run_token_eval_grouped
    preds = run_token_eval_grouped(model, proc, proj, lm, recs, args.overlay_dir, gf,
                                   gtok_id, yes_id, no_id, "cuda")

    out_path = args.preds_out or f"preds_token_{Path(args.ckpt).name}.pt"
    torch.save(preds, out_path)
    print(f"saved {len(preds)} preds -> {out_path}", flush=True)
    if args.nshards == 1:
        m = evaluate(build_mtgs_dicts(args.gtmeta, preds))
        print("\n" + format_metrics(m, title=f"{args.ckpt} ({len(preds)} preds)"), flush=True)


# ── blend CLI main ─────────────────────────────────────────────────────────────

def _main_eval_blend():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="graph feat cache .pt (v14graph_<split>.pt)")
    ap.add_argument("--pvlm", default="", help="VLM preds .pt; empty -> graph-only")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    feat = add_vis_mask(torch.load(args.feat, map_location="cpu", weights_only=False))
    pvlm = {}
    if args.pvlm:
        pvlm = orient(torch.load(args.pvlm, map_location="cpu", weights_only=False))
        pvlm = {k: v for k, v in pvlm.items() if k[0] in feat}
    print(f"[blend] samples={len(feat)}  pvlm_pairs={len(pvlm)}", flush=True)

    print(f"\n{'config':>22} {'F1_LAH':>8} {'F1_LAEO':>8} {'AP_SA':>8}")
    g = score(build_results(feat, {}, 0.0))
    print(f"{'(a) graph-only':>22} {g['F1_LAH']:>8.4f} {g['F1_LAEO']:>8.4f} {g['AP_SA']:>8.4f}")
    best = None   # (mean_of_3, alpha)
    for a in [float(x) for x in args.alphas.split(",")]:
        if a == 0 or not pvlm:
            continue
        m = score(build_results(feat, pvlm, a))
        print(f"{'blend a='+format(a,'.2f'):>22} {m['F1_LAH']:>8.4f} {m['F1_LAEO']:>8.4f} {m['AP_SA']:>8.4f}")
        mean3 = (m['F1_LAH'] + m['F1_LAEO'] + m['AP_SA']) / 3
        if best is None or mean3 > best[0]:
            best = (mean3, a)
    if pvlm:
        orac = oracle_pvlm(feat, pvlm)
        mo = score(build_results(feat, orac, 1.0))
        print(f"{'ORACLE ceiling':>22} {mo['F1_LAH']:>8.4f} {mo['F1_LAEO']:>8.4f} {mo['AP_SA']:>8.4f}  "
              f"(perfect router, {len(orac)} overrides)")

    # Full Main + Detail breakdown for the headline configs (graph-only + best blend).
    print("\n" + format_metrics(evaluate(build_results(feat, {}, 0.0)),
                                 title="graph-only"), flush=True)
    if best is not None:
        print("\n" + format_metrics(evaluate(build_results(feat, pvlm, best[1])),
                                    title=f"best blend a={best[1]:.2f} (by mean of 3)"), flush=True)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _CMDS = {
        "blend":   _main_eval_blend,
        "token":   _main_eval_lora_token,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        sys.exit("usage: python -m vlm.eval {" + "|".join(_CMDS) + "} [args]")
    _cmd = sys.argv.pop(1)
    _CMDS[_cmd]()
