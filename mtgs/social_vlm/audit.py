"""WP0 — baseline audit. Records provenance (git commit, checkpoint SHA256, resolved
config, package versions) and reproduces graph-only val/test metrics from the existing
feature cache (no re-extraction). Writes a JSON audit record.

Run: python -m mtgs.social_vlm.audit [--split val|test] [--ckpt <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

C = "/home/jinwoongjung/MTGS/data/vlm_feature"
DEFAULT_CKPT = "/home/jinwoongjung/MTGS/experiments/V18/train/checkpoints/best.ckpt"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd="/home/jinwoongjung/MTGS",
                                       text=True).strip()
    except Exception as e:
        return f"<git failed: {e}>"


def _versions():
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"}


def graph_only_metrics(split):
    """Graph-only preds (sigmoid(graph_logit)) via the SAME eval harness -> metrics.
    Also returns the per-sample logits dict for archival (LAH/LAEO/SA + labels)."""
    import math
    from vlm.injection import query_slots, graph_pair_logit
    from vlm.eval import build_mtgs_dicts, evaluate
    gf = torch.load(f"{C}/vlmgraph_{split}.pt", weights_only=False)
    preds = {}
    for line in open(f"{C}/manifest_{split}.jsonl"):
        r = json.loads(line)
        if r["sid"] not in gf:
            continue
        a, b, _, _ = query_slots(r)
        preds[(r["sid"], r["task"], r["i"], r["j"])] = \
            1.0 / (1.0 + math.exp(-graph_pair_logit(gf[r["sid"]], r["task"], a, b)))
    m = evaluate(build_mtgs_dicts(f"{C}/gtmeta_{split}.pt", preds,
                                  restrict_sids={k[0] for k in preds}))
    keep = ("F1_LAH", "F1_LAEO", "AP_SA", "LAH_AP", "LAH_AUC", "LAEO_AP", "LAEO_AUC",
            "SA_AP", "SA_AUC")
    return {k: m.get(k) for k in keep}, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="/home/jinwoongjung/MTGS/mtgs/social_vlm/audit_record.json")
    ap.add_argument("--save_logits", action="store_true",
                    help="also dump graph-only per-sample preds for archival")
    args = ap.parse_args()

    print(f"[audit] hashing checkpoint {args.ckpt} ...", flush=True)
    rec = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "checkpoint": args.ckpt,
        "checkpoint_sha256": _sha256(args.ckpt) if Path(args.ckpt).exists() else None,
        "versions": _versions(),
        "split": args.split,
    }
    print(f"[audit] reproducing graph-only {args.split} metrics from cache ...", flush=True)
    metrics, preds = graph_only_metrics(args.split)
    rec["graph_only_metrics"] = metrics
    rec["social_ap"] = (sum(metrics[k] for k in ("LAH_AP", "LAEO_AP", "SA_AP")) / 3
                        if all(metrics[k] is not None for k in ("LAH_AP", "LAEO_AP", "SA_AP"))
                        else None)

    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\n[audit] written -> {args.out}", flush=True)
    if args.save_logits:
        p = f"{C}/graph_only_{args.split}_preds.pt"
        torch.save(preds, p)
        print(f"[audit] graph-only preds -> {p}", flush=True)


if __name__ == "__main__":
    main()
