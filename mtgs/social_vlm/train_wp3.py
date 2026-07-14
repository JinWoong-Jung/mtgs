"""WP3 — external residual gate: train ONLY the PairResidualDecoder on cached frozen
features (graph bundle + Qwen scene memory). No Qwen/graph forward during training →
fast. Evaluates against the graph-only baseline through the SAME compute_metrics harness.

Go/No-Go: if this does not beat graph-only on val social_ap, do NOT proceed to WP4/5 —
analyse complementarity first (spec WP3 note).

Run: python -m mtgs.social_vlm.train_wp3 [--epochs 8] [--lr 3e-3]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import torch
from torch.utils.data import DataLoader, Dataset

from mtgs.social_vlm.conventions import manifest_record_to_indices
from mtgs.social_vlm.feature_bundle import bundle_from_cache_entry
from mtgs.social_vlm.residual_decoder import PairResidualDecoder
from vlm.eval import build_mtgs_dicts, evaluate

C = "/home/jinwoongjung/MTGS/data/vlm_feature"
TASKS = ("lah", "laeo", "sa")
LOGIT_KEY = {"lah": "lah_logits", "laeo": "laeo_logits", "sa": "sa_logits"}


class FrameSet(Dataset):
    """One item = one frame: (graph bundle-center dict, scene memory [K,D], GT matrices)."""

    def __init__(self, split):
        prefix = os.environ.get("SCENE_PREFIX", "scene_mem")   # "scene_dino" for ablation
        self.gf = torch.load(f"{C}/vlmgraph_{split}.pt", weights_only=False)
        self.mem = torch.load(f"{C}/{prefix}_{split}.pt", weights_only=False)
        self.gt = torch.load(f"{C}/gtmeta_{split}.pt", weights_only=False)
        self.sids = [s for s in self.mem if s in self.gf and s in self.gt]

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, k):
        sid = self.sids[k]
        b = bundle_from_cache_entry(sid, self.gf[sid]).center()   # each [1,...]
        b = {kk: v[0] for kk, v in b.items()}                     # drop batch dim -> [...]
        N = b["lah_logits"].shape[0]
        # GT matrices [N,N] from permutation-order pair vectors (looker,target convention)
        m = self.gt[sid]
        pairs = list(itertools.permutations(range(N), 2))
        gt = {t: -torch.ones(N, N) for t in TASKS}
        for q, (i, j) in enumerate(pairs):
            gt["lah"][j, i] = float(m["lah_gt"][q])               # dataset (i,j)="j looks at i"
            gt["laeo"][i, j] = float(m["laeo_gt"][q])
            gt["sa"][i, j] = float(m["coatt_gt"][q])
        return sid, b, self.mem[sid].float(), gt


def _pad_edge(e, N, Nmax):  # e: [N, N+2, De] -> [Nmax, Nmax+2, De]
    out = torch.zeros(Nmax, Nmax + 2, e.shape[-1], dtype=e.dtype)
    out[:N, :N] = e[:, :N]                       # person-person
    out[:N, Nmax] = e[:, N]                       # null_in
    out[:N, Nmax + 1] = e[:, N + 1]              # null_out
    return out


def collate(batch):
    sids = [b[0] for b in batch]
    Ns = [b[1]["lah_logits"].shape[0] for b in batch]
    Nmax = max(Ns)

    def padmat(x, N):  # [N,N,..] -> [Nmax,Nmax,..]
        pad = torch.zeros(Nmax, Nmax, *x.shape[2:], dtype=x.dtype)
        pad[:N, :N] = x
        return pad

    def padvec(x, N):  # [N,..] -> [Nmax,..]
        pad = torch.zeros(Nmax, *x.shape[1:], dtype=x.dtype)
        pad[:N] = x
        return pad

    out_b = {}
    for kk in ("lah_logits", "laeo_logits", "sa_logits", "alignment", "overlap", "pair_mask"):
        out_b[kk] = torch.stack([padmat(b[1][kk], Ns[bi]) for bi, b in enumerate(batch)])
    out_b["v_src"] = torch.stack([padvec(b[1]["v_src"], Ns[bi]) for bi, b in enumerate(batch)])
    out_b["v_tgt"] = torch.stack([  # [N+2,De] -> [Nmax+2,De] (persons padded, nulls kept last two)
        torch.cat([padvec(b[1]["v_tgt"][:-2], Ns[bi]), b[1]["v_tgt"][-2:]])
        for bi, b in enumerate(batch)])
    out_b["edge_states"] = torch.stack([_pad_edge(b[1]["edge_states"], Ns[bi], Nmax)
                                        for bi, b in enumerate(batch)])
    mem = torch.stack([b[2] for b in batch])
    gts = {t: torch.stack([padmat(b[3][t], Ns[bi]) for bi, b in enumerate(batch)]) for t in TASKS}
    return sids, out_b, mem, gts, torch.tensor(Ns)


def evaluate_split(dec, split, device, bs=32):
    ds = FrameSet(split)
    dl = DataLoader(ds, batch_size=bs, num_workers=6, collate_fn=collate)
    finals = {}   # sid -> {task: sigmoid(final)[N,N]}  (final indexed [looker,target])
    dec.eval()
    with torch.no_grad():
        for sids, b, mem, gts, Ns in dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = dec(b, mem.to(device))
            for bi, sid in enumerate(sids):
                N = int(Ns[bi])
                finals[sid] = {t: torch.sigmoid(out[t]["final"][bi, :N, :N]).cpu() for t in TASKS}
    dec.train()
    # Emit preds ONLY for manifest records (labelled pairs) — matches graph_only / the
    # frame pipeline. Emitting all ordered pairs pollutes the per-target-argmax F1.
    preds = {}
    for line in open(f"{C}/manifest_{split}.jsonl"):
        r = json.loads(line)
        if r["sid"] not in finals:
            continue
        looker, target = manifest_record_to_indices(r)
        preds[(r["sid"], r["task"], r["i"], r["j"])] = float(finals[r["sid"]][r["task"]][looker, target])
    m = evaluate(build_mtgs_dicts(f"{C}/gtmeta_{split}.pt", preds, restrict_sids=set(finals)))
    aps = [m.get(k) for k in ("LAH_AP", "LAEO_AP", "SA_AP")]
    m["social_ap"] = sum(aps) / 3 if all(a is not None for a in aps) else None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)     # moderate: high lr collapses the gate
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--d_mem", type=int, default=4096)
    ap.add_argument("--lam_gate", type=float, default=0.0)  # gate-open penalty: >0 keeps gate
    ap.add_argument("--tag", type=str, default="best")      # closed unless VLM truly helps (LAH/SA)
    args = ap.parse_args()
    device = "cuda"

    tr = FrameSet("train")
    print(f"[wp3] train frames={len(tr)}", flush=True)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=8, collate_fn=collate)
    dec = PairResidualDecoder(d_edge=256, d_model=256, d_mem=args.d_mem).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * max(1, len(dl)))
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    base = evaluate_split(dec, "val", device)   # decoder at init == graph-only
    print(f"[wp3] graph-only(val via decoder init): social_ap={base['social_ap']:.4f} "
          f"F1_LAH={base['F1_LAH']:.4f} F1_LAEO={base['F1_LAEO']:.4f} AP_SA={base['AP_SA']:.4f}", flush=True)

    best = None
    for ep in range(args.epochs):
        dec.train()
        run = n = 0
        for sids, b, mem, gts, Ns in dl:
            b = {k: v.to(device) for k, v in b.items()}
            gts = {t: gts[t].to(device) for t in TASKS}
            mem = mem.to(device)
            out = dec(b, mem)
            loss = 0.0
            for t in TASKS:
                y = gts[t]
                mask = (y >= 0)                       # -1 = no label; also excludes padded/self
                if mask.any():
                    loss = loss + bce(out[t]["final"][mask], y[mask])
                    # residual-magnitude penalty (only on labelled pairs): penalise the
                    # EFFECTIVE correction sigmoid(gate)*delta applied on top of the graph
                    # logit — NOT gate alone (delta blows up to game a gate-only penalty).
                    # graph-strong tasks (LAH/SA) then keep the residual ~0 unless it pays.
                    if args.lam_gate > 0:
                        res = torch.sigmoid(out[t]["gate"][mask]) * out[t]["delta"][mask]
                        loss = loss + args.lam_gate * res.abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step(); sched.step()
            run += float(loss); n += 1
        m = evaluate_split(dec, "val", device)
        gate = {t: torch.sigmoid(out[t]["gate"]).mean().item() for t in TASKS}
        print(f"[wp3] ep{ep} loss={run/max(n,1):.4f} VAL social_ap={m['social_ap']:.4f} "
              f"F1_LAH={m['F1_LAH']:.4f} F1_LAEO={m['F1_LAEO']:.4f} AP_SA={m['AP_SA']:.4f} "
              f"gate~{ {t: round(gate[t],3) for t in TASKS} }", flush=True)
        if best is None or m["social_ap"] > best:
            best = m["social_ap"]
            torch.save(dec.state_dict(), f"{C}/wp3_decoder_{args.tag}.pt")
    print(f"\n[wp3] GATE: graph-only social_ap={base['social_ap']:.4f} -> best decoder "
          f"social_ap={best:.4f}  (delta={best-base['social_ap']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
