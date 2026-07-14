from __future__ import annotations
"""Frame-level dataset for VLM Stage-2 (frame pipeline).

One example == one FRAME (not one pair). The frame is described once, with per-person
graph (<gtok>) + gaze-heatmap (<hmtok>) soft-tokens and a per-person anchor (<panc>);
ALL of the frame's queried pairs are supervised from the single shared VLM forward via
PairwiseSocialHead. Cuts forward count ~N× versus the per-pair TokenDS.

Reuses the exact graph feats / plain frame.png / manifest / gtmeta artifacts already
produced by graph_extract.sh — no re-extraction needed.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.injection import (
    gather_frame_feats,
    gather_frame_heatmaps,
    gather_pair_edges,
    graph_pair_logit,
    pair_belief,
    query_slots,
)
from vlm.overlay import build_frame_overlay, display_labels
from vlm.prompt import frame_prompt


def _valid_slots(bb):
    """Valid person slots == head-box area > 1e-4 (matches data_prep / add_vis_mask)."""
    ok = ((bb[:, 2] - bb[:, 0]) > 1e-4) & ((bb[:, 3] - bb[:, 1]) > 1e-4)
    return [k for k in range(bb.shape[0]) if bool(ok[k])]


class FrameDS(Dataset):
    def __init__(self, manifest, overlay_dir, graph_feats):
        recs = [json.loads(l) for l in open(manifest)]
        self.dir = Path(overlay_dir)
        self.gf = torch.load(graph_feats, weights_only=False)
        # Group manifest records by frame (sid), preserving order.
        by_sid = defaultdict(list)
        for r in recs:
            by_sid[r["sid"]].append(r)
        # Keep only frames present in the graph cache (defensive).
        self.sids = [s for s in by_sid if s in self.gf]
        self.by_sid = by_sid
        self._n_records = sum(len(by_sid[s]) for s in self.sids)

    def __len__(self):
        return len(self.sids)

    @property
    def num_records(self):
        return self._n_records

    def sample_weights(self, hard_floor=None):
        """Per-FRAME weights. Each record gets a (task, ans)-balanced weight, optionally
        scaled by graph hardness (hard_floor + |gt - p_graph|); the frame weight is the
        MEAN over its records so rare/hard pairs pull whole frames up without letting
        crowded frames dominate purely by pair count."""
        cnt = Counter((r["task"], r["ans"]) for s in self.sids for r in self.by_sid[s])
        w = torch.empty(len(self.sids))
        for idx, sid in enumerate(self.sids):
            d = self.gf[sid]
            acc = 0.0
            recs = self.by_sid[sid]
            for r in recs:
                bal = 1.0 / cnt[(r["task"], r["ans"])]
                if hard_floor is None:
                    acc += bal
                else:
                    a, b, _, _ = query_slots(r)
                    p = pair_belief(d, r["task"], a, b)["p"]
                    y = 1.0 if r["ans"] == "yes" else 0.0
                    acc += bal * (float(hard_floor) + abs(y - p))
            w[idx] = acc / max(len(recs), 1)
        return w

    def __getitem__(self, k):
        sid = self.sids[k]
        d = self.gf[sid]
        bb = d["head_bboxes"].float()
        slots = _valid_slots(bb)
        slot2local = {s: i for i, s in enumerate(slots)}
        _, lab = display_labels(torch.tensor([s in slot2local for s in range(bb.shape[0])]))
        labels = [lab[s] for s in slots]
        boxes = [bb[s] for s in slots]

        pil = Image.open(self.dir / sid / "frame.png").convert("RGB")
        pil = build_frame_overlay(pil, slots, bb, lab)
        prompt = frame_prompt(labels, boxes)
        gfeats, groles = gather_frame_feats(d, slots)          # (K,256),(K,)
        hms = gather_frame_heatmaps(d, slots)                  # (K,Hh,Ww)

        records = []
        for r in self.by_sid[sid]:
            a, b, _, _ = query_slots(r)                        # original slots
            if a not in slot2local or b not in slot2local:
                continue
            records.append({
                "task": r["task"],
                "la": slot2local[a], "lb": slot2local[b],      # local anchor indices
                "edges": gather_pair_edges(d, r["task"], a, b),
                "glogit": graph_pair_logit(d, r["task"], a, b),
                "y": 1.0 if r["ans"] == "yes" else 0.0,
                "key": (sid, r["task"], r["i"], r["j"]),        # preds convention (orig i,j)
            })
        return {"pil": pil, "prompt": prompt, "gfeats": gfeats, "groles": groles,
                "hms": hms, "K": len(slots), "records": records}


def make_frame_collate(processor):
    """Frame collate: base image+prompt SFT batch (no answer text / no LM labels) +
    flat per-person graph/heatmap soft-tokens + per-task record tensors with a frame
    pointer (into the batch) and LOCAL person indices for anchor gathering."""
    tok = processor.tokenizer

    def collate(batch):
        pils = [b["pil"] for b in batch]
        texts = [processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image", "image": b["pil"]},
                                                  {"type": "text", "text": b["prompt"]}]}],
                    tokenize=False, add_generation_prompt=True)
                 for b in batch]
        out = processor(text=texts, images=pils, return_tensors="pt", padding=True)
        out["graph_feats"] = torch.cat([b["gfeats"] for b in batch], dim=0)     # (ΣK,256)
        out["graph_role_ids"] = torch.cat([b["groles"] for b in batch], dim=0)  # (ΣK,)
        out["hm_feats"] = torch.cat([b["hms"] for b in batch], dim=0)           # (ΣK,Hh,Ww)
        out["frame_K"] = torch.tensor([b["K"] for b in batch], dtype=torch.long)

        # Group records across the batch by task -> stacked tensors + frame pointer.
        rec = {}
        for fi, b in enumerate(batch):
            for r in b["records"]:
                t = r["task"]
                slot = rec.setdefault(t, {"frame": [], "la": [], "lb": [], "glogit": [],
                                          "y": [], "edges": defaultdict(list), "keys": []})
                slot["frame"].append(fi)
                slot["la"].append(r["la"]); slot["lb"].append(r["lb"])
                slot["glogit"].append(r["glogit"]); slot["y"].append(r["y"])
                slot["keys"].append(r["key"])
                for name, v in r["edges"].items():
                    slot["edges"][name].append(v)
        packed = {}
        for t, s in rec.items():
            packed[t] = {
                "frame": torch.tensor(s["frame"], dtype=torch.long),
                "la": torch.tensor(s["la"], dtype=torch.long),
                "lb": torch.tensor(s["lb"], dtype=torch.long),
                "glogit": torch.tensor(s["glogit"], dtype=torch.float32),
                "y": torch.tensor(s["y"], dtype=torch.float32),
                "edges": {name: torch.stack(v) for name, v in s["edges"].items()},
                "keys": s["keys"],
            }
        out["records"] = packed
        return out

    return collate
