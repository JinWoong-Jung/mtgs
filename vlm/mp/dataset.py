from __future__ import annotations
"""Per-frame dataset for experiment F.

FEATURES come from vlmgraph_*.pt (v_src/v_tgt/edge_pp/edge_null_in/edge_null_out/head_bboxes).
GT comes from gtmeta_*.pt (lah_gt/laeo_gt/coatt_gt as flat permutations(n,2) -> reshaped to
N×N). The two align because head_bboxes match exactly (person index i is the same person).
NEVER use vlmgraph's own lah_gt/laeo_gt/sa_gt — they disagree with gtmeta (verified)."""

import itertools
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def gt_matrices(gtmeta_entry, n):
    """flat permutations(n,2) vectors -> (n,n) long matrices; diagonal = -1."""
    pairs = list(itertools.permutations(range(n), 2))
    lah = torch.full((n, n), -1, dtype=torch.long)
    laeo = torch.full((n, n), -1, dtype=torch.long)
    sa = torch.full((n, n), -1, dtype=torch.long)
    for q, (i, j) in enumerate(pairs):
        lah[i, j] = int(gtmeta_entry["lah_gt"][q])
        laeo[i, j] = int(gtmeta_entry["laeo_gt"][q])
        sa[i, j] = int(gtmeta_entry["coatt_gt"][q])
    return lah, laeo, sa


def person_feats(gf_entry, idxs):
    """[v_src ‖ v_tgt ‖ edge_null_in ‖ edge_null_out] for idxs -> (len(idxs), 1024)."""
    idx = torch.as_tensor(idxs, dtype=torch.long)
    return torch.cat([gf_entry["v_src"].float()[idx],
                      gf_entry["v_tgt"].float()[idx],
                      gf_entry["edge_null_in"].float()[idx],
                      gf_entry["edge_null_out"].float()[idx]], dim=1)


def _valid_people(bb):
    return [k for k in range(bb.shape[0])
            if (bb[k, 2] - bb[k, 0]) > 1e-4 and (bb[k, 3] - bb[k, 1]) > 1e-4]


class FrameDS(Dataset):
    def __init__(self, vlmgraph_path, gtmeta_path, overlay_dir, split,
                 num_people=4, seed=101):
        self.gf = torch.load(vlmgraph_path, weights_only=False)
        self.gt = torch.load(gtmeta_path, weights_only=False)
        self.dir = Path(overlay_dir)
        self.split = split
        self.num_people = num_people
        self.rng = random.Random(seed)
        # frames present in BOTH files with >=2 valid people
        self.sids = []
        for sid in self.gf:
            if sid not in self.gt:
                continue
            if len(_valid_people(self.gt[sid]["head_bboxes"].float())) >= 2:
                self.sids.append(sid)

    def __len__(self):
        return len(self.sids)

    def _select(self, valid, n):
        """train: exactly num_people slots (subsample valid, pad from invalid);
        val/test: all valid people."""
        if self.split != "train":
            return valid
        pool = list(valid)
        self.rng.shuffle(pool)
        chosen = pool[:self.num_people]
        if len(chosen) < self.num_people:              # pad with invalid idxs (masked by GT=-1)
            invalid = [k for k in range(n) if k not in valid]
            self.rng.shuffle(invalid)
            chosen = chosen + invalid[:self.num_people - len(chosen)]
        while len(chosen) < self.num_people:           # last resort: repeat
            chosen.append(chosen[-1])
        return chosen

    def __getitem__(self, k):
        sid = self.sids[k]
        g = self.gf[sid]
        m = self.gt[sid]
        bb_all = m["head_bboxes"].float()
        n = bb_all.shape[0]
        valid = _valid_people(bb_all)
        idxs = self._select(valid, n)
        idx = torch.as_tensor(idxs, dtype=torch.long)
        lah, laeo, sa = gt_matrices(m, n)
        pil = Image.open(self.dir / sid / "frame.png").convert("RGB")
        labels = [f"P{p+1}" for p in range(len(idxs))]
        return {
            "sid": sid,
            "pil": pil,
            "labels": labels,
            "bboxes": bb_all[idx],                          # (M,4)
            "feats": person_feats(g, idxs),                 # (M,1024)
            "edge_pp": g["edge_pp"].float()[idx][:, idx],   # (M,M,256)
            "lah": lah[idx][:, idx],                        # (M,M)
            "laeo": laeo[idx][:, idx],
            "sa": sa[idx][:, idx],
        }


def frame_collate(batch):
    """train collate: fixed num_people per item -> stack. Keeps pil/labels as lists."""
    return {
        "sid":    [b["sid"] for b in batch],
        "pil":    [b["pil"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "bboxes": torch.stack([b["bboxes"] for b in batch]),
        "feats":  torch.stack([b["feats"] for b in batch]),
        "edge_pp": torch.stack([b["edge_pp"] for b in batch]),
        "lah":    torch.stack([b["lah"] for b in batch]),
        "laeo":   torch.stack([b["laeo"] for b in batch]),
        "sa":     torch.stack([b["sa"] for b in batch]),
    }
