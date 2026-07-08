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
from torch.utils.data import Dataset, Sampler


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
    """Per-frame items with variable N (all valid people). `num_people="all"` (default)
    uses every valid person; an int keeps only the fixed-N subsample path (legacy). With
    N=all, batch via LengthBucketSampler + bucket_collate (variable-length lists)."""
    def __init__(self, vlmgraph_path, gtmeta_path, overlay_dir, split,
                 num_people="all", seed=101):
        self.gf = torch.load(vlmgraph_path, weights_only=False)
        self.gt = torch.load(gtmeta_path, weights_only=False)
        self.dir = Path(overlay_dir)
        self.split = split
        self.num_people = num_people
        self.rng = random.Random(seed)
        # frames present in BOTH files with >=2 valid people
        self.sids = []
        self.nps = []          # valid-people count per frame (for LengthBucketSampler)
        for sid in self.gf:
            if sid not in self.gt:
                continue
            nv = len(_valid_people(self.gt[sid]["head_bboxes"].float()))
            if nv >= 2:
                self.sids.append(sid)
                self.nps.append(nv)

    def __len__(self):
        return len(self.sids)

    def _select(self, valid, n):
        """N=all: every valid person. Int num_people: fixed-N slots (subsample valid,
        pad from invalid — legacy path, masked by GT=-1)."""
        if self.num_people == "all":
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


_KEYS = ("sid", "pil", "labels", "bboxes", "feats", "edge_pp", "lah", "laeo", "sa")


def bucket_collate(batch):
    """Variable-N collate: keep every field as a per-frame list (no stacking). The LM
    forward is still batched (processor pads token sequences); the head + loss run per
    frame on that frame's real N, so no padded-person slots are ever created."""
    return {k: [b[k] for b in batch] for k in _KEYS}


class LengthBucketSampler(Sampler):
    """Yield batches of frame indices with similar people-count N, so each batch's prompts
    are close in length and the processor pads minimally.

    `batch_size` caps frames/batch. `max_tokens` (optional) additionally caps the batch's
    estimated token cost `(#frames × est_tokens(maxN))` — so crowded frames (long prompts)
    form SMALLER batches, bounding activation memory (prevents OOM on 30+ person frames).
    est_tokens(n) = base_tokens + n × tokens_per_person."""
    def __init__(self, lengths, batch_size, max_tokens=None, base_tokens=350,
                 tokens_per_person=20, shuffle=True, seed=101):
        self.lengths = list(lengths)
        self.bs = batch_size
        self.max_tokens = max_tokens
        self.base_tokens = base_tokens
        self.tpp = tokens_per_person
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def _est(self, n):
        return self.base_tokens + n * self.tpp

    def _batches(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(idx)                       # break ties randomly
        idx.sort(key=lambda i: self.lengths[i])         # bucket by N
        if self.max_tokens is None:
            return [idx[i:i + self.bs] for i in range(0, len(idx), self.bs)]
        batches, cur = [], []
        for i in idx:
            est_i = self._est(self.lengths[i])          # sorted asc -> i is the batch max
            if cur and (len(cur) >= self.bs or (len(cur) + 1) * est_i > self.max_tokens):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            self.rng.shuffle(batches)                   # random batch order (mixes N sizes)
        yield from batches

    def __len__(self):
        return len(self._batches())
