from __future__ import annotations
"""LoRA training datasets for VLM Stage-2 (B/D/E: graph-text, C: graph-token).

LoRADatasetNoGraph: loads PRE-RENDERED overlay PNGs + builds the prompt.
  - graph_feats=None  -> vision-only baseline (experiment A)
  - graph_feats=<pt>  -> prepend graph-text block (experiment B)
  - env GRAPHTEXT_BLIND=1  -> answer-blind graph-text (experiment D)
  - env GRAPH_WRONG_UPWEIGHT>1  -> upsample graph-error cases (experiment E)

TokenDS: loads PRE-RENDERED overlay PNGs + graph soft-token features (experiment C).

Ported from peer sgg/datasets.py (LoRADatasetNoGraph + make_collate) and
peer sgg/train.py (_cmd_train_lora_token: TokenDS + make_token_collate).
Import swaps applied; peer LoRADataset (overlay-on-the-fly) NOT ported.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.injection import GTOK, N_TOK, gather_feats, graph_text_block
from vlm.prompt import nograph_prompt


# ---------------------------------------------------------------------------
# LoRADatasetNoGraph helpers
# ---------------------------------------------------------------------------

_BLIND = os.environ.get("GRAPHTEXT_BLIND") == "1"


def _gscore(d, task, i, j):
    """Graph's pair score (oriented: lah[a,b]='b looks at a' -> 'i looks at j'=lah[j,i])."""
    sig = lambda x: 1.0 / (1.0 + math.exp(-float(x)))
    if task == "lah":
        return sig(d["lah_logits"][j, i])
    if task == "laeo":
        return sig(0.5 * (float(d["laeo_logits"][i, j]) + float(d["laeo_logits"][j, i])))
    return sig(0.5 * (float(d["sa_logits"][i, j]) + float(d["sa_logits"][j, i])))


# ---------------------------------------------------------------------------
# Dataset: graph-text / vision-only (experiments A / B / D / E)
# ---------------------------------------------------------------------------

class LoRADatasetNoGraph(Dataset):
    """graph_feats=None -> vision-only; graph_feats=v14graph_<split>.pt -> prepend the
    graph-text block (same overlay, only the prompt text changes -> clean A/B)."""

    def __init__(self, manifest, overlay_dir, graph_feats=None):
        self.recs = [json.loads(l) for l in open(manifest)]
        self.dir = Path(overlay_dir)
        self.gf = torch.load(graph_feats, weights_only=False) if graph_feats else None

    def __len__(self):
        return len(self.recs)

    def sample_weights(self):
        """Balance every (task, answer) cell equally (LAEO yes is rare ~8%). When
        GRAPH_WRONG_UPWEIGHT>1 (and graph features present), additionally upsample records
        where the GRAPH is wrong, so the VLM gets more supervision on exactly the cases the
        verifier must correct (VERITAS: raise VLM recovery on the graph-error region)."""
        cnt = Counter((r["task"], r["ans"]) for r in self.recs)
        up = float(os.environ.get("GRAPH_WRONG_UPWEIGHT", "1") or 1)
        w = []
        for r in self.recs:
            base = 1.0 / cnt[(r["task"], r["ans"])]
            if up > 1 and self.gf is not None and r["sid"] in self.gf:
                if (_gscore(self.gf[r["sid"]], r["task"], r["i"], r["j"]) > 0.5) != (r["ans"] == "yes"):
                    base *= up
            w.append(base)
        return torch.tensor(w)

    def __getitem__(self, k):
        r = self.recs[k]
        png = self.dir / r["sid"] / f"{r['i']}_{r['j']}.png"
        pil = Image.open(png).convert("RGB")
        prompt = nograph_prompt(r["task"], r["li"], r["lj"])
        if self.gf is not None and r["sid"] in self.gf:
            prompt = graph_text_block(r["task"], r["i"], r["j"], self.gf[r["sid"]],
                                      r["li"], r["lj"], answer_blind=_BLIND) + "\n" + prompt
        return pil, prompt, r["ans"]


# ---------------------------------------------------------------------------
# Collate: graph-text / vision-only (make_collate)
# ---------------------------------------------------------------------------

def make_collate(processor):
    tok = processor.tokenizer
    tok.padding_side = "right"

    def collate(batch):
        pils, prompts, answers = zip(*batch)
        full_texts, prompt_texts = [], []
        for pil, prompt, answer in batch:
            msgs = [{"role": "user", "content": [{"type": "image", "image": pil},
                                                  {"type": "text", "text": prompt}]}]
            pt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(pt)
            full_texts.append(pt + answer)
        full = processor(text=list(full_texts), images=list(pils), return_tensors="pt", padding=True)
        prm = processor(text=list(prompt_texts), images=list(pils), return_tensors="pt", padding=True)
        plen = prm["attention_mask"].sum(1)           # prompt token counts
        flen = full["attention_mask"].sum(1)          # full token counts
        labels = full["input_ids"].clone()
        for i in range(labels.shape[0]):
            labels[i, :plen[i]] = -100                # mask prompt
            labels[i, flen[i]:] = -100                # mask right padding
        full["labels"] = labels
        return full

    return collate


# ---------------------------------------------------------------------------
# Dataset: graph soft-tokens (experiment C)
# ---------------------------------------------------------------------------

class TokenDS(Dataset):
    def __init__(self, manifest, overlay_dir, graph_feats):
        self.recs = [json.loads(l) for l in open(manifest)]
        self.dir = Path(overlay_dir)
        self.gf = torch.load(graph_feats, weights_only=False)

    def __len__(self):
        return len(self.recs)

    def sample_weights(self):
        cnt = Counter((r["task"], r["ans"]) for r in self.recs)
        return torch.tensor([1.0 / cnt[(r["task"], r["ans"])] for r in self.recs])

    def __getitem__(self, k):
        r = self.recs[k]
        pil = Image.open(self.dir / r["sid"] / f"{r['i']}_{r['j']}.png").convert("RGB")
        prompt = (GTOK * N_TOK) + "\n" + nograph_prompt(r["task"], r["li"], r["lj"])
        feats = gather_feats(self.gf[r["sid"]], r["i"], r["j"])  # (N_TOK,256)
        return pil, prompt, r["ans"], feats


# ---------------------------------------------------------------------------
# Collate: graph soft-tokens (make_token_collate)
# ---------------------------------------------------------------------------

def make_token_collate(processor):
    base = make_collate(processor)

    def collate(batch):
        feats = torch.stack([b[3] for b in batch])              # (B,N_TOK,256)
        out = base([(b[0], b[1], b[2]) for b in batch])
        out["graph_feats"] = feats
        return out

    return collate
