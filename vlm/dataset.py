from __future__ import annotations
"""LoRA training datasets for VLM Stage-2 (token path: graph soft-token injection).

TokenDS: loads PRE-RENDERED overlay PNGs + graph soft-token features (experiment C).

Ported from peer sgg/datasets.py (make_collate) and
peer sgg/train.py (_cmd_train_lora_token: TokenDS + make_token_collate).
"""

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.injection import gather_feats
from vlm.prompt import token_prompt


# ---------------------------------------------------------------------------
# Collate: graph-text / vision-only (make_collate)
# ---------------------------------------------------------------------------

def make_collate(processor):
    tok = processor.tokenizer
    tok.padding_side = "right"

    def collate(batch):
        pils = [b[0] for b in batch]
        full_texts, ans_lens = [], []
        for pil, prompt, answer in batch:
            msgs = [{"role": "user", "content": [{"type": "image", "image": pil},
                                                  {"type": "text", "text": prompt}]}]
            pt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            full_texts.append(pt + answer)
            ans_lens.append(len(tok(answer, add_special_tokens=False).input_ids))
        # SINGLE processor pass. The original ran the processor a 2nd time on the SAME
        # images (prompt-only) purely to get the prompt length for label masking — that
        # re-did the expensive vision preprocessing every batch. Since full = prompt +
        # answer (right-padded) the answer occupies exactly the trailing `ans_len` tokens
        # (verified: "yes"/"no" are single tokens and tokenize identically at the prompt
        # boundary), so we supervise only that trailing span and mask everything else.
        full = processor(text=list(full_texts), images=pils, return_tensors="pt", padding=True)
        flen = full["attention_mask"].sum(1)          # full (non-pad) token counts
        labels = torch.full_like(full["input_ids"], -100)
        for i in range(labels.shape[0]):
            a = ans_lens[i]
            labels[i, flen[i] - a:flen[i]] = full["input_ids"][i, flen[i] - a:flen[i]]
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
        gfd = self.gf[r["sid"]]
        bb = gfd["head_bboxes"]
        pil = Image.open(self.dir / r["sid"] / "frame.png").convert("RGB")
        prompt = token_prompt(r["task"], r["li"], r["lj"], bb[r["i"]], bb[r["j"]])
        feats, roles = gather_feats(gfd, r["task"], r["i"], r["j"])   # (K,256),(K,)
        return pil, prompt, r["ans"], feats, roles


# ---------------------------------------------------------------------------
# Collate: graph soft-tokens (make_token_collate)
# ---------------------------------------------------------------------------

def make_token_collate(processor):
    """Variable-length token collate: base SFT batch + flat concatenated graph feats/roles.
    <gtok> positions are filled by the hook in row-major (sample, seq) order, so the flat
    concat order (sample-major, then in-prompt order) matches exactly."""
    base = make_collate(processor)

    def collate(batch):
        out = base([(b[0], b[1], b[2]) for b in batch])
        out["graph_feats"] = torch.cat([b[3] for b in batch], dim=0)       # (ΣK, 256)
        out["graph_role_ids"] = torch.cat([b[4] for b in batch], dim=0)    # (ΣK,)
        return out

    return collate
