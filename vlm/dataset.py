from __future__ import annotations
"""LoRA training dataset for VLM Stage-2 (social-gaze specialist).

TokenDS: loads pre-rendered frame PNGs + per-frame graph features, and builds one
(image, prompt, graph soft-tokens, heatmap soft-tokens) example per manifest record.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.injection import gather_feats, gather_heatmaps, query_slots
from vlm.overlay import build_token_overlay
from vlm.prompt import token_prompt


# ---------------------------------------------------------------------------
# Shared example builder (train TokenDS + eval _TokenRecDS use the SAME path)
# ---------------------------------------------------------------------------

def build_example(rec, gfd, overlay_dir):
    """Render one (image, prompt, feats, roles, hms) example for a manifest record.

    Query orientation, overlay roles, graph soft-tokens and heatmap soft-tokens all
    go through injection.query_slots so they stay consistent by construction.
      pil   : frame with ONLY the A(red)/B(blue) head boxes
      feats : (K,256) graph node/edge embeddings   + roles (K,)
      hms   : (M,Hh,Ww) predicted gaze heatmaps for the queried persons
    """
    a, b, la, lb = query_slots(rec)
    bb = gfd["head_bboxes"].float()
    pil = Image.open(Path(overlay_dir) / rec["sid"] / "frame.png").convert("RGB")
    pil = build_token_overlay(pil, rec["task"], a, b, bb, {a: la, b: lb})
    prompt = token_prompt(rec["task"], la, lb, bb[a], bb[b])
    feats, roles = gather_feats(gfd, rec["task"], a, b)        # (K,256),(K,)
    hms = gather_heatmaps(gfd, rec["task"], a, b)              # (M,Hh,Ww)
    return pil, prompt, feats, roles, hms


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
# Dataset: graph + heatmap soft-tokens
# ---------------------------------------------------------------------------

class TokenDS(Dataset):
    def __init__(self, manifest, overlay_dir, graph_feats):
        self.recs = [json.loads(l) for l in open(manifest)]
        self.dir = Path(overlay_dir)
        self.gf = torch.load(graph_feats, weights_only=False)

    def __len__(self):
        return len(self.recs)

    def sample_weights(self, hard_floor=None):
        """(task, ans)-balanced weights, optionally scaled by graph hardness.

        hard_floor=None keeps the original pure class balance. Otherwise each
        record is scaled by (hard_floor + |gt - p_graph|) in [floor, floor+1]:
        records the graph already gets right with confidence contribute ~floor,
        graph-wrong/uncertain records (the oracle-router headroom) dominate.
        p_graph is the graph readout probability (input feature, not GT)."""
        cnt = Counter((r["task"], r["ans"]) for r in self.recs)
        w = torch.tensor([1.0 / cnt[(r["task"], r["ans"])] for r in self.recs])
        if hard_floor is None:
            return w
        from vlm.injection import pair_belief
        hard = torch.empty(len(self.recs))
        for k, r in enumerate(self.recs):
            a, b, _, _ = query_slots(r)
            p = pair_belief(self.gf[r["sid"]], r["task"], a, b)["p"]
            y = 1.0 if r["ans"] == "yes" else 0.0
            hard[k] = abs(y - p)
        return w * (float(hard_floor) + hard)

    def __getitem__(self, k):
        r = self.recs[k]
        pil, prompt, feats, roles, hms = build_example(r, self.gf[r["sid"]], self.dir)
        return pil, prompt, r["ans"], feats, roles, hms


# ---------------------------------------------------------------------------
# Collate: graph soft-tokens (make_token_collate)
# ---------------------------------------------------------------------------

def make_token_collate(processor):
    """Variable-length token collate: base SFT batch + flat concatenated graph
    feats/roles + heatmaps. <gtok>/<hmtok> positions are filled by the hook in
    row-major (sample, seq) order, so each modality's sample-major flat concat
    (then in-prompt order) matches its placeholder order exactly."""
    base = make_collate(processor)

    def collate(batch):
        out = base([(b[0], b[1], b[2]) for b in batch])
        out["graph_feats"] = torch.cat([b[3] for b in batch], dim=0)       # (ΣK, 256)
        out["graph_role_ids"] = torch.cat([b[4] for b in batch], dim=0)    # (ΣK,)
        out["hm_feats"] = torch.cat([b[5] for b in batch], dim=0)          # (ΣM, Hh, Ww)
        return out

    return collate
