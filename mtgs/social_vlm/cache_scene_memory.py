"""WP3 — cache frozen Qwen scene memory per frame (single center frame).

For each frame: overlay ALL valid persons (labelled boxes) on the plain frame.png, run a
FROZEN Qwen3-VL forward with a short neutral prompt, gather the last-layer hidden states
at the image-token positions, and adaptive-pool them to K tokens. Saves {sid: [K, D] fp16}
-> scene_mem_<split>.pt. No graph injection, no LoRA (WP3 external residual baseline).

Run: python -m mtgs.social_vlm.cache_scene_memory --split val [--limit N] [--bs 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from vlm.cfg import QWEN
from vlm.overlay import build_frame_overlay, display_labels
from vlm.patches import patch_qwen3vl_patch_embed

C = "/home/jinwoongjung/MTGS/data/vlm_feature"
K_TOKENS = 16
PROMPT = ("The image shows several people, each marked with a colored labelled head box. "
          "Analyze the scene and the people's gaze.")


def _valid_slots(bb):
    ok = ((bb[:, 2] - bb[:, 0]) > 1e-4) & ((bb[:, 3] - bb[:, 1]) > 1e-4)
    return [k for k in range(bb.shape[0]) if bool(ok[k])]


class _FrameDS(Dataset):
    def __init__(self, split, gf):
        self.dir = Path(f"{C}/overlays/{split}")
        self.gf = gf
        self.sids = [s for s in gf if (self.dir / s / "frame.png").exists()]

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, k):
        sid = self.sids[k]
        bb = self.gf[sid]["head_bboxes"].float()
        slots = _valid_slots(bb)
        _, lab = display_labels(torch.tensor([s in slots for s in range(bb.shape[0])]))
        pil = Image.open(self.dir / sid / "frame.png").convert("RGB")
        pil = build_frame_overlay(pil, slots, bb, lab)
        return sid, pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"

    proc = AutoProcessor.from_pretrained(QWEN)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, dtype=torch.bfloat16, device_map=device).eval()
    patch_qwen3vl_patch_embed(model)
    img_id = model.config.image_token_id
    lm = model.model.language_model
    cap = {}
    lm.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    D = model.config.text_config.hidden_size

    gf = torch.load(f"{C}/vlmgraph_{args.split}.pt", weights_only=False)
    ds = _FrameDS(args.split, gf)
    if args.limit:
        ds.sids = ds.sids[:args.limit]
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.num_workers,
                    collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1]))
    print(f"[scene] split={args.split} frames={len(ds)} K={K_TOKENS} img_token_id={img_id}", flush=True)

    out = {}
    with torch.no_grad():
        for sids, pils in tqdm(dl, desc=f"scene:{args.split}", file=sys.stdout):
            msgs = [[{"role": "user", "content": [{"type": "image", "image": p},
                                                  {"type": "text", "text": PROMPT}]}] for p in pils]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inp = proc(text=texts, images=list(pils), return_tensors="pt", padding=True).to(device)
            try:
                model(**inp, logits_to_keep=1)
            except TypeError:
                model(**inp)
            hidden = cap["h"]                                   # (B, L, D)
            img_mask = (inp["input_ids"] == img_id)             # (B, L)
            for b, sid in enumerate(sids):
                h = hidden[b][img_mask[b]].float()              # (n_img, D)
                # adaptive-pool the image tokens to K along the token axis
                pooled = F.adaptive_avg_pool1d(h.t().unsqueeze(0), K_TOKENS).squeeze(0).t()  # (K, D)
                out[sid] = pooled.half().cpu()

    p = f"{C}/scene_mem_{args.split}.pt"
    torch.save(out, p)
    print(f"[scene] saved {len(out)} scene memories [K={K_TOKENS},D={D}] -> {p}", flush=True)


if __name__ == "__main__":
    main()
