"""ABLATION — cache frozen DINOv2 scene memory per frame (non-VLM control).

Identical pipeline to cache_scene_memory.py (SAME overlaid center frame, SAME K-token
adaptive pool) but the encoder is a frozen DINOv2 ViT-B/14 instead of Qwen3-VL. This
isolates whether the WP3 residual gain comes from the VLM specifically or from any strong
frozen scene encoder. Saves {sid: [K, 768] fp16} -> scene_dino_<split>.pt.

Run: python -m mtgs.social_vlm.cache_scene_memory_dino --split val [--limit N] [--bs 32]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vlm.overlay import build_frame_overlay, display_labels

C = "/home/jinwoongjung/MTGS/data/vlm_feature"
K_TOKENS = 16
# DINOv2 ImageNet normalisation; 224 -> 16x16=256 patch tokens (pooled to K anyway)
_IMG = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


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
        pil = build_frame_overlay(pil, slots, bb, lab)          # SAME overlay as Qwen path
        return sid, _IMG(pil)


def _load_dino(device):
    cache = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(cache):
        m = torch.hub.load(cache, "dinov2_vitb14", source="local")
    else:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"

    model = _load_dino(device)
    gf = torch.load(f"{C}/vlmgraph_{args.split}.pt", weights_only=False)
    ds = _FrameDS(args.split, gf)
    if args.limit:
        ds.sids = ds.sids[:args.limit]
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.num_workers,
                    collate_fn=lambda b: (tuple(x[0] for x in b), torch.stack([x[1] for x in b])))
    print(f"[dino] split={args.split} frames={len(ds)} K={K_TOKENS}", flush=True)

    model = model.to(torch.bfloat16)   # Blackwell (sm_120) xformers has no fp32 kernel
    out = {}
    with torch.no_grad():
        for sids, imgs in tqdm(dl, desc=f"dino:{args.split}", file=sys.stdout):
            feats = model.forward_features(imgs.to(device, torch.bfloat16))["x_norm_patchtokens"]  # (B,Np,768)
            for b, sid in enumerate(sids):
                h = feats[b].float()                                   # (Np, 768)
                pooled = F.adaptive_avg_pool1d(h.t().unsqueeze(0), K_TOKENS).squeeze(0).t()  # (K,768)
                out[sid] = pooled.half().cpu()

    p = f"{C}/scene_dino_{args.split}.pt"
    torch.save(out, p)
    print(f"[dino] saved {len(out)} DINOv2 scene memories [K={K_TOKENS},D=768] -> {p}", flush=True)


if __name__ == "__main__":
    main()
