from __future__ import annotations
"""Experiment F eval: one forward per frame -> N×N logits -> preds dict -> reuse the
graph-baseline metric axis (vlm.eval.build_mtgs_dicts + evaluate)."""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from vlm.mp.model import symmetrize


def logits_to_preds(sid, logits, idxs):
    """logits (M,M,3) over selected people (idxs = original person indices) ->
    preds keyed by ORIGINAL indices. LAH directed (all i!=j); LAEO/SA symmetric, i<j."""
    lah = torch.sigmoid(logits[..., 0])
    laeo = torch.sigmoid(symmetrize(logits[..., 1]))
    sa = torch.sigmoid(symmetrize(logits[..., 2]))
    preds = {}
    M = len(idxs)
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            i, j = idxs[a], idxs[b]
            preds[(sid, "lah", i, j)] = float(lah[a, b])
            lo, hi = (a, b) if a < b else (b, a)
            oi, oj = idxs[lo], idxs[hi]
            preds[(sid, "laeo", oi, oj)] = float(laeo[lo, hi])
            preds[(sid, "sa", oi, oj)] = float(sa[lo, hi])
    return preds


def _main_eval_mp():
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from vlm.cfg import QWEN
    from vlm.patches import patch_qwen3vl_patch_embed
    from vlm.mp.prompt import PTOK, frame_prompt
    from vlm.mp.model import (PersonTokenProjector, SocialHead,
                              install_ptok_hook, read_person_hidden)
    from vlm.mp.dataset import person_feats, _valid_people
    from vlm.eval import build_mtgs_dicts, evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mtgs/config/config_vlm_mp.yaml")
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--vlmgraph", required=True)
    ap.add_argument("--gtmeta", required=True)
    ap.add_argument("--overlay_dir", required=True)
    ap.add_argument("--preds_out", default="")
    args = ap.parse_args()
    device = "cuda"

    cfg = OmegaConf.load(args.config)
    if not args.ckpt:
        xc = cfg.experiment
        args.ckpt = str(Path(str(xc.out_root)) / str(xc.name) / "train" / "checkpoints" / args.which)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    proc = AutoProcessor.from_pretrained(QWEN)
    proc.tokenizer.add_special_tokens({"additional_special_tokens": [PTOK]})
    ptok_id = proc.tokenizer.convert_tokens_to_ids(PTOK)
    base = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map=device)
    base.resize_token_embeddings(len(proc.tokenizer))
    patch_qwen3vl_patch_embed(base)
    D = base.config.text_config.hidden_size
    model = PeftModel.from_pretrained(base, args.ckpt).merge_and_unload().eval()
    proj = PersonTokenProjector(out_dim=D).to(device, torch.bfloat16)
    proj.load_state_dict(torch.load(Path(args.ckpt) / "projector.pt", weights_only=True))
    proj.eval()
    head = SocialHead(d_model=D).to(device, torch.bfloat16)
    head.load_state_dict(torch.load(Path(args.ckpt) / "social_head.pt", weights_only=True))
    head.eval()
    lm = model.model.language_model
    install_ptok_hook(lm)

    gf = torch.load(args.vlmgraph, weights_only=False)
    gt = torch.load(args.gtmeta, weights_only=False)
    overlay = Path(args.overlay_dir)
    preds = {}
    with torch.no_grad():
        for sid in tqdm([s for s in gf if s in gt], desc="mp-eval", unit="frame", file=sys.stdout):
            m = gt[sid]
            g = gf[sid]
            bb = m["head_bboxes"].float()
            valid = _valid_people(bb)
            if len(valid) < 2:
                continue
            vidx = torch.as_tensor(valid)
            labels = [f"P{p+1}" for p in range(len(valid))]
            prompt = frame_prompt(labels, bb[vidx])
            pil = Image.open(overlay / sid / "frame.png").convert("RGB")
            txt = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": pil},
                                              {"type": "text", "text": prompt}]}],
                tokenize=False, add_generation_prompt=True)
            inp = proc(text=[txt], images=[pil], return_tensors="pt", padding=True).to(device)
            feats = person_feats(g, valid).to(device, torch.bfloat16)
            mask = (inp["input_ids"] == ptok_id)
            lm._ptok = {"tokens": proj(feats), "mask": mask}
            out = model(**inp, output_hidden_states=True)
            h = read_person_hidden(out.hidden_states[-1], mask)[0]      # (len(valid), D)
            edge = g["edge_pp"].float()[vidx][:, vidx].to(device, torch.bfloat16)
            logits = head(h, edge).float().cpu()
            preds.update(logits_to_preds(sid, logits, valid))

    out_path = args.preds_out or f"preds_mp_{Path(args.ckpt).name}.pt"
    torch.save(preds, out_path)
    m = evaluate(build_mtgs_dicts(args.gtmeta, preds))
    print(f"[mp-eval] F1_LAH={m['F1_LAH']:.4f} F1_LAEO={m['F1_LAEO']:.4f} AP_SA={m['AP_SA']:.4f}", flush=True)


if __name__ == "__main__":
    _main_eval_mp()
