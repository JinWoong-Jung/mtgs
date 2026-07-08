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
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from vlm.cfg import QWEN
    from vlm.patches import patch_qwen3vl_patch_embed
    from vlm.mp.prompt import PTOK, frame_prompt
    from vlm.mp.model import (PersonTokenProjector, SocialHead,
                              install_ptok_hook, read_person_hidden)
    from vlm.mp.dataset import FrameDS, bucket_collate, LengthBucketSampler, _valid_people
    from vlm.eval import build_mtgs_dicts, evaluate
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mtgs/config/config_vlm_mp.yaml")
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--run_dir", default="", help="run dir (default: <out_root>/<name>)")
    ap.add_argument("--vlmgraph", required=True)
    ap.add_argument("--gtmeta", required=True)
    ap.add_argument("--overlay_dir", required=True)
    ap.add_argument("--preds_out", default="")
    ap.add_argument("--bs", type=int, default=8, help="frame batch cap (length-bucketed)")
    ap.add_argument("--max_tokens", type=int, default=3000, help="per-batch token budget (OOM guard)")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--split_tag", default="test", help="W&B metric prefix (test/val)")
    ap.add_argument("--wandb_off", action="store_true", help="disable W&B logging")
    args = ap.parse_args()
    device = "cuda"

    cfg = OmegaConf.load(args.config)
    if not args.ckpt:
        base = Path(args.run_dir) if args.run_dir else \
            Path(str(cfg.experiment.out_root)) / str(cfg.experiment.name)
        args.ckpt = str(base / "train" / "checkpoints" / args.which)

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

    ds = FrameDS(args.vlmgraph, args.gtmeta, args.overlay_dir, split="test", num_people="all")
    sampler = LengthBucketSampler(ds.nps, batch_size=args.bs, max_tokens=args.max_tokens, shuffle=False)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=bucket_collate)
    preds = {}
    with torch.no_grad():
        for batch in tqdm(dl, desc="mp-eval", unit="batch", file=sys.stdout):
            B = len(batch["pil"])
            texts = [proc.apply_chat_template(
                        [{"role": "user", "content": [
                            {"type": "image", "image": batch["pil"][b]},
                            {"type": "text", "text": frame_prompt(batch["labels"][b], batch["bboxes"][b])}]}],
                        tokenize=False, add_generation_prompt=True) for b in range(B)]
            inp = proc(text=texts, images=list(batch["pil"]), return_tensors="pt", padding=True).to(device)
            feats_all = torch.cat(batch["feats"], dim=0).to(device, torch.bfloat16)
            mask = (inp["input_ids"] == ptok_id)
            lm._ptok = {"tokens": proj(feats_all), "mask": mask}
            out = model(**inp, output_hidden_states=True, logits_to_keep=1)
            hs = read_person_hidden(out.hidden_states[-1], mask)
            for b in range(B):
                logits = head(hs[b], batch["edge_pp"][b].to(device, torch.bfloat16)).float().cpu()
                sid = batch["sid"][b]
                valid = _valid_people(ds.gt[sid]["head_bboxes"].float())
                preds.update(logits_to_preds(sid, logits, valid))

    out_path = args.preds_out or f"preds_mp_{Path(args.ckpt).name}.pt"
    torch.save(preds, out_path)
    m = evaluate(build_mtgs_dicts(args.gtmeta, preds))
    sc = (m["F1_LAH"] + m["F1_LAEO"] + m["AP_SA"]) / 3
    print(f"[mp-eval] F1_LAH={m['F1_LAH']:.4f} F1_LAEO={m['F1_LAEO']:.4f} "
          f"AP_SA={m['AP_SA']:.4f} mean={sc:.4f}", flush=True)

    # Log to W&B: resume the training run (id saved by train.py) so test/* lands next to
    # train/val on the same run; fall back to a standalone run if no id file is present.
    if not args.wandb_off:
        import wandb
        tag = args.split_tag
        rid_file = Path(args.ckpt).parent / "wandb_run_id.txt"
        rid = rid_file.read_text().strip() if rid_file.exists() else None
        wandb.init(project="MTGS", entity="gaze-social", group="vlm-stage2",
                   id=rid, resume="allow" if rid else None,
                   name=None if rid else f"{Path(args.ckpt).parents[2].name}-{tag}")
        wandb.log({f"{tag}/F1_LAH": m["F1_LAH"], f"{tag}/F1_LAEO": m["F1_LAEO"],
                   f"{tag}/AP_SA": m["AP_SA"], f"{tag}/mean_social": sc})
        wandb.summary[f"{tag}_mean_social"] = sc
        wandb.finish()


if __name__ == "__main__":
    _main_eval_mp()
