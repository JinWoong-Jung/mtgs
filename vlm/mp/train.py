from __future__ import annotations
"""Experiment F training: multi-person forward per frame, dense social head, masked BCE.
Only social_bce is unit-tested (CPU). _cmd_train_mp is the GPU entry the user runs."""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from vlm.mp.model import symmetrize


def social_bce(logits, lah, laeo, sa, pos_weight=(3.0, 2.0, 2.0)):
    """logits (B,N,N,3); lah/laeo/sa (B,N,N) in {-1,0,1}. Masked BCE-with-logits:
    LAH over all i!=j with gt!=-1; LAEO/SA over i<j with gt!=-1 (symmetric logits).
    pos_weight=(LAH,LAEO,SA) positive-class weights (graph baseline uses 3/2/2)."""
    N = lah.shape[1]
    device = logits.device
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)

    lah_l = logits[..., 0].float()
    laeo_l = symmetrize(logits[..., 1]).float()
    sa_l = symmetrize(logits[..., 2]).float()
    pw = [torch.tensor(float(w), device=device) for w in pos_weight]

    terms = []
    lah_m = (lah != -1) & (~eye)
    if lah_m.any():
        terms.append(F.binary_cross_entropy_with_logits(
            lah_l[lah_m], lah[lah_m].clamp(min=0).float(), pos_weight=pw[0]))
    for (gt, lg), w in zip(((laeo, laeo_l), (sa, sa_l)), (pw[1], pw[2])):
        m = (gt != -1) & upper
        if m.any():
            terms.append(F.binary_cross_entropy_with_logits(
                lg[m], gt[m].clamp(min=0).float(), pos_weight=w))
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _cmd_train_mp():
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_scheduler
    from peft import LoraConfig, get_peft_model
    from vlm.cfg import QWEN
    from vlm.patches import patch_qwen3vl_patch_embed
    from vlm.mp.prompt import PTOK, frame_prompt
    from vlm.mp.model import (PersonTokenProjector, SocialHead,
                              install_ptok_hook, read_person_hidden)
    from vlm.mp.dataset import FrameDS, bucket_collate, LengthBucketSampler, _valid_people
    from vlm.mp.eval import logits_to_preds
    from vlm.eval import build_mtgs_dicts, evaluate
    import wandb

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mtgs/config/config_vlm_mp.yaml")
    ap.add_argument("--vlmgraph_train", required=True)
    ap.add_argument("--gtmeta_train", required=True)
    ap.add_argument("--overlay_train", required=True)
    ap.add_argument("--vlmgraph_val", default="")
    ap.add_argument("--gtmeta_val", default="")
    ap.add_argument("--overlay_val", default="")
    ap.add_argument("--run_dir", default="", help="output dir (default: <out_root>/<name>)")
    ap.add_argument("--wandb_name", default="", help="W&B run name (default: experiment.name)")
    ap.add_argument("--wandb_off", action="store_true", help="disable W&B logging")
    args = ap.parse_args()
    device = "cuda"

    cfg = OmegaConf.load(args.config)
    epochs = int(cfg.train.epochs)
    bs = int(cfg.train.bs)
    accum = max(1, int(cfg.train.accum))
    num_workers = int(cfg.train.num_workers)
    rank = int(cfg.train.rank)
    seed = int(cfg.train.get("seed", 101))
    num_people = cfg.data.get("num_people", "all")     # "all" (variable N) or int (legacy)
    if num_people != "all":
        num_people = int(num_people)
    max_tokens = int(cfg.train.get("max_tokens", 3000))   # per-batch token budget (OOM guard)
    pos_weight = tuple(float(w) for w in cfg.train.get("pos_weight", [3.0, 2.0, 2.0]))  # LAH/LAEO/SA
    lr = float(cfg.optim.lr)
    weight_decay = float(cfg.optim.weight_decay)
    grad_clip = float(cfg.optim.grad_clip)
    sched_name = str(cfg.optim.scheduler).lower()
    warmup_ratio = float(cfg.optim.warmup_ratio)
    lora_targets = set(cfg.train.get("lora_targets",
                       ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    lora_vision = bool(cfg.train.get("lora_vision", False))
    lora_vision_targets = set(cfg.train.get("lora_vision_targets",
                              ["qkv", "proj", "linear_fc1", "linear_fc2"]))
    torch.manual_seed(seed)

    exp = cfg.experiment
    # run_dir (from launcher) mirrors train_vsgaze.sh: experiments/<date>/<EXP_NAME>/...
    # falls back to the config's out_root/name when not given.
    run_dir = Path(args.run_dir) if args.run_dir else Path(str(exp.out_root)) / str(exp.name)
    ckpt_dir = run_dir / "train" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_name or run_dir.name
    print(f"[mp] exp={run_name} epochs={epochs} bs={bs} accum={accum} lr={lr} "
          f"sched={sched_name} N={num_people} pos_w={pos_weight} vision_lora={lora_vision} "
          f"-> {ckpt_dir}", flush=True)

    use_wandb = not args.wandb_off
    if use_wandb:
        run = wandb.init(
            project="MTGS", entity="gaze-social", group="vlm-stage2",
            name=run_name,
            config={
                "mode": "mp", "lr": lr, "rank": rank, "epochs": epochs, "bs": bs,
                "accum": accum, "scheduler": sched_name, "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay, "num_people": num_people, "graph_feats": True,
                "pos_weight": list(pos_weight), "lora_vision": lora_vision,
            },
        )
        # persist run id so the post-training test eval (separate process) can resume
        # this SAME run and log test/* metrics alongside train/val.
        (ckpt_dir / "wandb_run_id.txt").write_text(run.id)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    proc = AutoProcessor.from_pretrained(QWEN)
    proc.tokenizer.add_special_tokens({"additional_special_tokens": [PTOK]})
    ptok_id = proc.tokenizer.convert_tokens_to_ids(PTOK)
    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map=device)
    model.resize_token_embeddings(len(proc.tokenizer))
    patch_qwen3vl_patch_embed(model)
    D = model.config.text_config.hidden_size
    # LoRA on LM projections; optionally on the vision tower (model.visual) too.
    # isinstance(nn.Linear) guard excludes the Conv3d patch-embed `proj` (name collides).
    targets = []
    for n, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        leaf = n.split(".")[-1]
        if "language_model" in n and leaf in lora_targets:
            targets.append(n)
        elif lora_vision and "visual" in n and leaf in lora_vision_targets:
            targets.append(n)
    nv = sum("visual" in n for n in targets)
    print(f"[mp] LoRA targets: {len(targets)} ({nv} vision, {len(targets)-nv} lm)", flush=True)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                           target_modules=targets, task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.enable_input_require_grads()
    # Gradient checkpointing: recompute layer activations in backward instead of storing
    # them — essential to fit crowded frames (up to ~830 tokens) at batch>1 in bf16.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    proj = PersonTokenProjector(out_dim=D).to(device, torch.bfloat16)
    head = SocialHead(d_model=D).to(device, torch.bfloat16)
    lm = model.base_model.model.model.language_model
    install_ptok_hook(lm)

    ds = FrameDS(args.vlmgraph_train, args.gtmeta_train, args.overlay_train,
                 split="train", num_people=num_people, seed=seed)
    train_sampler = LengthBucketSampler(ds.nps, batch_size=bs, max_tokens=max_tokens,
                                        shuffle=True, seed=seed)
    dl = DataLoader(ds, batch_sampler=train_sampler, num_workers=num_workers,
                    collate_fn=bucket_collate, pin_memory=True)
    print(f"[mp] train frames={len(ds)} batches/epoch={len(dl)}", flush=True)

    # Val dataset loaded ONCE (avoids reloading the ~0.5GB vlmgraph each epoch).
    val_ds = None
    if args.vlmgraph_val and args.gtmeta_val and args.overlay_val:
        val_ds = FrameDS(args.vlmgraph_val, args.gtmeta_val, args.overlay_val, split="val")
        print(f"[mp] val frames={len(val_ds)}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad] + \
             list(proj.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, math.ceil(len(dl) / accum))
    sched = get_scheduler("constant" if sched_name == "none" else sched_name, opt,
                          num_warmup_steps=int(warmup_ratio * total_steps),
                          num_training_steps=total_steps)

    def _chat(pil, prompt):
        return proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": pil},
                                          {"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True)

    def forward_batch(batch):
        """One bucketed batch of variable-N frames. Batched LM forward (processor pads
        token sequences); head runs per frame on real N. Returns list of (n_b,n_b,3)."""
        B = len(batch["pil"])
        texts = [_chat(batch["pil"][b], frame_prompt(batch["labels"][b], batch["bboxes"][b]))
                 for b in range(B)]
        inp = proc(text=texts, images=list(batch["pil"]), return_tensors="pt", padding=True).to(device)
        # project only REAL people (concat across the batch, person-major) -> matches the
        # row-major <ptok> mask over (B, L): frame 0's ptok first, then frame 1's, ...
        feats_all = torch.cat(batch["feats"], dim=0).to(device, torch.bfloat16)   # (sum n_b, 1024)
        mask = (inp["input_ids"] == ptok_id)
        lm._ptok = {"tokens": proj(feats_all), "mask": mask}
        # logits_to_keep=1: we never use vocab logits (dense head reads hidden states), so
        # skip the (B,L,vocab~151k) lm_head tensor — big memory saving.
        out = model(**inp, output_hidden_states=True, logits_to_keep=1)
        hs = read_person_hidden(out.hidden_states[-1], mask)     # list of (n_b, D)
        return [head(hs[b], batch["edge_pp"][b].to(device, torch.bfloat16)) for b in range(B)]

    def batch_loss(logits, batch):
        """Per-frame masked social BCE, averaged over the batch."""
        losses = [social_bce(logits[b].unsqueeze(0),
                             batch["lah"][b].unsqueeze(0).to(device),
                             batch["laeo"][b].unsqueeze(0).to(device),
                             batch["sa"][b].unsqueeze(0).to(device),
                             pos_weight=pos_weight)
                  for b in range(len(logits))]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def run_val():
        if val_ds is None:
            return None
        model.eval()
        vsampler = LengthBucketSampler(val_ds.nps, batch_size=bs, max_tokens=max_tokens,
                                       shuffle=False)
        vdl = DataLoader(val_ds, batch_sampler=vsampler, num_workers=num_workers,
                         collate_fn=bucket_collate)
        preds = {}
        for batch in tqdm(vdl, desc="val", unit="batch", file=sys.stdout, leave=False):
            logits = forward_batch(batch)
            for b in range(len(logits)):
                sid = batch["sid"][b]
                valid = _valid_people(val_ds.gt[sid]["head_bboxes"].float())
                preds.update(logits_to_preds(sid, logits[b].float().cpu(), valid))
        model.train()
        return evaluate(build_mtgs_dicts(args.gtmeta_val, preds))

    def save_ckpt(dst):
        model.save_pretrained(dst)
        torch.save(proj.state_dict(), Path(dst) / "projector.pt")
        torch.save(head.state_dict(), Path(dst) / "social_head.pt")

    best = None
    model.train()
    step = 0
    for ep in range(epochs):
        opt.zero_grad()
        run = 0.0
        pbar = tqdm(dl, desc=f"mp ep{ep}", unit="batch", file=sys.stdout)
        for it, batch in enumerate(pbar):
            logits = forward_batch(batch)
            loss = batch_loss(logits, batch) / accum
            loss.backward()
            run += float(loss) * accum
            if (it + 1) % accum == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                cur_lr = sched.get_last_lr()[0]
                pbar.set_postfix(loss=f"{run/(it+1):.3f}", lr=f"{cur_lr:.2e}")
                if use_wandb:
                    wandb.log({"train/loss": run / (it + 1), "train/lr": cur_lr, "step": step})
        save_ckpt(ckpt_dir / "last")
        m = run_val()
        if m is not None:
            sc = (m["F1_LAH"] + m["F1_LAEO"] + m["AP_SA"]) / 3
            print(f"[mp] ep{ep} F1_LAH={m['F1_LAH']:.4f} F1_LAEO={m['F1_LAEO']:.4f} "
                  f"AP_SA={m['AP_SA']:.4f} mean={sc:.4f}", flush=True)
            if use_wandb:
                wandb.log({"val/F1_LAH": m["F1_LAH"], "val/F1_LAEO": m["F1_LAEO"],
                           "val/AP_SA": m["AP_SA"], "val/mean_social": sc, "epoch": ep})
            if best is None or sc > best:
                best = sc
                save_ckpt(ckpt_dir / "best")
                print(f"[mp] ep{ep} new BEST mean={sc:.4f}", flush=True)
        else:
            print(f"[mp] ep{ep} val skipped", flush=True)
    if not (ckpt_dir / "best").exists():
        save_ckpt(ckpt_dir / "best")
    print(f"[mp] done. best mean={best}", flush=True)
    if use_wandb:
        wandb.summary["best_mean_social"] = best
        wandb.finish()


if __name__ == "__main__":
    _cmd_train_mp()
