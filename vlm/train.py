from __future__ import annotations
"""LoRA-SFT of Qwen3-VL-8B for VLM Stage-2.

Subcommands:
  token    – graph soft-token injection (experiment C)

CLI:
  python -m vlm.train token --manifest ... --overlay_dir ... --graph_feats ... --config ...
  (experiment name, output dir, and all hyperparameters come from the --config YAML)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model

from vlm.cfg import QWEN
from vlm.dataset import TokenDS, make_token_collate
from vlm.eval import build_mtgs_dicts, evaluate as eval_metrics, _TokenRecDS, _coll
from vlm.injection import GTOK, GraphTokenProjector, install_hook
from vlm.patches import patch_qwen3vl_patch_embed

PROJ = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ---------------------------------------------------------------------------
# token (experiment C)
# ---------------------------------------------------------------------------

def _cmd_train_lora_token():
    """Graph-TOKEN LoRA SFT of Qwen3-VL-8B: overlay + variable graph soft-tokens (latent
    fusion) + query -> yes/no for LAH/LAEO/SA. Trains LoRA (LM) + GraphTokenProjector.
    Injects projected graph embeddings at <gtok> placeholders via a forward hook.
    """
    _MODE = "token"

    def inject(lm, proj, batch, gtok_id, device):
        feats = batch.pop("graph_feats").to(device)          # (ΣK, 256)
        roles = batch.pop("graph_role_ids").to(device)       # (ΣK,)
        gtokens = proj(feats.to(torch.bfloat16), roles)      # (ΣK, D)
        mask = (batch["input_ids"] == gtok_id).to(device)
        lm._gtok = {"tokens": gtokens, "mask": mask}

    def eval_val(model, lm, proj, proc, manifest, overlay_dir, gtmeta, gf,
                 gtok_id, vlm_bs, num_workers, device):
        model.eval()
        old = proc.tokenizer.padding_side
        proc.tokenizer.padding_side = "left"
        yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = proc.tokenizer.encode("no", add_special_tokens=False)[0]
        recs = [json.loads(l) for l in open(manifest)]
        # Prefetch via DataLoader workers (same _TokenRecDS/_coll the standalone vlm.eval
        # uses): overlaps the CPU-side PIL decode + graph gather with the GPU forward, so
        # the GPU no longer stalls on per-batch image loading. Semantically identical to
        # the old inline loop — same records, same preds.
        dl = DataLoader(_TokenRecDS(recs, overlay_dir, gf), batch_size=vlm_bs,
                        num_workers=num_workers, collate_fn=_coll, pin_memory=False)
        preds = {}
        # Release the training-step allocator cache before val: the val forward runs
        # under no_grad (no autograd graph retained), but training weights+optimizer
        # still occupy the device, so free reserved-but-unused blocks first.
        torch.cuda.empty_cache()
        with torch.no_grad():
            for keys, pils, prompts, feats, roles in tqdm(dl, desc="val", unit="batch",
                                                          leave=False, file=sys.stdout):
                texts = [proc.apply_chat_template(
                            [{"role": "user", "content": [
                                {"type": "image", "image": p},
                                {"type": "text", "text": t}]}],
                            tokenize=False, add_generation_prompt=True)
                         for p, t in zip(pils, prompts)]
                inp = proc(text=texts, images=list(pils), return_tensors="pt", padding=True).to(device)
                gtokens = proj(feats.to(device, torch.bfloat16), roles.to(device))
                lm._gtok = {"tokens": gtokens, "mask": (inp["input_ids"] == gtok_id)}
                logits = model(**inp).logits[:, -1]
                pyes = torch.softmax(
                    torch.stack([logits[:, yes_id], logits[:, no_id]], -1), -1
                )[:, 0]
                for k, p in zip(keys, pyes.float().tolist()):
                    preds[k] = p
        torch.cuda.empty_cache()   # hand memory back to the training step
        m = eval_metrics(build_mtgs_dicts(gtmeta, preds))
        proc.tokenizer.padding_side = old
        model.train()
        return m

    def main():
        import wandb

        ap = argparse.ArgumentParser()
        ap.add_argument("--manifest", required=True)
        ap.add_argument("--overlay_dir", required=True)
        ap.add_argument("--graph_feats", required=True)
        ap.add_argument("--val_manifest", default="")
        ap.add_argument("--val_overlay_dir", default="")
        ap.add_argument("--val_gtmeta", default="")
        ap.add_argument("--val_graph_feats", default="")
        ap.add_argument("--config", default="mtgs/config/config_vlm.yaml",
                        help="hyperparameter + experiment YAML")
        ap.add_argument("--wandb_name", default="", help="W&B run name (default: experiment.name)")
        ap.add_argument("--wandb_off", action="store_true", help="disable W&B logging")
        args = ap.parse_args()
        device = "cuda"

        # Hyperparameters live in the YAML (paths/run-name stay on the CLI); edit that file
        # to tune batch size, lr, scheduler, epochs, etc. See mtgs/config/config_vlm.yaml.
        cfg = OmegaConf.load(args.config)
        epochs = int(cfg.train.epochs)
        bs = int(cfg.train.bs)
        accum = max(1, int(cfg.train.accum))
        steps_per_epoch = int(cfg.train.steps_per_epoch)
        num_workers = int(cfg.train.num_workers)
        rank = int(cfg.train.rank)
        seed = int(cfg.train.get("seed", 0))
        lora_targets = set(cfg.train.get("lora_targets", list(PROJ)))
        lr = float(cfg.optim.lr)
        weight_decay = float(cfg.optim.weight_decay)
        grad_clip = float(cfg.optim.grad_clip)
        sched_name = str(cfg.optim.scheduler).lower()
        warmup_ratio = float(cfg.optim.warmup_ratio)
        vlm_bs = int(cfg.val.vlm_bs)
        torch.manual_seed(seed)

        # Experiment location + best-ckpt monitor (mirrors the graph run layout:
        # <out_root>/<name>/train/checkpoints/{best,last}).
        exp_name = str(cfg.experiment.name)
        exp_dir = Path(str(cfg.experiment.out_root)) / exp_name
        ckpt_dir = exp_dir / "train" / "checkpoints"
        monitor = str(cfg.experiment.get("monitor", "mean_social"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp_dir / "config_vlm.yaml")     # config snapshot (like .hydra)
        metrics_path = exp_dir / "metrics.jsonl"
        print(f"[token] config={args.config}: exp={exp_name} epochs={epochs} bs={bs} accum={accum} "
              f"lr={lr} sched={sched_name} warmup={warmup_ratio} steps/ep={steps_per_epoch} "
              f"monitor={monitor}", flush=True)
        print(f"[token] out -> {exp_dir}", flush=True)

        # Fixed-shape (448x448) inputs -> let cuDNN pick fast kernels once; allow TF32 on
        # any residual fp32 matmuls. Free speedups, no effect on the bf16 model's outputs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        use_wandb = not args.wandb_off
        if use_wandb:
            wandb.init(
                project="MTGS", entity="gaze-social", group="vlm-stage2",
                name=args.wandb_name or exp_name,
                config={
                    "mode": _MODE, "lr": lr, "rank": rank, "epochs": epochs,
                    "bs": bs, "accum": accum, "scheduler": sched_name,
                    "warmup_ratio": warmup_ratio, "weight_decay": weight_decay,
                    "steps_per_epoch": steps_per_epoch, "graph_feats": True,
                },
            )

        proc = AutoProcessor.from_pretrained(QWEN)
        proc.tokenizer.add_special_tokens({"additional_special_tokens": [GTOK]})
        gtok_id = proc.tokenizer.convert_tokens_to_ids(GTOK)
        ds = TokenDS(args.manifest, args.overlay_dir, args.graph_feats)
        val_gf = torch.load(args.val_graph_feats, weights_only=False) if args.val_graph_feats else None
        print(f"[token] records={len(ds)} gtok_id={gtok_id}", flush=True)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN, dtype=torch.bfloat16, device_map=device)
        model.resize_token_embeddings(len(proc.tokenizer))
        patch_qwen3vl_patch_embed(model)   # Blackwell slow_conv_dilated3d bypass (~48x fwd speedup)
        D = model.config.text_config.hidden_size
        targets = [n for n, _ in model.named_modules()
                   if "language_model" in n and n.split(".")[-1] in lora_targets]
        print(f"[token] LoRA targets={sorted(lora_targets)} ({len(targets)} modules)", flush=True)
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
            target_modules=targets, task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.enable_input_require_grads()
        proj = GraphTokenProjector(out_dim=D).to(device, torch.bfloat16)
        # hook on the base text model (peft wraps the ForConditionalGeneration)
        lm = model.base_model.model.model.language_model
        install_hook(lm)

        sampler = WeightedRandomSampler(ds.sample_weights(), num_samples=steps_per_epoch,
                                        replacement=True)
        dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=num_workers,
                        collate_fn=make_token_collate(proc), pin_memory=True)
        params = [p for p in model.parameters() if p.requires_grad] + list(proj.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        # LR schedule over the true optimizer-step count (batches // accum, per epoch).
        opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / bs) // accum)
        total_steps = epochs * opt_steps_per_epoch
        warmup_steps = int(warmup_ratio * total_steps)
        sched = get_scheduler("constant" if sched_name == "none" else sched_name,
                              opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print(f"[token] scheduler={sched_name} total_steps={total_steps} "
              f"warmup_steps={warmup_steps}", flush=True)

        def save_ckpt(dst):
            model.save_pretrained(dst)
            torch.save(proj.state_dict(), Path(dst) / "projector.pt")

        def monitor_score(mm):
            if mm is None:
                return None
            if monitor == "mean_social":
                vals = [mm.get(k) for k in ("F1_LAH", "F1_LAEO", "AP_SA")]
                vals = [v for v in vals if v is not None]
                return sum(vals) / len(vals) if vals else None
            return mm.get(monitor)

        best_score = None
        best_epoch = -1

        model.train()
        proj.train()
        step = 0
        for ep in range(epochs):
            opt.zero_grad()
            pbar = tqdm(dl, desc=f"token ep{ep}", unit="batch", file=sys.stdout)
            run = 0.0
            correct = total = 0
            for it, batch in enumerate(pbar):
                inject(lm, proj, batch, gtok_id, device)
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out_ = model(**batch)
                loss = out_.loss / accum
                loss.backward()
                run += float(out_.loss)
                with torch.no_grad():
                    pred = out_.logits[:, :-1].argmax(-1)
                    lbl = batch["labels"][:, 1:]
                    mask = lbl != -100
                    correct += int(((pred == lbl) & mask).sum())
                    total += int(mask.sum())
                if (it + 1) % accum == 0:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    step += 1
                    cur_lr = sched.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=f"{run/(it+1):.3f}",
                        acc=f"{correct/max(total,1):.3f}",
                        lr=f"{cur_lr:.2e}",
                    )
                    if use_wandb:
                        wandb.log({
                            "train/loss": run / (it + 1),
                            "train/answer_acc": correct / max(total, 1),
                            "train/lr": cur_lr,
                            "step": step,
                        })
            mean_loss = run / max(len(dl), 1)
            train_acc = correct / max(total, 1)
            # 'last' = most recent epoch (always overwritten), like Lightning save_last=True
            save_ckpt(ckpt_dir / "last")
            print(f"[token] ep{ep} mean_loss={mean_loss:.4f} acc={train_acc:.4f} "
                  f"-> {ckpt_dir/'last'}", flush=True)
            # end-of-epoch VAL eval (model selection)
            m = None
            if val_gf is not None and args.val_manifest and Path(args.val_gtmeta).exists():
                try:
                    m = eval_val(model, lm, proj, proc, args.val_manifest, args.val_overlay_dir,
                                 args.val_gtmeta, val_gf, gtok_id, vlm_bs,
                                 num_workers, device)
                    print(
                        f"[token] ep{ep} VAL  F1_LAH={m.get('F1_LAH')}  "
                        f"F1_LAEO={m.get('F1_LAEO')}  AP_SA={m.get('AP_SA')}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[token] ep{ep} val failed: {e!r}", flush=True)
            else:
                print(f"[token] ep{ep} val skipped", flush=True)

            # 'best' = highest monitored val score so far
            sc = monitor_score(m)
            if sc is not None and (best_score is None or sc > best_score):
                best_score, best_epoch = sc, ep
                save_ckpt(ckpt_dir / "best")
                print(f"[token] ep{ep} new BEST {monitor}={sc:.4f} -> {ckpt_dir/'best'}", flush=True)

            # per-epoch metrics ledger (append)
            rec = {"epoch": ep, "step": step, "train_loss": mean_loss, "train_acc": train_acc,
                   "monitor": monitor, "monitor_score": sc}
            if m is not None:
                rec.update({k: m.get(k) for k in
                            ("Dist", "AP_IO", "F1_LAH_PP", "F1_LAEO_PP", "F1_LAH", "F1_LAEO", "AP_SA",
                             "LAH_AP", "LAH_AUC", "LAEO_AP", "LAEO_AUC", "SA_AP", "SA_AUC")})
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            if use_wandb and m is not None:
                wandb.log(
                    {f"val/{k}": m[k] for k in
                     ("Dist", "AP_IO", "F1_LAH_PP", "F1_LAEO_PP", "F1_LAH", "F1_LAEO", "AP_SA",
                      "LAH_AP", "LAH_AUC", "LAEO_AP", "LAEO_AUC", "SA_AP", "SA_AUC")
                     if m.get(k) is not None}
                    | {"epoch": ep}
                )
        # if val never ran (no best written), fall back to last as best so eval has a ckpt
        if not (ckpt_dir / "best").exists():
            save_ckpt(ckpt_dir / "best")
        print(f"[token] done. best={monitor}={best_score} @ep{best_epoch}  "
              f"ckpts -> {ckpt_dir} (best, last)", flush=True)
        if use_wandb:
            wandb.summary["best_epoch"] = best_epoch
            wandb.summary["best_" + monitor] = best_score
            wandb.finish()

    main()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    _CMDS = {"token": _cmd_train_lora_token}
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        sys.exit("usage: python -m vlm.train {" + "|".join(_CMDS) + "} [args]")
    _cmd = sys.argv.pop(1)
    _CMDS[_cmd]()
