from __future__ import annotations
"""LoRA-SFT of Qwen3-VL-8B for VLM Stage-2 (social-gaze specialist).

Per query pair, the model sees the frame with the A(red)/B(blue) head boxes plus two
injected latent modalities — graph node/edge embeddings (<gtok>) and the predicted
gaze heatmaps (<hmtok>) — and answers yes/no for LAH/LAEO/SA. Trains the LM LoRA +
GraphTokenProjector + HeatmapEncoder.

CLI:
  python -m vlm.train --manifest ... --overlay_dir ... --graph_feats ... --config ...
  (experiment name, output dir, and all hyperparameters come from the --config YAML)
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_scheduler
from peft import LoraConfig, PeftModel, get_peft_model

from vlm.cfg import QWEN
from vlm.dataset import TokenDS, make_token_collate
from vlm.eval import build_mtgs_dicts, evaluate as eval_metrics, run_token_eval_batched
from vlm.injection import GTOK, HMTOK, GraphTokenProjector, HeatmapEncoder, install_hook
from vlm.patches import patch_qwen3vl_patch_embed

PROJ = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

# Metric keys logged per split (val/test) + written to the JSONL ledger.
METRIC_KEYS = ("Dist", "AP_IO", "F1_LAH_PP", "F1_LAEO_PP", "F1_LAH", "F1_LAEO", "AP_SA",
               "LAH_AP", "LAH_AUC", "LAEO_AP", "LAEO_AUC", "SA_AP", "SA_AUC",
               "social_ap", "social_auc")


def _subsample_by_sid(recs, limit, seed=101):
    """Deterministic FRAME-level subsample for in-training val: pick whole sids
    (all pairs of a frame kept) until ~limit records. Frame-level (not record-level)
    keeps every frame's pair set complete, so per-target-argmax F1/AP stay valid on
    the subset; metrics are then computed on exactly the covered sids. Same seed
    every epoch -> identical subset -> epochs comparable."""
    if not limit or len(recs) <= limit:
        return recs
    by = {}
    for r in recs:
        by.setdefault(r["sid"], []).append(r)
    sids = sorted(by)
    random.Random(seed).shuffle(sids)
    out = []
    for s in sids:
        out.extend(by[s])
        if len(out) >= limit:
            break
    return out


def _augment_social(m):
    """Add social_ap / social_auc = mean of per-task {LAH,LAEO,SA} AP / AUC — the same
    axis the graph run monitors as metric/val/social_ap (mean of lah/laeo/coatt AP)."""
    if m is None:
        return m
    aps = [m.get(k) for k in ("LAH_AP", "LAEO_AP", "SA_AP") if m.get(k) is not None]
    aucs = [m.get(k) for k in ("LAH_AUC", "LAEO_AUC", "SA_AUC") if m.get(k) is not None]
    m["social_ap"] = sum(aps) / len(aps) if aps else None
    m["social_auc"] = sum(aucs) / len(aucs) if aucs else None
    return m


def train_lora():
    """Train the LM LoRA + GraphTokenProjector + HeatmapEncoder. Graph/heatmap
    embeddings are injected at <gtok>/<hmtok> placeholders via a forward hook."""

    _rms_diag = {"txt": None, "done": False}   # one-shot soft-token scale diagnostic

    def inject(lm, proj, hmenc, batch, gtok_id, hmtok_id, device):
        feats = batch.pop("graph_feats").to(device)          # (ΣK, 256)
        roles = batch.pop("graph_role_ids").to(device)       # (ΣK,)
        gtokens = proj(feats.to(torch.bfloat16), roles)      # (ΣK, D)
        lm._gtok = {"tokens": gtokens,
                    "mask": (batch["input_ids"] == gtok_id).to(device)}
        hms = batch.pop("hm_feats").to(device)               # (ΣM, Hh, Ww)
        hmtokens = hmenc(hms.to(torch.bfloat16))             # (ΣM, D)
        lm._hmtok = {"tokens": hmtokens,
                     "mask": (batch["input_ids"] == hmtok_id).to(device)}
        if not _rms_diag["done"]:
            _rms_diag["done"] = True
            with torch.no_grad():
                g_rms = gtokens.float().pow(2).mean(-1).sqrt().mean().item()
                h_rms = hmtokens.float().pow(2).mean(-1).sqrt().mean().item()
            print(f"[token] RMS diag: text-emb={_rms_diag['txt']:.4f}  "
                  f"gtok={g_rms:.4f}  hmtok={h_rms:.4f} "
                  f"(should be same order of magnitude)", flush=True)

    def eval_val(model, lm, proj, hmenc, proc, manifest, overlay_dir, gtmeta, gf,
                 gtok_id, hmtok_id, vlm_bs, num_workers, device, limit=0):
        model.eval()
        yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = proc.tokenizer.encode("no", add_special_tokens=False)[0]
        recs = [json.loads(l) for l in open(manifest)]
        # Model-selection val on a fixed frame-level subset (val.limit records) —
        # full-split numbers come from the final test eval.
        sub = _subsample_by_sid(recs, limit)
        if len(sub) < len(recs):
            print(f"[token] val subset: {len(sub)}/{len(recs)} records "
                  f"({len({r['sid'] for r in sub})} frames)", flush=True)
        # Release the training-step allocator cache before val: the val forward runs
        # under no_grad (no autograd graph retained), but training weights+optimizer
        # still occupy the device, so free reserved-but-unused blocks first.
        torch.cuda.empty_cache()
        # Batched eval (bs=vlm_bs): the SAME path standalone vlm.eval uses. On Blackwell the
        # LM dominates the forward (~80%), so full-batch LM throughput beats per-frame vision
        # reuse at bs=1. Same records, same preds as the standalone test eval.
        preds = run_token_eval_batched(model, proc, proj, hmenc, lm, sub, overlay_dir, gf,
                                       gtok_id, hmtok_id, yes_id, no_id, device,
                                       vlm_bs, num_workers)
        torch.cuda.empty_cache()   # hand memory back to the training step
        # Restrict metric frames to the evaluated sids: pairs without preds would
        # otherwise default to 0 and pollute AP/F1.
        m = eval_metrics(build_mtgs_dicts(gtmeta, preds,
                                          restrict_sids={k[0] for k in preds}))
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
        # Optional TEST eval on the BEST ckpt after training, logged to the same W&B run.
        ap.add_argument("--test_manifest", default="")
        ap.add_argument("--test_overlay_dir", default="")
        ap.add_argument("--test_gtmeta", default="")
        ap.add_argument("--test_graph_feats", default="")
        ap.add_argument("--test_preds_out", default="")
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
        # From-scratch modules (projector/heatmap-encoder) get their own, higher LR:
        # LoRA fine-tunes pretrained weights, but these start random and need to learn
        # the graph->LM translation fast.
        new_module_lr = float(cfg.optim.get("new_module_lr", 5.0e-4))
        weight_decay = float(cfg.optim.weight_decay)
        grad_clip = float(cfg.optim.grad_clip)
        sched_name = str(cfg.optim.scheduler).lower()
        warmup_ratio = float(cfg.optim.warmup_ratio)
        vlm_bs = int(cfg.val.vlm_bs)
        val_limit = int(cfg.val.get("limit", 0))   # 0 = full val split
        # Hard-example sampling (graph-hardness weighted).
        scfg = cfg.get("sampler", None)
        hard_floor = None
        if scfg is not None and bool(scfg.get("hard_weight", False)):
            hard_floor = float(scfg.get("hard_floor", 0.25))
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
              f"lr={lr} new_module_lr={new_module_lr} sched={sched_name} warmup={warmup_ratio} "
              f"steps/ep={steps_per_epoch} val_limit={val_limit} monitor={monitor}", flush=True)
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
                    "lr": lr, "new_module_lr": new_module_lr, "rank": rank,
                    "epochs": epochs, "bs": bs, "accum": accum, "scheduler": sched_name,
                    "warmup_ratio": warmup_ratio, "weight_decay": weight_decay,
                    "steps_per_epoch": steps_per_epoch, "graph_feats": True,
                    "heatmap_token": True, "hard_floor": hard_floor,
                    "val_limit": val_limit,
                },
            )

        proc = AutoProcessor.from_pretrained(QWEN)
        proc.tokenizer.add_special_tokens({"additional_special_tokens": [GTOK, HMTOK]})
        gtok_id = proc.tokenizer.convert_tokens_to_ids(GTOK)
        hmtok_id = proc.tokenizer.convert_tokens_to_ids(HMTOK)
        ds = TokenDS(args.manifest, args.overlay_dir, args.graph_feats)
        val_gf = torch.load(args.val_graph_feats, weights_only=False) if args.val_graph_feats else None
        print(f"[token] records={len(ds)} gtok_id={gtok_id} hmtok_id={hmtok_id} "
              f"hard_floor={hard_floor}", flush=True)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN, dtype=torch.bfloat16, device_map=device)
        model.resize_token_embeddings(len(proc.tokenizer))
        patch_qwen3vl_patch_embed(model)   # Blackwell slow_conv_dilated3d bypass (~48x fwd speedup)
        D = model.config.text_config.hidden_size
        # Mean per-token RMS of the (pretrained) text embeddings — the target scale for
        # the injected soft tokens (projector/hmenc gain init; see RMS diag print).
        with torch.no_grad():
            txt_rms = (model.get_input_embeddings().weight.float()
                       .pow(2).mean(-1).sqrt().mean().item())
        _rms_diag["txt"] = txt_rms
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
        hmenc = HeatmapEncoder(out_dim=D).to(device, torch.bfloat16)
        # Scale-match the soft tokens to real token embeddings from step 0.
        with torch.no_grad():
            proj.gain.fill_(txt_rms)
            hmenc.gain.fill_(txt_rms)
        print(f"[token] soft-token gain init: text-emb RMS={txt_rms:.4f}", flush=True)
        # hook on the base text model (peft wraps the ForConditionalGeneration)
        lm = model.base_model.model.model.language_model
        install_hook(lm)

        sampler = WeightedRandomSampler(ds.sample_weights(hard_floor=hard_floor),
                                        num_samples=steps_per_epoch,
                                        replacement=True)
        dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=num_workers,
                        collate_fn=make_token_collate(proc), pin_memory=True)
        # Two param groups: LoRA adapters at the base LR, from-scratch modules
        # (projector + heatmap encoder) at new_module_lr. HF schedulers scale each
        # group off its own initial lr, so the ratio is preserved through warmup/decay.
        lora_params = [p for p in model.parameters() if p.requires_grad]
        new_params = list(proj.parameters()) + list(hmenc.parameters())
        params = lora_params + new_params   # flat list for grad clipping
        opt = torch.optim.AdamW(
            [{"params": lora_params, "lr": lr},
             {"params": new_params, "lr": new_module_lr}],
            lr=lr, weight_decay=weight_decay)
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
            torch.save(hmenc.state_dict(), Path(dst) / "hmencoder.pt")

        def monitor_score(mm):
            if mm is None:
                return None
            if monitor == "mean_social":
                vals = [mm.get(k) for k in ("F1_LAH", "F1_LAEO", "AP_SA")]
                vals = [v for v in vals if v is not None]
                return sum(vals) / len(vals) if vals else None
            # social_ap / social_auc (added by _augment_social) or any single metric key
            return mm.get(monitor)

        best_score = None
        best_epoch = -1

        model.train()
        proj.train()
        hmenc.train()
        step = 0
        for ep in range(epochs):
            opt.zero_grad()
            pbar = tqdm(dl, desc=f"token ep{ep}", unit="batch", file=sys.stdout)
            run = 0.0
            correct = total = 0
            for it, batch in enumerate(pbar):
                inject(lm, proj, hmenc, batch, gtok_id, hmtok_id, device)
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
                    m = eval_val(model, lm, proj, hmenc, proc, args.val_manifest,
                                 args.val_overlay_dir, args.val_gtmeta, val_gf, gtok_id,
                                 hmtok_id, vlm_bs, num_workers, device, limit=val_limit)
                    _augment_social(m)
                    print(
                        f"[token] ep{ep} VAL  social_ap={m.get('social_ap')}  "
                        f"LAH_AP={m.get('LAH_AP')} LAEO_AP={m.get('LAEO_AP')} "
                        f"SA_AP={m.get('AP_SA')}",
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
                rec.update({k: m.get(k) for k in METRIC_KEYS})
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

            if use_wandb and m is not None:
                wandb.log(
                    {f"val/{k}": m[k] for k in METRIC_KEYS if m.get(k) is not None}
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

        # ── TEST eval on the BEST ckpt, logged to the SAME W&B run (graph-run parity) ──
        if args.test_manifest and Path(args.test_gtmeta).exists() and (ckpt_dir / "best").exists():
            # Free the training model + optimizer before loading a fresh eval copy of BEST.
            del opt, sched, params, dl, sampler, model, proj, hmenc, lm
            torch.cuda.empty_cache()
            best_dir = ckpt_dir / "best"
            print(f"[test] loading BEST ckpt {best_dir} for test eval ...", flush=True)
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                QWEN, dtype=torch.bfloat16, device_map=device)
            base.resize_token_embeddings(len(proc.tokenizer))
            patch_qwen3vl_patch_embed(base)
            tmodel = PeftModel.from_pretrained(base, str(best_dir)).merge_and_unload().eval()
            tproj = GraphTokenProjector(out_dim=D).to(device, torch.bfloat16)
            tproj.load_state_dict(torch.load(best_dir / "projector.pt", weights_only=True))
            tproj.eval()
            thmenc = HeatmapEncoder(out_dim=D).to(device, torch.bfloat16)
            thmenc.load_state_dict(torch.load(best_dir / "hmencoder.pt", weights_only=True))
            thmenc.eval()
            tlm = tmodel.model.language_model
            install_hook(tlm)
            yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_id = proc.tokenizer.encode("no", add_special_tokens=False)[0]
            # Test uses eval.vlm_bs / eval.num_workers (falls back to val settings).
            ecfg = cfg.get("eval", None)
            test_bs = int(ecfg.vlm_bs) if ecfg is not None else vlm_bs
            test_nw = int(ecfg.num_workers) if ecfg is not None else num_workers
            test_recs = [json.loads(l) for l in open(args.test_manifest)]
            test_gf = torch.load(args.test_graph_feats, weights_only=False)
            print(f"[test] records={len(test_recs)} bs={test_bs} nw={test_nw}", flush=True)
            preds = run_token_eval_batched(tmodel, proc, tproj, thmenc, tlm, test_recs,
                                           args.test_overlay_dir, test_gf, gtok_id, hmtok_id,
                                           yes_id, no_id, device, test_bs, test_nw)
            if args.test_preds_out:
                torch.save(preds, args.test_preds_out)
                print(f"[test] saved {len(preds)} preds -> {args.test_preds_out}", flush=True)
            mt = _augment_social(eval_metrics(build_mtgs_dicts(args.test_gtmeta, preds)))
            print(f"[test] social_ap={mt.get('social_ap')}  F1_LAH={mt.get('F1_LAH')} "
                  f"F1_LAEO={mt.get('F1_LAEO')} AP_SA={mt.get('AP_SA')}", flush=True)
            trec = {"split": "test"} | {k: mt.get(k) for k in METRIC_KEYS}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(trec) + "\n")
            if use_wandb:
                wandb.log({f"test/{k}": mt[k] for k in METRIC_KEYS if mt.get(k) is not None})
                for k in METRIC_KEYS:
                    if mt.get(k) is not None:
                        wandb.summary[f"test_{k}"] = mt[k]

        if use_wandb:
            wandb.finish()

    main()


if __name__ == "__main__":
    train_lora()
