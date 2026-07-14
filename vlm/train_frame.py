from __future__ import annotations
"""Frame-pipeline LoRA-SFT of Qwen3-VL-8B for VLM Stage-2 (social-gaze specialist).

ONE forward per FRAME: the model sees the frame with every person's head box plus per-
person graph (<gtok>) + gaze-heatmap (<hmtok>) soft-tokens and a per-person anchor
(<panc>). PairwiseSocialHead reads the anchor hidden states and predicts a CORRECTION
Δ on top of the frozen graph logit for every queried pair (LAH/LAEO/SA), trained with
BCE. No text answer is decoded. Trains LM LoRA + GraphTokenProjector + HeatmapEncoder +
PairwiseSocialHead. gate/Δ zero-init => step-0 == graph-only (never below baseline).

CLI mirrors vlm.train (paths on CLI, hyperparameters in --config YAML).
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
from vlm.eval import (
    build_mtgs_dicts,
    evaluate as eval_metrics,
    frame_forward,
    install_norm_hook,
    run_frame_eval_batched,
)
from vlm.frame_dataset import FrameDS, make_frame_collate
from vlm.injection import (
    GTOK, HMTOK, PANC, GraphTokenProjector, HeatmapEncoder, install_hook,
    query_slots, graph_pair_logit,
)
from vlm.patches import patch_qwen3vl_patch_embed
from vlm.prompt import TASKS
from vlm.social_head import PairwiseSocialHead

PROJ = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

METRIC_KEYS = ("Dist", "AP_IO", "F1_LAH_PP", "F1_LAEO_PP", "F1_LAH", "F1_LAEO", "AP_SA",
               "LAH_AP", "LAH_AUC", "LAEO_AP", "LAEO_AUC", "SA_AP", "SA_AUC",
               "social_ap", "social_auc")


def _augment_social(m):
    if m is None:
        return m
    aps = [m.get(k) for k in ("LAH_AP", "LAEO_AP", "SA_AP") if m.get(k) is not None]
    aucs = [m.get(k) for k in ("LAH_AUC", "LAEO_AUC", "SA_AUC") if m.get(k) is not None]
    m["social_ap"] = sum(aps) / len(aps) if aps else None
    m["social_auc"] = sum(aucs) / len(aucs) if aucs else None
    return m


# VLM metric-dict key -> MTGS+graph wandb key (same axis as models.py; SA == coatt).
# Lets the frame-VLM AP/AUC/dist overlay the graph run's metric/<split>/... panels.
_GRAPH_KEY = {
    "LAH_AP": "lah_ap",   "LAH_AUC": "lah_auc",
    "LAEO_AP": "laeo_ap", "LAEO_AUC": "laeo_auc",
    "SA_AP": "coatt_ap",  "SA_AUC": "coatt_auc",   # SA is 'coatt' in the graph run
    "social_ap": "social_ap", "social_auc": "social_auc",
    "Dist": "dist", "AP_IO": "ap_io",
    "F1_LAH": "f1_lah", "F1_LAEO": "f1_laeo",       # graph logs F1 only in test text, but keep for reference
}


def _graph_metric_log(m, split, tag=""):
    """Map a VLM metric dict to the graph run's wandb keys: metric/<split>/[<tag>_]<name>."""
    out = {}
    if m is None:
        return out
    pre = f"{tag}_" if tag else ""
    for src, dst in _GRAPH_KEY.items():
        v = m.get(src)
        if v is not None:
            out[f"metric/{split}/{pre}{dst}"] = v
    return out


def _graph_only_preds(manifest, graph_feats):
    """Graph-only preds {(sid,task,i,j): sigmoid(graph_logit)} straight from the graph
    cache — NO VLM. Evaluating these through the SAME build_mtgs_dicts+compute_metrics
    harness gives the ONLY apples-to-apples graph baseline to overlay the blended val
    against (the graph *training* run's native metrics are a different harness and are
    NOT comparable — e.g. its lah_auc is computed differently)."""
    gf = torch.load(graph_feats, weights_only=False)
    preds = {}
    for line in open(manifest):
        r = json.loads(line)
        if r["sid"] not in gf:
            continue
        a, b, _, _ = query_slots(r)
        lg = graph_pair_logit(gf[r["sid"]], r["task"], a, b)
        preds[(r["sid"], r["task"], r["i"], r["j"])] = 1.0 / (1.0 + math.exp(-lg))
    return preds


def _pos_weights(ds, device):
    """Per-task BCE pos_weight = n_neg / n_pos from the train manifest (rare-positive
    upweighting, esp. LAEO). Clamped to [0.2, 5] to avoid extreme scaling."""
    from collections import Counter
    c = Counter((r["task"], r["ans"]) for s in ds.sids for r in ds.by_sid[s])
    pw = {}
    for t in TASKS:
        npos = c.get((t, "yes"), 0)
        nneg = c.get((t, "no"), 0)
        pw[t] = torch.tensor(min(max(nneg / max(npos, 1), 0.2), 5.0), device=device)
    return pw


def train_frame():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--overlay_dir", required=True)
    ap.add_argument("--graph_feats", required=True)
    ap.add_argument("--val_manifest", default="")
    ap.add_argument("--val_overlay_dir", default="")
    ap.add_argument("--val_gtmeta", default="")
    ap.add_argument("--val_graph_feats", default="")
    ap.add_argument("--test_manifest", default="")
    ap.add_argument("--test_overlay_dir", default="")
    ap.add_argument("--test_gtmeta", default="")
    ap.add_argument("--test_graph_feats", default="")
    ap.add_argument("--test_preds_out", default="")
    ap.add_argument("--config", default="mtgs/config/config_vlm.yaml")
    ap.add_argument("--wandb_name", default="")
    ap.add_argument("--wandb_off", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    import wandb

    cfg = OmegaConf.load(args.config)
    epochs = int(cfg.train.epochs)
    bs = int(cfg.train.bs)
    accum = max(1, int(cfg.train.accum))
    frames_per_epoch = int(cfg.train.get("frames_per_epoch", cfg.train.get("steps_per_epoch", 12000)))
    num_workers = int(cfg.train.num_workers)
    rank = int(cfg.train.rank)
    seed = int(cfg.train.get("seed", 0))
    lora_targets = set(cfg.train.get("lora_targets", list(PROJ)))
    lr = float(cfg.optim.lr)
    new_module_lr = float(cfg.optim.get("new_module_lr", 5.0e-4))
    weight_decay = float(cfg.optim.weight_decay)
    grad_clip = float(cfg.optim.grad_clip)
    sched_name = str(cfg.optim.scheduler).lower()
    warmup_ratio = float(cfg.optim.warmup_ratio)
    aux_weight = float(cfg.optim.get("aux_weight", 0.5))   # weight on standalone vlm_logit BCE
    blend_w = float(cfg.get("fusion", {}).get("graph_weight", 0.5))  # fixed graph blend weight
    vlm_bs = int(cfg.val.vlm_bs)
    val_limit = int(cfg.val.get("limit", 0))   # 0 = all val frames (frame pipeline is cheap)
    scfg = cfg.get("sampler", None)
    hard_floor = None
    if scfg is not None and bool(scfg.get("hard_weight", False)):
        hard_floor = float(scfg.get("hard_floor", 0.25))
    torch.manual_seed(seed)

    exp_name = str(cfg.experiment.name)
    exp_dir = Path(str(cfg.experiment.out_root)) / exp_name
    ckpt_dir = exp_dir / "train" / "checkpoints"
    monitor = str(cfg.experiment.get("monitor", "social_ap"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config_vlm.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"[frame] config={args.config}: exp={exp_name} epochs={epochs} bs={bs} accum={accum} "
          f"lr={lr} new_module_lr={new_module_lr} aux_weight={aux_weight} blend_w={blend_w} "
          f"sched={sched_name} warmup={warmup_ratio} frames/ep={frames_per_epoch} "
          f"val_limit={val_limit} monitor={monitor}", flush=True)
    print(f"[frame] out -> {exp_dir}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    use_wandb = not args.wandb_off
    if use_wandb:
        wandb.init(project="MTGS", entity="gaze-social", group="vlm-stage2-frame",
                   name=args.wandb_name or exp_name,
                   config={"lr": lr, "new_module_lr": new_module_lr, "aux_weight": aux_weight,
                           "rank": rank,
                           "epochs": epochs, "bs": bs, "accum": accum, "scheduler": sched_name,
                           "warmup_ratio": warmup_ratio, "weight_decay": weight_decay,
                           "frames_per_epoch": frames_per_epoch, "pipeline": "frame",
                           "hard_floor": hard_floor, "val_limit": val_limit})

    proc = AutoProcessor.from_pretrained(QWEN)
    proc.tokenizer.add_special_tokens({"additional_special_tokens": [GTOK, HMTOK, PANC]})
    gtok_id = proc.tokenizer.convert_tokens_to_ids(GTOK)
    hmtok_id = proc.tokenizer.convert_tokens_to_ids(HMTOK)
    panc_id = proc.tokenizer.convert_tokens_to_ids(PANC)
    ds = FrameDS(args.manifest, args.overlay_dir, args.graph_feats)
    print(f"[frame] frames={len(ds)} records={ds.num_records} "
          f"gtok={gtok_id} hmtok={hmtok_id} panc={panc_id} hard_floor={hard_floor}", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, dtype=torch.bfloat16, device_map=device)
    model.resize_token_embeddings(len(proc.tokenizer))
    # Newly-added tokens (<gtok>/<hmtok> get replaced by injection, but <panc> is READ):
    # init all three to the mean pretrained embedding so the anchor position starts from a
    # sensible vector rather than a wild random row (embeddings are frozen during training).
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean_emb = emb[:-3].mean(0)
        for tid in (gtok_id, hmtok_id, panc_id):
            emb[tid] = mean_emb
    patch_qwen3vl_patch_embed(model)
    D = model.config.text_config.hidden_size
    with torch.no_grad():
        txt_rms = emb.float().pow(2).mean(-1).sqrt().mean().item()
    targets = [n for n, _ in model.named_modules()
               if "language_model" in n and n.split(".")[-1] in lora_targets]
    print(f"[frame] LoRA targets={sorted(lora_targets)} ({len(targets)} modules)", flush=True)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.enable_input_require_grads()

    proj = GraphTokenProjector(out_dim=D).to(device, torch.bfloat16)
    hmenc = HeatmapEncoder(out_dim=D).to(device, torch.bfloat16)
    with torch.no_grad():
        proj.gain.fill_(txt_rms)
        hmenc.gain.fill_(txt_rms)
    head = PairwiseSocialHead(d_lm=D, blend_w=blend_w).to(device)  # fixed graph:vlm blend
    print(f"[frame] soft-token gain init: text-emb RMS={txt_rms:.4f}", flush=True)

    lm = model.base_model.model.model.language_model
    install_hook(lm)                 # <gtok>/<hmtok> injection
    cap = install_norm_hook(lm)      # capture final hidden state for anchors
    pos_w = _pos_weights(ds, device)
    print(f"[frame] pos_weight " + " ".join(f"{t}={pos_w[t].item():.2f}" for t in TASKS), flush=True)

    sampler = WeightedRandomSampler(ds.sample_weights(hard_floor=hard_floor),
                                    num_samples=frames_per_epoch, replacement=True)
    dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=num_workers,
                    collate_fn=make_frame_collate(proc), pin_memory=True)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    new_params = list(proj.parameters()) + list(hmenc.parameters()) + list(head.parameters())
    params = lora_params + new_params
    opt = torch.optim.AdamW([{"params": lora_params, "lr": lr},
                             {"params": new_params, "lr": new_module_lr}],
                            lr=lr, weight_decay=weight_decay)
    opt_steps_per_epoch = max(1, math.ceil(frames_per_epoch / bs) // accum)
    total_steps = epochs * opt_steps_per_epoch
    warmup_steps = int(warmup_ratio * total_steps)
    sched = get_scheduler("constant" if sched_name == "none" else sched_name,
                          opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"[frame] scheduler={sched_name} total_steps={total_steps} warmup_steps={warmup_steps}",
          flush=True)

    def save_ckpt(dst):
        model.save_pretrained(dst)
        torch.save(proj.state_dict(), Path(dst) / "projector.pt")
        torch.save(hmenc.state_dict(), Path(dst) / "hmencoder.pt")
        torch.save(head.state_dict(), Path(dst) / "social_head.pt")

    def monitor_score(mm):
        if mm is None:
            return None
        if monitor == "mean_social":
            vals = [mm.get(k) for k in ("F1_LAH", "F1_LAEO", "AP_SA")]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None
        return mm.get(monitor)

    def eval_val():
        model.eval(); proj.eval(); hmenc.eval(); head.eval()
        vds = FrameDS(args.val_manifest, args.val_overlay_dir, args.val_graph_feats)
        # Optional frame cap for speed (0 = all): deterministic first-N frames.
        if val_limit and len(vds.sids) > val_limit:
            vds.sids = vds.sids[:val_limit]
        torch.cuda.empty_cache()
        preds = run_frame_eval_batched(model, proc, proj, hmenc, head, lm, vds,
                                       gtok_id, hmtok_id, panc_id, device,
                                       vlm_bs, num_workers, cap)
        torch.cuda.empty_cache()
        m = eval_metrics(build_mtgs_dicts(args.val_gtmeta, preds,
                                          restrict_sids={k[0] for k in preds}))
        model.train(); proj.train(); hmenc.train(); head.train()
        return _augment_social(m)

    # Graph-only baseline via the SAME frame harness (constant; the correct overlay
    # reference for the blended val — logged flat each epoch as metric/val/graph_only_*).
    graph_val = None
    if args.val_manifest and Path(args.val_gtmeta).exists():
        try:
            gp = _graph_only_preds(args.val_manifest, args.val_graph_feats)
            graph_val = _augment_social(eval_metrics(build_mtgs_dicts(
                args.val_gtmeta, gp, restrict_sids={k[0] for k in gp})))
            print(f"[frame] graph-only(val, same harness): social_ap={graph_val.get('social_ap'):.4f} "
                  f"LAH_AP={graph_val.get('LAH_AP'):.4f} LAEO_AP={graph_val.get('LAEO_AP'):.4f} "
                  f"SA_AP={graph_val.get('SA_AP'):.4f}", flush=True)
        except Exception as e:
            print(f"[frame] graph-only baseline failed: {e!r}", flush=True)

    best_score, best_epoch = None, -1
    model.train(); proj.train(); hmenc.train(); head.train()
    step = 0
    for ep in range(epochs):
        opt.zero_grad()
        pbar = tqdm(dl, desc=f"frame ep{ep}", unit="batch", file=sys.stdout)
        run = 0.0
        correct = total = 0
        a_sum = {t: 0.0 for t in TASKS}   # running per-pair α mean per task (router monitor)
        a_cnt = {t: 0 for t in TASKS}
        for it, batch in enumerate(pbar):
            out = frame_forward(model, lm, head, proj, hmenc, batch, gtok_id, hmtok_id,
                                panc_id, device, cap)
            losses, aux_losses = [], []
            per_task_bce = {}   # for graph-comparable loss/train/<task> logging (SA->coatt)
            for t, o in out.items():
                # main: blended prediction; aux: standalone vlm_logit (keeps it alive/
                # calibrated regardless of α — no blend deadlock, no saturation stall).
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    o["logit"], o["y"], pos_weight=pos_w[t])
                aux = torch.nn.functional.binary_cross_entropy_with_logits(
                    o["vlm_logit"], o["y"], pos_weight=pos_w[t])
                losses.append(bce); aux_losses.append(aux)
                per_task_bce["coatt" if t == "sa" else t] = float(bce)
                with torch.no_grad():
                    correct += int((((o["logit"] > 0).float()) == o["y"]).sum())
                    total += int(o["y"].numel())
                    a_sum[t] += float(o["alpha"].sum()); a_cnt[t] += o["alpha"].numel()
            if not losses:      # batch with no annotated pairs (defensive; shouldn't happen)
                continue
            loss = (torch.stack(losses).mean()
                    + aux_weight * torch.stack(aux_losses).mean()) / accum
            loss.backward()
            run += float(loss) * accum
            if (it + 1) % accum == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                cur_lr = sched.get_last_lr()[0]
                pbar.set_postfix(loss=f"{run/(it+1):.3f}", acc=f"{correct/max(total,1):.3f}",
                                 lr=f"{cur_lr:.2e}")
                if use_wandb:
                    wandb.log({"train/loss": run / (it + 1),
                               "train/answer_acc": correct / max(total, 1),
                               "train/lr": cur_lr, "step": step,
                               # graph-run-comparable loss keys (combined + per-task, SA->coatt)
                               "loss/train": run / (it + 1)}
                              | {f"loss/train/{t}": v for t, v in per_task_bce.items()})
        mean_loss = run / max(len(dl), 1)
        train_acc = correct / max(total, 1)
        mean_alpha = {t: a_sum[t] / max(a_cnt[t], 1) for t in TASKS}   # per-task mean router α
        save_ckpt(ckpt_dir / "last")
        print(f"[frame] ep{ep} mean_loss={mean_loss:.4f} acc={train_acc:.4f} -> {ckpt_dir/'last'}",
              flush=True)

        m = None
        if args.val_manifest and Path(args.val_gtmeta).exists():
            try:
                m = eval_val()
                print(f"[frame] ep{ep} VAL social_ap={m.get('social_ap')} "
                      f"F1_LAH={m.get('F1_LAH')} F1_LAEO={m.get('F1_LAEO')} "
                      f"AP_SA={m.get('AP_SA')}  mean_alpha(graph_wt)="
                      f"{ {t: round(mean_alpha[t], 3) for t in TASKS} }", flush=True)
            except Exception as e:
                print(f"[frame] ep{ep} val failed: {e!r}", flush=True)
        else:
            print(f"[frame] ep{ep} val skipped", flush=True)

        sc = monitor_score(m)
        if sc is not None and (best_score is None or sc > best_score):
            best_score, best_epoch = sc, ep
            save_ckpt(ckpt_dir / "best")
            print(f"[frame] ep{ep} new BEST {monitor}={sc:.4f} -> {ckpt_dir/'best'}", flush=True)

        rec = {"epoch": ep, "step": step, "train_loss": mean_loss, "train_acc": train_acc,
               "monitor": monitor, "monitor_score": sc}
        if m is not None:
            rec.update({k: m.get(k) for k in METRIC_KEYS})
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if use_wandb and m is not None:
            wandb.log({f"val/{k}": m[k] for k in METRIC_KEYS if m.get(k) is not None}
                      | _graph_metric_log(m, "val")                       # blended VLM
                      | _graph_metric_log(graph_val, "val", tag="graph_only")  # same-harness baseline
                      | {f"train/alpha_{t}": mean_alpha[t] for t in TASKS}
                      | {"epoch": ep})

    if not (ckpt_dir / "best").exists():
        save_ckpt(ckpt_dir / "best")
    print(f"[frame] done. best={monitor}={best_score} @ep{best_epoch} ckpts -> {ckpt_dir}",
          flush=True)
    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_" + monitor] = best_score

    # ── TEST eval on BEST ckpt (same W&B run) ──
    if args.test_manifest and Path(args.test_gtmeta).exists() and (ckpt_dir / "best").exists():
        del opt, sched, params, dl, sampler, model, proj, hmenc, head, lm
        torch.cuda.empty_cache()
        best_dir = ckpt_dir / "best"
        print(f"[test] loading BEST ckpt {best_dir} for test eval ...", flush=True)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN, dtype=torch.bfloat16, device_map=device)
        base.resize_token_embeddings(len(proc.tokenizer))
        # Re-apply the SAME frozen mean-init to the 3 added tokens as at train time. The
        # PEFT adapter does not persist base embeddings, so without this the <panc> anchor
        # row would be a random resize default here — a train/test input mismatch (the
        # mean is deterministic from the frozen pretrained embeddings, so this reproduces
        # exactly what training used).
        with torch.no_grad():
            temb = base.get_input_embeddings().weight
            tmean = temb[:-3].mean(0)
            for tid in (gtok_id, hmtok_id, panc_id):
                temb[tid] = tmean
        patch_qwen3vl_patch_embed(base)
        tmodel = PeftModel.from_pretrained(base, str(best_dir)).merge_and_unload().eval()
        tproj = GraphTokenProjector(out_dim=D).to(device, torch.bfloat16)
        tproj.load_state_dict(torch.load(best_dir / "projector.pt", weights_only=True)); tproj.eval()
        thmenc = HeatmapEncoder(out_dim=D).to(device, torch.bfloat16)
        thmenc.load_state_dict(torch.load(best_dir / "hmencoder.pt", weights_only=True)); thmenc.eval()
        thead = PairwiseSocialHead(d_lm=D, blend_w=blend_w).to(device)
        thead.load_state_dict(torch.load(best_dir / "social_head.pt", weights_only=True)); thead.eval()
        tlm = tmodel.model.language_model
        install_hook(tlm)
        tcap = install_norm_hook(tlm)
        ecfg = cfg.get("eval", None)
        test_bs = int(ecfg.vlm_bs) if ecfg is not None else vlm_bs
        test_nw = int(ecfg.num_workers) if ecfg is not None else num_workers
        tds = FrameDS(args.test_manifest, args.test_overlay_dir, args.test_graph_feats)
        print(f"[test] frames={len(tds)} records={tds.num_records} bs={test_bs} nw={test_nw}", flush=True)
        preds = run_frame_eval_batched(tmodel, proc, tproj, thmenc, thead, tlm, tds,
                                       gtok_id, hmtok_id, panc_id, device, test_bs, test_nw, tcap)
        if args.test_preds_out:
            torch.save(preds, args.test_preds_out)
            print(f"[test] saved {len(preds)} preds -> {args.test_preds_out}", flush=True)
        mt = _augment_social(eval_metrics(build_mtgs_dicts(args.test_gtmeta, preds)))
        print(f"[test] social_ap={mt.get('social_ap')} F1_LAH={mt.get('F1_LAH')} "
              f"F1_LAEO={mt.get('F1_LAEO')} AP_SA={mt.get('AP_SA')}", flush=True)
        trec = {"split": "test"} | {k: mt.get(k) for k in METRIC_KEYS}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(trec) + "\n")
        if use_wandb:
            wandb.log({f"test/{k}": mt[k] for k in METRIC_KEYS if mt.get(k) is not None}
                      | _graph_metric_log(mt, "test"))   # graph-run-comparable keys
            for k in METRIC_KEYS:
                if mt.get(k) is not None:
                    wandb.summary[f"test_{k}"] = mt[k]
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train_frame()
