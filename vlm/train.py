from __future__ import annotations
"""LoRA-SFT of Qwen3-VL-8B for VLM Stage-2.

Subcommands:
  token    – graph soft-token injection (experiment C)

CLI:
  python -m vlm.train token   --manifest ... --overlay_dir ... --graph_feats ... --out ...
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from vlm.cfg import QWEN
from vlm.dataset import TokenDS, make_token_collate
from vlm.eval import build_mtgs_dicts, evaluate as eval_metrics
from vlm.injection import GTOK, gather_feats, GraphTokenProjector, install_hook
from vlm.prompt import token_prompt

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
                 gtok_id, vlm_bs, device):
        model.eval()
        old = proc.tokenizer.padding_side
        proc.tokenizer.padding_side = "left"
        yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = proc.tokenizer.encode("no", add_special_tokens=False)[0]
        recs = [json.loads(l) for l in open(manifest)]
        overlay_dir = Path(overlay_dir)
        preds = {}
        # Release the training-step allocator cache before val: the val forward runs
        # under no_grad (no autograd graph retained), but training weights+optimizer
        # still occupy the device, so free reserved-but-unused blocks first.
        torch.cuda.empty_cache()
        with torch.no_grad():
            for b0 in range(0, len(recs), vlm_bs):
                chunk = recs[b0:b0 + vlm_bs]
                pils, texts, feats_list, roles_list = [], [], [], []
                for r in chunk:
                    gfd = gf[r["sid"]]
                    bb = gfd["head_bboxes"]
                    pil = Image.open(overlay_dir / r["sid"] / f"{r['i']}_{r['j']}.png").convert("RGB")
                    prompt = token_prompt(r["task"], r["li"], r["lj"], bb[r["i"]], bb[r["j"]])
                    msgs = [{"role": "user", "content": [
                        {"type": "image", "image": pil},
                        {"type": "text", "text": prompt},
                    ]}]
                    texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                    pils.append(pil)
                    f, ro = gather_feats(gfd, r["task"], r["i"], r["j"])
                    feats_list.append(f); roles_list.append(ro)
                inp = proc(text=texts, images=pils, return_tensors="pt", padding=True).to(device)
                gtokens = proj(torch.cat(feats_list).to(device, torch.bfloat16),
                               torch.cat(roles_list).to(device))
                lm._gtok = {"tokens": gtokens, "mask": (inp["input_ids"] == gtok_id)}
                logits = model(**inp).logits[:, -1]
                pyes = torch.softmax(
                    torch.stack([logits[:, yes_id], logits[:, no_id]], -1), -1
                )[:, 0]
                for r, p in zip(chunk, pyes.float().tolist()):
                    preds[(r["sid"], r["task"], r["i"], r["j"])] = p
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
        ap.add_argument("--out", default="results/vlm_lora_token")
        ap.add_argument("--val_manifest", default="")
        ap.add_argument("--val_overlay_dir", default="")
        ap.add_argument("--val_gtmeta", default="")
        ap.add_argument("--val_graph_feats", default="")
        ap.add_argument("--bs", type=int, default=8)
        ap.add_argument("--accum", type=int, default=1)
        ap.add_argument("--lr", type=float, default=1e-4)
        ap.add_argument("--epochs", type=int, default=3)
        ap.add_argument("--steps_per_epoch", type=int, default=20000)
        ap.add_argument("--rank", type=int, default=16)
        ap.add_argument("--num_workers", type=int, default=6)
        ap.add_argument("--vlm_bs", type=int, default=48)
        ap.add_argument("--wandb_name", default="", help="W&B run name (default: out dir name)")
        ap.add_argument("--wandb_off", action="store_true", help="disable W&B logging")
        args = ap.parse_args()
        device = "cuda"

        use_wandb = not args.wandb_off
        if use_wandb:
            wandb.init(
                project="MTGS", entity="gaze-social", group="vlm-stage2",
                name=args.wandb_name or Path(args.out).name,
                config={
                    "mode": _MODE,
                    "lr": args.lr,
                    "rank": args.rank,
                    "epochs": args.epochs,
                    "graph_feats": True,
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
        D = model.config.text_config.hidden_size
        targets = [n for n, _ in model.named_modules()
                   if "language_model" in n and n.split(".")[-1] in PROJ]
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
            target_modules=targets, task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.enable_input_require_grads()
        proj = GraphTokenProjector(out_dim=D).to(device, torch.bfloat16)
        # hook on the base text model (peft wraps the ForConditionalGeneration)
        lm = model.base_model.model.model.language_model
        install_hook(lm)

        sampler = WeightedRandomSampler(ds.sample_weights(), num_samples=args.steps_per_epoch,
                                        replacement=True)
        dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, num_workers=args.num_workers,
                        collate_fn=make_token_collate(proc), pin_memory=True)
        params = [p for p in model.parameters() if p.requires_grad] + list(proj.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

        model.train()
        proj.train()
        step = 0
        for ep in range(args.epochs):
            opt.zero_grad()
            pbar = tqdm(dl, desc=f"token ep{ep}")
            run = 0.0
            correct = total = 0
            for it, batch in enumerate(pbar):
                inject(lm, proj, batch, gtok_id, device)
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out_ = model(**batch)
                loss = out_.loss / args.accum
                loss.backward()
                run += float(out_.loss)
                with torch.no_grad():
                    pred = out_.logits[:, :-1].argmax(-1)
                    lbl = batch["labels"][:, 1:]
                    mask = lbl != -100
                    correct += int(((pred == lbl) & mask).sum())
                    total += int(mask.sum())
                if (it + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    pbar.set_postfix(
                        loss=f"{run/(it+1):.3f}",
                        acc=f"{correct/max(total,1):.3f}",
                    )
                    if use_wandb:
                        wandb.log({
                            "train/loss": run / (it + 1),
                            "train/answer_acc": correct / max(total, 1),
                            "step": step,
                        })
            model.save_pretrained(out / f"ep{ep}")
            torch.save(proj.state_dict(), out / f"ep{ep}" / "projector.pt")
            print(
                f"[token] ep{ep} mean_loss={run/max(len(dl),1):.4f} "
                f"acc={correct/max(total,1):.4f} -> {out/f'ep{ep}'}",
                flush=True,
            )
            # end-of-epoch VAL eval (model selection)
            m = None
            if val_gf is not None and args.val_manifest and Path(args.val_gtmeta).exists():
                try:
                    m = eval_val(model, lm, proj, proc, args.val_manifest, args.val_overlay_dir,
                                 args.val_gtmeta, val_gf, gtok_id, args.vlm_bs, device)
                    print(
                        f"[token] ep{ep} VAL  F1_LAH={m.get('F1_LAH')}  "
                        f"F1_LAEO={m.get('F1_LAEO')}  AP_SA={m.get('AP_SA')}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[token] ep{ep} val failed: {e!r}", flush=True)
            else:
                print(f"[token] ep{ep} val skipped", flush=True)
            if use_wandb and m is not None:
                wandb.log(
                    {f"val/{k}": m[k] for k in
                     ("Dist", "AP_IO", "F1_LAH_PP", "F1_LAEO_PP", "F1_LAH", "F1_LAEO", "AP_SA")
                     if m.get(k) is not None}
                    | {"epoch": ep}
                )
        model.save_pretrained(out / "final")
        torch.save(proj.state_dict(), out / "final" / "projector.pt")
        print(f"[token] done -> {out/'final'}", flush=True)
        if use_wandb:
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
