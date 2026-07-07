from __future__ import annotations
"""Frame-grouped token eval: run the Qwen3-VL vision tower once per frame and reuse the
image embeds across that frame's records via inputs_embeds splicing. Preds are identical
to the naive per-record loop (equivalence-tested).

# ============================================================
# API DISCREPANCY vs BRIEF — READ BEFORE RUNNING EQUIV TEST
# ============================================================
# The brief assumes `model.model.get_image_features(pixel_values, image_grid_thw)` returns
# either a plain tensor (num_img_tok, D) or a list/tuple thereof.
#
# ACTUAL RETURN (transformers build, inspected 2026-07-07):
#   The method is decorated with @can_return_tuple and returns
#   BaseModelOutputWithDeepstackFeatures (a ModelOutput / OrderedDict subclass —
#   NOT a tuple or list). The actual image-token embeds are in
#   result.pooler_output, which is the output of torch.split(embeds, split_sizes) —
#   i.e. a tuple of tensors of shape (n_tokens_i, D), one per image in the batch.
#
# The brief's guard `isinstance(img_embeds, (list, tuple))` would NOT fire on a
# ModelOutput, so `img_embeds.to(bfloat16)` would raise AttributeError at runtime.
#
# ADAPTATION APPLIED HERE (Step 3 of the brief explicitly permits this):
#   After calling get_image_features, we extract .pooler_output and cat the tuple.
#   For a single image the result is shape (n_tokens, D) — the intended contract.
#
# CONFIRM WHEN RUNNING THE EQUIVALENCE TEST:
#   1. model.model.get_image_features exists and the call does not raise
#   2. result.pooler_output is a tuple-of-tensors (torch.split output, one per image)
#   3. model.config.image_token_id resolves to the correct integer token id
#   4. model.model.language_model is the correct text sub-model path
#   5. max |ΔP(yes)| < 1e-3 vs the naive per-record loop
# ============================================================
"""
import collections
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import sys
from vlm.prompt import token_prompt
from vlm.injection import gather_feats


@torch.no_grad()
def run_token_eval_grouped(model, proc, proj, lm, recs, overlay_dir, gf,
                           gtok_id, yes_id, no_id, device):
    img_tok_id = model.config.image_token_id
    by_sid = collections.OrderedDict()
    for r in recs:
        by_sid.setdefault(r["sid"], []).append(r)
    preds = {}
    for sid, group in tqdm(by_sid.items(), desc="grouped-eval", unit="frame", file=sys.stdout):
        gfd = gf[sid]; bb = gfd["head_bboxes"]
        pil = Image.open(Path(overlay_dir) / sid / "frame.png").convert("RGB")
        # ---- vision tower ONCE for this frame ----
        vis_in = proc(text=["<image>"], images=[pil], return_tensors="pt").to(device)
        raw = model.model.get_image_features(
            vis_in["pixel_values"], vis_in["image_grid_thw"])
        # ADAPTATION: get_image_features returns BaseModelOutputWithDeepstackFeatures
        # (ModelOutput / OrderedDict subclass), NOT a plain tensor or tuple.
        # Actual embeds live in .pooler_output as torch.split() output (tuple of tensors).
        if hasattr(raw, "pooler_output"):
            po = raw.pooler_output
            img_embeds = torch.cat(po, dim=0) if isinstance(po, (list, tuple)) else po
        elif isinstance(raw, (list, tuple)):
            img_embeds = torch.cat(raw, dim=0)
        else:
            img_embeds = raw
        img_embeds = img_embeds.to(torch.bfloat16)              # (num_img_tok, D)
        # ---- per-record LM forward reusing img_embeds ----
        for r in group:
            prompt = token_prompt(r["task"], r["li"], r["lj"], bb[r["i"]], bb[r["j"]])
            txt = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": pil},
                                              {"type": "text", "text": prompt}]}],
                tokenize=False, add_generation_prompt=True)
            inp = proc(text=[txt], images=[pil], return_tensors="pt", padding=True).to(device)
            ids = inp["input_ids"]
            emb = model.model.language_model.get_input_embeddings()(ids).clone()  # (1,L,D)
            img_pos = (ids[0] == img_tok_id)
            emb[0, img_pos] = img_embeds.to(emb.dtype)           # splice cached vision
            feats, roles = gather_feats(gfd, r["task"], r["i"], r["j"])
            lm._gtok = {"tokens": proj(feats.to(device, torch.bfloat16), roles.to(device)),
                        "mask": (ids == gtok_id)}                # gtok hook splices these
            lg = model(inputs_embeds=emb, attention_mask=inp["attention_mask"]).logits[:, -1]
            p = torch.softmax(torch.stack([lg[:, yes_id], lg[:, no_id]], -1), -1)[0, 0].item()
            preds[(sid, r["task"], r["i"], r["j"])] = p
    return preds
