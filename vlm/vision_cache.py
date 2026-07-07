from __future__ import annotations
"""Frame-grouped token eval: run the Qwen3-VL vision tower once per frame and reuse it
across that frame's records. Preds are numerically identical to the naive per-record
loop (`model(**inp)`) by construction (equivalence-tested in vlm/tests_vision_cache.py).

Approach
--------
We do NOT touch inputs_embeds or manually splice image embeds. Instead we monkeypatch
`model.model.get_image_features` (the Qwen3VLModel method that runs the expensive vision
tower `self.visual`) with a per-frame cache, and call `model(**inp)` NORMALLY per record
(with pixel_values + image_grid_thw). Because the real `Qwen3VLModel.forward` runs
unchanged, deepstack feature injection (config deepstack_visual_indexes), M-RoPE 3D
position ids, the image-embed scatter, and the `lm._gtok` hook are all handled by the
model itself. The ONLY thing deduped is `get_image_features`, which for one frame runs
once (first record) and returns the cached BaseModelOutputWithDeepstackFeatures for the
rest — every record of a frame shares an identical frame.png, hence identical vision
output. This makes equivalence hold by construction.
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
    inner = model.model                      # Qwen3VLModel; forward calls self.get_image_features(...)
    orig_gif = inner.get_image_features      # bound method (call WITHOUT self)
    cache = {"val": None}

    def _cached_gif(pixel_values, image_grid_thw=None, **kw):
        # All records of one frame share an identical frame.png -> identical vision output.
        if cache["val"] is not None:
            return cache["val"]
        cache["val"] = orig_gif(pixel_values, image_grid_thw, **kw)
        return cache["val"]

    inner.get_image_features = _cached_gif   # instance attr shadows the method (no self passed)

    by_sid = collections.OrderedDict()
    for r in recs:
        by_sid.setdefault(r["sid"], []).append(r)
    preds = {}
    try:
        for sid, group in tqdm(by_sid.items(), desc="grouped-eval", unit="frame", file=sys.stdout):
            cache["val"] = None                                  # reset: recompute vision once per frame
            gfd = gf[sid]; bb = gfd["head_bboxes"]
            pil = Image.open(Path(overlay_dir) / sid / "frame.png").convert("RGB")
            for r in group:
                prompt = token_prompt(r["task"], r["li"], r["lj"], bb[r["i"]], bb[r["j"]])
                txt = proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image", "image": pil},
                                                  {"type": "text", "text": prompt}]}],
                    tokenize=False, add_generation_prompt=True)
                inp = proc(text=[txt], images=[pil], return_tensors="pt", padding=True).to(device)
                feats, roles = gather_feats(gfd, r["task"], r["i"], r["j"])
                lm._gtok = {"tokens": proj(feats.to(device, torch.bfloat16), roles.to(device)),
                            "mask": (inp["input_ids"] == gtok_id)}
                lg = model(**inp).logits[:, -1]
                p = torch.softmax(torch.stack([lg[:, yes_id], lg[:, no_id]], -1), -1)[0, 0].item()
                preds[(sid, r["task"], r["i"], r["j"])] = p
    finally:
        inner.get_image_features = orig_gif                      # ALWAYS restore
    return preds
