"""Diagnostic: how much does the VLM correction actually move the graph logit?

Loads the trained frame-pipeline BEST ckpt and, over a val subset, measures per task:
  mean|graph_logit|, mean|correction| (=|logit_final - graph_logit|), their ratio,
  and the DECISION FLIP rate (fraction of pairs whose yes/no decision the VLM changed).
Run on GPU AFTER training frees the device.
"""
import collections
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

from vlm.cfg import QWEN
from vlm.frame_dataset import FrameDS, make_frame_collate
from vlm.injection import GTOK, HMTOK, PANC, GraphTokenProjector, HeatmapEncoder, install_hook
from vlm.patches import patch_qwen3vl_patch_embed
from vlm.social_head import PairwiseSocialHead
from vlm.eval import install_norm_hook, frame_forward

C = "/home/jinwoongjung/MTGS/data/vlm_feature"
CK = "experiments/vlm_frame/VLM_Frame_v4/train/checkpoints/best"
device = "cuda"
N_FRAMES = 1500   # val subset for speed

proc = AutoProcessor.from_pretrained(QWEN)
proc.tokenizer.add_special_tokens({"additional_special_tokens": [GTOK, HMTOK, PANC]})
gtok_id = proc.tokenizer.convert_tokens_to_ids(GTOK)
hmtok_id = proc.tokenizer.convert_tokens_to_ids(HMTOK)
panc_id = proc.tokenizer.convert_tokens_to_ids(PANC)

base = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map=device)
base.resize_token_embeddings(len(proc.tokenizer))
with torch.no_grad():
    temb = base.get_input_embeddings().weight
    tmean = temb[:-3].mean(0)
    for tid in (gtok_id, hmtok_id, panc_id):
        temb[tid] = tmean
patch_qwen3vl_patch_embed(base)
D = base.config.text_config.hidden_size
model = PeftModel.from_pretrained(base, CK).merge_and_unload().eval()
proj = GraphTokenProjector(out_dim=D).to(device, torch.bfloat16)
proj.load_state_dict(torch.load(CK + "/projector.pt", weights_only=True)); proj.eval()
hmenc = HeatmapEncoder(out_dim=D).to(device, torch.bfloat16)
hmenc.load_state_dict(torch.load(CK + "/hmencoder.pt", weights_only=True)); hmenc.eval()
head = PairwiseSocialHead(d_lm=D).to(device)
head.load_state_dict(torch.load(CK + "/social_head.pt", weights_only=True)); head.eval()
lm = model.model.language_model
install_hook(lm)
cap = install_norm_hook(lm)

vds = FrameDS(C + "/manifest_val.jsonl", C + "/overlays/val", C + "/vlmgraph_val.pt")
vds.sids = vds.sids[:N_FRAMES]
dl = DataLoader(vds, batch_size=8, shuffle=False, num_workers=6,
                collate_fn=make_frame_collate(proc), pin_memory=False)

agg = collections.defaultdict(lambda: {"g": [], "c": [], "flip": 0, "n": 0})
for t in agg:
    agg[t]["a"] = []   # per-pair router α samples
with torch.no_grad():
    for batch in dl:
        glogit_by_task = {t: s["glogit"].clone() for t, s in batch["records"].items()}
        out = frame_forward(model, lm, head, proj, hmenc, batch,
                            gtok_id, hmtok_id, panc_id, device, cap)
        for t, o in out.items():
            lf = o["logit"].float().cpu()
            gl = glogit_by_task[t].float()
            corr = lf - gl
            a = agg[t]
            a["g"].append(gl.abs()); a["c"].append(corr.abs())
            a.setdefault("a", []).append(o["alpha"].float().cpu())
            a["flip"] += int(((lf > 0) != (gl > 0)).sum())
            a["n"] += lf.numel()

print("\n===== VLM correction magnitude diagnostic (val subset) =====")
print(f"{'task':>6} {'pairs':>8} {'mean|graph|':>12} {'mean|corr|':>11} "
      f"{'|corr|/|graph|':>14} {'decision_flip%':>15} {'mean_alpha':>11} {'alpha_std':>10}")
for t in ["lah", "laeo", "sa"]:
    a = agg[t]
    if a["n"] == 0:
        continue
    g = torch.cat(a["g"]).mean().item()
    c = torch.cat(a["c"]).mean().item()
    al = torch.cat(a["a"])
    print(f"{t:>6} {a['n']:>8} {g:>12.4f} {c:>11.4f} {c/g:>13.1%} "
          f"{100*a['flip']/a['n']:>14.2f}% {al.mean().item():>11.4f} {al.std().item():>10.4f}")
