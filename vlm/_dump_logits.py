"""Dump raw (graph_logit, vlm_logit, alpha) per test pair from a frame ckpt, so we can
compare graph-only / vlm-only / fixed-blend / learned-router / α-sweep WITHOUT re-running
the VLM. Saves {(sid,task,i,j): (graph_logit, vlm_logit, alpha)} to a .pt."""
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
OUT = C + "/logits_VLM_Frame_v4_test.pt"
SPLIT = "test"
device = "cuda"

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

ds = FrameDS(f"{C}/manifest_{SPLIT}.jsonl", f"{C}/overlays/{SPLIT}", f"{C}/vlmgraph_{SPLIT}.pt")
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=10,
                collate_fn=make_frame_collate(proc), pin_memory=False)
print(f"[dump] frames={len(ds)} records={ds.num_records}", flush=True)

out = {}
from tqdm import tqdm
import sys
with torch.no_grad():
    for batch in tqdm(dl, desc="dump", file=sys.stdout):
        glogit_by_task = {t: s["glogit"].clone() for t, s in batch["records"].items()}
        o = frame_forward(model, lm, head, proj, hmenc, batch,
                          gtok_id, hmtok_id, panc_id, device, cap)
        for t, d in o.items():
            gl = glogit_by_task[t]
            vl = d["vlm_logit"].float().cpu()
            al = d["alpha"].float().cpu()
            for k, g, v, a in zip(d["keys"], gl.tolist(), vl.tolist(), al.tolist()):
                out[k] = (g, v, a)

torch.save(out, OUT)
print(f"[dump] saved {len(out)} (graph,vlm,alpha) logits -> {OUT}", flush=True)
