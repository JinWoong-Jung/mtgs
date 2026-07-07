"""Equivalence: grouped vision-reuse eval == naive per-record eval (same plain images)."""
import json, torch
from pathlib import Path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from vlm.cfg import QWEN
from vlm.injection import GTOK, GraphTokenProjector, install_hook, gather_feats
from vlm.patches import patch_qwen3vl_patch_embed
from vlm.prompt import token_prompt
from vlm.eval import _TokenRecDS, _coll
from vlm.vision_cache import run_token_eval_grouped

def main(ckpt, manifest, overlay_dir, graph_feats, n_frames=3):
    proc = AutoProcessor.from_pretrained(QWEN); proc.tokenizer.padding_side="left"
    proc.tokenizer.add_special_tokens({"additional_special_tokens":[GTOK]})
    gtok_id = proc.tokenizer.convert_tokens_to_ids(GTOK)
    yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = proc.tokenizer.encode("no",  add_special_tokens=False)[0]
    base = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda")
    base.resize_token_embeddings(len(proc.tokenizer)); patch_qwen3vl_patch_embed(base)
    D = base.config.text_config.hidden_size
    model = PeftModel.from_pretrained(base, ckpt).merge_and_unload().eval()
    proj = GraphTokenProjector(out_dim=D).to("cuda", torch.bfloat16)
    proj.load_state_dict(torch.load(Path(ckpt)/"projector.pt", weights_only=True)); proj.eval()
    lm = model.model.language_model; install_hook(lm)
    gf = torch.load(graph_feats, weights_only=False)
    recs = [json.loads(l) for l in open(manifest)]
    # limit to first n_frames worth of sids
    sids=[];
    for r in recs:
        if r["sid"] not in sids: sids.append(r["sid"])
        if len(sids) > n_frames: break
    recs = [r for r in recs if r["sid"] in sids[:n_frames]]
    # naive per-record loop (reference)
    ref = {}
    from PIL import Image
    with torch.no_grad():
        for r in recs:
            gfd=gf[r["sid"]]; bb=gfd["head_bboxes"]
            pil=Image.open(Path(overlay_dir)/r["sid"]/"frame.png").convert("RGB")
            prompt=token_prompt(r["task"],r["li"],r["lj"],bb[r["i"]],bb[r["j"]])
            feats,roles=gather_feats(gfd,r["task"],r["i"],r["j"])
            txt=proc.apply_chat_template([{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":prompt}]}],tokenize=False,add_generation_prompt=True)
            inp=proc(text=[txt],images=[pil],return_tensors="pt",padding=True).to("cuda")
            lm._gtok={"tokens":proj(feats.to("cuda",torch.bfloat16),roles.to("cuda")),"mask":(inp["input_ids"]==gtok_id)}
            lg=model(**inp).logits[:,-1]
            ref[(r["sid"],r["task"],r["i"],r["j"])]=torch.softmax(torch.stack([lg[:,yes_id],lg[:,no_id]],-1),-1)[0,0].item()
    got = run_token_eval_grouped(model, proc, proj, lm, recs, overlay_dir, gf, gtok_id, yes_id, no_id, "cuda")
    assert set(got)==set(ref), "key mismatch"
    md = max(abs(got[k]-ref[k]) for k in ref)
    print(f"max |Δ P(yes)| = {md:.2e} over {len(ref)} records")
    assert md < 1e-3, f"not equivalent: {md}"
    print("EQUIVALENCE OK")

if __name__=="__main__":
    import sys; main(*sys.argv[1:])
