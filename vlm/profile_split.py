import time, torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vlm.cfg import QWEN
from vlm.patches import patch_qwen3vl_patch_embed
from PIL import Image
import sys

@torch.no_grad()
def main(img_path):
    proc = AutoProcessor.from_pretrained(QWEN)
    m = Qwen3VLForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda")
    patch_qwen3vl_patch_embed(m)
    pil = Image.open(img_path).convert("RGB")
    inp = proc(text=["<image> hello"], images=[pil], return_tensors="pt").to("cuda")
    for _ in range(3): m(**inp)               # warmup
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(20):
        ie = m.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"])
    torch.cuda.synchronize(); tv=(time.time()-t0)/20
    t0=time.time()
    for _ in range(20): m(**inp)
    torch.cuda.synchronize(); tf=(time.time()-t0)/20
    print(f"vision={tv*1e3:.1f}ms  full_fwd={tf*1e3:.1f}ms  vision_frac={tv/tf:.1%}")

if __name__=="__main__": main(sys.argv[1])
