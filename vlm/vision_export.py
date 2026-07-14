"""Export frozen Qwen3-VL vision outputs into a frame-keyed disk cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from vlm.patches import patch_qwen3vl_patch_embed
from vlm.pair_model import _find_multimodal_model
from vlm.vision_cache import DiskVisionFrame, VisionDiskCacheWriter


def _batches(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qwen", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("vision export requires a CUDA GPU")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    frame_root = Path(args.frame_root)
    paths = sorted(frame_root.glob("*/frame.png"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no frame.png files under {frame_root}")
    processor = AutoProcessor.from_pretrained(args.qwen, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    patch_qwen3vl_patch_embed(model)
    multimodal = _find_multimodal_model(model)
    visual = multimodal.visual
    metadata = {
        "qwen": args.qwen,
        "max_pixels": str(args.max_pixels),
        "dtype": "bfloat16",
        "out_hidden_size": str(visual.config.out_hidden_size),
        "deepstack_visual_indexes": ",".join(map(str, visual.config.deepstack_visual_indexes)),
    }
    writer = VisionDiskCacheWriter(
        args.out, metadata, shard_size=args.shard_size, overwrite=args.overwrite
    )
    try:
        for batch_paths in tqdm(list(_batches(paths, args.batch_size)), desc="vision-export"):
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            image_inputs = processor.image_processor(images=images, return_tensors="pt")
            grid = image_inputs["image_grid_thw"]
            with torch.no_grad():
                output = multimodal.get_image_features(
                    image_inputs["pixel_values"].to(model.device), grid.to(model.device), return_dict=True
                )
            merged_sizes = (grid.prod(-1) // int(visual.spatial_merge_size) ** 2).tolist()
            deep_splits = [torch.split(layer, [int(size) for size in merged_sizes]) for layer in output.deepstack_features]
            for index, path in enumerate(batch_paths):
                writer.add(
                    str(path.resolve()),
                    DiskVisionFrame(
                        grid[index].cpu(),
                        output.pooler_output[index].detach().cpu(),
                        tuple(layer[index].detach().cpu() for layer in deep_splits),
                    ),
                )
    finally:
        writer.close()
    print(f"[vision-export] saved {len(paths)} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
