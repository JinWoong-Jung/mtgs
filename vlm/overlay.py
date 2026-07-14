from __future__ import annotations
"""Overlay rendering helpers for VLM Stage-2 (ported from peer sgg/vlm.py).

Provides PIL-based bounding-box drawing utilities and graph-informed or
graph-free overlay construction for the VLM specialist inputs.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from mtgs.utils.image import IMG_MEAN, IMG_STD


CAND_COLORS = ["blue", "green", "gold"]


SOURCE_COLOR = "red"


PARTNER_COLOR = "blue"


CONTEXT_COLOR = "gray"


def denormalize_to_pil(img_chw, mean, std):
    """img_chw: [3,H,W] normalized tensor -> PIL RGB image."""
    import torch
    mean = torch.tensor(mean, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(std, device=img_chw.device).view(3, 1, 1)
    x = (img_chw.float() * std + mean).clamp(0, 1)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


_FONT_CACHE = {}


_FONT_PATHS = [
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size):
    if size not in _FONT_CACHE:
        f = None
        for p in _FONT_PATHS:
            try:
                f = ImageFont.truetype(p, size); break
            except OSError:
                continue
        _FONT_CACHE[size] = f or ImageFont.load_default()
    return _FONT_CACHE[size]


def _draw(draw, bbox_norm, W, H, color, label, width, font_size=26):
    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    box = [x1 * W, y1 * H, x2 * W, y2 * H]
    draw.rectangle(box, outline=color, width=width)
    if label:
        f = _font(font_size)
        try:
            l, t, r, b = draw.textbbox((0, 0), label, font=f)
            tw, th = r - l, b - t
        except Exception:
            tw, th = int(len(label) * font_size * 0.6), font_size
        bw, bh = tw + 6, th + 5
        above = box[1] - bh
        ty = above if above >= 0 else box[1] + width   # inside box-top if it'd clip
        tx = min(max(box[0], 0), max(W - bw, 0))        # keep within right edge
        draw.rectangle([tx, ty, tx + bw, ty + bh], fill=color)   # colored bg
        draw.text((tx + 3, ty + 1), label, fill="white", font=f)


def build_pointer_image(image_pil, task, i, j, cand_slots, bboxes_norm,
                        valid_slots, labels, null_in_slot, null_out_slot):
    """Graph-informed overlay. Only valid persons are drawn; labels are the
    contiguous P1..Pk display names. cand_slots: top-K indices into [N+2]."""
    overlay = image_pil.convert("RGB").copy()
    W, H = overlay.size
    draw = ImageDraw.Draw(overlay)
    person_cands = [int(k) for k in cand_slots if int(k) in valid_slots and int(k) != i]

    if task == "lah":
        _draw(draw, bboxes_norm[i], W, H, SOURCE_COLOR, labels[i], 4)
        for k, c in zip(person_cands[:3], CAND_COLORS):
            _draw(draw, bboxes_norm[k], W, H, c, labels[k], 3)
        for k in valid_slots:
            if k != i and k not in person_cands[:3]:
                _draw(draw, bboxes_norm[k], W, H, CONTEXT_COLOR, labels[k], 2)
    else:  # laeo / sa : highlight the pair
        _draw(draw, bboxes_norm[i], W, H, SOURCE_COLOR, labels[i], 4)
        _draw(draw, bboxes_norm[j], W, H, PARTNER_COLOR, labels[j], 4)
        for k in valid_slots:
            if k != i and k != j:
                _draw(draw, bboxes_norm[k], W, H, CONTEXT_COLOR, labels[k], 2)
    return overlay


def build_token_overlay(image_pil, task, a, b, bboxes_norm, labels):
    """Query overlay for the token path: draw ONLY the queried pair's head boxes —
    A = red box, B = blue box. No gaze markers (gaze location is supplied as the
    <hmtok> heatmap soft-token instead). Other people stay un-boxed to avoid clutter
    on crowd frames. (a, b) = injection.query_slots(rec); labels = {slot: label}."""
    overlay = image_pil.convert("RGB").copy()
    W, H = overlay.size
    draw = ImageDraw.Draw(overlay)
    _draw(draw, bboxes_norm[a], W, H, SOURCE_COLOR, labels[a], 4)    # red  = A
    _draw(draw, bboxes_norm[b], W, H, PARTNER_COLOR, labels[b], 4)   # blue = B
    return overlay


def build_canonical_pair_overlay(image_pil, bbox_a, bbox_b):
    """Draw the new pair pipeline's task-independent Person A/B overlay.

    Person A is always RED and Person B is always BLUE. Directional semantics have
    already been canonicalised by :class:`vlm.pair_dataset.PairSample`, so this helper
    never receives a task and never swaps the boxes. The source image is not mutated.
    """
    overlay = image_pil.convert("RGB").copy()
    W, H = overlay.size
    draw = ImageDraw.Draw(overlay)
    _draw(draw, bbox_a, W, H, SOURCE_COLOR, "A", 4)
    _draw(draw, bbox_b, W, H, PARTNER_COLOR, "B", 4)
    return overlay


def build_overlay_pair(image_pil, i, j, bboxes_norm, labels):
    """Graph-FREE overlay: draw ONLY the query pair — source i=red, target j=blue.
    Other people stay un-boxed (the full scene is still visible in the photo).
    This avoids clutter on crowd frames (videocoatt up to ~22 people) where boxing
    everyone obscures the scene and buries the queried pair. Other people's info,
    if needed, is supplied as TEXT in the prompt, not drawn."""
    overlay = image_pil.convert("RGB").copy()
    W, H = overlay.size
    draw = ImageDraw.Draw(overlay)
    _draw(draw, bboxes_norm[i], W, H, SOURCE_COLOR, labels[i], 4)    # red  = source
    _draw(draw, bboxes_norm[j], W, H, PARTNER_COLOR, labels[j], 4)   # blue = target
    return overlay


FRAME_PALETTE = ["red", "blue", "green", "gold", "magenta", "cyan", "orange",
                 "lime", "purple", "brown", "pink", "teal"]


def build_frame_overlay(image_pil, slots, bboxes_norm, labels):
    """Frame pipeline overlay: draw ALL listed persons' head boxes with P1..PK labels,
    cycling a colour palette. (slots) = ordered valid slot indices; labels = {slot: name}.
    The plain frame.png (no boxes) is reused; boxing happens here at load time."""
    overlay = image_pil.convert("RGB").copy()
    W, H = overlay.size
    draw = ImageDraw.Draw(overlay)
    for k, slot in enumerate(slots):
        _draw(draw, bboxes_norm[slot], W, H, FRAME_PALETTE[k % len(FRAME_PALETTE)],
              labels[slot], 3, font_size=20)
    return overlay


def display_labels(person_mask):
    """Contiguous human-facing labels for valid persons: {slot: 'P1'..'Pk'}."""
    valid = [s for s in range(person_mask.shape[-1]) if bool(person_mask[s])]
    return valid, {s: f"P{k + 1}" for k, s in enumerate(valid)}
