from __future__ import annotations
"""Multi-person prompt for experiment F. One <ptok> per person, no question text —
the social head reads per-person hidden states, so the LM only needs the scene image +
per-person slots. Person order in the prompt == order of `labels`/`bboxes` rows, which is
the order the <ptok> mask selects hidden states in (row-major)."""

import torch

PTOK = "<ptok>"


def _fmt_box(bb) -> str:
    return (f"[{float(bb[0]):.2f},{float(bb[1]):.2f},"
            f"{float(bb[2]):.2f},{float(bb[3]):.2f}]")


def frame_prompt(labels: list[str], bboxes: torch.Tensor) -> str:
    """labels[k] = person display label (e.g. 'P1'); bboxes = (N,4) head boxes.
    Emits exactly len(labels) <ptok> placeholders, one per person, in order."""
    parts = [f"Person {labels[k]} {PTOK} has head box {_fmt_box(bboxes[k])}."
             for k in range(len(labels))]
    return "Scene with people. " + " ".join(parts)
