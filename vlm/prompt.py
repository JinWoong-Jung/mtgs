from __future__ import annotations
"""Prompt-building helpers for VLM Stage-2 (token path)."""

from vlm.injection import GTOK


TASKS = ("lah", "laeo", "sa")


TASK_ID = {t: i for i, t in enumerate(TASKS)}


def _fmt_box(bb):
    return (f"[{float(bb[0]):.2f},{float(bb[1]):.2f},"
            f"{float(bb[2]):.2f},{float(bb[3]):.2f}]")


def token_prompt(task, li, lj, bb_i, bb_j):
    """Inline graph-token prompt. Emits exactly TOK_COUNT[task] '<gtok>' placeholders,
    in the SAME order as gather_feats(gf, task, i, j). bb_i/bb_j = head boxes [x1,y1,x2,y2].

    <gtok> order per task (must match vlm.injection.gather_feats):
      lah  : SRC(li), TGT(lj), EDGE_FWD             (3)
      laeo : SRC(li), SRC(lj), EDGE_FWD, EDGE_BWD   (4)
      sa   : SRC(li), SRC(lj), NULL_IN(li), NULL_IN(lj), EDGE_FWD, EDGE_BWD  (6)
    """
    red = f"Person {li} {GTOK} has head box {_fmt_box(bb_i)}. "
    blue = f"Person {lj} {GTOK} has head box {_fmt_box(bb_j)}. "
    if task == "lah":
        return (red + blue
                + f"Their gaze relation: {GTOK}. "
                + f"Is {li} looking at {lj}? Answer only: yes or no.")
    if task == "laeo":
        return (red + blue
                + f"Their mutual gaze relation: {GTOK} {GTOK}. "
                + f"Are {li} and {lj} looking at each other? Answer only: yes or no.")
    if task == "sa":
        return (red + blue
                + f"Scene-gaze cues: {GTOK} {GTOK}. Their relation: {GTOK} {GTOK}. "
                + f"Are {li} and {lj} looking at the same thing or person? "
                + f"Answer only: yes or no.")
    raise ValueError(f"unknown task {task!r}")


