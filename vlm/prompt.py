from __future__ import annotations
"""Prompt-building helpers for VLM Stage-2 (token path).

The prompt describes the visual overlay (A/B head boxes only) and injects two
latent modalities as placeholder tokens:
  <gtok>  graph node/edge embeddings (see vlm.injection.gather_feats)
  <hmtok> predicted gaze heatmap per queried person (see gather_heatmaps)
"""

from vlm.injection import GTOK, HMTOK, PANC


TASKS = ("lah", "laeo", "sa")


TASK_ID = {t: i for i, t in enumerate(TASKS)}


def _fmt_box(bb):
    return (f"[{float(bb[0]):.2f},{float(bb[1]):.2f},"
            f"{float(bb[2]):.2f},{float(bb[3]):.2f}]")


def frame_prompt(labels, bboxes):
    """Frame-level prompt for the frame pipeline: describe EVERY listed person once
    (per-person graph <gtok> + gaze-heatmap <hmtok> soft-tokens), THEN a trailing
    per-person anchor block ('P1 <panc> P2 <panc> ...') whose hidden states the readout
    head reads as person tokens. No yes/no question — social predictions come from the
    head, not from decoded text.

    Why the anchor block is at the END (OmniGF-style whole-context anchor): the LM is
    causal, so an anchor placed inline right after person Pi could NOT attend to the
    later persons' descriptions. Emitting ALL <panc> after ALL descriptions lets every
    anchor attend to the full multi-person context (image + every person's tokens) —
    exactly what relational (who-looks-at-whom) reasoning needs.

      labels : list of display names ["P1", "P2", ...] in person order
      bboxes : list of head boxes [x1,y1,x2,y2] (normalised), same order

    Emits exactly len(labels) of each: <gtok>, <hmtok> (in the descriptions) and <panc>
    (in the trailing block) — matching injection.gather_frame_feats /
    gather_frame_heatmaps order and the head's anchor gather (<panc>, row-major)."""
    people = [
        f"Person {lab}: graph {GTOK}, gaze heatmap {HMTOK}, head box {_fmt_box(bb)}."
        for lab, bb in zip(labels, bboxes)
    ]
    anchors = " ".join(f"{lab} {PANC}" for lab in labels)
    return (
        f"This image shows {len(labels)} people, each marked with a colored labelled "
        f"head box. " + " ".join(people)
        + " Analyse each person's gaze direction and who or what they are looking at, to "
        "reason about who looks at whom, who look at each other, and who share attention. "
        f"Per-person summary: {anchors}"
    )


def token_prompt(task, la, lb, bb_a, bb_b):
    """Inline soft-token prompt for query slots (a, b) = injection.query_slots(rec).
    la = first-named person A (the LOOKER for lah), lb = person B; bb_a/bb_b are the
    A/B head boxes [x1,y1,x2,y2] (normalised) drawn as the RED/BLUE boxes in the image.

    Emits exactly TOK_COUNT[task] '<gtok>' + HM_COUNT[task] '<hmtok>' placeholders,
    matching gather_feats(gf,task,a,b) / gather_heatmaps(gf,task,a,b) in order.

      <gtok> order   lah : SRC(A) [red], TGT(B) [blue], EDGE_FWD(A→B) [relation],
                           SRC(A), TGT(B) [question]                             (5)
                     laeo: SRC(A) [red], SRC(B) [blue],
                           EDGE_FWD(A→B), EDGE_BWD(B→A) [relations],
                           SRC(A), SRC(B) [question]                             (6)
                     sa  : SRC(A) [red], SRC(B) [blue],
                           NULL_IN(A), NULL_IN(B) [scene-gaze],
                           SRC(A), SRC(B) [question]                             (6)
      <hmtok> order  lah : hm(A)                               (1)
                     laeo/sa: hm(A), hm(B)                     (2)
    """
    red = f"Person {la} {GTOK} is in the RED box {_fmt_box(bb_a)}. "
    blue = f"Person {lb} {GTOK} is in the BLUE box {_fmt_box(bb_b)}. "
    if task == "lah":
        # single directed edge A->B (EDGE_FWD)
        return (red + blue
                + f"The gaze relation from {la} to {lb}: {GTOK}. "
                + f"Person {la} gaze heatmap: {HMTOK}. "
                + f"Is {la} {GTOK} looking at {lb} {GTOK}? Answer only: yes or no.")
    if task == "laeo":
        # both directions: A->B (EDGE_FWD) then B->A (EDGE_BWD)
        return (red + blue
                + f"The gaze relation from {la} to {lb}: {GTOK}. "
                + f"The gaze relation from {lb} to {la}: {GTOK}. "
                + f"Person {la} gaze heatmap: {HMTOK}, Person {lb} gaze heatmap: {HMTOK}. "
                + f"Are {la} {GTOK} and {lb} {GTOK} looking at each other? Answer only: yes or no.")
    if task == "sa":
        # each person's scene-gaze channel (NULL_IN): A then B
        return (red + blue
                + f"The scene-gaze of {la}: {GTOK}. The scene-gaze of {lb}: {GTOK}. "
                + f"Person {la} gaze heatmap: {HMTOK}, Person {lb} gaze heatmap: {HMTOK}. "
                + f"Are {la} {GTOK} and {lb} {GTOK} looking at the same thing or person? "
                + f"Answer only: yes or no.")
    raise ValueError(f"unknown task {task!r}")
