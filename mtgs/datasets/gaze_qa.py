# mtgs/datasets/gaze_qa.py
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class QAPair:
    batch_idx: int           # which item in the batch
    task: str                # "lah" | "laeo" | "sa"
    src_idx: int             # looker (LAH subject) or first person (LAEO/SA)
    dst_idx: int             # target (LAH object) or second person (LAEO/SA)
    label: int               # 1 = Yes, 0 = No
    src_bbox: Tuple[float, float, float, float] = field(default=(0., 0., 1., 1.))
    dst_bbox: Tuple[float, float, float, float] = field(default=(0., 0., 1., 1.))


class GazeQACollator:
    """Generates Yes/No QA pairs from a padded batch at the center frame.

    Pair convention (from mtgs_net.py):
        pairs = list(itertools.permutations(range(N), 2))
        pairs[k] = (src_k, dst_k)
        lah_labels[b, t, k] = 1  means  person src_k looks at person dst_k

    LAH uses ordered pairs (directed): (i→j) ≠ (j→i).
    LAEO/SA use unordered pairs (symmetric): combinations to avoid duplicates.

    Only pairs with explicit annotation (label 0 or 1) are included.
    Valid persons are right-aligned: indices [N - nv, N).
    Bboxes extracted from head_bboxes[b, t_c, idx] (normalized [x1,y1,x2,y2]).
    """

    _TASKS = [
        ("lah",  "lah_labels",   True),
        ("laeo", "laeo_labels",  False),
        ("sa",   "coatt_labels", False),
    ]

    def __call__(self, batch: dict) -> List[QAPair]:
        B = batch["lah_labels"].shape[0]
        T = batch["lah_labels"].shape[1]
        t_c = T // 2
        P = batch["lah_labels"].shape[2]
        N_padded = int(round((1 + math.sqrt(1 + 4 * P)) / 2))

        perm_to_k = {(s, d): k
                     for k, (s, d) in enumerate(itertools.permutations(range(N_padded), 2))}

        head_bboxes = batch.get("head_bboxes")  # (B, T, N, 4) or None

        all_pairs: List[QAPair] = []
        for b in range(B):
            nv = int(batch["num_valid_people"][b, 0].item())
            valid_start = N_padded - nv
            valid = range(valid_start, N_padded)

            for task, label_key, is_directed in self._TASKS:
                labels = batch[label_key][b, t_c]  # (P,)
                pairs_iter = (itertools.permutations(valid, 2) if is_directed
                              else itertools.combinations(valid, 2))
                for src_k, dst_k in pairs_iter:
                    k = perm_to_k[(src_k, dst_k)]
                    lbl = int(labels[k].item())
                    if lbl == -1:
                        continue
                    src_bbox = tuple(head_bboxes[b, t_c, src_k].tolist()) \
                        if head_bboxes is not None else (0., 0., 1., 1.)
                    dst_bbox = tuple(head_bboxes[b, t_c, dst_k].tolist()) \
                        if head_bboxes is not None else (0., 0., 1., 1.)
                    all_pairs.append(QAPair(b, task, src_k, dst_k, lbl, src_bbox, dst_bbox))

        return all_pairs
