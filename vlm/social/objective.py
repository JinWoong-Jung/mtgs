"""Generative yes/no objective for the text-evidence social-gaze VLM.

The production VLM is a single generative model: the MTGS graph evidence enters the
prompt as natural-language text, and the model is SFT'd to emit the one-token answer
``yes`` or ``no``. Evaluation runs one prompt forward and reads
``sigmoid(logit_yes - logit_no)`` at the answer position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from vlm.social.model import TextGenerativeVLM


def generative_answer_token_ids(tokenizer) -> tuple[int, int]:
    """Return exact one-token ids for text-SFT targets ``yes`` and ``no``.

    The assistant chat template already supplies its own boundary token, so the
    supervised target is ``yes``/``no`` without a leading whitespace token.
    """
    ids = []
    for answer in ("yes", "no"):
        encoded = tokenizer.encode(answer, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"generative answer {answer!r} must be one token, got {encoded}"
            )
        ids.append(int(encoded[0]))
    if ids[0] == ids[1]:
        raise ValueError(f"yes/no generative answer tokens must be distinct, got {ids}")
    return ids[0], ids[1]


@dataclass
class GenerativeOutput:
    loss: torch.Tensor | None = None
    prob: torch.Tensor | None = None      # [B] eval probability (candidate scoring)


class GenerativeObjective(nn.Module):
    """Generative yes/no objective for the text-evidence VLM.

    * train: standard next-token CE on the one-token ``yes`` or ``no`` target.
    * eval: one prompt forward, then ``sigmoid(logit_yes - logit_no)`` at the
      answer position.
    """

    def __init__(
        self,
        vlm: TextGenerativeVLM,
        *,
        direct_yes_no_token_ids: tuple[int, int] | None = None,
        pos_weight_by_task_id: torch.Tensor | None = None,
    ):
        super().__init__()
        self.vlm = vlm
        self.direct_yes_no_token_ids = direct_yes_no_token_ids
        # [num_tasks] positive-class loss weight indexed by SOCIAL_TASK_ID; ``None`` (or
        # an all-ones tensor) leaves the CE unweighted. Only positive (label==1) examples
        # are up-weighted, to counter the "no"-heavy answer distribution that drives the
        # VLM to under-predict "yes". Registered as a buffer so it follows .to(device).
        if pos_weight_by_task_id is None:
            self.register_buffer("pos_weight_by_task_id", None, persistent=False)
        else:
            self.register_buffer(
                "pos_weight_by_task_id",
                pos_weight_by_task_id.to(dtype=torch.float),
                persistent=False,
            )

    def close(self) -> None:
        self.vlm.close()

    def forward(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        task_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> GenerativeOutput:
        model_inputs = self._maybe_attach_pos_weight(model_inputs, task_ids, labels)
        out = self.vlm(model_inputs)                 # backbone computes CE loss from labels
        return GenerativeOutput(loss=out.loss)

    def _maybe_attach_pos_weight(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        task_ids: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> Mapping[str, torch.Tensor]:
        """Add a per-example ``pair_pos_weight`` [B] so positive pairs' CE is up-weighted.

        A positive pair on task ``t`` gets weight ``pos_weight_by_task_id[t]``; every
        negative pair (and every example when weighting is disabled) gets weight 1.0.
        """
        if self.pos_weight_by_task_id is None:
            return model_inputs
        if task_ids is None or labels is None:
            raise ValueError(
                "loss.pos_weight is configured but task_ids/pair_labels were not supplied"
            )
        weight = self.pos_weight_by_task_id.to(task_ids.device)[task_ids]
        pair_pos_weight = torch.where(
            labels.to(task_ids.device) == 1, weight, torch.ones_like(weight)
        )
        return {**model_inputs, "pair_pos_weight": pair_pos_weight}

    @torch.no_grad()
    def score(self, model_inputs: Mapping[str, torch.Tensor], num_pairs: int) -> torch.Tensor:
        """Return P(yes) via direct text answer logits at the answer position."""
        if self.direct_yes_no_token_ids is None:
            raise ValueError("generative scoring requires direct_yes_no_token_ids")
        if not isinstance(self.vlm, TextGenerativeVLM):
            raise TypeError("direct yes/no scoring requires TextGenerativeVLM")
        yes_id, no_id = self.direct_yes_no_token_ids
        answer_logits = self.vlm.direct_answer_logits(
            model_inputs, yes_token_id=yes_id, no_token_id=no_id
        )
        if answer_logits.shape != (num_pairs, 2):
            raise ValueError(
                f"direct answer logits must be ({num_pairs},2), got {tuple(answer_logits.shape)}"
            )
        return torch.sigmoid((answer_logits[:, 0] - answer_logits[:, 1]).float())
