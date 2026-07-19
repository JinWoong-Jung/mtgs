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
    ):
        super().__init__()
        self.vlm = vlm
        self.direct_yes_no_token_ids = direct_yes_no_token_ids

    def close(self) -> None:
        self.vlm.close()

    def forward(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        task_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> GenerativeOutput:
        out = self.vlm(model_inputs)                 # backbone computes CE loss from labels
        return GenerativeOutput(loss=out.loss)

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
