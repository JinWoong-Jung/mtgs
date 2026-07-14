import random
from types import SimpleNamespace

import torch
import torch.nn as nn

from vlm.pair_head import PairGenerativeObjective, answer_loglik
from vlm.pair_model import GraphTokenProjector, graph_token_masks
from vlm.pair_prompt import (
    FINAL_PROBABILITY_QUESTION,
    GRAPH_EVIDENCE_INTRO,
    GRAPH_TOKENS,
    GRAPH_TOKEN_COUNT,
    QUESTION_BANK,
    SOCIAL_RELATION_TOKEN,
    compose_generative_prompt,
    generative_answer_json,
    parse_label_probability,
    validate_generative_pair_prompt,
)


def test_compositional_prompt_bank_samples_and_keeps_task_graph_tokens():
    box_a, box_b = [0.12, 0.18, 0.26, 0.42], [0.58, 0.21, 0.73, 0.46]
    for task in ("lah", "laeo", "sa"):
        validate_generative_pair_prompt(task, box_a, box_b)
        text = compose_generative_prompt(task, box_a, box_b, rng=random.Random(3))
        assert "0.12" in text and "0.58" in text                 # bbox coords substituted
        assert SOCIAL_RELATION_TOKEN not in text
        for k, token in enumerate(GRAPH_TOKENS):
            assert text.count(token) == (1 if k < GRAPH_TOKEN_COUNT[task] else 0)
        assert '[{"label": y}]' in text
        assert GRAPH_EVIDENCE_INTRO in text
        assert FINAL_PROBABILITY_QUESTION in text
        assert len(QUESTION_BANK[task]) == 10                    # ten question paraphrases


def test_answer_json_and_probability_parser():
    assert generative_answer_json(1) == '[{"label": 1}]'
    assert generative_answer_json(0) == '[{"label": 0}]'
    assert parse_label_probability('{"label": 0.83}') == 0.83
    assert parse_label_probability('{"label": 9.0}') == 1.0      # clamp
    assert parse_label_probability("no json", default=0.5) == 0.5


def test_graph_token_masks_allow_variable_presence():
    token_ids = {tok: 900 + i for i, tok in enumerate(GRAPH_TOKENS)}
    # SA uses only gtok0, gtok1
    row = [7, 900, 5, 901, 5]
    masks = graph_token_masks(torch.tensor([row]), token_ids)
    assert masks.shape == (1, len(row), 4)
    assert masks.sum().item() == 2
    assert masks[0, :, 2].sum().item() == 0                      # absent slots have no position


def test_graph_token_projector_shapes():
    proj = GraphTokenProjector(graph_dim=16, output_dim=8)
    out = proj(torch.randn(3, 4, 16))
    assert out.shape == (3, 4, 8)


def test_answer_loglik_and_score():
    logits = torch.zeros(2, 4, 5)
    logits[0, 2, 3] = 10.0
    labels = torch.full((2, 4), -100)
    labels[:, 3] = 3
    ll = answer_loglik(logits, labels)
    assert ll[0] > ll[1]

    class _FakeGenVLM(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.w = nn.Parameter(torch.randn(n, 4, 5))

        def forward(self, model_inputs):
            return SimpleNamespace(logits=self.w, loss=self.w.mean())

        def close(self):
            pass

    num_pairs = 3
    obj = PairGenerativeObjective(_FakeGenVLM(2 * num_pairs))
    lab = torch.full((2 * num_pairs, 4), -100)
    lab[:, 3] = 1
    prob = obj.score({"labels": lab}, num_pairs)
    assert prob.shape == (num_pairs,)
    assert bool(((prob >= 0) & (prob <= 1)).all())
