import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vlm.pair_dataset import PairSample
from vlm.pair_features import TextGraphEvidence
from vlm.pair_head import PairGenerativeObjective, answer_loglik
from vlm.pair_input import PairVLMInput
from vlm.pair_model import (
    GraphTokenProjector,
    TextGenerativeVLM,
    graph_token_masks,
    make_text_generative_collate,
    make_text_generative_eval_collate,
)
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


class _StubBackbone(torch.nn.Module):
    """Minimal backbone: returns .loss and .logits, ignores vision/graph kwargs."""
    def __init__(self, vocab=32):
        super().__init__()
        self.vocab = vocab
        self.lin = torch.nn.Embedding(vocab, vocab)

    def forward(self, input_ids=None, labels=None, **kw):
        logits = self.lin(input_ids)                      # [B,L,V]
        out = type("O", (), {})()
        out.logits = logits
        out.loss = None
        if labels is not None:
            shift = logits[:, :-1].reshape(-1, self.vocab)
            tgt = labels[:, 1:].reshape(-1)
            out.loss = torch.nn.functional.cross_entropy(shift, tgt.clamp_min(0),
                                                          ignore_index=-100)
        return out


def test_text_generative_vlm_runs_without_graph_features():
    vlm = TextGenerativeVLM(_StubBackbone())
    inp = {"input_ids": torch.randint(0, 32, (2, 6)),
           "labels": torch.randint(0, 32, (2, 6))}
    out = vlm(inp)
    assert out.logits.shape == (2, 6, 32)
    assert out.loss is not None


def test_text_generative_vlm_accepts_vision_cache_size_and_exposes_cache_info():
    vlm = TextGenerativeVLM(_StubBackbone(), vision_cache_size=4)
    try:
        info = vlm.vision_cache_info()
        assert info.max_items == 4
        assert info.curr_items == 0
    finally:
        vlm.close()


def test_text_generative_reuse_forward_returns_logits_and_next_token_ce():
    class _ReuseBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 8, bias=False)

        def get_output_embeddings(self):
            return self.head

    class _ReuseVLM(TextGenerativeVLM):
        def _forward_with_reused_vision(self, kwargs, device):
            assert "labels" not in kwargs
            batch, length = kwargs["input_ids"].shape
            hidden = torch.randn(batch, length, 4, device=device, requires_grad=True)
            return SimpleNamespace(
                last_hidden_state=hidden,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

    vlm = _ReuseVLM(_ReuseBackbone(), vision_cache_size=1)
    labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    try:
        output = vlm(
            {
                "input_ids": torch.zeros(2, 4, dtype=torch.long),
                "labels": labels,
                "vision_reuse_indices": torch.tensor([0, 0]),
            }
        )
        assert output.logits.shape == (2, 4, 8)
        assert output.loss is not None and torch.isfinite(output.loss)
        expected = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, 8), labels[:, 1:].reshape(-1)
        )
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        assert vlm.backbone.head.weight.grad is not None
    finally:
        vlm.close()


def _local_qwen_processor():
    transformers = pytest.importorskip("transformers")
    model_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    snapshots = sorted((model_root / "snapshots").glob("*"))
    if not snapshots:
        pytest.skip("local Qwen3-VL processor cache is unavailable")
    return transformers.AutoProcessor.from_pretrained(
        snapshots[-1], local_files_only=True
    )


def _unmarked_text_item():
    return PairVLMInput(
        annotation=PairSample(
            sid="tiny",
            task="lah",
            person_i=0,
            person_j=1,
            label=1,
            raw_i=1,
            raw_j=0,
        ),
        image=Image.new("RGB", (56, 56), "black"),
        prompt="Is Person A looking at Person B? Answer yes or no.",
        evidence=TextGraphEvidence(task="lah", p_ab=0.5),
        vision_cache_key="/split/tiny/frame.png",
    )


def test_make_text_generative_collate_reuse_flag_produces_vision_reuse_keys():
    processor = _local_qwen_processor()
    item = _unmarked_text_item()

    batch = make_text_generative_collate(processor, reuse_vision=True)([item, item])

    assert batch["vision_reuse_indices"].tolist() == [0, 0]
    assert batch["vision_unique_grid_thw"].shape == (1, 3)
    assert batch["vision_frame_ids"] == ("/split/tiny/frame.png",)


def test_make_text_generative_eval_collate_reuse_dedups_across_2b_candidates():
    processor = _local_qwen_processor()
    item = _unmarked_text_item()

    batch = make_text_generative_eval_collate(processor, reuse_vision=True)([item, item])

    assert batch["num_pairs"] == 2
    assert batch["vision_reuse_indices"].tolist() == [0, 0, 0, 0]
    assert batch["vision_unique_grid_thw"].shape == (1, 3)
    assert batch["vision_frame_ids"] == ("/split/tiny/frame.png",)


def test_text_objective_score_returns_probability_per_pair():
    vlm = TextGenerativeVLM(_StubBackbone())
    obj = PairGenerativeObjective(vlm)
    # 2 pairs -> [2B]=4 rows: pos_0,pos_1,neg_0,neg_1
    inp = {"input_ids": torch.randint(0, 32, (4, 6)),
           "labels": torch.randint(0, 32, (4, 6))}
    prob = obj.score(inp, num_pairs=2)
    assert prob.shape == (2,)
    assert torch.all((prob >= 0) & (prob <= 1))


def test_text_score_prefers_yes_when_backbone_prefers_yes():
    """logP(yes)-logP(no) sign wiring: a backbone that always emits high logit for a fixed
    'yes' token id must score the yes-candidate above the no-candidate."""
    import torch
    from vlm.pair_head import PairGenerativeObjective
    from vlm.pair_model import TextGenerativeVLM

    YES_ROW, NO_ROW = 5, 6      # pretend token ids; yes-candidate label row uses YES_ROW

    class Prefer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # TextGenerativeVLM.forward resolves the target device via
            # next(self.backbone.parameters()); real backbones (Qwen etc.) always have
            # parameters, so give the stub one inert buffer-like param to match that contract.
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, labels=None, **kw):
            B, L = input_ids.shape
            logits = torch.zeros(B, L, 8)
            logits[..., YES_ROW] = 3.0        # always confident about the yes token
            out = type("O", (), {})()
            out.logits, out.loss = logits, None
            return out

    obj = PairGenerativeObjective(TextGenerativeVLM(Prefer()))
    # pair 0: yes-candidate labels point at YES_ROW; no-candidate at NO_ROW
    labels = torch.full((2, 4), -100)
    labels[0, 1:] = YES_ROW      # positive (yes) candidate
    labels[1, 1:] = NO_ROW       # negative (no) candidate
    inp = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "labels": labels}
    prob = obj.score(inp, num_pairs=1)
    assert prob.item() > 0.5


def test_text_generative_vlm_reuse_matches_no_reuse_numerically():
    """Frozen-vision reuse must preserve answer-position logits exactly enough."""
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    model_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    snapshots = sorted((model_root / "snapshots").glob("*"))
    if not snapshots:
        pytest.skip("local Qwen3-VL processor cache is unavailable")

    processor = transformers.AutoProcessor.from_pretrained(
        snapshots[-1], local_files_only=True
    )
    base_config = transformers.AutoConfig.from_pretrained(
        snapshots[-1], local_files_only=True
    )
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": len(processor.tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "max_position_embeddings": 2048,
            "pad_token_id": processor.tokenizer.pad_token_id,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": (0, 1),
        },
        image_token_id=base_config.image_token_id,
        video_token_id=base_config.video_token_id,
        vision_start_token_id=base_config.vision_start_token_id,
        vision_end_token_id=base_config.vision_end_token_id,
    )
    model = transformers.Qwen3VLForConditionalGeneration(config)
    model.requires_grad_(False)
    targets = [
        name
        for name, _ in model.named_modules()
        if "language_model" in name
        and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    ]
    backbone = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    wrapper = TextGenerativeVLM(backbone, vision_cache_size=4).eval()
    items = [_unmarked_text_item(), _unmarked_text_item()]
    plain_batch = make_text_generative_collate(processor, reuse_vision=False)(items)
    reused_batch = make_text_generative_collate(processor, reuse_vision=True)(items)

    try:
        assert torch.equal(plain_batch["input_ids"], reused_batch["input_ids"])
        assert torch.equal(plain_batch["labels"], reused_batch["labels"])
        with torch.no_grad():
            plain = wrapper(plain_batch)
            reused = wrapper(reused_batch)

        # Causal logits at t-1 predict the supervised answer token at t.
        answer_mask = plain_batch["labels"][:, 1:] != -100
        plain_answer_logits = plain.logits[:, :-1][answer_mask]
        reused_answer_logits = reused.logits[:, :-1][answer_mask]
        torch.testing.assert_close(
            reused_answer_logits, plain_answer_logits, rtol=1e-4, atol=1e-4
        )
        assert wrapper.vision_cache_info() == (0, 1, 4, 1)
    finally:
        wrapper.close()
