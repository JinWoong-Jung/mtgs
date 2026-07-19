from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vlm.social.data import SocialSample
from vlm.social.objective import GenerativeObjective, generative_answer_token_ids
from vlm.social.input import SocialVLMInput
from vlm.social.model import (
    TextGenerativeVLM,
    make_text_generative_collate,
    make_text_generative_direct_eval_collate,
    make_text_generative_eval_collate,
)


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


def test_text_generative_reuse_forward_projects_only_supervised_tokens_for_next_token_ce():
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
            self.last_hidden = hidden
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
        # The reuse training path projects only the three supervised causal
        # positions per sequence, rather than a full [B, L, V] tensor.
        assert output.logits.shape == (6, 8)
        assert output.loss is not None and torch.isfinite(output.loss)
        mask = labels[:, 1:] != -100
        expected_logits = vlm.backbone.head(vlm.last_hidden[:, :-1][mask])
        expected = torch.nn.functional.cross_entropy(
            expected_logits.float(), labels[:, 1:][mask]
        )
        torch.testing.assert_close(output.logits, expected_logits)
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
    return SocialVLMInput(
        annotation=SocialSample(
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
        # Reuse training returns only these answer-position logits; the normal
        # backbone path retains its full sequence logits for compatibility.
        answer_mask = plain_batch["labels"][:, 1:] != -100
        plain_answer_logits = plain.logits[:, :-1][answer_mask]
        torch.testing.assert_close(
            reused.logits, plain_answer_logits, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(reused.loss, plain.loss, rtol=1e-4, atol=1e-4)
        assert wrapper.vision_cache_info() == (0, 1, 4, 1)

        # Direct evaluation must equal selecting only yes/no at the final prompt row,
        # without materialising the full [B,L,V] tensor in production.
        direct_batch = make_text_generative_direct_eval_collate(
            processor, reuse_vision=True
        )(items)
        yes_id, no_id = generative_answer_token_ids(processor.tokenizer)
        with torch.no_grad():
            direct_logits = wrapper.direct_answer_logits(
                direct_batch, yes_token_id=yes_id, no_token_id=no_id
            )
            full_prompt_logits = wrapper(direct_batch).logits
        last = direct_batch["attention_mask"].sum(dim=1).long() - 1
        expected = full_prompt_logits[
            torch.arange(len(items)), last
        ][:, [yes_id, no_id]]
        torch.testing.assert_close(direct_logits, expected, rtol=1e-4, atol=1e-4)
    finally:
        wrapper.close()


def test_direct_yesno_score_is_sigmoid_of_the_two_answer_logits():
    class _DirectTextVLM(TextGenerativeVLM):
        def direct_answer_logits(self, model_inputs, *, yes_token_id, no_token_id):
            assert (yes_token_id, no_token_id) == (3, 4)
            return torch.tensor([[3.0, 1.0], [1.0, 4.0]])

    obj = GenerativeObjective(
        _DirectTextVLM(_StubBackbone()), direct_yes_no_token_ids=(3, 4)
    )
    probability = obj.score({}, num_pairs=2)
    torch.testing.assert_close(
        probability, torch.sigmoid(torch.tensor([2.0, -3.0]))
    )


def test_generative_score_requires_direct_token_ids():
    obj = GenerativeObjective(TextGenerativeVLM(_StubBackbone()))
    with pytest.raises(ValueError, match="direct_yes_no_token_ids"):
        obj.score({}, num_pairs=1)


def test_text_direct_eval_collate_has_one_prompt_per_pair():
    processor = _local_qwen_processor()
    item = _unmarked_text_item()
    batch = make_text_generative_direct_eval_collate(
        processor, reuse_vision=True
    )([item, item])
    assert batch["num_pairs"] == 2
    assert "labels" not in batch
    assert batch["vision_reuse_indices"].tolist() == [0, 0]
    assert batch["vision_unique_grid_thw"].shape == (1, 3)


def test_generative_answer_token_ids_use_unspaced_supervision_tokens():
    class _Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"yes": [17], "no": [23]}[text]

    assert generative_answer_token_ids(_Tokenizer()) == (17, 23)
