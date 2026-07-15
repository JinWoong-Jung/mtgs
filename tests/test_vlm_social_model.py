from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vlm.social.data import SocialSample
from vlm.social.evidence import GraphBatch, GraphEvidence
from vlm.social.input import SocialVLMInput
from vlm.social.model import (
    SocialVLM,
    make_social_collate,
    out_of_place_soft_token_replace,
    placeholder_masks,
    prepare_social_tokens,
)
from vlm.social.prompt import SOCIAL_SPECIAL_TOKENS, task_prompt


def _token_ids():
    return {token: index + 10 for index, token in enumerate(SOCIAL_SPECIAL_TOKENS)}


def _input_ids(batch_size=2):
    ids = torch.tensor([3, *range(10, 17), 4], dtype=torch.long)
    return ids.repeat(batch_size, 1)


def _graph_batch() -> GraphBatch:
    generator = torch.Generator().manual_seed(11)
    return GraphBatch(
        tasks=("lah", "sa"),
        person_features=torch.randn(2, 2, 3, 4, generator=generator),
        person_channel_present=torch.tensor([
            [[True, False, False], [False, True, False]],
            [[True, False, True], [True, False, True]],
        ]),
        relation_features=torch.randn(2, 2, 4, generator=generator),
        relation_present=torch.tensor([[True, False], [True, True]]),
        heatmap_features=torch.randn(2, 2, 16, 16, generator=generator),
        heatmap_present=torch.tensor([[True, False], [True, True]]),
        graph_logits=torch.tensor([0.1, -0.2]),
    )


class _StubLanguageModel(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embed_tokens = embedding
        self.norm = nn.LayerNorm(embedding.embedding_dim)
        self.last_inputs = None

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        self.last_inputs = inputs_embeds
        return self.norm(torch.tanh(inputs_embeds.cumsum(dim=1)))


class _StubModelContainer(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model


class _StubBackbone(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__()
        embedding = nn.Embedding(vocab_size, hidden_size)
        self.model = _StubModelContainer(_StubLanguageModel(embedding))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size)
        )

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        del kwargs
        embeds = self.get_input_embeddings()(input_ids)
        hidden = self.model.language_model(inputs_embeds=embeds)
        return SimpleNamespace(
            hidden_states=(hidden,) if output_hidden_states else None,
            logits=hidden,
        )


def test_placeholder_masks_require_exactly_one_ordered_occurrence():
    masks, positions = placeholder_masks(_input_ids(), _token_ids())
    assert masks.shape == (2, 9, 7)
    assert positions.tolist() == [list(range(1, 8)), list(range(1, 8))]

    missing = _input_ids()
    missing[:, 3] = 1
    with pytest.raises(ValueError, match="once"):
        placeholder_masks(missing, _token_ids())

    out_of_order = _input_ids()
    out_of_order[:, [1, 2]] = out_of_order[:, [2, 1]]
    with pytest.raises(ValueError, match="out of order"):
        placeholder_masks(out_of_order, _token_ids())


def test_soft_token_replacement_is_out_of_place_and_differentiable():
    base = torch.randn(2, 9, 8, requires_grad=True)
    original = base.detach().clone()
    masks, _ = placeholder_masks(_input_ids(), _token_ids())
    tokens = torch.randn(2, 7, 8, requires_grad=True)

    replaced = out_of_place_soft_token_replace(base, masks, tokens)
    torch.testing.assert_close(base, original)
    for batch in range(2):
        torch.testing.assert_close(replaced[batch, 1:8], tokens[batch])
    replaced.square().mean().backward()
    assert base.grad is not None and base.grad.abs().sum() > 0
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0


def test_evidence_wrapper_injects_six_slots_and_reads_social_hidden_with_gradients():
    backbone = _StubBackbone()
    wrapper = SocialVLM(
        backbone,
        _token_ids(),
        graph_dim=4,
        graph_hidden_dim=32,
        heatmap_conv_dim=32,
    )
    try:
        result = wrapper({"input_ids": _input_ids()}, _graph_batch())
        assert result.h_social.shape == (2, 16)
        assert result.evidence_tokens.shape == (2, 6, 16)
        assert result.backbone_output.hidden_states is None

        injected = backbone.model.language_model.last_inputs
        assert injected is not None
        torch.testing.assert_close(injected[:, 1:7], result.evidence_tokens)
        expected_social = wrapper.social_query.view(1, -1).expand(2, -1)
        torch.testing.assert_close(injected[:, 7], expected_social)

        result.h_social.square().mean().backward()
        assert wrapper.social_query.grad is not None
        assert wrapper.social_query.grad.abs().sum() > 0
        assert wrapper.projector.person.na_channels.grad is not None
        assert wrapper.projector.person.na_channels.grad.abs().sum() > 0
    finally:
        wrapper.close()


def _single_lah_input() -> SocialVLMInput:
    annotation = SocialSample(
        sid="tiny",
        task="lah",
        person_i=0,
        person_j=1,
        label=1,
        raw_i=1,
        raw_j=0,
    )
    graph = _graph_batch()
    evidence = GraphEvidence(
        task="lah",
        person_features=graph.person_features[0],
        person_channel_present=graph.person_channel_present[0],
        relation_features=graph.relation_features[0],
        relation_present=graph.relation_present[0],
        heatmap_features=graph.heatmap_features[0],
        heatmap_present=graph.heatmap_present[0],
        graph_logit=graph.graph_logits[0],
    )
    return SocialVLMInput(
        annotation=annotation,
        image=Image.new("RGB", (56, 56), "black"),
        prompt=task_prompt("lah"),
        evidence=evidence,
    )


def test_actual_tiny_qwen_forward_keeps_vision_deepstack_and_mrope():
    transformers = pytest.importorskip("transformers")
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
    model.gradient_checkpointing_enable()
    model.train()
    assert model.is_gradient_checkpointing
    token_ids = prepare_social_tokens(processor.tokenizer, model)
    batch = make_social_collate(processor)([_single_lah_input()])
    graph = batch.pop("pair_graph")
    batch.pop("task_ids")
    batch.pop("pair_labels")
    batch.pop("eval_keys")
    assert "pixel_values" in batch
    assert "image_grid_thw" in batch
    assert "mm_token_type_ids" in batch

    captured = {}

    def capture_language_inputs(module, args, kwargs):
        del module, args
        captured["deepstack"] = kwargs.get("deepstack_visual_embeds")
        captured["visual_mask"] = kwargs.get("visual_pos_masks")

    wrapper = SocialVLM(
        model,
        token_ids,
        graph_dim=4,
        graph_hidden_dim=32,
        heatmap_conv_dim=32,
    )
    capture_handle = model.model.language_model.register_forward_pre_hook(
        capture_language_inputs, with_kwargs=True
    )
    try:
        with patch.object(
            model.model,
            "get_rope_index",
            wraps=model.model.get_rope_index,
        ) as get_rope_index:
            result = wrapper(batch, graph)
        assert get_rope_index.call_count == 1
        assert result.h_social.shape == (1, 32)
        assert torch.isfinite(result.h_social).all()
        assert result.backbone_output.hidden_states is None
        assert captured["visual_mask"] is not None
        assert captured["visual_mask"].any()
        assert captured["deepstack"] is not None
        assert len(captured["deepstack"]) == 2
        assert all(tensor.numel() > 0 for tensor in captured["deepstack"])
        result.h_social.square().mean().backward()
        assert wrapper.social_query.grad is not None
        assert wrapper.social_query.grad.abs().sum() > 0
    finally:
        capture_handle.remove()
        wrapper.close()


def test_actual_tiny_qwen_reuses_unmarked_vision_with_deepstack_and_mrope():
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
    model.gradient_checkpointing_enable()
    model.train()
    token_ids = prepare_social_tokens(processor.tokenizer, model)
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
    core_model = backbone.base_model.model.model

    source = _single_lah_input()
    unmarked = SocialVLMInput(
        annotation=source.annotation,
        image=source.image,
        prompt=task_prompt("lah"),
        evidence=source.evidence,
        vision_cache_key="/split/tiny/frame.png",
    )
    batch = make_social_collate(processor, reuse_vision=True)([unmarked, unmarked])
    graph = batch.pop("pair_graph")
    batch.pop("task_ids")
    batch.pop("pair_labels")
    batch.pop("eval_keys")
    assert batch["image_grid_thw"].shape[0] == 2
    assert batch["vision_unique_grid_thw"].shape[0] == 1
    assert batch["vision_reuse_indices"].tolist() == [0, 0]

    wrapper = SocialVLM(
        backbone,
        token_ids,
        graph_dim=4,
        graph_hidden_dim=32,
        heatmap_conv_dim=32,
        vision_cache_size=2,
    )
    try:
        with patch.object(
            core_model.visual,
            "forward",
            wraps=core_model.visual.forward,
        ) as visual_forward, patch.object(
            core_model,
            "get_rope_index",
            wraps=core_model.get_rope_index,
        ) as get_rope_index:
            first = wrapper(batch, graph)
            second = wrapper(batch, graph)
        assert visual_forward.call_count == 1
        assert get_rope_index.call_count == 2
        assert first.h_social.shape == second.h_social.shape == (2, 32)
        assert wrapper.vision_cache_info() == (1, 1, 2, 1)
        first.h_social.square().mean().backward()
        assert wrapper.social_query.grad is not None
        assert wrapper.social_query.grad.abs().sum() > 0
        lora_grads = [
            parameter.grad
            for name, parameter in backbone.named_parameters()
            if "lora_" in name
        ]
        assert any(
            grad is not None and grad.abs().sum() > 0 for grad in lora_grads
        )
    finally:
        wrapper.close()
