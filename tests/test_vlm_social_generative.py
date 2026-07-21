from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from vlm.social.data import SocialSample
from vlm.social.objective import GenerativeObjective, generative_answer_token_ids
from vlm.social.input import SocialVLMInput
from vlm.social.graph_tokens import (
    GRAPH_TOKEN_SLOTS,
    GraphTokenAdapter,
    GraphTokenPayload,
    configure_graph_tokenizer,
    extract_graph_token_payload,
)
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


def test_graph_token_adapter_and_inline_replacement_preserve_slot_contract():
    hidden_size, edge_dim = 8, 4
    token_ids = {slot.name: 10 + index for index, slot in enumerate(GRAPH_TOKEN_SLOTS)}
    adapter = GraphTokenAdapter(edge_dim=edge_dim, hidden_size=hidden_size, dropout=0.0)
    vlm = TextGenerativeVLM(
        _StubBackbone(), graph_token_adapter=adapter, graph_token_ids=token_ids
    )
    # A LAH sample owns only heatmap_a and edge_ab. The absent B slots must not
    # change any ordinary text embedding or be treated as prompt markers.
    input_ids = torch.tensor([[token_ids["heatmap_a"], 1, token_ids["edge_ab"], 2]])
    embeds = torch.zeros(1, 4, hidden_size)
    kwargs = {
        "graph_token_present": torch.tensor([[True, False, True, False]]),
        "graph_token_heatmaps": torch.ones(1, 2, 8, 8),
        "graph_token_edges": torch.ones(1, 2, edge_dim),
    }
    injected = vlm._inject_graph_token_embeddings(
        input_ids=input_ids, inputs_embeds=embeds, kwargs=kwargs
    )
    assert kwargs == {}
    assert torch.count_nonzero(injected[:, 0]) > 0
    assert torch.count_nonzero(injected[:, 2]) > 0
    assert torch.count_nonzero(injected[:, 1]) == 0
    assert torch.count_nonzero(injected[:, 3]) == 0
    injected.sum().backward()
    assert adapter.edge_encoder[1].weight.grad is not None
    assert adapter.heatmap_encoder[0].weight.grad is not None

    with pytest.raises(ValueError, match="mismatch"):
        vlm._inject_graph_token_embeddings(
            input_ids=input_ids,
            inputs_embeds=embeds,
            kwargs={
                "graph_token_present": torch.tensor([[False, False, True, False]]),
                "graph_token_heatmaps": torch.ones(1, 2, 8, 8),
                "graph_token_edges": torch.ones(1, 2, edge_dim),
            },
        )


def test_graph_token_payload_detaches_cache_and_trains_only_adapter():
    edge_dim, hidden_size = 4, 8
    cache = {
        "gaze_heatmap": torch.randn(2, 8, 8, requires_grad=True),
        "edge_pp": torch.randn(2, 2, edge_dim, requires_grad=True),
        # Label-shaped tensors are intentionally present but must remain inaccessible.
        "lah_gt": torch.ones(2, 2),
        "inout_gt": torch.tensor([1, 0]),
    }
    payload = extract_graph_token_payload(
        task="laeo", person_a=0, person_b=1, cache=cache,
        features=("gaze_heatmap", "edge_pp"),
    )
    assert set(payload.values) == {"heatmap_a", "heatmap_b", "edge_ab", "edge_ba"}
    assert not any(value.requires_grad for value in payload.values.values())

    adapter = GraphTokenAdapter(edge_dim=edge_dim, hidden_size=hidden_size, dropout=0.0)
    heatmaps = torch.stack(
        (payload.values["heatmap_a"], payload.values["heatmap_b"])
    ).unsqueeze(0)
    edges = torch.stack(
        (payload.values["edge_ab"], payload.values["edge_ba"])
    ).unsqueeze(0)
    adapter(heatmaps, edges).sum().backward()

    assert cache["gaze_heatmap"].grad is None
    assert cache["edge_pp"].grad is None
    assert adapter.heatmap_encoder[0].weight.grad is not None
    assert adapter.edge_encoder[1].weight.grad is not None


def test_text_generative_vlm_rejects_graph_tokens_without_vision_reuse():
    adapter = GraphTokenAdapter(edge_dim=4, hidden_size=8, dropout=0.0)
    ids = {slot.name: 10 + index for index, slot in enumerate(GRAPH_TOKEN_SLOTS)}
    vlm = TextGenerativeVLM(_StubBackbone(), graph_token_adapter=adapter, graph_token_ids=ids)
    with pytest.raises(ValueError, match="reuse_frozen_vision"):
        vlm({
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "graph_token_present": torch.zeros(1, 4, dtype=torch.bool),
        })


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


def _graph_token_text_item():
    return SocialVLMInput(
        annotation=SocialSample(
            sid="token",
            task="lah",
            person_i=0,
            person_j=1,
            label=1,
            raw_i=1,
            raw_j=0,
        ),
        image=Image.new("RGB", (56, 56), "black"),
        prompt=(
            "Person A's predicted gaze-distribution feature <|graph_heatmap_a|> and the "
            "directed graph edge <|graph_edge_ab|> estimate P(Person A looks at Person B) = 0.5."
        ),
        vision_cache_key="/split/token/frame.png",
        graph_token_payload=GraphTokenPayload({
            "heatmap_a": torch.ones(8, 8),
            "edge_ab": torch.full((4,), 2.0),
        }),
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


def test_text_token_collate_packs_prompt_aligned_payloads():
    processor = _local_qwen_processor()
    token_ids = configure_graph_tokenizer(processor.tokenizer)
    item = _graph_token_text_item()

    batch = make_text_generative_collate(
        processor,
        reuse_vision=True,
        graph_token_ids=token_ids,
        graph_token_edge_dim=4,
    )([item])

    assert batch["graph_token_present"].tolist() == [[True, False, True, False]]
    assert batch["graph_token_heatmaps"].shape == (1, 2, 8, 8)
    assert batch["graph_token_edges"].shape == (1, 2, 4)
    torch.testing.assert_close(batch["graph_token_edges"][0, 0], torch.full((4,), 2.0))
    assert int(batch["input_ids"].eq(token_ids["heatmap_a"]).sum()) == 1
    assert int(batch["input_ids"].eq(token_ids["edge_ab"]).sum()) == 1


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

        # The same real Qwen reuse path must accept inline dense graph tokens. These
        # marker ids are resized before use and each is replaced before the language
        # model sees the embedding sequence.
        token_ids = configure_graph_tokenizer(processor.tokenizer)
        backbone.resize_token_embeddings(len(processor.tokenizer))
        wrapper.graph_token_adapter = GraphTokenAdapter(
            edge_dim=4, hidden_size=32, dropout=0.0
        )
        wrapper.graph_token_ids = token_ids
        token_batch = make_text_generative_collate(
            processor,
            reuse_vision=True,
            graph_token_ids=token_ids,
            graph_token_edge_dim=4,
        )([_graph_token_text_item()])
        with torch.no_grad():
            token_output = wrapper(token_batch)
        assert token_output.loss is not None and torch.isfinite(token_output.loss)
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


def test_weighted_supervised_ce_without_weights_matches_plain_mean():
    from vlm.social.model import weighted_supervised_ce

    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]])
    labels = torch.tensor([0, 2])
    got = weighted_supervised_ce(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(got, expected)


def test_weighted_supervised_ce_upweights_positive_rows_by_weight_normalised_mean():
    from vlm.social.model import weighted_supervised_ce

    # Two examples, one supervised answer token each (rows 0 and 1).
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]])
    labels = torch.tensor([0, 2])
    rows = torch.tensor([0, 1])
    weights = torch.tensor([3.0, 1.0])  # example 0 up-weighted 3x
    got = weighted_supervised_ce(logits, labels, rows, weights)
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = (weights * ce).sum() / weights.sum()  # sum(w*ce)/sum(w)
    torch.testing.assert_close(got, expected)
    # A pure up-weight of one example must sit between the two per-example losses.
    assert min(ce).item() <= got.item() <= max(ce).item()


def test_objective_attaches_per_example_pos_weight_only_for_positive_pairs():
    # task_ids: [lah=0, laeo=1, sa=2, lah=0]; labels: [pos, neg, pos, neg].
    captured = {}

    class _CaptureVLM(TextGenerativeVLM):
        def forward(self, model_inputs):
            captured["pair_pos_weight"] = model_inputs.get("pair_pos_weight")
            return SimpleNamespace(loss=torch.tensor(0.0))

    pos_weight = torch.tensor([2.0, 5.0, 3.0])  # per task id
    obj = GenerativeObjective(
        _CaptureVLM(_StubBackbone()), pos_weight_by_task_id=pos_weight
    )
    task_ids = torch.tensor([0, 1, 2, 0])
    labels = torch.tensor([1, 0, 1, 0])
    obj({}, task_ids, labels)
    # positives inherit their task weight; negatives stay 1.0.
    torch.testing.assert_close(
        captured["pair_pos_weight"], torch.tensor([2.0, 1.0, 3.0, 1.0])
    )


def test_objective_without_pos_weight_leaves_batch_untouched():
    captured = {}

    class _CaptureVLM(TextGenerativeVLM):
        def forward(self, model_inputs):
            captured["keys"] = set(model_inputs)
            return SimpleNamespace(loss=torch.tensor(0.0))

    obj = GenerativeObjective(_CaptureVLM(_StubBackbone()))
    obj({"input_ids": torch.zeros(2, 3, dtype=torch.long)},
        torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert "pair_pos_weight" not in captured["keys"]


def test_objective_pos_weight_requires_task_ids_and_labels():
    obj = GenerativeObjective(
        TextGenerativeVLM(_StubBackbone()),
        pos_weight_by_task_id=torch.tensor([2.0, 2.0, 2.0]),
    )
    with pytest.raises(ValueError, match="task_ids/pair_labels"):
        obj({"input_ids": torch.zeros(1, 3, dtype=torch.long)})


def test_reuse_forward_applies_positive_class_weight_to_the_supervised_ce():
    class _ReuseBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 8, bias=False)

        def get_output_embeddings(self):
            return self.head

    class _ReuseVLM(TextGenerativeVLM):
        def _forward_with_reused_vision(self, kwargs, device):
            batch, length = kwargs["input_ids"].shape
            hidden = torch.arange(batch * length * 4, dtype=torch.float).reshape(
                batch, length, 4
            )
            self.last_hidden = hidden
            return SimpleNamespace(
                last_hidden_state=hidden, past_key_values=None,
                hidden_states=None, attentions=None,
            )

    vlm = _ReuseVLM(_ReuseBackbone(), vision_cache_size=1)
    labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])  # one supervised token per row
    base = {"input_ids": torch.zeros(2, 4, dtype=torch.long),
            "labels": labels, "vision_reuse_indices": torch.tensor([0, 0])}
    try:
        unweighted = vlm(dict(base)).loss
        weighted = vlm({**base, "pair_pos_weight": torch.tensor([3.0, 1.0])}).loss
        mask = labels[:, 1:] != -100
        logits = vlm.backbone.head(vlm.last_hidden[:, :-1][mask])
        ce = torch.nn.functional.cross_entropy(
            logits.float(), labels[:, 1:][mask], reduction="none"
        )
        rows = mask.nonzero(as_tuple=True)[0]
        w = torch.tensor([3.0, 1.0])[rows]
        torch.testing.assert_close(weighted, (w * ce).sum() / w.sum())
        assert not torch.allclose(weighted, unweighted)
    finally:
        vlm.close()


def test_non_reuse_path_rejects_nontrivial_pos_weight():
    vlm = TextGenerativeVLM(_StubBackbone())
    with pytest.raises(ValueError, match="reuse_frozen_vision"):
        vlm({"input_ids": torch.zeros(2, 4, dtype=torch.long),
             "labels": torch.randint(0, 32, (2, 4)),
             "pair_pos_weight": torch.tensor([2.0, 1.0])})
