"""Pair-VLM collation, out-of-place soft-token injection and hidden readout."""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from vlm.pair_dataset import SOCIAL_TASK_ID
from vlm.pair_features import PairGraphBatch, stack_pair_graph_evidence
from vlm.pair_input import PairVLMInput
from vlm.pair_projection import PairEvidenceProjector
from vlm.pair_prompt import (
    GRAPH_TOKENS,
    PAIR_EVIDENCE_TOKENS,
    PAIR_SPECIAL_TOKENS,
    SOCIAL_RELATION_TOKEN,
    add_pair_special_tokens,
    generative_answer_json,
    generative_answer_yesno,
    pair_special_token_ids,
    social_readout_prompt,
    task_conditioned_pair_instruction,
)


def prepare_pair_tokens(tokenizer: Any, model: nn.Module) -> dict[str, int]:
    """Register seven pair tokens and resize the model embedding table when needed."""
    add_pair_special_tokens(tokenizer)
    embeddings = model.get_input_embeddings()
    if embeddings.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return pair_special_token_ids(tokenizer)


def _pair_chat_text(processor: Any, item: PairVLMInput) -> str:
    task = item.annotation.task
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": item.image},
                {
                    "type": "text",
                    "text": task_conditioned_pair_instruction(
                        task, draw_bboxes=item.draw_bboxes
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": social_readout_prompt(task)}],
        },
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if not text.endswith(SOCIAL_RELATION_TOKEN):
        raise ValueError("Qwen chat template did not preserve social readout as final text")
    return text


def _encode_reused_frame_batch(
    processor: Any, texts: Sequence[str], items: Sequence[PairVLMInput]
) -> dict[str, Any]:
    """Process each unique unmarked frame once while retaining one image per text."""
    unique_images = []
    unique_sids = []
    sid_to_index: dict[str, int] = {}
    reuse_indices = []
    for item in items:
        if item.draw_bboxes:
            raise ValueError("vision reuse requires unmodified images (draw_bboxes=false)")
        sid = item.vision_cache_key or item.annotation.sid
        unique_index = sid_to_index.get(sid)
        if unique_index is None:
            unique_index = len(unique_sids)
            sid_to_index[sid] = unique_index
            unique_sids.append(sid)
            unique_images.append(item.image)
        reuse_indices.append(unique_index)

    image_inputs = dict(
        processor.image_processor(images=unique_images, return_tensors="pt")
    )
    unique_grid = image_inputs.get("image_grid_thw")
    if not torch.is_tensor(unique_grid) or unique_grid.shape != (len(unique_sids), 3):
        shape = tuple(unique_grid.shape) if torch.is_tensor(unique_grid) else None
        raise ValueError(
            f"unique image_grid_thw must have shape ({len(unique_sids)},3), got {shape}"
        )
    reuse = torch.tensor(reuse_indices, dtype=torch.long)
    expanded_grid = unique_grid.index_select(0, reuse)
    merge_length = int(processor.image_processor.merge_size) ** 2
    image_token = str(processor.image_token)
    expanded_texts = []
    for text, grid in zip(texts, expanded_grid):
        if text.count(image_token) != 1:
            raise ValueError("each pair chat must contain exactly one Qwen image token")
        num_tokens = int(grid.prod().item()) // merge_length
        placeholder = "<|placeholder|>" * num_tokens
        expanded_texts.append(
            text.replace(image_token, placeholder, 1).replace(
                "<|placeholder|>", image_token
            )
        )

    encoded = dict(
        processor.tokenizer(
            expanded_texts,
            return_tensors="pt",
            padding=True,
        )
    )
    encoded["mm_token_type_ids"] = torch.as_tensor(
        processor.create_mm_token_type_ids(encoded["input_ids"]),
        dtype=torch.long,
    )
    image_inputs["image_grid_thw"] = expanded_grid
    encoded.update(image_inputs)
    encoded["vision_unique_grid_thw"] = unique_grid
    encoded["vision_reuse_indices"] = reuse
    encoded["vision_frame_ids"] = tuple(unique_sids)
    return encoded


def make_pair_collate(processor: Any, *, reuse_vision: bool = False):
    """Build Qwen image chats with an assistant-side ``<social_relation>`` prefill."""
    processor.tokenizer.padding_side = "right"

    def collate(items: Sequence[PairVLMInput]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty pair batch")
        texts = [_pair_chat_text(processor, item) for item in items]
        if reuse_vision:
            encoded = _encode_reused_frame_batch(processor, texts, items)
        else:
            encoded = processor(
                text=texts,
                images=[item.image for item in items],
                return_tensors="pt",
                padding=True,
            )
        out = dict(encoded)
        out["pair_graph"] = stack_pair_graph_evidence([item.evidence for item in items])
        out["task_ids"] = torch.tensor(
            [SOCIAL_TASK_ID[item.annotation.task] for item in items], dtype=torch.long
        )
        out["pair_labels"] = torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32
        )
        out["eval_keys"] = [item.annotation.eval_key for item in items]
        return out

    return collate


# ── EyeVLM-style GENERATIVE path ──────────────────────────────────────────────
def evidence_placeholder_masks(input_ids: torch.Tensor, token_ids: Mapping[str, int]):
    """[B,L,6] masks + [B,6] positions for the six graph evidence tokens (generative)."""
    ids = torch.tensor(
        [token_ids[t] for t in PAIR_EVIDENCE_TOKENS],
        dtype=input_ids.dtype, device=input_ids.device,
    )
    masks = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1))
    counts = masks.sum(dim=1)
    if not bool(torch.all(counts == 1)):
        raise ValueError(f"each evidence token must occur once per sample; counts={counts.tolist()}")
    positions = masks.to(torch.int64).argmax(dim=1)
    # No fixed prompt order required: bmm-injection routes each token by identity, so the
    # grouped per-person layout may interleave the six tokens freely.
    return masks, positions


def graph_token_masks(input_ids: torch.Tensor, token_ids: Mapping[str, int]):
    """[B, L, 4] masks for the four <gtok*> slots. Each slot appears 0 or 1 times (a task
    uses 2-4 of them); bmm-injection places only the slots actually present in the prompt."""
    ids = torch.tensor(
        [token_ids[t] for t in GRAPH_TOKENS], dtype=input_ids.dtype, device=input_ids.device
    )
    masks = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1))          # [B, L, 4]
    if not bool(torch.all(masks.sum(dim=1) <= 1)):
        raise ValueError("a <gtok*> token occurs more than once in a sample")
    return masks


class GraphTokenProjector(nn.Module):
    """Project up-to-4 raw graph vectors (v_src/v_tgt/edge, De=256) into Qwen hidden space.

    A per-slot embedding disambiguates the four positions; a magnitude gain matches the text
    embedding RMS so the soft-tokens sit at a sensible scale. Simple by design (no CNN, no
    null_in), per the current scope."""

    def __init__(self, graph_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(graph_dim, output_dim)
        self.slot = nn.Parameter(torch.zeros(len(GRAPH_TOKENS), output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def set_output_gain(self, value: float) -> None:
        with torch.no_grad():
            self.gain.fill_(value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, 4, De] -> [B, 4, H]
        x = self.proj(features.to(self.proj.weight.dtype)) + self.slot.unsqueeze(0)
        return self.gain * self.norm(x)


def _user_message(item: PairVLMInput):
    return {"role": "user", "content": [
        {"type": "image", "image": item.image},
        {"type": "text", "text": item.prompt},
    ]}


def make_generative_collate(processor: Any):
    """JSON-SFT collate (EyeVLM): prompt + target ``[{"label": 1/0}]`` with next-token CE
    supervised ONLY on the answer JSON (the prompt is masked to -100). The prompt-only
    encoding gives the per-sample prefix length used for masking."""
    processor.tokenizer.padding_side = "right"

    def collate(items: Sequence[PairVLMInput]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty pair batch")
        images = [item.image for item in items]
        prompt_texts, full_texts = [], []
        for item in items:
            user = _user_message(item)
            prompt_texts.append(processor.apply_chat_template(
                [user], tokenize=False, add_generation_prompt=True))
            answer = generative_answer_json(int(item.annotation.label))
            full_texts.append(processor.apply_chat_template(
                [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                tokenize=False))
        prompt_enc = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)          # per-sample prefix length
        encoded = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        out = dict(encoded)
        labels = out["input_ids"].clone()
        labels[out["attention_mask"] == 0] = -100                     # padding
        for i, plen in enumerate(prompt_lens.tolist()):
            labels[i, :plen] = -100                                    # prompt tokens
        out["labels"] = labels
        out["graph_features"] = torch.stack([item.evidence.features for item in items])  # [B,4,De]
        out["graph_present"] = torch.stack([item.evidence.present for item in items])     # [B,4]
        out["task_ids"] = torch.tensor(
            [SOCIAL_TASK_ID[item.annotation.task] for item in items], dtype=torch.long)
        out["pair_labels"] = torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32)
        out["eval_keys"] = [item.annotation.eval_key for item in items]
        return out

    return collate


def make_generative_eval_collate(processor: Any):
    """Eval collate: for each pair emit TWO teacher-forced sequences (positive/negative JSON
    candidates) so the objective can score them by answer log-likelihood. Order = [pos_0..
    pos_{B-1}, neg_0..neg_{B-1}]; a [B] image set is duplicated for the two candidates."""
    processor.tokenizer.padding_side = "right"
    pos_json, neg_json = generative_answer_json(1), generative_answer_json(0)

    def collate(items: Sequence[PairVLMInput]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty pair batch")
        images, prompt_texts, full_texts = [], [], []
        for answer in (pos_json, neg_json):                            # positives then negatives
            for item in items:
                user = _user_message(item)
                images.append(item.image)
                prompt_texts.append(processor.apply_chat_template(
                    [user], tokenize=False, add_generation_prompt=True))
                full_texts.append(processor.apply_chat_template(
                    [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                    tokenize=False))
        prompt_enc = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        encoded = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        out = dict(encoded)
        labels = out["input_ids"].clone()
        labels[out["attention_mask"] == 0] = -100
        for i, plen in enumerate(prompt_lens.tolist()):
            labels[i, :plen] = -100
        out["labels"] = labels
        feats = torch.stack([item.evidence.features for item in items])   # [B,4,De]
        present = torch.stack([item.evidence.present for item in items])  # [B,4]
        out["graph_features"] = torch.cat([feats, feats], dim=0)          # [2B] pos+neg share
        out["graph_present"] = torch.cat([present, present], dim=0)
        out["pair_labels"] = torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32)
        out["eval_keys"] = [item.annotation.eval_key for item in items]
        out["num_pairs"] = len(items)
        return out

    return collate


_NON_MODEL_KEYS = (
    "graph_features", "graph_present", "task_ids", "pair_labels", "eval_keys", "num_pairs",
)


class PairGenerativeVLM(nn.Module):
    """Generative EyeVLM-style pair model: the task-specific graph soft-tokens (v_src/v_tgt/
    edge, up to 4 <gtok*> slots — our contribution) are injected into the prompt and the
    frozen+LoRA LM is SFT'd to generate the binary JSON ``[{"label": 1/0}]``. ``forward`` returns the raw
    backbone ModelOutput (``.loss`` from ``labels`` for train, ``.logits`` for eval scoring)."""

    def __init__(
        self,
        backbone: nn.Module,
        token_ids: Mapping[str, int],
        graph_dim: int = 256,
        graph_hidden_dim: int = 1024,
        heatmap_conv_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.token_ids = dict(token_ids)
        hidden_dim = int(backbone.config.text_config.hidden_size)
        embeddings = backbone.get_input_embeddings().weight
        self.projector = GraphTokenProjector(graph_dim, hidden_dim).to(
            device=embeddings.device, dtype=embeddings.dtype)
        with torch.no_grad():
            text_rms = embeddings.float().pow(2).mean(-1).sqrt().mean().item()
        self.projector.set_output_gain(text_rms)
        self._injector = _PairInjectionHook(_find_language_model(backbone))

    def close(self) -> None:
        self._injector.close()

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def forward(self, model_inputs: Mapping[str, torch.Tensor]):
        """Inject the graph tokens and run the backbone. ``labels`` (added by the collate)
        makes the backbone return next-token CE ``.loss``; ``.logits`` feeds eval scoring."""
        device = next(self.projector.parameters()).device
        input_ids = model_inputs["input_ids"].to(device)
        masks = graph_token_masks(input_ids, self.token_ids)            # [B, L, 4]
        features = model_inputs["graph_features"].to(device)           # [B, 4, De]
        graph_tokens = self.projector(features)                        # [B, 4, H]
        kwargs = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in model_inputs.items() if key not in _NON_MODEL_KEYS
        }
        kwargs.update({
            "output_hidden_states": False, "return_dict": True, "use_cache": False,
        })
        self._injector.pending = (masks, graph_tokens)
        self._injector.calls = 0
        try:
            output = self.backbone(**kwargs)
        finally:
            self._injector.pending = None
        if self._injector.calls != 1:
            raise RuntimeError(f"pair injection hook ran {self._injector.calls} times, expected once")
        return output


def _text_generative_collate(processor: Any, answers_for):
    """Shared body for text-mode SFT / eval collates. ``answers_for(items)`` yields the
    ordered (answer_text, item) pairs to teacher-force; no graph feature tensors are added."""
    processor.tokenizer.padding_side = "right"

    def collate(items: Sequence[PairVLMInput]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty pair batch")
        images, prompt_texts, full_texts = [], [], []
        for answer, item in answers_for(items):
            user = _user_message(item)
            images.append(item.image)
            prompt_texts.append(processor.apply_chat_template(
                [user], tokenize=False, add_generation_prompt=True))
            full_texts.append(processor.apply_chat_template(
                [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                tokenize=False))
        prompt_enc = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        encoded = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        out = dict(encoded)
        labels = out["input_ids"].clone()
        labels[out["attention_mask"] == 0] = -100
        for i, plen in enumerate(prompt_lens.tolist()):
            labels[i, :plen] = -100
        out["labels"] = labels
        return out

    return collate


def make_text_generative_collate(processor: Any, *, reuse_vision: bool = False):
    """Text-mode SFT with optional one-vision-encoding-per-unique-frame reuse."""
    processor.tokenizer.padding_side = "right"

    def collate(items: Sequence[PairVLMInput]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty pair batch")
        answers = [
            (generative_answer_yesno(int(item.annotation.label)), item)
            for item in items
        ]
        prompt_texts, full_texts = [], []
        for answer, item in answers:
            user = _user_message(item)
            prompt_texts.append(
                processor.apply_chat_template(
                    [user], tokenize=False, add_generation_prompt=True
                )
            )
            full_texts.append(
                processor.apply_chat_template(
                    [
                        user,
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        },
                    ],
                    tokenize=False,
                )
            )
        if reuse_vision:
            prompt_enc = _encode_reused_frame_batch(processor, prompt_texts, items)
            encoded = _encode_reused_frame_batch(processor, full_texts, items)
        else:
            images = [item.image for item in items]
            prompt_enc = processor(
                text=prompt_texts, images=images, return_tensors="pt", padding=True
            )
            encoded = processor(
                text=full_texts, images=images, return_tensors="pt", padding=True
            )
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        out = dict(encoded)
        labels = out["input_ids"].clone()
        labels[out["attention_mask"] == 0] = -100
        for index, prompt_len in enumerate(prompt_lens.tolist()):
            labels[index, :prompt_len] = -100
        out["labels"] = labels
        out["task_ids"] = torch.tensor(
            [SOCIAL_TASK_ID[item.annotation.task] for item in items], dtype=torch.long
        )
        out["pair_labels"] = torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32
        )
        out["eval_keys"] = [item.annotation.eval_key for item in items]
        return out

    return collate


def make_text_generative_eval_collate(processor: Any):
    """Text-mode eval: [2B] rows = yes-candidate for every pair, then no-candidate."""
    def answers_for(items):
        return ([("yes", it) for it in items] + [("no", it) for it in items])
    base = _text_generative_collate(processor, answers_for)

    def collate(items: Sequence[PairVLMInput]) -> dict[str, Any]:
        out = base(items)
        out["pair_labels"] = torch.tensor(
            [it.annotation.label for it in items], dtype=torch.float32)
        out["eval_keys"] = [it.annotation.eval_key for it in items]
        out["num_pairs"] = len(items)
        return out

    return collate


def placeholder_masks(
    input_ids: torch.Tensor, token_ids: Mapping[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[B,L,7]`` masks and ``[B,7]`` positions with strict count/order checks."""
    ids = torch.tensor(
        [token_ids[token] for token in PAIR_SPECIAL_TOKENS],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    masks = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1))
    counts = masks.sum(dim=1)
    if not bool(torch.all(counts == 1)):
        raise ValueError(f"every pair special token must occur once per sample; counts={counts.tolist()}")
    positions = masks.to(torch.int64).argmax(dim=1)
    if not bool(torch.all(positions[:, 1:] > positions[:, :-1])):
        raise ValueError(f"pair special tokens are out of order; positions={positions.tolist()}")
    return masks, positions


def out_of_place_soft_token_replace(
    inputs_embeds: torch.Tensor,
    masks: torch.Tensor,
    soft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Differentiably replace selected embeddings without indexed in-place writes."""
    if masks.shape[:2] != inputs_embeds.shape[:2]:
        raise ValueError("placeholder mask and input embedding sequence shapes differ")
    if soft_tokens.shape != (inputs_embeds.shape[0], masks.shape[2], inputs_embeds.shape[2]):
        raise ValueError(
            f"soft_tokens must have shape {(inputs_embeds.shape[0], masks.shape[2], inputs_embeds.shape[2])}, "
            f"got {tuple(soft_tokens.shape)}"
        )
    replacement = torch.bmm(masks.to(soft_tokens.dtype), soft_tokens)
    selected = masks.any(dim=-1, keepdim=True)
    return torch.where(selected, replacement.to(inputs_embeds.dtype), inputs_embeds)


def _find_language_model(model: nn.Module) -> nn.Module:
    candidates = []
    for name, module in model.named_modules():
        if name.endswith("language_model") and hasattr(module, "embed_tokens"):
            candidates.append((name, module))
    unique = {id(module): (name, module) for name, module in candidates}
    if len(unique) != 1:
        names = [name for name, _ in candidates]
        raise ValueError(f"expected one Qwen language_model, found {names}")
    return next(iter(unique.values()))[1]


def _find_multimodal_model(model: nn.Module) -> nn.Module:
    candidates = []
    for name, module in model.named_modules():
        if (
            hasattr(module, "get_image_features")
            and hasattr(module, "get_placeholder_mask")
            and hasattr(module, "compute_3d_position_ids")
            and hasattr(module, "language_model")
            and hasattr(module, "visual")
        ):
            candidates.append((name, module))
    unique = {id(module): (name, module) for name, module in candidates}
    if len(unique) != 1:
        names = [name for name, _ in candidates]
        raise ValueError(f"expected one Qwen multimodal model, found {names}")
    return next(iter(unique.values()))[1]


class _PairInjectionHook:
    def __init__(self, language_model: nn.Module):
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self.calls = 0
        self.handle = language_model.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        del module
        if self.pending is None:
            return args, kwargs
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            raise ValueError("Qwen language_model hook requires inputs_embeds")
        masks, tokens = self.pending
        kwargs["inputs_embeds"] = out_of_place_soft_token_replace(
            inputs_embeds,
            masks.to(inputs_embeds.device),
            tokens.to(inputs_embeds.device, inputs_embeds.dtype),
        )
        self.calls += 1
        return args, kwargs

    def close(self) -> None:
        self.handle.remove()


class _SocialReadoutHook:
    """Capture only the social-query row from Qwen's post-final-norm output."""

    def __init__(self, language_model: nn.Module):
        norm = getattr(language_model, "norm", None)
        if not isinstance(norm, nn.Module):
            raise ValueError("Qwen language_model must expose a final norm module")
        self.pending_mask: torch.Tensor | None = None
        self.captured: torch.Tensor | None = None
        self.calls = 0
        self.handle = norm.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        del module, args
        if self.pending_mask is None:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise ValueError("Qwen final norm must return a [B,L,H] tensor")
        mask = self.pending_mask.to(hidden.device)
        if mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"social mask shape {tuple(mask.shape)} does not match final hidden "
                f"shape {tuple(hidden.shape[:2])}"
            )
        captured = hidden[mask]
        if captured.shape != (hidden.shape[0], hidden.shape[2]):
            raise ValueError(f"invalid social readout shape {tuple(captured.shape)}")
        self.captured = captured
        self.calls += 1

    def close(self) -> None:
        self.captured = None
        self.pending_mask = None
        self.handle.remove()


@dataclass
class PairSocialVLMOutput:
    h_social: torch.Tensor
    evidence_tokens: torch.Tensor
    placeholder_positions: torch.Tensor
    backbone_output: Any


VisionCacheInfo = namedtuple("VisionCacheInfo", "hits misses max_items curr_items")


@dataclass(frozen=True)
class _CachedVisionFrame:
    grid_thw: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: tuple[torch.Tensor, ...]


class _VisionReuseMixin:
    """Reuse frozen per-frame Qwen vision features across pair text sequences.

    The cache and masked-scatter forward path require unmodified images. A subclass
    must set ``self.backbone`` before calling ``_init_vision_reuse``.
    """

    def _init_vision_reuse(self, vision_cache_size: int) -> None:
        if vision_cache_size < 0:
            raise ValueError(
                f"vision_cache_size must be non-negative, got {vision_cache_size}"
            )
        self.vision_cache_size = int(vision_cache_size)
        self._vision_cache: OrderedDict[str, _CachedVisionFrame] = OrderedDict()
        self._vision_cache_hits = 0
        self._vision_cache_misses = 0
        self._multimodal_model: nn.Module | None = None

    def clear_vision_cache(self) -> None:
        self._vision_cache.clear()
        self._vision_cache_hits = 0
        self._vision_cache_misses = 0

    def vision_cache_info(self) -> VisionCacheInfo:
        return VisionCacheInfo(
            self._vision_cache_hits,
            self._vision_cache_misses,
            self.vision_cache_size,
            len(self._vision_cache),
        )

    def _vision_model(self) -> nn.Module:
        if self._multimodal_model is None:
            self._multimodal_model = _find_multimodal_model(self.backbone)
        return self._multimodal_model

    def _remember_vision_frame(self, sid: str, value: _CachedVisionFrame) -> None:
        if not self.vision_cache_size:
            return
        self._vision_cache[sid] = value
        self._vision_cache.move_to_end(sid)
        while len(self._vision_cache) > self.vision_cache_size:
            self._vision_cache.popitem(last=False)

    def _reused_vision_features(
        self,
        *,
        pixel_values: torch.Tensor,
        unique_grid_thw: torch.Tensor,
        expanded_grid_thw: torch.Tensor,
        reuse_indices: torch.Tensor,
        frame_ids: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        unique_count = len(frame_ids)
        if unique_grid_thw.shape != (unique_count, 3):
            raise ValueError("vision_unique_grid_thw and vision_frame_ids disagree")
        if reuse_indices.ndim != 1 or expanded_grid_thw.shape != (
            len(reuse_indices),
            3,
        ):
            raise ValueError("invalid expanded vision-reuse metadata")
        if reuse_indices.numel() and (
            int(reuse_indices.min()) < 0
            or int(reuse_indices.max()) >= unique_count
        ):
            raise ValueError("vision_reuse_indices is outside unique frame range")

        patch_sizes = unique_grid_thw.prod(-1).tolist()
        if sum(int(size) for size in patch_sizes) != pixel_values.shape[0]:
            raise ValueError("unique pixel_values do not match unique image grids")
        pixel_parts = torch.split(pixel_values, [int(size) for size in patch_sizes])
        resolved: dict[int, _CachedVisionFrame] = {}
        missing_indices = []
        for index, sid in enumerate(frame_ids):
            cached = self._vision_cache.get(sid)
            if cached is not None:
                if not torch.equal(cached.grid_thw.cpu(), unique_grid_thw[index].cpu()):
                    raise ValueError(f"cached vision grid changed for frame {sid!r}")
                self._vision_cache_hits += 1
                self._vision_cache.move_to_end(sid)
                resolved[index] = cached
            else:
                self._vision_cache_misses += 1
                missing_indices.append(index)

        vision_model = self._vision_model()
        if missing_indices:
            missing_pixels = torch.cat(
                [pixel_parts[index] for index in missing_indices], dim=0
            ).to(device)
            missing_grids = unique_grid_thw[missing_indices].to(device)
            with torch.no_grad():
                vision_output = vision_model.get_image_features(
                    missing_pixels,
                    missing_grids,
                    return_dict=True,
                )
            poolers = tuple(vision_output.pooler_output)
            merged_sizes = (
                missing_grids.prod(-1)
                // int(vision_model.visual.spatial_merge_size) ** 2
            ).tolist()
            deep_splits = [
                torch.split(layer, [int(size) for size in merged_sizes])
                for layer in vision_output.deepstack_features
            ]
            if len(poolers) != len(missing_indices):
                raise RuntimeError(
                    "Qwen vision output count does not match missing frames"
                )
            for local_index, unique_index in enumerate(missing_indices):
                value = _CachedVisionFrame(
                    grid_thw=missing_grids[local_index].detach().cpu(),
                    pooler_output=poolers[local_index].detach(),
                    deepstack_features=tuple(
                        layer[local_index].detach() for layer in deep_splits
                    ),
                )
                resolved[unique_index] = value
                self._remember_vision_frame(frame_ids[unique_index], value)

        sample_indices = reuse_indices.tolist()
        image_embeds = torch.cat(
            [resolved[index].pooler_output for index in sample_indices], dim=0
        )
        layer_count = len(next(iter(resolved.values())).deepstack_features)
        deepstack = [
            torch.cat(
                [resolved[index].deepstack_features[layer] for index in sample_indices],
                dim=0,
            )
            for layer in range(layer_count)
        ]
        expected_grids = unique_grid_thw.index_select(0, reuse_indices.cpu())
        if not torch.equal(expected_grids.cpu(), expanded_grid_thw.cpu()):
            raise ValueError("expanded image_grid_thw does not match reuse mapping")
        return image_embeds, deepstack

    def _forward_with_reused_vision(
        self, kwargs: dict[str, Any], device: torch.device
    ) -> Any:
        required = (
            "input_ids",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
            "vision_unique_grid_thw",
            "vision_reuse_indices",
            "vision_frame_ids",
        )
        missing = [key for key in required if key not in kwargs]
        if missing:
            raise ValueError(f"vision reuse inputs are missing {missing}")
        input_ids = kwargs.pop("input_ids").to(device)
        pixel_values = kwargs.pop("pixel_values")
        expanded_grid = kwargs.pop("image_grid_thw")
        unique_grid = kwargs.pop("vision_unique_grid_thw")
        reuse_indices = kwargs.pop("vision_reuse_indices")
        frame_ids = kwargs.pop("vision_frame_ids")
        mm_token_type_ids = kwargs.pop("mm_token_type_ids").to(device)
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        expanded_grid_device = expanded_grid.to(device)

        vision_model = self._vision_model()
        inputs_embeds = vision_model.get_input_embeddings()(input_ids)
        image_embeds, deepstack = self._reused_vision_features(
            pixel_values=pixel_values,
            unique_grid_thw=unique_grid,
            expanded_grid_thw=expanded_grid,
            reuse_indices=reuse_indices,
            frame_ids=frame_ids,
            device=device,
        )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = vision_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        position_ids = vision_model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=expanded_grid_device,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        kwargs.pop("logits_to_keep", None)
        return vision_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask[..., 0],
            deepstack_visual_embeds=deepstack,
            **kwargs,
        )


@dataclass
class TextGenerativeVLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None


class TextGenerativeVLM(_VisionReuseMixin, nn.Module):
    """Text-evidence generative VLM with optional cross-pair vision reuse."""

    def __init__(self, backbone: nn.Module, vision_cache_size: int = 0):
        super().__init__()
        self.backbone = backbone
        self._init_vision_reuse(vision_cache_size)

    def close(self) -> None:
        self.clear_vision_cache()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_cache_size:
            self._vision_model().visual.eval()
        return self

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def _reused_generative_output(
        self, hidden_output: Any, labels: torch.Tensor | None
    ) -> TextGenerativeVLMOutput:
        """Apply the frozen LM head and causal CE bypassed by the low-level reuse path."""
        hidden_states = (
            hidden_output.last_hidden_state
            if hasattr(hidden_output, "last_hidden_state")
            else hidden_output[0]
        )
        logits = self.get_output_embeddings()(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return TextGenerativeVLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(hidden_output, "past_key_values", None),
            hidden_states=getattr(hidden_output, "hidden_states", None),
            attentions=getattr(hidden_output, "attentions", None),
        )

    def forward(self, model_inputs: Mapping[str, torch.Tensor]):
        device = next(self.backbone.parameters()).device
        reuse_vision = "vision_reuse_indices" in model_inputs
        kwargs = {
            key: value
            for key, value in model_inputs.items()
            if key not in _NON_MODEL_KEYS
        }
        if not reuse_vision:
            kwargs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
            kwargs.update(
                {"output_hidden_states": False, "return_dict": True, "use_cache": False}
            )
            return self.backbone(**kwargs)

        labels = kwargs.pop("labels", None)
        if torch.is_tensor(labels):
            labels = labels.to(device)
        kwargs.update(
            {"output_hidden_states": False, "return_dict": True, "use_cache": False}
        )
        hidden_output = self._forward_with_reused_vision(kwargs, device)
        return self._reused_generative_output(hidden_output, labels)


class PairSocialVLM(_VisionReuseMixin, nn.Module):
    """Qwen forward with six graph slots and one learned social-query readout token."""

    def __init__(
        self,
        backbone: nn.Module,
        token_ids: Mapping[str, int],
        graph_dim: int = 256,
        graph_hidden_dim: int = 1024,
        heatmap_conv_dim: int = 128,
        vision_cache_size: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.token_ids = dict(token_ids)
        self._init_vision_reuse(vision_cache_size)
        hidden_dim = int(backbone.config.text_config.hidden_size)
        embeddings = backbone.get_input_embeddings().weight
        self.projector = PairEvidenceProjector(
            graph_dim,
            hidden_dim,
            graph_hidden_dim=graph_hidden_dim,
            heatmap_conv_dim=heatmap_conv_dim,
        ).to(device=embeddings.device, dtype=embeddings.dtype)
        with torch.no_grad():
            text_rms = embeddings.float().pow(2).mean(-1).sqrt().mean().item()
            social_id = self.token_ids[SOCIAL_RELATION_TOKEN]
            social_init = embeddings[social_id].detach().clone()
        self.projector.set_output_gain(text_rms)
        self.social_query = nn.Parameter(social_init)
        language_model = _find_language_model(backbone)
        self._injector = _PairInjectionHook(language_model)
        self._readout = _SocialReadoutHook(language_model)

    def close(self) -> None:
        self.clear_vision_cache()
        self._injector.close()
        self._readout.close()

    def train(self, mode: bool = True):
        super().train(mode)
        # Cached outputs must not depend on dropout/stochastic training state. The
        # vision tower is frozen by contract, so eval mode is correct in both phases.
        if self.vision_cache_size:
            self._vision_model().visual.eval()
        return self

    def get_output_embeddings(self) -> nn.Module:
        """Return Qwen's pretrained LM head for optional one-token supervision."""
        output_embeddings = self.backbone.get_output_embeddings()
        if not isinstance(output_embeddings, nn.Module):
            raise ValueError("Qwen backbone does not expose output embeddings")
        return output_embeddings

    def forward(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        graph: PairGraphBatch,
    ) -> PairSocialVLMOutput:
        if "input_ids" not in model_inputs:
            raise ValueError("model_inputs must include input_ids")
        device = self.social_query.device
        input_ids = model_inputs["input_ids"].to(device)
        masks, positions = placeholder_masks(input_ids, self.token_ids)
        graph = graph.to(device)
        evidence_tokens = self.projector(graph)
        social = self.social_query.view(1, 1, -1).expand(input_ids.shape[0], 1, -1)
        soft_tokens = torch.cat((evidence_tokens, social), dim=1)

        reuse_vision = "vision_reuse_indices" in model_inputs
        kwargs = dict(model_inputs)
        if not reuse_vision:
            kwargs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
        kwargs.update({
            # A final-norm hook captures only h_social. Requesting hidden states would
            # retain every decoder layer and undermine gradient-checkpointing savings.
            "output_hidden_states": False,
            "return_dict": True,
            "use_cache": False,
            "logits_to_keep": 1,
        })
        self._injector.pending = (masks, soft_tokens)
        self._injector.calls = 0
        self._readout.pending_mask = masks[..., -1]
        self._readout.captured = None
        self._readout.calls = 0
        try:
            output = (
                self._forward_with_reused_vision(kwargs, device)
                if reuse_vision
                else self.backbone(**kwargs)
            )
        except Exception:
            self._readout.captured = None
            raise
        finally:
            self._injector.pending = None
            self._readout.pending_mask = None
        if self._injector.calls != 1:
            raise RuntimeError(f"pair injection hook ran {self._injector.calls} times, expected once")
        if self._readout.calls != 1 or self._readout.captured is None:
            raise RuntimeError(
                f"social final-norm hook ran {self._readout.calls} times, expected once"
            )
        h_social = self._readout.captured
        self._readout.captured = None
        return PairSocialVLMOutput(
            h_social=h_social,
            evidence_tokens=evidence_tokens,
            placeholder_positions=positions,
            backbone_output=output,
        )
