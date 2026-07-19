"""Pair-VLM text-generative collation and frozen-vision-reuse forward path."""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from vlm.social.data import SOCIAL_TASK_ID
from vlm.social.input import SocialVLMInput
from vlm.runtime.vision_cache import VisionDiskCache
from vlm.social.prompt import generative_answer_yesno


# Batch keys that carry labels/metadata rather than model inputs; stripped before the
# backbone forward.
_NON_MODEL_KEYS = (
    "graph_features", "graph_present", "task_ids", "pair_labels", "eval_keys", "num_pairs",
)


def _encode_reused_frame_batch(
    processor: Any, texts: Sequence[str], items: Sequence[SocialVLMInput]
) -> dict[str, Any]:
    """Process each unique unmarked frame once while retaining one image per text."""
    unique_images = []
    unique_sids = []
    sid_to_index: dict[str, int] = {}
    reuse_indices = []
    for item in items:
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


def _user_message(item: SocialVLMInput):
    return {"role": "user", "content": [
        {"type": "image", "image": item.image},
        {"type": "text", "text": item.prompt},
    ]}


def _text_generative_collate(processor: Any, answers_for):
    """Shared body for text-mode SFT / eval collates. ``answers_for(items)`` yields the
    ordered (answer_text, item) pairs to teacher-force; no graph feature tensors are added."""
    processor.tokenizer.padding_side = "right"

    def collate(items: Sequence[SocialVLMInput]) -> dict[str, Any]:
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

    def collate(items: Sequence[SocialVLMInput]) -> dict[str, Any]:
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


def make_text_generative_direct_eval_collate(
    processor: Any, *, reuse_vision: bool = False
):
    """Build one generation prompt per pair for direct one-token yes/no scoring.

    Unlike candidate scoring this never duplicates a pair into yes/no continuations.
    The objective reads the final prompt hidden state and scores only the two answer
    tokens.  Right padding keeps ``attention_mask.sum() - 1`` equal to the next-token
    prediction position for every row.
    """
    processor.tokenizer.padding_side = "right"

    def collate(items: Sequence[SocialVLMInput]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty pair batch")
        prompt_texts = []
        for item in items:
            user = _user_message(item)
            prompt_texts.append(
                processor.apply_chat_template(
                    [user], tokenize=False, add_generation_prompt=True
                )
            )
        if reuse_vision:
            out = _encode_reused_frame_batch(processor, prompt_texts, items)
        else:
            out = dict(processor(
                text=prompt_texts,
                images=[item.image for item in items],
                return_tensors="pt",
                padding=True,
            ))
        out["pair_labels"] = torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32
        )
        out["eval_keys"] = [item.annotation.eval_key for item in items]
        out["num_pairs"] = len(items)
        return out

    return collate



def make_text_generative_eval_collate(
    processor: Any, *, reuse_vision: bool = False
):
    """Build [yes_0..yes_B-1, no_0..no_B-1], optionally reusing each frame once."""
    processor.tokenizer.padding_side = "right"

    def collate(items: Sequence[SocialVLMInput]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty pair batch")
        ordered = (
            [("yes", item) for item in items]
            + [("no", item) for item in items]
        )
        expanded_items = [item for _, item in ordered]
        prompt_texts, full_texts = [], []
        for answer, item in ordered:
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
            prompt_enc = _encode_reused_frame_batch(
                processor, prompt_texts, expanded_items
            )
            encoded = _encode_reused_frame_batch(
                processor, full_texts, expanded_items
            )
        else:
            images = [item.image for item in expanded_items]
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
        out["pair_labels"] = torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32
        )
        out["eval_keys"] = [item.annotation.eval_key for item in items]
        out["num_pairs"] = len(items)
        return out

    return collate


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

    def _init_vision_reuse(
        self,
        vision_cache_size: int,
        vision_disk_cache: str | None = None,
        vision_disk_metadata: Mapping[str, str] | None = None,
    ) -> None:
        if vision_cache_size < 0:
            raise ValueError(
                f"vision_cache_size must be non-negative, got {vision_cache_size}"
            )
        self.vision_cache_size = int(vision_cache_size)
        self._vision_cache: OrderedDict[str, _CachedVisionFrame] = OrderedDict()
        self._vision_disk_cache = (
            VisionDiskCache(vision_disk_cache, vision_disk_metadata)
            if vision_disk_cache else None
        )
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
                disk_frame = (
                    None if self._vision_disk_cache is None
                    else self._vision_disk_cache.get(sid)
                )
                if disk_frame is not None:
                    if not torch.equal(disk_frame.grid_thw.cpu(), unique_grid_thw[index].cpu()):
                        raise ValueError(f"disk vision grid changed for frame {sid!r}")
                    value = _CachedVisionFrame(
                        grid_thw=disk_frame.grid_thw,
                        pooler_output=disk_frame.pooler_output,
                        deepstack_features=disk_frame.deepstack_features,
                    )
                    resolved[index] = value
                    self._remember_vision_frame(sid, value)
                else:
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

    def __init__(
        self,
        backbone: nn.Module,
        vision_cache_size: int = 0,
        vision_disk_cache: str | None = None,
        vision_disk_metadata: Mapping[str, str] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self._init_vision_reuse(
            vision_cache_size, vision_disk_cache, vision_disk_metadata
        )

    def close(self) -> None:
        self.clear_vision_cache()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_cache_size:
            self._vision_model().visual.eval()
        return self

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def direct_answer_logits(
        self, model_inputs: Mapping[str, torch.Tensor], *, yes_token_id: int, no_token_id: int
    ) -> torch.Tensor:
        """Return [B,2] next-token logits for yes/no without any [B,L,V] projection.

        This evaluation-only path runs the language model once on the generation
        prompt, gathers each row's final valid hidden state, then applies only the
        two requested LM-head rows.  It requires frozen-vision reuse; the active
        text-evidence configuration always enables that path.
        """
        if "vision_reuse_indices" not in model_inputs:
            raise ValueError("direct answer scoring requires vision_reuse_indices")
        device = next(self.backbone.parameters()).device
        kwargs = {
            key: value
            for key, value in model_inputs.items()
            if key not in _NON_MODEL_KEYS
        }
        # Direct evaluation intentionally supplies no labels, but discard them defensively.
        kwargs.pop("labels", None)
        attention_mask = kwargs.get("attention_mask")
        if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
            raise ValueError("direct answer scoring requires [B,L] attention_mask")
        kwargs.update(
            {"output_hidden_states": False, "return_dict": True, "use_cache": False}
        )
        hidden_output = self._forward_with_reused_vision(kwargs, device)
        hidden = (
            hidden_output.last_hidden_state
            if hasattr(hidden_output, "last_hidden_state")
            else hidden_output[0]
        )
        if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
            raise ValueError(
                "hidden sequence must align with prompt attention_mask: "
                f"hidden={tuple(hidden.shape)}, mask={tuple(attention_mask.shape)}"
            )
        last_positions = attention_mask.to(hidden.device, dtype=torch.long).sum(dim=1) - 1
        if bool(torch.any(last_positions < 0)):
            raise ValueError("direct answer scoring received an empty prompt")
        final_hidden = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), last_positions
        ]
        lm_head = self.get_output_embeddings()
        if not hasattr(lm_head, "weight"):
            raise TypeError("direct answer scoring requires an LM head with a weight matrix")
        ids = torch.tensor([yes_token_id, no_token_id], device=hidden.device)
        weight = lm_head.weight.index_select(0, ids)
        bias = getattr(lm_head, "bias", None)
        if bias is not None:
            bias = bias.index_select(0, ids)
        return torch.nn.functional.linear(final_hidden, weight, bias)  # [B,2]

    def _reused_generative_output(
        self, hidden_output: Any, labels: torch.Tensor | None
    ) -> TextGenerativeVLMOutput:
        """Apply causal NLL after the low-level vision-reuse language-model forward.

        ``language_model`` returns hidden states, rather than the top-level VLM's
        logits/loss. Do *not* project every sequence position to the full Qwen
        vocabulary here: text SFT supervises only the answer continuation. Project
        precisely those causal positions and apply the mathematically identical CE.
        This is the reuse-path counterpart of Qwen's ``logits_to_keep`` optimisation
        and avoids materialising a [B, L, V] logits tensor during training.
        """
        hidden_states = (
            hidden_output.last_hidden_state
            if hasattr(hidden_output, "last_hidden_state")
            else hidden_output[0]
        )
        loss = None
        if labels is None:
            # Kept for diagnostic/non-reuse-equivalence callers. Production text
            # evaluation uses direct_answer_logits(), which projects only yes/no.
            logits = self.get_output_embeddings()(hidden_states)
        else:
            shift_labels = labels[:, 1:].to(hidden_states.device)
            supervised = shift_labels.ne(-100)
            if not bool(supervised.any()):
                raise ValueError("generative labels contain no supervised answer tokens")
            # Position t-1 predicts label t. Boolean indexing preserves a gradient
            # path only through supervised answer positions while retaining exactly
            # the same CE denominator over the complete frozen vocabulary.
            answer_hidden = hidden_states[:, :-1, :][supervised]
            answer_labels = shift_labels[supervised]
            logits = self.get_output_embeddings()(answer_hidden)
            loss = torch.nn.functional.cross_entropy(logits.float(), answer_labels)
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


