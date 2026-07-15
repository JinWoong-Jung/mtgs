"""Pair-wise image/prompt/evidence assembly with a bounded raw-frame cache."""

from __future__ import annotations

import hashlib
import random

from collections import Counter, OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.social.data import (
    SocialAnnotationDataset,
    SocialSample,
    SOCIAL_TASK_ID,
    frame_path,
)
from vlm.social.evidence import (
    GraphEvidence,
    assemble_generative_graph,
    assemble_graph_evidence,
    assemble_text_graph_evidence,
    stack_graph_evidence,
)
from vlm.social.prompt import (
    compose_generative_prompt,
    compose_text_prompt,
    task_prompt,
)


FrameCacheInfo = namedtuple("FrameCacheInfo", "hits misses max_items curr_items")


class RawFrameCache:
    """Process-local LRU of decoded, unmodified RGB frames.

    DataLoader workers each own their cache. A small bound avoids multiplying a large
    image cache by ``num_workers``. Callers must treat the returned raw image as read-only;
    pair identity is supplied by normalized bbox coordinates in the text prompt.
    """

    def __init__(self, frame_root: str | Path, max_items: int = 32):
        if max_items < 0:
            raise ValueError(f"max_items must be non-negative, got {max_items}")
        self.frame_root = Path(frame_root)
        self.max_items = int(max_items)
        self._frames: OrderedDict[str, Image.Image] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, sid: str) -> Image.Image:
        cached = self._frames.get(sid)
        if cached is not None:
            self._hits += 1
            self._frames.move_to_end(sid)
            return cached

        self._misses += 1
        path = self.frame_root / sid / "frame.png"
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(f"cannot load cached raw frame for {sid!r}: {path}") from exc

        if self.max_items:
            self._frames[sid] = image
            self._frames.move_to_end(sid)
            while len(self._frames) > self.max_items:
                self._frames.popitem(last=False)
        return image

    def clear(self) -> None:
        self._frames.clear()
        self._hits = 0
        self._misses = 0

    def cache_info(self) -> FrameCacheInfo:
        return FrameCacheInfo(self._hits, self._misses, self.max_items, len(self._frames))


@dataclass(frozen=True)
class SocialVLMInput:
    """One complete pre-tokenization pair sample."""

    annotation: SocialSample
    image: Image.Image
    prompt: str
    evidence: GraphEvidence
    vision_cache_key: str | None = None


class SocialInputDataset(Dataset):
    """Connect labelled pairs to cached frames and cached frozen graph evidence."""

    def __init__(
        self,
        manifest: str | Path | SocialAnnotationDataset,
        frame_root: str | Path,
        graph_cache: Mapping[str, Mapping[str, object]],
        *,
        raw_image_cache_size: int = 32,
        output_mode: str = "yesno",
        graph_evidence: str = "gtoken",
        generative_prompt_seed: int | None = None,
        include_graph_evidence: bool = True,
    ):
        if output_mode not in ("yesno", "generative"):
            raise ValueError(f"output_mode must be yesno/generative, got {output_mode!r}")
        if graph_evidence not in ("gtoken", "text"):
            raise ValueError(f"graph_evidence must be gtoken/text, got {graph_evidence!r}")
        if not include_graph_evidence and graph_evidence != "text":
            raise ValueError(
                "include_graph_evidence=False is only meaningful for graph_evidence='text' "
                f"(the ablation applies to the natural-language prompt); got {graph_evidence!r}"
            )
        self.graph_evidence = graph_evidence
        self.include_graph_evidence = bool(include_graph_evidence)
        self.annotations = (
            manifest if isinstance(manifest, SocialAnnotationDataset)
            else SocialAnnotationDataset(manifest)
        )
        self.graph_cache = graph_cache
        self.frames = RawFrameCache(frame_root, max_items=raw_image_cache_size)
        self.output_mode = output_mode
        self.generative_prompt_seed = (
            None if generative_prompt_seed is None else int(generative_prompt_seed)
        )
        required_sids = {sample.sid for sample in self.annotations}
        missing = sorted(required_sids.difference(graph_cache))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"graph cache is missing {len(missing)} manifest frames: {preview}{suffix}"
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def _generative_rng(self, sample: SocialSample):
        """Stable prompt sampling for validation/test; ``None`` preserves train augmentation."""
        if self.generative_prompt_seed is None:
            return None
        material = (
            f"{self.generative_prompt_seed}|{sample.sid}|{sample.task}|"
            f"{sample.raw_i}|{sample.raw_j}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "little")
        return random.Random(seed)

    def __getitem__(self, index: int) -> SocialVLMInput:
        sample = self.annotations[index]
        cache = self.graph_cache[sample.sid]
        text_mode = self.output_mode == "generative" and self.graph_evidence == "text"
        if self.output_mode == "generative":
            evidence = (
                assemble_text_graph_evidence(sample, cache) if text_mode
                else assemble_generative_graph(sample, cache)
            )
        else:
            evidence = assemble_graph_evidence(sample, cache)

        raw = self.frames.get(sample.sid)
        need_boxes = self.output_mode == "generative"
        box_a = box_b = None
        if need_boxes:
            bboxes = cache.get("head_bboxes")
            if not torch.is_tensor(bboxes) or bboxes.ndim != 2 or bboxes.shape[1] != 4:
                shape = (
                    tuple(bboxes.shape)
                    if torch.is_tensor(bboxes)
                    else type(bboxes).__name__
                )
                raise ValueError(
                    f"head_bboxes for {sample.sid!r} must have shape [N,4], got {shape}"
                )
            max_index = max(sample.person_i, sample.person_j)
            if max_index >= bboxes.shape[0]:
                raise IndexError(
                    f"pair person index {max_index} exceeds {sample.sid!r} bbox count "
                    f"{bboxes.shape[0]}"
                )
            box_a = bboxes[sample.person_i]
            box_b = bboxes[sample.person_j]
        # RawFrameCache images are immutable by contract. Returning the shared object
        # lets the collator deduplicate image preprocessing and vision encoding by frame.
        image = raw
        if text_mode:
            prompt = compose_text_prompt(
                sample.task, box_a.tolist(), box_b.tolist(), evidence,
                rng=self._generative_rng(sample),
                include_graph_evidence=self.include_graph_evidence,
            )
        elif self.output_mode == "generative":
            prompt = compose_generative_prompt(
                sample.task, box_a.tolist(), box_b.tolist(), rng=self._generative_rng(sample)
            )
        else:
            prompt = task_prompt(sample.task)
        return SocialVLMInput(
            annotation=sample,
            image=image,
            prompt=prompt,
            evidence=evidence,
            vision_cache_key=str(
                (self.frames.frame_root / sample.sid / "frame.png").resolve()
            ),
        )

    def raw_frame_path(self, index: int) -> Path:
        return frame_path(self.frames.frame_root, self.annotations[index])

    def sample_weights(
        self,
        *,
        balance_mode: str = "task",
        hard_floor: float | None = None,
        route_threshold: float | None = None,
    ) -> torch.Tensor:
        return sample_weights(
            self.annotations,
            self.graph_cache,
            balance_mode=balance_mode,
            hard_floor=hard_floor,
            route_threshold=route_threshold,
        )


def sample_graph_logit(sample: SocialSample, cache: Mapping[str, object]) -> torch.Tensor:
    """Read only the frozen pair logit, without assembling image/graph-token evidence."""
    value = cache.get(f"{sample.task}_logits")
    if not torch.is_tensor(value) or value.ndim != 2 or not value.is_floating_point():
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ValueError(
            f"{sample.task}_logits for {sample.sid!r} must be a floating [N,N] tensor, "
            f"got {shape}"
        )
    if value.shape[0] != value.shape[1]:
        raise ValueError(f"{sample.task}_logits must be square, got {tuple(value.shape)}")
    a, b = sample.person_i, sample.person_j
    if not 0 <= a < value.shape[0] or not 0 <= b < value.shape[0]:
        raise IndexError(
            f"pair ({a},{b}) is outside {sample.task}_logits shape {tuple(value.shape)}"
        )
    if sample.task == "lah":
        return value[a, b].detach().float().reshape(())
    return (0.5 * (value[a, b] + value[b, a])).detach().float().reshape(())


def graph_confidence(sample: SocialSample, cache: Mapping[str, object]) -> float:
    """Frozen-graph decision confidence for one pair: ``max(p, 1-p)`` in [0.5, 1]."""
    probability = torch.sigmoid(sample_graph_logit(sample, cache))
    return float(torch.maximum(probability, 1.0 - probability))


def partition_by_graph_confidence(
    annotations: SocialAnnotationDataset,
    graph_cache: Mapping[str, Mapping[str, object]],
    threshold: float,
) -> tuple[list[int], list[int]]:
    """Split pair indices into ``(high_confidence, low_confidence)`` by graph confidence.

    Pairs with ``conf > threshold`` are answered by the frozen graph, so the VLM is queried
    for the ``conf <= threshold`` remainder. ``threshold`` is on the ``max(p,1-p)``
    scale, hence in [0.5, 1].
    """
    if not 0.5 <= threshold <= 1.0:
        raise ValueError(f"routing threshold must be in [0.5, 1], got {threshold}")
    high: list[int] = []
    low: list[int] = []
    for index, sample in enumerate(annotations):
        if graph_confidence(sample, graph_cache[sample.sid]) > threshold:
            high.append(index)
        else:
            low.append(index)
    return high, low


def sample_weights(
    annotations: SocialAnnotationDataset,
    graph_cache: Mapping[str, Mapping[str, object]],
    *,
    balance_mode: str = "task",
    hard_floor: float | None = None,
    route_threshold: float | None = None,
) -> torch.Tensor:
    """Task/label balancing, optionally multiplied by frozen-graph error hardness.

    When ``route_threshold`` is set, high-confidence pairs (answered by the frozen graph at
    inference) get zero weight, so training draws only from the low-confidence pairs the VLM
    is responsible for.
    """
    if balance_mode not in ("none", "task", "task_label"):
        raise ValueError(
            f"balance_mode must be none/task/task_label, got {balance_mode!r}"
        )
    if hard_floor is not None and not 0 <= hard_floor <= 1:
        raise ValueError(f"hard_floor must be in [0,1], got {hard_floor}")
    routed_out = set()
    if route_threshold is not None:
        high, _low = partition_by_graph_confidence(annotations, graph_cache, route_threshold)
        routed_out = set(high)
        if len(routed_out) == len(annotations):
            raise ValueError(
                "routing threshold leaves no low-confidence training pairs; lower it"
            )
    if balance_mode == "task_label":
        keys = [(sample.task, sample.label) for sample in annotations.samples]
    elif balance_mode == "task":
        keys = [sample.task for sample in annotations.samples]
    else:
        keys = [None] * len(annotations)
    counts = Counter(keys)
    weights = torch.empty(len(annotations), dtype=torch.double)
    for index, sample in enumerate(annotations.samples):
        if index in routed_out:
            weights[index] = 0.0
            continue
        weight = 1.0 if balance_mode == "none" else 1.0 / counts[keys[index]]
        if hard_floor is not None:
            graph_logit = sample_graph_logit(sample, graph_cache[sample.sid])
            probability = torch.sigmoid(graph_logit).item()
            weight *= hard_floor + abs(float(sample.label) - probability)
        weights[index] = weight
    return weights


def task_pos_weights(
    annotations: SocialAnnotationDataset,
    *,
    minimum: float = 0.2,
    maximum: float = 5.0,
) -> dict[str, float]:
    """Compute clamped ``negative/positive`` ratios for every social task."""
    if not 0 < minimum <= maximum:
        raise ValueError(f"invalid pos-weight clamp [{minimum},{maximum}]")
    counts = Counter((sample.task, sample.label) for sample in annotations.samples)
    output = {}
    for task in SOCIAL_TASK_ID:
        positive = counts[(task, 1)]
        negative = counts[(task, 0)]
        if positive == 0 or negative == 0:
            raise ValueError(
                f"task {task!r} needs both labels for pos_weight, got pos={positive}, neg={negative}"
            )
        output[task] = min(max(negative / positive, minimum), maximum)
    return output


@dataclass(frozen=True)
class GraphControlInput:
    annotation: SocialSample
    graph_logit: torch.Tensor


class GraphControlDataset(Dataset):
    """Same labelled rows as SocialInputDataset, with no image or Qwen dependency."""

    def __init__(
        self,
        manifest: str | Path | SocialAnnotationDataset,
        graph_cache: Mapping[str, Mapping[str, object]],
    ):
        self.annotations = (
            manifest if isinstance(manifest, SocialAnnotationDataset)
            else SocialAnnotationDataset(manifest)
        )
        self.graph_cache = graph_cache
        missing = sorted({s.sid for s in self.annotations.samples}.difference(graph_cache))
        if missing:
            raise ValueError(f"graph cache is missing {len(missing)} manifest frames")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> GraphControlInput:
        sample = self.annotations[index]
        return GraphControlInput(
            annotation=sample,
            graph_logit=sample_graph_logit(sample, self.graph_cache[sample.sid]),
        )

    def sample_weights(
        self, *, balance_mode: str = "task", hard_floor: float | None = None
    ) -> torch.Tensor:
        return sample_weights(
            self.annotations,
            self.graph_cache,
            balance_mode=balance_mode,
            hard_floor=hard_floor,
        )


@dataclass(frozen=True)
class GraphFeatureControlInput:
    annotation: SocialSample
    evidence: GraphEvidence


class GraphFeatureControlDataset(Dataset):
    """Six-slot graph evidence with no image decoding, processor, or Qwen path."""

    def __init__(
        self,
        manifest: str | Path | SocialAnnotationDataset,
        graph_cache: Mapping[str, Mapping[str, object]],
    ):
        self.annotations = (
            manifest if isinstance(manifest, SocialAnnotationDataset)
            else SocialAnnotationDataset(manifest)
        )
        self.graph_cache = graph_cache
        missing = sorted({s.sid for s in self.annotations.samples}.difference(graph_cache))
        if missing:
            raise ValueError(f"graph cache is missing {len(missing)} manifest frames")
        if not self.annotations.samples:
            raise ValueError("graph feature control requires at least one annotation")
        # Establish the MLP input contract before optimizer construction. Every later
        # sample is independently validated by assemble_graph_evidence.
        first = self.annotations.samples[0]
        self.feature_dim = assemble_graph_evidence(
            first, self.graph_cache[first.sid]
        ).feature_dim

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> GraphFeatureControlInput:
        sample = self.annotations[index]
        evidence = assemble_graph_evidence(
            sample, self.graph_cache[sample.sid]
        )
        if evidence.feature_dim != self.feature_dim:
            raise ValueError(
                f"inconsistent graph feature dimension for {sample.sid!r}: "
                f"expected {self.feature_dim}, got {evidence.feature_dim}"
            )
        return GraphFeatureControlInput(annotation=sample, evidence=evidence)

    def sample_weights(
        self, *, balance_mode: str = "task", hard_floor: float | None = None
    ) -> torch.Tensor:
        return sample_weights(
            self.annotations,
            self.graph_cache,
            balance_mode=balance_mode,
            hard_floor=hard_floor,
        )


def control_collate(items: list[GraphControlInput]) -> dict[str, object]:
    if not items:
        raise ValueError("cannot collate an empty graph-control batch")
    return {
        "graph_logits": torch.stack([item.graph_logit for item in items]),
        "task_ids": torch.tensor(
            [SOCIAL_TASK_ID[item.annotation.task] for item in items], dtype=torch.long
        ),
        "pair_labels": torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32
        ),
        "eval_keys": [item.annotation.eval_key for item in items],
    }


def feature_control_collate(
    items: list[GraphFeatureControlInput],
) -> dict[str, object]:
    if not items:
        raise ValueError("cannot collate an empty graph-feature-control batch")
    return {
        "pair_graph": stack_graph_evidence([item.evidence for item in items]),
        "task_ids": torch.tensor(
            [SOCIAL_TASK_ID[item.annotation.task] for item in items], dtype=torch.long
        ),
        "pair_labels": torch.tensor(
            [item.annotation.label for item in items], dtype=torch.float32
        ),
        "eval_keys": [item.annotation.eval_key for item in items],
    }
