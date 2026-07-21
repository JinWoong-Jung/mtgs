"""Pair-wise image/prompt/evidence assembly with a bounded raw-frame cache."""

from __future__ import annotations

import hashlib
import math
import random

from collections import Counter, OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.social.data import (
    SocialAnnotationDataset,
    SocialSample,
    SOCIAL_TASK_ID,
    frame_path,
)
from vlm.cache.overlay import build_overlay_pair
from vlm.social.evidence import assemble_text_graph_evidence
from vlm.social.graph_tokens import (
    GraphTokenPayload,
    extract_graph_token_payload,
    graph_token_markers,
    normalize_graph_evidence_mode,
    normalize_graph_token_features,
)
from vlm.social.prompt import compose_text_prompt


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
    vision_cache_key: str | None = None
    graph_token_payload: GraphTokenPayload | None = None


class SocialInputDataset(Dataset):
    """Connect labelled pairs to cached frames and cached frozen graph evidence.

    When ``routing_threshold`` is supplied, low-confidence pairs carry a compact prompt
    cue that the graph needs visual review. The predicate is deliberately identical to
    :func:`partition_by_graph_confidence`: graph answers only ``conf > threshold`` and
    the VLM-reviewed remainder is ``conf <= threshold``.
    """

    def __init__(
        self,
        manifest: str | Path | SocialAnnotationDataset,
        frame_root: str | Path,
        graph_cache: Mapping[str, Mapping[str, object]],
        *,
        raw_image_cache_size: int = 32,
        generative_prompt_seed: int | None = None,
        include_graph_evidence: bool = True,
        routing_threshold: float | None = None,
        graph_evidence_mode: str = "text",
        graph_token_features: Sequence[str] | None = None,
        draw_pair_bboxes: bool = False,
        draw_gaze_arrows: bool = False,
    ):
        self.draw_pair_bboxes = bool(draw_pair_bboxes)
        self.draw_gaze_arrows = bool(draw_gaze_arrows)
        if self.draw_gaze_arrows and not self.draw_pair_bboxes:
            raise ValueError("draw_gaze_arrows requires draw_pair_bboxes=True")
        self.include_graph_evidence = bool(include_graph_evidence)
        self.graph_evidence_mode = normalize_graph_evidence_mode(graph_evidence_mode)
        if self.graph_evidence_mode == "text_tokens":
            if not self.include_graph_evidence:
                raise ValueError("text_tokens requires include_graph_evidence=True")
            self.graph_token_features = normalize_graph_token_features(graph_token_features)
        else:
            self.graph_token_features = ()
        if routing_threshold is not None and not 0.5 <= float(routing_threshold) <= 1.0:
            raise ValueError(
                "routing_threshold must be in [0.5, 1], "
                f"got {routing_threshold}"
            )
        self.routing_threshold = (
            None if routing_threshold is None else float(routing_threshold)
        )
        self.annotations = (
            manifest if isinstance(manifest, SocialAnnotationDataset)
            else SocialAnnotationDataset(manifest)
        )
        self.graph_cache = graph_cache
        self.frames = RawFrameCache(frame_root, max_items=raw_image_cache_size)
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
        evidence = assemble_text_graph_evidence(sample, cache)

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
        image = self.frames.get(sample.sid)
        frame_key = str((self.frames.frame_root / sample.sid / "frame.png").resolve())
        if self.draw_pair_bboxes:
            # Draw the queried pair's head boxes (A=red source, B=blue target) so the VLM
            # sees which people the prompt's "Person A/B" refer to, instead of resolving
            # text coordinates against the raw frame. build_overlay_pair copies the image,
            # so the shared cache frame is never mutated. Each pair now has a distinct
            # image, so the vision-reuse cache key must be pair-unique (otherwise the
            # collator would encode only the first pair's boxed frame and reuse it for the
            # rest of the frame). This intentionally forgoes per-frame vision reuse.
            gaze_vecs = None
            if self.draw_gaze_arrows:
                gaze_vecs = cache.get("gaze_vecs")
                if not torch.is_tensor(gaze_vecs) or gaze_vecs.ndim != 2 or gaze_vecs.shape[1] != 2:
                    shape = tuple(gaze_vecs.shape) if torch.is_tensor(gaze_vecs) else type(gaze_vecs).__name__
                    raise ValueError(f"gaze_vecs for {sample.sid!r} must have shape [N,2], got {shape}")
            image = build_overlay_pair(
                image,
                sample.person_i,
                sample.person_j,
                bboxes,
                {sample.person_i: "Person A", sample.person_j: "Person B"},
                task=sample.task if gaze_vecs is not None else None,
                gaze_vecs=gaze_vecs,
            )
            frame_key = f"{frame_key}::A{sample.person_i}B{sample.person_j}"
        graph_needs_visual_review = (
            self.routing_threshold is not None
            and graph_confidence(sample, cache) <= self.routing_threshold
        )
        token_markers = (
            graph_token_markers(sample.task, self.graph_token_features)
            if self.graph_evidence_mode == "text_tokens"
            else None
        )
        graph_token_payload = (
            extract_graph_token_payload(
                task=sample.task,
                person_a=sample.person_i,
                person_b=sample.person_j,
                cache=cache,
                features=self.graph_token_features,
            )
            if self.graph_evidence_mode == "text_tokens"
            else None
        )
        prompt = compose_text_prompt(
            sample.task, box_a.tolist(), box_b.tolist(), evidence,
            rng=self._generative_rng(sample),
            include_graph_evidence=self.include_graph_evidence,
            graph_needs_visual_review=graph_needs_visual_review,
            graph_token_markers=token_markers,
        )
        return SocialVLMInput(
            annotation=sample,
            image=image,
            prompt=prompt,
            vision_cache_key=frame_key,
            graph_token_payload=graph_token_payload,
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


def boost_low_confidence_weights(
    weights: torch.Tensor,
    annotations: SocialAnnotationDataset,
    graph_cache: Mapping[str, Mapping[str, object]],
    *,
    threshold: float,
    multiplier: float,
) -> tuple[torch.Tensor, list[int]]:
    """Oversample the VLM-routed remainder without discarding graph-fallback pairs.

    ``multiplier=1`` is a no-op. Larger values multiply only the weights of pairs
    whose frozen-graph confidence is ``<= threshold`` -- precisely the pairs the
    validation/test router sends to the VLM. This is sampler weighting, not loss
    weighting: generative yes/no CE remains unchanged for every sampled example.
    """
    if weights.ndim != 1 or len(weights) != len(annotations):
        raise ValueError(
            "weights must be a rank-1 tensor aligned with annotations: "
            f"got shape={tuple(weights.shape)}, annotations={len(annotations)}"
        )
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError(
            "low-confidence sampling multiplier must be finite and >= 1, "
            f"got {multiplier}"
        )
    _high, low = partition_by_graph_confidence(annotations, graph_cache, threshold)
    if not low:
        raise ValueError(
            "routing threshold leaves no low-confidence pairs to boost; raise it"
        )
    boosted = weights.clone()
    if multiplier != 1.0:
        boosted[low] *= multiplier
    return boosted, low


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
