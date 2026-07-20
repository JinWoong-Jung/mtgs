"""Frozen MTGS graph features injected as inline Qwen token embeddings.

The graph cache is an inference-only input.  This module deliberately exposes a
small, whitelisted token schema instead of passing cache dictionaries through the
VLM: labels and other evaluation metadata can therefore never enter the model.

``text`` mode has no dependency on this module at runtime.  ``text_tokens`` keeps
the ordinary natural-language graph evidence and additionally replaces registered
placeholder tokens with projected dense graph features.  Adding a future feature
requires one new :class:`GraphTokenSlot` plus an adapter branch; callers continue
to use the generic payload/collator contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


GRAPH_EVIDENCE_MODES = ("text", "text_tokens")
GRAPH_TOKEN_FEATURES = ("gaze_heatmap", "edge_pp")


@dataclass(frozen=True)
class GraphTokenSlot:
    """One inline placeholder and its cache-backed feature family."""

    name: str
    feature: str
    marker: str


# Slot order is serialized in checkpoints and used for compact batch tensors.  Do not
# reorder existing entries; append future slots instead.
GRAPH_TOKEN_SLOTS = (
    GraphTokenSlot("heatmap_a", "gaze_heatmap", "<|graph_heatmap_a|>"),
    GraphTokenSlot("heatmap_b", "gaze_heatmap", "<|graph_heatmap_b|>"),
    GraphTokenSlot("edge_ab", "edge_pp", "<|graph_edge_ab|>"),
    GraphTokenSlot("edge_ba", "edge_pp", "<|graph_edge_ba|>"),
)
GRAPH_TOKEN_SLOT_INDEX = {slot.name: index for index, slot in enumerate(GRAPH_TOKEN_SLOTS)}
GRAPH_TOKEN_SLOT_BY_NAME = {slot.name: slot for slot in GRAPH_TOKEN_SLOTS}


@dataclass(frozen=True)
class GraphTokenPayload:
    """Whitelisted frozen cache tensors for one social-relation prompt.

    Keys are slot names such as ``heatmap_a`` or ``edge_ab``.  Values are detached
    cache views; the adapter receives them as data and no graph gradients exist.
    """

    values: Mapping[str, torch.Tensor]


def normalize_graph_evidence_mode(value: str | None) -> str:
    mode = "text" if value is None else str(value).strip().lower()
    if mode not in GRAPH_EVIDENCE_MODES:
        raise ValueError(
            f"graph_evidence_mode must be one of {GRAPH_EVIDENCE_MODES}, got {value!r}"
        )
    return mode


def normalize_graph_token_features(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        values = GRAPH_TOKEN_FEATURES
    features = tuple(str(value).strip() for value in values)
    if not features:
        raise ValueError("text_tokens requires at least one graph token feature")
    unknown = set(features).difference(GRAPH_TOKEN_FEATURES)
    if unknown:
        raise ValueError(
            f"unsupported graph token features: {sorted(unknown)}; "
            f"available={GRAPH_TOKEN_FEATURES}"
        )
    if len(set(features)) != len(features):
        raise ValueError(f"graph token features must be unique, got {features}")
    return features


def graph_token_slots_for(task: str, features: Sequence[str]) -> tuple[GraphTokenSlot, ...]:
    """Return the task-specific inline slot sequence.

    LAH evaluates only A->B.  LAEO and SA need both people/directions; for SA the
    two edge tokens are pair context rather than an assertion that either directed
    looking relation holds.
    """
    enabled = set(features)
    if task == "lah":
        names = ("heatmap_a", "edge_ab")
    elif task in ("laeo", "sa"):
        names = ("heatmap_a", "edge_ab", "heatmap_b", "edge_ba")
    else:
        raise ValueError(f"unknown social task {task!r}")
    return tuple(
        GRAPH_TOKEN_SLOT_BY_NAME[name]
        for name in names
        if GRAPH_TOKEN_SLOT_BY_NAME[name].feature in enabled
    )


def graph_token_markers(task: str, features: Sequence[str]) -> dict[str, str]:
    """Prompt-facing marker strings for the selected task/features."""
    return {slot.name: slot.marker for slot in graph_token_slots_for(task, features)}


def graph_token_strings() -> tuple[str, ...]:
    """All stable special tokens; saved tokenizer schemas always use this order."""
    return tuple(slot.marker for slot in GRAPH_TOKEN_SLOTS)


def configure_graph_tokenizer(tokenizer) -> dict[str, int]:
    """Register stable one-token placeholders and return their vocabulary ids.

    This is called only in ``text_tokens`` mode.  It is intentionally not invoked
    for ``text`` so the established baseline tokenizer/model remain untouched.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": list(graph_token_strings())})
    ids: dict[str, int] = {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for slot in GRAPH_TOKEN_SLOTS:
        token_id = int(tokenizer.convert_tokens_to_ids(slot.marker))
        if token_id < 0 or (unk_id is not None and token_id == int(unk_id)):
            raise ValueError(f"graph placeholder {slot.marker!r} was not registered as one token")
        encoded = tokenizer.encode(slot.marker, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(
                f"graph placeholder {slot.marker!r} must tokenize to one id, got {encoded}"
            )
        ids[slot.name] = token_id
    return ids


def extract_graph_token_payload(
    *,
    task: str,
    person_a: int,
    person_b: int,
    cache: Mapping[str, object],
    features: Sequence[str],
) -> GraphTokenPayload:
    """Read only prediction features for one canonical A/B relation from cache."""
    slots = graph_token_slots_for(task, features)
    values: dict[str, torch.Tensor] = {}
    need_heatmap = any(slot.feature == "gaze_heatmap" for slot in slots)
    need_edge = any(slot.feature == "edge_pp" for slot in slots)

    heatmap = None
    if need_heatmap:
        heatmap = cache.get("gaze_heatmap")
        if not torch.is_tensor(heatmap) or heatmap.ndim != 3 or not heatmap.is_floating_point():
            shape = tuple(heatmap.shape) if torch.is_tensor(heatmap) else type(heatmap).__name__
            raise ValueError(f"gaze_heatmap must be floating [N,H,W], got {shape}")

    edge = None
    if need_edge:
        edge = cache.get("edge_pp")
        if not torch.is_tensor(edge) or edge.ndim != 3 or not edge.is_floating_point():
            shape = tuple(edge.shape) if torch.is_tensor(edge) else type(edge).__name__
            raise ValueError(f"edge_pp must be floating [N,N,D], got {shape}")
        if edge.shape[0] != edge.shape[1]:
            raise ValueError(f"edge_pp must be square in source/target axes, got {tuple(edge.shape)}")

    n = int(heatmap.shape[0] if heatmap is not None else edge.shape[0])
    if not 0 <= person_a < n or not 0 <= person_b < n:
        raise IndexError(f"canonical pair ({person_a},{person_b}) is outside graph cache with N={n}")
    if heatmap is not None and heatmap.shape[0] != n:
        raise ValueError("gaze_heatmap and edge_pp disagree on person count")

    for slot in slots:
        if slot.name == "heatmap_a":
            values[slot.name] = heatmap[person_a].detach()
        elif slot.name == "heatmap_b":
            values[slot.name] = heatmap[person_b].detach()
        elif slot.name == "edge_ab":
            values[slot.name] = edge[person_a, person_b].detach()
        elif slot.name == "edge_ba":
            values[slot.name] = edge[person_b, person_a].detach()
        else:  # Defensive: future registry additions require an explicit extractor branch.
            raise ValueError(f"no cache extractor registered for graph token slot {slot.name!r}")
    return GraphTokenPayload(values=values)


class GraphTokenAdapter(nn.Module):
    """Project whitelisted heatmap/edge cache features into Qwen token embeddings."""

    def __init__(self, *, edge_dim: int, hidden_size: int, dropout: float = 0.10):
        super().__init__()
        if edge_dim <= 0 or hidden_size <= 0:
            raise ValueError("edge_dim and hidden_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"graph token dropout must be in [0,1), got {dropout}")
        self.edge_dim = int(edge_dim)
        self.hidden_size = int(hidden_size)
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(64),
            nn.Linear(64, hidden_size),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, 512),
            nn.GELU(),
            nn.Linear(512, hidden_size),
        )
        self.slot_embedding = nn.Embedding(len(GRAPH_TOKEN_SLOTS), hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, heatmaps: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Return ``[B,4,D]`` in fixed ``GRAPH_TOKEN_SLOTS`` order.

        Heatmaps are cache logits. A spatial softmax converts them to a normalized
        spatial distribution before the tiny CNN. The final global pooling intentionally
        yields one summary token, so this branch can learn distributional structure
        (for example, concentration and spread) but does not explicitly preserve an
        absolute ``(x, y)`` location. The prompt's gaze-point text carries that coordinate.
        """
        if heatmaps.ndim != 4 or heatmaps.shape[1] != 2:
            raise ValueError(f"heatmaps must be [B,2,H,W], got {tuple(heatmaps.shape)}")
        if edges.ndim != 3 or edges.shape[1:] != (2, self.edge_dim):
            raise ValueError(
                f"edges must be [B,2,{self.edge_dim}], got {tuple(edges.shape)}"
            )
        if heatmaps.shape[0] != edges.shape[0]:
            raise ValueError("heatmap and edge batches must have the same size")
        batch, _, height, width = heatmaps.shape
        normalized_heatmaps = F.softmax(heatmaps.float().reshape(batch * 2, -1), dim=-1)
        normalized_heatmaps = normalized_heatmaps.reshape(batch * 2, 1, height, width)
        heat_tokens = self.heatmap_encoder(normalized_heatmaps).reshape(batch, 2, -1)
        edge_tokens = self.edge_encoder(edges.float())
        tokens = torch.stack(
            (heat_tokens[:, 0], heat_tokens[:, 1], edge_tokens[:, 0], edge_tokens[:, 1]),
            dim=1,
        )
        slot_ids = torch.arange(len(GRAPH_TOKEN_SLOTS), device=tokens.device)
        return self.dropout(tokens + self.slot_embedding(slot_ids).unsqueeze(0))
