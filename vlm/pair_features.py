"""Task-conditioned MTGS graph evidence in one fixed six-slot layout.

This module is the boundary between cached MTGS graph tensors and the new pair-wise
VLM.  It only gathers and validates evidence; projection into the VLM hidden size and
learned N/A embeddings belong to the later injection module.

Every task has the same semantic slot order::

    person_a, person_b, relation_ab, relation_ba, heatmap_a, heatmap_b

Person slots have three fixed channels ``[v_src, v_tgt, E(person->Null_in)]`` plus a
channel-presence mask.  Missing relation/heatmap slots are zero-filled here and marked
absent, so a downstream projector can replace them with learned N/A tokens instead of
mistaking a zero vector for real evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from vlm.pair_dataset import PairSample, SOCIAL_TASKS


SLOT_NAMES = (
    "person_a",
    "person_b",
    "relation_ab",
    "relation_ba",
    "heatmap_a",
    "heatmap_b",
)
PERSON_CHANNEL_NAMES = ("src", "tgt", "null_in")
PERSON_CHANNEL = {name: index for index, name in enumerate(PERSON_CHANNEL_NAMES)}


def _check_shape(tensor: torch.Tensor, expected: tuple[int | None, ...], name: str) -> None:
    if tensor.ndim != len(expected):
        raise ValueError(f"{name} must have {len(expected)} dimensions, got {tuple(tensor.shape)}")
    for axis, (actual, wanted) in enumerate(zip(tensor.shape, expected)):
        if wanted is not None and actual != wanted:
            raise ValueError(
                f"{name} axis {axis} must have size {wanted}, got {tuple(tensor.shape)}"
            )


@dataclass(frozen=True)
class PairGraphEvidence:
    """Fixed-shape graph evidence for one canonical :class:`PairSample`."""

    task: str
    person_features: torch.Tensor          # [2, 3, D]: A/B x src/tgt/null_in
    person_channel_present: torch.Tensor   # [2, 3] bool
    relation_features: torch.Tensor        # [2, D]: A->B, B->A
    relation_present: torch.Tensor         # [2] bool
    heatmap_features: torch.Tensor         # [2, H, W]: HM_A, HM_B
    heatmap_present: torch.Tensor          # [2] bool
    graph_logit: torch.Tensor              # [] frozen graph residual base

    def __post_init__(self) -> None:
        if self.task not in SOCIAL_TASKS:
            raise ValueError(f"unknown social task {self.task!r}")
        _check_shape(self.person_features, (2, 3, None), "person_features")
        feature_dim = self.person_features.shape[-1]
        _check_shape(self.person_channel_present, (2, 3), "person_channel_present")
        _check_shape(self.relation_features, (2, feature_dim), "relation_features")
        _check_shape(self.relation_present, (2,), "relation_present")
        _check_shape(self.heatmap_features, (2, None, None), "heatmap_features")
        _check_shape(self.heatmap_present, (2,), "heatmap_present")
        _check_shape(self.graph_logit, (), "graph_logit")
        for name, mask in (
            ("person_channel_present", self.person_channel_present),
            ("relation_present", self.relation_present),
            ("heatmap_present", self.heatmap_present),
        ):
            if mask.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean, got {mask.dtype}")

    @property
    def feature_dim(self) -> int:
        return self.person_features.shape[-1]

    @property
    def slot_presence(self) -> torch.Tensor:
        """Presence in ``SLOT_NAMES`` order; both person slots always exist."""
        people = torch.ones(2, dtype=torch.bool, device=self.relation_present.device)
        return torch.cat((people, self.relation_present, self.heatmap_present))

    def to(self, device: torch.device | str) -> "PairGraphEvidence":
        return PairGraphEvidence(
            task=self.task,
            person_features=self.person_features.to(device),
            person_channel_present=self.person_channel_present.to(device),
            relation_features=self.relation_features.to(device),
            relation_present=self.relation_present.to(device),
            heatmap_features=self.heatmap_features.to(device),
            heatmap_present=self.heatmap_present.to(device),
            graph_logit=self.graph_logit.to(device),
        )


@dataclass(frozen=True)
class PairGraphBatch:
    """A stacked batch preserving the same six-slot contract."""

    tasks: tuple[str, ...]
    person_features: torch.Tensor          # [B, 2, 3, D]
    person_channel_present: torch.Tensor   # [B, 2, 3]
    relation_features: torch.Tensor        # [B, 2, D]
    relation_present: torch.Tensor         # [B, 2]
    heatmap_features: torch.Tensor         # [B, 2, H, W]
    heatmap_present: torch.Tensor          # [B, 2]
    graph_logits: torch.Tensor             # [B]

    @property
    def slot_presence(self) -> torch.Tensor:
        people = torch.ones(
            (len(self.tasks), 2), dtype=torch.bool, device=self.relation_present.device
        )
        return torch.cat((people, self.relation_present, self.heatmap_present), dim=1)

    def to(self, device: torch.device | str) -> "PairGraphBatch":
        return PairGraphBatch(
            tasks=self.tasks,
            person_features=self.person_features.to(device),
            person_channel_present=self.person_channel_present.to(device),
            relation_features=self.relation_features.to(device),
            relation_present=self.relation_present.to(device),
            heatmap_features=self.heatmap_features.to(device),
            heatmap_present=self.heatmap_present.to(device),
            graph_logits=self.graph_logits.to(device),
        )


def _tensor(cache: Mapping[str, object], name: str, ndim: int) -> torch.Tensor:
    value = cache.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"graph cache field {name!r} is missing or is not a tensor")
    if value.ndim != ndim:
        raise ValueError(f"graph cache field {name!r} must be {ndim}D, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise ValueError(f"graph cache field {name!r} must be floating point, got {value.dtype}")
    # Keep the cached dtype here. The fixed output buffers below cast only the selected
    # pair slices to float32, avoiding a full per-frame half->float copy for every pair.
    return value.detach()


def _validate_cache_and_pair(
    sample: PairSample, cache: Mapping[str, object]
) -> tuple[torch.Tensor, ...]:
    v_src = _tensor(cache, "v_src", 2)
    v_tgt = _tensor(cache, "v_tgt", 2)
    edge_pp = _tensor(cache, "edge_pp", 3)
    edge_null_in = _tensor(cache, "edge_null_in", 2)
    heatmaps = _tensor(cache, "gaze_heatmap", 3)
    logits = _tensor(cache, f"{sample.task}_logits", 2)

    num_people, feature_dim = v_src.shape
    if feature_dim <= 0 or num_people <= 0:
        raise ValueError(f"v_src has invalid shape {tuple(v_src.shape)}")
    _check_shape(v_tgt, (None, feature_dim), "v_tgt")
    if v_tgt.shape[0] < num_people:
        raise ValueError(
            f"v_tgt must contain at least {num_people} person targets, got {v_tgt.shape[0]}"
        )
    _check_shape(edge_pp, (num_people, num_people, feature_dim), "edge_pp")
    _check_shape(edge_null_in, (num_people, feature_dim), "edge_null_in")
    _check_shape(heatmaps, (num_people, None, None), "gaze_heatmap")
    _check_shape(logits, (num_people, num_people), f"{sample.task}_logits")

    devices = {tensor.device for tensor in (v_src, v_tgt, edge_pp, edge_null_in, heatmaps, logits)}
    if len(devices) != 1:
        raise ValueError(f"graph cache tensors must share one device, got {sorted(map(str, devices))}")

    for name, index in (("person_i", sample.person_i), ("person_j", sample.person_j)):
        if not 0 <= index < num_people:
            raise IndexError(f"{name}={index} is outside graph cache person range [0, {num_people})")

    visibility = cache.get("vis_mask", cache.get("person_mask"))
    if visibility is not None:
        if not torch.is_tensor(visibility) or visibility.ndim != 1 or len(visibility) != num_people:
            shape = tuple(visibility.shape) if torch.is_tensor(visibility) else type(visibility).__name__
            raise ValueError(f"vis/person mask must have shape ({num_people},), got {shape}")
        invisible = [
            name
            for name, index in (("person_i", sample.person_i), ("person_j", sample.person_j))
            if not bool(visibility[index])
        ]
        if invisible:
            raise ValueError(f"pair references non-visible graph slots: {', '.join(invisible)}")

    return v_src, v_tgt, edge_pp, edge_null_in, heatmaps, logits


def assemble_pair_graph_evidence(
    sample: PairSample, cache: Mapping[str, object]
) -> PairGraphEvidence:
    """Gather one pair without any further index transposition.

    ``sample.person_i`` is Person A and ``sample.person_j`` is Person B. For LAH,
    Unit 1 already guarantees that this means A looks at B.
    """
    v_src, v_tgt, edge_pp, edge_null_in, heatmaps, logits = _validate_cache_and_pair(
        sample, cache
    )
    a, b = sample.person_i, sample.person_j
    feature_dim = v_src.shape[-1]
    device = v_src.device

    person = torch.zeros((2, 3, feature_dim), dtype=torch.float32, device=device)
    person_present = torch.zeros((2, 3), dtype=torch.bool, device=device)
    relation = torch.zeros((2, feature_dim), dtype=torch.float32, device=device)
    relation_present = torch.zeros(2, dtype=torch.bool, device=device)
    heatmap = torch.zeros((2, *heatmaps.shape[-2:]), dtype=torch.float32, device=device)
    heatmap_present = torch.zeros(2, dtype=torch.bool, device=device)

    if sample.task == "lah":
        person[0, PERSON_CHANNEL["src"]] = v_src[a]
        person[1, PERSON_CHANNEL["tgt"]] = v_tgt[b]
        person_present[0, PERSON_CHANNEL["src"]] = True
        person_present[1, PERSON_CHANNEL["tgt"]] = True
    elif sample.task == "laeo":
        person[0, PERSON_CHANNEL["src"]] = v_src[a]
        person[0, PERSON_CHANNEL["tgt"]] = v_tgt[a]
        person[1, PERSON_CHANNEL["src"]] = v_src[b]
        person[1, PERSON_CHANNEL["tgt"]] = v_tgt[b]
        person_present[:, :2] = True
    elif sample.task == "sa":
        person[0, PERSON_CHANNEL["src"]] = v_src[a]
        person[0, PERSON_CHANNEL["null_in"]] = edge_null_in[a]
        person[1, PERSON_CHANNEL["src"]] = v_src[b]
        person[1, PERSON_CHANNEL["null_in"]] = edge_null_in[b]
        person_present[:, PERSON_CHANNEL["src"]] = True
        person_present[:, PERSON_CHANNEL["null_in"]] = True
    else:  # PairSample validation should make this unreachable.
        raise ValueError(f"unknown social task {sample.task!r}")

    # All tasks use A->B. LAEO and SA additionally use B->A.
    relation[0] = edge_pp[a, b]
    relation_present[0] = True
    if sample.task in ("laeo", "sa"):
        relation[1] = edge_pp[b, a]
        relation_present[1] = True

    # LAH needs the looker's heatmap (A); mutual/shared tasks need both.
    heatmap[0] = heatmaps[a]
    heatmap_present[0] = True
    if sample.task in ("laeo", "sa"):
        heatmap[1] = heatmaps[b]
        heatmap_present[1] = True

    if sample.task == "lah":
        graph_logit = logits[a, b].float()
    else:
        graph_logit = 0.5 * (logits[a, b].float() + logits[b, a].float())

    return PairGraphEvidence(
        task=sample.task,
        person_features=person,
        person_channel_present=person_present,
        relation_features=relation,
        relation_present=relation_present,
        heatmap_features=heatmap,
        heatmap_present=heatmap_present,
        graph_logit=graph_logit.reshape(()),
    )


def stack_pair_graph_evidence(items: Sequence[PairGraphEvidence]) -> PairGraphBatch:
    """Stack compatible evidence objects for the future pair-wise collate function."""
    if not items:
        raise ValueError("cannot stack an empty evidence sequence")
    return PairGraphBatch(
        tasks=tuple(item.task for item in items),
        person_features=torch.stack([item.person_features for item in items]),
        person_channel_present=torch.stack([item.person_channel_present for item in items]),
        relation_features=torch.stack([item.relation_features for item in items]),
        relation_present=torch.stack([item.relation_present for item in items]),
        heatmap_features=torch.stack([item.heatmap_features for item in items]),
        heatmap_present=torch.stack([item.heatmap_present for item in items]),
        graph_logits=torch.stack([item.graph_logit for item in items]),
    )


# ── EyeVLM-generative graph tokens (our contribution): v_src / v_tgt / edge only ──────────
@dataclass(frozen=True)
class GenGraphEvidence:
    """Up-to-4 task-specific graph feature vectors for the generative prompt's <gtok> slots.

    LAH  : [v_src[i], v_tgt[j], E[i->j]]           present=[1,1,1,0]
    LAEO : [v_src[i], v_src[j], E[i->j], E[j->i]]  present=[1,1,1,1]
    SA   : [v_src[i], v_src[j]]                    present=[1,1,0,0]
    """
    task: str
    features: torch.Tensor          # [4, De] (unused slots zero-filled)
    present: torch.Tensor           # [4] bool

    def to(self, device):
        return GenGraphEvidence(self.task, self.features.to(device), self.present.to(device))


def assemble_generative_graph(sample: PairSample, cache: Mapping[str, object]) -> GenGraphEvidence:
    v_src = _tensor(cache, "v_src", 2)
    v_tgt = _tensor(cache, "v_tgt", 2)
    edge = _tensor(cache, "edge_pp", 3)
    n = v_src.shape[0]
    i, j = sample.person_i, sample.person_j
    for name, idx in (("person_i", i), ("person_j", j)):
        if not 0 <= idx < n:
            raise IndexError(f"{name}={idx} outside person range [0,{n})")
    de = v_src.shape[-1]
    feats = torch.zeros(4, de, dtype=torch.float32)
    present = torch.zeros(4, dtype=torch.bool)
    if sample.task == "lah":
        feats[0], feats[1], feats[2] = v_src[i].float(), v_tgt[j].float(), edge[i, j].float()
        present[:3] = True
    elif sample.task == "laeo":
        feats[0], feats[1] = v_src[i].float(), v_src[j].float()
        feats[2], feats[3] = edge[i, j].float(), edge[j, i].float()
        present[:4] = True
    elif sample.task == "sa":
        feats[0], feats[1] = v_src[i].float(), v_src[j].float()
        present[:2] = True
    else:
        raise ValueError(f"unknown social task {sample.task!r}")
    return GenGraphEvidence(sample.task, feats, present)
