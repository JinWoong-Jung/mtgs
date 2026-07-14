"""Project the fixed MTGS six-slot evidence layout into Qwen's hidden space."""

from __future__ import annotations

import torch
import torch.nn as nn

from vlm.pair_features import PairGraphBatch


def _expect_shape(tensor: torch.Tensor, shape: tuple[int | None, ...], name: str) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must be {len(shape)}D, got {tuple(tensor.shape)}")
    for axis, (actual, expected) in enumerate(zip(tensor.shape, shape)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} axis {axis} must have size {expected}, got {tuple(tensor.shape)}"
            )


class PersonSlotProjector(nn.Module):
    """Two person slots from ``[src,tgt,null_in]`` channels and their presence mask.

    Absent channels are replaced by channel-specific learned N/A graph-space vectors
    before projection. This is essential for LAH, where A has src only and B has tgt
    only. The mask is also concatenated explicitly so zero-valued real evidence cannot
    be confused with absence.
    """

    def __init__(self, graph_dim: int, output_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.graph_dim = graph_dim
        self.na_channels = nn.Parameter(torch.empty(3, graph_dim))
        nn.init.normal_(self.na_channels, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(3 * graph_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.out_norm = nn.LayerNorm(output_dim)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def resolve_channels(self, features: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        _expect_shape(features, (None, 2, 3, self.graph_dim), "person_features")
        _expect_shape(present, (features.shape[0], 2, 3), "person_channel_present")
        if present.dtype != torch.bool:
            raise ValueError("person_channel_present must be boolean")
        # Graph caches remain FP32 while the Qwen/projector path normally runs BF16.
        # Cast at the projector boundary so Linear sees the same dtype as its weights.
        features = features.to(dtype=self.na_channels.dtype)
        na = self.na_channels.view(1, 1, 3, self.graph_dim)
        return torch.where(present.unsqueeze(-1), features, na)

    def forward(self, features: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        resolved = self.resolve_channels(features, present)
        flat = resolved.flatten(start_dim=2)
        x = torch.cat((flat, present.to(flat.dtype)), dim=-1)
        return self.gain * self.out_norm(self.mlp(x))


class RelationSlotProjector(nn.Module):
    """A->B/B->A graph edges with a learned N/A edge for absent directions."""

    def __init__(self, graph_dim: int, output_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.graph_dim = graph_dim
        self.na_feature = nn.Parameter(torch.empty(graph_dim))
        nn.init.normal_(self.na_feature, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.out_norm = nn.LayerNorm(output_dim)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        _expect_shape(features, (None, 2, self.graph_dim), "relation_features")
        _expect_shape(present, (features.shape[0], 2), "relation_present")
        if present.dtype != torch.bool:
            raise ValueError("relation_present must be boolean")
        features = features.to(dtype=self.na_feature.dtype)
        resolved = torch.where(
            present.unsqueeze(-1), features, self.na_feature.view(1, 1, -1)
        )
        x = torch.cat((resolved, present.unsqueeze(-1).to(resolved.dtype)), dim=-1)
        return self.gain * self.out_norm(self.mlp(x))


class HeatmapSlotProjector(nn.Module):
    """Encode each gaze heatmap to one token; use a learned hidden-space N/A token."""

    def __init__(self, output_dim: int, conv_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, conv_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, conv_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(conv_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)
        self.na_token = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.na_token, std=0.02)
        self.na_norm = nn.LayerNorm(output_dim)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, heatmaps: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        _expect_shape(heatmaps, (None, 2, None, None), "heatmap_features")
        _expect_shape(present, (heatmaps.shape[0], 2), "heatmap_present")
        if present.dtype != torch.bool:
            raise ValueError("heatmap_present must be boolean")
        batch, slots, height, width = heatmaps.shape
        total = batch * slots
        present_flat = present.reshape(total)
        present_indices = present_flat.nonzero(as_tuple=False).flatten()

        # Start from learned N/A tokens and run the CNN only for present heatmaps.
        # This avoids encoding LAH's absent Person-B heatmap just to discard it.
        na = self.na_norm(self.na_token).view(1, -1).expand(total, -1)
        if present_indices.numel() == 0:
            resolved = na
        else:
            selected = heatmaps.reshape(total, height, width).index_select(
                0, present_indices
            )
            flat = selected.reshape(selected.shape[0], -1).float()
            spatial = torch.softmax(flat, dim=-1).reshape(
                selected.shape[0], 1, height, width
            )
            encoded = self.proj(self.net(spatial.to(self.proj.weight.dtype)))
            encoded = self.out_norm(encoded)
            resolved = na.index_copy(0, present_indices, encoded)
        return self.gain * resolved.reshape(batch, slots, -1)


class PairEvidenceProjector(nn.Module):
    """Produce tokens in ``person_a,person_b,relation_ab,relation_ba,hm_a,hm_b`` order."""

    def __init__(
        self,
        graph_dim: int,
        output_dim: int,
        graph_hidden_dim: int = 1024,
        heatmap_conv_dim: int = 128,
    ):
        super().__init__()
        self.person = PersonSlotProjector(graph_dim, output_dim, graph_hidden_dim)
        self.relation = RelationSlotProjector(graph_dim, output_dim, graph_hidden_dim)
        self.heatmap = HeatmapSlotProjector(output_dim, heatmap_conv_dim)

    def set_output_gain(self, value: float) -> None:
        with torch.no_grad():
            self.person.gain.fill_(value)
            self.relation.gain.fill_(value)
            self.heatmap.gain.fill_(value)

    def forward(self, batch: PairGraphBatch) -> torch.Tensor:
        people = self.person(batch.person_features, batch.person_channel_present)
        relations = self.relation(batch.relation_features, batch.relation_present)
        heatmaps = self.heatmap(batch.heatmap_features, batch.heatmap_present)
        return torch.cat((people, relations, heatmaps), dim=1)
