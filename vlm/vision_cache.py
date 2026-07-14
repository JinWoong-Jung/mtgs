"""Persistent, frame-keyed cache for frozen Qwen vision features."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FORMAT_VERSION = 1


@dataclass(frozen=True)
class DiskVisionFrame:
    grid_thw: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: tuple[torch.Tensor, ...]


class VisionDiskCache:
    """Read completed frame-keyed safetensor shards; a missing key returns ``None``."""

    def __init__(self, root: str | Path, expected_metadata: Mapping[str, str] | None = None):
        self.root = Path(root)
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"vision cache index does not exist: {index_path}")
        self.index = json.loads(index_path.read_text())
        if self.index.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported vision cache format")
        self.metadata = dict(self.index.get("metadata", {}))
        for key, value in (expected_metadata or {}).items():
            if str(self.metadata.get(key)) != str(value):
                raise ValueError(
                    f"vision cache metadata mismatch for {key!r}: "
                    f"cache={self.metadata.get(key)!r}, expected={value!r}"
                )
        self.frames = self.index.get("frames", {})

    def __len__(self) -> int:
        return len(self.frames)

    def get(self, frame_id: str) -> DiskVisionFrame | None:
        entry = self.frames.get(str(frame_id))
        if entry is None:
            return None
        path = self.root / entry["shard"]
        prefix = entry["prefix"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            grid = handle.get_tensor(f"{prefix}.grid_thw")
            pooler = handle.get_tensor(f"{prefix}.pooler_output")
            deepstack = tuple(
                handle.get_tensor(f"{prefix}.deepstack_{layer}")
                for layer in range(int(entry["deepstack_count"]))
            )
        return DiskVisionFrame(grid, pooler, deepstack)


class VisionDiskCacheWriter:
    """Append frames and atomically publish safetensor shards plus one index."""

    def __init__(
        self,
        root: str | Path,
        metadata: Mapping[str, str],
        *,
        shard_size: int = 32,
        overwrite: bool = False,
    ):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if self.index_path.exists() and not overwrite:
            raise FileExistsError(f"vision cache already exists: {self.index_path}")
        self.metadata = {str(key): str(value) for key, value in metadata.items()}
        self.shard_size = int(shard_size)
        self.frames: dict[str, dict[str, object]] = {}
        self.pending: list[tuple[str, DiskVisionFrame]] = []
        self.shard_index = 0

    def add(self, frame_id: str, frame: DiskVisionFrame) -> None:
        if not frame.deepstack_features:
            raise ValueError("vision frame has no deepstack features")
        self.pending.append((str(frame_id), frame))
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        shard_name = f"shard-{self.shard_index:05d}.safetensors"
        final = self.root / shard_name
        temporary = final.with_suffix(".safetensors.pending")
        tensors: dict[str, torch.Tensor] = {}
        for local_index, (frame_id, frame) in enumerate(self.pending):
            prefix = str(local_index)
            if frame_id in self.frames:
                raise ValueError(f"duplicate vision frame id: {frame_id}")
            tensors[f"{prefix}.grid_thw"] = frame.grid_thw.detach().cpu().contiguous()
            tensors[f"{prefix}.pooler_output"] = frame.pooler_output.detach().cpu().contiguous()
            for layer, value in enumerate(frame.deepstack_features):
                tensors[f"{prefix}.deepstack_{layer}"] = value.detach().cpu().contiguous()
            self.frames[frame_id] = {
                "shard": shard_name,
                "prefix": prefix,
                "deepstack_count": len(frame.deepstack_features),
            }
        save_file(tensors, str(temporary), metadata=self.metadata)
        os.replace(temporary, final)
        self.pending.clear()
        self.shard_index += 1

    def close(self) -> None:
        self.flush()
        payload = {
            "format_version": FORMAT_VERSION,
            "metadata": self.metadata,
            "frames": self.frames,
        }
        temporary = self.index_path.with_suffix(".json.pending")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        os.replace(temporary, self.index_path)
