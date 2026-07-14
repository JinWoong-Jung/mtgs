"""Validate exported MTGS graph features before a VLM run consumes them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _manifest_sids(path: Path) -> set[str]:
    with path.open() as handle:
        return {json.loads(line)["sid"] for line in handle if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_feats", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gtmeta", default="")
    parser.add_argument("--require_direct_laeo", action="store_true")
    args = parser.parse_args()

    graph_path = Path(args.graph_feats)
    cache = torch.load(graph_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or not cache:
        raise ValueError(f"graph cache must be a non-empty dict: {graph_path}")
    manifest_sids = _manifest_sids(Path(args.manifest))
    cache_sids = set(cache)
    missing = sorted(manifest_sids - cache_sids)
    if missing:
        raise ValueError(f"manifest sids missing from graph cache: {missing[:10]}")

    max_direct_difference = 0.0
    required = {"lah_logits", "laeo_logits", "sa_logits", "v_src", "v_tgt", "edge_pp"}
    for sid in manifest_sids:
        record = cache[sid]
        absent = sorted(required - set(record))
        if absent:
            raise ValueError(f"{sid}: graph cache record lacks {absent}")
        lah, laeo = record["lah_logits"].float(), record["laeo_logits"].float()
        if lah.shape != laeo.shape or lah.ndim != 2 or lah.shape[0] != lah.shape[1]:
            raise ValueError(f"{sid}: incompatible LAH/LAEO shapes {lah.shape}/{laeo.shape}")
        derived = torch.minimum(lah, lah.transpose(-1, -2))
        max_direct_difference = max(max_direct_difference, float((laeo - derived).abs().max()))

    if args.gtmeta:
        gtmeta = torch.load(args.gtmeta, map_location="cpu", weights_only=False)
        if isinstance(gtmeta, dict):
            gt_sids = set(gtmeta)
            absent = sorted(manifest_sids - gt_sids)
            if absent:
                raise ValueError(f"manifest sids missing from gtmeta: {absent[:10]}")
        else:
            print(f"[validate] gtmeta type={type(gtmeta).__name__}; no key-level check", flush=True)
    if args.require_direct_laeo and max_direct_difference <= 1e-6:
        raise ValueError("LAEO equals LAH-min for every manifest frame; direct decoder was not used")
    print(
        f"[validate] graph={len(cache_sids):,} manifest_frames={len(manifest_sids):,} "
        f"extra_graph={len(cache_sids - manifest_sids):,} "
        f"max_abs_laeo_vs_lah_min={max_direct_difference:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
