"""Filter a social-gaze manifest down to the low-confidence pairs (graph conf < threshold).

Confidence-gated routing answers high-confidence pairs with the frozen graph directly and
queries the VLM only on the pairs the graph is unsure about
(``max(p, 1-p) < threshold``). This builds the low-confidence TRAINING manifest so the VLM
specialises on exactly the pairs it will be asked at eval time -- where ``vlm.social``'s
router makes the same split live. It reuses ``graph_confidence()`` (the same function the
eval-time router uses), so the train and eval partitions are guaranteed to match.

CLI:
  python -m vlm.cache.filter_lowconf_manifest \
      --manifest .../manifest_train.jsonl --graph_feats .../vlmgraph_train.pt \
      --out .../manifest_train.jsonl --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from vlm.social.data import SocialAnnotationDataset
from vlm.social.input import graph_confidence


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--graph_feats", required=True, help="vlmgraph_<split>.pt for the same split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="keep pairs with graph confidence max(p,1-p) STRICTLY below this")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    if not 0.5 <= args.threshold <= 1.0:
        raise SystemExit(f"threshold must be in [0.5, 1.0], got {args.threshold}")

    raw = [line for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    annotations = SocialAnnotationDataset(args.manifest)
    if len(annotations.samples) != len(raw):
        raise SystemExit(
            f"manifest line/sample mismatch: {len(raw)} lines vs {len(annotations.samples)} samples"
        )
    cache = torch.load(args.graph_feats, map_location="cpu", weights_only=False)

    kept_lines: list[str] = []
    kept: Counter = Counter()
    total: Counter = Counter()
    for line, sample in zip(raw, annotations.samples):
        total[sample.task] += 1
        if sample.sid not in cache:
            raise SystemExit(f"graph cache missing sid {sample.sid!r}")
        if graph_confidence(sample, cache[sample.sid]) < args.threshold:
            kept_lines.append(line)
            kept[sample.task] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(("\n".join(kept_lines) + "\n") if kept_lines else "")

    summary = {
        "threshold": args.threshold,
        "input_pairs": len(raw),
        "kept_pairs": len(kept_lines),
        "per_task": {t: {"kept": kept[t], "total": total[t]} for t in ("lah", "laeo", "sa")},
    }
    print(f"[filter-lowconf] threshold={args.threshold} kept {len(kept_lines)}/{len(raw)} "
          f"pairs -> {out_path}", flush=True)
    for t in ("lah", "laeo", "sa"):
        print(f"    {t}: kept {kept[t]}/{total[t]}", flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2))
    if not kept_lines:
        raise SystemExit("no low-confidence pairs kept; raise --threshold")


if __name__ == "__main__":
    main()
