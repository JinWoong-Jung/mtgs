"""WP0 convention tests. Run: python -m mtgs.social_vlm.test_conventions

  (a) dataset↔graph LAH DIRECTION: graph_lah[looker,target] separates the manifest
      answer (AUC high); the transposed index does not (AUC ~chance) — proves the
      convention, and that direction actually matters.
  (b) synthetic gaze-alignment GEOMETRY: looker(0.25,0.5) gaze(1,0) pointing AT
      target(0.75,0.5) → alignment[looker,target] = -1 (graph's dir is target→looker,
      so "looking at" is NEGATIVE — verified, matches the graph's cached align).
  (b2) EMPIRICAL alignment sign on real val: LAH=yes align ≪ LAH=no align (both < 0).
"""

from __future__ import annotations

import json

import torch
from torchmetrics.functional import auroc

from mtgs.social_vlm.conventions import (
    manifest_record_to_indices,
    pairwise_alignment,
)

C = "/home/jinwoongjung/MTGS/data/vlm_feature"


def test_direction(max_frames=1500):
    gf = torch.load(f"{C}/vlmgraph_val.pt", weights_only=False)
    recs = [json.loads(l) for l in open(f"{C}/manifest_val.jsonl")]
    correct, wrong, y = [], [], []
    seen = set()
    for r in recs:
        if r["task"] != "lah" or r["sid"] not in gf:
            continue
        seen.add(r["sid"])
        if len(seen) > max_frames:
            break
        lah = gf[r["sid"]]["lah_logits"].float()          # (N,N) [looker,target]
        looker, target = manifest_record_to_indices(r)
        correct.append(lah[looker, target])               # convention direction
        wrong.append(lah[target, looker])                 # transposed (should be worse)
        y.append(1 if r["ans"] == "yes" else 0)
    y = torch.tensor(y)
    auc_c = auroc(torch.stack(correct), y, task="binary").item()
    auc_w = auroc(torch.stack(wrong), y, task="binary").item()
    print(f"[a] LAH direction: convention AUC={auc_c:.4f}  transposed AUC={auc_w:.4f}  "
          f"(n={len(y)})")
    assert auc_c > 0.90, f"convention direction AUC too low ({auc_c:.3f})"
    assert auc_c - auc_w > 0.20, "direction does not matter (convention unverified)"
    return auc_c, auc_w


def test_geometry():
    centers = torch.tensor([[0.25, 0.5], [0.75, 0.5]])
    gaze = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])         # P0 looks right (at P1); P1 left (at P0)
    a = pairwise_alignment(centers, gaze)
    print(f"[b] geometry: alignment[looker=0,target=1]={a[0,1].item():+.4f} "
          f"(looking-at => -1)  alignment[1,0]={a[1,0].item():+.4f}")
    # graph convention: dir is target→looker, so gaze pointing AT target gives -1.
    assert torch.isclose(a[0, 1], torch.tensor(-1.0), atol=1e-5), "looker→target 'looking at' should be -1"
    assert torch.isclose(a[1, 0], torch.tensor(-1.0), atol=1e-5), "P1 gaze(-1,0) points at P0 => -1"


def test_alignment_convention(max_frames=2000):
    gf = torch.load(f"{C}/vlmgraph_val.pt", weights_only=False)
    recs = [json.loads(l) for l in open(f"{C}/manifest_val.jsonl")]
    yes, no = [], []
    seen = set()
    for r in recs:
        if r["task"] != "lah" or r["sid"] not in gf:
            continue
        seen.add(r["sid"])
        if len(seen) > max_frames:
            break
        d = gf[r["sid"]]
        bb = d["head_bboxes"].float()
        centers = (bb[:, :2] + bb[:, 2:]) * 0.5
        al = pairwise_alignment(centers, d["gaze_vecs"].float())
        looker, target = manifest_record_to_indices(r)
        (yes if r["ans"] == "yes" else no).append(float(al[looker, target]))
    my = sum(yes) / len(yes)
    mn = sum(no) / len(no)
    print(f"[b2] empirical align[looker,target]: YES mean={my:+.3f}  NO mean={mn:+.3f} "
          f"=> looking-at is NEGATIVE")
    assert my < mn - 0.3, "LAH=yes should have far more negative alignment than LAH=no"
    assert my < 0, "looking-at alignment should be negative under the graph convention"


if __name__ == "__main__":
    test_geometry()
    test_alignment_convention()
    test_direction()
    print("\nWP0 convention tests PASSED")
