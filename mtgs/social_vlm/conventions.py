"""WP0 — LAH/LAEO/SA looker↔target index conventions (single source of truth).

ALL dataset↔graph index conversion in mtgs/social_vlm/ goes through this file. No ad-hoc
`src`/`dst`/`i`/`j` swapping elsewhere — use looker_idx / target_idx / person_a_idx /
person_b_idx and the helpers below.

Conventions (verified against the trained graph, see test_conventions.py):
  * Graph matrices are indexed [looker_idx, target_idx]:
        graph_lah[looker_idx, target_idx] = logit( looker looks at target )
    (In mtgs_net the p2p edge E[i→j] and lah readout are "i looks at j", i.e. i=looker.)
  * Dataset pair vectors follow itertools.permutations(range(N), 2) order, and a pair
    (a, b) means "b looks at a"  ==>  target_idx = a, looker_idx = b.
    (This is why the LAH GT uses pair_vector_to_matrix(..., reverse=True).)
  * A manifest record stores {"i", "j", "task"} where, for LAH, the answer means
    "j looks at i"  ==>  looker_idx = j, target_idx = i. LAEO/SA are symmetric.
  * Pairwise gaze alignment is indexed [looker_idx, target_idx], BUT the graph defines
    the direction as dir = center[looker] - center[target] (i.e. target→looker), so
        alignment[looker_idx, target_idx] = cos( gaze[looker], center[looker]-center[target] )
    which is NEGATIVE (≈ -1) when the looker's gaze points AT the target. VERIFIED
    empirically on val (test_conventions.test_alignment_convention): LAH=yes mean ≈ -0.90
    vs LAH=no mean ≈ -0.24. This matches the trained graph's cached `align` field exactly
    (max_abs_err ~2e-4) and is fed as a raw feature to a learned edge layer, so the sign
    is internally consistent even though it is the opposite of the naive expectation.
    Downstream code MUST use this sign convention (do not flip) to stay graph-consistent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── dataset (a,b) ⇄ graph [looker, target] ────────────────────────────────────

def dataset_pair_to_graph_indices(target_idx: int, looker_idx: int):
    """Dataset pair (a=target, b=looker) → graph matrix indices (looker, target)."""
    return looker_idx, target_idx


def graph_indices_to_dataset_pair(looker_idx: int, target_idx: int):
    """Graph matrix indices (looker, target) → dataset pair (a=target, b=looker)."""
    return target_idx, looker_idx


def manifest_record_to_indices(rec: dict):
    """Manifest record {"i","j","task"} → (looker_idx, target_idx) for LAH, or
    (person_a_idx, person_b_idx) for the symmetric tasks (laeo/sa; order kept).

    LAH: dataset stores (i=target, j=looker); answer = "j looks at i" → looker=j, target=i.
    """
    if rec["task"] == "lah":
        return rec["j"], rec["i"]          # looker_idx, target_idx
    return rec["i"], rec["j"]              # person_a_idx, person_b_idx (symmetric)


# ── geometry: pairwise gaze alignment (matches mtgs_net GazeGraphBlock) ────────

def pairwise_alignment(centers: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
    """alignment[looker, target] = cos( gaze[looker], normalize(center[looker]-center[target]) ).

    Reproduces the trained graph's `align` field EXACTLY (adaptor_modules.GazeGraphBlock:
    dir = centers.unsqueeze(-3) - centers.unsqueeze(-2), align = gaze·dir). Because dir is
    target→looker, the value is NEGATIVE (≈ -1) when the looker gazes AT the target — see
    the module docstring / test_alignment_convention. Do NOT flip the sign downstream.

    centers : (N, 2) head-box centers (normalised)
    gaze    : (N, 2) predicted gaze unit vectors
    Returns : (N, N); diagonal is undefined (dir is 0) and must be ignored.
    """
    dir_ij = F.normalize(centers.unsqueeze(1) - centers.unsqueeze(0), dim=-1)  # (N,N,2)
    return (gaze.unsqueeze(1) * dir_ij).sum(-1)                                # (N,N)
