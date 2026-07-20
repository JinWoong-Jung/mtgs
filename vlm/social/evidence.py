"""Task-conditioned MTGS graph evidence rendered as natural-language prompt text.

This module is the boundary between the cached MTGS graph tensors and the pair-wise
generative VLM. It reads the frozen graph's per-pair probabilities and summarizes them
as plain sentences (:class:`TextGraphEvidence`); the prompt builder writes those into the
model input. There is exactly one evidence contract: prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from vlm.social.data import SocialSample


def _check_shape(tensor: torch.Tensor, expected: tuple[int | None, ...], name: str) -> None:
    if tensor.ndim != len(expected):
        raise ValueError(f"{name} must have {len(expected)} dimensions, got {tuple(tensor.shape)}")
    for axis, (actual, wanted) in enumerate(zip(tensor.shape, expected)):
        if wanted is not None and actual != wanted:
            raise ValueError(
                f"{name} axis {axis} must have size {wanted}, got {tuple(tensor.shape)}"
            )


def _tensor(cache: Mapping[str, object], name: str, ndim: int) -> torch.Tensor:
    value = cache.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"graph cache field {name!r} is missing or is not a tensor")
    if value.ndim != ndim:
        raise ValueError(f"graph cache field {name!r} must be {ndim}D, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise ValueError(f"graph cache field {name!r} must be floating point, got {value.dtype}")
    return value.detach()


# ── Text graph evidence: natural-language probabilities (no feature injection) ───────────
@dataclass(frozen=True)
class PersonGazeText:
    """SA-only per-person summary rendered as text.

    ``third_*`` is the most-likely OTHER person this person gazes at (excluding the pair
    partner and non-visible slots); ``None`` when no such person exists.
    ``nonperson_prob`` = sigmoid(null_in_logits) is the graph's probability that this gaze
    target is within the image but is not another annotated person. ``gaze_xy`` is retained
    only for backwards-compatible construction of diagnostic objects; prompts deliberately
    do not render the brittle argmax coordinate."""
    third_bbox: tuple[float, float, float, float] | None
    third_prob: float | None
    nonperson_prob: float | None
    gaze_xy: tuple[float, float] | None = None
    # Retained internally for diagnostics.
    third_person_index: int | None = None


@dataclass(frozen=True)
class AltTargetText:
    """LAH/LAEO-only: the single highest-probability OTHER candidate for one person's
    outgoing LAH edge (excluding the pair partner and non-visible slots).

    Only constructed when it beats the queried pair probability -- i.e. only when the
    graph's actual top pick for this person is NOT the partner being asked about, so the
    comparison is informative rather than merely confirmatory. ``person_index`` is the raw
    graph-cache index, used by the prompt builder to detect when Person A's and Person B's
    alternate both name the same third person (shared label) versus two different people
    (distinct labels)."""
    bbox: tuple[float, float, float, float]
    prob: float
    person_index: int


@dataclass(frozen=True)
class TextGraphEvidence:
    """Graph predictions for one pair, ready to be written into the prompt as sentences.

    LAH : p_ab only.  LAEO: p_ab/p_ba plus the direct mutual-gaze decoder probability.
    SA  : direct shared-attention probability plus each person's most likely third-person
          target and non-person probability.

    ``task_prob`` uses the task decoder's canonical pair logit. Symmetric task directions
    are averaged in logit space before sigmoid, matching the locked graph evaluator.

    ``temporal_probs`` contains the relation probability at the previous, current, and next
    cached context positions. With the current MTGS settings these positions correspond to
    raw-frame offsets ``[-3, 0, +3]``. The middle value is checked against the exported
    center logit before the evidence is returned.

    ``gaze_a_xy`` / ``gaze_b_xy`` remain as compatibility-only fields. They are not populated
    by :func:`assemble_text_graph_evidence` and are never rendered into new prompts.

    ``alt_a`` / ``alt_b`` (LAH/LAEO only) are Person A's / Person B's best OTHER candidate,
    surfaced only when it outscores the queried pair probability (see ``AltTargetText``).
    LAH sets only ``alt_a`` (A is the sole looker being evaluated); LAEO sets both.
    """
    task: str
    p_ab: float | None = None
    p_ba: float | None = None
    task_prob: float | None = None
    temporal_probs: tuple[float, float, float] | None = None
    person_a: PersonGazeText | None = None
    person_b: PersonGazeText | None = None
    gaze_a_xy: tuple[float, float] | None = None
    gaze_b_xy: tuple[float, float] | None = None
    alt_a: AltTargetText | None = None
    alt_b: AltTargetText | None = None


def _sa_person_text(
    self_idx: int,
    partner_idx: int,
    lah_logits: torch.Tensor,
    null_in_logits: torch.Tensor,
    head_bboxes: torch.Tensor,
    vis_mask: torch.Tensor | None,
) -> PersonGazeText:
    n = lah_logits.shape[0]
    # (1) top person target: highest LAH logit to a visible third person (not self/partner).
    best_k, best_logit = None, None
    for k in range(n):
        if k in (self_idx, partner_idx):
            continue
        if vis_mask is not None and not bool(vis_mask[k]):
            continue
        logit = float(lah_logits[self_idx, k])
        if best_logit is None or logit > best_logit:
            best_logit, best_k = logit, k
    if best_k is None:
        third_bbox, third_prob = None, None
    else:
        third_bbox = tuple(round(float(v), 2) for v in head_bboxes[best_k].tolist())
        third_prob = float(torch.sigmoid(torch.tensor(best_logit)))

    # (2) P(non-person scene) = sigmoid(null_in). The predicted argmax gaze point is
    # intentionally not exposed: it behaved as a near-hard negative whenever it landed just
    # outside the queried head box, even for genuinely positive relations.
    nonperson = float(torch.sigmoid(null_in_logits[self_idx].float()))
    return PersonGazeText(
        third_bbox=third_bbox,
        third_prob=third_prob,
        nonperson_prob=nonperson,
        third_person_index=best_k,
    )


def _best_alt_target(
    self_idx: int,
    partner_idx: int,
    pair_prob: float,
    lah_logits: torch.Tensor,
    head_bboxes: torch.Tensor,
    vis_mask: torch.Tensor | None,
) -> AltTargetText | None:
    """The self person's highest-probability OTHER target, gated to only fire when it's
    informative: requires >=3 people (an "other" candidate must exist) and the alternate's
    probability strictly exceeding ``pair_prob`` (otherwise the partner already IS the
    graph's top pick and showing a weaker alternate adds no signal)."""
    n = lah_logits.shape[0]
    if n < 3:
        return None
    best_k, best_prob = None, None
    for k in range(n):
        if k in (self_idx, partner_idx):
            continue
        if vis_mask is not None and not bool(vis_mask[k]):
            continue
        p = float(torch.sigmoid(lah_logits[self_idx, k]))
        if best_prob is None or p > best_prob:
            best_prob, best_k = p, k
    if best_k is None or best_prob <= pair_prob:
        return None
    bbox = tuple(round(float(v), 2) for v in head_bboxes[best_k].tolist())
    return AltTargetText(bbox=bbox, prob=best_prob, person_index=best_k)


def _symmetric_task_probability(
    logits: torch.Tensor, person_a: int, person_b: int
) -> float:
    pair_logit = 0.5 * (logits[person_a, person_b] + logits[person_b, person_a])
    return float(torch.sigmoid(pair_logit.float()))


def _temporal_relation_probabilities(
    cache: Mapping[str, object],
    *,
    field: str,
    center_logits: torch.Tensor,
    person_a: int,
    person_b: int,
    symmetric: bool,
) -> tuple[float, float, float]:
    """Return probabilities at the cached context slots immediately around center.

    The exporter stores the complete odd-length temporal window as ``[T,N,N]``. We use
    slots ``center-1, center, center+1`` rather than the outermost context positions. With
    the current ``temporal_stride=3`` this means raw-frame offsets ``[-3, 0, +3]``.
    """
    frames = _tensor(cache, field, 3).float()
    n = center_logits.shape[0]
    _check_shape(frames, (None, n, n), field)
    if frames.shape[0] < 3 or frames.shape[0] % 2 == 0:
        raise ValueError(f"{field} must have an odd temporal length >= 3, got {frames.shape[0]}")

    center = frames.shape[0] // 2
    selected = frames[center - 1:center + 2, person_a, person_b]
    if symmetric:
        selected = 0.5 * (
            selected + frames[center - 1:center + 2, person_b, person_a]
        )
        center_logit = 0.5 * (
            center_logits[person_a, person_b] + center_logits[person_b, person_a]
        )
    else:
        center_logit = center_logits[person_a, person_b]
    if not bool(torch.isfinite(selected).all()) or not bool(torch.isfinite(center_logit)):
        raise ValueError(f"{field} contains a non-finite relation logit")

    probabilities = torch.sigmoid(selected)
    center_probability = torch.sigmoid(center_logit.float())
    if not torch.isclose(probabilities[1], center_probability, atol=2e-3, rtol=2e-3):
        raise ValueError(
            f"{field} center probability {float(probabilities[1]):.6f} does not match "
            f"the exported center probability {float(center_probability):.6f}"
        )
    return tuple(float(value) for value in probabilities)


def assemble_text_graph_evidence(
    sample: SocialSample, cache: Mapping[str, object]
) -> TextGraphEvidence:
    """Read the frozen graph's per-pair probabilities for the natural-language prompt."""
    lah = _tensor(cache, "lah_logits", 2).float()
    a, b = sample.person_i, sample.person_j
    n = lah.shape[0]
    for name, idx in (("person_i", a), ("person_j", b)):
        if not 0 <= idx < n:
            raise IndexError(f"{name}={idx} outside person range [0,{n})")

    bboxes = _tensor(cache, "head_bboxes", 2)
    if bboxes.shape[1] != 4:
        raise ValueError(f"head_bboxes must be [N,4] for {sample.sid!r}")
    vis = cache.get("vis_mask")
    vis = vis if torch.is_tensor(vis) else None

    if sample.task == "lah":
        p_ab = float(torch.sigmoid(lah[a, b]))
        return TextGraphEvidence(
            task="lah",
            p_ab=p_ab,
            temporal_probs=_temporal_relation_probabilities(
                cache, field="lah_logits_frames", center_logits=lah,
                person_a=a, person_b=b, symmetric=False,
            ),
            alt_a=_best_alt_target(a, b, p_ab, lah, bboxes, vis),
        )
    if sample.task == "laeo":
        laeo = _tensor(cache, "laeo_logits", 2).float()
        _check_shape(laeo, (n, n), "laeo_logits")
        p_ab = float(torch.sigmoid(lah[a, b]))
        p_ba = float(torch.sigmoid(lah[b, a]))
        return TextGraphEvidence(
            task="laeo",
            p_ab=p_ab,
            p_ba=p_ba,
            task_prob=_symmetric_task_probability(laeo, a, b),
            temporal_probs=_temporal_relation_probabilities(
                cache, field="laeo_logits_frames", center_logits=laeo,
                person_a=a, person_b=b, symmetric=True,
            ),
            alt_a=_best_alt_target(a, b, p_ab, lah, bboxes, vis),
            alt_b=_best_alt_target(b, a, p_ba, lah, bboxes, vis),
        )
    if sample.task == "sa":
        sa = _tensor(cache, "sa_logits", 2).float()
        _check_shape(sa, (n, n), "sa_logits")
        null_in = _tensor(cache, "null_in_logits", 1)
        return TextGraphEvidence(
            task="sa",
            task_prob=_symmetric_task_probability(sa, a, b),
            temporal_probs=_temporal_relation_probabilities(
                cache, field="sa_logits_frames", center_logits=sa,
                person_a=a, person_b=b, symmetric=True,
            ),
            person_a=_sa_person_text(a, b, lah, null_in, bboxes, vis),
            person_b=_sa_person_text(b, a, lah, null_in, bboxes, vis),
        )
    raise ValueError(f"unknown social task {sample.task!r}")
