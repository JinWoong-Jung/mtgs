"""Natural-language prompt for the pair-wise social-gaze VLM.

There is exactly one prompt contract: LAH, LAEO and SA share a stable scaffold, and the
MTGS graph evidence is written into it as plain sentences (text mode). The model answers
with a single word, ``yes`` or ``no``. Surface realizations of the role, person-location,
task-question, graph-introduction and verification fields are sampled independently for
augmentation; Person A/B identity, LAH direction, evidence ordering, the final question
and the answer schema are never augmented.
"""

from __future__ import annotations

import random
import re
from typing import Mapping


def _coords(box):
    # Head boxes come from detection/tracking and can extend slightly past the frame;
    # clip to [0,1] so they never contradict the "normalized to the image" framing below.
    return [round(min(1.0, max(0.0, float(v))), 2) for v in box]


TEXT_ROLE_BANK = (
    "You are a visual reasoning assistant specializing in social gaze.",
    "You are a vision assistant trained to analyze social gaze between people.",
    "You are an image reasoning assistant with expertise in human gaze behavior.",
    "You are a visual analyst determining gaze relationships in social scenes.",
)
TEXT_ROLE = TEXT_ROLE_BANK[0]
TEXT_SHARED_INSTRUCTION = (
    "Determine the gaze relationship between Person A and Person B using both the image "
    "and the auxiliary estimates produced by a pretrained social-gaze graph model."
)
# No-graph-evidence ablation variant: identical scaffold, minus the graph clause.
TEXT_SHARED_INSTRUCTION_NO_GRAPH = (
    "Determine the gaze relationship between Person A and Person B using the image."
)

TEXT_PERSON_NOUNS = ("person", "individual", "subject", "human")
TEXT_PERSON_LOCATION_TEMPLATES = (
    "- Person {label} is the {noun} whose head bounding box is {box}.",
    "- Person {label} identifies the {noun} with head coordinates {box}.",
    "- Person {label} is the {noun} whose head lies inside {box}.",
    "- The {noun} labeled Person {label} has the head bounding box {box}.",
    "- Person {label} refers to the {noun} occupying the head region {box}.",
)

TEXT_GRAPH_INTRO_BANK = (
    "Auxiliary graph evidence is listed below. Treat it as supporting context rather than "
    "a guaranteed answer.",
    "The pretrained graph model provides the following auxiliary estimates. These values "
    "may be useful, but they can be wrong.",
    "Use the following graph predictions as supplementary clues alongside the image.",
    "The graph model independently produced the evidence below; verify it against the "
    "visible scene before deciding.",
)
TEXT_CORRECTION_BANK = (
    "Inspect the visible head poses, eye directions, body orientations, and surrounding "
    "scene to decide whether the visual evidence supports or contradicts the graph estimate.",
    "Check the image yourself and use its visual gaze cues to verify or correct the graph's "
    "prediction.",
    "Do not copy a graph probability mechanically. Weigh it against the people and scene "
    "shown in the image, then make the final decision.",
    "Base the final judgment on the image and graph evidence together, correcting the graph "
    "whenever the visual evidence disagrees.",
)
TEXT_CORRECTION = TEXT_CORRECTION_BANK[0]
# Routing-only replacement for the ordinary graph-correction sentence. It occupies the
# same prompt slot (rather than adding a new line), so low-confidence prompts explicitly
# tell the VLM why visual reasoning is needed without growing the prompt contract.
TEXT_ROUTED_CORRECTION = (
    "The graph is not confident enough about this relation, so resolve it from the image."
)
# No-graph-evidence ablation variant of TEXT_CORRECTION_BANK: same register and length,
# minus any mention of the graph (there is nothing to correct/verify against).
TEXT_CORRECTION_BANK_NO_GRAPH = (
    "Inspect the visible head poses, eye directions, body orientations, and surrounding "
    "scene to reach your judgment.",
    "Check the image yourself and use its visual gaze cues to reach a decision.",
    "Weigh the people and gaze cues visible in the image, then make the final decision.",
    "Base the final judgment on the visible evidence in the image alone.",
)
TEXT_OUTPUT_INSTRUCTION = 'Answer with a single word, "yes" or "no".'

TEXT_TASK_QUESTIONS = {
    "lah": [
        "Does Person A look at Person B?",
        "Is Person A directing their gaze toward Person B?",
        "Is Person B the current gaze target of Person A?",
        "Does the gaze direction Person A -> Person B occur in this image?",
        "Based on the image, is Person A visually attending to Person B?",
        "Does Person A's line of sight appear to reach Person B?",
        "Would you judge that Person A is looking toward Person B?",
        "Is Person A's visual attention directed at Person B?",
        "Does Person A appear to have Person B as their gaze target?",
        "Determine whether the directed looking relation from Person A to Person B holds.",
    ],
    "laeo": [
        "Are Person A and Person B looking at each other?",
        "Do Person A and Person B make mutual eye contact?",
        "Is there mutual gaze between Person A and Person B?",
        "Are the two people directing their gazes toward one another?",
        "Does the image show a bidirectional gaze relationship between Person A and Person B?",
        "Are both Person A -> Person B and Person B -> Person A visually present?",
        "Do Person A and Person B appear to be mutually looking at one another?",
        "Would you judge that Person A and Person B are making eye contact?",
        "Are the gazes of Person A and Person B oriented toward each other?",
        "Determine whether mutual looking occurs between Person A and Person B.",
    ],
    "sa": [
        "Are Person A and Person B looking at the same target?",
        "Do Person A and Person B share a common visual target?",
        "Is shared visual attention present between Person A and Person B?",
        "Are Person A and Person B directing their gazes toward a common target?",
        "Are the two people jointly attending to the same target?",
        "Does the image show a common gaze target for Person A and Person B?",
        "Are both people visually attending to the same entity or scene region?",
        "Would you judge that Person A and Person B share attention?",
        "Do the gaze directions of Person A and Person B converge on one target?",
        "Determine whether Person A and Person B have the same gaze target.",
    ],
}

TEXT_TASK_SEMANTICS = {
    "lah": (
        "Person A is the source of the gaze and Person B is the candidate target. "
        "The direction being evaluated is strictly Person A -> Person B."
    ),
    "laeo": (
        "Mutual gaze requires visual support in both directions: Person A -> Person B "
        "and Person B -> Person A."
    ),
    "sa": (
        "The common target may be another person, an object, or a location in the scene."
    ),
}

# Repeating the task immediately before the answer instruction keeps the final readout
# anchored on the requested relation after the longer graph-evidence block.
TEXT_FINAL_QUESTIONS = {
    "lah": "Final question: Is Person A looking at Person B?",
    "laeo": "Final question: Are Person A and Person B looking at one another?",
    "sa": "Final question: Are Person A and Person B looking at the same target?",
}

_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _fmt_prob(p: float) -> str:
    return f"{float(p):.2f}"


def _fmt_temporal_probs(probabilities) -> str:
    if probabilities is None or len(probabilities) != 3:
        raise ValueError("temporal graph probabilities must contain previous/current/next")
    return "[" + ", ".join(_fmt_prob(value) for value in probabilities) + "]"


def _fmt_box(box) -> str:
    return str(_coords(box))


def _person_location_line(label: str, box, rng) -> str:
    return rng.choice(TEXT_PERSON_LOCATION_TEMPLATES).format(
        label=label,
        noun=rng.choice(TEXT_PERSON_NOUNS),
        box=_fmt_box(box),
    )


def _alt_target_line(name: str, alt) -> str:
    # Referenced by bbox only, like SA's third-person line -- no "Person C" label is
    # introduced (never formally defined in "People under consideration"). When Person A's
    # and Person B's alternates are the same third person, the identical bbox appearing in
    # both lines already signals that without naming anyone.
    return (
        f"- Among other visible people, the graph's strongest alternative target for {name} "
        f"is the head at {_coords(list(alt.bbox))}, with probability {_fmt_prob(alt.prob)}."
    )


def _text_evidence_block(
    evidence, graph_token_markers: Mapping[str, str] | None = None
) -> str:
    """Render graph evidence without exposing predicted gaze-point coordinates.

    Dense heatmap markers in ``text_tokens`` mode remain available as distribution
    features, but are no longer tied to a brittle argmax coordinate.
    """
    markers = graph_token_markers or {}
    task = evidence.task
    if task == "lah":
        heat, edge = markers.get("heatmap_a"), markers.get("edge_ab")
        if evidence.temporal_probs is not None:
            features = []
            if heat is not None:
                features.append(f"Person A's predicted gaze-distribution feature {heat}")
            if edge is not None:
                features.append(f"the directed graph edge {edge}")
            prefix = f"Using {' and '.join(features)}, the graph" if features else "The graph"
            line = (
                f"- {prefix} estimates P(Person A looks at Person B) across the previous, "
                "current, and next context positions as "
                f"{_fmt_temporal_probs(evidence.temporal_probs)}, respectively"
            )
        else:
            relation = f"The directed graph edge {edge}" if edge is not None else "The graph"
            heat_text = (
                f", together with Person A's predicted gaze-distribution feature {heat},"
                if heat is not None else ""
            )
            line = (
                f"- {relation}{heat_text} estimates P(Person A looks at Person B) = "
                f"{_fmt_prob(evidence.p_ab)}"
            )
        lines = ["Auxiliary graph evidence:", line + "."]
        if evidence.alt_a is not None:
            lines.append(_alt_target_line("Person A", evidence.alt_a))
        return "\n".join(lines)

    if task == "laeo":
        heat_a, edge_ab = markers.get("heatmap_a"), markers.get("edge_ab")
        heat_b, edge_ba = markers.get("heatmap_b"), markers.get("edge_ba")

        def _direction_line(source, target, probability, heat, edge):
            features = []
            if heat is not None:
                features.append(f"{source}'s predicted gaze-distribution feature {heat}")
            if edge is not None:
                features.append(f"the directed graph edge {edge}")
            if features:
                return (
                    f"- {' and '.join(features)} support P({source} looks at {target}) = "
                    f"{_fmt_prob(probability)}"
                )
            return f"- P({source} looks at {target}) = {_fmt_prob(probability)}"

        line_a = _direction_line("Person A", "Person B", evidence.p_ab, heat_a, edge_ab)
        line_b = _direction_line("Person B", "Person A", evidence.p_ba, heat_b, edge_ba)
        lines = ["Auxiliary graph evidence:", line_a + "."]
        if evidence.alt_a is not None:
            lines.append(_alt_target_line("Person A", evidence.alt_a))
        lines.append(line_b + ".")
        if evidence.alt_b is not None:
            lines.append(_alt_target_line("Person B", evidence.alt_b))
        if evidence.temporal_probs is not None:
            lines.append(
                "- Across the previous, current, and next context positions, the graph's "
                "direct LAEO decoder estimates the mutual-gaze probabilities as "
                f"{_fmt_temporal_probs(evidence.temporal_probs)}, respectively."
            )
        elif evidence.task_prob is not None:
            lines.append(
                "- The graph's direct LAEO decoder estimates the probability of mutual "
                f"gaze as {_fmt_prob(evidence.task_prob)}."
            )
        return "\n".join(lines)

    if task == "sa":
        lines = ["Auxiliary graph evidence:"]
        for name, person in (("Person A", evidence.person_a), ("Person B", evidence.person_b)):
            if person is None:
                raise ValueError(f"SA text evidence is missing {name}'s gaze summary")
            slot = "heatmap_a" if name == "Person A" else "heatmap_b"
            heat = markers.get(slot)
            if heat is not None:
                lines.append(
                    f"- {name}'s predicted gaze-distribution feature {heat} accompanies the "
                    f"graph's probability {_fmt_prob(person.nonperson_prob)} that {name} is "
                    "looking within the image, but not at another annotated person."
                )
            else:
                lines.append(
                    f"- The graph estimates probability {_fmt_prob(person.nonperson_prob)} that "
                    f"{name} is looking within the image, but not at another annotated person."
                )
            if person.third_bbox is not None:
                lines.append(
                    f"- {name}'s most likely person target is the head at "
                    f"{_coords(list(person.third_bbox))} with probability "
                    f"{_fmt_prob(person.third_prob)}."
                )
            else:
                lines.append(f"- No other visible person is a likely gaze target for {name}.")

        if evidence.temporal_probs is not None:
            edge_ab, edge_ba = markers.get("edge_ab"), markers.get("edge_ba")
            if edge_ab is not None or edge_ba is not None:
                edges = " ".join(edge for edge in (edge_ab, edge_ba) if edge is not None)
                lines.append(
                    f"- Using the directed pair-edge features {edges}, across the previous, "
                    "current, and next context positions the graph's direct SA decoder estimates "
                    "the shared-attention probabilities as "
                    f"{_fmt_temporal_probs(evidence.temporal_probs)}, respectively."
                )
            else:
                lines.append(
                    "- Across the previous, current, and next context positions, the graph's "
                    "direct SA decoder estimates the shared-attention probabilities as "
                    f"{_fmt_temporal_probs(evidence.temporal_probs)}, respectively."
                )
        elif evidence.task_prob is not None:
            lines.append(
                "- The graph's direct SA decoder estimates the probability of shared "
                f"attention as {_fmt_prob(evidence.task_prob)}."
            )
        return "\n".join(lines)
    raise ValueError(f"unknown social task {task!r}")


def compose_text_prompt(
    task,
    box_a,
    box_b,
    evidence=None,
    *,
    rng=None,
    include_graph_evidence: bool = True,
    graph_needs_visual_review: bool = False,
    graph_token_markers: Mapping[str, str] | None = None,
) -> str:
    """Render a compositional, task-stable natural-language prompt.

    ``include_graph_evidence=True`` (default) requires ``evidence`` and inserts the
    graph's probability estimate(s) as an auxiliary evidence block. Set it to ``False``
    to render the graph-evidence ablation variant: the graph-introduction line and the
    evidence block are omitted entirely, and the shared instruction / correction sentence
    are swapped for image-only wording that never mentions the graph. ``evidence`` is
    ignored (and may be left ``None``) in that case.

    ``graph_needs_visual_review`` is set only for pairs routed to the VLM because the
    frozen graph did not exceed the confidence threshold. It replaces the ordinary
    graph-correction sentence in the same slot; it never adds an instruction line.

    ``graph_token_markers`` is ``None`` for the legacy ``text`` mode.  In
    ``text_tokens`` mode it maps task-specific token-slot names to registered special
    placeholders; each marker stays adjacent to the graph value it represents.

    Training callers use the module RNG, producing fresh surface forms when a sample is
    revisited. Validation/test callers provide a sample-derived seeded RNG, making their
    prompt deterministic. Person identity, LAH direction, evidence ordering, the final
    question and the answer schema never change between augmented surface forms.
    """
    if task not in TEXT_TASK_QUESTIONS:
        raise ValueError(f"unknown social task {task!r}")
    if include_graph_evidence:
        if evidence is None:
            raise ValueError("evidence is required when include_graph_evidence=True")
        if evidence.task != task:
            raise ValueError(f"evidence task {evidence.task!r} != prompt task {task!r}")
    r = rng if rng is not None else random
    parts = [
        r.choice(TEXT_ROLE_BANK),
        TEXT_SHARED_INSTRUCTION if include_graph_evidence else TEXT_SHARED_INSTRUCTION_NO_GRAPH,
        "People under consideration (all coordinates are normalized to the image and "
        "clipped to the image boundary):",
        _person_location_line("A", box_a, r),
        _person_location_line("B", box_b, r),
        "Task:",
        r.choice(TEXT_TASK_QUESTIONS[task]),
        TEXT_TASK_SEMANTICS[task],
    ]
    if include_graph_evidence:
        parts.append(r.choice(TEXT_GRAPH_INTRO_BANK))
        parts.append(_text_evidence_block(evidence, graph_token_markers))
        parts.append(
            TEXT_ROUTED_CORRECTION
            if graph_needs_visual_review
            else r.choice(TEXT_CORRECTION_BANK)
        )
    else:
        parts.append(r.choice(TEXT_CORRECTION_BANK_NO_GRAPH))
    parts.append(TEXT_FINAL_QUESTIONS[task])
    parts.append(TEXT_OUTPUT_INSTRUCTION)
    return "\n".join(parts)


def generative_answer_yesno(label: int) -> str:
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    return "yes" if label == 1 else "no"


def parse_yesno_probability(text: str, default: float = 0.5) -> float:
    match = _YESNO_RE.search(text or "")
    if match is None:
        return float(default)
    return 1.0 if match.group(1).lower() == "yes" else 0.0


def validate_text_prompt(
    task, box_a, box_b, evidence=None, *, include_graph_evidence: bool = True
) -> None:
    text = compose_text_prompt(
        task, box_a, box_b, evidence, rng=random.Random(0),
        include_graph_evidence=include_graph_evidence,
    )
    if "Person A" not in text or "Person B" not in text:
        raise ValueError("text prompt must name Person A and Person B")
    for name, box in (("Person A", box_a), ("Person B", box_b)):
        if _fmt_box(box) not in text:
            raise ValueError(f"text prompt must include {name}'s head bounding box")
    if TEXT_TASK_SEMANTICS[task] not in text:
        raise ValueError("text prompt must contain the fixed task semantics")
    if not text.rstrip().endswith(TEXT_OUTPUT_INSTRUCTION):
        raise ValueError("text prompt must end with the yes/no output instruction")
    if f"{TEXT_FINAL_QUESTIONS[task]}\n{TEXT_OUTPUT_INSTRUCTION}" not in text:
        raise ValueError("text prompt must repeat the final task question before answering")
    if include_graph_evidence and "graph" not in text.lower():
        raise ValueError("graph-evidence prompt must reference the graph")
    if not include_graph_evidence and "graph" in text.lower():
        raise ValueError("no-graph-evidence prompt must not mention the graph")
