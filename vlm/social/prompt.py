"""Fixed-schema prompt and special-token contract for pair-wise social VLM training.

LAH, LAEO and SA share one schema and one readout. Only a short natural-language task
definition changes, allowing Qwen's pretrained language semantics to condition how the
same six evidence slots are interpreted.
"""

from __future__ import annotations

import random
import re
from typing import Any

from vlm.social.data import SOCIAL_TASKS
from vlm.social.evidence import SLOT_NAMES


SOCIAL_EVIDENCE_TOKENS = tuple(f"<{name}>" for name in SLOT_NAMES)
(
    PERSON_A_TOKEN,
    PERSON_B_TOKEN,
    RELATION_AB_TOKEN,
    RELATION_BA_TOKEN,
    HEATMAP_A_TOKEN,
    HEATMAP_B_TOKEN,
) = SOCIAL_EVIDENCE_TOKENS
SOCIAL_RELATION_TOKEN = "<social_relation>"
SOCIAL_SPECIAL_TOKENS = SOCIAL_EVIDENCE_TOKENS + (SOCIAL_RELATION_TOKEN,)

UNMARKED_SOCIAL_IDENTITY = (
    "The image is unmodified; Person A and Person B are identified by their supplied "
    "latent evidence."
)


TASK_DEFINITIONS = {
    "lah": "Determine whether Person A is looking at Person B.",
    "laeo": "Determine whether Person A and Person B are looking at each other.",
    "sa": "Determine whether Person A and Person B are looking at the same target.",
}

SOCIAL_INSTRUCTION_TEMPLATE = "\n".join((
    "Infer the social relation between the two specified people using the image and the "
    "supplied latent evidence.",
    "Task definition: {task_definition}",
    "{social_identity}",
    f"Person A evidence: {PERSON_A_TOKEN}",
    f"Person B evidence: {PERSON_B_TOKEN}",
    f"Relation from Person A to Person B: {RELATION_AB_TOKEN}",
    f"Relation from Person B to Person A: {RELATION_BA_TOKEN}",
    f"Person A gaze heatmap: {HEATMAP_A_TOKEN}",
    f"Person B gaze heatmap: {HEATMAP_B_TOKEN}",
))
# Per-task yes/no question so the answer slot naturally elicits the frozen LM head's
# " yes"/" no" logits (read at <social_relation>).
READOUT_QUESTIONS = {
    "lah": "Does Person A look at Person B?",
    "laeo": "Do Person A and Person B look at each other?",
    "sa": "Do Person A and Person B look at the same target?",
}
SOCIAL_READOUT_TEMPLATE = "Question: {question} Answer (yes or no) : {token}"


def social_readout_prompt(task: str) -> str:
    """Assistant-side yes/no readout ending in the ``<social_relation>`` token."""
    try:
        question = READOUT_QUESTIONS[task]
    except KeyError as exc:
        raise ValueError(f"unknown social task {task!r}") from exc
    return SOCIAL_READOUT_TEMPLATE.format(question=question, token=SOCIAL_RELATION_TOKEN)


# ── EyeVLM-style GENERATIVE prompt (compositional reproduction bank + graph tokens) ────
#   question template x person-location expression x person-word are sampled INDEPENDENTLY
#   and combined (EyeVLM). Our contribution: task-specific graph soft-tokens (v_src/v_tgt/
#   edge only; heatmap and null_in deferred). The model is SFT'd to generate [{"label": 1/0}].
GEN_ROLE = "You are a vision assistant specializing in human gaze analysis."

# Up-to-4 graph soft-token slots; how many each task uses (see GRAPH_BLOCK / graph assembly):
#   LAH  : v_src[i], v_tgt[j], E[i->j]                    (3)
#   LAEO : v_src[i], v_src[j], E[i->j], E[j->i]           (4)
#   SA   : v_src[i], v_src[j]                             (2)
GRAPH_TOKENS = tuple(f"<gtok{i}>" for i in range(4))
GRAPH_TOKEN_COUNT = {"lah": 3, "laeo": 4, "sa": 2}

LAEO_QUESTIONS = [
    "Are <TwoPersons> looking at one another in the image?",
    "Is eye contact occurring between <TwoPersons>?",
    "The image contains <TwoPersons>. Do they seem to be making eye contact?",
    "<TwoPersons> are visible in the image. Are their gazes oriented toward one another?",
    "Are <TwoPersons> mutually looking at one another?",
    "Estimate the probability that <TwoPersons> are engaged in eye contact.",
    "What is the likelihood that <OnePerson_A> and <OnePerson_B> are looking at each other?",
    "Is there eye contact between <OnePerson_A> and <OnePerson_B>?",
    "Do <TwoPersons> appear to be looking at each other?",
    "Would you judge that <TwoPersons> are making eye contact?",
]
LAH_QUESTIONS = [
    "Is <SourcePerson> looking at <TargetPerson>?",
    "The image contains <TwoPersons>. Does <SourcePerson> appear to be looking at <TargetPerson>?",
    "<TwoPersons> are visible in the image. Is <SourcePerson> looking at <TargetPerson>?",
    "Estimate the probability that <SourcePerson> is looking at <TargetPerson>.",
    "What is the likelihood that <SourcePerson> is looking at <TargetPerson>?",
    "Does <SourcePerson> seem to be looking toward <TargetPerson>?",
    "How likely is it that <SourcePerson> is looking at <TargetPerson>?",
    "Provide the probability that <SourcePerson> is looking at <TargetPerson>.",
    "There are <TwoPersons> visible in the image. Does <SourcePerson> seem to be looking at <TargetPerson>?",
    "Would you say that <SourcePerson> is looking at <TargetPerson>?",
]
SA_QUESTIONS = [
    "Are <TwoPersons> sharing attention in the image?",
    "Is there shared attention between <TwoPersons>?",
    "The image contains <TwoPersons>. Do they appear to be sharing attention?",
    "<TwoPersons> are visible in the image. Are their gazes directed toward a common target?",
    "Are <TwoPersons> jointly attending to the same target?",
    "Estimate the probability that <TwoPersons> are sharing attention.",
    "What is the likelihood that <OnePerson_A> and <OnePerson_B> share attention?",
    "Is shared attention present between <OnePerson_A> and <OnePerson_B>?",
    "Do <TwoPersons> appear to be sharing attention?",
    "Would you say that <TwoPersons> are sharing attention?",
]
QUESTION_BANK = {"lah": LAH_QUESTIONS, "laeo": LAEO_QUESTIONS, "sa": SA_QUESTIONS}

ONE_PERSON_TEMPLATES = [
    "the <person> whose head is inside the bounding box [{xmin}, {ymin}, {xmax}, {ymax}]",
    "the <person> located at [{xmin}, {ymin}, {xmax}, {ymax}]",
    "the <person> whose head is enclosed by the bounding box [{xmin}, {ymin}, {xmax}, {ymax}]",
    "the <person> whose head is bounded by [{xmin}, {ymin}, {xmax}, {ymax}]",
    "the <person> whose head is identified in the region [{xmin}, {ymin}, {xmax}, {ymax}]",
]
TWO_PERSON_TEMPLATES = [
    "the <persons> whose heads are inside the bounding boxes "
    "[{xmin1}, {ymin1}, {xmax1}, {ymax1}] and [{xmin2}, {ymin2}, {xmax2}, {ymax2}]",
    "the <persons> located at "
    "[{xmin1}, {ymin1}, {xmax1}, {ymax1}] and [{xmin2}, {ymin2}, {xmax2}, {ymax2}]",
    "the <persons> whose heads are enclosed by the bounding boxes "
    "[{xmin1}, {ymin1}, {xmax1}, {ymax1}] and [{xmin2}, {ymin2}, {xmax2}, {ymax2}]",
    "the <persons> whose heads are bounded by "
    "[{xmin1}, {ymin1}, {xmax1}, {ymax1}] and [{xmin2}, {ymin2}, {xmax2}, {ymax2}]",
    "the <persons> whose heads are identified in the regions "
    "[{xmin1}, {ymin1}, {xmax1}, {ymax1}] and [{xmin2}, {ymin2}, {xmax2}, {ymax2}]",
]
PERSON_WORDS = ["person", "subject", "individual", "human"]
PERSONS_WORDS = ["people", "subjects", "individuals", "humans"]

_GRAPH_ROLE_LABEL = {
    "lah": (
        "Person A's outgoing gaze",
        "Person B as a potential gaze target",
        "the directed relation from Person A to Person B",
    ),
    "laeo": (
        "Person A's outgoing gaze",
        "Person B's outgoing gaze",
        "the directed relation from Person A to Person B",
        "the directed relation from Person B to Person A",
    ),
    "sa": ("Person A's outgoing gaze", "Person B's outgoing gaze"),
}

GRAPH_EVIDENCE_INTRO = (
    "To complement your visual inspection, additional evidence from a pretrained "
    "social-gaze graph is provided below. Each learned token summarizes the indicated "
    "gaze-related evidence. Use it as supplementary context together with the image "
    "and the head bounding boxes."
)

FINAL_PROBABILITY_QUESTION = (
    "Based on the image, the head bounding boxes, and the supplementary graph evidence, "
    "what is the probability that the described social-gaze relation holds?"
)


def _graph_block(task: str) -> str:
    labels = _GRAPH_ROLE_LABEL[task]
    parts = [f"Evidence about {labels[k]}: {GRAPH_TOKENS[k]}" for k in range(len(labels))]
    return "\n".join((GRAPH_EVIDENCE_INTRO, *parts))


GEN_OUTPUT_INSTRUCTION = (
    'Respond ONLY with JSON of the form [{"label": y}], where y is exactly 1 if '
    "the described social-gaze relation holds and exactly 0 otherwise."
)

_LABEL_RE = re.compile(r'"label"\s*:\s*([0-9]*\.?[0-9]+)')


def _coords(box):
    return [round(float(v), 2) for v in box]


def _one_person_expr(box, rng) -> str:
    tmpl = rng.choice(ONE_PERSON_TEMPLATES).replace("<person>", rng.choice(PERSON_WORDS))
    x = _coords(box)
    return tmpl.format(xmin=x[0], ymin=x[1], xmax=x[2], ymax=x[3])


def _two_person_expr(box_a, box_b, rng) -> str:
    tmpl = rng.choice(TWO_PERSON_TEMPLATES).replace("<persons>", rng.choice(PERSONS_WORDS))
    a, b = _coords(box_a), _coords(box_b)
    return tmpl.format(
        xmin1=a[0], ymin1=a[1], xmax1=a[2], ymax1=a[3],
        xmin2=b[0], ymin2=b[1], xmax2=b[2], ymax2=b[3],
    )


def compose_generative_prompt(task: str, box_a, box_b, rng=None) -> str:
    """Sample a question template, person-location expressions and person words independently
    and combine, then append the task's graph-token block + the JSON output instruction.
    For LAH the first box is the source (looker), the second the target — order preserved."""
    if task not in QUESTION_BANK:
        raise ValueError(f"unknown social task {task!r}")
    r = rng if rng is not None else random
    question = r.choice(QUESTION_BANK[task])
    substitutions = (
        ("<TwoPersons>", lambda: _two_person_expr(box_a, box_b, r)),
        ("<SourcePerson>", lambda: _one_person_expr(box_a, r)),
        ("<OnePerson_A>", lambda: _one_person_expr(box_a, r)),
        ("<TargetPerson>", lambda: _one_person_expr(box_b, r)),
        ("<OnePerson_B>", lambda: _one_person_expr(box_b, r)),
    )
    for placeholder, make in substitutions:
        while placeholder in question:                 # independent sample per occurrence
            question = question.replace(placeholder, make(), 1)
    return "\n".join([
        GEN_ROLE,
        question,
        _graph_block(task),
        FINAL_PROBABILITY_QUESTION,
        GEN_OUTPUT_INSTRUCTION,
    ])


def generative_answer_json(label: int) -> str:
    """Paper-format binary target: ``[{\"label\": 1}]`` or ``[{\"label\": 0}]``."""
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    return '[{"label": %d}]' % label


def parse_label_probability(text: str, default: float = 0.5) -> float:
    """Parse a binary ``label`` value from generated JSON; clamp defensively to [0,1]."""
    match = _LABEL_RE.search(text or "")
    if match is None:
        return float(default)
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return float(default)


def validate_generative_prompt(task: str, box_a, box_b) -> None:
    """The composed prompt carries exactly the task's graph tokens once and no leftover
    <...> placeholder / <social_relation> token."""
    text = compose_generative_prompt(task, box_a, box_b, rng=random.Random(0))
    used = GRAPH_TOKEN_COUNT[task]
    for k, token in enumerate(GRAPH_TOKENS):
        expected = 1 if k < used else 0
        if text.count(token) != expected:
            raise ValueError(f"graph token {token!r} count {text.count(token)} != {expected}")
    if SOCIAL_RELATION_TOKEN in text or "<Two" in text or "<One" in text or "<Source" in text:
        raise ValueError("composed prompt still contains an unresolved placeholder")


def task_instruction(task: str) -> str:
    """User-side instruction for a plain image and fixed evidence-slot schema."""
    try:
        task_definition = TASK_DEFINITIONS[task]
    except KeyError as exc:
        raise ValueError(f"unknown social task {task!r}") from exc
    return SOCIAL_INSTRUCTION_TEMPLATE.format(
        task_definition=task_definition,
        social_identity=UNMARKED_SOCIAL_IDENTITY,
    )


def task_prompt(task: str) -> str:
    """Human-readable whole prompt; chat collation puts its final line in assistant."""
    return task_instruction(task) + "\n" + social_readout_prompt(task)


def validate_prompt(task: str, prompt: str | None = None) -> None:
    """Fail if task text or a placeholder is duplicated, missing, or reordered."""
    prompt = task_prompt(task) if prompt is None else prompt
    for token in SOCIAL_SPECIAL_TOKENS:
        count = prompt.count(token)
        if count != 1:
            raise ValueError(f"pair prompt must contain {token!r} once, got {count}")
    positions = [prompt.index(token) for token in SOCIAL_SPECIAL_TOKENS]
    if positions != sorted(positions):
        raise ValueError("pair special tokens do not follow the fixed six-slot/readout order")
    expected_definition = f"Task definition: {TASK_DEFINITIONS[task]}"
    if prompt.count(expected_definition) != 1:
        raise ValueError(
            f"pair prompt must contain task definition {expected_definition!r} once"
        )
    expected_readout = social_readout_prompt(task)
    if not prompt.endswith(expected_readout):
        raise ValueError(f"pair prompt must end with {expected_readout!r}")
    if prompt.count(UNMARKED_SOCIAL_IDENTITY) != 1:
        raise ValueError("pair prompt must contain the plain-image identity text once")


def add_social_special_tokens(tokenizer: Any) -> int:
    """Register all seven tokens without discarding a tokenizer's existing specials.

    Model embedding resize is intentionally not performed here; Unit 4 owns the model
    wrapper and must call ``resize_token_embeddings(len(tokenizer))`` exactly once.
    """
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    wanted = list(SOCIAL_SPECIAL_TOKENS) + list(GRAPH_TOKENS)   # yes/no slots + generative gtoks
    merged = existing + [token for token in wanted if token not in existing]
    try:
        return int(tokenizer.add_special_tokens(
            {"additional_special_tokens": merged},
            replace_additional_special_tokens=False,
        ))
    except TypeError:
        # Compatibility with tokenizers lacking replace_additional_special_tokens.
        return int(tokenizer.add_special_tokens({"additional_special_tokens": merged}))


def special_token_ids(tokenizer: Any) -> dict[str, int]:
    """Return distinct single-token ids, rejecting split or unknown placeholders."""
    ids: dict[str, int] = {}
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    tokens = list(SOCIAL_SPECIAL_TOKENS) + list(GRAPH_TOKENS)
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"special token {token!r} was split into ids {encoded}")
        token_id = int(encoded[0])
        if unknown_id is not None and token_id == unknown_id:
            raise ValueError(f"special token {token!r} maps to unk_token_id={unknown_id}")
        ids[token] = token_id
    if len(set(ids.values())) != len(tokens):
        raise ValueError(f"pair special-token ids are not distinct: {ids}")
    return ids


def validate_tokenized_prompt(tokenizer: Any, text: str) -> None:
    """Verify every registered placeholder survives tokenization exactly once/in order."""
    token_ids = special_token_ids(tokenizer)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    positions = []
    for token in SOCIAL_SPECIAL_TOKENS:
        token_id = token_ids[token]
        count = encoded.count(token_id)
        if count != 1:
            raise ValueError(f"tokenized prompt must contain {token!r} once, got {count}")
        positions.append(encoded.index(token_id))
    if positions != sorted(positions):
        raise ValueError("tokenized pair placeholders are out of fixed order")


# ── Natural-language graph-evidence prompt (text mode) + yes/no answer ────────────────────
# Keep one stable semantic scaffold while independently sampling surface realizations of
# its role, person-location, task-question, graph-introduction and verification fields.
# Person A/B identity, LAH direction, evidence ordering, final question and answer schema
# are deliberately never augmented.
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


def _fmt_box(box) -> str:
    return str(_coords(box))


def _person_location_line(label: str, box, rng) -> str:
    return rng.choice(TEXT_PERSON_LOCATION_TEMPLATES).format(
        label=label,
        noun=rng.choice(TEXT_PERSON_NOUNS),
        box=_fmt_box(box),
    )


def _text_evidence_block(evidence) -> str:
    task = evidence.task
    if task == "lah":
        return "\n".join((
            "Auxiliary graph evidence:",
            f"- The directed graph edge Person A -> Person B has probability "
            f"{_fmt_prob(evidence.p_ab)}.",
            f"- In other words, the graph estimates P(Person A looks at Person B) = "
            f"{_fmt_prob(evidence.p_ab)}.",
        ))
    if task == "laeo":
        lines = [
            "Auxiliary graph evidence:",
            f"- P(Person A looks at Person B) = {_fmt_prob(evidence.p_ab)}.",
            f"- P(Person B looks at Person A) = {_fmt_prob(evidence.p_ba)}.",
        ]
        if evidence.task_prob is not None:
            lines.append(
                "- The graph's direct LAEO decoder estimates the probability of mutual "
                f"gaze as {_fmt_prob(evidence.task_prob)}."
            )
        return "\n".join(lines)
    if task == "sa":
        lines = ["Auxiliary graph evidence:"]
        if evidence.task_prob is not None:
            lines.append(
                "- The graph's direct SA decoder estimates the probability of shared "
                f"attention as {_fmt_prob(evidence.task_prob)}."
            )
        for name, person in (("Person A", evidence.person_a), ("Person B", evidence.person_b)):
            if person is None:
                raise ValueError(f"SA text evidence is missing {name}'s gaze summary")
            if person.third_bbox is not None:
                lines.append(
                    f"- {name}'s highest-scoring other-person target has head bounding box "
                    f"{_coords(list(person.third_bbox))}, with probability "
                    f"{_fmt_prob(person.third_prob)}. The probability that {name} instead "
                    f"looks at a non-person location is {_fmt_prob(person.nonperson_prob)}."
                )
            else:
                lines.append(
                    f"- No visible person outside the query pair is available as a target for "
                    f"{name}. The probability that {name} looks at a non-person location is "
                    f"{_fmt_prob(person.nonperson_prob)}."
                )
        a_index = evidence.person_a.third_person_index
        b_index = evidence.person_b.third_person_index
        if a_index is not None and b_index is not None:
            if a_index == b_index:
                lines.append(
                    "- The graph selects the same other person as the highest-scoring target "
                    "for both Person A and Person B."
                )
            else:
                lines.append(
                    "- The graph selects different highest-scoring person targets for Person A "
                    "and Person B."
                )
        else:
            lines.append(
                "- A same-person target comparison is unavailable because at least one person "
                "has no visible target candidate outside the query pair."
            )
        return "\n".join(lines)
    raise ValueError(f"unknown social task {task!r}")


def compose_text_prompt(
    task, box_a, box_b, evidence=None, *, rng=None, include_graph_evidence: bool = True
) -> str:
    """Render a compositional, task-stable natural-language prompt.

    ``include_graph_evidence=True`` (default) requires ``evidence`` and inserts the
    graph's probability estimate(s) as an auxiliary evidence block. Set it to ``False``
    to render the graph-evidence ablation variant: the graph-introduction line and the
    evidence block are omitted entirely, and the shared instruction / correction sentence
    are swapped for image-only wording that never mentions the graph. ``evidence`` is
    ignored (and may be left ``None``) in that case.

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
        "People under consideration:",
        _person_location_line("A", box_a, r),
        _person_location_line("B", box_b, r),
        "Task:",
        r.choice(TEXT_TASK_QUESTIONS[task]),
        TEXT_TASK_SEMANTICS[task],
    ]
    if include_graph_evidence:
        parts.append(r.choice(TEXT_GRAPH_INTRO_BANK))
        parts.append(_text_evidence_block(evidence))
        parts.append(r.choice(TEXT_CORRECTION_BANK))
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


for _task in SOCIAL_TASKS:
    validate_prompt(_task)
