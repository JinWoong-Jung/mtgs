"""Fixed-schema prompt and special-token contract for pair-wise social VLM training.

LAH, LAEO and SA share one schema and one readout. Only a short natural-language task
definition changes, allowing Qwen's pretrained language semantics to condition how the
same six evidence slots are interpreted.
"""

from __future__ import annotations

import random
import re
from typing import Any

from vlm.pair_dataset import SOCIAL_TASKS
from vlm.pair_features import SLOT_NAMES


PAIR_EVIDENCE_TOKENS = tuple(f"<{name}>" for name in SLOT_NAMES)
(
    PERSON_A_TOKEN,
    PERSON_B_TOKEN,
    RELATION_AB_TOKEN,
    RELATION_BA_TOKEN,
    HEATMAP_A_TOKEN,
    HEATMAP_B_TOKEN,
) = PAIR_EVIDENCE_TOKENS
SOCIAL_RELATION_TOKEN = "<social_relation>"
PAIR_SPECIAL_TOKENS = PAIR_EVIDENCE_TOKENS + (SOCIAL_RELATION_TOKEN,)

MARKED_PAIR_IDENTITY = (
    "Person A is marked with a RED box and Person B is marked with a BLUE box."
)
UNMARKED_PAIR_IDENTITY = (
    "The image is unmodified; Person A and Person B are identified by their supplied "
    "latent evidence."
)


TASK_DEFINITIONS = {
    "lah": "Determine whether Person A is looking at Person B.",
    "laeo": "Determine whether Person A and Person B are looking at each other.",
    "sa": "Determine whether Person A and Person B are looking at the same target.",
}

PAIR_INSTRUCTION_TEMPLATE = "\n".join((
    "Infer the social relation between the two specified people using the image and the "
    "supplied latent evidence.",
    "Task definition: {task_definition}",
    "{pair_identity}",
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


def validate_generative_pair_prompt(task: str, box_a, box_b) -> None:
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


def task_conditioned_pair_instruction(task: str, *, draw_bboxes: bool = True) -> str:
    """User-side instruction: fixed schema with one plain-text task definition."""
    try:
        task_definition = TASK_DEFINITIONS[task]
    except KeyError as exc:
        raise ValueError(f"unknown social task {task!r}") from exc
    return PAIR_INSTRUCTION_TEMPLATE.format(
        task_definition=task_definition,
        pair_identity=MARKED_PAIR_IDENTITY if draw_bboxes else UNMARKED_PAIR_IDENTITY,
    )


def task_conditioned_pair_prompt(task: str, *, draw_bboxes: bool = True) -> str:
    """Human-readable whole prompt; chat collation puts its final line in assistant."""
    return (
        task_conditioned_pair_instruction(task, draw_bboxes=draw_bboxes)
        + "\n"
        + social_readout_prompt(task)
    )


def validate_pair_prompt(
    task: str, prompt: str | None = None, *, draw_bboxes: bool = True
) -> None:
    """Fail if task text or a placeholder is duplicated, missing, or reordered."""
    prompt = (
        task_conditioned_pair_prompt(task, draw_bboxes=draw_bboxes)
        if prompt is None
        else prompt
    )
    for token in PAIR_SPECIAL_TOKENS:
        count = prompt.count(token)
        if count != 1:
            raise ValueError(f"pair prompt must contain {token!r} once, got {count}")
    positions = [prompt.index(token) for token in PAIR_SPECIAL_TOKENS]
    if positions != sorted(positions):
        raise ValueError("pair special tokens do not follow the fixed six-slot/readout order")
    expected_definition = f"Task definition: {TASK_DEFINITIONS[task]}"
    if prompt.count(expected_definition) != 1:
        raise ValueError(f"pair prompt must contain task definition {expected_definition!r} once")
    expected_readout = social_readout_prompt(task)
    if not prompt.endswith(expected_readout):
        raise ValueError(f"pair prompt must end with {expected_readout!r}")
    expected_identity = MARKED_PAIR_IDENTITY if draw_bboxes else UNMARKED_PAIR_IDENTITY
    unexpected_identity = UNMARKED_PAIR_IDENTITY if draw_bboxes else MARKED_PAIR_IDENTITY
    if prompt.count(expected_identity) != 1 or unexpected_identity in prompt:
        raise ValueError("pair prompt identity text does not match draw_bboxes")


def add_pair_special_tokens(tokenizer: Any) -> int:
    """Register all seven tokens without discarding a tokenizer's existing specials.

    Model embedding resize is intentionally not performed here; Unit 4 owns the model
    wrapper and must call ``resize_token_embeddings(len(tokenizer))`` exactly once.
    """
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    wanted = list(PAIR_SPECIAL_TOKENS) + list(GRAPH_TOKENS)   # yes/no slots + generative gtoks
    merged = existing + [token for token in wanted if token not in existing]
    try:
        return int(tokenizer.add_special_tokens(
            {"additional_special_tokens": merged},
            replace_additional_special_tokens=False,
        ))
    except TypeError:
        # Compatibility with tokenizers lacking replace_additional_special_tokens.
        return int(tokenizer.add_special_tokens({"additional_special_tokens": merged}))


def pair_special_token_ids(tokenizer: Any) -> dict[str, int]:
    """Return distinct single-token ids, rejecting split or unknown placeholders."""
    ids: dict[str, int] = {}
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    tokens = list(PAIR_SPECIAL_TOKENS) + list(GRAPH_TOKENS)
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


def validate_tokenized_pair_prompt(tokenizer: Any, text: str) -> None:
    """Verify every registered placeholder survives tokenization exactly once/in order."""
    token_ids = pair_special_token_ids(tokenizer)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    positions = []
    for token in PAIR_SPECIAL_TOKENS:
        token_id = token_ids[token]
        count = encoded.count(token_id)
        if count != 1:
            raise ValueError(f"tokenized prompt must contain {token!r} once, got {count}")
        positions.append(encoded.index(token_id))
    if positions != sorted(positions):
        raise ValueError("tokenized pair placeholders are out of fixed order")


# ── Natural-language graph-evidence prompt (text mode) + yes/no answer ────────────────────
TEXT_ROLE = (
    "You are a vision assistant specializing in human gaze analysis, working alongside a "
    "pretrained social-gaze graph model. The graph already produced the estimate(s) below, "
    "but it was NOT confident — that is exactly why your visual judgment is needed here."
)
TEXT_MARKED_IDENTITY = (
    "In this image Person A is marked with a RED box and Person B is marked with a BLUE box."
)
TEXT_CORRECTION = (
    "Do not simply repeat the graph's estimate(s). Inspect the image and the head bounding "
    "boxes yourself, and decide whether the relation actually holds — confirm, correct, or "
    "override the graph's prediction as your visual evidence dictates."
)
TEXT_OUTPUT_INSTRUCTION = 'Answer with a single word, "yes" or "no".'

TEXT_TASK_QUESTIONS = {
    "lah": [
        "Is Person A, located at {a}, looking at Person B, located at {b}?",
        "Does Person A ({a}) appear to be looking at Person B ({b})?",
        "Is Person A at {a} directing their gaze toward Person B at {b}?",
    ],
    "laeo": [
        "Are Person A, located at {a}, and Person B, located at {b}, looking at one another?",
        "Are Person A ({a}) and Person B ({b}) making eye contact?",
        "Do Person A at {a} and Person B at {b} appear to look at each other?",
    ],
    "sa": [
        "Are Person A, located at {a}, and Person B, located at {b}, looking at the same target?",
        "Do Person A ({a}) and Person B ({b}) share attention on a common target?",
        "Are Person A at {a} and Person B at {b} jointly attending to the same target?",
    ],
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


def _text_evidence_block(evidence) -> str:
    task = evidence.task
    if task == "lah":
        return f"Graph's uncertain estimate: P(Person A looks at Person B) = {_fmt_prob(evidence.p_ab)}"
    if task == "laeo":
        return "\n".join((
            "Graph's uncertain estimates:",
            f"- P(Person A looks at Person B) = {_fmt_prob(evidence.p_ab)}",
            f"- P(Person B looks at Person A) = {_fmt_prob(evidence.p_ba)}",
        ))
    if task == "sa":
        lines = ["Graph's uncertain estimates:"]
        for name, person in (("Person A", evidence.person_a), ("Person B", evidence.person_b)):
            if person.third_bbox is not None:
                lines.append(
                    f"- {name} most likely gazes at the person at {_coords(list(person.third_bbox))} "
                    f"(probability {_fmt_prob(person.third_prob)}); probability of gazing at a "
                    f"non-person location instead: {_fmt_prob(person.nonperson_prob)}"
                )
            else:
                lines.append(
                    f"- {name}: no other person is a likely gaze target; probability of gazing "
                    f"at a non-person location: {_fmt_prob(person.nonperson_prob)}"
                )
        return "\n".join(lines)
    raise ValueError(f"unknown social task {task!r}")


def compose_text_prompt(task, box_a, box_b, evidence, *, draw_bboxes: bool = True, rng=None) -> str:
    """Render the graph's predictions as natural-language sentences (text evidence mode)."""
    if task not in TEXT_TASK_QUESTIONS:
        raise ValueError(f"unknown social task {task!r}")
    if evidence.task != task:
        raise ValueError(f"evidence task {evidence.task!r} != prompt task {task!r}")
    r = rng if rng is not None else random
    question = r.choice(TEXT_TASK_QUESTIONS[task]).format(a=_fmt_box(box_a), b=_fmt_box(box_b))
    parts = [TEXT_ROLE]
    if draw_bboxes:
        parts.append(TEXT_MARKED_IDENTITY)
    parts.extend([
        question,
        _text_evidence_block(evidence),
        TEXT_CORRECTION,
        TEXT_FINAL_QUESTIONS[task],
        TEXT_OUTPUT_INSTRUCTION,
    ])
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


def validate_text_pair_prompt(task, box_a, box_b, evidence, *, draw_bboxes: bool = True) -> None:
    text = compose_text_prompt(task, box_a, box_b, evidence, draw_bboxes=draw_bboxes,
                               rng=random.Random(0))
    if "Person A" not in text or "Person B" not in text:
        raise ValueError("text prompt must name Person A and Person B")
    if not text.rstrip().endswith(TEXT_OUTPUT_INSTRUCTION):
        raise ValueError("text prompt must end with the yes/no output instruction")
    if f"{TEXT_FINAL_QUESTIONS[task]}\n{TEXT_OUTPUT_INSTRUCTION}" not in text:
        raise ValueError("text prompt must repeat the final task question before answering")
    if draw_bboxes and TEXT_MARKED_IDENTITY not in text:
        raise ValueError("text prompt with draw_bboxes must include the red/blue identity line")


for _task in SOCIAL_TASKS:
    validate_pair_prompt(_task)
    validate_pair_prompt(_task, draw_bboxes=False)
