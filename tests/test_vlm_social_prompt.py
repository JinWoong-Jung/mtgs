from pathlib import Path

import pytest
from PIL import Image

from vlm.social.evidence import SLOT_NAMES
from vlm.social.prompt import (
    SOCIAL_EVIDENCE_TOKENS,
    SOCIAL_INSTRUCTION_TEMPLATE,
    SOCIAL_SPECIAL_TOKENS,
    SOCIAL_RELATION_TOKEN,
    READOUT_QUESTIONS,
    TASK_DEFINITIONS,
    UNMARKED_SOCIAL_IDENTITY,
    add_social_special_tokens,
    special_token_ids,
    social_readout_prompt,
    task_instruction,
    task_prompt,
    validate_prompt,
    validate_tokenized_prompt,
)


class _FakeTokenizer:
    def __init__(self):
        self.unk_token_id = 0
        self.additional_special_tokens = ["<existing>"]
        self.vocab = {"<unk>": 0, "<existing>": 1}

    def add_special_tokens(self, spec, replace_additional_special_tokens=True):
        added = 0
        for token in spec["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        if replace_additional_special_tokens:
            self.additional_special_tokens = list(spec["additional_special_tokens"])
        else:
            for token in spec["additional_special_tokens"]:
                if token not in self.additional_special_tokens:
                    self.additional_special_tokens.append(token)
        return added

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        specials = sorted(self.vocab, key=len, reverse=True)
        ids = []
        index = 0
        while index < len(text):
            token = next((item for item in specials if text.startswith(item, index)), None)
            if token is not None:
                ids.append(self.vocab[token])
                index += len(token)
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return ids


def test_task_conditioning_changes_only_one_plain_text_field():
    assert tuple(token[1:-1] for token in SOCIAL_EVIDENCE_TOKENS) == SLOT_NAMES
    normalized = []
    for task, definition in TASK_DEFINITIONS.items():
        instruction = task_instruction(task)
        prompt = task_prompt(task)
        assert instruction == SOCIAL_INSTRUCTION_TEMPLATE.format(
            task_definition=definition,
            social_identity=UNMARKED_SOCIAL_IDENTITY,
        )
        assert f"Task definition: {definition}" in prompt
        for token in SOCIAL_SPECIAL_TOKENS:
            assert prompt.count(token) == 1
        readout = social_readout_prompt(task)
        assert readout == (
            f"Question: {READOUT_QUESTIONS[task]} Answer (yes or no) : "
            f"{SOCIAL_RELATION_TOKEN}"
        )
        assert prompt.endswith(readout)
        assert prompt.endswith(SOCIAL_RELATION_TOKEN)
        assert "yes or no" in prompt.lower()      # readout now elicits a yes/no answer
        validate_prompt(task)
        normalized.append(
            prompt.replace(definition, "{TASK_DEFINITION}").replace(readout, "{READOUT}")
        )
    assert len(set(normalized)) == 1


def test_plain_prompt_is_explicit_and_keeps_the_fixed_slot_schema():
    prompt = task_prompt("lah")
    assert prompt.count(UNMARKED_SOCIAL_IDENTITY) == 1
    for token in SOCIAL_SPECIAL_TOKENS:
        assert prompt.count(token) == 1
    validate_prompt("lah", prompt)


def test_special_registration_preserves_existing_tokens_and_is_idempotent():
    from vlm.social.prompt import GRAPH_TOKENS
    total = len(SOCIAL_SPECIAL_TOKENS) + len(GRAPH_TOKENS)   # yes/no slots + generative gtoks
    tokenizer = _FakeTokenizer()
    assert add_social_special_tokens(tokenizer) == total
    assert tokenizer.additional_special_tokens[0] == "<existing>"
    assert set(SOCIAL_SPECIAL_TOKENS).issubset(tokenizer.additional_special_tokens)
    assert set(GRAPH_TOKENS).issubset(tokenizer.additional_special_tokens)
    assert add_social_special_tokens(tokenizer) == 0
    ids = special_token_ids(tokenizer)
    assert len(ids) == total
    validate_tokenized_prompt(tokenizer, task_prompt("lah"))


def test_prompt_validator_rejects_missing_or_duplicate_placeholders():
    prompt = task_prompt("lah")
    with pytest.raises(ValueError, match="once"):
        validate_prompt("lah", prompt.replace("<person_a>", ""))
    with pytest.raises(ValueError, match="once"):
        validate_prompt("lah", prompt + " <person_a>")
    with pytest.raises(ValueError, match="unknown social task"):
        task_prompt("other")


def test_local_qwen_chat_template_preserves_all_social_tokens():
    transformers = pytest.importorskip("transformers")
    model_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    snapshots = sorted((model_root / "snapshots").glob("*"))
    if not snapshots:
        pytest.skip("local Qwen3-VL processor cache is unavailable")

    processor = transformers.AutoProcessor.from_pretrained(
        snapshots[-1], local_files_only=True
    )
    tokenizer = processor.tokenizer
    add_social_special_tokens(tokenizer)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (32, 32), "black")},
                {"type": "text", "text": task_instruction("laeo")},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": social_readout_prompt("laeo")}],
        },
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    validate_tokenized_prompt(tokenizer, rendered)
    positions = [rendered.index(token) for token in SOCIAL_SPECIAL_TOKENS]
    assert positions == sorted(positions)
    assert rendered.endswith(SOCIAL_RELATION_TOKEN)
    encoded = tokenizer.encode(rendered, add_special_tokens=False)
    assert encoded[-1] == special_token_ids(tokenizer)[SOCIAL_RELATION_TOKEN]


# ── Text prompt composition + yes/no answer (Task 2) ────────────────────────────
import random
from vlm.social.evidence import TextGraphEvidence, PersonGazeText
from vlm.social.prompt import (
    compose_text_prompt,
    generative_answer_yesno,
    parse_yesno_probability,
    validate_text_prompt,
)

BOX_A = [0.12, 0.18, 0.26, 0.42]
BOX_B = [0.58, 0.21, 0.73, 0.46]


def test_text_prompt_no_longer_claims_the_graph_is_unconfident():
    ev = TextGraphEvidence(task="lah", p_ab=0.95)   # a HIGH-confidence estimate on purpose
    text = compose_text_prompt("lah", [0.1, 0.1, 0.2, 0.2], [0.5, 0.1, 0.6, 0.2], ev,
                               rng=random.Random(0))
    assert "not confident" not in text.lower()
    assert "uncertain" not in text.lower()
    # still frames the graph's number as evidence the VLM must weigh, not blindly trust
    assert "0.95" in text
    assert "final" in text.lower() or "determine" in text.lower() or "your own" in text.lower()


def test_text_prompt_lah_has_ab_labels_prob_and_correction_framing():
    ev = TextGraphEvidence(task="lah", p_ab=0.82)
    text = compose_text_prompt("lah", BOX_A, BOX_B, ev, rng=random.Random(0))
    assert "Person A" in text and "Person B" in text
    assert "0.82" in text                       # rendered probability
    assert "0.12" in text and "0.58" in text    # both bboxes present
    assert "weigh" in text.lower() or "evidence" in text.lower()  # frames graph output as evidence to weigh
    assert "yes" in text.lower() and "no" in text.lower()   # output instruction
    validate_text_prompt("lah", BOX_A, BOX_B, ev)


def test_text_prompt_laeo_shows_both_directions():
    ev = TextGraphEvidence(task="laeo", p_ab=0.82, p_ba=0.61, task_prob=0.74)
    text = compose_text_prompt("laeo", BOX_A, BOX_B, ev, rng=random.Random(0))
    assert "0.82" in text and "0.61" in text and "0.74" in text
    assert "direct LAEO decoder" in text
    assert text.endswith(
        'Final question: Are Person A and Person B looking at one another?\n'
        'Answer with a single word, "yes" or "no".'
    )


def test_text_prompt_sa_includes_third_person_bbox_and_nonperson():
    ev = TextGraphEvidence(
        task="sa",
        task_prob=0.66,
        person_a=PersonGazeText((0.123456, 0.1, 0.7, 0.3), 0.35, 0.58, 2),
        person_b=PersonGazeText((0.05, 0.3, 0.16, 0.4), 0.20, 0.71, 3),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, ev, rng=random.Random(0))
    assert all(value in text for value in ("0.66", "0.35", "0.58", "0.20", "0.71"))
    assert "different highest-scoring person targets" in text
    # Verify that third-person bbox coords are rounded to 2 decimals via _coords
    assert "0.12" in text                         # rounded form should be present
    assert "0.123456" not in text                 # unrounded form must NOT be present


def test_text_prompt_sa_states_when_graph_targets_the_same_person():
    ev = TextGraphEvidence(
        task="sa",
        task_prob=0.77,
        person_a=PersonGazeText((0.2, 0.3, 0.4, 0.5), 0.72, 0.10, 2),
        person_b=PersonGazeText((0.2, 0.3, 0.4, 0.5), 0.69, 0.12, 2),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, ev, rng=random.Random(1))
    assert "same other person" in text
    assert "direct SA decoder" in text


def test_text_prompt_augmentation_changes_surface_not_semantic_contract():
    ev = TextGraphEvidence(task="lah", p_ab=0.82)
    prompts = {
        compose_text_prompt("lah", BOX_A, BOX_B, ev, rng=random.Random(seed))
        for seed in range(12)
    }
    assert len(prompts) > 1
    for text in prompts:
        assert "Person A -> Person B" in text
        assert str(BOX_A) in text and str(BOX_B) in text
        assert text.endswith(
            'Final question: Is Person A looking at Person B?\n'
            'Answer with a single word, "yes" or "no".'
        )


def test_text_prompt_sa_without_third_person_omits_that_clause():
    ev = TextGraphEvidence(
        task="sa",
        person_a=PersonGazeText(None, None, 0.58),
        person_b=PersonGazeText(None, None, 0.71),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, ev, rng=random.Random(0))
    assert "0.58" in text and "0.71" in text
    validate_text_prompt("sa", BOX_A, BOX_B, ev)


def test_yesno_answer_and_parser():
    assert generative_answer_yesno(1) == "yes"
    assert generative_answer_yesno(0) == "no"
    assert parse_yesno_probability("yes") == 1.0
    assert parse_yesno_probability("no") == 0.0
    assert parse_yesno_probability("maybe", default=0.5) == 0.5


# ── Graph-evidence ablation (include_graph_evidence=False) ──────────────────────
def test_no_graph_prompt_omits_graph_entirely():
    ev = TextGraphEvidence(task="lah", p_ab=0.82)
    text = compose_text_prompt(
        "lah", BOX_A, BOX_B, ev, rng=random.Random(0), include_graph_evidence=False
    )
    assert "graph" not in text.lower()
    assert "0.82" not in text                    # the graph's probability must not leak in
    validate_text_prompt(
        "lah", BOX_A, BOX_B, include_graph_evidence=False
    )


def test_no_graph_prompt_evidence_is_optional():
    # include_graph_evidence=False must not require an evidence object at all.
    text = compose_text_prompt(
        "sa", BOX_A, BOX_B, rng=random.Random(0), include_graph_evidence=False
    )
    assert "graph" not in text.lower()


def test_no_graph_prompt_keeps_identity_direction_and_output_contract():
    text = compose_text_prompt(
        "laeo", BOX_A, BOX_B, rng=random.Random(0), include_graph_evidence=False
    )
    assert "Person A" in text and "Person B" in text
    assert str(BOX_A) in text and str(BOX_B) in text
    assert text.endswith(
        'Final question: Are Person A and Person B looking at one another?\n'
        'Answer with a single word, "yes" or "no".'
    )


def test_with_graph_prompt_still_requires_evidence():
    with pytest.raises(ValueError):
        compose_text_prompt("lah", BOX_A, BOX_B, rng=random.Random(0))


def test_ablation_flag_is_the_only_difference_between_variants():
    ev = TextGraphEvidence(task="lah", p_ab=0.82)
    with_graph = compose_text_prompt("lah", BOX_A, BOX_B, ev, rng=random.Random(7))
    no_graph = compose_text_prompt(
        "lah", BOX_A, BOX_B, ev, rng=random.Random(7), include_graph_evidence=False
    )
    # Person location lines and the final question/output block are drawn from RNG calls
    # that occur before/after the graph-only block, so they must be byte-identical given
    # the same seed; only the graph-evidence interior differs.
    assert with_graph.splitlines()[2:5] == no_graph.splitlines()[2:5]  # People + A/B lines
    assert with_graph.splitlines()[-2:] == no_graph.splitlines()[-2:]  # final Q + output
    assert with_graph != no_graph
