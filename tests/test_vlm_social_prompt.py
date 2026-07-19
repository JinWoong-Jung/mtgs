import random

import pytest

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


def test_text_prompt_sa_gives_person_target_gaze_point_and_nonperson_prob():
    ev = TextGraphEvidence(
        task="sa",
        task_prob=0.66,
        person_a=PersonGazeText((0.123456, 0.1, 0.7, 0.3), 0.35, 0.58,
                                gaze_xy=(0.63, 0.29), third_person_index=2),
        person_b=PersonGazeText((0.05, 0.3, 0.16, 0.4), 0.72, 0.10,
                                gaze_xy=(0.64, 0.28), third_person_index=3),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, ev, rng=random.Random(0))
    assert "shared attention as 0.66" in text
    # Both people: person-target line + unconditional gaze-point / non-person-prob line.
    assert "most likely person target is the head at [0.12, 0.1, 0.7, 0.3]" in text
    assert "0.35" in text
    assert "gaze point is near [0.63, 0.29]" in text and "0.58" in text
    assert "gaze point is near [0.64, 0.28]" in text and "0.10" in text
    assert "0.123456" not in text                 # bbox coords rounded via _coords


def test_text_prompt_sa_without_person_target_still_gives_gaze_point():
    ev = TextGraphEvidence(
        task="sa",
        task_prob=0.5,
        person_a=PersonGazeText(None, None, 0.71, gaze_xy=(0.3, 0.3)),
        person_b=PersonGazeText(None, None, 0.60, gaze_xy=(0.7, 0.2)),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, ev, rng=random.Random(1))
    assert "No other visible person is a likely gaze target for Person A" in text
    assert "gaze point is near [0.3, 0.3]" in text and "0.71" in text
    assert "gaze point is near [0.7, 0.2]" in text and "0.60" in text
    assert "direct SA decoder" in text
    validate_text_prompt("sa", BOX_A, BOX_B, ev)


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
