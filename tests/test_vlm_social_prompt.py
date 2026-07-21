import random

import pytest

from vlm.social.evidence import PersonGazeText, TextGraphEvidence
from vlm.social.prompt import (
    compose_text_prompt,
    generative_answer_yesno,
    parse_yesno_probability,
    validate_text_prompt,
)

BOX_A = [0.12, 0.18, 0.26, 0.42]
BOX_B = [0.58, 0.21, 0.73, 0.46]
APPROXIMATION_CUE = (
    "The predicted gaze-point coordinates are approximate; inspect the surrounding visual "
    "region rather than treating an exact coordinate as decisive."
)


def test_routed_prompt_replaces_correction_with_visual_review_cue_without_new_line():
    ev = TextGraphEvidence(task="lah", p_ab=0.50, gaze_a_xy=(0.44, 0.31))
    standard = compose_text_prompt("lah", BOX_A, BOX_B, ev, rng=random.Random(0))
    routed = compose_text_prompt(
        "lah", BOX_A, BOX_B, ev, rng=random.Random(0), graph_needs_visual_review=True
    )
    cue = "The graph is not confident enough about this relation, so resolve it from the image."
    assert cue in routed and cue not in standard
    assert routed.count("\n") == standard.count("\n")
    assert "0.50" in routed and "[0.44, 0.31]" in routed


def test_text_tokens_keep_text_baseline_unchanged_when_markers_are_absent():
    evidence = TextGraphEvidence(task="lah", p_ab=0.82, gaze_a_xy=(0.44, 0.31))
    legacy = compose_text_prompt("lah", BOX_A, BOX_B, evidence, rng=random.Random(4))
    empty = compose_text_prompt(
        "lah", BOX_A, BOX_B, evidence, rng=random.Random(4), graph_token_markers={}
    )
    assert empty == legacy and "<|graph_" not in legacy


def test_text_tokens_place_inline_markers_and_keep_gaze_coordinate_cue():
    lah = TextGraphEvidence(task="lah", p_ab=0.82, gaze_a_xy=(0.44, 0.31))
    text = compose_text_prompt(
        "lah", BOX_A, BOX_B, lah, rng=random.Random(0),
        graph_token_markers={"heatmap_a": "<|graph_heatmap_a|>", "edge_ab": "<|graph_edge_ab|>"},
    )
    assert "<|graph_heatmap_a|>" in text and "directed graph edge <|graph_edge_ab|>" in text
    assert "[0.44, 0.31]" in text and APPROXIMATION_CUE in text


def test_laeo_renders_scalar_probabilities_both_gaze_points_and_no_temporal_wording():
    evidence = TextGraphEvidence(
        task="laeo", p_ab=0.82, p_ba=0.61, task_prob=0.74,
        gaze_a_xy=(0.44, 0.31), gaze_b_xy=(0.21, 0.35),
    )
    text = compose_text_prompt("laeo", BOX_A, BOX_B, evidence, rng=random.Random(0))
    assert "0.82" in text and "0.61" in text and "0.74" in text
    assert "[0.44, 0.31]" in text and "[0.21, 0.35]" in text
    assert APPROXIMATION_CUE in text
    assert "previous, current, and next context positions" not in text
    assert text.endswith(
        'Final question: Are Person A and Person B looking at one another?\n'
        'Answer with a single word, "yes" or "no".'
    )


def test_sa_renders_scalar_probability_gaze_points_and_approximation_once():
    evidence = TextGraphEvidence(
        task="sa", task_prob=0.66,
        person_a=PersonGazeText((0.123456, 0.1, 0.7, 0.3), 0.35, 0.58,
                                 gaze_xy=(0.63, 0.29), third_person_index=2),
        person_b=PersonGazeText(None, None, 0.10, gaze_xy=(0.64, 0.28)),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, evidence, rng=random.Random(0))
    assert "shared attention as 0.66" in text
    assert "most likely person target is the head at [0.12, 0.1, 0.7, 0.3]" in text
    assert "[0.63, 0.29]" in text and "[0.64, 0.28]" in text
    assert text.count(APPROXIMATION_CUE) == 1
    assert "previous, current, and next context positions" not in text
    validate_text_prompt("sa", BOX_A, BOX_B, evidence)


def test_lah_gaze_point_and_direction_share_a_single_combined_line():
    evidence = TextGraphEvidence(
        task="lah", p_ab=0.82, gaze_a_xy=(0.44, 0.31), gaze_a_dir="upper-right",
    )
    text = compose_text_prompt("lah", BOX_A, BOX_B, evidence, rng=random.Random(0))
    assert text.count(
        "Person A's gaze point is near [0.44, 0.31] (direction: upper-right)."
    ) == 1
    assert "Person A's gaze direction is upper-right." not in text  # merged, no separate line
    assert "Person B's gaze point" not in text and "Person B's gaze direction" not in text
    assert text.count(APPROXIMATION_CUE) == 1


def test_laeo_gaze_point_and_direction_share_a_single_combined_line_for_both_people():
    evidence = TextGraphEvidence(
        task="laeo", p_ab=0.82, p_ba=0.61, task_prob=0.74,
        gaze_a_xy=(0.44, 0.31), gaze_b_xy=(0.21, 0.35),
        gaze_a_dir="up", gaze_b_dir="lower-left",
    )
    text = compose_text_prompt("laeo", BOX_A, BOX_B, evidence, rng=random.Random(0))
    assert text.count("Person A's gaze point is near [0.44, 0.31] (direction: up).") == 1
    assert text.count("Person B's gaze point is near [0.21, 0.35] (direction: lower-left).") == 1
    assert text.count(APPROXIMATION_CUE) == 1


def test_sa_gaze_point_and_direction_share_a_single_combined_line_for_both_people():
    evidence = TextGraphEvidence(
        task="sa", task_prob=0.66,
        person_a=PersonGazeText((0.12, 0.1, 0.7, 0.3), 0.35, 0.58,
                                 gaze_xy=(0.63, 0.29), gaze_dir="down",
                                 third_person_index=2),
        person_b=PersonGazeText(None, None, 0.10, gaze_xy=(0.64, 0.28), gaze_dir="left"),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, evidence, rng=random.Random(0))
    assert text.count("Person A's gaze point is near [0.63, 0.29] (direction: down).") == 1
    assert text.count("Person B's gaze point is near [0.64, 0.28] (direction: left).") == 1
    assert text.count(APPROXIMATION_CUE) == 1


def test_text_prompt_augmentation_keeps_semantic_contract():
    evidence = TextGraphEvidence(task="lah", p_ab=0.82, gaze_a_xy=(0.3, 0.4))
    prompts = {compose_text_prompt("lah", BOX_A, BOX_B, evidence, rng=random.Random(seed)) for seed in range(12)}
    assert len(prompts) > 1
    for text in prompts:
        assert "Person A -> Person B" in text
        assert str(BOX_A) in text and str(BOX_B) in text
        assert "[0.3, 0.4]" in text and APPROXIMATION_CUE in text


def test_yesno_answer_and_parser():
    assert generative_answer_yesno(1) == "yes"
    assert generative_answer_yesno(0) == "no"
    assert parse_yesno_probability("yes") == 1.0
    assert parse_yesno_probability("no") == 0.0
    assert parse_yesno_probability("maybe", default=0.5) == 0.5


def test_no_graph_prompt_omits_graph_evidence():
    text = compose_text_prompt(
        "lah", BOX_A, BOX_B, rng=random.Random(0), include_graph_evidence=False
    )
    assert "graph" not in text.lower() and APPROXIMATION_CUE not in text
    validate_text_prompt("lah", BOX_A, BOX_B, include_graph_evidence=False)


def test_with_graph_prompt_still_requires_evidence():
    with pytest.raises(ValueError):
        compose_text_prompt("lah", BOX_A, BOX_B, rng=random.Random(0))
