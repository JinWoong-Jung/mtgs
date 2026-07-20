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


def test_routed_prompt_replaces_correction_with_visual_review_cue_without_new_line():
    ev = TextGraphEvidence(task="lah", p_ab=0.50)
    standard = compose_text_prompt("lah", BOX_A, BOX_B, ev, rng=random.Random(0))
    routed = compose_text_prompt(
        "lah",
        BOX_A,
        BOX_B,
        ev,
        rng=random.Random(0),
        graph_needs_visual_review=True,
    )

    cue = "The graph is not confident enough about this relation, so resolve it from the image."
    assert cue in routed
    assert cue not in standard
    assert routed.count(cue) == 1
    assert routed.count("\n") == standard.count("\n")  # correction slot is replaced, not appended
    assert "0.50" in routed
    assert routed.endswith('Answer with a single word, "yes" or "no".')


def test_text_prompt_lah_has_ab_labels_prob_and_correction_framing():
    ev = TextGraphEvidence(task="lah", p_ab=0.82)
    text = compose_text_prompt("lah", BOX_A, BOX_B, ev, rng=random.Random(0))
    assert "Person A" in text and "Person B" in text
    assert "0.82" in text                       # rendered probability
    assert "0.12" in text and "0.58" in text    # both bboxes present
    assert "weigh" in text.lower() or "evidence" in text.lower()  # frames graph output as evidence to weigh
    assert "yes" in text.lower() and "no" in text.lower()   # output instruction
    validate_text_prompt("lah", BOX_A, BOX_B, ev)


def test_text_tokens_keep_text_baseline_unchanged_when_markers_are_absent():
    evidence = TextGraphEvidence(task="lah", p_ab=0.82, gaze_a_xy=(0.44, 0.31))
    legacy = compose_text_prompt("lah", BOX_A, BOX_B, evidence, rng=random.Random(4))
    empty_markers = compose_text_prompt(
        "lah", BOX_A, BOX_B, evidence, rng=random.Random(4), graph_token_markers={}
    )
    assert empty_markers == legacy
    assert "<|graph_" not in legacy


def test_text_tokens_place_inline_markers_without_gaze_point_coordinates():
    lah = TextGraphEvidence(
        task="lah", p_ab=0.82, temporal_probs=(0.71, 0.82, 0.79),
        gaze_a_xy=(0.44, 0.31),
    )
    lah_text = compose_text_prompt(
        "lah", BOX_A, BOX_B, lah, rng=random.Random(0),
        graph_token_markers={
            "heatmap_a": "<|graph_heatmap_a|>",
            "edge_ab": "<|graph_edge_ab|>",
        },
    )
    assert "<|graph_heatmap_a|>" in lah_text
    assert "directed graph edge <|graph_edge_ab|>" in lah_text
    assert "[0.71, 0.82, 0.79]" in lah_text
    assert "gaze point" not in lah_text
    assert "[0.44, 0.31]" not in lah_text

    laeo = TextGraphEvidence(
        task="laeo", p_ab=0.82, p_ba=0.61, task_prob=0.74,
        temporal_probs=(0.70, 0.74, 0.76),
        gaze_a_xy=(0.44, 0.31), gaze_b_xy=(0.21, 0.35),
    )
    laeo_text = compose_text_prompt(
        "laeo", BOX_A, BOX_B, laeo, rng=random.Random(0),
        graph_token_markers={
            "heatmap_a": "<|graph_heatmap_a|>",
            "edge_ab": "<|graph_edge_ab|>",
            "heatmap_b": "<|graph_heatmap_b|>",
            "edge_ba": "<|graph_edge_ba|>",
        },
    )
    for marker in (
        "<|graph_heatmap_a|>", "<|graph_edge_ab|>",
        "<|graph_heatmap_b|>", "<|graph_edge_ba|>",
    ):
        assert marker in laeo_text
    assert "[0.70, 0.74, 0.76]" in laeo_text
    assert "gaze point" not in laeo_text
    assert "[0.44, 0.31]" not in laeo_text and "[0.21, 0.35]" not in laeo_text

    sa = TextGraphEvidence(
        task="sa", task_prob=0.66, temporal_probs=(0.60, 0.66, 0.70),
        person_a=PersonGazeText(None, None, 0.58, gaze_xy=(0.63, 0.29)),
        person_b=PersonGazeText(None, None, 0.10, gaze_xy=(0.64, 0.28)),
    )
    sa_text = compose_text_prompt(
        "sa", BOX_A, BOX_B, sa, rng=random.Random(0),
        graph_token_markers={
            "heatmap_a": "<|graph_heatmap_a|>",
            "edge_ab": "<|graph_edge_ab|>",
            "heatmap_b": "<|graph_heatmap_b|>",
            "edge_ba": "<|graph_edge_ba|>",
        },
    )
    for marker in (
        "<|graph_heatmap_a|>", "<|graph_edge_ab|>",
        "<|graph_heatmap_b|>", "<|graph_edge_ba|>",
    ):
        assert marker in sa_text
    assert "[0.60, 0.66, 0.70]" in sa_text
    assert "gaze point" not in sa_text
    assert "[0.63, 0.29]" not in sa_text and "[0.64, 0.28]" not in sa_text
def test_text_evidence_never_renders_compatibility_gaze_coordinates():
    lah = TextGraphEvidence(task="lah", p_ab=0.82, gaze_a_xy=(0.44, 0.31))
    lah_text = compose_text_prompt("lah", BOX_A, BOX_B, lah, rng=random.Random(0))
    assert "gaze point" not in lah_text
    assert "[0.44, 0.31]" not in lah_text

    laeo = TextGraphEvidence(
        task="laeo", p_ab=0.82, p_ba=0.61,
        gaze_a_xy=(0.44, 0.31), gaze_b_xy=(0.21, 0.35),
    )
    laeo_text = compose_text_prompt("laeo", BOX_A, BOX_B, laeo, rng=random.Random(0))
    assert "gaze point" not in laeo_text
    assert "[0.44, 0.31]" not in laeo_text
    assert "[0.21, 0.35]" not in laeo_text
def test_text_prompt_laeo_shows_both_directions():
    ev = TextGraphEvidence(task="laeo", p_ab=0.82, p_ba=0.61, task_prob=0.74)
    text = compose_text_prompt("laeo", BOX_A, BOX_B, ev, rng=random.Random(0))
    assert "0.82" in text and "0.61" in text and "0.74" in text
    assert "direct LAEO decoder" in text
    assert text.endswith(
        'Final question: Are Person A and Person B looking at one another?\n'
        'Answer with a single word, "yes" or "no".'
    )


def test_temporal_probabilities_replace_duplicate_center_task_probability():
    lah = TextGraphEvidence(task="lah", p_ab=0.50, temporal_probs=(0.42, 0.50, 0.47))
    lah_text = compose_text_prompt("lah", BOX_A, BOX_B, lah, rng=random.Random(0))
    assert "[0.42, 0.50, 0.47]" in lah_text
    assert "previous, current, and next context positions" in lah_text

    laeo = TextGraphEvidence(
        task="laeo", p_ab=0.61, p_ba=0.44, task_prob=0.48,
        temporal_probs=(0.41, 0.48, 0.58),
    )
    laeo_text = compose_text_prompt("laeo", BOX_A, BOX_B, laeo, rng=random.Random(0))
    assert "[0.41, 0.48, 0.58]" in laeo_text
    assert "mutual-gaze probabilities" in laeo_text
    assert "probability of mutual gaze as 0.48" not in laeo_text

    sa = TextGraphEvidence(
        task="sa", task_prob=0.52, temporal_probs=(0.38, 0.52, 0.61),
        person_a=PersonGazeText(None, None, 0.62),
        person_b=PersonGazeText(None, None, 0.57),
    )
    sa_text = compose_text_prompt("sa", BOX_A, BOX_B, sa, rng=random.Random(0))
    assert "[0.38, 0.52, 0.61]" in sa_text
    assert "shared-attention probabilities" in sa_text
    assert "probability of shared attention as 0.52" not in sa_text


def test_text_prompt_sa_gives_person_target_and_nonperson_prob_without_gaze_point():
    evidence = TextGraphEvidence(
        task="sa",
        task_prob=0.66,
        person_a=PersonGazeText((0.123456, 0.1, 0.7, 0.3), 0.35, 0.58,
                                gaze_xy=(0.63, 0.29), third_person_index=2),
        person_b=PersonGazeText((0.05, 0.3, 0.16, 0.4), 0.72, 0.10,
                                gaze_xy=(0.64, 0.28), third_person_index=3),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, evidence, rng=random.Random(0))
    assert "shared attention as 0.66" in text
    assert "most likely person target is the head at [0.12, 0.1, 0.7, 0.3]" in text
    assert "0.35" in text and "0.58" in text and "0.10" in text
    assert "within the image, but not at another annotated person" in text
    assert "gaze point" not in text
    assert "[0.63, 0.29]" not in text and "[0.64, 0.28]" not in text
    assert "0.123456" not in text
def test_text_prompt_sa_without_person_target_still_gives_nonperson_probability():
    evidence = TextGraphEvidence(
        task="sa",
        task_prob=0.5,
        person_a=PersonGazeText(None, None, 0.71, gaze_xy=(0.3, 0.3)),
        person_b=PersonGazeText(None, None, 0.60, gaze_xy=(0.7, 0.2)),
    )
    text = compose_text_prompt("sa", BOX_A, BOX_B, evidence, rng=random.Random(1))
    assert "No other visible person is a likely gaze target for Person A" in text
    assert "0.71" in text and "0.60" in text
    assert "gaze point" not in text
    assert "[0.3, 0.3]" not in text and "[0.7, 0.2]" not in text
    assert "direct SA decoder" in text
    validate_text_prompt("sa", BOX_A, BOX_B, evidence)
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
