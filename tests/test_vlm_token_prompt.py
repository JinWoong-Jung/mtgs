# tests/test_vlm_token_prompt.py
import torch
from vlm.prompt import token_prompt
from vlm.injection import GTOK, HMTOK, TOK_COUNT, HM_COUNT, gather_feats


BB = [0.10, 0.20, 0.30, 0.40]


def test_gtok_and_hmtok_counts_match():
    for task in ("lah", "laeo", "sa"):
        s = token_prompt(task, "P1", "P2", BB, BB)
        assert s.count(GTOK) == TOK_COUNT[task], (task, "gtok", s)
        assert s.count(HMTOK) == HM_COUNT[task], (task, "hmtok", s)


def test_prompt_contains_boxes_and_question():
    bb_a = [0.11, 0.22, 0.33, 0.44]
    bb_b = [0.55, 0.66, 0.77, 0.88]
    s = token_prompt("lah", "P1", "P2", bb_a, bb_b)
    assert "[0.11,0.22,0.33,0.44]" in s
    assert "[0.55,0.66,0.77,0.88]" in s
    # the first-named person (A, from query_slots = the LOOKER) is asked about
    assert f"Is P1 {GTOK} looking at P2 {GTOK}?" in s
    assert s.strip().endswith("yes or no.")
    # A gets the RED box (matches build_token_overlay), B the BLUE box
    assert s.index("RED") < s.index("BLUE")


def test_laeo_sa_questions():
    assert "looking at each other" in token_prompt("laeo", "A", "B", BB, BB)
    assert "same thing or person" in token_prompt("sa", "A", "B", BB, BB)


def test_no_belief_or_marker_text():
    """Belief sentence and gaze-marker text were removed; gaze location is now the
    <hmtok> heatmap token, not drawn arrows or a prior probability sentence."""
    for task in ("lah", "laeo", "sa"):
        s = token_prompt(task, "P1", "P2", BB, BB)
        assert "prior" not in s.lower()
        assert "arrow" not in s.lower()


def test_heatmap_sentence_persons():
    # lah: only A's heatmap; laeo/sa: both A and B
    s = token_prompt("lah", "P1", "P2", BB, BB)
    assert f"Person P1 gaze heatmap: {HMTOK}" in s and "Person P2 gaze heatmap" not in s
    for task in ("laeo", "sa"):
        s = token_prompt(task, "P1", "P2", BB, BB)
        assert f"Person P1 gaze heatmap: {HMTOK}" in s
        assert f"Person P2 gaze heatmap: {HMTOK}" in s


def test_relation_wording_per_task():
    # LAH: single directed relation A->B
    s = token_prompt("lah", "P1", "P2", BB, BB)
    assert f"The gaze relation from P1 to P2: {GTOK}" in s
    assert "The gaze relation from P2 to P1" not in s
    # LAEO: both directions explicitly
    s = token_prompt("laeo", "P1", "P2", BB, BB)
    assert f"The gaze relation from P1 to P2: {GTOK}" in s
    assert f"The gaze relation from P2 to P1: {GTOK}" in s
    # SA: per-person scene-gaze, no pairwise 'gaze relation' edge
    s = token_prompt("sa", "P1", "P2", BB, BB)
    assert f"The scene-gaze of P1: {GTOK}" in s
    assert f"The scene-gaze of P2: {GTOK}" in s
    assert "gaze relation from" not in s


def test_flat_concat_order_matches_gtok_row_major():
    """The collate concatenates feats sample-major then in-prompt order; the hook fills
    <gtok> positions row-major. This test pins the ordering invariant at the feats level."""
    g = torch.Generator().manual_seed(1)
    N, De = 4, 256
    gf = {"v_src": torch.randn(N, De, generator=g),
          "v_tgt": torch.randn(N, De, generator=g),
          "edge_pp": torch.randn(N, N, De, generator=g),
          "edge_null_in": torch.randn(N, De, generator=g),
          "head_bboxes": torch.rand(N, 4, generator=g)}
    batch = [gather_feats(gf, "lah", 0, 1), gather_feats(gf, "sa", 2, 3)]
    feats = torch.cat([b[0] for b in batch], dim=0)
    roles = torch.cat([b[1] for b in batch], dim=0)
    assert feats.shape == (5 + 6, De)      # lah=5, sa=6
    assert roles.shape == (11,)
    assert torch.equal(feats[:5], batch[0][0])
    assert torch.equal(feats[5:], batch[1][0])
