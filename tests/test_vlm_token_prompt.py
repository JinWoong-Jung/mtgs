# tests/test_vlm_token_prompt.py
import torch
from vlm.prompt import token_prompt
from vlm.injection import GTOK, TOK_COUNT, gather_feats


def test_gtok_count_matches_tok_count():
    bb = [0.10, 0.20, 0.30, 0.40]
    for task in ("lah", "laeo", "sa"):
        s = token_prompt(task, "P1", "P2", bb, bb)
        assert s.count(GTOK) == TOK_COUNT[task], (task, s)


def test_prompt_contains_boxes_and_question():
    bb_i = [0.11, 0.22, 0.33, 0.44]
    bb_j = [0.55, 0.66, 0.77, 0.88]
    s = token_prompt("lah", "P1", "P2", bb_i, bb_j)
    assert "[0.11,0.22,0.33,0.44]" in s
    assert "[0.55,0.66,0.77,0.88]" in s
    assert "Is P1 looking at P2?" in s
    assert s.strip().endswith("yes or no.")


def test_laeo_sa_questions():
    bb = [0.0, 0.0, 0.1, 0.1]
    assert "looking at each other" in token_prompt("laeo", "A", "B", bb, bb)
    assert "same thing or person" in token_prompt("sa", "A", "B", bb, bb)


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
    assert feats.shape == (3 + 6, De)
    assert roles.shape == (9,)
    # first 3 rows are the LAH sample, next 6 the SA sample
    assert torch.equal(feats[:3], batch[0][0])
    assert torch.equal(feats[3:], batch[1][0])
