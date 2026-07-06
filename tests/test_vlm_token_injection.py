# tests/test_vlm_token_injection.py
import torch
import torch.nn as nn
from vlm.injection import gather_feats, GraphTokenProjector, ROLE, TOK_COUNT, N_ROLES, install_hook, GTOK


def _fake_gf(N=4, De=256):
    g = torch.Generator().manual_seed(0)
    return {
        "v_src": torch.randn(N, De, generator=g),
        "v_tgt": torch.randn(N, De, generator=g),
        "edge_pp": torch.randn(N, N, De, generator=g),
        "edge_null_in": torch.randn(N, De, generator=g),
        "head_bboxes": torch.rand(N, 4, generator=g),
    }


def test_counts_and_roles_per_task():
    gf = _fake_gf()
    i, j = 1, 2
    feats, roles = gather_feats(gf, "lah", i, j)
    assert feats.shape == (3, 256) and roles.tolist() == [ROLE["SRC"], ROLE["TGT"], ROLE["EDGE_FWD"]]
    feats, roles = gather_feats(gf, "laeo", i, j)
    assert feats.shape == (4, 256)
    assert roles.tolist() == [ROLE["SRC"], ROLE["SRC"], ROLE["EDGE_FWD"], ROLE["EDGE_BWD"]]
    feats, roles = gather_feats(gf, "sa", i, j)
    assert feats.shape == (6, 256)
    assert roles.tolist() == [ROLE["SRC"], ROLE["SRC"], ROLE["NULL_IN"],
                              ROLE["NULL_IN"], ROLE["EDGE_FWD"], ROLE["EDGE_BWD"]]
    for t in ("lah", "laeo", "sa"):
        f, r = gather_feats(gf, t, i, j)
        assert f.shape[0] == TOK_COUNT[t] == r.shape[0]


def test_edge_orientation_fwd_is_j_i():
    """EDGE_FWD ('i looks at j') must be the tensor slice edge_pp[j, i]."""
    gf = _fake_gf()
    i, j = 0, 3
    feats, roles = gather_feats(gf, "lah", i, j)
    fwd = feats[roles.tolist().index(ROLE["EDGE_FWD"])]
    assert torch.equal(fwd, gf["edge_pp"][j, i].float())
    feats, roles = gather_feats(gf, "laeo", i, j)
    r = roles.tolist()
    assert torch.equal(feats[r.index(ROLE["EDGE_FWD"])], gf["edge_pp"][j, i].float())
    assert torch.equal(feats[r.index(ROLE["EDGE_BWD"])], gf["edge_pp"][i, j].float())


def test_node_selection():
    gf = _fake_gf()
    i, j = 2, 0
    feats, _ = gather_feats(gf, "lah", i, j)
    assert torch.equal(feats[0], gf["v_src"][i].float())      # SRC = v_src[i]
    assert torch.equal(feats[1], gf["v_tgt"][j].float())      # TGT = v_tgt[j]
    feats, _ = gather_feats(gf, "sa", i, j)
    assert torch.equal(feats[0], gf["v_src"][i].float())
    assert torch.equal(feats[1], gf["v_src"][j].float())
    assert torch.equal(feats[2], gf["edge_null_in"][i].float())
    assert torch.equal(feats[3], gf["edge_null_in"][j].float())


def test_projector_shape_and_role_conditioning():
    proj = GraphTokenProjector(out_dim=64)
    assert proj.role_emb.shape == (N_ROLES, 256)
    feats = torch.randn(6, 256)
    roles = torch.tensor([0, 0, 4, 4, 2, 3])
    out = proj(feats, roles)
    assert out.shape == (6, 64)
    # same feature, different role -> different output (role actually used)
    # Re-init role_emb to non-zero so the test is non-vacuous (zero-init makes roles identical).
    g = torch.Generator().manual_seed(42)
    proj.role_emb.data = torch.randn(proj.role_emb.shape, generator=g)
    a = proj(torch.zeros(1, 256), torch.tensor([ROLE["SRC"]]))
    b = proj(torch.zeros(1, 256), torch.tensor([ROLE["TGT"]]))
    assert not torch.allclose(a, b)


class _StubLM(nn.Module):
    """Minimal stand-in for the Qwen text model: records the inputs_embeds it receives."""
    def __init__(self, D=8):
        super().__init__()
        self.D = D
        self.seen = None

    def forward(self, inputs_embeds=None, **kw):
        self.seen = inputs_embeds
        return inputs_embeds


def test_hook_overwrites_only_gtok_positions_variable_length():
    lm = _StubLM(D=8)
    install_hook(lm)
    proj = GraphTokenProjector(out_dim=8)
    # batch of 2: LAH (3 tokens) then SA (6 tokens) = 9 gtok positions
    gf = {
        "v_src": torch.randn(4, 256), "v_tgt": torch.randn(4, 256),
        "edge_pp": torch.randn(4, 4, 256), "edge_null_in": torch.randn(4, 256),
        "head_bboxes": torch.rand(4, 4),
    }
    f0, r0 = gather_feats(gf, "lah", 0, 1)
    f1, r1 = gather_feats(gf, "sa", 2, 3)
    feats = torch.cat([f0, f1]); roles = torch.cat([r0, r1])
    tokens = proj(feats, roles)                       # (9, 8)
    B, L = 2, 12
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, 1:4] = True                               # 3 positions row 0
    mask[1, 2:8] = True                               # 6 positions row 1
    assert int(mask.sum()) == tokens.shape[0]
    emb = torch.zeros(B, L, 8)
    lm._gtok = {"tokens": tokens, "mask": mask}
    _, kwargs = None, {"inputs_embeds": emb}
    # emulate the pre-hook path
    out_args, out_kwargs = lm._forward_pre_hooks[list(lm._forward_pre_hooks)[0]](
        lm, (), {"inputs_embeds": emb})
    new_emb = out_kwargs["inputs_embeds"]
    assert torch.allclose(new_emb[mask], tokens.to(new_emb.dtype))
    assert torch.count_nonzero(new_emb[~mask]) == 0   # non-gtok untouched
