# tests/test_vlm_token_injection.py
import torch
import torch.nn as nn
from vlm.injection import (gather_feats, gather_heatmaps, pair_belief, query_slots,
                           GraphTokenProjector, HeatmapEncoder, ROLE, TOK_COUNT, HM_COUNT,
                           N_ROLES, install_hook, GTOK)


def _fake_gf(N=4, De=256, Hh=64, Ww=64):
    g = torch.Generator().manual_seed(0)
    return {
        "v_src": torch.randn(N, De, generator=g),
        "v_tgt": torch.randn(N, De, generator=g),
        "edge_pp": torch.randn(N, N, De, generator=g),
        "edge_null_in": torch.randn(N, De, generator=g),
        "head_bboxes": torch.rand(N, 4, generator=g),
        "lah_logits": torch.randn(N, N, generator=g),
        "laeo_logits": torch.randn(N, N, generator=g),
        "sa_logits": torch.randn(N, N, generator=g),
        "overlap": torch.rand(N, N, generator=g),
        "gaze_heatmap": torch.randn(N, Hh, Ww, generator=g),
    }


def test_query_slots_direction():
    """LAH record (i, j) means "j looks at i" (verified: gaze_point[j] in bbox[i]
    for 77% of yes vs 7% of no; lah_logits[j,i] AUC 0.94). The first-named person
    A in the question ("Is A looking at B?") must therefore be slot j."""
    rec = {"task": "lah", "i": 2, "j": 3, "li": "P1", "lj": "P2"}
    a, b, la, lb = query_slots(rec)
    assert (a, b) == (3, 2)          # A = looker = slot j
    assert (la, lb) == ("P2", "P1")
    rec = {"task": "laeo", "i": 2, "j": 3, "li": "P1", "lj": "P2"}
    assert query_slots(rec) == (2, 3, "P1", "P2")   # symmetric: unchanged
    rec = {"task": "sa", "i": 0, "j": 1, "li": "P1", "lj": "P2"}
    assert query_slots(rec) == (0, 1, "P1", "P2")


def test_counts_and_roles_per_task():
    gf = _fake_gf()
    a, b = 1, 2
    feats, roles = gather_feats(gf, "lah", a, b)
    assert feats.shape == (5, 256) and roles.tolist() == [
        ROLE["SRC"], ROLE["TGT"], ROLE["EDGE_FWD"], ROLE["SRC"], ROLE["TGT"]]
    feats, roles = gather_feats(gf, "laeo", a, b)
    assert feats.shape == (6, 256)
    assert roles.tolist() == [ROLE["SRC"], ROLE["SRC"], ROLE["EDGE_FWD"], ROLE["EDGE_BWD"],
                              ROLE["SRC"], ROLE["SRC"]]
    feats, roles = gather_feats(gf, "sa", a, b)   # SA: null_in only, no p2p edge
    assert feats.shape == (6, 256)
    assert roles.tolist() == [ROLE["SRC"], ROLE["SRC"], ROLE["NULL_IN"], ROLE["NULL_IN"],
                              ROLE["SRC"], ROLE["SRC"]]
    for t in ("lah", "laeo", "sa"):
        f, r = gather_feats(gf, t, a, b)
        assert f.shape[0] == TOK_COUNT[t] == r.shape[0]


def test_gather_heatmaps_persons_and_count():
    gf = _fake_gf()
    a, b = 1, 3
    hm = gather_heatmaps(gf, "lah", a, b)         # A only
    assert hm.shape == (HM_COUNT["lah"], 64, 64)
    assert torch.equal(hm[0], gf["gaze_heatmap"][a].float())
    for t in ("laeo", "sa"):
        hm = gather_heatmaps(gf, t, a, b)         # A, B
        assert hm.shape == (HM_COUNT[t], 64, 64)
        assert torch.equal(hm[0], gf["gaze_heatmap"][a].float())
        assert torch.equal(hm[1], gf["gaze_heatmap"][b].float())


def test_heatmap_encoder_shape_and_softmax_invariance():
    enc = HeatmapEncoder(out_dim=32)
    hm = torch.randn(3, 64, 64)
    out = enc(hm)
    assert out.shape == (3, 32)
    # spatial-softmax input -> adding a constant to a heatmap leaves the token unchanged
    out2 = enc(hm + 5.0)
    assert torch.allclose(out, out2, atol=1e-4)


def test_edge_orientation_fwd_is_a_b():
    """EDGE_FWD ("A looks at B", the asked direction) must be edge_pp[a, b]
    (edge_pp[x, y] = E[x→y]; readout = lah_logits[x, y])."""
    gf = _fake_gf()
    a, b = 0, 3
    feats, roles = gather_feats(gf, "lah", a, b)
    fwd = feats[roles.tolist().index(ROLE["EDGE_FWD"])]
    assert torch.equal(fwd, gf["edge_pp"][a, b].float())
    feats, roles = gather_feats(gf, "laeo", a, b)
    r = roles.tolist()
    assert torch.equal(feats[r.index(ROLE["EDGE_FWD"])], gf["edge_pp"][a, b].float())
    assert torch.equal(feats[r.index(ROLE["EDGE_BWD"])], gf["edge_pp"][b, a].float())


def test_node_selection():
    gf = _fake_gf()
    a, b = 2, 0
    feats, _ = gather_feats(gf, "lah", a, b)
    assert torch.equal(feats[0], gf["v_src"][a].float())      # SRC = v_src[A] (looker)
    assert torch.equal(feats[1], gf["v_tgt"][b].float())      # TGT = v_tgt[B] (target)
    feats, _ = gather_feats(gf, "sa", a, b)
    assert torch.equal(feats[0], gf["v_src"][a].float())
    assert torch.equal(feats[1], gf["v_src"][b].float())
    assert torch.equal(feats[2], gf["edge_null_in"][a].float())   # scene-gaze A
    assert torch.equal(feats[3], gf["edge_null_in"][b].float())   # scene-gaze B


def test_pair_belief_orientation_and_symmetry():
    gf = _fake_gf()
    a, b = 1, 3
    bl = pair_belief(gf, "lah", a, b)
    assert abs(bl["p"] - torch.sigmoid(gf["lah_logits"][a, b]).item()) < 1e-6
    assert abs(bl["ov"] - float(gf["overlap"][a, b])) < 1e-6
    for t in ("laeo", "sa"):
        assert abs(pair_belief(gf, t, a, b)["p"] - pair_belief(gf, t, b, a)["p"]) < 1e-6


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


def test_hook_fills_both_gtok_and_hmtok_positions():
    lm = _StubLM(D=8)
    install_hook(lm)
    proj = GraphTokenProjector(out_dim=8)
    enc = HeatmapEncoder(out_dim=8)
    # batch of 2: LAH (5 gtok, 1 hmtok) then SA (6 gtok, 2 hmtok)
    gf = _fake_gf()
    f0, r0 = gather_feats(gf, "lah", 0, 1)
    f1, r1 = gather_feats(gf, "sa", 2, 3)
    gtokens = proj(torch.cat([f0, f1]), torch.cat([r0, r1]))     # (11, 8)
    hms = torch.cat([gather_heatmaps(gf, "lah", 0, 1), gather_heatmaps(gf, "sa", 2, 3)])
    hmtokens = enc(hms)                                          # (3, 8)
    B, L = 2, 20
    gmask = torch.zeros(B, L, dtype=torch.bool)
    gmask[0, 1:6] = True                              # 5 gtok row 0
    gmask[1, 2:8] = True                              # 6 gtok row 1
    hmask = torch.zeros(B, L, dtype=torch.bool)
    hmask[0, 6] = True                                # 1 hmtok row 0
    hmask[1, 8:10] = True                             # 2 hmtok row 1
    assert int(gmask.sum()) == gtokens.shape[0]
    assert int(hmask.sum()) == hmtokens.shape[0]
    assert int((gmask & hmask).sum()) == 0           # disjoint placeholders
    emb = torch.zeros(B, L, 8)
    lm._gtok = {"tokens": gtokens, "mask": gmask}
    lm._hmtok = {"tokens": hmtokens, "mask": hmask}
    _, out_kwargs = lm._forward_pre_hooks[list(lm._forward_pre_hooks)[0]](
        lm, (), {"inputs_embeds": emb})
    new_emb = out_kwargs["inputs_embeds"]
    assert torch.allclose(new_emb[gmask], gtokens.to(new_emb.dtype))
    assert torch.allclose(new_emb[hmask], hmtokens.to(new_emb.dtype))
    assert torch.count_nonzero(new_emb[~(gmask | hmask)]) == 0   # rest untouched
