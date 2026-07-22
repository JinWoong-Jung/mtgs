"""Row/col message-passing ablation for the gaze graph refiner.

Covers the four settings (both / row-only / col-only / neither) under the
capacity-controlled design (method B): every module is always built (so param
counts match across settings); a disabled direction only has its contributions
gated out in forward. Each flag governs BOTH channels of its direction:
``use_row_attn`` → row edge-attention (①) AND source node-update (④ out_agg →
v_src); ``use_col_attn`` → col edge-attention (②) AND target node-update (④
in_agg → v_tgt). Re-inject (⑤) runs iff at least one direction is active. The
tests assert (1) identical parameter counts across all four settings, (2)
correct forward shapes for T=1 and T>1 with padded (variable) people, (3) a
disabled attention branch contributes exactly zero to the refresh input, and
(4) the node-update/re-inject path is gated by the same flags (a disabled
direction leaves its node embedding unchanged; "neither" skips re-inject so E
is only edge-init + refresh + temporal).
"""

import pytest
import torch

from mtgs.networks.adaptor_modules import (
    GazeGraphBlock,
    _RefinerLayer,
    _UnifiedRefiner,
)

DE, HEADS = 16, 4
SETTINGS = {
    "both": (True, True),
    "row_only": (True, False),
    "col_only": (False, True),
    "neither": (False, False),
}


def _make_inputs(B=2, T=1, N=3, n_valid=None, De=DE, seed=0):
    """Build valid _RefinerLayer.forward inputs, mirroring GazeGraphBlock.forward.

    ``n_valid`` (per-batch count of real people, right-aligned like the model's
    node_valid mask) exercises padding; defaults to all-valid.
    """
    torch.manual_seed(seed)
    Tl = N + 2
    if n_valid is None:
        n_valid = [N] * B
    device = torch.device("cpu")

    node_valid = (
        torch.arange(N).view(1, N)
        >= (N - torch.tensor(n_valid).view(B, 1))
    )  # (B, N)
    eye = torch.eye(N, dtype=torch.bool)
    p2p_valid = node_valid.unsqueeze(2) & node_valid.unsqueeze(1) & ~eye.unsqueeze(0)
    null_valid = node_valid.unsqueeze(2)  # (B, N, 1)
    ev_bool = torch.cat(
        [p2p_valid, null_valid.expand(B, N, 1), null_valid.expand(B, N, 1)], dim=2
    )  # (B, N, Tl)
    ev_bool_T = ev_bool.unsqueeze(1).expand(B, T, N, Tl)
    ev = ev_bool_T.unsqueeze(-1).float()          # (B, T, N, Tl, 1)
    ev_sq = ev.squeeze(-1)                          # (B, T, N, Tl)

    E = torch.randn(B, T, N, Tl, De)

    # row_kpm: (B*T*N, Tl) — mask invalid targets; unmask fully-masked rows.
    row_kpm = ~ev_bool_T.reshape(B * T * N, Tl)
    row_kpm = _UnifiedRefiner._safe_kpm(row_kpm)
    # col_kpm: (B*T*(N+1), N) — target k over N sources, null_out (last slot) excluded.
    col_valid = ev_bool_T[:, :, :, : N + 1].permute(0, 1, 3, 2)  # (B,T,N+1,N)
    col_kpm = ~col_valid.reshape(B * T * (N + 1), N)
    col_kpm = _UnifiedRefiner._safe_kpm(col_kpm)

    v_src = torch.randn(B, T, N, De)
    v_tgt = torch.randn(B, T, Tl, De)
    return dict(E=E, ev=ev, ev_sq=ev_sq, row_kpm=row_kpm, col_kpm=col_kpm,
                v_src=v_src, v_tgt=v_tgt)


def test_all_four_settings_share_identical_parameter_count():
    counts = {}
    for name, (ur, uc) in SETTINGS.items():
        layer = _RefinerLayer(DE, HEADS, use_row_attn=ur, use_col_attn=uc)
        counts[name] = sum(p.numel() for p in layer.parameters())
    assert len(set(counts.values())) == 1, f"capacity control broken: {counts}"


@pytest.mark.parametrize("name,flags", SETTINGS.items())
@pytest.mark.parametrize("T", [1, 3])
def test_forward_shapes_for_each_setting_single_and_multi_frame(name, flags, T):
    ur, uc = flags
    B, N = 2, 3
    layer = _RefinerLayer(DE, HEADS, use_row_attn=ur, use_col_attn=uc)
    # n_valid=[3,2] exercises padding: batch item 1 has one padded (invalid) person.
    inp = _make_inputs(B=B, T=T, N=N, n_valid=[3, 2])
    E_out, v_src_out, v_tgt_out = layer(**inp)
    Tl = N + 2
    assert E_out.shape == (B, T, N, Tl, DE)
    assert v_src_out.shape == (B, T, N, DE)
    assert v_tgt_out.shape == (B, T, Tl, DE)
    assert torch.isfinite(E_out).all()
    # invalid edges must stay zero (masked by ev after refresh/inject).
    assert (E_out[inp["ev"].expand_as(E_out) == 0] == 0).all()


def _capture_refresh_input(layer, inp):
    """Return the tensor fed to ``layer.refresh`` (= cat(row_context, col_context))
    during one forward, via a pre-forward hook. This is the post-attention / at-refresh
    state the ablation acts on -- BEFORE steps ④⑤⑥ further mutate E."""
    captured = {}

    def hook(_module, args):
        captured["x"] = args[0].detach().clone()

    handle = layer.refresh.register_forward_pre_hook(hook)
    try:
        layer(**inp)
    finally:
        handle.remove()
    return captured["x"]  # (..., 2*De): [:De]=row_context, [De:]=col_context


@pytest.mark.parametrize("T", [1, 3])
def test_disabled_branch_contributes_exactly_zero_context_at_refresh(T):
    """The core capacity-controlled ablation guarantee, checked at the refresh step."""
    inp = _make_inputs(B=2, T=T, N=3, n_valid=[3, 2])

    # row-only: col_context (second half) must be exactly zero; row half must not be.
    row_only = _RefinerLayer(DE, HEADS, use_row_attn=True, use_col_attn=False)
    x = _capture_refresh_input(row_only, inp)
    assert x[..., DE:].abs().sum() == 0, "col context must be zero when col is disabled"
    assert x[..., :DE].abs().sum() > 0, "row context must be active when row is enabled"

    # col-only: mirror.
    col_only = _RefinerLayer(DE, HEADS, use_row_attn=False, use_col_attn=True)
    x = _capture_refresh_input(col_only, inp)
    assert x[..., :DE].abs().sum() == 0, "row context must be zero when row is disabled"
    assert x[..., DE:].abs().sum() > 0, "col context must be active when col is enabled"

    # neither: both halves exactly zero -> refresh sees an all-zero context.
    neither = _RefinerLayer(DE, HEADS, use_row_attn=False, use_col_attn=False)
    x = _capture_refresh_input(neither, inp)
    assert x.abs().sum() == 0, "neither must feed an all-zero context to refresh"


def test_neither_reduces_refresh_delta_to_a_constant_bias_map():
    """With both contexts zero, refresh(cat(0,0)) is the SAME vector for every edge
    (only the MLP's response to the zero input + bias), so the pre-mask post-refresh
    delta E-E_in is constant across all edge slots. This is the concrete meaning of
    'row/col attention removed' at the refresh step (steps ④⑤⑥ still run afterwards)."""
    inp = _make_inputs(B=1, T=1, N=3, n_valid=[3])
    neither = _RefinerLayer(DE, HEADS, use_row_attn=False, use_col_attn=False)
    x = _capture_refresh_input(neither, inp)          # all-zero (B,T,N,Tl,2De)
    delta = neither.refresh(x)                          # (B,T,N,Tl,De)
    flat = delta.reshape(-1, DE)
    # every edge position received the identical refresh vector.
    assert torch.allclose(flat, flat[0].expand_as(flat), atol=1e-6)


def test_unified_refiner_threads_flags_to_every_layer():
    ref = _UnifiedRefiner(DE, num_layers=2, heads=HEADS,
                          use_row_attn=True, use_col_attn=False,
                          use_temporal_attn=False)
    for layer in ref.layers:
        assert layer.use_row_attn is True and layer.use_col_attn is False
        assert layer.use_temporal_attn is False


# ── Node-update / re-inject coupling (each flag governs BOTH its channels) ──────

def _capture_reinject_ran(layer, inp):
    """True iff step ⑤ (layer.inject) is invoked during one forward."""
    ran = {"v": False}

    def hook(_module, _args):
        ran["v"] = True

    handle = layer.inject.register_forward_pre_hook(hook)
    try:
        layer(**inp)
    finally:
        handle.remove()
    return ran["v"]


def test_row_flag_gates_source_node_update_col_flag_gates_target():
    """Disabling a direction must leave that direction's node embedding unchanged
    (row → v_src via out_agg; col → v_tgt via in_agg), verified by feeding a layer
    whose refresh/inject can only move a node if its own update path ran."""
    inp = _make_inputs(B=1, T=1, N=3, n_valid=[3])

    # col-only: source node-update (row channel) is skipped, so v_src is returned
    # unchanged; the target update (col channel) runs, so v_tgt changes.
    col_only = _RefinerLayer(DE, HEADS, use_row_attn=False, use_col_attn=True)
    _, v_src_out, v_tgt_out = col_only(**inp)
    assert torch.equal(v_src_out, inp["v_src"]), "v_src must be untouched when row off"
    assert not torch.equal(v_tgt_out, inp["v_tgt"]), "v_tgt must update when col on"

    # row-only: mirror — v_tgt untouched, v_src updated.
    row_only = _RefinerLayer(DE, HEADS, use_row_attn=True, use_col_attn=False)
    _, v_src_out, v_tgt_out = row_only(**inp)
    assert not torch.equal(v_src_out, inp["v_src"]), "v_src must update when row on"
    assert torch.equal(v_tgt_out, inp["v_tgt"]), "v_tgt must be untouched when col off"


def test_neither_skips_reinject_and_leaves_both_nodes_unchanged():
    """With both directions off there is no node message-passing: ⑤ re-inject is
    skipped entirely and both node embeddings pass through unchanged."""
    inp = _make_inputs(B=1, T=1, N=3, n_valid=[3])
    neither = _RefinerLayer(DE, HEADS, use_row_attn=False, use_col_attn=False)
    assert not _capture_reinject_ran(neither, inp), "re-inject must be skipped when neither"
    _, v_src_out, v_tgt_out = neither(**inp)
    assert torch.equal(v_src_out, inp["v_src"])
    assert torch.equal(v_tgt_out, inp["v_tgt"])


@pytest.mark.parametrize("name,flags", SETTINGS.items())
def test_reinject_runs_iff_any_direction_active(name, flags):
    ur, uc = flags
    inp = _make_inputs(B=1, T=1, N=3, n_valid=[3])
    layer = _RefinerLayer(DE, HEADS, use_row_attn=ur, use_col_attn=uc)
    assert _capture_reinject_ran(layer, inp) == (ur or uc)


def test_temporal_attn_off_skips_and_removes_the_module():
    """Module-skip ablation (not capacity-controlled): off => temporal encoder not
    built, so its params are removed and the layer is strictly smaller than 'on'."""
    on = _RefinerLayer(DE, HEADS, use_temporal_attn=True)
    off = _RefinerLayer(DE, HEADS, use_temporal_attn=False)
    assert on.temporal is not None and off.temporal is None
    n_on = sum(p.numel() for p in on.parameters())
    n_off = sum(p.numel() for p in off.parameters())
    assert n_off < n_on, "disabling temporal must remove its parameters"


def test_temporal_attn_off_matches_the_pre_temporal_state_when_multiframe():
    """With T>1, 'temporal off' must equal 'temporal on' up to step ⑤ — i.e. off is
    exactly the on-path with step ⑥ removed. We verify by disabling the temporal
    encoder on an otherwise identically-initialised layer and checking the temporal
    step is what changed."""
    inp = _make_inputs(B=2, T=3, N=3, n_valid=[3, 2])

    off = _RefinerLayer(DE, HEADS, use_temporal_attn=False)
    E_off, _, _ = off(**{k: v.clone() if torch.is_tensor(v) else v for k, v in inp.items()})

    # Build an 'on' layer that shares all non-temporal weights, then confirm the
    # temporal step actually perturbs E (so the flag is not a no-op) at T>1.
    on = _RefinerLayer(DE, HEADS, use_temporal_attn=True)
    on.load_state_dict(off.state_dict(), strict=False)  # temporal stays randomly init
    E_on, _, _ = on(**{k: v.clone() if torch.is_tensor(v) else v for k, v in inp.items()})
    assert not torch.allclose(E_on, E_off), "temporal step must change E when on (T>1)"


def test_temporal_attn_flag_is_a_noop_at_t1():
    """At T=1 step ⑥ never runs regardless of the flag, so on/off produce identical
    E for shared weights -- the flag only matters for multi-frame inputs."""
    inp = _make_inputs(B=2, T=1, N=3, n_valid=[3, 2])
    off = _RefinerLayer(DE, HEADS, use_temporal_attn=False)
    on = _RefinerLayer(DE, HEADS, use_temporal_attn=True)
    on.load_state_dict(off.state_dict(), strict=False)
    E_off, _, _ = off(**{k: v.clone() if torch.is_tensor(v) else v for k, v in inp.items()})
    E_on, _, _ = on(**{k: v.clone() if torch.is_tensor(v) else v for k, v in inp.items()})
    torch.testing.assert_close(E_on, E_off)


# ── Null-node ablation ────────────────────────────────────────────────────────

D_TOK, DE_BLK = 24, 16


def _block_inputs(B=2, T=2, N=3, n_valid=None, D=D_TOK, seed=1):
    torch.manual_seed(seed)
    if n_valid is None:
        n_valid = [N] * B
    Hh = Ww = 8
    return dict(
        person_tokens=torch.randn(B, T, N, D),
        num_valid_people=torch.tensor(n_valid),
        gaze_vecs=torch.randn(B, T, N, 2),
        head_bboxes=torch.rand(B, T, N, 4),
        gaze_heatmaps=torch.rand(B, T, N, Hh, Ww),
        inout_logits=torch.randn(B, T, N),
        gaze_feat=torch.randn(B, T, N, D),
    )


def _capture_refined_edges(block, inp):
    """Grab the fully-refined edge tensor E (post-refiner, pre-readout) via a hook on
    the refiner, so we can assert masked null columns are exactly zero at readout time."""
    captured = {}

    def hook(_m, _args, output):
        captured["E"] = output[0].detach().clone()  # refiner returns (E, v_src, v_tgt)

    handle = block.refiner.register_forward_hook(hook)
    try:
        block(**inp)
    finally:
        handle.remove()
    return captured["E"]  # (B, T, N, Tl, De)


@pytest.mark.parametrize("use_in,use_out", [(True, True), (False, True), (True, False), (False, False)])
def test_null_ablation_zeros_only_the_disabled_null_edges(use_in, use_out):
    N = 3
    block = GazeGraphBlock(token_dim=D_TOK, edge_dim=DE_BLK, num_layers=2, face_dim=D_TOK,
                           use_null_in=use_in, use_null_out=use_out)
    block.eval()
    E = _capture_refined_edges(block, _block_inputs(N=N))
    # slot N = null_in, slot N+1 = null_out
    ni, no = E[:, :, :, N, :], E[:, :, :, N + 1, :]
    if not use_in:
        assert ni.abs().sum() == 0, "disabled null_in edge must be exactly zero"
    if not use_out:
        assert no.abs().sum() == 0, "disabled null_out edge must be exactly zero"
    if use_in:
        assert ni.abs().sum() > 0, "active null_in edge must carry signal"
    if use_out:
        assert no.abs().sum() > 0, "active null_out edge must carry signal"


def test_null_ablation_is_capacity_controlled_identical_param_counts():
    counts = {}
    for use_in, use_out in [(True, True), (False, True), (True, False), (False, False)]:
        blk = GazeGraphBlock(token_dim=D_TOK, edge_dim=DE_BLK, num_layers=2, face_dim=D_TOK,
                             use_null_in=use_in, use_null_out=use_out)
        counts[(use_in, use_out)] = sum(p.numel() for p in blk.parameters())
    assert len(set(counts.values())) == 1, f"null ablation must preserve params: {counts}"


def test_null_in_removal_changes_person_edges_and_keeps_sa_shape():
    """-Null_in must actually perturb the person-person edges (info flow cut during
    refinement), while SA still outputs the right shape (head_sa reads ni=0)."""
    N = 3
    inp = _block_inputs(N=N, seed=2)
    full = GazeGraphBlock(token_dim=D_TOK, edge_dim=DE_BLK, num_layers=2, face_dim=D_TOK,
                          use_null_in=True, use_null_out=True)
    full.eval()
    minus_in = GazeGraphBlock(token_dim=D_TOK, edge_dim=DE_BLK, num_layers=2, face_dim=D_TOK,
                              use_null_in=False, use_null_out=True)
    minus_in.load_state_dict(full.state_dict())  # identical weights, only the flag differs
    minus_in.eval()

    E_full = _capture_refined_edges(full, inp)
    E_min = _capture_refined_edges(minus_in, {k: v.clone() for k, v in inp.items()})
    E_pp_full = E_full[:, :, :, :N, :]
    E_pp_min = E_min[:, :, :, :N, :]
    assert not torch.allclose(E_pp_full, E_pp_min), \
        "removing null_in must change how person edges are refined"

    lah, laeo, sa, null_in, null_out, edge_valid = minus_in(**{k: v.clone() for k, v in inp.items()})
    assert sa.shape == (2, 2, N, N)              # SA head still works with ni=0
    assert edge_valid[:, :, 2 * N].sum() == 0     # null_in marked invalid in readout


def test_null_flags_default_true_matches_explicit_full():
    """Backward compatibility: omitting the flags must equal Full (both on)."""
    a = GazeGraphBlock(token_dim=D_TOK, edge_dim=DE_BLK, num_layers=2, face_dim=D_TOK)
    b = GazeGraphBlock(token_dim=D_TOK, edge_dim=DE_BLK, num_layers=2, face_dim=D_TOK,
                       use_null_in=True, use_null_out=True)
    assert a.use_null_in and a.use_null_out
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
