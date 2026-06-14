# mtgs/networks/vlm/mtgs_builder.py
"""Shared helper to construct a frozen MTGS backbone from the Hydra config.

Used by both the Stage-B trainer (online feature extraction fallback) and the
offline feature-extraction script. Mirrors the constructor argument mapping in
mtgs/networks/models.py::MTGSModel.__init__ so the backbone is identical to the
one trained in Stage A.
"""
import torch

from mtgs.networks.mtgs_net import MTGS


def build_mtgs(cfg) -> MTGS:
    """Instantiate an MTGS backbone using the same arg mapping as MTGSModel."""
    return MTGS(
        patch_size=cfg.model.patch_size,
        token_dim=cfg.model.token_dim,
        image_size=cfg.model.image_size,
        gaze_feature_dim=cfg.model.gaze_feature_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_num_heads=cfg.model.encoder_num_heads,
        encoder_num_global_tokens=cfg.model.encoder_num_global_tokens,
        encoder_mlp_ratio=cfg.model.encoder_mlp_ratio,
        encoder_use_qkv_bias=cfg.model.encoder_use_qkv_bias,
        encoder_drop_rate=cfg.model.encoder_drop_rate,
        encoder_attn_drop_rate=cfg.model.encoder_attn_drop_rate,
        encoder_drop_path_rate=cfg.model.encoder_drop_path_rate,
        decoder_feature_dim=cfg.model.decoder_feature_dim,
        decoder_hooks=cfg.model.decoder_hooks,
        decoder_hidden_dims=cfg.model.decoder_hidden_dims,
        decoder_use_bn=cfg.model.decoder_use_bn,
        temporal_context=cfg.data.temporal_context,
        output=cfg.model.output,
        gaze_graph_num_layers=cfg.gaze_graph.num_layers,
        gaze_graph_edge_dim=cfg.gaze_graph.edge_dim,
        gaze_graph_use_prior=cfg.gaze_graph.use_prior,
        gaze_graph_prior_weight=cfg.gaze_graph.prior_weight,
        gaze_graph_use_node_xattn=cfg.gaze_graph.use_node_xattn,
        gaze_graph_laeo_derive=cfg.gaze_graph.laeo_derive,
        gaze_graph_use=cfg.gaze_graph.use,
    )


def load_stage_a_into(mtgs: MTGS, ckpt_path: str) -> None:
    """Load a Stage-A MTGSModel checkpoint (model.* prefix) into an MTGS module."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    missing, unexpected = mtgs.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_stage_a] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[load_stage_a] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")


def attach_graph_state_hooks(mtgs: MTGS, store: dict):
    """Register forward hooks capturing refiner (E, v_src, v_tgt) + edge_valid.

    Mutates `store` in place on each forward; returns the hook handles so the
    caller can remove them if desired.
    """
    def _refiner_hook(module, inp, output):
        E, v_src, v_tgt = output
        store["E"] = E.detach()
        store["v_src"] = v_src.detach()
        store["v_tgt"] = v_tgt.detach()

    def _block_hook(module, inp, output):
        # output = (lah, laeo, sa, null_in, null_out, edge_valid)
        store["edge_valid"] = output[5].detach()

    h1 = mtgs.gaze_graph_block.refiner.register_forward_hook(_refiner_hook)
    h2 = mtgs.gaze_graph_block.register_forward_hook(_block_hook)
    return [h1, h2]
