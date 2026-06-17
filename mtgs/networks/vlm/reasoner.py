# mtgs/networks/vlm/reasoner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from transformers import AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration

from mtgs.networks.vlm.graph_tokenizer import GraphEvidenceTokenizer
from mtgs.networks.vlm.memory_attn import MemoryCrossAttn, MemoryAugmentedLayer
from mtgs.datasets.gaze_qa import QAPair
from mtgs.utils.image import IMG_MEAN, IMG_STD


# Qwen3-VL-8B (Qwen3-8B backbone): 36 layers, full_attention_interval=4
# GatedAttention (full attn) at every 4th layer starting from index 3
_DEFAULT_CROSS_ATTN_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35]


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _fmt_bbox(bbox: Tuple[float, float, float, float]) -> str:
    return "[{:.2f},{:.2f},{:.2f},{:.2f}]".format(*bbox)


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _download_snapshot(repo_id: str, local_dir: Path) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download a missing VLM backbone. "
            "Install it or place the model files under vlm.backbone."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {repo_id} to {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return str(local_dir)


def _resolve_backbone(backbone: str, default_namespace: str = "Qwen") -> str:
    """Resolve vlm.backbone, downloading missing local paths in-place."""
    raw = str(backbone).strip()
    if not raw:
        raise ValueError("vlm.backbone must not be empty.")

    path_like = raw.startswith(("/", "./", "../")) or raw.startswith(
        ("model/", "models/", "checkpoints/")
    )
    if path_like:
        local_path = Path(raw).expanduser()
        if _is_model_dir(local_path):
            return str(local_path)
        repo_id = f"{default_namespace}/{local_path.name}"
        return _download_snapshot(repo_id, local_path)

    if raw.count("/") == 1:
        return raw

    return f"{default_namespace}/{raw}"


class EvidenceAugmentedVLM(nn.Module):
    """Frozen Qwen3-VL-4B augmented with graph evidence via cross-attention.

    Entity grounding:
      - Text level: bbox [x1,y1,x2,y2] appended after each <P> token in prompt
      - Embedding level: Emb(<P>) += W_node · v  (v = v_src or v_tgt)

    Trainable: MemoryCrossAttn layers (9), W_node, <P> token embedding, Q_g, W_proj.
    Frozen: all other VLM parameters and MTGS pipeline.

    All QA pairs in a step are forwarded through the LLM in a single batched call
    (left-padded; attention_mask + position_ids keep positions correct).
    """

    def __init__(self, cfg):
        super().__init__()
        llm_cfg = cfg.vlm
        edge_dim = cfg.gaze_graph.edge_dim
        self._prompt_templates = dict(llm_cfg.prompt_templates)

        # ── Load tokenizer + VLM (text-only path) ─────────────────────────────
        backbone = _resolve_backbone(llm_cfg.backbone)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            backbone, trust_remote_code=True, local_files_only=True
        )
        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            backbone,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        for p in self.llm.parameters():
            p.requires_grad_(False)

        d_llm = self.llm.config.text_config.hidden_size  # e.g. 2560 for Qwen3-VL-4B

        # ── Special <P> token ─────────────────────────────────────────────────
        self.hf_tokenizer.add_special_tokens({"additional_special_tokens": ["<P>"]})
        self.llm.resize_token_embeddings(len(self.hf_tokenizer))
        self._P_token_id = self.hf_tokenizer.convert_tokens_to_ids("<P>")
        self._yes_id = self.hf_tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id  = self.hf_tokenizer.encode("No",  add_special_tokens=False)[0]

        # ── Entity grounding: edge_dim → d_llm ───────────────────────────────
        self.W_node = nn.Linear(edge_dim, d_llm, bias=False)

        # ── Graph tokenizer (Stage 4) ─────────────────────────────────────────
        self.graph_tokenizer = GraphEvidenceTokenizer(
            edge_dim=edge_dim,
            d_llm=d_llm,
            m=llm_cfg.memory_tokens_m,
            depth=llm_cfg.tokenizer_depth,
        )

        # ── Visual encoder (optional: prepend scene tokens as soft prefix) ──────
        self.use_visual_encoder = _as_bool(getattr(llm_cfg, "visual_encoder", True))
        if self.use_visual_encoder:
            self.img_processor = AutoProcessor.from_pretrained(
                backbone, trust_remote_code=True, local_files_only=True
            )
            self.register_buffer(
                "_img_mean", torch.tensor(IMG_MEAN).view(3, 1, 1), persistent=False
            )
            self.register_buffer(
                "_img_std", torch.tensor(IMG_STD).view(3, 1, 1), persistent=False
            )
        else:
            self.img_processor = None

        # ── Wrap Full-Attention layers with MemoryCrossAttn ───────────────────
        indices = list(llm_cfg.get("cross_attn_layer_indices",
                                   _DEFAULT_CROSS_ATTN_INDICES))
        self._cross_attn_indices = indices
        text_layers = self.llm.model.language_model.layers
        for idx in indices:
            orig = text_layers[idx]
            cross_attn = MemoryCrossAttn(d_llm, num_heads=8)
            text_layers[idx] = MemoryAugmentedLayer(orig, cross_attn)

    # ── Visual encoder ───────────────────────────────────────────────────────

    def _encode_scene(self, scene_img: Tensor) -> Tensor:
        """scene_img: (C, H, W) MTGS-normalized → visual prefix tokens (1, L_vis, d_llm)."""
        raw = (scene_img.cpu().float() * self._img_std.cpu()
               + self._img_mean.cpu()).clamp(0.0, 1.0)
        pil = Image.fromarray((raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        return self._encode_scene_pil(pil, device=scene_img.device)

    def _encode_scene_pil(self, pil: Image.Image, device=None) -> Tensor:
        """PIL image → visual prefix tokens (1, L_vis, d_llm). No MTGS denorm needed.

        Called when the cache stores JPEG bytes (has_image_jpeg=true): the bytes are
        decoded directly to PIL in vlm_trainer and passed here, bypassing any MTGS
        normalisation/denormalisation round-trip.
        """
        if device is None:
            device = self._img_mean.device
        proc = self.img_processor.image_processor(images=[pil], return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device, dtype=torch.bfloat16)
        image_grid_thw = proc["image_grid_thw"].to(device)
        with torch.no_grad():
            vision_out = self.llm.model.get_image_features(pixel_values, image_grid_thw)
            vis_feats = vision_out.pooler_output[0]  # (L_vis, d_llm)
        return vis_feats.unsqueeze(0).to(torch.bfloat16)  # (1, L_vis, d_llm)

    # ── G_LLM injection helpers ───────────────────────────────────────────────

    def _set_G(self, G: Tensor):
        for idx in self._cross_attn_indices:
            self.llm.model.language_model.layers[idx]._G_LLM = G

    def _clear_G(self):
        for idx in self._cross_attn_indices:
            self.llm.model.language_model.layers[idx]._G_LLM = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        E_c: Tensor,
        edge_valid: Tensor,
        v_src_c: Tensor,
        v_tgt_c: Tensor,
        qa_pairs: List[QAPair],
        vis_prefix: "Optional[dict[int, Tensor]]" = None,
    ) -> Tensor:
        loss, _ = self._loss_and_scores(
            E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs,
            vis_prefix=vis_prefix, return_scores=False
        )
        return loss

    def loss_and_scores(
        self,
        E_c: Tensor,
        edge_valid: Tensor,
        v_src_c: Tensor,
        v_tgt_c: Tensor,
        qa_pairs: List[QAPair],
        vis_prefix: "Optional[dict[int, Tensor]]" = None,
    ) -> tuple[Tensor, Tensor]:
        """Return full-vocab NLL loss and per-pair Yes-vs-No scores in one pass."""
        return self._loss_and_scores(
            E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs,
            vis_prefix=vis_prefix, return_scores=True
        )

    def _loss_and_scores(
        self,
        E_c: Tensor,
        edge_valid: Tensor,
        v_src_c: Tensor,
        v_tgt_c: Tensor,
        qa_pairs: List[QAPair],
        vis_prefix: "Optional[dict[int, Tensor]]" = None,
        return_scores: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Single batched LLM forward over all QA pairs in the step.

        All pairs are left-padded to the same length and forwarded together as
        (P, max_len, d_llm). attention_mask and position_ids keep each sequence's
        positional context correct. logits[:, -1, :] is the prediction after
        "Answer:" for every pair regardless of padding length.

        Args:
            vis_prefix: pre-computed visual prefix as {batch_idx: (1, L_vis, d_llm) bf16},
                or None when visual encoder is disabled. Built by VLMReasonerModel
                (_build_vis_prefix) from either cached vis_tokens or online _encode_scene.
        """
        if not qa_pairs:
            loss = torch.tensor(0.0, requires_grad=True,
                                device=E_c.device, dtype=torch.float32)
            return loss, torch.zeros(0, device=E_c.device)

        device = E_c.device
        P = len(qa_pairs)

        G_LLM = self.graph_tokenizer(E_c, edge_valid)   # (B, M, d_llm)
        G_LLM_bf16 = G_LLM.to(torch.bfloat16)

        # vis_prefix is already computed by VLMReasonerModel._build_vis_prefix()
        # (either from cached vis_tokens or online _encode_scene; None = disabled)
        if vis_prefix is None:
            vis_prefix = {}

        # 1. Tokenize all prompts on CPU
        seqs: list[Tensor] = []
        p_pos_list: list[Tensor] = []
        for qa in qa_pairs:
            prompt = self._prompt_templates[qa.task].format(
                src_bbox=_fmt_bbox(qa.src_bbox), dst_bbox=_fmt_bbox(qa.dst_bbox)
            )
            ids = self.hf_tokenizer(prompt, return_tensors="pt").input_ids[0]  # (L,)
            seqs.append(ids)
            p_pos_list.append((ids == self._P_token_id).nonzero(as_tuple=True)[0])

        # 2. Left-pad ids; build base embeddings (bf16, no grad from frozen emb_table)
        max_text_len = max(s.shape[0] for s in seqs)
        pad_id = self.hf_tokenizer.pad_token_id or 0
        padded_ids = torch.full((P, max_text_len), pad_id, dtype=torch.long, device=device)
        text_attn_mask = torch.zeros(P, max_text_len, dtype=torch.long, device=device)
        for i, ids in enumerate(seqs):
            L = ids.shape[0]
            padded_ids[i, max_text_len - L:] = ids.to(device)
            text_attn_mask[i, max_text_len - L:] = 1

        emb_table = self.llm.get_input_embeddings()
        with torch.no_grad():
            embeds = emb_table(padded_ids).to(torch.bfloat16)  # (P, max_text_len, d_llm)

        # 3. Apply W_node grounding in-place (grad flows from W_node → embeds → LLM)
        for i, (qa, p_pos) in enumerate(zip(qa_pairs, p_pos_list)):
            offset = max_text_len - seqs[i].shape[0]
            b = qa.batch_idx
            v_subj = self.W_node(v_src_c[b, qa.src_idx].float()).to(torch.bfloat16)
            v_obj = (
                self.W_node(v_tgt_c[b, qa.dst_idx].float())
                if qa.task == "lah"
                else self.W_node(v_src_c[b, qa.dst_idx].float())
            ).to(torch.bfloat16)
            if len(p_pos) >= 1:
                pos = offset + p_pos[0].item()
                embeds[i, pos] = embeds[i, pos] + v_subj
            if len(p_pos) >= 2:
                pos = offset + p_pos[1].item()
                embeds[i, pos] = embeds[i, pos] + v_obj

        # 4. Prepend visual prefix per pair (if visual encoder enabled)
        if vis_prefix:
            L_vis = next(iter(vis_prefix.values())).shape[1]
            d = embeds.shape[-1]
            vis_stack = torch.stack([
                vis_prefix.get(
                    qa.batch_idx,
                    torch.zeros(1, L_vis, d, device=device, dtype=torch.bfloat16),
                ).squeeze(0)
                for qa in qa_pairs
            ]).to(device)                                  # (P, L_vis, d)
            embeds = torch.cat([vis_stack, embeds], dim=1)
            vis_mask = torch.ones(P, L_vis, dtype=torch.long, device=device)
            attention_mask = torch.cat([vis_mask, text_attn_mask], dim=1)
        else:
            attention_mask = text_attn_mask                # (P, max_text_len)

        # Causal-LM position_ids for left-padded sequences
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        # 5. Inject per-pair G_LLM into cross-attn layers
        batch_idx_t = torch.tensor([qa.batch_idx for qa in qa_pairs], device=device)
        self._set_G(G_LLM_bf16[batch_idx_t])              # (P, M, d_llm)

        # 6. Single batched LLM forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.llm(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        self._clear_G()

        # 7. Last-token logits → NLL loss + optional scores
        # With left-padding, position -1 is always the "Answer:" token for every row.
        logits_f = out.logits[:, -1, :].float()           # (P, vocab)
        log_probs = F.log_softmax(logits_f, dim=-1)
        target_ids = torch.tensor(
            [self._yes_id if qa.label == 1 else self._no_id for qa in qa_pairs],
            device=device,
        )
        loss = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).mean()
        score_tensor = (
            logits_f[:, self._yes_id] - logits_f[:, self._no_id]
            if return_scores
            else torch.zeros(0, device=device)
        )
        return loss, score_tensor

    @torch.no_grad()
    def score_pairs(
        self,
        E_c: Tensor,
        edge_valid: Tensor,
        v_src_c: Tensor,
        v_tgt_c: Tensor,
        qa_pairs: List[QAPair],
        vis_prefix: "Optional[dict[int, Tensor]]" = None,
    ) -> Tensor:
        """Inference: per-pair score = logit(Yes) - logit(No). Returns (len(qa_pairs),)."""
        if not qa_pairs:
            return torch.zeros(0, device=E_c.device)
        _, scores = self._loss_and_scores(
            E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs,
            vis_prefix=vis_prefix, return_scores=True,
        )
        return scores
