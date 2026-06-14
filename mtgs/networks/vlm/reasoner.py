# mtgs/networks/vlm/reasoner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

from mtgs.networks.vlm.graph_tokenizer import GraphEvidenceTokenizer
from mtgs.networks.vlm.memory_attn import MemoryCrossAttn, MemoryAugmentedLayer
from mtgs.datasets.gaze_qa import QAPair


# Qwen3-VL-8B (Qwen3-8B backbone): 36 layers, full_attention_interval=4
# GatedAttention (full attn) at every 4th layer starting from index 3
_DEFAULT_CROSS_ATTN_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35]


def _fmt_bbox(bbox: Tuple[float, float, float, float]) -> str:
    return "[{:.2f},{:.2f},{:.2f},{:.2f}]".format(*bbox)


class EvidenceAugmentedVLM(nn.Module):
    """Frozen Qwen3-VL-8B augmented with graph evidence via cross-attention.

    Entity grounding:
      - Text level: bbox [x1,y1,x2,y2] appended after each <P> token in prompt
      - Embedding level: Emb(<P>) += W_node · v  (v = v_src or v_tgt)

    Trainable: MemoryCrossAttn layers (9), W_node, <P> token embedding, Q_g, W_proj.
    Frozen: all other VLM parameters and MTGS pipeline.
    """

    def __init__(self, cfg):
        super().__init__()
        llm_cfg = cfg.vlm
        edge_dim = cfg.gaze_graph.edge_dim
        self._prompt_templates = dict(llm_cfg.prompt_templates)

        # ── Load tokenizer + VLM (text-only path) ─────────────────────────────
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            llm_cfg.backbone, trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_cfg.backbone,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in self.llm.parameters():
            p.requires_grad_(False)

        d_llm = self.llm.config.hidden_size  # e.g. 3584 for Qwen3-VL-8B

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

        # ── Wrap Full-Attention layers with MemoryCrossAttn ───────────────────
        indices = list(llm_cfg.get("cross_attn_layer_indices",
                                   _DEFAULT_CROSS_ATTN_INDICES))
        self._cross_attn_indices = indices
        for idx in indices:
            orig = self.llm.model.layers[idx]
            cross_attn = MemoryCrossAttn(d_llm, num_heads=8)
            self.llm.model.layers[idx] = MemoryAugmentedLayer(orig, cross_attn)

    # ── G_LLM injection helpers ───────────────────────────────────────────────

    def _set_G(self, G: Tensor):
        for idx in self._cross_attn_indices:
            self.llm.model.layers[idx]._G_LLM = G

    def _clear_G(self):
        for idx in self._cross_attn_indices:
            self.llm.model.layers[idx]._G_LLM = None

    # ── Query construction with bbox text + node feature grounding ────────────

    def _build_input_embeds(
        self,
        qa: QAPair,
        v_src_c: Tensor,   # (B, N, De)
        v_tgt_c: Tensor,   # (B, Tl, De)  Tl = N+2
        device: torch.device,
    ) -> Tensor:
        """Tokenize the bbox-augmented prompt and apply W_node grounding to <P> positions.

        Prompt format examples:
          LAH:  "Does <P> [0.10,0.20,0.30,0.40] look at <P> [0.50,0.60,0.70,0.80]? Answer:"
          LAEO: "Do <P> [..] and <P> [..] look at each other? Answer:"
          SA:   "Do <P> [..] and <P> [..] attend to the same target? Answer:"

        <P> embedding is further grounded: Emb(<P>) += W_node(v)
          subject: v_src[src_idx]
          object:  v_tgt[dst_idx]  ← LAH
                   v_src[dst_idx]  ← LAEO/SA

        Returns: (1, seq_len, d_llm)
        """
        src_bbox_str = _fmt_bbox(qa.src_bbox)
        dst_bbox_str = _fmt_bbox(qa.dst_bbox)

        prompt = self._prompt_templates[qa.task].format(
            src_bbox=src_bbox_str, dst_bbox=dst_bbox_str
        )
        input_ids = self.hf_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        emb_table = self.llm.get_input_embeddings()
        embeds = emb_table(input_ids).float()  # (1, seq_len, d_llm)

        P_id = self._P_token_id
        p_positions = (input_ids[0] == P_id).nonzero(as_tuple=True)[0]

        b = qa.batch_idx
        v_subj = self.W_node(v_src_c[b, qa.src_idx].float())
        if qa.task == "lah":
            v_obj = self.W_node(v_tgt_c[b, qa.dst_idx].float())
        else:
            v_obj = self.W_node(v_src_c[b, qa.dst_idx].float())

        if len(p_positions) >= 1:
            embeds[0, p_positions[0]] = embeds[0, p_positions[0]] + v_subj
        if len(p_positions) >= 2:
            embeds[0, p_positions[1]] = embeds[0, p_positions[1]] + v_obj

        return embeds

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        E_c: Tensor,           # (B, N, Tl, De)
        edge_valid: Tensor,    # (B, N, 2N+2)
        v_src_c: Tensor,       # (B, N, De)
        v_tgt_c: Tensor,       # (B, Tl, De)
        qa_pairs: List[QAPair],
    ) -> Tensor:
        if not qa_pairs:
            return torch.tensor(0.0, requires_grad=True,
                                device=E_c.device, dtype=torch.float32)

        device = E_c.device
        G_LLM = self.graph_tokenizer(E_c, edge_valid)       # (B, M, d_llm)
        G_LLM_bf16 = G_LLM.to(torch.bfloat16)

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for qa in qa_pairs:
            self._set_G(G_LLM_bf16[qa.batch_idx : qa.batch_idx + 1])
            embeds = self._build_input_embeds(qa, v_src_c, v_tgt_c, device)
            embeds_bf16 = embeds.to(torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = self.llm(inputs_embeds=embeds_bf16)

            logits = out.logits[0, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_id = self._yes_id if qa.label == 1 else self._no_id
            loss = -log_probs[target_id]
            total_loss = total_loss + loss
            count += 1

        self._clear_G()
        return total_loss / count
