"""Task-specific social decoders and graph-residual BCE objective.

The primary prediction contract is::

    final_logit = stop_gradient(graph_logit) + delta_task(h_social)

Each task owns a decoder, while the input prompt and six evidence slots retain one
shared schema. Decoder output layers are exactly zero-initialized, so an untrained
model is graph-equivalent. This makes ``delta_logits`` a correction, not a calibrated
standalone VLM probability; standalone evaluation is an explicit ablation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm.pair_dataset import SOCIAL_TASK_ID, SOCIAL_TASKS
from vlm.pair_features import PairGraphBatch
from vlm.pair_model import PairGenerativeVLM, PairSocialVLM, PairSocialVLMOutput


def _task_decoder(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    output = nn.Linear(hidden_dim, 1)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        output,
    )


def _validate_vector(tensor: torch.Tensor, batch: int, name: str) -> None:
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must have shape ({batch},), got {tuple(tensor.shape)}")


def _validate_task_ids(task_ids: torch.Tensor, batch: int) -> None:
    _validate_vector(task_ids, batch, "task_ids")
    if task_ids.dtype != torch.long:
        raise ValueError(f"task_ids must be torch.long, got {task_ids.dtype}")
    if not bool(torch.all((0 <= task_ids) & (task_ids < len(SOCIAL_TASKS)))):
        raise ValueError(f"task_ids must be in [0,{len(SOCIAL_TASKS)}), got {task_ids.tolist()}")


def answer_token_ids(tokenizer) -> tuple[int, int]:
    """Return existing single-token ids for the continuations ``' yes'``/``' no'``."""
    ids = []
    for answer in (" yes", " no"):
        encoded = tokenizer.encode(answer, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"LM auxiliary answer {answer!r} must be one existing token, got {encoded}"
            )
        ids.append(int(encoded[0]))
    if ids[0] == ids[1]:
        raise ValueError(f"yes/no answer tokens must be distinct, got {ids}")
    return ids[0], ids[1]


@dataclass
class PairDecoderOutput:
    logits: torch.Tensor             # [B], graph + selected delta
    delta_logits: torch.Tensor       # [B], selected task correction
    all_delta_logits: torch.Tensor   # [B,3], LAH/LAEO/SA decoder order
    graph_logits: torch.Tensor       # [B], detached residual base


class PairTaskResidualDecoder(nn.Module):
    """Route one post-norm social hidden state through three task-specific heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0,1), got {dropout}")
        self.input_dim = int(input_dim)
        self.decoders = nn.ModuleDict({
            task: _task_decoder(self.input_dim, hidden_dim, dropout)
            for task in SOCIAL_TASKS
        })

    def forward(
        self,
        h_social: torch.Tensor,
        task_ids: torch.Tensor,
        graph_logits: torch.Tensor,
    ) -> PairDecoderOutput:
        if h_social.ndim != 2 or h_social.shape[1] != self.input_dim:
            raise ValueError(
                f"h_social must have shape [B,{self.input_dim}], got {tuple(h_social.shape)}"
            )
        batch = h_social.shape[0]
        _validate_task_ids(task_ids, batch)
        _validate_vector(graph_logits, batch, "graph_logits")

        # Keep the small decoder in its configured dtype (normally FP32) even when
        # Qwen's final hidden state is BF16. The cast remains differentiable to Qwen.
        decoder_dtype = next(self.decoders.parameters()).dtype
        hidden = h_social.to(dtype=decoder_dtype)
        all_delta = torch.cat(
            [self.decoders[task](hidden) for task in SOCIAL_TASKS], dim=1
        )
        routed_ids = task_ids.to(device=all_delta.device)
        delta = all_delta.gather(1, routed_ids.unsqueeze(1)).squeeze(1)

        # Frozen graph evidence is the immutable base. Only the VLM correction learns.
        graph_base = graph_logits.detach().to(device=delta.device, dtype=delta.dtype)
        return PairDecoderOutput(
            logits=graph_base + delta,
            delta_logits=delta,
            all_delta_logits=all_delta,
            graph_logits=graph_base,
        )


class PairYesNoResidualHead(nn.Module):
    """Head driven by the FROZEN LM head's yes/no log-odds at ``h_social``.

    Instead of learning a fresh MLP over the hidden state, the score is read straight from
    the pretrained LM head's own belief that the answer token is ``" yes"`` vs ``" no"``::

        yesno = h_social · (W_lm[yes] - W_lm[no])        # log P(yes)/P(no); softmax cancels
        delta = scale_task * yesno + bias_task

    Two modes (``use_graph_residual``):
      * True  (default): ``final = stop_gradient(graph_logit) + delta`` — VLM corrects graph.
        With ``scale_init=0`` an untrained model is EXACTLY graph-equivalent.
      * False (standalone): ``final = delta`` — pure VLM yes/no, graph not used. Use
        ``scale_init=1`` so the prediction starts as the raw LM yes/no log-odds.

    Only ``scale``/``bias`` (2 params/task) learn here; gradient still flows through the
    frozen LM head into ``h_social`` (LoRA / projector / social_query). ``graph_logits`` is
    still returned (detached) for logging/analysis even in standalone mode.
    """

    def __init__(
        self,
        yes_token_id: int,
        no_token_id: int,
        *,
        use_graph_residual: bool = True,
        scale_init: float = 0.0,
    ):
        super().__init__()
        if yes_token_id < 0 or no_token_id < 0 or yes_token_id == no_token_id:
            raise ValueError(f"invalid yes/no ids: yes={yes_token_id}, no={no_token_id}")
        self.use_graph_residual = bool(use_graph_residual)
        self.register_buffer(
            "answer_ids", torch.tensor([no_token_id, yes_token_id], dtype=torch.long)
        )
        self.scale = nn.Parameter(torch.full((len(SOCIAL_TASKS),), float(scale_init)))
        self.bias = nn.Parameter(torch.zeros(len(SOCIAL_TASKS)))

    def forward(
        self,
        h_social: torch.Tensor,
        task_ids: torch.Tensor,
        graph_logits: torch.Tensor,
        lm_head: nn.Module,
    ) -> PairDecoderOutput:
        if h_social.ndim != 2:
            raise ValueError(f"h_social must be [B,H], got {tuple(h_social.shape)}")
        batch = h_social.shape[0]
        _validate_task_ids(task_ids, batch)
        _validate_vector(graph_logits, batch, "graph_logits")
        weight = getattr(lm_head, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise ValueError("lm_head must expose a 2D weight tensor")
        no_id, yes_id = int(self.answer_ids[0]), int(self.answer_ids[1])
        # Only the yes/no direction is needed; the full-vocab softmax normalisation
        # cancels in (yes_logit - no_logit), so two LM-head rows suffice. The LM head is a
        # FROZEN read-only semantic direction (detached), so no gradient reaches it.
        direction = (weight[yes_id] - weight[no_id]).detach().to(device=h_social.device).float()
        yesno = (h_social.float() * direction).sum(-1)                        # [B]
        task_ids = task_ids.to(yesno.device)
        scale = self.scale.to(yesno.device)
        bias = self.bias.to(yesno.device)
        all_delta = scale.unsqueeze(0) * yesno.unsqueeze(1) + bias.unsqueeze(0)  # [B,T]
        delta = all_delta.gather(1, task_ids.unsqueeze(1)).squeeze(1)            # [B]
        graph_base = graph_logits.detach().to(device=delta.device, dtype=delta.dtype)
        logits = graph_base + delta if self.use_graph_residual else delta
        return PairDecoderOutput(
            logits=logits,
            delta_logits=delta,
            all_delta_logits=all_delta,
            graph_logits=graph_base,
        )


class GraphLogitMLPControl(nn.Module):
    """Vision-free control: task MLP sees only the frozen scalar graph logit."""

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.decoder = PairTaskResidualDecoder(
            input_dim=1, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(
        self, graph_logits: torch.Tensor, task_ids: torch.Tensor
    ) -> PairDecoderOutput:
        if graph_logits.ndim != 1:
            raise ValueError(
                f"graph_logits must have shape [B], got {tuple(graph_logits.shape)}"
            )
        graph_base = graph_logits.detach()
        return self.decoder(graph_base.unsqueeze(1), task_ids, graph_base)


class GraphFeatureMLPControl(nn.Module):
    """Image-free control over flattened person/relation graph evidence.

    Heatmaps are optional because they are graph predictions but retain a spatial
    representation. When enabled, deterministic adaptive pooling bounds the MLP input
    size; no image encoder or learned convolution is introduced.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        *,
        include_heatmaps: bool = False,
        heatmap_pool_size: int = 8,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if heatmap_pool_size <= 0:
            raise ValueError(
                f"heatmap_pool_size must be positive, got {heatmap_pool_size}"
            )
        self.feature_dim = int(feature_dim)
        self.include_heatmaps = bool(include_heatmaps)
        self.heatmap_pool_size = int(heatmap_pool_size)
        # 2*3 person channels + 2 relations, followed by their 6+2 presence bits.
        input_dim = 8 * self.feature_dim + 8
        if self.include_heatmaps:
            input_dim += 2 * self.heatmap_pool_size**2 + 2
        self.input_dim = input_dim
        self.decoder = PairTaskResidualDecoder(
            input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def _flat_features(self, graph: PairGraphBatch) -> torch.Tensor:
        batch = len(graph.tasks)
        expected_shapes = {
            "person_features": (batch, 2, 3, self.feature_dim),
            "person_channel_present": (batch, 2, 3),
            "relation_features": (batch, 2, self.feature_dim),
            "relation_present": (batch, 2),
        }
        for name, expected in expected_shapes.items():
            actual = tuple(getattr(graph, name).shape)
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")

        person_mask = graph.person_channel_present.detach()
        relation_mask = graph.relation_present.detach()
        person = graph.person_features.detach() * person_mask.unsqueeze(-1)
        relation = graph.relation_features.detach() * relation_mask.unsqueeze(-1)
        pieces = [
            person.flatten(1),
            relation.flatten(1),
            person_mask.to(person.dtype).flatten(1),
            relation_mask.to(person.dtype).flatten(1),
        ]
        if self.include_heatmaps:
            expected_heatmap_prefix = (batch, 2)
            if graph.heatmap_features.ndim != 4 or tuple(
                graph.heatmap_features.shape[:2]
            ) != expected_heatmap_prefix:
                raise ValueError(
                    "heatmap_features must have shape [B,2,H,W], got "
                    f"{tuple(graph.heatmap_features.shape)}"
                )
            if tuple(graph.heatmap_present.shape) != expected_heatmap_prefix:
                raise ValueError(
                    f"heatmap_present must have shape {expected_heatmap_prefix}, got "
                    f"{tuple(graph.heatmap_present.shape)}"
                )
            heatmap_mask = graph.heatmap_present.detach()
            heatmaps = graph.heatmap_features.detach() * heatmap_mask[..., None, None]
            pooled = F.adaptive_avg_pool2d(
                heatmaps.reshape(batch * 2, 1, *heatmaps.shape[-2:]),
                self.heatmap_pool_size,
            ).reshape(batch, -1)
            pieces.extend((pooled, heatmap_mask.to(person.dtype)))
        features = torch.cat(pieces, dim=1)
        if features.shape != (batch, self.input_dim):
            raise RuntimeError(
                f"graph feature control produced {tuple(features.shape)}, expected "
                f"{(batch, self.input_dim)}"
            )
        return features

    def forward(
        self, graph: PairGraphBatch, task_ids: torch.Tensor
    ) -> PairDecoderOutput:
        batch = len(graph.tasks)
        _validate_task_ids(task_ids, batch)
        expected = torch.tensor(
            [SOCIAL_TASK_ID[task] for task in graph.tasks],
            dtype=torch.long,
            device=task_ids.device,
        )
        if not torch.equal(task_ids, expected):
            raise ValueError("task_ids do not match graph task order")
        decoder_device = next(self.decoder.parameters()).device
        # Graph caches/collation stay on CPU. Move only the compact flattened vector,
        # rather than the full optional [B,2,H,W] heatmaps, onto the accelerator.
        features = self._flat_features(graph).to(decoder_device)
        return self.decoder(features, task_ids, graph.graph_logits)


@dataclass
class PairBCELossOutput:
    loss: torch.Tensor
    per_sample: torch.Tensor


class PairTaskBCELoss(nn.Module):
    """Binary loss with optional per-task positive weights in LAH/LAEO/SA order."""

    def __init__(
        self,
        pos_weight: Mapping[str, float] | Sequence[float] | None = None,
    ):
        super().__init__()
        if pos_weight is None:
            values = [1.0] * len(SOCIAL_TASKS)
        elif isinstance(pos_weight, Mapping):
            unknown = set(pos_weight).difference(SOCIAL_TASKS)
            missing = set(SOCIAL_TASKS).difference(pos_weight)
            if unknown or missing:
                raise ValueError(
                    f"pos_weight keys must be exactly {SOCIAL_TASKS}; "
                    f"missing={sorted(missing)}, unknown={sorted(unknown)}"
                )
            values = [float(pos_weight[task]) for task in SOCIAL_TASKS]
        else:
            values = [float(value) for value in pos_weight]
            if len(values) != len(SOCIAL_TASKS):
                raise ValueError(
                    f"pos_weight must contain {len(SOCIAL_TASKS)} values, got {len(values)}"
                )
        weights = torch.tensor(values, dtype=torch.float32)
        if not bool(torch.all(torch.isfinite(weights) & (weights > 0))):
            raise ValueError(f"pos_weight values must be finite and positive, got {values}")
        self.register_buffer("pos_weight", weights)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> PairBCELossOutput:
        batch = logits.shape[0] if logits.ndim == 1 else -1
        _validate_vector(logits, batch, "logits")
        _validate_vector(labels, batch, "labels")
        _validate_task_ids(task_ids, batch)
        if not bool(torch.all((labels == 0) | (labels == 1))):
            raise ValueError(f"labels must be exactly binary, got {labels.tolist()}")

        logits_fp32 = logits.float()
        labels_fp32 = labels.to(device=logits.device, dtype=torch.float32)
        task_ids = task_ids.to(device=logits.device)
        positive_weights = self.pos_weight.to(logits.device).index_select(0, task_ids)
        sample_weights = torch.where(
            labels_fp32.bool(), positive_weights, torch.ones_like(positive_weights)
        )
        per_sample = F.binary_cross_entropy_with_logits(
            logits_fp32, labels_fp32, reduction="none"
        ) * sample_weights
        return PairBCELossOutput(loss=per_sample.mean(), per_sample=per_sample)


@dataclass
class PairNextTokenLossOutput:
    loss: torch.Tensor
    per_sample: torch.Tensor
    accuracy: torch.Tensor


class PairNextTokenLoss(nn.Module):
    """Full-vocabulary yes/no next-token loss evaluated only at ``h_social``."""

    def __init__(self, yes_token_id: int, no_token_id: int):
        super().__init__()
        if yes_token_id < 0 or no_token_id < 0 or yes_token_id == no_token_id:
            raise ValueError(
                f"invalid yes/no token ids: yes={yes_token_id}, no={no_token_id}"
            )
        self.register_buffer(
            "answer_ids", torch.tensor([no_token_id, yes_token_id], dtype=torch.long)
        )

    def forward(
        self,
        h_social: torch.Tensor,
        labels: torch.Tensor,
        output_embeddings: nn.Module,
    ) -> PairNextTokenLossOutput:
        if h_social.ndim != 2:
            raise ValueError(f"h_social must be 2D, got {tuple(h_social.shape)}")
        batch = h_social.shape[0]
        _validate_vector(labels, batch, "labels")
        if not bool(torch.all((labels == 0) | (labels == 1))):
            raise ValueError(f"labels must be exactly binary, got {labels.tolist()}")
        weight = getattr(output_embeddings, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise ValueError("output_embeddings must expose a 2D weight tensor")
        hidden = h_social.to(device=weight.device, dtype=weight.dtype)
        vocabulary_logits = output_embeddings(hidden).float()
        answer_ids = self.answer_ids.to(vocabulary_logits.device)
        targets = answer_ids.index_select(
            0, labels.to(vocabulary_logits.device, torch.long)
        )
        per_sample = F.cross_entropy(
            vocabulary_logits, targets, reduction="none"
        )
        accuracy = vocabulary_logits.argmax(dim=-1).eq(targets).float().mean()
        return PairNextTokenLossOutput(
            loss=per_sample.mean(), per_sample=per_sample, accuracy=accuracy
        )


@dataclass
class PairSocialObjectiveOutput:
    vlm: PairSocialVLMOutput
    decoder: PairDecoderOutput
    loss: torch.Tensor | None = None
    per_sample_loss: torch.Tensor | None = None
    residual_loss: torch.Tensor | None = None
    lm_aux_loss: torch.Tensor | None = None
    lm_aux_accuracy: torch.Tensor | None = None


class PairSocialObjective(nn.Module):
    """End-to-end Unit-5 composition; optimizer/dataloader policy remains Unit 6."""

    def __init__(
        self,
        vlm: PairSocialVLM,
        decoder: PairTaskResidualDecoder,
        criterion: PairTaskBCELoss | None = None,
        *,
        lm_aux_weight: float = 0.0,
        lm_auxiliary: PairNextTokenLoss | None = None,
    ):
        super().__init__()
        if lm_aux_weight < 0:
            raise ValueError(f"lm_aux_weight must be non-negative, got {lm_aux_weight}")
        if lm_aux_weight > 0 and lm_auxiliary is None:
            raise ValueError("positive lm_aux_weight requires PairNextTokenLoss")
        self.vlm = vlm
        self.decoder = decoder
        self.criterion = criterion if criterion is not None else PairTaskBCELoss()
        self.lm_aux_weight = float(lm_aux_weight)
        self.lm_auxiliary = lm_auxiliary

    def close(self) -> None:
        self.vlm.close()

    def forward(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        graph: PairGraphBatch,
        task_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> PairSocialObjectiveOutput:
        batch = len(graph.tasks)
        _validate_task_ids(task_ids, batch)
        expected = torch.tensor(
            [SOCIAL_TASK_ID[task] for task in graph.tasks],
            dtype=torch.long,
            device=task_ids.device,
        )
        if not torch.equal(task_ids, expected):
            raise ValueError(
                f"task_ids do not match graph task order: ids={task_ids.tolist()}, "
                f"tasks={graph.tasks}"
            )

        vlm_output = self.vlm(model_inputs, graph)
        # The yes/no head reads the frozen LM head's yes/no log-odds at h_social.
        decoder_output = self.decoder(
            vlm_output.h_social, task_ids, graph.graph_logits,
            self.vlm.get_output_embeddings(),
        )
        if labels is None:
            return PairSocialObjectiveOutput(vlm=vlm_output, decoder=decoder_output)
        loss_output = self.criterion(decoder_output.logits, labels, task_ids)
        lm_output = None
        total_loss = loss_output.loss
        if self.lm_aux_weight > 0:
            assert self.lm_auxiliary is not None
            lm_output = self.lm_auxiliary(
                vlm_output.h_social,
                labels,
                self.vlm.get_output_embeddings(),
            )
            total_loss = total_loss + self.lm_aux_weight * lm_output.loss
        return PairSocialObjectiveOutput(
            vlm=vlm_output,
            decoder=decoder_output,
            loss=total_loss,
            per_sample_loss=loss_output.per_sample,
            residual_loss=loss_output.loss,
            lm_aux_loss=None if lm_output is None else lm_output.loss,
            lm_aux_accuracy=None if lm_output is None else lm_output.accuracy,
        )


@dataclass
class PairGenerativeOutput:
    loss: torch.Tensor | None = None
    prob: torch.Tensor | None = None      # [B] eval probability (candidate scoring)


def answer_loglik(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence answer log-likelihood: sum log P(token) over positions with a label.

    ``logits`` [N,L,V], ``labels`` [N,L] with -100 on masked (prompt/pad) positions.
    Uses the standard next-token shift, so position i's logits score token i+1.
    """
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    logp = torch.log_softmax(shifted_logits, dim=-1)
    mask = (shifted_labels != -100)
    safe = shifted_labels.clamp_min(0)
    token_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)   # [N, L-1]
    return (token_logp * mask).sum(dim=1)                          # [N]


class PairGenerativeObjective(nn.Module):
    """EyeVLM-style generative objective: the frozen+LoRA LM is SFT'd to generate a JSON
    binary label ``[{"label": 1/0}]`` after a prompt carrying text bbox coords  + graph evidence soft-tokens (our contribution).

    * train (``forward``): next-token CE on the answer JSON (prompt masked in ``labels``).
    * eval  (``score``): teacher-force both JSON candidates and read the model's preference
      ``sigmoid(LL(positive) - LL(negative))`` -> [0,1] for the locked AP/AUC/F1 harness.
    """

    def __init__(self, vlm: PairGenerativeVLM):
        super().__init__()
        self.vlm = vlm

    def close(self) -> None:
        self.vlm.close()

    def forward(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        task_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> PairGenerativeOutput:
        out = self.vlm(model_inputs)                 # backbone computes CE loss from labels
        return PairGenerativeOutput(loss=out.loss)

    @torch.no_grad()
    def score(self, model_inputs: Mapping[str, torch.Tensor], num_pairs: int) -> torch.Tensor:
        """Return P(positive) per pair from the [2B] positive/negative candidate batch."""
        out = self.vlm(model_inputs)
        ll = answer_loglik(out.logits, model_inputs["labels"].to(out.logits.device))  # [2B]
        return torch.sigmoid(ll[:num_pairs] - ll[num_pairs:])                          # [B]
