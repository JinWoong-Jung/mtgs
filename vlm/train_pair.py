"""Pair-wise MTGS+Qwen training and the vision-free graph-logit control.

Unit 6 intentionally stops at train/validation loss and checkpointing. Reconstructing
the locked VSGaze prediction dictionaries and AP/AUC/F1 evaluation belongs to Unit 7.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import (
    DataLoader,
    Sampler,
    SequentialSampler,
    Subset,
    WeightedRandomSampler,
)
from tqdm import tqdm

from vlm.cfg import QWEN
from vlm.pair_head import (
    GraphFeatureMLPControl,
    GraphLogitMLPControl,
    PairGenerativeObjective,
    PairSocialObjective,
    PairTaskBCELoss,
    PairYesNoResidualHead,
    answer_token_ids,
)
from vlm.pair_input import (
    GraphControlDataset,
    GraphFeatureControlDataset,
    PairInputDataset,
    pair_control_collate,
    pair_feature_control_collate,
    pair_pos_weights,
    partition_by_graph_confidence,
    sample_graph_logit,
)
from vlm.pair_eval import (
    PairPredictionCollector,
    evaluate_pair_predictions,
    format_pair_metrics,
    metric_payload,
    raw_graph_predictions,
)
from vlm.pair_model import (
    PairGenerativeVLM,
    PairSocialVLM,
    TextGenerativeVLM,
    make_generative_collate,
    make_generative_eval_collate,
    make_pair_collate,
    make_text_generative_collate,
    make_text_generative_eval_collate,
    prepare_pair_tokens,
)
from vlm.patches import patch_qwen3vl_patch_embed


LORA_PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)
TRAIN_MODES = ("vlm", "graph_mlp", "features_mlp")


@dataclass
class EpochStats:
    loss: float
    residual_loss: float
    accuracy: float
    examples: int
    lm_aux_loss: float | None = None
    lm_aux_accuracy: float | None = None


def format_epoch_report(
    *,
    epoch: int,
    epochs: int,
    train_stats: EpochStats,
    val_stats: EpochStats | None,
    val_metrics: Mapping[str, object] | None,
    selection_name: str,
    selection_score: float,
    best_score: float | None,
    improved: bool,
) -> str:
    """Compact, metrics-only block written once after a completed epoch."""

    def stats_line(split: str, stats: EpochStats) -> str:
        values = (
            f"loss={stats.loss:.6f} residual_loss={stats.residual_loss:.6f} "
            f"accuracy={stats.accuracy:.6f} examples={stats.examples}"
        )
        if stats.lm_aux_loss is not None:
            values += (
                f" lm_aux_loss={stats.lm_aux_loss:.6f} "
                f"lm_aux_accuracy={stats.lm_aux_accuracy:.6f}"
            )
        return f"  {split}: {values}"

    lines = [f"[epoch {epoch + 1}/{epochs}]", stats_line("train", train_stats)]
    if val_stats is not None:
        lines.append(stats_line("val", val_stats))
    if val_metrics is not None:
        lines.append(format_pair_metrics(val_metrics, "validation"))
    best = "N/A" if best_score is None else f"{best_score:.6f}"
    lines.append(
        f"  selection: {selection_name}={selection_score:.6f} "
        f"best={best} improved={improved}"
    )
    return "\n".join(lines)


def append_epoch_report(path: str | Path | None, report: str) -> None:
    """Append one epoch block to the Slurm metrics file when configured."""
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(report.rstrip() + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_steps_per_epoch(num_samples: int, batch_size: int, accumulation: int) -> int:
    if min(num_samples, batch_size, accumulation) <= 0:
        raise ValueError("num_samples, batch_size and accumulation must be positive")
    return math.ceil(math.ceil(num_samples / batch_size) / accumulation)


class FrameGroupedBatchSampler(Sampler[list[int]]):
    """Keep sampled rows from the same frame contiguous for frozen-vision reuse.

    The wrapped sampler still decides the exact sample multiset and therefore preserves
    task/hardness weighting. This class only reorders those indices before batching.
    """

    def __init__(
        self,
        sampler: Iterable[int],
        frame_ids: list[str],
        batch_size: int,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not hasattr(sampler, "__len__"):
            raise ValueError("frame-grouped sampler requires a finite base sampler")
        self.sampler = sampler
        self.frame_ids = frame_ids
        self.batch_size = int(batch_size)

    def __iter__(self):
        buckets: OrderedDict[str, list[int]] = OrderedDict()
        for raw_index in self.sampler:
            index = int(raw_index)
            if not 0 <= index < len(self.frame_ids):
                raise IndexError(f"sample index {index} is outside frame-id table")
            buckets.setdefault(self.frame_ids[index], []).append(index)
        ordered = [index for bucket in buckets.values() for index in bucket]
        for start in range(0, len(ordered), self.batch_size):
            yield ordered[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)


def _dataset_frame_ids(dataset) -> list[str]:
    annotations = getattr(dataset, "annotations", None)
    samples = getattr(annotations, "samples", None)
    if samples is None or len(samples) != len(dataset):
        raise ValueError("frame grouping requires a pair dataset with annotations")
    return [sample.sid for sample in samples]


def make_epoch_loader(
    dataset,
    collate_fn,
    weights: torch.Tensor,
    *,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
    pin_memory: bool | None = None,
    group_by_frame: bool = False,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed + epoch)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    common = {
        "dataset": dataset,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        # A new epoch loader gets a new deterministic sampler; do not leave the old
        # loader's worker pool alive between epochs.
        "persistent_workers": False,
    }
    if group_by_frame:
        return DataLoader(
            **common,
            batch_sampler=FrameGroupedBatchSampler(
                sampler, _dataset_frame_ids(dataset), batch_size
            ),
        )
    return DataLoader(**common, batch_size=batch_size, sampler=sampler)


def make_validation_loader(
    dataset,
    collate_fn,
    batch_size: int,
    num_workers: int,
    *,
    pin_memory: bool | None = None,
    group_by_frame: bool = False,
) -> DataLoader:
    common = {
        "dataset": dataset,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if group_by_frame:
        return DataLoader(
            **common,
            batch_sampler=FrameGroupedBatchSampler(
                SequentialSampler(dataset), _dataset_frame_ids(dataset), batch_size
            ),
        )
    return DataLoader(**common, batch_size=batch_size, shuffle=False)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _trainable(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def partition_vlm_parameters(objective: PairSocialObjective):
    """Return disjoint LoRA/new-module groups and validate the frozen-base contract."""
    lora_named = [
        (name, parameter)
        for name, parameter in objective.vlm.backbone.named_parameters()
        if parameter.requires_grad
    ]
    if not lora_named:
        raise ValueError("Qwen backbone has no trainable LoRA parameters")
    invalid = [name for name, _ in lora_named if "lora_" not in name]
    if invalid:
        raise ValueError(f"non-LoRA Qwen parameters are trainable: {invalid[:10]}")

    lora = [parameter for _, parameter in lora_named]
    if isinstance(objective, PairGenerativeObjective):
        if isinstance(objective.vlm, TextGenerativeVLM):
            # text mode: graph evidence is already text in the prompt; only LoRA trains.
            new = []
        else:
            # gtoken generative: only the graph projector learns besides LoRA.
            new = list(objective.vlm.projector.parameters())
    else:
        new = (
            list(objective.vlm.projector.parameters())
            + [objective.vlm.social_query]
            + list(objective.decoder.parameters())
        )
    new = _trainable(new)
    overlap = {id(parameter) for parameter in lora}.intersection(
        id(parameter) for parameter in new
    )
    if overlap:
        raise ValueError("LoRA and new-module optimizer groups overlap")
    return lora, new


def _scheduler(optimizer, name: str, warmup_steps: int, total_steps: int):
    from transformers import get_scheduler

    name = "constant" if name.lower() == "none" else name.lower()
    return get_scheduler(
        name,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def _load_graph_cache(path: str | Path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache, Mapping):
        raise ValueError(f"graph cache must be a mapping, got {type(cache).__name__}")
    return cache


def _resolve_pos_weights(cfg, annotations) -> dict[str, float]:
    configured = cfg.loss.get("pos_weight", "auto")
    if str(configured).lower() == "auto":
        return pair_pos_weights(
            annotations,
            minimum=float(cfg.loss.get("pos_weight_min", 0.2)),
            maximum=float(cfg.loss.get("pos_weight_max", 5.0)),
        )
    values = OmegaConf.to_container(configured, resolve=True)
    if not isinstance(values, Mapping):
        raise ValueError("loss.pos_weight must be 'auto' or a task mapping")
    return {str(key): float(value) for key, value in values.items()}


def _hard_floor(cfg) -> float | None:
    if not bool(cfg.sampler.get("hard_weight", False)):
        return None
    return float(cfg.sampler.get("hard_floor", 0.25))


def _route_threshold(cfg) -> float | None:
    """Confidence-gated routing threshold (max(p,1-p) scale), or None when disabled."""
    routing = cfg.get("routing", {})
    if not bool(routing.get("use", False)):
        return None
    return float(routing.get("threshold", 0.9))


def validate_sampler_loss_compatibility(
    balance_mode: str, pos_weights: Mapping[str, float]
) -> None:
    """Ensure label imbalance is corrected by sampling or BCE, never both."""
    non_unit = {
        task: float(value)
        for task, value in pos_weights.items()
        if not math.isclose(float(value), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    }
    if balance_mode == "task_label" and non_unit:
        raise ValueError(
            "sampler.balance_mode='task_label' already balances labels; set every "
            f"loss.pos_weight to 1.0 or use balance_mode='task'. Non-unit values: {non_unit}"
        )


def checkpoint_score(
    val_stats: EpochStats | None,
    val_metrics: Mapping[str, object] | None,
    *,
    monitor: str,
    monitor_mode: str,
) -> tuple[str, str, float]:
    """Return the authoritative selection name/mode/value for one epoch."""
    if val_metrics is not None:
        if monitor_mode not in ("max", "min"):
            raise ValueError(f"experiment.monitor_mode must be max/min, got {monitor_mode!r}")
        value = val_metrics.get(monitor)
        if value is None:
            raise ValueError(
                f"validation metric {monitor!r} is unavailable; available="
                f"{sorted(key for key, item in val_metrics.items() if item is not None)}"
            )
        return monitor, monitor_mode, float(value)
    if val_stats is not None:
        return "val_loss", "min", float(val_stats.loss)
    raise ValueError("checkpoint selection requires validation stats or metrics")


def score_improved(value: float, best: float | None, mode: str) -> bool:
    if mode not in ("max", "min"):
        raise ValueError(f"score mode must be max/min, got {mode!r}")
    return best is None or (value > best if mode == "max" else value < best)


def _make_lora_backbone(cfg, processor, device: torch.device, resume: Path | None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration

    model_name = str(cfg.model.get("qwen", QWEN))
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=str(device),
        attn_implementation="sdpa",   # fast attention on Blackwell (avoid eager fallback)
    )
    token_ids = prepare_pair_tokens(processor.tokenizer, base)
    patch_qwen3vl_patch_embed(base)
    base.requires_grad_(False)

    targets = set(cfg.model.lora.get("targets", LORA_PROJECTIONS))
    unknown = targets.difference(LORA_PROJECTIONS)
    if unknown:
        raise ValueError(f"unsupported LoRA projections: {sorted(unknown)}")
    target_names = [
        name
        for name, _ in base.named_modules()
        if "language_model" in name and name.rsplit(".", 1)[-1] in targets
    ]
    if not target_names:
        raise ValueError(f"no Qwen language-model LoRA targets found for {sorted(targets)}")

    adapter_dir = None if resume is None else resume / "adapter"
    if adapter_dir is not None and adapter_dir.exists():
        backbone = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    else:
        rank = int(cfg.model.lora.rank)
        alpha = int(cfg.model.lora.get("alpha", 2 * rank))
        backbone = get_peft_model(
            base,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=float(cfg.model.lora.get("dropout", 0.05)),
                target_modules=target_names,
                task_type="CAUSAL_LM",
            ),
        )
    backbone.config.use_cache = False
    if bool(cfg.model.get("gradient_checkpointing", True)):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    backbone.enable_input_require_grads()
    return backbone, token_ids, target_names


def build_vlm_objective(cfg, processor, device: torch.device, resume: Path | None = None):
    backbone, token_ids, target_names = _make_lora_backbone(
        cfg, processor, device, resume
    )
    vlm = PairSocialVLM(
        backbone,
        token_ids,
        graph_dim=int(cfg.model.get("graph_dim", 256)),
        graph_hidden_dim=int(cfg.model.get("graph_hidden_dim", 1024)),
        heatmap_conv_dim=int(cfg.model.get("heatmap_conv_dim", 128)),
        vision_cache_size=int(
            cfg.get("input", {}).get("vision_cache_size", 0)
        ),
    )
    # Primary head: the frozen LM head's " yes"/" no" log-odds at h_social.
    #   graph_residual=true  -> final = graph_logit + yes/no correction (VLM refines graph)
    #   graph_residual=false -> final = yes/no only (pure VLM standalone; scale starts at 1)
    yes_id, no_id = answer_token_ids(processor.tokenizer)
    graph_residual = bool(cfg.model.get("graph_residual", True))
    decoder = PairYesNoResidualHead(
        yes_id,
        no_id,
        use_graph_residual=graph_residual,
        scale_init=0.0 if graph_residual else 1.0,
    ).to(device=device)
    return vlm, decoder, None, target_names


@dataclass(frozen=True)
class GenerativeBuilders:
    """Which generative collate/dataset wiring a run resolves to, keyed off
    ``model.graph_evidence``."""

    graph_evidence: str
    uses_text_collate: bool


def select_generative_builders(cfg) -> GenerativeBuilders:
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else cfg["model"]
    evidence = str(model_cfg.get("graph_evidence", "gtoken"))
    if evidence not in ("gtoken", "text"):
        raise ValueError(f"model.graph_evidence must be gtoken/text, got {evidence!r}")
    return GenerativeBuilders(graph_evidence=evidence, uses_text_collate=evidence == "text")


def build_generative_objective(cfg, processor, device: torch.device, resume: Path | None = None):
    """EyeVLM-style generative objective: LM generates yes/no.

    gtoken mode: graph evidence is injected as input soft-tokens (``PairGenerativeVLM`` +
    projector). text mode: graph evidence is already natural-language text in the prompt, so
    the backbone is wrapped bare (``TextGenerativeVLM``, no projector).
    """
    backbone, token_ids, target_names = _make_lora_backbone(cfg, processor, device, resume)
    if select_generative_builders(cfg).uses_text_collate:
        vlm = TextGenerativeVLM(backbone)
    else:
        vlm = PairGenerativeVLM(
            backbone,
            token_ids,
            graph_dim=int(cfg.model.get("graph_dim", 256)),
            graph_hidden_dim=int(cfg.model.get("graph_hidden_dim", 1024)),
            heatmap_conv_dim=int(cfg.model.get("heatmap_conv_dim", 128)),
        )
    objective = PairGenerativeObjective(vlm).to(device=device)
    return objective, target_names


def _restore_vlm_modules(objective: PairSocialObjective, checkpoint: Path) -> dict[str, Any]:
    modules_path = checkpoint / "pair_modules.pt"
    trainer_path = checkpoint / "trainer_state.pt"
    if not modules_path.exists() or not trainer_path.exists():
        raise FileNotFoundError(f"incomplete pair checkpoint: {checkpoint}")
    modules = torch.load(modules_path, map_location="cpu", weights_only=False)
    if "projector" in modules:
        objective.vlm.projector.load_state_dict(modules["projector"], strict=True)
    if not isinstance(objective, PairGenerativeObjective):
        with torch.no_grad():
            objective.vlm.social_query.copy_(
                modules["social_query"].to(
                    objective.vlm.social_query.device, objective.vlm.social_query.dtype
                )
            )
        objective.decoder.load_state_dict(modules["decoder"], strict=True)
        objective.criterion.load_state_dict(modules["criterion"], strict=True)
        if objective.lm_auxiliary is not None and modules.get("lm_auxiliary") is not None:
            objective.lm_auxiliary.load_state_dict(modules["lm_auxiliary"], strict=True)
    return torch.load(trainer_path, map_location="cpu", weights_only=False)


def save_vlm_checkpoint(
    path: Path,
    objective: PairSocialObjective,
    processor,
    optimizer,
    scheduler,
    state: Mapping[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    objective.vlm.backbone.save_pretrained(path / "adapter")
    processor.save_pretrained(path / "processor")
    if isinstance(objective, PairGenerativeObjective):
        if isinstance(objective.vlm, TextGenerativeVLM):
            # text mode: graph evidence is text in the prompt; no projector to save.
            modules = {"generative": True, "text_evidence": True}
        else:
            # gtoken generative: only the graph projector learns besides LoRA (no
            # social_query/decoder).
            modules = {"generative": True, "projector": _cpu_state_dict(objective.vlm.projector)}
    else:
        modules = {
            "projector": _cpu_state_dict(objective.vlm.projector),
            "social_query": objective.vlm.social_query.detach().cpu(),
            "decoder": _cpu_state_dict(objective.decoder),
            "criterion": _cpu_state_dict(objective.criterion),
            "lm_auxiliary": None
            if objective.lm_auxiliary is None
            else _cpu_state_dict(objective.lm_auxiliary),
        }
    torch.save(modules, path / "pair_modules.pt")
    torch.save(
        {
            **dict(state),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        path / "trainer_state.pt",
    )


def save_control_checkpoint(
    path: Path,
    control: torch.nn.Module,
    criterion: PairTaskBCELoss,
    optimizer,
    scheduler,
    state: Mapping[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **dict(state),
            "control": _cpu_state_dict(control),
            "criterion": _cpu_state_dict(criterion),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        path / "control.pt",
    )


def restore_control_checkpoint(
    path: Path, control, criterion, optimizer=None, scheduler=None
) -> dict[str, Any]:
    state = torch.load(path / "control.pt", map_location="cpu", weights_only=False)
    control.load_state_dict(state["control"], strict=True)
    criterion.load_state_dict(state["criterion"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def _vlm_batch(objective: PairSocialObjective, batch: dict[str, Any]):
    batch = dict(batch)
    graph = batch.pop("pair_graph")
    task_ids = batch.pop("task_ids")
    labels = batch.pop("pair_labels")
    batch.pop("eval_keys")
    return objective(batch, graph, task_ids, labels), labels


def _generative_batch(objective: PairGenerativeObjective, batch: dict[str, Any]):
    """EyeVLM-style JSON-SFT step: the backbone computes next-token CE from the token-level
    ``labels`` (answer JSON only). Prediction metrics come from ``objective.score`` on the
    eval collate, not from this teacher-forced pass, so a placeholder prediction is used."""
    batch = dict(batch)
    task_ids = batch.pop("task_ids")
    labels = batch.pop("pair_labels")          # [B] binary GT (for logging only here)
    batch.pop("eval_keys")
    # batch still carries token-level "labels" + graph_features/graph_present for the vlm.
    out = objective(batch, task_ids)
    zeros = torch.zeros(labels.numel(), device=labels.device)
    decoder = type("GenDecoder", (), {
        "logits": zeros, "graph_logits": None, "delta_logits": None,
    })()
    shim = type("GenBatchOutput", (), {
        "loss": out.loss, "residual_loss": out.loss,
        "lm_aux_loss": None, "lm_aux_accuracy": None, "decoder": decoder,
    })()
    return shim, labels


def _control_batch(control, criterion, batch: dict[str, Any], device: torch.device):
    task_ids = batch["task_ids"].to(device)
    labels = batch["pair_labels"].to(device)
    if "pair_graph" in batch:
        decoder = control(batch["pair_graph"], task_ids)
    else:
        decoder = control(batch["graph_logits"].to(device), task_ids)
    loss = criterion(decoder.logits, labels, task_ids)
    output = type("ControlBatchOutput", (), {
        "loss": loss.loss,
        "residual_loss": loss.loss,
        "lm_aux_loss": None,
        "lm_aux_accuracy": None,
        "decoder": decoder,
    })()
    return output, labels


def collect_generative_predictions(
    objective: PairGenerativeObjective,
    dataset,
    processor,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    description: str,
    route_threshold: float | None = None,
) -> PairPredictionCollector:
    """Fill a prediction collector with EyeVLM candidate-scoring probabilities: each pair's
    two JSON candidates are teacher-forced and P(positive)=sigmoid(LL_pos-LL_neg).

    With ``route_threshold`` set, confidence-gated routing answers high-confidence pairs with
    the frozen graph logit directly (no VLM forward) and queries the VLM only on the
    low-confidence remainder, cutting evaluation cost by the high-confidence fraction.
    """
    collector = PairPredictionCollector()
    eval_dataset = dataset
    uses_text_collate = getattr(dataset, "graph_evidence", "gtoken") == "text"
    if route_threshold is not None:
        high, low = partition_by_graph_confidence(
            dataset.annotations, dataset.graph_cache, route_threshold
        )
        if high:
            samples = [dataset.annotations[i] for i in high]
            graph_logits = torch.stack(
                [sample_graph_logit(s, dataset.graph_cache[s.sid]) for s in samples]
            ).float()
            labels = torch.tensor([s.label for s in samples], dtype=torch.float32)
            collector.add_batch([s.eval_key for s in samples], graph_logits, labels)
        print(
            f"[route] {description}: graph={len(high)} vlm={len(low)} "
            f"(vlm {len(low) / max(len(dataset), 1):.1%})",
            flush=True,
        )
        if not low:
            return collector
        eval_dataset = Subset(dataset, low)
    eval_collate = (
        make_text_generative_eval_collate(processor)
        if uses_text_collate
        else make_generative_eval_collate(processor)
    )
    loader = make_validation_loader(
        eval_dataset, eval_collate, batch_size, num_workers,
        pin_memory=device.type == "cuda",
    )
    was_training = objective.training
    objective.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=description, leave=False):
            num_pairs = int(batch["num_pairs"])
            model_inputs = {
                k: v for k, v in batch.items()
                if k not in ("pair_labels", "eval_keys", "num_pairs")
            }
            prob = objective.score(model_inputs, num_pairs).clamp(1e-6, 1 - 1e-6)
            logit = torch.log(prob / (1 - prob))          # collector sigmoids back to prob
            collector.add_batch(batch["eval_keys"], logit, batch["pair_labels"])
    objective.train(was_training)
    return collector


def run_epoch(
    module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: PairTaskBCELoss | None = None,
    optimizer=None,
    scheduler=None,
    accumulation: int = 1,
    grad_clip: float = 0.0,
    description: str = "epoch",
    prediction_collector: PairPredictionCollector | None = None,
    batch_log_interval: int = 0,
    batch_logger=None,
) -> EpochStats:
    if batch_log_interval < 0:
        raise ValueError("batch_log_interval must be non-negative")
    if batch_logger is not None and not callable(batch_logger):
        raise TypeError("batch_logger must be callable")
    training = optimizer is not None
    module.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    count = correct = 0
    loss_sum = residual_sum = lm_loss_sum = lm_acc_sum = 0.0
    lm_count = 0
    batches = len(loader)
    optimizer_steps = 0
    iterator = tqdm(loader, desc=description, leave=False)

    for index, batch in enumerate(iterator):
        group_start = (index // accumulation) * accumulation
        group_size = min(accumulation, batches - group_start)
        with torch.set_grad_enabled(training):
            if isinstance(module, PairSocialObjective):
                output, labels = _vlm_batch(module, batch)
            elif isinstance(module, PairGenerativeObjective):
                output, labels = _generative_batch(module, batch)
            else:
                if criterion is None:
                    raise ValueError("graph control requires a BCE criterion")
                output, labels = _control_batch(module, criterion, batch, device)
            if output.loss is None:
                raise RuntimeError("training objective did not return loss")
            if training:
                (output.loss / group_size).backward()

        batch_size = labels.numel()
        count += batch_size
        loss_sum += float(output.loss.detach()) * batch_size
        residual_sum += float(output.residual_loss.detach()) * batch_size
        predictions = output.decoder.logits.detach().gt(0).to(labels.device)
        correct += int(predictions.eq(labels.bool()).sum())
        if prediction_collector is not None:
            eval_keys = batch.get("eval_keys")
            if not isinstance(eval_keys, list) or len(eval_keys) != batch_size:
                raise ValueError("prediction collection requires one eval_key per sample")
            prediction_collector.add_batch(
                eval_keys,
                output.decoder.logits,
                labels,
                graph_logits=output.decoder.graph_logits,
                delta_logits=output.decoder.delta_logits,
            )
        if output.lm_aux_loss is not None:
            lm_count += batch_size
            lm_loss_sum += float(output.lm_aux_loss.detach()) * batch_size
            lm_acc_sum += float(output.lm_aux_accuracy.detach()) * batch_size

        group_end = (index + 1) % accumulation == 0 or index + 1 == batches
        if training and group_end:
            trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if batch_logger is not None and batch_log_interval and (
                optimizer_steps % batch_log_interval == 0 or index + 1 == batches
            ):
                batch_logger(optimizer_steps, {
                    "batch_loss": float(output.loss.detach()),
                    "running_loss": loss_sum / max(count, 1),
                    "examples": int(count),
                })
        iterator.set_postfix(loss=f"{loss_sum / max(count, 1):.4f}")

    return EpochStats(
        loss=loss_sum / max(count, 1),
        residual_loss=residual_sum / max(count, 1),
        accuracy=correct / max(count, 1),
        examples=count,
        lm_aux_loss=None if lm_count == 0 else lm_loss_sum / lm_count,
        lm_aux_accuracy=None if lm_count == 0 else lm_acc_sum / lm_count,
    )


def _processor(cfg, resume: Path | None):
    from transformers import AutoProcessor

    saved = None if resume is None else resume / "processor"
    source = saved if saved is not None and saved.exists() else str(cfg.model.get("qwen", QWEN))
    # Cap the visual-token budget (EyeVLM uses 200704 = 448x448, aspect ratio preserved).
    # Fewer image tokens -> shorter LM sequence -> faster train/val/test.
    kwargs = {}
    max_pixels = cfg.model.get("max_pixels", None)
    if max_pixels:
        kwargs["max_pixels"] = int(max_pixels)
        min_pixels = cfg.model.get("min_pixels", None)
        if min_pixels:
            kwargs["min_pixels"] = int(min_pixels)
    return AutoProcessor.from_pretrained(source, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mtgs/config/config_vlm_pair.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--graph_feats", required=True)
    parser.add_argument("--val_manifest", default="")
    parser.add_argument("--val_frame_root", default="")
    parser.add_argument("--val_graph_feats", default="")
    parser.add_argument("--val_gtmeta", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--wandb_off", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mode = str(cfg.experiment.get("mode", "vlm"))
    if mode not in TRAIN_MODES:
        raise ValueError(f"experiment.mode must be one of {TRAIN_MODES}, got {mode!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "vlm" and device.type != "cuda":
        raise RuntimeError("Qwen pair training requires CUDA; use graph_mlp mode on CPU")
    seed = int(cfg.train.get("seed", 101))
    monitor = str(cfg.experiment.get("monitor", "social_ap"))
    monitor_mode = str(cfg.experiment.get("monitor_mode", "max")).lower()
    threshold = float(cfg.val.get("threshold", 0.5))
    seed_everything(seed)

    experiment_dir = Path(str(cfg.experiment.out_root)) / str(cfg.experiment.name)
    checkpoint_dir = experiment_dir / "train" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, experiment_dir / "config_vlm_pair.yaml")
    ledger = experiment_dir / "metrics.jsonl"
    epoch_metrics_path = os.environ.get("PAIR_EPOCH_METRICS_PATH")
    resume = Path(args.resume) if args.resume else None

    train_cache = _load_graph_cache(args.graph_feats)
    val_cache = _load_graph_cache(args.val_graph_feats) if args.val_graph_feats else None
    hard_floor = _hard_floor(cfg)
    balance_mode = str(cfg.sampler.get("balance_mode", "task"))
    route_threshold = _route_threshold(cfg)
    epochs = int(cfg.train.epochs)
    num_samples = int(cfg.train.get("samples_per_epoch", 0))
    num_workers = int(cfg.train.get("num_workers", 4))
    accumulation = int(cfg.train.get("accum", 1))
    grad_clip = float(cfg.optim.get("grad_clip", 1.0))
    input_cfg = cfg.get("input", {})
    output_mode = str(cfg.get("model", {}).get("output", "yesno"))
    generative = output_mode == "generative"
    builders = select_generative_builders(cfg)
    draw_bboxes = bool(input_cfg.get("draw_bboxes", True))
    reuse_vision = (
        not draw_bboxes
        and bool(input_cfg.get("reuse_frozen_vision", True))
        and mode == "vlm"
        and not generative          # generative collate does not do cross-pair vision reuse yet
    )
    group_by_frame = reuse_vision and bool(
        input_cfg.get("group_by_frame", True)
    )

    processor = objective = None
    if mode == "vlm":
        processor = _processor(cfg, resume)
        train_dataset = PairInputDataset(
            args.manifest,
            args.frame_root,
            train_cache,
            raw_image_cache_size=int(cfg.train.get("raw_image_cache_size", 16)),
            draw_bboxes=draw_bboxes,
            output_mode=output_mode,
            graph_evidence=builders.graph_evidence,
        )
        if generative:
            collate = (
                make_text_generative_collate(processor)
                if builders.uses_text_collate
                else make_generative_collate(processor)
            )
        else:
            collate = make_pair_collate(processor, reuse_vision=reuse_vision)
        batch_size = int(cfg.train.bs)
        val_batch_size = int(cfg.val.bs)
    elif mode == "graph_mlp":
        train_dataset = GraphControlDataset(args.manifest, train_cache)
        collate = pair_control_collate
        batch_size = int(cfg.control.get("bs", 1024))
        val_batch_size = int(cfg.control.get("val_bs", batch_size))
        accumulation = int(cfg.control.get("accum", 1))
    else:
        train_dataset = GraphFeatureControlDataset(args.manifest, train_cache)
        collate = pair_feature_control_collate
        batch_size = int(cfg.control.get("feature_bs", cfg.control.get("bs", 1024)))
        val_batch_size = int(
            cfg.control.get("feature_val_bs", cfg.control.get("val_bs", batch_size))
        )
        accumulation = int(cfg.control.get("accum", 1))
    if num_samples <= 0:
        num_samples = len(train_dataset)

    # Confidence-gated routing applies only to the VLM generative path; graph controls are
    # cheap enough to run on every pair.
    train_route = route_threshold if (mode == "vlm" and generative) else None
    weights = train_dataset.sample_weights(
        balance_mode=balance_mode, hard_floor=hard_floor, route_threshold=train_route
    )
    if train_route is not None:
        kept = int((weights > 0).sum())
        print(
            f"[route] train: low-confidence pairs kept={kept}/{len(train_dataset)} "
            f"(threshold={train_route})",
            flush=True,
        )
    pos_weights = _resolve_pos_weights(cfg, train_dataset.annotations)
    # Generative uses next-token CE (no pos_weight), so EyeVLM-style balanced re-sampling
    # (balance_mode=task_label -> pos/neg even across the 3 tasks) has nothing to conflict
    # with. Only the BCE paths need the sampler<->pos_weight double-correction guard.
    if not generative:
        validate_sampler_loss_compatibility(balance_mode, pos_weights)
    criterion = PairTaskBCELoss(pos_weights).to(device)

    if mode == "vlm":
        if generative:
            objective, target_names = build_generative_objective(
                cfg, processor, device, resume
            )
        else:
            vlm, decoder, _auxiliary, target_names = build_vlm_objective(
                cfg, processor, device, resume
            )
            # yes/no head is the primary prediction now; the old LM auxiliary is subsumed.
            objective = PairSocialObjective(vlm, decoder, criterion)
        module = objective
        lora_params, new_params = partition_vlm_parameters(objective)
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": float(cfg.optim.lr)},
                {"params": new_params, "lr": float(cfg.optim.new_module_lr)},
            ],
            weight_decay=float(cfg.optim.weight_decay),
        )
        print(f"[pair] LoRA targets={len(target_names)}, params={sum(p.numel() for p in lora_params):,}")
    elif mode == "graph_mlp":
        module = GraphLogitMLPControl(
            hidden_dim=int(cfg.control.get("hidden_dim", 32)),
            dropout=float(cfg.control.get("dropout", 0.0)),
        ).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(cfg.control.get("lr", cfg.optim.new_module_lr)),
            weight_decay=float(cfg.optim.weight_decay),
        )
    else:
        module = GraphFeatureMLPControl(
            feature_dim=train_dataset.feature_dim,
            hidden_dim=int(cfg.control.get("feature_hidden_dim", 512)),
            dropout=float(cfg.control.get("dropout", 0.0)),
            include_heatmaps=bool(cfg.control.get("include_heatmaps", False)),
            heatmap_pool_size=int(cfg.control.get("heatmap_pool_size", 8)),
        ).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(cfg.control.get("lr", cfg.optim.new_module_lr)),
            weight_decay=float(cfg.optim.weight_decay),
        )

    steps_per_epoch = optimizer_steps_per_epoch(num_samples, batch_size, accumulation)
    total_steps = epochs * steps_per_epoch
    scheduler = _scheduler(
        optimizer,
        str(cfg.optim.scheduler),
        int(float(cfg.optim.warmup_ratio) * total_steps),
        total_steps,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_score = None
    global_step = 0
    if resume is not None:
        if mode == "vlm":
            state = _restore_vlm_modules(objective, resume)
        else:
            state = restore_control_checkpoint(
                resume, module, criterion, optimizer, scheduler
            )
        if mode == "vlm":
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        if state.get("best_score") is not None:
            best_score = float(state["best_score"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])

    val_loader = None
    val_dataset = None
    if args.val_manifest and val_cache is not None:
        if mode == "vlm":
            val_dataset = PairInputDataset(
                args.val_manifest,
                args.val_frame_root,
                val_cache,
                raw_image_cache_size=int(cfg.val.get("raw_image_cache_size", 16)),
                draw_bboxes=draw_bboxes,
                output_mode=output_mode,
                graph_evidence=builders.graph_evidence,
                generative_prompt_seed=seed if generative else None,
            )
        elif mode == "graph_mlp":
            val_dataset = GraphControlDataset(args.val_manifest, val_cache)
        else:
            val_dataset = GraphFeatureControlDataset(args.val_manifest, val_cache)
        val_loader = make_validation_loader(
            val_dataset,
            collate,
            val_batch_size,
            int(cfg.val.get("num_workers", num_workers)),
            group_by_frame=group_by_frame,
        )

    graph_val_metrics = None
    if args.val_gtmeta:
        if val_dataset is None:
            raise ValueError("--val_gtmeta requires --val_manifest and --val_graph_feats")
        if not Path(args.val_gtmeta).exists():
            raise FileNotFoundError(f"validation gtmeta does not exist: {args.val_gtmeta}")
        graph_collector = raw_graph_predictions(val_dataset.annotations, val_cache)
        graph_val_metrics = evaluate_pair_predictions(
            args.val_gtmeta,
            graph_collector.probabilities,
            expected_sids={sample.sid for sample in val_dataset.annotations},
            threshold=threshold,
        )
        print(format_pair_metrics(graph_val_metrics, "raw_graph:val"), flush=True)
        del graph_collector

    use_wandb = not args.wandb_off
    wandb_batch_log_interval = int(cfg.train.get("wandb_log_interval", 25))
    wandb = None
    if use_wandb:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(
            project="MTGS",
            entity="gaze-social",
            group="vlm-pair",
            name=str(cfg.experiment.name),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    print(
        f"[pair] mode={mode} train={len(train_dataset)} samples/ep={num_samples} "
        f"bs={batch_size} accum={accumulation} pos_weight={pos_weights} "
        f"lm_aux={float(cfg.loss.get('lm_aux_weight', 0.0))} "
        f"draw_bboxes={draw_bboxes} reuse_vision={reuse_vision} "
        f"group_by_frame={group_by_frame} out={experiment_dir}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        loader = make_epoch_loader(
            train_dataset,
            collate,
            weights,
            num_samples=num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            epoch=epoch,
            group_by_frame=group_by_frame,
        )
        def log_train_batch(local_step: int, payload: Mapping[str, float | int]) -> None:
            if wandb is None:
                return
            step = global_step + local_step
            event = {
                "epoch": epoch,
                "global_step": step,
                **{f"train/{key}": value for key, value in payload.items()},
            }
            for group_index, group in enumerate(optimizer.param_groups):
                event[f"optim/lr_group_{group_index}"] = float(group["lr"])
            wandb.log(event, step=step)

        train_stats = run_epoch(
            module,
            loader,
            device=device,
            criterion=criterion if mode != "vlm" else None,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation=accumulation,
            grad_clip=grad_clip,
            description=f"{mode}:train:{epoch}",
            batch_log_interval=wandb_batch_log_interval if wandb is not None else 0,
            batch_logger=log_train_batch if wandb is not None else None,
        )
        global_step += steps_per_epoch
        val_stats = None
        val_collector = None
        val_metrics = None
        if val_loader is not None:
            if generative:
                # metrics via EyeVLM candidate scoring (teacher-forced JSON log-likelihood).
                val_collector = collect_generative_predictions(
                    module, val_dataset, processor,
                    batch_size=val_batch_size,
                    num_workers=int(cfg.val.get("num_workers", num_workers)),
                    device=device, description=f"gen:val:{epoch}",
                    route_threshold=route_threshold,
                )
            else:
                val_collector = PairPredictionCollector()
                val_stats = run_epoch(
                    module,
                    val_loader,
                    device=device,
                    criterion=criterion if mode != "vlm" else None,
                    description=f"{mode}:val:{epoch}",
                    prediction_collector=val_collector,
                )
            val_collector.assert_complete(
                sample.eval_key for sample in val_dataset.annotations
            )
            if args.val_gtmeta:
                val_metrics = evaluate_pair_predictions(
                    args.val_gtmeta,
                    val_collector.probabilities,
                    expected_sids={sample.sid for sample in val_dataset.annotations},
                    threshold=threshold,
                )
                print(format_pair_metrics(val_metrics, f"{mode}:val:{epoch}"), flush=True)

        if val_metrics is not None or val_stats is not None:
            selection_name, selection_mode, selection_score = checkpoint_score(
                val_stats,
                val_metrics,
                monitor=monitor,
                monitor_mode=monitor_mode,
            )
        else:
            selection_name, selection_mode, selection_score = (
                "train_loss", "min", float(train_stats.loss)
            )
        improved = score_improved(selection_score, best_score, selection_mode)
        next_best = selection_score if improved else best_score
        best_val_loss = min(
            best_val_loss,
            val_stats.loss if val_stats is not None else train_stats.loss,
        )
        state = {
            "mode": mode,
            "epoch": epoch,
            "global_step": global_step,
            "monitor": selection_name,
            "monitor_mode": selection_mode,
            "selection_score": selection_score,
            "best_score": next_best,
            "best_val_loss": best_val_loss,
            "train": asdict(train_stats),
            "val": None if val_stats is None else asdict(val_stats),
            "val_metrics": None if val_metrics is None else metric_payload(val_metrics),
            "graph_val_metrics": None
            if graph_val_metrics is None
            else metric_payload(graph_val_metrics),
        }
        if mode == "vlm":
            save_vlm_checkpoint(
                checkpoint_dir / "last", objective, processor, optimizer, scheduler, state
            )
        else:
            save_control_checkpoint(
                checkpoint_dir / "last", module, criterion, optimizer, scheduler, state
            )
        if val_collector is not None:
            val_collector.save(checkpoint_dir / "last" / "val_predictions.pt")
        if improved:
            best_score = selection_score
            state["best_score"] = best_score
            if mode == "vlm":
                save_vlm_checkpoint(
                    checkpoint_dir / "best", objective, processor, optimizer, scheduler, state
                )
            else:
                save_control_checkpoint(
                    checkpoint_dir / "best", module, criterion, optimizer, scheduler, state
                )
            if val_collector is not None:
                val_collector.save(checkpoint_dir / "best" / "val_predictions.pt")
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(state) + "\n")
        if wandb is not None:
            log = {
                "train/loss": train_stats.loss,
                "train/examples": train_stats.examples,
            }
            if val_stats is not None:
                log.update({f"val/{key}": value for key, value in asdict(val_stats).items() if value is not None})
            if val_metrics is not None:
                log.update({
                    f"metric/val/{key}": value
                    for key, value in metric_payload(val_metrics).items()
                    if value is not None
                })
            if graph_val_metrics is not None:
                log.update({
                    f"metric/val/graph_only_{key}": value
                    for key, value in metric_payload(graph_val_metrics).items()
                    if value is not None
                })
            log.update({
                "epoch": epoch,
                "global_step": global_step,
                f"selection/{selection_name}": selection_score,
            })
            wandb.log(log, step=global_step)
        epoch_report = format_epoch_report(
            epoch=epoch,
            epochs=epochs,
            train_stats=train_stats,
            val_stats=val_stats,
            val_metrics=val_metrics,
            selection_name=selection_name,
            selection_score=selection_score,
            best_score=best_score,
            improved=improved,
        )
        append_epoch_report(epoch_metrics_path, epoch_report)
        print(epoch_report, flush=True)
        print(
            f"[pair] epoch={epoch} train={train_stats} val={val_stats} "
            f"selection={selection_name}:{selection_score:.6f} best={best_score}",
            flush=True,
        )

    if objective is not None:
        objective.close()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
