"""Pair-wise MTGS+Qwen training and the vision-free graph-logit control.

Unit 6 intentionally stops at train/validation loss and checkpointing. Reconstructing
the locked VSGaze prediction dictionaries and AP/AUC/F1 evaluation belongs to Unit 7.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset,
    WeightedRandomSampler,
)
from tqdm import tqdm

from vlm.cache.config import QWEN
from vlm.social.objective import (
    GenerativeObjective,
    generative_answer_token_ids,
)
from vlm.social.input import (
    SocialInputDataset,
    task_pos_weights,
    partition_by_graph_confidence,
    sample_graph_logit,
)
from vlm.social.evaluation import (
    PredictionCollector,
    evaluate_predictions,
    format_graph_model_table,
    format_metrics,
    format_routing_comparison_table,
    metric_payload,
    raw_graph_predictions,
    routing_low_confidence_keys,
)
from vlm.social.model import (
    TextGenerativeVLM,
    make_text_generative_collate,
    make_text_generative_direct_eval_collate,
    make_text_generative_eval_collate,
)
from vlm.runtime.qwen import patch_qwen3vl_patch_embed


LORA_PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)

# Under confidence-gated routing, per-pair scores come from two independently
# calibrated models (graph logit vs VLM logit-yes/no), so any ranking-based metric
# is not meaningful -- see format_metrics()'s f1_only docstring. Keep this key set in
# sync with what format_metrics(f1_only=True) drops.
_ROUTING_INVALID_METRIC_KEYS = {
    "social_ap", "social_auc",
    "LAH_AP", "LAH_AUC", "LAEO_AP", "LAEO_AUC", "SA_AP", "SA_AUC", "AP_SA",
}


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
    f1_only: bool = False,
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
        lines.append(format_metrics(val_metrics, "validation", f1_only=f1_only))
    best = "N/A" if best_score is None else f"{best_score:.6f}"
    lines.append(
        f"  selection: {selection_name}={selection_score:.6f} "
        f"best={best} improved={improved}"
    )
    return "\n".join(lines)


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
    # Confidence-gated routing wraps the low-confidence remainder in a Subset
    # (collect_generative_predictions), which has no .annotations of its own --
    # unwrap to the base dataset's annotations and index through .indices instead.
    if isinstance(dataset, Subset):
        annotations = getattr(dataset.dataset, "annotations", None)
        samples = getattr(annotations, "samples", None)
        if samples is None:
            raise ValueError("frame grouping requires a pair dataset with annotations")
        return [samples[i].sid for i in dataset.indices]
    annotations = getattr(dataset, "annotations", None)
    samples = getattr(annotations, "samples", None)
    if samples is None or len(samples) != len(dataset):
        raise ValueError("frame grouping requires a pair dataset with annotations")
    return [sample.sid for sample in samples]


def make_epoch_loader(
    dataset,
    collate_fn,
    weights: torch.Tensor | None = None,
    *,
    sampling_strategy: str = "uniform",
    num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
    pin_memory: bool | None = None,
    group_by_frame: bool = False,
) -> DataLoader:
    """Construct one deterministic train epoch.

    ``once`` visits every row in the selected manifest exactly once, in a
    seeded permutation. The other strategies deliberately resample with replacement.
    """
    generator = torch.Generator().manual_seed(seed + epoch)
    if sampling_strategy == "once":
        if num_samples not in (0, len(dataset)):
            raise ValueError(
                "sampling.strategy=once requires samples_per_epoch=0 or the "
                f"dataset length ({len(dataset)}), got {num_samples}"
            )
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
    elif sampling_strategy in ("uniform", "task_balanced", "task_label_balanced"):
        if weights is None:
            raise ValueError(f"sampling strategy {sampling_strategy!r} requires weights")
        if num_samples <= 0:
            num_samples = len(dataset)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
    else:
        raise ValueError(
            "sampling.strategy must be once/uniform/task_balanced/task_label_balanced, "
            f"got {sampling_strategy!r}"
        )
    common = {
        "dataset": dataset,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
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


def partition_vlm_parameters(objective: GenerativeObjective):
    """Return disjoint LoRA/new-module groups and validate the frozen-base contract.

    The text-evidence generative VLM has no new trainable modules besides the LoRA
    adapters: graph evidence is already natural-language prompt text.
    """
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
    new = _trainable([])
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
    # BCE weighting is only used by separate graph-control experiments. The primary
    # generative VLM intentionally has no loss section.
    loss_cfg = cfg.get("loss", {})
    configured = loss_cfg.get("pos_weight", "auto")
    if str(configured).lower() == "auto":
        return task_pos_weights(
            annotations,
            minimum=float(loss_cfg.get("pos_weight_min", 0.2)),
            maximum=float(loss_cfg.get("pos_weight_max", 5.0)),
        )
    values = OmegaConf.to_container(configured, resolve=True)
    if not isinstance(values, Mapping):
        raise ValueError("loss.pos_weight must be 'auto' or a task mapping")
    return {str(key): float(value) for key, value in values.items()}


def _hard_floor(cfg) -> float | None:
    sampler_cfg = cfg.get("sampler", {})
    if not bool(sampler_cfg.get("hard_weight", False)):
        return None
    return float(sampler_cfg.get("hard_floor", 0.25))

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
    return backbone, target_names


@dataclass(frozen=True)
class GenerativeBuilders:
    """Text-evidence generative VLM policy (the single supported contract)."""

    reuse_vision: bool
    include_graph_evidence: bool


def select_generative_builders(cfg) -> GenerativeBuilders:
    """Return the single supported VLM contract: generative text evidence."""
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else cfg["model"]
    input_cfg = cfg.get("input", {}) if hasattr(cfg, "get") else {}
    return GenerativeBuilders(
        reuse_vision=bool(input_cfg.get("reuse_frozen_vision", False)),
        include_graph_evidence=bool(model_cfg.get("include_graph_evidence", True)),
    )


def build_generative_objective(cfg, processor, device: torch.device, resume: Path | None = None):
    """Generative yes/no objective with graph evidence written in the prompt as text."""
    backbone, target_names = _make_lora_backbone(cfg, processor, device, resume)
    builders = select_generative_builders(cfg)
    cache_size = (
        int(cfg.get("input", {}).get("vision_cache_size", 0))
        if builders.reuse_vision
        else 0
    )
    disk_cache = str(cfg.get("input", {}).get("vision_disk_cache", "")) or None
    disk_metadata = None if disk_cache is None else {
        "qwen": str(cfg.model.get("qwen", QWEN)),
        "max_pixels": str(int(cfg.model.get("max_pixels", 200704))),
    }
    vlm = TextGenerativeVLM(
        backbone,
        vision_cache_size=cache_size,
        vision_disk_cache=disk_cache,
        vision_disk_metadata=disk_metadata,
    )
    direct_ids = generative_answer_token_ids(processor.tokenizer)
    objective = GenerativeObjective(
        vlm, direct_yes_no_token_ids=direct_ids
    ).to(device=device)
    return objective, target_names


def _restore_vlm_modules(objective: GenerativeObjective, checkpoint: Path) -> dict[str, Any]:
    modules_path = checkpoint / "pair_modules.pt"
    trainer_path = checkpoint / "trainer_state.pt"
    if not modules_path.exists() or not trainer_path.exists():
        raise FileNotFoundError(f"incomplete pair checkpoint: {checkpoint}")
    # Text-evidence generative VLM has no extra trainable modules besides the LoRA
    # adapter (restored from disk by PeftModel.from_pretrained); nothing to load here.
    return torch.load(trainer_path, map_location="cpu", weights_only=False)


def save_vlm_checkpoint(
    path: Path,
    objective: GenerativeObjective,
    processor,
    optimizer,
    scheduler,
    state: Mapping[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    objective.vlm.backbone.save_pretrained(path / "adapter")
    processor.save_pretrained(path / "processor")
    # Text-evidence generative VLM: graph evidence is prompt text; only the LoRA adapter
    # (saved above) trains, so there are no extra modules to serialize.
    modules = {"generative": True, "text_evidence": True}
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


def _generative_batch(objective: GenerativeObjective, batch: dict[str, Any]):
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


def collect_generative_predictions(
    objective: GenerativeObjective,
    dataset,
    processor,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    description: str,
    reuse_vision: bool = False,
    group_by_frame: bool = False,
    route_threshold: float | None = None,
) -> PredictionCollector:
    """Fill a prediction collector with EyeVLM candidate-scoring probabilities: each pair's
    two JSON candidates are teacher-forced and P(positive)=sigmoid(LL_pos-LL_neg).

    With ``route_threshold`` set, confidence-gated routing answers high-confidence pairs with
    the frozen graph logit directly (no VLM forward) and queries the VLM only on the
    low-confidence remainder, cutting evaluation cost by the high-confidence fraction.
    """
    collector = PredictionCollector()
    eval_dataset = dataset
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
    direct_text_scoring = (
        reuse_vision
        and getattr(objective, "direct_yes_no_token_ids", None) is not None
    )
    eval_collate = (
        make_text_generative_direct_eval_collate(processor, reuse_vision=True)
        if direct_text_scoring
        else make_text_generative_eval_collate(processor, reuse_vision=reuse_vision)
    )
    loader = make_validation_loader(
        eval_dataset, eval_collate, batch_size, num_workers,
        pin_memory=device.type == "cuda",
        group_by_frame=group_by_frame,
    )
    was_training = objective.training
    objective.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=description, leave=False, file=sys.stdout):
            num_pairs = int(batch["num_pairs"])
            model_inputs = {
                k: v for k, v in batch.items()
                if k not in ("pair_labels", "eval_keys", "num_pairs")
            }
            prob = objective.score(model_inputs, num_pairs).clamp(1e-6, 1 - 1e-6)
            logit = torch.log(prob / (1 - prob))          # collector sigmoids back to prob
            collector.add_batch(batch["eval_keys"], logit, batch["pair_labels"])
    objective.train(was_training)
    if direct_text_scoring:
        print(
            f"[answer] {description}: direct yes/no logit scoring (one prompt forward per pair)",
            flush=True,
        )
    return collector


def run_epoch(
    module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    accumulation: int = 1,
    grad_clip: float = 0.0,
    description: str = "epoch",
    prediction_collector: PredictionCollector | None = None,
    batch_log_interval: int = 0,
    batch_logger=None,
    loss_ema_state: dict[str, float] | None = None,
    loss_ema_beta: float = 0.98,
) -> EpochStats:
    if batch_log_interval < 0:
        raise ValueError("batch_log_interval must be non-negative")
    if batch_logger is not None and not callable(batch_logger):
        raise TypeError("batch_logger must be callable")
    if not 0.0 < loss_ema_beta < 1.0:
        raise ValueError("loss_ema_beta must be in (0, 1)")
    training = optimizer is not None
    module.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    count = correct = 0
    loss_sum = residual_sum = lm_loss_sum = lm_acc_sum = 0.0
    lm_count = 0
    batches = len(loader)
    optimizer_steps = 0
    iterator = tqdm(loader, desc=description, leave=False, file=sys.stdout)

    for index, batch in enumerate(iterator):
        group_start = (index // accumulation) * accumulation
        group_size = min(accumulation, batches - group_start)
        with torch.set_grad_enabled(training):
            output, labels = _generative_batch(module, batch)
            if output.loss is None:
                raise RuntimeError("training objective did not return loss")
            if training:
                (output.loss / group_size).backward()

        batch_size = labels.numel()
        count += batch_size
        batch_loss = float(output.loss.detach())
        loss_sum += batch_loss * batch_size
        residual_sum += float(output.residual_loss.detach()) * batch_size
        if loss_ema_state is not None:
            # Dense EMA over every micro-batch; state lives in the caller so the curve
            # is continuous across epochs (unlike the epoch-cumulative loss_sum/count).
            prev = loss_ema_state.get("value")
            loss_ema_state["value"] = (
                batch_loss if prev is None
                else loss_ema_beta * prev + (1.0 - loss_ema_beta) * batch_loss
            )
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
                running = (
                    loss_ema_state["value"] if loss_ema_state is not None
                    else loss_sum / max(count, 1)
                )
                batch_logger(optimizer_steps, {
                    "batch_loss": batch_loss,
                    "running_loss": running,
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
    parser.add_argument("--config", default="mtgs/config/config_vlm.yaml")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Qwen VLM training requires CUDA")
    seed = int(cfg.train.get("seed", 101))
    monitor = str(cfg.experiment.get("monitor", "social_ap"))
    monitor_mode = str(cfg.experiment.get("monitor_mode", "max")).lower()
    threshold = float(cfg.val.get("threshold", 0.5))
    # Confidence-gated routing mixes the frozen graph's and the VLM's independent score
    # scales, so AP/AUC over the combined predictions are not meaningful once routed --
    # only report F1 in that case (see format_graph_model_table / format_metrics docstrings).
    routing_on = bool(cfg.get("routing", {}).get("use", False))
    seed_everything(seed)

    experiment_dir = Path(str(cfg.experiment.out_root)) / str(cfg.experiment.name)
    checkpoint_dir = experiment_dir / "train" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, experiment_dir / "config_vlm.yaml")
    ledger = experiment_dir / "metrics.jsonl"
    resume = Path(args.resume) if args.resume else None

    train_cache = _load_graph_cache(args.graph_feats)
    val_cache = _load_graph_cache(args.val_graph_feats) if args.val_graph_feats else None
    sampling_cfg = cfg.get("sampling", {})
    sampling_strategy = str(sampling_cfg.get("strategy", "once"))
    balance_modes = {
        "uniform": "none",
        "task_balanced": "task",
        "task_label_balanced": "task_label",
    }
    if sampling_strategy not in ("once", *balance_modes):
        raise ValueError(
            "sampling.strategy must be once/uniform/task_balanced/task_label_balanced, "
            f"got {sampling_strategy!r}"
        )
    # data.profile chooses a subset; sampling only traverses that selected population.
    balance_mode = balance_modes.get(sampling_strategy, "none")
    hard_floor = None
    # Confidence-gated routing (config_vlm.yaml `routing.use`): validation answers
    # high-confidence pairs with the frozen graph directly and queries the VLM only on
    # the low-confidence remainder, mirroring eval_vlm.sh's test-time behaviour exactly.
    # This requires --val_manifest to be the FULL (unfiltered) val set -- train_vlm.sh
    # filters only the TRAIN manifest to low-confidence pairs, never --val_manifest --
    # otherwise raw_graph_predictions() below would score the graph on the hard subset
    # only, not the full population it needs to be compared against.
    route_threshold = _route_threshold(cfg)
    epochs = int(cfg.train.epochs)
    num_samples = int(sampling_cfg.get("samples_per_epoch", 0))
    num_workers = int(cfg.train.get("num_workers", 4))
    accumulation = int(cfg.train.get("accum", 1))
    grad_clip = float(cfg.optim.get("grad_clip", 1.0))
    input_cfg = cfg.get("input", {})
    # The only VLM implementation is generative yes/no with text graph evidence.
    builders = select_generative_builders(cfg)
    reuse_vision = builders.reuse_vision
    group_by_frame = reuse_vision and bool(
        input_cfg.get("group_by_frame", False)
    )

    processor = _processor(cfg, resume)
    train_dataset = SocialInputDataset(
        args.manifest,
        args.frame_root,
        train_cache,
        raw_image_cache_size=int(cfg.train.get("raw_image_cache_size", 16)),
        include_graph_evidence=builders.include_graph_evidence,
    )
    collate = make_text_generative_collate(processor, reuse_vision=builders.reuse_vision)
    batch_size = int(cfg.train.bs)
    val_batch_size = int(cfg.val.bs)
    if sampling_strategy == "once":
        # Exact-once traversal never changes the selected population size.
        num_samples = len(train_dataset)
        weights = None
    else:
        if num_samples <= 0:
            num_samples = len(train_dataset)
        weights = train_dataset.sample_weights(
            balance_mode=balance_mode, hard_floor=hard_floor
        )

    # Generative training uses next-token CE (no BCE weighting).
    objective, target_names = build_generative_objective(cfg, processor, device, resume)
    module = objective
    lora_params, new_params = partition_vlm_parameters(objective)
    parameter_groups = [{"params": lora_params, "lr": float(cfg.optim.lr)}]
    if new_params:
        parameter_groups.append(
            {
                "params": new_params,
                "lr": float(cfg.optim.get("new_module_lr", cfg.optim.lr)),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg.optim.weight_decay),
    )
    print(f"[vlm] LoRA targets={len(target_names)}, params={sum(p.numel() for p in lora_params):,}")

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
        state = _restore_vlm_modules(objective, resume)
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
        val_dataset = SocialInputDataset(
            args.val_manifest,
            args.val_frame_root,
            val_cache,
            raw_image_cache_size=int(cfg.val.get("raw_image_cache_size", 16)),
            generative_prompt_seed=int(cfg.val.get("prompt_seed", seed)),
            include_graph_evidence=builders.include_graph_evidence,
        )
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
        graph_val_metrics = evaluate_predictions(
            args.val_gtmeta,
            graph_collector.probabilities,
            expected_sids={sample.sid for sample in val_dataset.annotations},
            threshold=threshold,
        )
        print(format_metrics(graph_val_metrics, "raw_graph:val", f1_only=routing_on), flush=True)
        if not routing_on:
            # Routing needs graph_val_collector.records for the final diagnostic table
            # (format_routing_comparison_table); otherwise free it now as before.
            del graph_collector
        else:
            graph_val_collector = graph_collector

    use_wandb = not args.wandb_off
    wandb_batch_log_interval = int(cfg.train.get("wandb_log_interval", 25))
    wandb = None
    if use_wandb:
        import wandb as wandb_module

        wandb = wandb_module
        run = wandb.init(
            project="MTGS",
            entity="gaze-social",
            group="vlm",
            name=str(cfg.experiment.name),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        (experiment_dir / "wandb_run_id.txt").write_text(str(run.id), encoding="utf-8")

    print(
        f"[vlm] train={len(train_dataset)} samples/ep={num_samples} "
        f"sampling={sampling_strategy} bs={batch_size} accum={accumulation} "
        f"reuse_vision={reuse_vision} group_by_frame={group_by_frame} "
        f"out={experiment_dir}",
        flush=True,
    )
    # Persistent across epochs so train/running_loss is a continuous EMA rather than an
    # epoch-cumulative mean that resets (and visibly jumps) at every epoch boundary.
    train_loss_ema: dict[str, float] = {}
    for epoch in range(start_epoch, epochs):
        loader = make_epoch_loader(
            train_dataset,
            collate,
            weights,
            sampling_strategy=sampling_strategy,
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
                **{
                    f"train/{key}": value
                    for key, value in payload.items()
                    if key != "examples"
                },
            }
            for group_index, group in enumerate(optimizer.param_groups):
                event[f"optim/lr_group_{group_index}"] = float(group["lr"])
            wandb.log(event, step=step)

        train_stats = run_epoch(
            module,
            loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation=accumulation,
            grad_clip=grad_clip,
            description=f"Epoch {epoch + 1}/{epochs} [train]",
            batch_log_interval=wandb_batch_log_interval if wandb is not None else 0,
            batch_logger=log_train_batch if wandb is not None else None,
            loss_ema_state=train_loss_ema,
        )
        global_step += steps_per_epoch
        val_stats = None
        val_collector = None
        val_metrics = None
        if val_loader is not None:
            # metrics via direct yes/no answer-logit scoring (one prompt forward per pair).
            val_collector = collect_generative_predictions(
                module, val_dataset, processor,
                batch_size=val_batch_size,
                num_workers=int(cfg.val.get("num_workers", num_workers)),
                device=device, description=f"Epoch {epoch + 1}/{epochs} [val]",
                reuse_vision=builders.reuse_vision,
                group_by_frame=group_by_frame,
                route_threshold=route_threshold,
            )
            val_collector.assert_complete(
                sample.eval_key for sample in val_dataset.annotations
            )
            if args.val_gtmeta:
                val_metrics = evaluate_predictions(
                    args.val_gtmeta,
                    val_collector.probabilities,
                    expected_sids={sample.sid for sample in val_dataset.annotations},
                    threshold=threshold,
                )
                print(
                    format_metrics(
                        val_metrics, f"epoch {epoch + 1}/{epochs} val", f1_only=routing_on
                    ),
                    flush=True,
                )

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
            "mode": "vlm",
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
        save_vlm_checkpoint(
            checkpoint_dir / "last", objective, processor, optimizer, scheduler, state
        )
        if val_collector is not None:
            val_collector.save(checkpoint_dir / "last" / "val_predictions.pt")
        if improved:
            best_score = selection_score
            state["best_score"] = best_score
            save_vlm_checkpoint(
                checkpoint_dir / "best", objective, processor, optimizer, scheduler, state
            )
            if val_collector is not None:
                val_collector.save(checkpoint_dir / "best" / "val_predictions.pt")
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(state) + "\n")
        if wandb is not None:
            log = {
                "train/loss": train_stats.loss,
            }
            if val_stats is not None:
                log.update({f"val/{key}": value for key, value in asdict(val_stats).items() if value is not None})
            if val_metrics is not None:
                excluded_keys = _ROUTING_INVALID_METRIC_KEYS if routing_on else {"AP_SA"}
                log.update({
                    f"metric/val/{key}": value
                    for key, value in metric_payload(val_metrics).items()
                    if value is not None and key not in excluded_keys
                })
            log.update({
                "epoch": epoch,
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
            f1_only=routing_on,
        )
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

    if graph_val_metrics is not None and val_metrics is not None and val_dataset is not None:
        # Final stdout block: last epoch's val metrics against the frozen graph baseline,
        # in the same table format eval_vlm.sh prints for a held-out test run.
        if routing_on and val_collector is not None:
            low_conf_keys = routing_low_confidence_keys(
                val_dataset.annotations, val_cache, route_threshold
            )
            print(
                format_routing_comparison_table(
                    graph_val_collector.records,
                    val_collector.records,
                    low_conf_keys,
                    graph_val_metrics,
                    val_metrics,
                    threshold=route_threshold,
                    model_name=str(cfg.experiment.name),
                ),
                flush=True,
            )
        else:
            print(
                format_graph_model_table(
                    graph_val_metrics, val_metrics, val_dataset.annotations,
                    model_name=str(cfg.experiment.name),
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
