"""Locked VSGaze evaluation utilities shared by every pair-model variant.

The model-facing LAH convention is ``looker -> target``. Evaluation keys deliberately
remain in the raw manifest convention, where an LAH row ``(i,j)`` means ``j looks at
i``. :class:`SocialSample` retains both orientations, so this module never performs an
additional LAH transpose: it consumes ``SocialSample.eval_key`` exactly once at the
model/evaluator boundary.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from vlm.social.data import SocialAnnotationDataset, SOCIAL_TASKS
from vlm.social.input import sample_graph_logit


EvalKey = tuple[str, str, int, int]
CORE_METRIC_KEYS = (
    "F1_LAH",
    "F1_LAEO",
    "F1_SA",
    "LAH_AP",
    "LAH_AUC",
    "LAEO_AP",
    "LAEO_AUC",
    "SA_AP",
    "SA_AUC",
    "social_ap",
    "social_auc",
    "mean_social_f1",
)


def normalize_eval_key(key: Sequence[object]) -> EvalKey:
    """Validate an evaluator key and canonicalize only symmetric task ordering."""
    if len(key) != 4:
        raise ValueError(f"eval key must have four fields, got {key!r}")
    sid, task, raw_i, raw_j = key
    if not isinstance(sid, str) or not sid:
        raise ValueError(f"eval sid must be a non-empty string, got {sid!r}")
    if task not in SOCIAL_TASKS:
        raise ValueError(f"eval task must be one of {SOCIAL_TASKS}, got {task!r}")
    if (
        isinstance(raw_i, bool)
        or isinstance(raw_j, bool)
        or not isinstance(raw_i, int)
        or not isinstance(raw_j, int)
        or raw_i < 0
        or raw_j < 0
        or raw_i == raw_j
    ):
        raise ValueError(f"eval person indices must be distinct non-negative ints: {key!r}")
    if task in ("laeo", "sa") and raw_i > raw_j:
        raw_i, raw_j = raw_j, raw_i
    # LAH stays raw: (target, looker). Do not transpose it here.
    return sid, task, raw_i, raw_j


@dataclass(frozen=True)
class PredictionRecord:
    key: EvalKey
    label: int
    probability: float
    logit: float
    graph_probability: float | None
    graph_logit: float | None
    delta_logit: float | None


class PredictionCollector:
    """Strict one-prediction-per-labelled-row accumulator."""

    def __init__(self):
        self.probabilities: dict[EvalKey, float] = {}
        self.records: list[PredictionRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add_batch(
        self,
        eval_keys: Sequence[Sequence[object]],
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        graph_logits: torch.Tensor | None = None,
        delta_logits: torch.Tensor | None = None,
    ) -> None:
        batch = len(eval_keys)
        tensors = {"logits": logits, "labels": labels}
        if graph_logits is not None:
            tensors["graph_logits"] = graph_logits
        if delta_logits is not None:
            tensors["delta_logits"] = delta_logits
        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor) or tensor.shape != (batch,):
                shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
                raise ValueError(f"{name} must have shape ({batch},), got {shape}")

        logits_cpu = logits.detach().float().cpu()
        labels_cpu = labels.detach().float().cpu()
        graph_cpu = None if graph_logits is None else graph_logits.detach().float().cpu()
        delta_cpu = None if delta_logits is None else delta_logits.detach().float().cpu()
        if not bool(torch.all(torch.isfinite(logits_cpu))):
            raise ValueError("prediction logits contain non-finite values")
        if not bool(torch.all((labels_cpu == 0) | (labels_cpu == 1))):
            raise ValueError(f"prediction labels must be binary, got {labels_cpu.tolist()}")
        if graph_cpu is not None and not bool(torch.all(torch.isfinite(graph_cpu))):
            raise ValueError("graph logits contain non-finite values")
        if delta_cpu is not None and not bool(torch.all(torch.isfinite(delta_cpu))):
            raise ValueError("delta logits contain non-finite values")

        probabilities = torch.sigmoid(logits_cpu)
        graph_probabilities = None if graph_cpu is None else torch.sigmoid(graph_cpu)
        for index, raw_key in enumerate(eval_keys):
            key = normalize_eval_key(raw_key)
            if key in self.probabilities:
                raise ValueError(f"duplicate normalized prediction key: {key}")
            probability = float(probabilities[index])
            self.probabilities[key] = probability
            self.records.append(
                PredictionRecord(
                    key=key,
                    label=int(labels_cpu[index]),
                    probability=probability,
                    logit=float(logits_cpu[index]),
                    graph_probability=None
                    if graph_probabilities is None
                    else float(graph_probabilities[index]),
                    graph_logit=None if graph_cpu is None else float(graph_cpu[index]),
                    delta_logit=None if delta_cpu is None else float(delta_cpu[index]),
                )
            )

    def assert_complete(self, expected_keys: Iterable[Sequence[object]]) -> None:
        normalized = [normalize_eval_key(key) for key in expected_keys]
        expected = set(normalized)
        if len(expected) != len(normalized):
            raise ValueError("expected annotations contain duplicate normalized eval keys")
        actual = set(self.probabilities)
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        if missing or extra:
            raise ValueError(
                f"prediction coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "probabilities": dict(self.probabilities),
            "records": [asdict(record) for record in self.records],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


def raw_graph_predictions(
    annotations: SocialAnnotationDataset,
    graph_cache: Mapping[str, Mapping[str, object]],
) -> PredictionCollector:
    """Build the raw-graph row predictions without an image or DataLoader."""
    collector = PredictionCollector()
    keys = []
    logits = []
    labels = []
    for sample in annotations:
        keys.append(sample.eval_key)
        logits.append(sample_graph_logit(sample, graph_cache[sample.sid]))
        labels.append(sample.label)
    if not keys:
        raise ValueError("cannot evaluate an empty annotation set")
    graph_logits = torch.stack(logits).float()
    collector.add_batch(
        keys,
        graph_logits,
        torch.tensor(labels, dtype=torch.float32),
        graph_logits=graph_logits,
        delta_logits=torch.zeros_like(graph_logits),
    )
    collector.assert_complete(sample.eval_key for sample in annotations)
    return collector


def _parse_f1(detail: str) -> dict[str, float | None]:
    values: dict[str, float | None] = {task: None for task in ("LAH", "LAEO", "SA")}
    section = None
    for line in detail.splitlines():
        text = line.strip()
        if text.startswith("----- LAEO"):
            section = "LAEO"
        elif text.startswith("----- LAH"):
            section = "LAH"
        elif text.startswith("----- CoAtt"):
            section = "SA"
        elif text.startswith("F1 ") and "thr=" in text and section is not None:
            match = re.search(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", text)
            values[section] = None if match is None else float(match.group())
    return values


def augment_social_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """Add SA F1 and complete three-task aggregate metrics to locked output."""
    output = dict(metrics)
    parsed = _parse_f1(str(output.get("detail", "")))
    if output.get("F1_LAH") is None:
        output["F1_LAH"] = parsed["LAH"]
    if output.get("F1_LAEO") is None:
        output["F1_LAEO"] = parsed["LAEO"]
    output["F1_SA"] = parsed["SA"]

    def complete_mean(keys: Sequence[str]) -> float | None:
        values = [output.get(key) for key in keys]
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values) / len(values)

    output["social_ap"] = complete_mean(("LAH_AP", "LAEO_AP", "SA_AP"))
    output["social_auc"] = complete_mean(("LAH_AUC", "LAEO_AUC", "SA_AUC"))
    output["mean_social_f1"] = complete_mean(("F1_LAH", "F1_LAEO", "F1_SA"))
    return output


def evaluate_predictions(
    gtmeta_path: str | Path,
    probabilities: Mapping[EvalKey, float],
    *,
    expected_sids: Iterable[str],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Run the repository's locked ``build_mtgs_dicts -> compute`` evaluator."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    expected = set(expected_sids)
    if not expected:
        raise ValueError("expected_sids must be non-empty")
    normalized = {normalize_eval_key(key): float(value) for key, value in probabilities.items()}
    if len(normalized) != len(probabilities):
        raise ValueError("probability mapping contains duplicate normalized keys")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized.values()):
        raise ValueError("probabilities must be finite values in [0,1]")

    # Delayed import keeps graph controls lightweight until the locked harness is used.
    from vlm.social.metrics import build_mtgs_dicts, evaluate

    samples = build_mtgs_dicts(gtmeta_path, normalized, restrict_sids=expected)
    if len(samples) != len(expected):
        raise ValueError(
            f"gtmeta coverage mismatch: expected {len(expected)} frames, built {len(samples)}"
        )
    return augment_social_metrics(evaluate(samples, thr=threshold))


def metric_payload(metrics: Mapping[str, object], *, detail: bool = False) -> dict[str, object]:
    """Return a JSON-safe metric dictionary, excluding the long log by default."""
    output = {}
    for key, value in metrics.items():
        if key == "detail" and not detail:
            continue
        if value is None or isinstance(value, str):
            output[key] = value
        elif isinstance(value, (int, float)):
            output[key] = float(value)
        elif hasattr(value, "item"):
            output[key] = float(value.item())
        else:
            output[key] = value
    return output


def metric_deltas(
    metrics: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, float | None]:
    output = {}
    for key in CORE_METRIC_KEYS:
        current, base = metrics.get(key), baseline.get(key)
        output[key] = (
            None if current is None or base is None else float(current) - float(base)
        )
    return output


def format_graph_model_table(
    graph_metrics: Mapping[str, object],
    model_metrics: Mapping[str, object],
    annotations: SocialAnnotationDataset,
    *,
    model_name: str = "VLM",
) -> str:
    """Format graph-versus-model metrics for the exact evaluated manifest."""

    task_keys = {
        "LAH": ("LAH_AP", "LAH_AUC", "F1_LAH", "lah"),
        "LAEO": ("LAEO_AP", "LAEO_AUC", "F1_LAEO", "laeo"),
        "SA": ("SA_AP", "SA_AUC", "F1_SA", "sa"),
        "Macro": ("social_ap", "social_auc", "mean_social_f1", None),
    }
    class_counts = annotations.class_counts

    def number(value: object) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    def delta(key: str) -> str:
        graph, model = graph_metrics.get(key), model_metrics.get(key)
        if graph is None or model is None:
            return "N/A"
        return f"{float(model) - float(graph):+.4f}"

    lines = [
        f"===== TEST: RAW GRAPH vs {model_name.upper()} (same manifest) =====",
        "| Task | Pos / Neg | Graph AP | " + model_name + " AP | ΔAP | Graph AUC | "
        + model_name + " AUC | ΔAUC | Graph F1 | " + model_name + " F1 | ΔF1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_positive = total_negative = 0
    for title, (ap_key, auc_key, f1_key, task) in task_keys.items():
        if task is None:
            count_text = f"{total_positive} / {total_negative}"
        else:
            counts = class_counts[task]
            total_positive += counts[1]
            total_negative += counts[0]
            count_text = f"{counts[1]} / {counts[0]}"
        lines.append(
            f"| {title} | {count_text} | {number(graph_metrics.get(ap_key))} | "
            f"{number(model_metrics.get(ap_key))} | {delta(ap_key)} | "
            f"{number(graph_metrics.get(auc_key))} | {number(model_metrics.get(auc_key))} | "
            f"{delta(auc_key)} | {number(graph_metrics.get(f1_key))} | "
            f"{number(model_metrics.get(f1_key))} | {delta(f1_key)} |"
        )
    return "\n".join(lines)


def format_metrics(metrics: Mapping[str, object], title: str = "") -> str:
    def value(key: str) -> str:
        item = metrics.get(key)
        return "N/A" if item is None else f"{float(item):.4f}"

    prefix = f"[{title}] " if title else ""
    return (
        f"{prefix}social_ap={value('social_ap')} social_auc={value('social_auc')} "
        f"mean_f1={value('mean_social_f1')}\n"
        f"  LAH : AP={value('LAH_AP')} AUC={value('LAH_AUC')} F1={value('F1_LAH')}\n"
        f"  LAEO: AP={value('LAEO_AP')} AUC={value('LAEO_AUC')} F1={value('F1_LAEO')}\n"
        f"  SA  : AP={value('SA_AP')} AUC={value('SA_AUC')} F1={value('F1_SA')}"
    )
