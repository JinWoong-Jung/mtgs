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
from vlm.social.input import partition_by_graph_confidence, sample_graph_logit


EvalKey = tuple[str, str, int, int]
CORE_METRIC_KEYS = (
    "F1_LAH",
    "F1_LAEO",
    "F1_SA",
    "Acc_LAH",
    "Acc_LAEO",
    "Acc_SA",
    "LAH_AP",
    "LAH_AUC",
    "LAEO_AP",
    "LAEO_AUC",
    "SA_AP",
    "SA_AUC",
    "social_ap",
    "social_auc",
    "mean_social_f1",
    "mean_social_accuracy",
)
_ACCURACY_KEYS = (("LAH", "lah_pred", "lah_gt"), ("LAEO", "laeo_pred", "laeo_gt"), ("SA", "coatt_pred", "coatt_gt"))


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


def binary_accuracy(
    samples: Sequence[Mapping[str, object]], *, threshold: float = 0.5
) -> dict[str, float | None]:
    """Flat per-task binary accuracy (TP+TN / total) over valid (gt != -1) pairs.

    Complements the locked per-target-argmax F1 with a simple threshold@0.5 accuracy,
    computed directly from the same ``build_mtgs_dicts`` sample dicts (never re-derives
    predictions/labels, so it cannot diverge from what ``evaluate()`` scored).
    """
    output: dict[str, float | None] = {}
    for name, pred_key, gt_key in _ACCURACY_KEYS:
        correct = 0
        count = 0
        for sample in samples:
            pred = sample[pred_key].reshape(-1)
            gt = sample[gt_key].reshape(-1)
            valid = gt != -1
            if not bool(valid.any()):
                continue
            pred_label = (pred[valid] >= threshold).long()
            correct += int((pred_label == gt[valid]).sum())
            count += int(valid.sum())
        output[f"Acc_{name}"] = None if count == 0 else correct / count
    return output


def augment_social_metrics(
    metrics: Mapping[str, object],
    samples: Sequence[Mapping[str, object]] | None = None,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Add SA F1, binary accuracy, and complete three-task aggregate metrics."""
    output = dict(metrics)
    parsed = _parse_f1(str(output.get("detail", "")))
    if output.get("F1_LAH") is None:
        output["F1_LAH"] = parsed["LAH"]
    if output.get("F1_LAEO") is None:
        output["F1_LAEO"] = parsed["LAEO"]
    output["F1_SA"] = parsed["SA"]
    if samples is not None:
        output.update(binary_accuracy(samples, threshold=threshold))

    def complete_mean(keys: Sequence[str]) -> float | None:
        values = [output.get(key) for key in keys]
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values) / len(values)

    output["social_ap"] = complete_mean(("LAH_AP", "LAEO_AP", "SA_AP"))
    output["social_auc"] = complete_mean(("LAH_AUC", "LAEO_AUC", "SA_AUC"))
    output["mean_social_f1"] = complete_mean(("F1_LAH", "F1_LAEO", "F1_SA"))
    output["mean_social_accuracy"] = complete_mean(("Acc_LAH", "Acc_LAEO", "Acc_SA"))
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
    return augment_social_metrics(evaluate(samples, thr=threshold), samples, threshold=threshold)


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
    f1_only: bool = False,
) -> str:
    """Format graph-versus-model metrics for the exact evaluated manifest.

    ``f1_only`` drops the AP/AUC columns and prints only F1: under confidence-gated
    routing, ``model_metrics`` mixes the frozen graph's score scale (high-confidence
    pairs) with the VLM's independent score scale (low-confidence pairs), so a ranking
    metric (AP/AUC) computed over that mix is not meaningful. F1 thresholds each pair at
    0.5 independently and stays valid regardless of which scale answered it. Callers pass
    ``f1_only=cfg.routing.use`` so routed runs report the graph-only vs graph+VLM-routing
    comparison as F1-only automatically.
    """

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

    graph_label = "Graph-only"
    model_label = f"Graph+{model_name} routing" if f1_only else model_name
    if f1_only:
        lines = [
            f"===== TEST: {graph_label.upper()} vs {model_label.upper()} (same manifest, "
            "F1 only -- routing mixes graph/VLM score scales, AP/AUC omitted) =====",
            f"| Task | Pos / Neg | {graph_label} F1 | {model_label} F1 | ΔF1 |",
            "|---|---:|---:|---:|---:|",
        ]
    else:
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
        if f1_only:
            lines.append(
                f"| {title} | {count_text} | {number(graph_metrics.get(f1_key))} | "
                f"{number(model_metrics.get(f1_key))} | {delta(f1_key)} |"
            )
        else:
            lines.append(
                f"| {title} | {count_text} | {number(graph_metrics.get(ap_key))} | "
                f"{number(model_metrics.get(ap_key))} | {delta(ap_key)} | "
                f"{number(graph_metrics.get(auc_key))} | {number(model_metrics.get(auc_key))} | "
                f"{delta(auc_key)} | {number(graph_metrics.get(f1_key))} | "
                f"{number(model_metrics.get(f1_key))} | {delta(f1_key)} |"
            )
    return "\n".join(lines)


def _binary_f1_from_pairs(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Threshold@0.5 F1 over ``(label, pred)`` pairs; ``None`` if ``pairs`` is empty."""
    if not pairs:
        return None
    tp = sum(1 for label, pred in pairs if label == 1 and pred == 1)
    fp = sum(1 for label, pred in pairs if label == 0 and pred == 1)
    fn = sum(1 for label, pred in pairs if label == 1 and pred == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def routing_low_confidence_keys(
    annotations: SocialAnnotationDataset,
    graph_cache: Mapping[str, Mapping[str, object]],
    threshold: float,
) -> set[EvalKey]:
    """Normalized eval keys of pairs the router sends to the VLM (graph conf < threshold)."""
    _high, low = partition_by_graph_confidence(annotations, graph_cache, threshold)
    return {normalize_eval_key(annotations.samples[i].eval_key) for i in low}


def format_routing_comparison_table(
    graph_records: Sequence[PredictionRecord],
    model_records: Sequence[PredictionRecord],
    low_conf_keys: set[EvalKey],
    graph_metrics: Mapping[str, object],
    model_metrics: Mapping[str, object],
    *,
    threshold: float,
    model_name: str = "VLM",
) -> str:
    """4-row confidence-gated routing diagnostic (F1 only; AP/AUC are not meaningful once
    graph and VLM score scales are mixed by routing).

    Rows 1-2: flat threshold@0.5 F1 restricted to exactly the pairs the router sends to
    the VLM (graph conf < threshold), for the graph and the VLM separately over the SAME
    key set -- directly comparable, and answers "does the VLM actually help on the cases
    it's asked about".

    Rows 3-4: flat threshold@0.5 F1 over the complementary retained pairs (graph
    conf >= threshold), for the graph and for the routed system separately. Row 4 is a
    consistency check, not a new measurement: routing answers these pairs with the raw
    graph logit (no VLM forward), so rows 3 and 4 are expected to be identical -- a
    divergence here would mean the router leaked a VLM prediction into a high-confidence
    slot.

    Rows 5-6: the locked per-target-pooled F1 (``compute_metrics.compute()``, via
    ``evaluate_predictions``) over the FULL population, for the graph alone and for the
    graph+VLM routed system -- answers "does the combined system beat the graph baseline
    overall". ``model_records``/``model_metrics`` come from a routed collector/evaluation
    (high-confidence pairs answered by the graph, low-confidence pairs by the VLM); rows
    1-2 recover the VLM's own predictions on the low-confidence pairs by filtering that
    same collector's records down to ``low_conf_keys`` -- no extra forward pass needed.
    """

    def flat_f1(records: Sequence[PredictionRecord], keys: set[EvalKey]) -> dict[str, float | None]:
        by_task: dict[str, list[tuple[int, int]]] = {"lah": [], "laeo": [], "sa": []}
        for record in records:
            if record.key in keys:
                by_task[record.key[1]].append(
                    (record.label, 1 if record.probability >= 0.5 else 0)
                )
        out = {task: _binary_f1_from_pairs(pairs) for task, pairs in by_task.items()}
        values = [value for value in out.values() if value is not None]
        out["macro"] = sum(values) / len(values) if values else None
        return out

    def num(value: object) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    all_keys = {record.key for record in model_records}
    high_conf_keys = all_keys - low_conf_keys

    row1 = flat_f1(graph_records, low_conf_keys)
    row2 = flat_f1(model_records, low_conf_keys)
    row3 = flat_f1(graph_records, high_conf_keys)
    row4 = flat_f1(model_records, high_conf_keys)
    n_low = sum(1 for record in model_records if record.key in low_conf_keys)
    n_high = len(high_conf_keys)
    n_full = len(model_records)

    lines = [
        f"===== ROUTING DIAGNOSTIC (threshold={threshold}): Graph-only vs {model_name} "
        "(F1 only -- AP/AUC omitted, routing mixes score scales) =====",
        "| Row | Scope | n | LAH F1 | LAEO F1 | SA F1 | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| 1. Graph-only | conf<{threshold} | {n_low} | {num(row1['lah'])} | "
        f"{num(row1['laeo'])} | {num(row1['sa'])} | {num(row1['macro'])} |",
        f"| 2. {model_name} | conf<{threshold} | {n_low} | {num(row2['lah'])} | "
        f"{num(row2['laeo'])} | {num(row2['sa'])} | {num(row2['macro'])} |",
        f"| 3. Graph-only | conf>={threshold} | {n_high} | {num(row3['lah'])} | "
        f"{num(row3['laeo'])} | {num(row3['sa'])} | {num(row3['macro'])} |",
        f"| 4. {model_name} | conf>={threshold} | {n_high} | {num(row4['lah'])} | "
        f"{num(row4['laeo'])} | {num(row4['sa'])} | {num(row4['macro'])} |",
        f"| 5. Graph-only | full | {n_full} | {num(graph_metrics.get('F1_LAH'))} | "
        f"{num(graph_metrics.get('F1_LAEO'))} | {num(graph_metrics.get('F1_SA'))} | "
        f"{num(graph_metrics.get('mean_social_f1'))} |",
        f"| 6. Graph+{model_name} routing | full | {n_full} | "
        f"{num(model_metrics.get('F1_LAH'))} | {num(model_metrics.get('F1_LAEO'))} | "
        f"{num(model_metrics.get('F1_SA'))} | {num(model_metrics.get('mean_social_f1'))} |",
    ]
    return "\n".join(lines)


def format_metrics_table(
    metrics: Mapping[str, object],
    annotations: SocialAnnotationDataset,
    *,
    title: str = "results",
) -> str:
    """Format one model's per-task AP/AUC/F1/Acc metrics as a readable table.

    Sibling of :func:`format_graph_model_table` for runs with no comparison model
    (e.g. a standalone ``raw_graph`` evaluation).
    """

    task_keys = {
        "LAH": ("LAH_AP", "LAH_AUC", "F1_LAH", "Acc_LAH", "lah"),
        "LAEO": ("LAEO_AP", "LAEO_AUC", "F1_LAEO", "Acc_LAEO", "laeo"),
        "SA": ("SA_AP", "SA_AUC", "F1_SA", "Acc_SA", "sa"),
        "Macro": ("social_ap", "social_auc", "mean_social_f1", "mean_social_accuracy", None),
    }
    class_counts = annotations.class_counts

    def number(value: object) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    lines = [
        f"===== {title.upper()} =====",
        "| Task | Pos / Neg | AP | AUC | F1 | Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_positive = total_negative = 0
    for name, (ap_key, auc_key, f1_key, acc_key, task) in task_keys.items():
        if task is None:
            count_text = f"{total_positive} / {total_negative}"
        else:
            counts = class_counts[task]
            total_positive += counts[1]
            total_negative += counts[0]
            count_text = f"{counts[1]} / {counts[0]}"
        lines.append(
            f"| {name} | {count_text} | {number(metrics.get(ap_key))} | "
            f"{number(metrics.get(auc_key))} | {number(metrics.get(f1_key))} | "
            f"{number(metrics.get(acc_key))} |"
        )
    return "\n".join(lines)


def format_metrics(metrics: Mapping[str, object], title: str = "", *, f1_only: bool = False) -> str:
    """One-line-per-task metric summary.

    ``f1_only`` drops AP/AUC/social_ap/social_auc: under confidence-gated routing these
    mix the graph's and VLM's independent score scales and are not meaningful (see
    :func:`format_graph_model_table`). Pass ``f1_only=cfg.routing.use``.
    """

    def value(key: str) -> str:
        item = metrics.get(key)
        return "N/A" if item is None else f"{float(item):.4f}"

    prefix = f"[{title}] " if title else ""
    if f1_only:
        return (
            f"{prefix}mean_f1={value('mean_social_f1')} mean_acc={value('mean_social_accuracy')} "
            "(AP/AUC omitted under routing)\n"
            f"  LAH : F1={value('F1_LAH')} Acc={value('Acc_LAH')}\n"
            f"  LAEO: F1={value('F1_LAEO')} Acc={value('Acc_LAEO')}\n"
            f"  SA  : F1={value('F1_SA')} Acc={value('Acc_SA')}"
        )
    return (
        f"{prefix}social_ap={value('social_ap')} social_auc={value('social_auc')} "
        f"mean_f1={value('mean_social_f1')} mean_acc={value('mean_social_accuracy')}\n"
        f"  LAH : AP={value('LAH_AP')} AUC={value('LAH_AUC')} F1={value('F1_LAH')} Acc={value('Acc_LAH')}\n"
        f"  LAEO: AP={value('LAEO_AP')} AUC={value('LAEO_AUC')} F1={value('F1_LAEO')} Acc={value('Acc_LAEO')}\n"
        f"  SA  : AP={value('SA_AP')} AUC={value('SA_AUC')} F1={value('F1_SA')} Acc={value('Acc_SA')}"
    )
