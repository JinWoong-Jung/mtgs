import json
from types import SimpleNamespace

import pytest
import torch

import vlm.social.evaluate as evaluate_module
from vlm.social.evaluate import compare_results, run_evaluation
from vlm.social.data import SocialAnnotationDataset
from vlm.social.evaluation import (
    PredictionCollector,
    PredictionRecord,
    augment_social_metrics,
    format_graph_model_table,
    format_metrics,
    format_metrics_table,
    format_routing_comparison_table,
    normalize_eval_key,
    raw_graph_predictions,
    routing_low_confidence_keys,
)


def _manifest(path):
    records = [
        # Raw LAH means person 1 looks at person 0. Internal graph index is [1,0].
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "no"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "yes"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _logits_only_cache():
    """Minimal graph cache: only the per-task pair logits the raw baseline needs."""
    logits = torch.tensor([[0.0, -0.2], [0.3, 0.0]])
    return {"s0": {task + "_logits": logits.clone() for task in ("lah", "laeo", "sa")}}


def test_eval_key_keeps_raw_lah_direction_and_sorts_only_symmetric_tasks():
    assert normalize_eval_key(("s", "lah", 0, 1)) == ("s", "lah", 0, 1)
    assert normalize_eval_key(("s", "lah", 1, 0)) == ("s", "lah", 1, 0)
    assert normalize_eval_key(("s", "laeo", 1, 0)) == ("s", "laeo", 0, 1)
    assert normalize_eval_key(("s", "sa", 1, 0)) == ("s", "sa", 0, 1)


def test_prediction_collector_is_strict_and_preserves_graph_delta_diagnostics():
    collector = PredictionCollector()
    keys = [("s", "lah", 0, 1), ("s", "sa", 1, 0)]
    logits = torch.tensor([2.0, -1.0])
    graph = torch.tensor([1.5, -0.5])
    collector.add_batch(
        keys,
        logits,
        torch.tensor([1.0, 0.0]),
        graph_logits=graph,
        delta_logits=logits - graph,
    )
    collector.assert_complete([("s", "lah", 0, 1), ("s", "sa", 0, 1)])
    assert set(collector.probabilities) == {
        ("s", "lah", 0, 1),
        ("s", "sa", 0, 1),
    }
    assert collector.records[0].graph_logit == 1.5
    assert collector.records[0].delta_logit == 0.5

    with pytest.raises(ValueError, match="duplicate normalized"):
        collector.add_batch(
            [("s", "sa", 0, 1)], torch.zeros(1), torch.zeros(1)
        )
    with pytest.raises(ValueError, match="coverage mismatch"):
        collector.assert_complete([("s", "lah", 0, 1)])


def test_raw_graph_predictions_use_canonical_indices_but_raw_eval_keys(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    annotations = SocialAnnotationDataset(manifest)
    logits = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    cache = {
        "s0": {
            "lah_logits": logits,
            "laeo_logits": logits + 10,
            "sa_logits": logits + 20,
        }
    }
    collector = raw_graph_predictions(annotations, cache)
    records = {record.key: record for record in collector.records}
    # Key remains raw (target=0, looker=1), but value came from graph[looker=1,target=0].
    lah = records[("s0", "lah", 0, 1)]
    assert lah.graph_logit == 2.0
    assert lah.logit == 2.0 and lah.delta_logit == 0.0
    # Symmetric task logit is the two graph directions' mean.
    assert records[("s0", "laeo", 0, 1)].graph_logit == 11.5
    assert records[("s0", "sa", 0, 1)].graph_logit == 21.5


def test_metric_augmentation_adds_sa_f1_and_complete_three_task_means():
    detail = """
----- LAEO -----
F1    : 0.6000  (thr=0.5)
----- LAH -----
F1    : 0.7000  (thr=0.5)
----- CoAtt (SA) -----
F1    : 0.8000  (thr=0.5)
"""
    metrics = augment_social_metrics({
        "F1_LAH": 0.7,
        "F1_LAEO": 0.6,
        "LAH_AP": 0.9,
        "LAEO_AP": 0.6,
        "SA_AP": 0.75,
        "LAH_AUC": 0.8,
        "LAEO_AUC": 0.7,
        "SA_AUC": 0.9,
        "detail": detail,
    })
    assert metrics["F1_SA"] == 0.8
    assert metrics["social_ap"] == pytest.approx(0.75)
    assert metrics["social_auc"] == pytest.approx(0.8)
    assert metrics["mean_social_f1"] == pytest.approx(0.7)


def test_graph_model_table_uses_exact_manifest_class_counts(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    annotations = SocialAnnotationDataset(manifest)
    graph = {
        "LAH_AP": 0.4, "LAH_AUC": 0.5, "F1_LAH": 0.2,
        "LAEO_AP": 0.3, "LAEO_AUC": 0.4, "F1_LAEO": 0.1,
        "SA_AP": 0.5, "SA_AUC": 0.6, "F1_SA": 0.3,
        "social_ap": 0.4, "social_auc": 0.5, "mean_social_f1": 0.2,
    }
    model = {key: value + 0.1 for key, value in graph.items()}
    table = format_graph_model_table(graph, model, annotations, model_name="VLM")
    assert "===== TEST: RAW GRAPH vs VLM (same manifest) =====" in table
    assert "| LAH | 1 / 0 | 0.4000 | 0.5000 | +0.1000 |" in table
    assert "| Macro | 2 / 1 | 0.4000 | 0.5000 | +0.1000 |" in table


def test_graph_model_table_f1_only_drops_ap_auc_columns(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    annotations = SocialAnnotationDataset(manifest)
    graph = {
        "LAH_AP": 0.4, "LAH_AUC": 0.5, "F1_LAH": 0.2,
        "LAEO_AP": 0.3, "LAEO_AUC": 0.4, "F1_LAEO": 0.1,
        "SA_AP": 0.5, "SA_AUC": 0.6, "F1_SA": 0.3,
        "social_ap": 0.4, "social_auc": 0.5, "mean_social_f1": 0.2,
    }
    model = {key: value + 0.1 for key, value in graph.items()}
    table = format_graph_model_table(graph, model, annotations, model_name="VLM", f1_only=True)
    assert "GRAPH-ONLY" in table
    assert "GRAPH+VLM ROUTING" in table
    assert "F1 only" in table
    assert "| Task | Pos / Neg | Graph-only F1 | Graph+VLM routing F1 | ΔF1 |" in table
    assert "| LAH | 1 / 0 | 0.2000 | 0.3000 | +0.1000 |" in table
    assert "| Macro | 2 / 1 | 0.2000 | 0.3000 | +0.1000 |" in table
    # AP/AUC values must not leak into the routed table.
    assert "0.4000 | 0.5000 |" not in table  # graph AP/AUC pair would render this way
    assert "AP" not in table.split("\n")[1]  # header row has no AP column


def test_format_metrics_f1_only_drops_ap_auc(tmp_path):
    metrics = {
        "LAH_AP": 0.4, "LAH_AUC": 0.5, "F1_LAH": 0.2, "Acc_LAH": 0.6,
        "LAEO_AP": 0.3, "LAEO_AUC": 0.4, "F1_LAEO": 0.1, "Acc_LAEO": 0.5,
        "SA_AP": 0.5, "SA_AUC": 0.6, "F1_SA": 0.3, "Acc_SA": 0.7,
        "social_ap": 0.4, "social_auc": 0.5, "mean_social_f1": 0.2, "mean_social_accuracy": 0.6,
    }
    text = format_metrics(metrics, "vlm", f1_only=True)
    assert "AP=" not in text
    assert "AUC=" not in text
    assert "F1=0.2000" in text
    assert "mean_f1=0.2000" in text
    assert "mean_acc=0.6000" in text
    # Default (f1_only=False) keeps AP/AUC for non-routed runs.
    full = format_metrics(metrics, "raw_graph")
    assert "AP=0.4000" in full
    assert "AUC=0.5000" in full


def _rec(sid, task, i, j, label, prob):
    return PredictionRecord(
        key=normalize_eval_key((sid, task, i, j)),
        label=label,
        probability=prob,
        logit=0.0,
        graph_probability=None,
        graph_logit=None,
        delta_logit=None,
    )


def test_routing_comparison_table_four_rows():
    # Only (s0, lah, 0, 1) is in the low-confidence set; (s1, lah, 0, 1) is not, so it's
    # excluded from rows 1-2 but still counted in rows 3-4 (n_full).
    graph_records = [
        _rec("s0", "lah", 0, 1, label=1, prob=0.4),   # low-conf: graph predicts 0 (wrong)
        _rec("s1", "lah", 0, 1, label=0, prob=0.5),   # not low-conf: ignored by rows 1-2
    ]
    model_records = [
        _rec("s0", "lah", 0, 1, label=1, prob=0.9),   # low-conf: VLM predicts 1 (correct)
        _rec("s1", "lah", 0, 1, label=0, prob=0.5),
    ]
    low_conf_keys = {normalize_eval_key(("s0", "lah", 0, 1))}
    graph_metrics = {"F1_LAH": 0.5, "F1_LAEO": 0.6, "F1_SA": 0.7, "mean_social_f1": 0.6}
    model_metrics = {"F1_LAH": 0.8, "F1_LAEO": 0.7, "F1_SA": 0.9, "mean_social_f1": 0.8}

    table = format_routing_comparison_table(
        graph_records, model_records, low_conf_keys, graph_metrics, model_metrics,
        threshold=0.8, model_name="VLM",
    )
    assert "ROUTING DIAGNOSTIC (threshold=0.8)" in table
    assert "F1 only" in table
    # Row 1: graph on the low-conf pair alone -- label=1, pred=0 -> F1=0.
    assert "| 1. Graph-only | conf<0.8 | 1 | 0.0000 |" in table
    # Row 2: VLM on the SAME low-conf pair -- label=1, pred=1 -> F1=1.
    assert "| 2. VLM | conf<0.8 | 1 | 1.0000 |" in table
    # Rows 3-4: the official pooled metrics passed straight through, full population (n=2).
    assert "| 3. Graph-only | full | 2 | 0.5000 |" in table
    assert "| 4. Graph+VLM routing | full | 2 | 0.8000 |" in table


def test_routing_low_confidence_keys_matches_graph_confidence(tmp_path):
    from vlm.social.input import graph_confidence

    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    annotations = SocialAnnotationDataset(manifest)
    cache = _logits_only_cache()
    threshold = 0.6

    expected = {
        normalize_eval_key(sample.eval_key)
        for sample in annotations.samples
        if graph_confidence(sample, cache[sample.sid]) < threshold
    }
    got = routing_low_confidence_keys(annotations, cache, threshold)
    assert got == expected
    assert got  # sanity: this fixture has at least one low-confidence pair at 0.6


def test_metrics_table_uses_exact_manifest_class_counts(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    annotations = SocialAnnotationDataset(manifest)
    metrics = {
        "LAH_AP": 0.4, "LAH_AUC": 0.5, "F1_LAH": 0.2, "Acc_LAH": 0.6,
        "LAEO_AP": 0.3, "LAEO_AUC": 0.4, "F1_LAEO": 0.1, "Acc_LAEO": 0.5,
        "SA_AP": 0.5, "SA_AUC": 0.6, "F1_SA": 0.3, "Acc_SA": 0.7,
        "social_ap": 0.4, "social_auc": 0.5, "mean_social_f1": 0.2, "mean_social_accuracy": 0.6,
    }
    table = format_metrics_table(metrics, annotations, title="raw_graph")
    assert "===== RAW_GRAPH =====" in table
    assert "| Task | Pos / Neg | AP | AUC | F1 | Acc |" in table
    assert "| LAH | 1 / 0 | 0.4000 | 0.5000 | 0.2000 | 0.6000 |" in table
    assert "| Macro | 2 / 1 | 0.4000 | 0.5000 | 0.2000 | 0.6000 |" in table


def _result(path, name, mode, social_ap, graph_ap=0.7):
    base = {
        "social_ap": graph_ap,
        "LAH_AP": 0.8,
        "LAEO_AP": 0.6,
        "SA_AP": 0.7,
    }
    metrics = dict(base, social_ap=social_ap)
    value = {
        "name": name,
        "mode": mode,
        "manifest": "/same/manifest.jsonl",
        "graph_feats": "/same/graph.pt",
        "gtmeta": "/same/gtmeta.pt",
        "threshold": 0.5,
        "metrics": metrics,
        "raw_graph_metrics": base,
        "delta_vs_graph": {"social_ap": social_ap - graph_ap},
    }
    path.write_text(json.dumps(value))


def test_compare_results_requires_same_provenance_and_raw_baseline(tmp_path):
    raw = tmp_path / "raw.json"
    other = tmp_path / "vlm.json"
    _result(raw, "raw graph", "raw_graph", 0.7)
    _result(other, "graph VLM", "vlm", 0.73)
    payload, table = compare_results([raw, other])
    assert payload["ranking_by_social_ap"] == ["graph VLM", "raw graph"]
    assert "| graph VLM | vlm | 0.7300 | 0.0300 |" in table

    mismatch = tmp_path / "mismatch.json"
    _result(mismatch, "bad baseline", "vlm", 0.75, graph_ap=0.69)
    with pytest.raises(ValueError, match="raw graph metric mismatch"):
        compare_results([raw, mismatch])


def test_raw_graph_eval_runs_from_minimal_logits_cache(tmp_path, monkeypatch):
    """Regression: the raw-graph baseline must evaluate from a cache holding ONLY the
    per-task ``*_logits`` (no v_src/edge_pp/heatmap tensor stack, which was removed).
    It builds a plain SocialAnnotationDataset and scores via sample_graph_logit only."""
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    graph_path = tmp_path / "graph.pt"
    torch.save(_logits_only_cache(), graph_path)
    gtmeta = tmp_path / "gtmeta.pt"
    torch.save({}, gtmeta)
    config = tmp_path / "config.yaml"
    config.write_text("val:\n  num_workers: 0\n")

    fixed_metrics = {
        "F1_LAH": 0.7, "F1_LAEO": 0.6, "F1_SA": 0.5,
        "LAH_AP": 0.8, "LAEO_AP": 0.7, "SA_AP": 0.6,
        "LAH_AUC": 0.9, "LAEO_AUC": 0.8, "SA_AUC": 0.7,
        "social_ap": 0.7, "social_auc": 0.8, "mean_social_f1": 0.6,
        "detail": "locked metric detail",
    }
    monkeypatch.setattr(
        evaluate_module, "evaluate_predictions",
        lambda *args, **kwargs: dict(fixed_metrics),
    )

    output = tmp_path / "output"
    result = run_evaluation(SimpleNamespace(
        mode="raw_graph", name="raw", checkpoint="",
        config=str(config), manifest=str(manifest), frame_root="",
        graph_feats=str(graph_path), gtmeta=str(gtmeta), output_dir=str(output),
        batch_size=0, num_workers=0, threshold=0.5, device="cpu",
    ))
    assert result["mode"] == "raw_graph"
    assert result["checkpoint"] is None
    saved = torch.load(output / "predictions.pt", weights_only=False)
    assert len(saved["probabilities"]) == 3


def test_text_generative_eval_uses_text_dataset_and_generative_objective(tmp_path, monkeypatch):
    """Standalone VLM eval rebuilds the text-evidence input contract and scores via the
    generative objective (no gtoken/control paths)."""
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    graph_path = tmp_path / "graph.pt"
    torch.save(_logits_only_cache(), graph_path)
    gtmeta = tmp_path / "gtmeta.pt"
    torch.save({}, gtmeta)
    config = tmp_path / "config.yaml"
    config.write_text("""
input:
  reuse_frozen_vision: true
  group_by_frame: true
  vision_cache_size: 4
train:
  seed: 7
val:
  bs: 2
  num_workers: 0
""")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    captured = {}

    class FakeDataset:
        def __init__(self, manifest, frame_root, graph_cache, **kwargs):
            captured["dataset_kwargs"] = kwargs
            self.annotations = SocialAnnotationDataset(manifest)
            self.graph_cache = graph_cache

        def __len__(self):
            return len(self.annotations)

    class FakeObjective:
        def close(self):
            captured["closed"] = True

    fixed_metrics = {
        "F1_LAH": 0.7, "F1_LAEO": 0.6, "F1_SA": 0.5,
        "LAH_AP": 0.8, "LAEO_AP": 0.7, "SA_AP": 0.6,
        "LAH_AUC": 0.9, "LAEO_AUC": 0.8, "SA_AUC": 0.7,
        "social_ap": 0.7, "social_auc": 0.8, "mean_social_f1": 0.6,
        "detail": "locked metric detail",
    }
    monkeypatch.setattr(evaluate_module, "_resolve_device", lambda *_: torch.device("cpu"))
    monkeypatch.setattr(evaluate_module, "_processor", lambda *_: object())
    monkeypatch.setattr(evaluate_module, "SocialInputDataset", FakeDataset)
    monkeypatch.setattr(
        evaluate_module, "build_generative_objective",
        lambda *args: (FakeObjective(), ()),
    )
    monkeypatch.setattr(
        evaluate_module, "_restore_vlm_modules",
        lambda *args: {"mode": "vlm", "epoch": 1},
    )
    monkeypatch.setattr(
        evaluate_module, "collect_generative_predictions",
        lambda module, dataset, processor, **kwargs: (
            captured.setdefault("collect_kwargs", kwargs),
            raw_graph_predictions(dataset.annotations, dataset.graph_cache),
        )[1],
    )
    monkeypatch.setattr(
        evaluate_module, "evaluate_predictions",
        lambda *args, **kwargs: dict(fixed_metrics),
    )

    result = run_evaluation(SimpleNamespace(
        mode="vlm", name="text evidence", checkpoint=str(checkpoint),
        config=str(config), manifest=str(manifest), frame_root=str(tmp_path),
        graph_feats=str(graph_path), gtmeta=str(gtmeta), output_dir=str(tmp_path / "output"),
        batch_size=2, num_workers=0, threshold=0.5, device="cpu",
    ))
    # SocialInputDataset is constructed without any graph_evidence/output_mode selector.
    assert "graph_evidence" not in captured["dataset_kwargs"]
    assert "output_mode" not in captured["dataset_kwargs"]
    assert captured["collect_kwargs"]["reuse_vision"] is True
    assert captured["collect_kwargs"]["group_by_frame"] is True
    assert captured["closed"] is True
    assert result["variant"]["graph_evidence"] == "text"
    assert result["variant"]["reuse_frozen_vision"] is True
