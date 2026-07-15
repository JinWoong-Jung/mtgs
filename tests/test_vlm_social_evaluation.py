import json
from types import SimpleNamespace

import pytest
import torch

import vlm.social.evaluate as evaluate_module
from vlm.social.evaluate import compare_results, run_evaluation
from vlm.social.objective import GraphLogitMLPControl, TaskBCELoss
from vlm.social.training import save_control_checkpoint
from vlm.social.data import SocialAnnotationDataset
from vlm.social.evaluation import (
    PredictionCollector,
    augment_social_metrics,
    format_graph_model_table,
    normalize_eval_key,
    raw_graph_predictions,
)


def _manifest(path):
    records = [
        # Raw LAH means person 1 looks at person 0. Internal graph index is [1,0].
        {"sid": "s0", "task": "lah", "i": 0, "j": 1, "ans": "yes"},
        {"sid": "s0", "task": "laeo", "i": 0, "j": 1, "ans": "no"},
        {"sid": "s0", "task": "sa", "i": 0, "j": 1, "ans": "yes"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


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
    feature = tmp_path / "feature.json"
    _result(raw, "raw graph", "raw_graph", 0.7)
    _result(feature, "features MLP", "features_mlp", 0.73)
    payload, table = compare_results([raw, feature])
    assert payload["ranking_by_social_ap"] == ["features MLP", "raw graph"]
    assert "| features MLP | features_mlp | 0.7300 | 0.0300 |" in table

    mismatch = tmp_path / "mismatch.json"
    _result(mismatch, "bad baseline", "vlm", 0.75, graph_ap=0.69)
    with pytest.raises(ValueError, match="raw graph metric mismatch"):
        compare_results([raw, mismatch])


def test_graph_control_checkpoint_runs_through_eval_cli_path(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    records = []
    cache = {}
    for index in range(6):
        task = ("lah", "laeo", "sa")[index % 3]
        sid = f"s{index}"
        records.append({
            "sid": sid,
            "task": task,
            "i": 0,
            "j": 1,
            "ans": "yes" if index % 2 else "no",
        })
        cache[sid] = {
            "lah_logits": torch.tensor([[0.0, -0.2], [0.3, 0.0]]),
            "laeo_logits": torch.tensor([[0.0, -0.4], [0.2, 0.0]]),
            "sa_logits": torch.tensor([[0.0, 0.5], [-0.1, 0.0]]),
        }
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    graph_path = tmp_path / "graph.pt"
    torch.save(cache, graph_path)
    gtmeta = tmp_path / "gtmeta.pt"
    torch.save({}, gtmeta)
    config = tmp_path / "config.yaml"
    config.write_text("""
control:
  hidden_dim: 8
  dropout: 0.0
  val_bs: 2
val:
  num_workers: 0
""")

    checkpoint = tmp_path / "checkpoint"
    control = GraphLogitMLPControl(hidden_dim=8)
    criterion = TaskBCELoss()
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    save_control_checkpoint(
        checkpoint,
        control,
        criterion,
        optimizer,
        scheduler,
        {"mode": "graph_mlp", "epoch": 1, "global_step": 4},
    )

    fixed_metrics = {
        "F1_LAH": 0.7,
        "F1_LAEO": 0.6,
        "F1_SA": 0.5,
        "LAH_AP": 0.8,
        "LAEO_AP": 0.7,
        "SA_AP": 0.6,
        "LAH_AUC": 0.9,
        "LAEO_AUC": 0.8,
        "SA_AUC": 0.7,
        "social_ap": 0.7,
        "social_auc": 0.8,
        "mean_social_f1": 0.6,
        "detail": "locked metric detail",
    }
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_predictions",
        lambda *args, **kwargs: dict(fixed_metrics),
    )
    output = tmp_path / "output"
    result = run_evaluation(SimpleNamespace(
        mode="graph_mlp",
        name="scalar control",
        checkpoint=str(checkpoint),
        config=str(config),
        manifest=str(manifest),
        frame_root="",
        graph_feats=str(graph_path),
        gtmeta=str(gtmeta),
        output_dir=str(output),
        batch_size=2,
        num_workers=0,
        threshold=0.5,
        device="cpu",
    ))
    assert result["mode"] == "graph_mlp"
    assert result["checkpoint_state"]["epoch"] == 1
    assert result["predictions"] == 6
    assert (output / "result.json").exists()
    saved = torch.load(output / "predictions.pt", weights_only=False)
    assert len(saved["probabilities"]) == 6



def test_text_generative_eval_uses_text_dataset_and_collate(tmp_path, monkeypatch):
    """Standalone eval must rebuild the same text-evidence input contract as training."""
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest)
    cache = {
        "s0": {
            task + "_logits": torch.tensor([[0.0, -0.2], [0.3, 0.0]])
            for task in ("lah", "laeo", "sa")
        }
    }
    graph_path = tmp_path / "graph.pt"
    torch.save(cache, graph_path)
    gtmeta = tmp_path / "gtmeta.pt"
    torch.save({}, gtmeta)
    config = tmp_path / "config.yaml"
    config.write_text("""
input:
  reuse_frozen_vision: true
  group_by_frame: true
  vision_cache_size: 4
loss:
  lm_aux_weight: 0.0
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
            captured["dataset"] = kwargs
            self.annotations = SocialAnnotationDataset(manifest)
            self.graph_cache = graph_cache
            self.graph_evidence = kwargs["graph_evidence"]

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
        evaluate_module, "make_text_generative_collate",
        lambda processor, *, reuse_vision: captured.setdefault(
            "text_collate_reuse", reuse_vision
        ) or object(),
    )
    monkeypatch.setattr(
        evaluate_module, "make_generative_collate",
        lambda processor: pytest.fail("text evidence must not select gtoken collate"),
    )
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
    assert captured["dataset"]["graph_evidence"] == "text"
    assert "draw_bboxes" not in captured["dataset"]
    assert captured["text_collate_reuse"] is True
    assert captured["collect_kwargs"]["reuse_vision"] is True
    assert captured["collect_kwargs"]["group_by_frame"] is True
    assert captured["closed"] is True
    assert result["variant"]["graph_evidence"] == "text"
    assert result["variant"]["reuse_frozen_vision"] is True
    assert "draw_bboxes" not in result["variant"]
