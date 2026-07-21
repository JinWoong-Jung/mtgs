"""Checkpoint evaluation and comparison for the pair-wise social-gaze VLM.

Two modes only: the frozen-graph ``raw_graph`` baseline and the generative text-evidence
``vlm``. Each run reports both, so a single ``vlm`` run yields the graph-vs-VLM table.

Examples::

    python -m vlm.social.evaluate run --mode raw_graph ...
    python -m vlm.social.evaluate run --mode vlm --name my_vlm --checkpoint .../best ...
    python -m vlm.social.evaluate compare result_dirs_or_json_files...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from omegaconf import OmegaConf

from vlm.social.evaluation import (
    CORE_METRIC_KEYS,
    PredictionCollector,
    evaluate_predictions,
    format_graph_model_table,
    format_metrics,
    format_metrics_table,
    format_routing_comparison_table,
    metric_deltas,
    metric_payload,
    raw_graph_predictions,
    routing_low_confidence_keys,
)
from vlm.social.data import SocialAnnotationDataset
from vlm.social.input import SocialInputDataset
from vlm.social.training import (
    _load_graph_cache,
    _processor,
    _restore_vlm_modules,
    _route_threshold,
    build_generative_objective,
    collect_generative_predictions,
    select_generative_builders,
)


EVAL_MODES = ("raw_graph", "vlm")


def _resolve_device(value: str, mode: str) -> torch.device:
    if value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if mode == "vlm" and device.type != "cuda":
        raise RuntimeError("Qwen pair evaluation requires CUDA")
    return device


def _validate_checkpoint_mode(state: Mapping[str, Any], mode: str) -> None:
    saved = state.get("mode")
    if saved is not None and saved != mode:
        raise ValueError(f"checkpoint mode is {saved!r}, requested evaluator mode is {mode!r}")


def _checkpoint_summary(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    keys = (
        "mode",
        "epoch",
        "global_step",
        "monitor",
        "monitor_mode",
        "selection_score",
        "best_score",
    )
    return {key: state.get(key) for key in keys if key in state}


def _variant_settings(cfg, mode: str) -> dict[str, Any]:
    if mode == "vlm":
        input_cfg = cfg.get("input", {})
        builders = select_generative_builders(cfg)
        reuse_vision = builders.reuse_vision
        return {
            "output": "generative",
            "graph_evidence": builders.graph_evidence_mode,
            "graph_token_features": list(builders.graph_token_features),
            "include_graph_evidence": builders.include_graph_evidence,
            "lm_aux_weight": float(cfg.get("loss", {}).get("lm_aux_weight", 0.0)),
            "reuse_frozen_vision": reuse_vision,
            "group_by_frame": reuse_vision
            and bool(input_cfg.get("group_by_frame", False)),
            "vision_cache_size": int(input_cfg.get("vision_cache_size", 0)),
        }
    return {}


def run_evaluation(args) -> dict[str, Any]:
    mode = str(args.mode)
    if mode not in EVAL_MODES:
        raise ValueError(f"mode must be one of {EVAL_MODES}, got {mode!r}")
    cfg = OmegaConf.load(args.config)
    # Confidence-gated routing mixes the frozen graph's and the VLM's independent score
    # scales, so AP/AUC over the combined predictions are not meaningful once routed --
    # only report F1 in that case (see format_graph_model_table / format_metrics docstrings).
    routing_on = mode == "vlm" and bool(cfg.get("routing", {}).get("use", False))
    routing_threshold = float(cfg.get("routing", {}).get("threshold", 0.8))
    graph_cache = _load_graph_cache(args.graph_feats)
    device = _resolve_device(args.device, mode)
    threshold = float(args.threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if not getattr(args, "wandb_off", True):
        import wandb as wandb_module

        wandb = wandb_module
        run_id = str(getattr(args, "wandb_run_id", "")).strip()
        init_kwargs = dict(
            project="MTGS",
            entity="gaze-social",
            group="vlm",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if run_id:
            init_kwargs.update(id=run_id, resume="must")
        else:
            init_kwargs.update(
                job_type="test", name=args.name or f"{cfg.experiment.name}_{mode}"
            )
        wandb.init(**init_kwargs)
    processor = objective = None
    dataset = None
    input_cfg = cfg.get("input", {})
    generative_builders = select_generative_builders(cfg) if mode == "vlm" else None
    reuse_vision = mode == "vlm" and generative_builders.reuse_vision
    group_by_frame = reuse_vision and bool(
        input_cfg.get("group_by_frame", False)
    )
    if mode == "vlm":
        if not args.frame_root:
            raise ValueError("vlm evaluation requires --frame_root")
        if not args.checkpoint:
            raise ValueError("vlm evaluation requires --checkpoint")
        checkpoint = Path(args.checkpoint)
        processor = _processor(cfg, checkpoint)
        dataset = SocialInputDataset(
            args.manifest,
            args.frame_root,
            graph_cache,
            raw_image_cache_size=int(cfg.val.get("raw_image_cache_size", 16)),
            generative_prompt_seed=int(cfg.val.get("prompt_seed", cfg.train.get("seed", 101))),
            include_graph_evidence=generative_builders.include_graph_evidence,
            routing_threshold=_route_threshold(cfg),
            graph_evidence_mode=generative_builders.graph_evidence_mode,
            graph_token_features=generative_builders.graph_token_features,
            draw_pair_bboxes=generative_builders.draw_pair_bboxes,
            draw_gaze_arrows=generative_builders.draw_gaze_arrows,
        )
        annotations = dataset.annotations
    else:  # raw_graph: no VLM, no frames -- only the frozen graph logits are scored.
        checkpoint = None
        annotations = SocialAnnotationDataset(args.manifest)

    expected_keys = [sample.eval_key for sample in annotations.samples]
    expected_sids = {sample.sid for sample in annotations.samples}
    graph_collector = raw_graph_predictions(annotations.samples, graph_cache)
    graph_metrics = evaluate_predictions(
        args.gtmeta,
        graph_collector.probabilities,
        expected_sids=expected_sids,
        threshold=threshold,
    )
    print(format_metrics(graph_metrics, "raw_graph", f1_only=routing_on), flush=True)
    if wandb is not None:
        wandb.log({
            f"metric/test_graph/{key}": value
            for key, value in metric_payload(graph_metrics).items()
            if value is not None
        })

    state = None
    stats = None
    if mode == "raw_graph":
        collector = graph_collector
        metrics = graph_metrics
    else:
        if not routing_on:
            # Routing needs graph_collector.records for the final diagnostic table
            # (format_routing_comparison_table); otherwise free it now as before.
            del graph_collector
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        objective, _ = build_generative_objective(cfg, processor, device, checkpoint)
        module = objective
        state = _restore_vlm_modules(objective, checkpoint)
        _validate_checkpoint_mode(state, mode)

        batch_size = int(cfg.val.bs) if args.batch_size <= 0 else int(args.batch_size)
        default_workers = int(cfg.val.get("num_workers", 4))
        num_workers = default_workers if args.num_workers < 0 else int(args.num_workers)
        collector = PredictionCollector()
        try:
            collector = collect_generative_predictions(
                module, dataset, processor,
                batch_size=batch_size, num_workers=num_workers,
                device=device, description=f"Test [{args.name or mode}]",
                reuse_vision=generative_builders.reuse_vision,
                group_by_frame=group_by_frame,
                route_threshold=_route_threshold(cfg),
            )
        finally:
            if objective is not None:
                objective.close()
        collector.assert_complete(expected_keys)
        metrics = evaluate_predictions(
            args.gtmeta,
            collector.probabilities,
            expected_sids=expected_sids,
            threshold=threshold,
        )
        print(format_metrics(metrics, args.name or mode, f1_only=routing_on), flush=True)

    if wandb is not None:
        wandb.log({
            f"metric/test/{key}": value
            for key, value in metric_payload(metrics).items()
            if value is not None
        })

    collector.save(output_dir / "predictions.pt")
    details = (
        "===== RAW GRAPH =====\n"
        + str(graph_metrics.get("detail", ""))
        + "\n\n===== MODEL =====\n"
        + str(metrics.get("detail", ""))
    )
    (output_dir / "detail.txt").write_text(details, encoding="utf-8")
    result = {
        "name": args.name or mode,
        "mode": mode,
        "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "checkpoint_state": _checkpoint_summary(state),
        "variant": _variant_settings(cfg, mode),
        "config": str(Path(args.config).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "graph_feats": str(Path(args.graph_feats).resolve()),
        "gtmeta": str(Path(args.gtmeta).resolve()),
        "frame_root": None if not args.frame_root else str(Path(args.frame_root).resolve()),
        "threshold": threshold,
        "frames": len(expected_sids),
        "predictions": len(collector),
        "eval_stats": None if stats is None else asdict(stats),
        "metrics": metric_payload(metrics),
        "raw_graph_metrics": metric_payload(graph_metrics),
        "delta_vs_graph": metric_deltas(metrics, graph_metrics),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[pair-eval] wrote {result_path}", flush=True)
    # Keep this as the final stdout block for a redirected test .out file.
    if mode != "raw_graph":
        if routing_on:
            low_conf_keys = routing_low_confidence_keys(
                dataset.annotations, graph_cache, routing_threshold
            )
            print(
                format_routing_comparison_table(
                    graph_collector.records,
                    collector.records,
                    low_conf_keys,
                    graph_metrics,
                    metrics,
                    threshold=routing_threshold,
                    model_name=args.name or mode,
                ),
                flush=True,
            )
        else:
            print(
                format_graph_model_table(
                    graph_metrics,
                    metrics,
                    dataset.annotations,
                    model_name=args.name or mode,
                ),
                flush=True,
            )
    else:
        print(
            format_metrics_table(metrics, annotations, title=args.name or mode),
            flush=True,
        )
    if wandb is not None:
        wandb.finish()
    return result


def _result_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "result.json" if path.is_dir() else path


def load_result(path: str | Path) -> dict[str, Any]:
    resolved = _result_path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "metrics" not in value:
        raise ValueError(f"not a pair evaluation result: {resolved}")
    return value


def compare_results(paths: Sequence[str | Path]) -> tuple[dict[str, Any], str]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two result files")
    results = [load_result(path) for path in paths]
    names = [str(result.get("name")) for result in results]
    if len(set(names)) != len(names):
        raise ValueError(f"comparison result names must be unique, got {names}")
    provenance_keys = ("manifest", "graph_feats", "gtmeta", "threshold")
    reference = {key: results[0].get(key) for key in provenance_keys}
    for result in results[1:]:
        current = {key: result.get(key) for key in provenance_keys}
        if current != reference:
            raise ValueError(
                f"comparison provenance mismatch for {result.get('name')}: "
                f"expected {reference}, got {current}"
            )
        for key in CORE_METRIC_KEYS:
            expected = results[0]["raw_graph_metrics"].get(key)
            actual = result["raw_graph_metrics"].get(key)
            if expected is None or actual is None:
                if expected != actual:
                    raise ValueError(f"raw graph metric mismatch for {key}")
            elif abs(float(expected) - float(actual)) > 1e-9:
                raise ValueError(f"raw graph metric mismatch for {key}: {expected} vs {actual}")

    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append({
            "name": result["name"],
            "mode": result["mode"],
            "metrics": {key: metrics.get(key) for key in CORE_METRIC_KEYS},
            "delta_vs_graph": result.get("delta_vs_graph", {}),
        })
    ranked = sorted(
        rows,
        key=lambda row: float("-inf")
        if row["metrics"].get("social_ap") is None
        else float(row["metrics"]["social_ap"]),
        reverse=True,
    )
    payload = {
        "provenance": reference,
        "rows": rows,
        "ranking_by_social_ap": [row["name"] for row in ranked],
    }

    def number(value) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    header = (
        "| name | mode | social AP | Δgraph | LAH AP | LAEO AP | SA AP | "
        "LAH F1 | LAEO F1 | SA F1 |"
    )
    separator = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        metrics = row["metrics"]
        delta = row["delta_vs_graph"].get("social_ap")
        lines.append(
            f"| {row['name']} | {row['mode']} | {number(metrics.get('social_ap'))} | "
            f"{number(delta)} | {number(metrics.get('LAH_AP'))} | "
            f"{number(metrics.get('LAEO_AP'))} | {number(metrics.get('SA_AP'))} | "
            f"{number(metrics.get('F1_LAH'))} | {number(metrics.get('F1_LAEO'))} | "
            f"{number(metrics.get('F1_SA'))} |"
        )
    return payload, "\n".join(lines)


def _run_parser(subparsers) -> None:
    parser = subparsers.add_parser("run", help="evaluate one raw/control/VLM variant")
    parser.add_argument("--mode", required=True, choices=EVAL_MODES)
    parser.add_argument("--name", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="mtgs/config/config_vlm.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame_root", default="")
    parser.add_argument("--graph_feats", required=True)
    parser.add_argument("--gtmeta", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wandb_run_id", default="")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb_off", action="store_true")


def _compare_parser(subparsers) -> None:
    parser = subparsers.add_parser("compare", help="compare common-harness result files")
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", default="")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    _run_parser(subparsers)
    _compare_parser(subparsers)
    args = parser.parse_args()
    if args.command == "run":
        run_evaluation(args)
    else:
        payload, table = compare_results(args.results)
        print(table)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[pair-eval] comparison -> {path}")


if __name__ == "__main__":
    main()
