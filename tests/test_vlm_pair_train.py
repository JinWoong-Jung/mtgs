import pytest
import torch

from vlm.pair_head import (
    GraphFeatureMLPControl,
    GraphLogitMLPControl,
    PairGenerativeObjective,
    PairSocialObjective,
    PairTaskBCELoss,
    PairTaskResidualDecoder,
)
from vlm.pair_model import PairSocialVLM, TextGenerativeVLM
from vlm.pair_features import PairGraphBatch
from vlm.pair_eval import PairPredictionCollector
from vlm.pair_prompt import PAIR_SPECIAL_TOKENS
from vlm.train_pair import (
    EpochStats,
    FrameGroupedBatchSampler,
    append_epoch_report,
    checkpoint_score,
    format_epoch_report,
    make_epoch_loader,
    optimizer_steps_per_epoch,
    partition_vlm_parameters,
    restore_control_checkpoint,
    run_epoch,
    save_control_checkpoint,
    score_improved,
    validate_sampler_loss_compatibility,
)


class _TinyControlDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.graph = torch.tensor([-2.0, -1.0, 1.0, 2.0, -0.5, 0.5])
        self.tasks = torch.tensor([0, 1, 2, 0, 1, 2])
        self.labels = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, index):
        return self.graph[index], self.tasks[index], self.labels[index]


def _collate(items):
    graph, tasks, labels = zip(*items)
    return {
        "graph_logits": torch.stack(graph),
        "task_ids": torch.stack(tasks),
        "pair_labels": torch.stack(labels),
        "eval_keys": [None] * len(items),
    }


def _feature_collate(items):
    indices = torch.tensor(items, dtype=torch.long)
    tasks = tuple(("lah", "laeo", "sa")[index % 3] for index in items)
    batch = len(items)
    return {
        "pair_graph": PairGraphBatch(
            tasks=tasks,
            person_features=torch.randn(batch, 2, 3, 4),
            person_channel_present=torch.ones(batch, 2, 3, dtype=torch.bool),
            relation_features=torch.randn(batch, 2, 4),
            relation_present=torch.ones(batch, 2, dtype=torch.bool),
            heatmap_features=torch.randn(batch, 2, 8, 8),
            heatmap_present=torch.ones(batch, 2, dtype=torch.bool),
            graph_logits=torch.linspace(-1.0, 1.0, batch),
        ),
        "task_ids": indices.remainder(3),
        "pair_labels": indices.remainder(2).float(),
        "eval_keys": [
            (f"s{index}", task, 0, 1) for index, task in zip(items, tasks)
        ],
    }


def test_optimizer_step_count_handles_partial_accumulation_group():
    assert optimizer_steps_per_epoch(10, batch_size=4, accumulation=2) == 2
    assert optimizer_steps_per_epoch(8, batch_size=4, accumulation=2) == 1


def test_route_threshold_none_when_routing_disabled():
    from vlm.train_pair import _route_threshold

    cfg = {"routing": {"use": False, "threshold": 0.8}}
    assert _route_threshold(cfg) is None


def test_epoch_sampler_is_deterministic_per_seed_and_epoch():
    dataset = _TinyControlDataset()
    weights = torch.ones(len(dataset), dtype=torch.double)
    kwargs = dict(
        dataset=dataset,
        collate_fn=_collate,
        weights=weights,
        num_samples=12,
        batch_size=3,
        num_workers=0,
        seed=17,
        pin_memory=False,
    )
    a = [batch["graph_logits"].tolist() for batch in make_epoch_loader(**kwargs, epoch=2)]
    b = [batch["graph_logits"].tolist() for batch in make_epoch_loader(**kwargs, epoch=2)]
    c = [batch["graph_logits"].tolist() for batch in make_epoch_loader(**kwargs, epoch=3)]
    assert a == b
    assert a != c


def test_frame_grouped_batch_sampler_preserves_multiset_and_contiguity():
    sampled = [0, 3, 1, 4, 2, 5, 0, 4]
    frame_ids = ["a", "b", "a", "c", "b", "a"]
    batches = list(FrameGroupedBatchSampler(sampled, frame_ids, batch_size=3))
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == sorted(sampled)
    ordered_frames = [frame_ids[index] for index in flattened]
    for frame in set(ordered_frames):
        positions = [i for i, value in enumerate(ordered_frames) if value == frame]
        assert positions == list(range(min(positions), max(positions) + 1))
    assert all(1 <= len(batch) <= 3 for batch in batches)


def test_graph_control_epoch_and_checkpoint_roundtrip(tmp_path):
    dataset = _TinyControlDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=_collate)
    control = GraphLogitMLPControl(hidden_dim=8)
    criterion = PairTaskBCELoss()
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    before = run_epoch(
        control,
        loader,
        device=torch.device("cpu"),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=2,
        description="test",
    )
    assert before.examples == len(dataset)
    assert torch.isfinite(torch.tensor(before.loss))
    expected = {name: value.detach().clone() for name, value in control.state_dict().items()}
    save_control_checkpoint(
        tmp_path, control, criterion, optimizer, scheduler,
        {"epoch": 2, "global_step": 7, "best_val_loss": before.loss},
    )

    restored = GraphLogitMLPControl(hidden_dim=8)
    restored_criterion = PairTaskBCELoss()
    state = restore_control_checkpoint(tmp_path, restored, restored_criterion)
    assert state["epoch"] == 2 and state["global_step"] == 7
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_epoch_batch_logger_tracks_optimizer_updates_not_microbatches():
    dataset = _TinyControlDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=_collate)
    control = GraphLogitMLPControl(hidden_dim=8)
    criterion = PairTaskBCELoss()
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    events = []
    run_epoch(
        control, loader, device=torch.device("cpu"), criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, accumulation=2,
        batch_log_interval=1,
        batch_logger=lambda step, payload: events.append((step, payload)),
        description="batch-log-test",
    )
    # Three microbatches with accumulation=2 produce exactly two optimizer updates.
    assert [step for step, _ in events] == [1, 2]
    assert all(event["examples"] > 0 for _, event in events)
    assert all(torch.isfinite(torch.tensor(event["batch_loss"])) for _, event in events)
    assert all(
        set(event) == {"batch_loss", "running_loss", "examples"}
        for _, event in events
    )


def test_graph_feature_control_runs_through_shared_epoch_loop():
    loader = torch.utils.data.DataLoader(
        list(range(6)), batch_size=2, collate_fn=_feature_collate
    )
    control = GraphFeatureMLPControl(feature_dim=4, hidden_dim=8)
    criterion = PairTaskBCELoss()
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    stats = run_epoch(
        control,
        loader,
        device=torch.device("cpu"),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        accumulation=2,
        description="feature-control-test",
    )
    assert stats.examples == 6
    assert torch.isfinite(torch.tensor(stats.loss))

    collector = PairPredictionCollector()
    eval_stats = run_epoch(
        control,
        loader,
        device=torch.device("cpu"),
        criterion=criterion,
        prediction_collector=collector,
        description="feature-control-eval-test",
    )
    assert eval_stats.examples == 6 and len(collector) == 6
    collector.assert_complete(
        (f"s{index}", ("lah", "laeo", "sa")[index % 3], 0, 1)
        for index in range(6)
    )


def test_checkpoint_selection_prefers_locked_metric_and_falls_back_to_val_loss():
    stats = EpochStats(loss=0.4, residual_loss=0.4, accuracy=0.5, examples=8)
    assert checkpoint_score(
        stats,
        {"social_ap": 0.81},
        monitor="social_ap",
        monitor_mode="max",
    ) == ("social_ap", "max", 0.81)
    # Generative validation has locked metrics but no teacher-forced EpochStats.
    assert checkpoint_score(
        None,
        {"social_ap": 0.82},
        monitor="social_ap",
        monitor_mode="max",
    ) == ("social_ap", "max", 0.82)
    assert checkpoint_score(
        stats, None, monitor="social_ap", monitor_mode="max"
    ) == ("val_loss", "min", 0.4)
    assert score_improved(0.82, 0.81, "max")
    assert score_improved(0.39, 0.40, "min")
    assert not score_improved(0.80, 0.81, "max")


def test_epoch_metrics_file_gets_one_compact_block_per_completed_epoch(tmp_path):
    train = EpochStats(loss=0.5, residual_loss=0.5, accuracy=0.7, examples=60)
    val = EpochStats(loss=0.4, residual_loss=0.4, accuracy=0.8, examples=30)
    metrics = {
        "social_ap": 0.81,
        "social_auc": 0.82,
        "mean_social_f1": 0.73,
        "LAH_AP": 0.9,
        "LAH_AUC": 0.8,
        "F1_LAH": 0.7,
        "LAEO_AP": 0.8,
        "LAEO_AUC": 0.9,
        "F1_LAEO": 0.6,
        "SA_AP": 0.7,
        "SA_AUC": 0.75,
        "F1_SA": 0.65,
    }
    path = tmp_path / "vlmpair.err"
    first = format_epoch_report(
        epoch=0,
        epochs=3,
        train_stats=train,
        val_stats=val,
        val_metrics=metrics,
        selection_name="social_ap",
        selection_score=0.81,
        best_score=0.81,
        improved=True,
    )
    second = format_epoch_report(
        epoch=1,
        epochs=3,
        train_stats=train,
        val_stats=val,
        val_metrics=metrics,
        selection_name="social_ap",
        selection_score=0.80,
        best_score=0.81,
        improved=False,
    )
    append_epoch_report(path, first)
    append_epoch_report(path, second)
    text = path.read_text()
    assert text.count("[epoch 1/3]") == 1
    assert text.count("[epoch 2/3]") == 1
    assert text.count("[validation] social_ap=") == 2
    assert "tqdm" not in text and "wandb:" not in text


def test_task_label_sampler_rejects_any_non_unit_pos_weight():
    validate_sampler_loss_compatibility(
        "task", {"lah": 2.0, "laeo": 0.5, "sa": 1.0}
    )
    validate_sampler_loss_compatibility(
        "task_label", {"lah": 1.0, "laeo": 1.0, "sa": 1.0}
    )
    with pytest.raises(ValueError, match="already balances labels"):
        validate_sampler_loss_compatibility(
            "task_label", {"lah": 2.0, "laeo": 1.0, "sa": 1.0}
        )
    with pytest.raises(ValueError, match="already balances labels"):
        validate_sampler_loss_compatibility(
            "task_label", {"lah": 1.0, "laeo": 0.5, "sa": 1.0}
        )


def test_actual_tiny_qwen_peft_keeps_only_lora_and_new_modules_trainable():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "max_position_embeddings": 256,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": (0, 1),
        },
        image_token_id=40,
        video_token_id=41,
        vision_start_token_id=42,
        vision_end_token_id=43,
    )
    base = transformers.Qwen3VLForConditionalGeneration(config)
    base.requires_grad_(False)
    targets = [
        name
        for name, _ in base.named_modules()
        if "language_model" in name
        and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    ]
    backbone = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    token_ids = {
        token: index + 10 for index, token in enumerate(PAIR_SPECIAL_TOKENS)
    }
    vlm = PairSocialVLM(
        backbone,
        token_ids,
        graph_dim=4,
        graph_hidden_dim=16,
        heatmap_conv_dim=32,
    )
    objective = PairSocialObjective(
        vlm,
        PairTaskResidualDecoder(32, 16, 0.0),
        PairTaskBCELoss(),
    )
    try:
        lora, new = partition_vlm_parameters(objective)
        assert lora and new
        assert all(parameter.requires_grad for parameter in lora + new)
        assert not backbone.get_output_embeddings().weight.requires_grad
        assert vlm.get_output_embeddings() is backbone.get_output_embeddings()
        assert not any(
            parameter.requires_grad
            for parameter in backbone.base_model.model.model.visual.parameters()
        )
    finally:
        objective.close()


def test_partition_vlm_parameters_text_mode_has_empty_new_group_and_no_overlap():
    """Text graph-evidence mode has no soft-token projector: partition_vlm_parameters must
    special-case TextGenerativeVLM to an empty 'new' group (see vlm/train_pair.py branch on
    isinstance(objective.vlm, TextGenerativeVLM)), with only LoRA-tagged params trainable."""

    class _LoraStubBackbone(torch.nn.Module):
        """Minimal frozen-base + LoRA-tagged-param backbone; mirrors how peft names LoRA
        adapter params (substring 'lora_') which partition_vlm_parameters keys off of."""

        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(4, 4)
            self.base.requires_grad_(False)
            self.lora_A = torch.nn.Parameter(torch.randn(4, 2))
            self.lora_B = torch.nn.Parameter(torch.randn(2, 4))

        def forward(self, input_ids=None, labels=None, **kw):
            raise NotImplementedError("partition_vlm_parameters never calls forward")

    vlm = TextGenerativeVLM(_LoraStubBackbone())
    objective = PairGenerativeObjective(vlm)
    lora, new = partition_vlm_parameters(objective)
    assert len(lora) == 2
    assert new == []
    assert not ({id(parameter) for parameter in lora} & {id(parameter) for parameter in new})


def test_graph_evidence_config_selects_text_collate(monkeypatch):
    # Guard: a text-mode config must resolve to the text collate + text dataset arg.
    from vlm.train_pair import select_generative_builders

    text_builders = select_generative_builders(
        {"model": {"output": "generative", "graph_evidence": "text"}}
    )
    assert text_builders.graph_evidence == "text"
    assert text_builders.uses_text_collate is True
    gtok_builders = select_generative_builders(
        {"model": {"output": "generative", "graph_evidence": "gtoken"}}
    )
    assert gtok_builders.graph_evidence == "gtoken"
    assert gtok_builders.uses_text_collate is False
    default_builders = select_generative_builders({"model": {"output": "generative"}})
    assert default_builders.graph_evidence == "gtoken"
    assert default_builders.uses_text_collate is False
    with pytest.raises(ValueError, match="graph_evidence"):
        select_generative_builders({"model": {"graph_evidence": "bogus"}})


def test_select_generative_builders_exposes_text_only_vision_reuse_flag():
    from vlm.train_pair import select_generative_builders

    text = select_generative_builders({
        "model": {"output": "generative", "graph_evidence": "text"},
        "input": {"reuse_frozen_vision": True},
    })
    assert text.reuse_vision is True

    gtoken = select_generative_builders({
        "model": {"output": "generative", "graph_evidence": "gtoken"},
        "input": {"reuse_frozen_vision": True},
    })
    assert gtoken.reuse_vision is False

    disabled = select_generative_builders({
        "model": {"output": "generative", "graph_evidence": "text"},
        "input": {"reuse_frozen_vision": False},
    })
    assert disabled.reuse_vision is False
