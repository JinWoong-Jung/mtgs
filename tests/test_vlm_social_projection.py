import torch

from vlm.social.evidence import GraphBatch
from vlm.social.projection import (
    HeatmapSlotProjector,
    EvidenceProjector,
    PersonSlotProjector,
)


def _graph_batch() -> GraphBatch:
    generator = torch.Generator().manual_seed(7)
    return GraphBatch(
        tasks=("lah", "sa"),
        person_features=torch.randn(2, 2, 3, 4, generator=generator),
        person_channel_present=torch.tensor([
            [[True, False, False], [False, True, False]],
            [[True, False, True], [True, False, True]],
        ]),
        relation_features=torch.randn(2, 2, 4, generator=generator),
        relation_present=torch.tensor([[True, False], [True, True]]),
        heatmap_features=torch.randn(2, 2, 16, 16, generator=generator),
        heatmap_present=torch.tensor([[True, False], [True, True]]),
        graph_logits=torch.tensor([0.2, -0.4]),
    )


def _replace(batch: GraphBatch, **changes) -> GraphBatch:
    values = {
        "tasks": batch.tasks,
        "person_features": batch.person_features,
        "person_channel_present": batch.person_channel_present,
        "relation_features": batch.relation_features,
        "relation_present": batch.relation_present,
        "heatmap_features": batch.heatmap_features,
        "heatmap_present": batch.heatmap_present,
        "graph_logits": batch.graph_logits,
    }
    values.update(changes)
    return GraphBatch(**values)


def test_lah_absent_person_channels_resolve_to_channel_specific_learned_na():
    batch = _graph_batch()
    projector = PersonSlotProjector(graph_dim=4, output_dim=8, hidden_dim=16)
    resolved = projector.resolve_channels(
        batch.person_features, batch.person_channel_present
    )

    assert torch.equal(resolved[0, 0, 0], batch.person_features[0, 0, 0])
    assert torch.equal(resolved[0, 1, 1], batch.person_features[0, 1, 1])
    for person, channel in ((0, 1), (0, 2), (1, 0), (1, 2)):
        assert torch.equal(resolved[0, person, channel], projector.na_channels[channel])


def test_person_projection_ignores_absent_values_and_trains_all_na_channels():
    batch = _graph_batch()
    projector = PersonSlotProjector(graph_dim=4, output_dim=8, hidden_dim=16)
    corrupted = batch.person_features.clone()
    corrupted[~batch.person_channel_present] = 1e6

    expected = projector(batch.person_features, batch.person_channel_present)
    actual = projector(corrupted, batch.person_channel_present)
    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    assert projector.na_channels.grad is not None
    assert torch.all(projector.na_channels.grad.abs().sum(dim=-1) > 0)


def test_graph_fp32_is_cast_at_bfloat16_projector_boundary():
    batch = _graph_batch()
    person = PersonSlotProjector(
        graph_dim=4, output_dim=8, hidden_dim=16
    ).to(dtype=torch.bfloat16)
    output = person(batch.person_features, batch.person_channel_present)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_heatmap_cnn_only_receives_present_slots():
    batch = _graph_batch()
    projector = HeatmapSlotProjector(output_dim=8, conv_dim=32)
    seen_batch_sizes = []

    def capture_batch(module, args):
        del module
        seen_batch_sizes.append(args[0].shape[0])

    handle = projector.net.register_forward_pre_hook(capture_batch)
    try:
        output = projector(batch.heatmap_features, batch.heatmap_present)
    finally:
        handle.remove()
    assert output.shape == (2, 2, 8)
    assert seen_batch_sizes == [int(batch.heatmap_present.sum())] == [3]


def test_all_absent_heatmaps_skip_cnn_and_train_na_token():
    projector = HeatmapSlotProjector(output_dim=8, conv_dim=32)
    calls = []
    handle = projector.net.register_forward_hook(lambda *args: calls.append(True))
    try:
        output = projector(
            torch.randn(2, 2, 16, 16),
            torch.zeros(2, 2, dtype=torch.bool),
        )
    finally:
        handle.remove()
    assert calls == []
    output.square().mean().backward()
    assert projector.na_token.grad is not None
    assert projector.na_token.grad.abs().sum() > 0


def test_six_slot_projector_masks_all_absent_raw_values_and_is_trainable():
    batch = _graph_batch()
    projector = EvidenceProjector(
        graph_dim=4, output_dim=16, graph_hidden_dim=32, heatmap_conv_dim=32
    )
    expected = projector(batch)
    assert expected.shape == (2, 6, 16)
    assert torch.isfinite(expected).all()

    people = batch.person_features.clone()
    relations = batch.relation_features.clone()
    heatmaps = batch.heatmap_features.clone()
    people[~batch.person_channel_present] = -1e6
    relations[~batch.relation_present] = 1e6
    heatmaps[~batch.heatmap_present] = -1e6
    corrupted = _replace(
        batch,
        person_features=people,
        relation_features=relations,
        heatmap_features=heatmaps,
    )
    torch.testing.assert_close(projector(corrupted), expected)

    expected.square().mean().backward()
    for parameter in (
        projector.person.na_channels,
        projector.relation.na_feature,
        projector.heatmap.na_token,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
