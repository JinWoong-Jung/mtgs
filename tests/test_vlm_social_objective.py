from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm.social.data import SOCIAL_TASK_ID
from vlm.social.evidence import GraphBatch
from vlm.social.objective import (
    GraphFeatureMLPControl,
    GraphLogitMLPControl,
    SocialObjective,
    TaskBCELoss,
    TaskResidualDecoder,
    YesNoResidualHead,
)
from vlm.social.model import SocialVLMOutput


def _graph_batch(graph_logits=None) -> GraphBatch:
    if graph_logits is None:
        graph_logits = torch.tensor([0.7, -0.3, 1.2])
    return GraphBatch(
        tasks=("lah", "laeo", "sa"),
        person_features=torch.zeros(3, 2, 3, 4),
        person_channel_present=torch.ones(3, 2, 3, dtype=torch.bool),
        relation_features=torch.zeros(3, 2, 4),
        relation_present=torch.ones(3, 2, dtype=torch.bool),
        heatmap_features=torch.zeros(3, 2, 8, 8),
        heatmap_present=torch.ones(3, 2, dtype=torch.bool),
        graph_logits=graph_logits,
    )


def _task_ids():
    return torch.tensor([
        SOCIAL_TASK_ID["lah"],
        SOCIAL_TASK_ID["laeo"],
        SOCIAL_TASK_ID["sa"],
    ])


def test_zero_init_residual_is_exactly_graph_equivalent():
    decoder = TaskResidualDecoder(input_dim=8, hidden_dim=16, dropout=0.0)
    graph = _graph_batch()
    output = decoder(torch.randn(3, 8), _task_ids(), graph.graph_logits)
    assert torch.count_nonzero(output.delta_logits) == 0
    assert torch.count_nonzero(output.all_delta_logits) == 0
    torch.testing.assert_close(output.logits, graph.graph_logits)
    assert not output.graph_logits.requires_grad


def test_task_id_routes_to_the_matching_decoder():
    decoder = TaskResidualDecoder(input_dim=4, hidden_dim=8, dropout=0.0)
    for task, bias in (("lah", 1.0), ("laeo", 2.0), ("sa", 3.0)):
        final = decoder.decoders[task][-1]
        with torch.no_grad():
            final.bias.fill_(bias)
    output = decoder(torch.zeros(3, 4), _task_ids(), torch.zeros(3))
    torch.testing.assert_close(output.delta_logits, torch.tensor([1.0, 2.0, 3.0]))


def test_graph_mlp_control_is_zero_init_and_vision_free():
    control = GraphLogitMLPControl(hidden_dim=8)
    graph_logits = torch.tensor([0.3, -0.7, 1.1], requires_grad=True)
    output = control(graph_logits, _task_ids())
    torch.testing.assert_close(output.logits, graph_logits.detach())
    assert graph_logits.grad is None


def test_graph_feature_mlp_is_zero_init_detached_and_heatmap_optional():
    graph = _graph_batch()
    graph.person_features.requires_grad_()
    graph.relation_features.requires_grad_()
    graph.heatmap_features.requires_grad_()
    graph.graph_logits.requires_grad_()

    control = GraphFeatureMLPControl(
        feature_dim=4, hidden_dim=8, dropout=0.0
    )
    assert control.input_dim == 8 * 4 + 8
    output = control(graph, _task_ids())
    torch.testing.assert_close(output.logits, graph.graph_logits.detach())

    # Make the correction depend on the flattened evidence, then verify cached graph
    # tensors remain an immutable input boundary.
    for decoder in control.decoder.decoders.values():
        with torch.no_grad():
            decoder[-1].weight.fill_(1.0)
    control(graph, _task_ids()).logits.sum().backward()
    assert graph.person_features.grad is None
    assert graph.relation_features.grad is None
    assert graph.heatmap_features.grad is None
    assert graph.graph_logits.grad is None

    with_heatmaps = GraphFeatureMLPControl(
        feature_dim=4,
        hidden_dim=8,
        dropout=0.0,
        include_heatmaps=True,
        heatmap_pool_size=4,
    )
    assert with_heatmaps.input_dim == 8 * 4 + 8 + 2 * 4 * 4 + 2
    torch.testing.assert_close(
        with_heatmaps(graph, _task_ids()).logits,
        graph.graph_logits.detach(),
    )


def test_graph_base_is_stop_gradient_but_vlm_path_learns_after_zero_init_step():
    decoder = TaskResidualDecoder(input_dim=4, hidden_dim=8, dropout=0.0)
    hidden = torch.randn(3, 4, requires_grad=True)
    graph_logits = torch.randn(3, requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    criterion = TaskBCELoss()
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)

    first = decoder(hidden, _task_ids(), graph_logits)
    criterion(first.logits, labels, _task_ids()).loss.backward()
    assert graph_logits.grad is None
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    for task in ("lah", "laeo", "sa"):
        assert decoder.decoders[task][-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    hidden.grad = None

    second = decoder(hidden, _task_ids(), graph_logits)
    criterion(second.logits, labels, _task_ids()).loss.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_task_weighted_bce_matches_manual_binary_loss():
    criterion = TaskBCELoss({"lah": 2.0, "laeo": 3.0, "sa": 4.0})
    logits = torch.tensor([0.2, -0.4, 0.8])
    labels = torch.tensor([1.0, 0.0, 1.0])
    output = criterion(logits, labels, _task_ids())
    base = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    expected = base * torch.tensor([2.0, 1.0, 4.0])
    torch.testing.assert_close(output.per_sample, expected)
    torch.testing.assert_close(output.loss, expected.mean())


@pytest.mark.parametrize(
    "labels,task_ids,match",
    [
        (torch.tensor([0.0, 0.5, 1.0]), _task_ids(), "binary"),
        (torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0, 1, 3]), r"in \[0,3\)"),
    ],
)
def test_bce_rejects_invalid_labels_or_tasks(labels, task_ids, match):
    with pytest.raises(ValueError, match=match):
        TaskBCELoss()(torch.zeros(3), labels, task_ids)


class _FakeVLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Parameter(hidden)
        self.output_embeddings = nn.Linear(hidden.shape[-1], 32, bias=False)
        self.closed = False

    def forward(self, model_inputs, graph):
        del model_inputs
        return SocialVLMOutput(
            h_social=self.hidden,
            evidence_tokens=torch.empty(len(graph.tasks), 6, self.hidden.shape[-1]),
            placeholder_positions=torch.empty(len(graph.tasks), 7, dtype=torch.long),
            backbone_output=SimpleNamespace(hidden_states=None),
        )

    def close(self):
        self.closed = True

    def get_output_embeddings(self):
        return self.output_embeddings


def test_objective_composes_vlm_yesno_head_and_bce_and_checks_task_order():
    graph = _graph_batch()
    vlm = _FakeVLM(torch.randn(3, 8))
    objective = SocialObjective(
        vlm,
        YesNoResidualHead(yes_token_id=7, no_token_id=9),
        TaskBCELoss(),
    )
    labels = torch.tensor([1.0, 0.0, 1.0])
    output = objective({}, graph, _task_ids(), labels)
    # zero-init scale/bias -> graph-equivalent prediction.
    torch.testing.assert_close(output.decoder.logits, graph.graph_logits)
    assert output.loss is not None and output.loss.ndim == 0
    assert output.per_sample_loss is not None

    with pytest.raises(ValueError, match="do not match"):
        objective({}, graph, _task_ids().roll(1), labels)
    objective.close()
    assert vlm.closed


def test_yesno_head_reads_lm_head_and_opens_vlm_gradient():
    graph = _graph_batch()
    vlm = _FakeVLM(torch.randn(3, 8))
    head = YesNoResidualHead(yes_token_id=7, no_token_id=9)
    objective = SocialObjective(vlm, head, TaskBCELoss())
    labels = torch.tensor([1.0, 0.0, 1.0])

    # At zero-init the yes/no correction is exactly zero (graph-equivalent) and the
    # scale/bias affine still receives gradient from the frozen LM head's yes/no logits.
    output = objective({}, graph, _task_ids(), labels)
    torch.testing.assert_close(output.decoder.delta_logits, torch.zeros(3))
    output.loss.backward()
    assert head.scale.grad is not None and head.scale.grad.abs().sum() > 0
    # LM head weights stay frozen (read-only semantic direction).
    assert vlm.output_embeddings.weight.grad is None

    # Once scale is non-zero the VLM path (h_social) is supervised too.
    vlm.hidden.grad = None
    head.zero_grad()
    with torch.no_grad():
        head.scale.fill_(0.5)
    objective({}, graph, _task_ids(), labels).loss.backward()
    assert vlm.hidden.grad is not None and vlm.hidden.grad.abs().sum() > 0


def test_yesno_head_standalone_ignores_graph_and_outputs_pure_vlm():
    graph = _graph_batch(graph_logits=torch.tensor([3.0, -2.0, 5.0]))
    vlm = _FakeVLM(torch.randn(3, 8))
    head = YesNoResidualHead(
        yes_token_id=7, no_token_id=9, use_graph_residual=False, scale_init=1.0
    )
    objective = SocialObjective(vlm, head, TaskBCELoss())
    out = objective({}, graph, _task_ids(), torch.tensor([1.0, 0.0, 1.0]))
    # standalone: final == delta (yes/no only), graph_logit NOT added ...
    torch.testing.assert_close(out.decoder.logits, out.decoder.delta_logits)
    # ... but the graph base is still exposed for logging/analysis, just unused.
    torch.testing.assert_close(out.decoder.graph_logits, graph.graph_logits)
    # scale_init=1 => the prediction starts as the raw LM yes/no log-odds (VLM drives it).
    out.loss.backward()
    assert vlm.hidden.grad is not None and vlm.hidden.grad.abs().sum() > 0
