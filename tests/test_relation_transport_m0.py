from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from src.models.relation_transport_m0 import (
    Model,
    ModalityRelations,
    _segment_statistics,
    preprocess_physical_graph,
)


def _cfg(mode: str = "orthogonal", **overrides):
    model = {
        "name": "relation_transport_m0",
        "hidden_dim": 128,
        "relation_dim": 64,
        "max_order": 3,
        "dropout": 0.0,
        "relation_dropout": 0.0,
        "transport_mode": mode,
        "transport_group_dim": 8,
        "transport_sweeps": 2,
        "transport_max_angle": math.pi / 2,
        "response_hidden_dim": 64,
        "compatibility_scale": 2.0,
        "anchor_beta": 0.1,
        "aggregation": "pna",
        "readout": "gpr",
        "relation_context": False,
        "edge_chunk_size": 2,
    }
    model.update(overrides)
    return OmegaConf.create({"model": model})


def _data_info():
    return {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_nodes": 5}


def _inputs():
    torch.manual_seed(17)
    x = torch.randn(5, 6)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 3, 4], [1, 0, 2, 1, 4, 3, 3]], dtype=torch.long
    )
    return x, edge_index


def _model(mode: str = "orthogonal"):
    torch.manual_seed(1234)
    return Model(_cfg(mode), _data_info())


def test_graph_preprocess_removes_self_loops():
    edge_index = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
    clean, *_ = preprocess_physical_graph(edge_index, 3)
    assert not torch.any(clean[0] == clean[1])


def test_graph_preprocess_is_bidirectional_and_coalesced():
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 0, 2]])
    clean, *_ = preprocess_physical_graph(edge_index, 3)
    pairs = list(zip(clean[0].tolist(), clean[1].tolist(), strict=True))
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == {(0, 1), (1, 0)}


def test_reverse_edge_changes_translation_primitive():
    relation = ModalityRelations(hidden_dim=4, relation_dim=4)
    target = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    source = torch.tensor([[-2.0, 0.5, 3.0, 1.5]])
    forward = relation(target, source)
    reverse = relation(source, target)
    assert not torch.allclose(forward, reverse)


def test_relation_shapes():
    model = _model()
    x, edge_index = _inputs()
    clean, *_ = preprocess_physical_graph(edge_index, x.size(0))
    h0_text = model.text_projector(x[:, :3])
    h0_visual = model.visual_projector(x[:, 3:])
    evidence = model._relation_evidence(h0_text, h0_visual, clean[0], clean[1])
    assert evidence[0].shape == evidence[1].shape == (clean.size(1),)
    assert evidence[2].shape == evidence[3].shape == (clean.size(1), 64)
    assert model._edge_angles(evidence[2], "text").shape == (clean.size(1), 8)


def test_conductance_is_in_zero_two():
    model = _model().eval()
    x, edge_index = _inputs()
    result = model._compute(x, edge_index)
    for modality in ("text", "visual"):
        conductance = result[f"conductance_{modality}"]
        assert torch.all(conductance > 0)
        assert torch.all(conductance < 2)


def test_zero_relation_correction_uses_similarity_prior():
    model = _model().eval()
    x, edge_index = _inputs()
    result = model._compute(x, edge_index)
    source, target = result["source"], result["target"]
    for modality in ("text", "visual"):
        h0 = result[f"H0_{modality}"]
        raw_tau = getattr(model, f"raw_tau_{modality}")
        expected = 2.0 * torch.sigmoid(
            F_softplus(raw_tau)
            * torch.nn.functional.cosine_similarity(h0[target], h0[source], dim=-1)
        )
        assert torch.allclose(result[f"conductance_{modality}"], expected, atol=1e-7)
        layer = getattr(model, f"{modality}_relation_correction")
        assert torch.count_nonzero(layer.weight) == 0
        assert torch.count_nonzero(layer.bias) == 0
        assert abs(float(F_softplus(raw_tau)) - 1.0) < 1e-6


def F_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(value)


def test_zero_angles_are_identity():
    source = torch.randn(11, 128)
    angles = torch.zeros(11, 8)
    assert torch.equal(Model._givens_transport(source, angles), source)


def test_givens_preserves_l2_norm():
    torch.manual_seed(92)
    source = torch.randn(23, 128)
    angles = (torch.rand(23, 8) - 0.5) * math.pi
    transported = Model._givens_transport(source, angles)
    assert torch.allclose(source.norm(dim=-1), transported.norm(dim=-1), atol=2e-6, rtol=2e-6)


def test_zero_angle_orthogonal_equals_identity_transport():
    identity = _model("identity").eval()
    orthogonal = _model("orthogonal").eval()
    identity_state = identity.state_dict()
    common_state = {
        name: value for name, value in orthogonal.state_dict().items() if name in identity_state
    }
    identity.load_state_dict(common_state, strict=True)
    x, edge_index = _inputs()
    actual, *_ = orthogonal(x, edge_index)
    expected, *_ = identity(x, edge_index)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_response_is_identity_at_initialization():
    response = _model().text_response
    value = torch.randn(7, 128)
    assert torch.equal(response(value), value)


def test_pna_matches_manual_toy_graph():
    messages = torch.tensor([[1.0, -1.0], [3.0, 1.0], [5.0, 3.0]])
    target = torch.tensor([1, 1, 2])
    degree = torch.tensor([0, 2, 1, 0])
    log_degree = torch.log1p(degree.float())
    mean_log = log_degree[degree > 0].mean()
    scalers = torch.stack(
        (
            torch.ones_like(log_degree),
            torch.where(degree > 0, log_degree / mean_log, torch.ones_like(log_degree)),
            torch.where(degree > 0, mean_log / log_degree.clamp_min(1e-8), torch.ones_like(log_degree)),
        ),
        dim=-1,
    )
    actual = _segment_statistics(
        [(messages, target)], degree, scalers, 2, messages.dtype, messages.device
    )
    expected_stats = torch.zeros(4, 4, 2)
    epsilon = torch.finfo(messages.dtype).eps
    expected_stats[1, 0] = torch.tensor([2.0, 0.0])
    expected_stats[1, 1] = torch.tensor([3.0, 1.0])
    expected_stats[1, 2] = torch.tensor([1.0, -1.0])
    expected_stats[1, 3] = torch.sqrt(torch.tensor([1.0, 1.0]).clamp_min(epsilon))
    expected_stats[2, 0] = torch.tensor([5.0, 3.0])
    expected_stats[2, 1] = torch.tensor([5.0, 3.0])
    expected_stats[2, 2] = torch.tensor([5.0, 3.0])
    expected_stats[2, 3] = torch.full((2,), math.sqrt(epsilon))
    expected = (expected_stats[:, :, None, :] * scalers[:, None, :, None]).reshape(4, 24)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_pna_is_finite_for_isolated_node():
    messages = torch.tensor([[2.0, -3.0]])
    target = torch.tensor([0])
    degree = torch.tensor([1, 0, 0])
    scalers = torch.ones((3, 3))
    actual = _segment_statistics(
        [(messages, target)], degree, scalers, 2, messages.dtype, messages.device
    )
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[1:], torch.zeros_like(actual[1:]))


def test_state_bank_has_orders_zero_to_three():
    model = _model().eval()
    x, edge_index = _inputs()
    result = model._compute(x, edge_index)
    for modality in ("text", "visual"):
        states = result[f"S_{modality}"]
        assert len(states) == 4
        assert torch.equal(states[0], result[f"H0_{modality}"])


def test_gpr_weights_sum_to_one():
    model = _model()
    assert torch.allclose(torch.softmax(model.gamma_logits_text, 0).sum(), torch.tensor(1.0))
    assert torch.allclose(torch.softmax(model.gamma_logits_visual, 0).sum(), torch.tensor(1.0))


def test_forward_shape_and_interface():
    model = _model().eval()
    x, edge_index = _inputs()
    z, aux1, aux2, aux_loss, metadata = model(x, edge_index)
    assert model.out_dim == 128
    assert z.shape == (x.size(0), 128)
    assert aux1 is None and aux2 is None
    assert aux_loss.item() == 0
    assert metadata == {}


def test_inference_matches_eval_forward():
    model = _model().eval()
    x, edge_index = _inputs()
    with torch.no_grad():
        expected, *_ = model(x, edge_index)
    actual = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    assert torch.allclose(actual, expected.cpu(), atol=1e-6, rtol=1e-6)


def test_relation_parameters_receive_gradient():
    model = _model()
    with torch.no_grad():
        model.text_relation_correction.weight.normal_(0.0, 0.05)
        model.visual_relation_correction.weight.normal_(0.0, 0.05)
    x, edge_index = _inputs()
    z, *_ = model(x, edge_index)
    (z * torch.randn_like(z)).sum().backward()
    for name in (
        "text_relations.diff.weight",
        "visual_relations.bilinear_source.weight",
        "shared_relation.0.weight",
    ):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0


def test_transport_angle_parameters_receive_gradient():
    model = _model("orthogonal")
    with torch.no_grad():
        model.text_angle_head.weight.normal_(0.0, 0.01)
    x, edge_index = _inputs()
    z, *_ = model(x, edge_index)
    (z * torch.randn_like(z)).sum().backward()
    grad = model.text_angle_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
