from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from src.models.relation_transport_m0 import Model as M01Model
from src.models.relation_transport_m02 import EdgeRelationContext, Model


def _cfg(m02: bool = True):
    model = {
        "name": "relation_transport_m02" if m02 else "relation_transport_m0",
        "hidden_dim": 128,
        "relation_dim": 64,
        "max_order": 3,
        "dropout": 0.0,
        "relation_dropout": 0.0,
        "transport_mode": "orthogonal",
        "transport_group_dim": 8,
        "transport_sweeps": 2,
        "transport_max_angle": math.pi / 2,
        "response_hidden_dim": 64,
        "compatibility_scale": 2.0,
        "anchor_beta": 0.1,
        "aggregation": "pna",
        "readout": "gpr",
        "relation_context": m02,
        "context_residual_scale": 0.5,
        "edge_chunk_size": 2,
    }
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


def _models():
    torch.manual_seed(1234)
    m01 = M01Model(_cfg(False), _data_info())
    torch.manual_seed(1234)
    m02 = Model(_cfg(True), _data_info())
    state = m01.state_dict()
    missing, unexpected = m02.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(name.startswith(("text_context_encoder.", "visual_context_encoder.", "text_edge_context.", "visual_edge_context.")) for name in missing)
    return m01, m02


def _evidence(model, x, edge_index, contextual=True):
    clean, *_ = model._get_graph(edge_index, x.size(0))
    h_text = model.text_projector(x[:, :3])
    h_visual = model.visual_projector(x[:, 3:])
    previous = getattr(model, "_relation_context_enabled", None)
    if previous is not None:
        model._relation_context_enabled = contextual
    try:
        return model._relation_evidence(h_text, h_visual, clean[0], clean[1])
    finally:
        if previous is not None:
            model._relation_context_enabled = previous


def test_relation_context_incoming_mean_std():
    relation = torch.tensor([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0], [9.0, 15.0]])
    source = torch.tensor([0, 2, 1, 1])
    target = torch.tensor([1, 1, 0, 2])
    stats = Model._relation_environment_statistics(relation, source, target, 4)
    assert torch.allclose(stats[1, :2], torch.tensor([2.0, 5.0]))
    assert torch.allclose(stats[1, 2:4], torch.tensor([1.0, 2.0]))


def test_relation_context_outgoing_mean_std():
    relation = torch.tensor([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0], [9.0, 15.0]])
    source = torch.tensor([0, 2, 1, 1])
    target = torch.tensor([1, 1, 0, 2])
    stats = Model._relation_environment_statistics(relation, source, target, 4)
    assert torch.allclose(stats[1, 4:6], torch.tensor([7.0, 13.0]))
    assert torch.allclose(stats[1, 6:8], torch.tensor([2.0, 2.0]))


def test_relation_context_handles_isolated_node():
    relation = torch.tensor([[1.0, 2.0]])
    source = torch.tensor([0])
    target = torch.tensor([1])
    stats = Model._relation_environment_statistics(relation, source, target, 3)
    assert torch.equal(stats[2], torch.zeros(8))
    model = Model(_cfg(), _data_info()).eval()
    wide_relation = torch.zeros(1, 64)
    wide_relation[0, :2] = relation[0]
    contextual, residual, context = model._contextualize_one_modality(
        wide_relation, source, target, 3, "text"
    )
    assert torch.isfinite(context).all()
    assert torch.equal(context[2], torch.zeros(64))
    assert contextual.shape == residual.shape == (1, 64)


def test_reverse_relation_has_distinct_in_out_context():
    relation = torch.tensor([[1.0, 2.0], [7.0, -3.0]])
    source = torch.tensor([0, 1])
    target = torch.tensor([1, 0])
    stats = Model._relation_environment_statistics(relation, source, target, 2)
    assert not torch.equal(stats[:, :4], stats[:, 4:])


def test_context_output_head_zero_initialized():
    edge = EdgeRelationContext(4)
    assert torch.count_nonzero(edge.output.weight) == 0
    assert torch.count_nonzero(edge.output.bias) == 0


def test_contextual_relation_equals_pair_relation_at_initialization():
    m01, m02 = _models()
    x, edge_index = _inputs()
    pair = _evidence(m02, x, edge_index, contextual=False)
    contextual = _evidence(m02, x, edge_index, contextual=True)
    for pair_value, context_value in zip(pair, contextual, strict=True):
        assert torch.equal(pair_value, context_value)


def test_m02_matches_m01_at_context_zero():
    m01, m02 = _models()
    m01.eval()
    m02.eval()
    x, edge_index = _inputs()
    with torch.no_grad():
        expected, *_ = m01(x, edge_index)
        actual, *_ = m02(x, edge_index)
    max_abs_error = (expected - actual).abs().max().item()
    assert max_abs_error <= 2e-6


def test_context_residual_is_vector_valued():
    model = Model(_cfg(), _data_info()).eval()
    with torch.no_grad():
        model.text_edge_context.output.weight.normal_(0, 0.05)
    x, edge_index = _inputs()
    evidence = _evidence(model, x, edge_index)
    pair = _evidence(model, x, edge_index, contextual=False)
    residual = evidence[2] - pair[2]
    assert residual.shape == (model._get_graph(edge_index, x.size(0))[0].size(1), 64)
    assert torch.count_nonzero(residual.std(dim=-1)) > 0


def test_context_residual_receives_gradient():
    model = Model(_cfg(), _data_info())
    with torch.no_grad():
        model.text_edge_context.output.weight.normal_(0, 0.03)
        model.text_relation_correction.weight.normal_(0, 0.05)
        model.text_angle_head.weight.normal_(0, 0.01)
    x, edge_index = _inputs()
    z, *_ = model(x, edge_index)
    (z * torch.randn_like(z)).sum().backward()
    for parameter in (
        model.text_edge_context.output.weight,
        model.text_context_encoder.linear.weight,
        model.text_edge_context.target.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_contextual_relation_changes_conductance_when_active():
    _m01, model = _models()
    with torch.no_grad():
        model.text_edge_context.output.weight.normal_(0, 0.05)
        model.text_relation_correction.weight.normal_(0, 0.05)
    x, edge_index = _inputs()
    pair = _evidence(model, x, edge_index, contextual=False)
    contextual = _evidence(model, x, edge_index, contextual=True)
    assert not torch.allclose(pair[0], contextual[0])


def test_contextual_relation_changes_angles_when_active():
    _m01, model = _models()
    with torch.no_grad():
        model.visual_edge_context.output.weight.normal_(0, 0.05)
        model.visual_angle_head.weight.normal_(0, 0.01)
    x, edge_index = _inputs()
    pair = _evidence(model, x, edge_index, contextual=False)
    contextual = _evidence(model, x, edge_index, contextual=True)
    pair_angles = model._edge_angles(pair[3], "visual")
    context_angles = model._edge_angles(contextual[3], "visual")
    assert not torch.allclose(pair_angles, context_angles)


def test_forward_without_context_matches_pair_path():
    m01, m02 = _models()
    m01.eval()
    m02.eval()
    x, edge_index = _inputs()
    with torch.no_grad():
        expected, *_ = m01(x, edge_index)
        actual, *_ = m02.forward_without_relation_context(x, edge_index)
    assert torch.equal(actual, expected)


def test_m02_forward_shape_and_interface():
    model = Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    z, aux1, aux2, aux_loss, metadata = model(x, edge_index)
    assert model.out_dim == 128
    assert z.shape == (5, 128)
    assert aux1 is None and aux2 is None
    assert aux_loss.item() == 0
    assert metadata == {}


def test_m02_inference_matches_eval_forward():
    model = Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    with torch.no_grad():
        expected, *_ = model(x, edge_index)
    actual = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    assert torch.allclose(actual, expected.cpu(), atol=1e-6, rtol=1e-6)


def test_m02_analyze_adds_scalar_context_summaries():
    model = Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    result = model.analyze(x, edge_index)
    assert set(result["relation_context_norm"]) == {"text", "visual"}
    assert set(result["contextual_relation_residual"]) == {"text", "visual"}
    assert set(result["context_conductance_abs_change"]) == {"text", "visual"}
    assert set(result["context_angle_mean_abs_change"]) == {"text", "visual"}
    for key in (
        "relation_context_norm",
        "contextual_relation_residual",
        "context_conductance_abs_change",
        "context_angle_mean_abs_change",
    ):
        for values in result[key].values():
            assert set(values) == {"mean", "std", "p10", "p50", "p90"}
            assert all(math.isfinite(value) for value in values.values())
