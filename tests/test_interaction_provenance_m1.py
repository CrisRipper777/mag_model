from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.interaction_provenance_m1 import Model as M1Model
from src.models.relation_transport_m0 import Model as M01Model, preprocess_physical_graph


def _cfg(name: str = "interaction_provenance_m1"):
    model = {
        "name": name,
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
        "relation_context": False,
        "edge_chunk_size": 2,
        "provenance_dim": 32,
        "provenance_dropout": 0.0,
        "provenance_composition": "distmult",
        "provenance_adapter": "residual",
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
    m01 = M01Model(_cfg("relation_transport_m0"), _data_info())
    torch.manual_seed(1234)
    m1 = M1Model(_cfg(), _data_info())
    m1.load_state_dict(m01.state_dict(), strict=False)
    return m01, m1


def test_provenance_g0_is_fixed_ones():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    result = model._compute(x, edge_index)
    for modality in ("text", "visual"):
        assert torch.equal(
            result[f"G_{modality}"][0],
            torch.ones_like(result[f"G_{modality}"][0]),
        )


def test_interaction_atom_shape():
    model = M1Model(_cfg(), _data_info()).eval()
    atom = model._interaction_atom(torch.randn(7, 64), torch.randn(7, 128), "text")
    assert atom.shape == (7, 32)


def test_interaction_atom_uses_relation_component():
    model = M1Model(_cfg(), _data_info()).eval()
    relation, effect = torch.randn(9, 64), torch.randn(9, 128)
    full = model._interaction_atom(relation, effect, "text")
    relation_off = model._interaction_atom(relation, effect, "text", "relation_off")
    assert not torch.allclose(full, relation_off)
    r = model.text_relation_provenance_norm(model.text_relation_to_provenance(relation))
    e = model.text_effect_provenance_norm(model.text_effect_to_provenance(effect))
    expected = torch.tanh(model.text_atom_norm(r + e + model.text_interaction_cross(r * e)))
    assert torch.allclose(full, expected)


def test_interaction_atom_uses_realized_effect_component():
    model = M1Model(_cfg(), _data_info()).eval()
    relation, effect = torch.randn(9, 64), torch.randn(9, 128)
    full = model._interaction_atom(relation, effect, "visual")
    effect_off = model._interaction_atom(relation, effect, "visual", "effect_off")
    assert not torch.allclose(full, effect_off)


def test_distmult_history_composition_matches_manual():
    history = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    atom = torch.randn(4, 3)
    source = torch.tensor([4, 1, 1, 0])
    expected = history[source] * atom
    actual = M1Model._compose_path_history(history, atom, source)
    assert torch.equal(actual, expected)


def test_provenance_pna_shape():
    model = M1Model(_cfg(), _data_info())
    x = torch.randn(4, 32)
    target = torch.tensor([1, 1, 2, 4])
    degree = torch.bincount(target, minlength=5)
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
    output = model.text_provenance_aggregator(
        [(x, target)], degree, scalers, x.dtype, x.device
    )
    assert output.shape == (5, 32)


def test_provenance_combine_shape():
    model = M1Model(_cfg(), _data_info())
    combined = model.text_provenance_combine(torch.randn(5, 64))
    assert combined.shape == (5, 32)
    assert model.text_provenance_state_norm(combined).shape == (5, 32)


def test_provenance_states_have_orders_zero_to_three():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    result = model._compute(x, edge_index)
    for modality in ("text", "visual"):
        assert len(result[f"G_{modality}"]) == 4
        assert all(state.shape == (5, 32) for state in result[f"G_{modality}"])


def test_provenance_adapter_zero_initialized():
    model = M1Model(_cfg(), _data_info())
    assert torch.count_nonzero(model.provenance_adapter.weight) == 0


def test_m1_matches_m01_at_initialization():
    m01, m1 = _models()
    m01.eval()
    m1.eval()
    x, edge_index = _inputs()
    z0, *_ = m01(x, edge_index)
    z1, *_ = m1(x, edge_index)
    assert torch.allclose(z1, z0, atol=2e-6, rtol=2e-6)


def test_m1_semantic_states_match_m01_at_initialization():
    m01, m1 = _models()
    m01.eval()
    m1.eval()
    x, edge_index = _inputs()
    result0 = m01._compute(x, edge_index)
    result1 = m1._compute(x, edge_index)
    for modality in ("text", "visual"):
        for state0, state1 in zip(
            result0[f"S_{modality}"], result1[f"S_{modality}"], strict=True
        ):
            assert torch.allclose(state0, state1, atol=2e-6, rtol=2e-6)


def test_provenance_parameters_receive_gradient_after_training_step():
    model = M1Model(_cfg(), _data_info())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, edge_index = _inputs()
    labels = torch.tensor([0, 1, 0, 1, 0])
    head = torch.nn.Linear(128, 2)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        z, *_ = model(x, edge_index)
        loss = F.cross_entropy(head(z), labels)
        loss.backward()
        optimizer.step()
    for name in (
        "text_relation_to_provenance.weight",
        "visual_effect_to_provenance.weight",
        "text_provenance_aggregator.projection.weight",
    ):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0


def _assert_semantics_equal(full, other):
    for modality in ("text", "visual"):
        for left, right in zip(full[f"S_{modality}"], other[f"S_{modality}"], strict=True):
            assert torch.allclose(left, right, atol=2e-6, rtol=2e-6)


def test_history_off_keeps_semantic_states_unchanged():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    history_off = model._compute(x, edge_index, provenance_mode="history_off")
    _assert_semantics_equal(full, history_off)


def test_relation_off_keeps_semantic_states_unchanged():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    relation_off = model._compute(x, edge_index, provenance_mode="relation_off")
    _assert_semantics_equal(full, relation_off)


def test_effect_off_keeps_semantic_states_unchanged():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    effect_off = model._compute(x, edge_index, provenance_mode="effect_off")
    _assert_semantics_equal(full, effect_off)


def test_history_shuffle_is_degree_matched():
    model = M1Model(_cfg(), _data_info())
    x, edge_index = _inputs()
    _, degree, *_ = preprocess_physical_graph(edge_index, x.size(0))
    permutation = model._history_shuffle_permutation(degree, 3407)
    for node, source in enumerate(permutation.tolist()):
        assert model._degree_bucket(int(degree[node])) == model._degree_bucket(int(degree[source]))


def test_history_shuffle_keeps_semantic_states_unchanged():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    shuffled = model._compute(
        x, edge_index, provenance_mode="history_shuffle", history_shuffle_seed=3407
    )
    _assert_semantics_equal(full, shuffled)


def test_inference_matches_eval_forward():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    with torch.no_grad():
        expected, *_ = model(x, edge_index)
    actual = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    assert torch.allclose(actual, expected.cpu(), atol=1e-6, rtol=1e-6)


def test_analyze_metrics_are_finite():
    model = M1Model(_cfg(), _data_info()).eval()
    x, edge_index = _inputs()
    diagnostics = model.analyze(x, edge_index)

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, (float, int)):
            assert math.isfinite(float(value))

    visit(diagnostics)
