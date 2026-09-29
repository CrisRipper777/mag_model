from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from scripts.run_problem_validation_p02 import _shuffle_feature_gate, load_p01_reference
from src.analysis.problem_validation.common import _tensor_hash, make_physical_graph
from src.analysis.problem_validation.p02_context_probe import build_context_readout_features
from src.analysis.problem_validation.p02_diagnostics import (
    aggregate_with_edge_states,
    context_rollout,
    exact_decomposition_audit,
    make_all_node_features,
    precompute_edge_states,
)
from src.analysis.problem_validation.p02_models import P02Model, relation_descriptor, same_relation_encoder_definition
from src.analysis.problem_validation.p02_training import IncomingCSR, load_fixed_split, safe_labels, _edge_batch_features
from src.data import MAGData


def _model(variant: str, classes: int = 3, dim: int = 128) -> P02Model:
    return P02Model(
        variant,
        num_classes=classes,
        semantic_dim=dim,
        relation_dim=64,
        relation_dropout=0.1,
        scalar_gate_scale=2.0,
        feature_modulation_scale=0.5,
        similarity_init_alpha=1.0,
        similarity_init_beta=0.0,
    )


def _toy_graph(num_nodes: int = 120) -> IncomingCSR:
    node = torch.arange(num_nodes)
    edges = torch.stack(
        (torch.cat((node, (node + 1) % num_nodes)), torch.cat(((node + 1) % num_nodes, node)))
    )
    physical = make_physical_graph(edges, num_nodes)
    return IncomingCSR(physical, num_nodes, torch.device("cpu"))


def _gate_config() -> dict:
    return {"analysis": {"scalar_gate_scale": 2.0, "feature_modulation_scale": 0.5, "relation_edge_chunk_size": 64}}


def test_relation_descriptor_direction() -> None:
    target = torch.tensor([[1.0, 2.0]])
    source = torch.tensor([[3.0, 5.0]])
    descriptor = relation_descriptor(target, source)
    assert torch.equal(descriptor, torch.tensor([[1.0, 2.0, 3.0, 5.0, 2.0, 3.0, 3.0, 10.0]]))
    assert torch.equal(descriptor[:, :2], target)
    assert torch.equal(descriptor[:, 2:4], source)


def test_uniform_matches_manual_mean() -> None:
    graph = _toy_graph(8)
    h_t, h_v = torch.randn(8, 128), torch.randn(8, 128)
    model = _model("uniform").eval()
    targets = torch.arange(8)
    features = _edge_batch_features(model, graph, h_t, h_v, targets)
    expected_t = torch.zeros_like(h_t)
    expected_v = torch.zeros_like(h_v)
    for target in range(8):
        start, stop = graph.row_ptr[target : target + 2].tolist()
        sources = graph.src[start:stop]
        if len(sources):
            expected_t[target] = h_t[sources].mean(0)
            expected_v[target] = h_v[sources].mean(0)
    assert torch.allclose(features[:, 256:384], expected_t, atol=1e-6)
    assert torch.allclose(features[:, 384:512], expected_v, atol=1e-6)


def test_similarity_scalar_shape_and_range() -> None:
    model = _model("similarity_scalar").eval()
    target, source = torch.randn(9, 128), torch.randn(9, 128)
    a, g = model.relation_function(0, target, source)
    assert a.shape == (9, 1) and g.shape == source.shape
    assert torch.all((a >= 0) & (a <= 2))


def test_learned_scalar_gate_range() -> None:
    model = _model("learned_scalar").eval()
    a, g = model.relation_function(1, torch.randn(11, 128), torch.randn(11, 128))
    assert a.shape == (11, 1) and g.shape == (11, 128)
    assert torch.all((a > 0) & (a < 2))
    assert torch.equal(g, torch.ones_like(g))


def test_conditional_feature_gate_range() -> None:
    model = _model("conditional_feature").eval()
    a, g = model.relation_function(0, torch.randn(17, 128), torch.randn(17, 128))
    assert a.shape == (17, 1) and g.shape == (17, 128)
    assert torch.all((a > 0) & (a < 2))
    assert torch.all((g >= 0.5) & (g <= 1.5))


def test_v2_v3_share_same_relation_encoder_definition() -> None:
    assert same_relation_encoder_definition(_model("learned_scalar"), _model("conditional_feature"))
    left = _model("learned_scalar")
    right = _model("conditional_feature")
    assert [tuple(x.shape) for x in left.relation_encoders[0].parameters()] == [
        tuple(x.shape) for x in right.relation_encoders[0].parameters()
    ]
    assert sum(x.numel() for x in right.feature_heads.parameters()) > 0


@pytest.mark.parametrize("variant", ["uniform", "similarity_scalar", "learned_scalar", "conditional_feature"])
def test_exact_edge_contribution_variants(variant: str) -> None:
    torch.manual_seed(41)
    graph = _toy_graph(120)
    h_t, h_v = torch.randn(120, 128), torch.randn(120, 128)
    model = _model(variant).eval()
    states = precompute_edge_states(model, graph, h_t, h_v, _gate_config())
    features = make_all_node_features(model, graph, h_t, h_v, states)
    target, neighbor = graph.dst[:200].cpu(), graph.src[:200].cpu()
    state_rows = {**states, "physical_edge_row": torch.arange(200)}
    result = exact_decomposition_audit(
        model,
        features,
        h_t,
        h_v,
        target,
        neighbor,
        graph.degree[target],
        state_rows,
        seed=42,
        sample_count=128,
    )
    assert result["sample_count"] >= 100
    assert result["dtype"] == "cpu_float32"
    assert result["max_abs_error"] < 1e-5


def test_identity_function_equals_g_one() -> None:
    model = _model("conditional_feature").eval()
    target, source = torch.randn(13, 128), torch.randn(13, 128)
    a, _ = model.relation_function(0, target, source)
    a_identity, g_identity = model.relation_function(0, target, source, identity_function=True)
    message, _, _ = model.edge_messages(0, target, source, identity_function=True)
    assert torch.equal(a, a_identity)
    assert torch.equal(g_identity, torch.ones_like(source))
    assert torch.allclose(message, a * source)


def test_function_shuffle_preserves_multiset_within_quintile() -> None:
    n = 100
    states = {}
    for name in ("text", "visual"):
        states[name] = {
            "a": torch.rand(n, 1),
            "g": torch.randn(n, 16),
            "probe_similarity": torch.linspace(-1, 1, n),
            "_embedding": torch.randn(n + 1, 16),
        }
    shuffled = _shuffle_feature_gate(states, seed=100)
    for name in ("text", "visual"):
        sim = states[name]["probe_similarity"].numpy()
        order = np.argsort(sim, kind="mergesort")
        for ids in np.array_split(order, 5):
            old_rows = Counter(map(tuple, states[name]["g"][torch.as_tensor(ids)].numpy()))
            new_rows = Counter(map(tuple, shuffled[name]["g"][torch.as_tensor(ids)].numpy()))
            assert old_rows == new_rows
    assert all(torch.equal(states[name]["a"], shuffled[name]["a"]) for name in states)


def test_context_rollout_manual_toy_graph() -> None:
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    physical = make_physical_graph(edges, 4)
    graph = IncomingCSR(physical, 4, torch.device("cpu"))
    h_t = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1
    h_v = h_t + 10
    states = {}
    for name, h in (("text", h_t), ("visual", h_v)):
        states[name] = {"a": torch.ones(graph.src.numel(), 1), "g": torch.ones(graph.src.numel(), 2), "_embedding": h}
    contexts, _ = context_rollout(graph, h_t, h_v, states, max_order=3, chunk_size=4)
    previous = h_t
    for order in range(1, 4):
        expected, _ = aggregate_with_edge_states(
            graph,
            previous,
            previous + 10,
            {"text": states["text"], "visual": states["visual"]},
            chunk_size=3,
        )
        assert torch.allclose(contexts["text"][order], expected, atol=1e-6)
        previous = expected


def test_context_relation_state_is_fixed_across_orders() -> None:
    graph = _toy_graph(9)
    h_t, h_v = torch.randn(9, 128), torch.randn(9, 128)
    model = _model("conditional_feature").eval()
    states = precompute_edge_states(model, graph, h_t, h_v, _gate_config())
    saved_a = {name: states[name]["a"].clone() for name in states}
    saved_g = {name: states[name]["g"].clone() for name in states}
    contexts, _ = context_rollout(graph, h_t, h_v, states, max_order=3, chunk_size=7)
    assert all(torch.equal(saved_a[name], states[name]["a"]) for name in states)
    assert all(torch.equal(saved_g[name], states[name]["g"]) for name in states)
    assert len(contexts["text"]) == len(contexts["visual"]) == 4


def test_context_readout_normalizes_each_block() -> None:
    h_t, h_v = torch.randn(5, 128), torch.randn(5, 128)
    contexts = {name: [h.clone() for _ in range(4)] for name, h in (("text", h_t), ("visual", h_v))}
    context_only = build_context_readout_features(h_t, h_v, contexts, "context_only")
    full_bank = build_context_readout_features(h_t, h_v, contexts, "full_bank")
    assert context_only.shape == (5, 6 * 128)
    assert full_bank.shape == (5, 8 * 128)
    assert torch.allclose(torch.linalg.vector_norm(context_only[:, :128], dim=-1), torch.ones(5), atol=1e-6)


def _toy_data() -> MAGData:
    return MAGData(
        name="toy", source="test", task="nc", x=torch.randn(10, 5), x_t=torch.randn(10, 3),
        x_i=torch.randn(10, 2), edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.arange(10) % 2, train_idx=torch.tensor([0, 1, 2, 3]),
        val_idx=torch.tensor([4, 5, 6]), test_idx=torch.tensor([7, 8, 9]), num_nodes=10, num_classes=2,
    )


def test_no_test_access() -> None:
    data = _toy_data()
    with pytest.raises(PermissionError):
        safe_labels(data, data.test_idx, torch.cat((data.train_idx, data.val_idx)))


def test_fixed_probe_split_reused(tmp_path) -> None:
    data = _toy_data()
    original_train = data.train_idx
    split_path = tmp_path / "fixed.pt"
    cache = {
        "source_split_hashes": {
            "original_train_sha256": _tensor_hash(data.train_idx),
            "original_val_sha256": _tensor_hash(data.val_idx),
        },
        "probe_train_idx": torch.tensor([0, 1]),
        "probe_calib_idx": torch.tensor([2, 3]),
        "analysis_target_nodes": data.val_idx.clone(),
        "sampled_edge_target_node": torch.tensor([4]),
        "sampled_edge_neighbor_node": torch.tensor([1]),
    }
    torch.save(cache, split_path)
    loaded = load_fixed_split(data, "toy", split_path)
    assert torch.equal(loaded["probe_train_idx"], cache["probe_train_idx"])
    assert torch.equal(loaded["probe_calib_idx"], cache["probe_calib_idx"])
    assert torch.equal(data.train_idx, original_train)
    with pytest.raises(FileNotFoundError):
        load_fixed_split(data, "toy", tmp_path / "missing.pt")


def test_p01_reference_edge_population_matches(tmp_path) -> None:
    data = _toy_data()
    physical = make_physical_graph(data.edge_index, data.num_nodes)
    graph = IncomingCSR(physical, data.num_nodes, torch.device("cpu"))
    split = {
        "sampled_edge_target_node": torch.tensor([1]),
        "sampled_edge_neighbor_node": torch.tensor([0]),
        "analysis_target_nodes": torch.tensor([1]),
    }
    reference = {
        "run_seed": 42,
        "target_node": torch.tensor([1]),
        "neighbor_node": torch.tensor([0]),
        "target_degree": torch.tensor([int(graph.degree[1])]),
        "analysis_target_nodes": torch.tensor([1]),
        "probe_sim_text": torch.tensor([0.2]),
        "probe_sim_visual": torch.tensor([0.3]),
        "utility_ce_text": torch.tensor([0.1]),
        "utility_ce_visual": torch.tensor([-0.1]),
        "utility_ce_joint": torch.tensor([0.0]),
    }
    path = tmp_path / "edge_analysis.pt"
    torch.save(reference, path)
    assert torch.equal(load_p01_reference(path, split, graph, 42)["target_node"], split["sampled_edge_target_node"])
    reference["neighbor_node"] = torch.tensor([2])
    torch.save(reference, path)
    with pytest.raises(RuntimeError, match="population mismatch"):
        load_p01_reference(path, split, graph, 42)
