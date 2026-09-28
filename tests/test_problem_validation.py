from __future__ import annotations

import pytest
import torch

from src.analysis.problem_validation.common import (
    audit_split,
    make_physical_graph,
    make_probe_split,
    mean_neighbor_messages,
    sample_analysis_population,
)
from src.analysis.problem_validation.edge_utility import (
    MessageProbe,
    build_message_features,
    check_decomposition_exactness,
    remove_message_utility,
)
from src.data import MAGData


def _toy_data() -> MAGData:
    labels = torch.arange(4).repeat_interleave(20)
    train = torch.arange(0, 48)
    val = torch.arange(48, 64)
    test = torch.arange(64, 80)
    x_t = torch.randn(80, 7)
    x_i = torch.randn(80, 5)
    return MAGData(
        name="toy",
        source="test",
        task="nc",
        x=torch.cat((x_t, x_i), dim=1),
        x_t=x_t,
        x_i=x_i,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=labels,
        train_idx=train,
        val_idx=val,
        test_idx=test,
        num_nodes=80,
        num_classes=4,
    )


def test_probe_split_no_overlap(tmp_path) -> None:
    data = _toy_data()
    split = make_probe_split(data, "Toy", tmp_path)
    ptrain, pcal = split["probe_train_idx"], split["probe_calib_idx"]
    assert set(ptrain.tolist()).isdisjoint(pcal.tolist())
    assert set(ptrain.tolist()) | set(pcal.tolist()) == set(data.train_idx.tolist())
    audit = audit_split(data, ptrain, pcal, data.val_idx)
    assert audit["probe_train_original_val_overlap"] == 0
    assert audit["probe_calib_original_val_overlap"] == 0


def test_analysis_never_uses_test_nodes() -> None:
    data = _toy_data()
    with pytest.raises(AssertionError):
        audit_split(data, torch.tensor([0]), torch.tensor([1]), torch.tensor([64]))


def test_message_mean_matches_manual_small_graph() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    edges = torch.tensor([[0, 1, 2, 0], [1, 0, 1, 2]])
    mean, isolated = mean_neighbor_messages(x, edges)
    expected = torch.tensor([[3.0, 4.0], [3.0, 4.0], [1.0, 2.0], [0.0, 0.0]])
    assert torch.equal(mean, expected)
    assert isolated.tolist() == [False, False, False, True]


def test_edge_contribution_decomposition() -> None:
    torch.manual_seed(7)
    num_nodes = 120
    ring = torch.arange(num_nodes)
    edges = torch.stack((torch.cat((ring, (ring + 1) % num_nodes)), torch.cat(((ring + 1) % num_nodes, ring))))
    graph = make_physical_graph(edges, num_nodes)
    h_t, h_v = torch.randn(num_nodes, 128), torch.randn(num_nodes, 128)
    features, _, _, isolated = build_message_features(h_t, h_v, graph)
    assert not isolated.any()
    model = MessageProbe(5)
    target = ring
    neighbor = (ring + 1) % num_nodes
    degree = torch.full((num_nodes,), 2)
    error = check_decomposition_exactness(model, features, h_t, h_v, target, neighbor, degree)
    assert error < 1e-5


def test_remove_message_utility_sign() -> None:
    label = torch.tensor([0])
    helpful_full = torch.tensor([[2.0, 0.0]])
    helpful_removed = torch.tensor([[0.0, 2.0]])
    harmful_full = helpful_removed
    harmful_removed = helpful_full
    assert remove_message_utility(helpful_full, helpful_removed, label).item() > 0
    assert remove_message_utility(harmful_full, harmful_removed, label).item() < 0


def test_deterministic_analysis_sampling() -> None:
    data = _toy_data()
    edges = torch.tensor([[i for i in range(79)], [i + 1 for i in range(79)]])
    graph = make_physical_graph(edges, data.num_nodes)
    one = sample_analysis_population(data, graph, seed=42)
    two = sample_analysis_population(data, graph, seed=42)
    assert all(torch.equal(one[k], two[k]) for k in one)


def test_no_self_loop_in_analysis_edges() -> None:
    edges = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 2, 2]])
    physical = make_physical_graph(edges, 3)
    assert not torch.any(physical[0] == physical[1])
    data = _toy_data()
    data.val_idx = torch.tensor([0, 1, 2, 3])
    population = sample_analysis_population(data, physical, seed=42)
    assert torch.all(population["target_node"] != population["neighbor_node"])
