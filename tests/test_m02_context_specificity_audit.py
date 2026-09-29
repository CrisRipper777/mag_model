from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from scripts.audit_m02_context_specificity import (
    DEGREE_BUCKET_LABELS,
    _residual_diversity,
    build_reverse_edge_mapping,
    degree_bucket_ids,
    degree_matched_permutation,
    encode_full_node_contexts,
    encode_leave_pair_out_contexts,
    edge_degree_bin_ids,
    leave_pair_out_statistics,
    shuffle_context_pair,
    validation_with_evidence,
)
from src.models.relation_transport_m02 import Model


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "relation_transport_m02",
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
                "relation_context": True,
                "context_residual_scale": 0.5,
                "edge_chunk_size": 2,
            }
        }
    )


def _data_info():
    return {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_nodes": 5, "num_classes": 2}


def _toy_bidirectional_graph():
    # Pair 0-1 is the held-out physical pair; nodes 1 and 0 retain other neighbors.
    source = torch.tensor([0, 1, 0, 2, 3, 1, 4, 1])
    target = torch.tensor([1, 0, 2, 0, 1, 3, 1, 4])
    relation = torch.tensor(
        [
            [1.0, 3.0], [2.0, 4.0], [3.0, 1.0], [4.0, -1.0],
            [5.0, 7.0], [6.0, 9.0], [9.0, 11.0], [8.0, 15.0],
        ]
    )
    return relation, source, target


def test_reverse_edge_mapping_is_involution():
    _relation, source, target = _toy_bidirectional_graph()
    reverse = build_reverse_edge_mapping(source, target, 5)
    ids = torch.arange(source.numel())
    assert torch.equal(reverse[reverse], ids)
    assert torch.equal(source[reverse], target)
    assert torch.equal(target[reverse], source)


def test_leave_pair_out_removes_current_and_reverse_pair():
    relation, source, target = _toy_bidirectional_graph()
    reverse = build_reverse_edge_mapping(source, target, 5)
    target_stats, source_stats, target_active, source_active = leave_pair_out_statistics(
        relation, source, target, reverse, 5
    )
    # The current pair contributes e0 (0->1) and reverse e1 (1->0); neither is retained.
    assert torch.allclose(target_stats[0, :2], torch.tensor([7.0, 9.0]))
    assert torch.allclose(source_stats[0, :2], torch.tensor([4.0, -1.0]))
    assert target_active[0] and source_active[0]


def test_leave_pair_out_mean_matches_manual_toy_graph():
    relation, source, target = _toy_bidirectional_graph()
    reverse = build_reverse_edge_mapping(source, target, 5)
    target_stats, source_stats, *_ = leave_pair_out_statistics(relation, source, target, reverse, 5)
    # Incoming/outgoing means for target node 1 after excluding nodes 0 in both directions.
    assert torch.allclose(target_stats[0, 0:2], torch.tensor([7.0, 9.0]))
    assert torch.allclose(target_stats[0, 4:6], torch.tensor([7.0, 12.0]))
    # Source node 0 retains only its physical pair with node 2.
    assert torch.allclose(source_stats[0, 0:2], torch.tensor([4.0, -1.0]))
    assert torch.allclose(source_stats[0, 4:6], torch.tensor([3.0, 1.0]))


def test_leave_pair_out_std_matches_manual_toy_graph():
    relation, source, target = _toy_bidirectional_graph()
    reverse = build_reverse_edge_mapping(source, target, 5)
    target_stats, source_stats, *_ = leave_pair_out_statistics(relation, source, target, reverse, 5)
    assert torch.allclose(target_stats[0, 2:4], torch.tensor([2.0, 2.0]))
    assert torch.allclose(target_stats[0, 6:8], torch.tensor([1.0, 3.0]))
    assert torch.equal(source_stats[0, 2:4], torch.zeros(2))


def test_degree_one_pair_has_zero_other_context_after_removal():
    relation = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = torch.tensor([0, 1])
    target = torch.tensor([1, 0])
    reverse = build_reverse_edge_mapping(source, target, 2)
    target_stats, source_stats, target_active, source_active = leave_pair_out_statistics(
        relation, source, target, reverse, 2
    )
    assert torch.equal(target_stats, torch.zeros_like(target_stats))
    assert torch.equal(source_stats, torch.zeros_like(source_stats))
    assert not target_active.any() and not source_active.any()


def test_leave_pair_out_keeps_other_neighbors():
    relation, source, target = _toy_bidirectional_graph()
    reverse = build_reverse_edge_mapping(source, target, 5)
    target_stats, source_stats, target_active, source_active = leave_pair_out_statistics(
        relation, source, target, reverse, 5
    )
    assert target_active[0] and source_active[0]
    assert target_stats[0].abs().sum() > 0
    assert source_stats[0].abs().sum() > 0


def test_degree_bucket_assignment():
    degree = torch.tensor([0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 100])
    buckets = degree_bucket_ids(degree)
    assert [DEGREE_BUCKET_LABELS[index] for index in buckets.tolist()] == [
        "0", "1", "2-3", "2-3", "4-7", "4-7", "8-15", "8-15",
        "16-31", "16-31", ">=32", ">=32",
    ]
    source = torch.tensor([1, 2, 4, 10])
    target = torch.tensor([2, 3, 8, 11])
    assert edge_degree_bin_ids(degree, source, target).tolist() == [0, 1, 2, 3]


def test_degree_matched_shuffle_is_permutation():
    degree = torch.tensor([0, 1, 1, 2, 3, 4, 6, 9, 16, 33])
    permutation = degree_matched_permutation(degree, 3407)
    assert torch.equal(torch.sort(permutation).values, torch.arange(degree.numel()))
    assert torch.equal(degree_bucket_ids(degree[permutation]), degree_bucket_ids(degree))
    assert not torch.equal(permutation, torch.arange(degree.numel()))


def test_same_shuffle_used_for_text_and_visual():
    permutation = torch.tensor([1, 0, 3, 2])
    contexts = {"text": torch.arange(12).reshape(4, 3), "visual": torch.arange(20).reshape(4, 5)}
    shuffled = shuffle_context_pair(contexts, permutation)
    assert torch.equal(shuffled["text"], contexts["text"][permutation])
    assert torch.equal(shuffled["visual"], contexts["visual"][permutation])
    assert torch.equal(shuffled["text"][0], contexts["text"][1])
    assert torch.equal(shuffled["visual"][0], contexts["visual"][1])


def test_shuffle_preserves_context_norm_multiset():
    contexts = {"text": torch.randn(20, 64), "visual": torch.randn(20, 64)}
    degree = torch.tensor([0, 1, 1, 2, 2, 3, 4, 4, 7, 8, 8, 9, 16, 17, 32, 33, 33, 5, 2, 1])
    permutation = degree_matched_permutation(degree, 3408)
    shuffled = shuffle_context_pair(contexts, permutation)
    for modality in contexts:
        assert torch.equal(
            torch.sort(contexts[modality].norm(dim=-1)).values,
            torch.sort(shuffled[modality].norm(dim=-1)).values,
        )


def _model_and_inputs():
    torch.manual_seed(904)
    model = Model(_cfg(), _data_info()).eval()
    with torch.no_grad():
        model.text_edge_context.output.weight.normal_(0, 0.04)
        model.visual_edge_context.output.weight.normal_(0, 0.04)
    torch.manual_seed(905)
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 3, 4], [1, 0, 2, 1, 4, 3, 3]])
    classifier = torch.nn.Linear(128, 2).eval()
    return model, classifier, x, edge_index


def _evidence(model, x, edge_index):
    clean, *_ = model._get_graph(edge_index, x.size(0))
    h_text = model.text_projector(x[:, :3])
    h_visual = model.visual_projector(x[:, 3:])
    return model._relation_evidence(h_text, h_visual, clean[0], clean[1]), clean, h_text, h_visual


def test_full_context_path_matches_original_m02():
    model, classifier, x, edge_index = _model_and_inputs()
    evidence, *_ = _evidence(model, x, edge_index)
    val_idx = torch.tensor([0, 2, 4])
    labels = torch.tensor([0, 1, 0])
    normal = validation_with_evidence(model, classifier, x, edge_index, val_idx, labels, 2)
    reconstructed = validation_with_evidence(
        model, classifier, x, edge_index, val_idx, labels, 2, evidence=evidence
    )
    assert normal == reconstructed


def test_context_off_path_matches_existing_helper():
    model, _classifier, x, edge_index = _model_and_inputs()
    with torch.inference_mode():
        helper, *_ = model.forward_without_relation_context(x, edge_index)
        previous = model._relation_context_enabled
        model._relation_context_enabled = False
        try:
            explicit, *_ = model(x, edge_index)
        finally:
            model._relation_context_enabled = previous
    assert torch.equal(helper, explicit)


def test_lpo_counterfactual_is_finite():
    model, classifier, x, edge_index = _model_and_inputs()
    evidence, clean, h_text, h_visual = _evidence(model, x, edge_index)
    relation_text, relation_visual = evidence[2], evidence[3]
    reverse = build_reverse_edge_mapping(clean[0], clean[1], x.size(0))
    lpo_text = encode_leave_pair_out_contexts(model, relation_text, clean[0], clean[1], reverse, x.size(0), "text")[0]
    lpo_visual = encode_leave_pair_out_contexts(model, relation_visual, clean[0], clean[1], reverse, x.size(0), "visual")[0]
    from scripts.audit_m02_context_specificity import relation_conductance
    lpo_evidence = (
        relation_conductance(model, h_text, lpo_text, clean[0], clean[1], "text"),
        relation_conductance(model, h_visual, lpo_visual, clean[0], clean[1], "visual"),
        lpo_text,
        lpo_visual,
    )
    metrics = validation_with_evidence(
        model, classifier, x, edge_index, torch.tensor([0, 2, 4]), torch.tensor([0, 1, 0]), 2,
        evidence=lpo_evidence,
    )
    assert all(math.isfinite(value) for value in metrics.values())


def test_shuffle_counterfactual_is_finite():
    model, classifier, x, edge_index = _model_and_inputs()
    evidence, clean, h_text, h_visual = _evidence(model, x, edge_index)
    source, target = clean
    contexts = {
        "text": encode_full_node_contexts(model, evidence[2], source, target, x.size(0), "text"),
        "visual": encode_full_node_contexts(model, evidence[3], source, target, x.size(0), "visual"),
    }
    degree = torch.bincount(source, minlength=x.size(0))
    permutation = degree_matched_permutation(degree, 3407)
    shuffled = shuffle_context_pair(contexts, permutation)
    from scripts.audit_m02_context_specificity import relation_conductance
    r_text = model.text_edge_context(evidence[2], shuffled["text"][target], shuffled["text"][source], 0.5)[0]
    r_visual = model.visual_edge_context(evidence[3], shuffled["visual"][target], shuffled["visual"][source], 0.5)[0]
    shuffled_evidence = (
        relation_conductance(model, h_text, r_text, source, target, "text"),
        relation_conductance(model, h_visual, r_visual, source, target, "visual"),
        r_text,
        r_visual,
    )
    metrics = validation_with_evidence(
        model, classifier, x, edge_index, torch.tensor([0, 2, 4]), torch.tensor([0, 1, 0]), 2,
        evidence=shuffled_evidence,
    )
    assert all(math.isfinite(value) for value in metrics.values())


def test_residual_diversity_metrics_are_finite():
    torch.manual_seed(901)
    result = _residual_diversity(torch.randn(300, 64), "Toy", 42, "text")
    assert result["sampled_edges"] == 300
    for value in result.values():
        if isinstance(value, float):
            assert math.isfinite(value)
