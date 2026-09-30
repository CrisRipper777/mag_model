from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from scripts.audit_interaction_provenance_m11 import (
    TOLERANCE,
    assert_semantic_invariance,
    build_source_path_messages,
    build_target_combine_input,
    interaction_atom_components,
    module_state_sha256,
    provenance_step,
)
from src.models.interaction_provenance_m1 import Model


def _model():
    cfg = OmegaConf.create({
        "model": {
            "name": "interaction_provenance_m1",
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
    })
    torch.manual_seed(123)
    return Model(cfg, {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_nodes": 5}).eval()


def _inputs():
    torch.manual_seed(71)
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    return x, edge_index


def _components(model, modality="text"):
    relation = torch.randn(7, 64)
    effect = torch.randn(7, 128)
    return relation, effect, interaction_atom_components(model, relation, effect, modality)


def test_full_atom_reconstruction_matches_model():
    model = _model()
    relation, effect, components = _components(model)
    actual = model._interaction_atom(relation, effect, "text", "full")
    assert torch.allclose(components["full"], actual, atol=TOLERANCE, rtol=TOLERANCE)


def test_corrected_relation_off_equals_tanh_norm_effect_only():
    model = _model()
    _, _, components = _components(model)
    expected = torch.tanh(model.text_atom_norm(components["e"]))
    assert torch.allclose(components["relation_off_corrected"], expected, atol=1e-7, rtol=1e-7)


def test_corrected_effect_off_equals_tanh_norm_relation_only():
    model = _model()
    _, _, components = _components(model)
    expected = torch.tanh(model.text_atom_norm(components["r"]))
    assert torch.allclose(components["effect_off_corrected"], expected, atol=1e-7, rtol=1e-7)


def test_cross_off_equals_tanh_norm_relation_plus_effect():
    model = _model()
    _, _, components = _components(model)
    expected = torch.tanh(model.text_atom_norm(components["r"] + components["e"]))
    assert torch.allclose(components["cross_off"], expected, atol=1e-7, rtol=1e-7)


def test_corrected_component_off_does_not_include_cross_bias():
    model = _model()
    relation, effect = torch.randn(7, 64), torch.randn(7, 128)
    before = interaction_atom_components(model, relation, effect, "text")
    with torch.no_grad():
        model.text_interaction_cross.bias.fill_(1.25)
    after = interaction_atom_components(model, relation, effect, "text")
    for key in ("relation_off_corrected", "effect_off_corrected", "cross_off"):
        assert torch.allclose(before[key], after[key], atol=1e-7, rtol=1e-7)
    buggy_legacy = model._interaction_atom(relation, effect, "text", "relation_off")
    assert not torch.allclose(after["relation_off_corrected"], buggy_legacy)
    assert not torch.allclose(before["full"], after["full"])


def test_local_source_history_off_keeps_target_memory_full():
    previous = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1
    atom = torch.arange(9, dtype=torch.float32).reshape(3, 3) + 1
    source = torch.tensor([0, 2, 3])
    path = build_source_path_messages(previous, atom, source, "off")
    combine_input = build_target_combine_input(previous, torch.zeros_like(previous), "full")
    assert torch.equal(path, atom)
    assert torch.equal(combine_input[:, :3], previous)
    assert not torch.equal(path, previous[source] * atom)


def test_local_target_memory_off_keeps_source_path_full():
    previous = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1
    atom = torch.arange(9, dtype=torch.float32).reshape(3, 3) + 1
    source = torch.tensor([0, 2, 3])
    path = build_source_path_messages(previous, atom, source, "full")
    combine_input = build_target_combine_input(previous, torch.zeros_like(previous), "off")
    assert torch.equal(path, previous[source] * atom)
    assert torch.equal(combine_input[:, :3], torch.ones_like(previous))


def test_corrected_local_shuffle_keeps_target_memory_unshuffled():
    previous = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1
    atom = torch.ones(3, 3)
    source = torch.tensor([0, 2, 3])
    permutation = torch.tensor([2, 0, 3, 1])
    path = build_source_path_messages(previous, atom, source, "shuffle", permutation)
    combine_input = build_target_combine_input(previous, torch.zeros_like(previous), "full")
    assert torch.equal(path, previous[permutation][source])
    assert torch.equal(combine_input[:, :3], previous)
    assert not torch.equal(combine_input[:, :3], previous[permutation])


def test_recursive_source_history_off_retains_target_memory():
    previous = torch.randn(4, 3)
    atom = torch.randn(2, 3)
    source = torch.tensor([0, 2])
    assert torch.equal(build_source_path_messages(previous, atom, source, "off"), atom)
    combined = build_target_combine_input(previous, torch.zeros_like(previous), "full")
    assert torch.equal(combined[:, :3], previous)


def test_recursive_target_memory_off_retains_source_path_composition():
    previous = torch.randn(4, 3)
    atom = torch.randn(2, 3)
    source = torch.tensor([0, 2])
    path = build_source_path_messages(previous, atom, source, "full")
    combined = build_target_combine_input(previous, torch.zeros_like(previous), "off")
    assert torch.equal(path, previous[source] * atom)
    assert torch.equal(combined[:, :3], torch.ones_like(previous))


def test_recursive_source_shuffle_only_shuffles_source_history():
    previous = torch.randn(4, 3)
    atom = torch.randn(2, 3)
    source = torch.tensor([0, 2])
    permutation = torch.tensor([2, 0, 3, 1])
    path = build_source_path_messages(previous, atom, source, "shuffle", permutation)
    combined = build_target_combine_input(previous, torch.zeros_like(previous), "full")
    assert torch.equal(path, previous[permutation][source] * atom)
    assert torch.equal(combined[:, :3], previous)


def test_full_provenance_step_reconstruction_matches_model():
    model = _model()
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    semantic = full["semantic_cache"]
    for modality in ("text", "visual"):
        for order in range(1, 4):
            rebuilt, _, _, _ = provenance_step(model, semantic, full[f"G_{modality}"][order - 1], modality, order)
            assert torch.allclose(rebuilt, full[f"G_{modality}"][order], atol=TOLERANCE, rtol=TOLERANCE)


def test_all_counterfactuals_preserve_semantic_states():
    model = _model()
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    semantic = full["semantic_cache"]
    permutation = torch.tensor([1, 0, 3, 2, 4])
    cases = (
        {"source_history_mode": "off"},
        {"target_memory_mode": "off"},
        {"source_history_mode": "shuffle", "permutation": permutation},
        {"atom_mode": "relation_off_corrected"},
        {"atom_mode": "effect_off_corrected"},
        {"atom_mode": "cross_off"},
    )
    for modality in ("text", "visual"):
        for kwargs in cases:
            current, _, _, _ = provenance_step(
                model, semantic, full[f"G_{modality}"][1], modality, 2, **kwargs
            )
            candidate = {f"S_{name}": semantic[f"S_{name}"] for name in ("text", "visual")}
            assert assert_semantic_invariance(full, candidate) <= TOLERANCE
            assert torch.isfinite(current).all()


def test_counterfactuals_are_finite():
    model = _model()
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    semantic = full["semantic_cache"]
    variants = (
        {"source_history_mode": "off"},
        {"target_memory_mode": "off"},
        {"atom_mode": "relation_off_corrected"},
        {"atom_mode": "effect_off_corrected"},
        {"atom_mode": "cross_off"},
    )
    for modality in ("text", "visual"):
        for kwargs in variants:
            state, aggregate, atoms, _ = provenance_step(
                model, semantic, full[f"G_{modality}"][0], modality, 1,
                sample_indices=torch.tensor([0, 1]), **kwargs,
            )
            assert torch.isfinite(state).all()
            assert torch.isfinite(aggregate).all()
            if atoms is not None:
                assert torch.isfinite(atoms).all()


def test_model_state_unchanged_after_audit():
    model = _model()
    before = module_state_sha256(model)
    x, edge_index = _inputs()
    full = model._compute(x, edge_index)
    provenance_step(
        model, full["semantic_cache"], full["G_text"][1], "text", 2,
        source_history_mode="shuffle", permutation=torch.tensor([1, 0, 2, 3, 4]),
    )
    assert module_state_sha256(model) == before
