from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.interaction_provenance_m1 import Model as M1Model
from src.models.provenance_evidence_m2 import Model as M2Model


def _cfg(conditioning: bool = True):
    return OmegaConf.create(
        {
            "model": {
                "name": "provenance_evidence_m2",
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
                "evidence_dim": 128,
                "evidence_num_heads": 4,
                "evidence_dropout": 0.0,
                "provenance_conditioning": conditioning,
            }
        }
    )


def _data_info():
    return {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_nodes": 6}


def _inputs():
    torch.manual_seed(17)
    x = torch.randn(6, 6)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 3, 4, 5], [1, 0, 2, 1, 4, 5, 3, 3]],
        dtype=torch.long,
    )
    return x, edge_index


def _paired_models(conditioning: bool = True):
    torch.manual_seed(1234)
    m1 = M1Model(_cfg(), _data_info()).eval()
    torch.manual_seed(5678)
    m2 = M2Model(_cfg(conditioning), _data_info()).eval()
    m2.load_state_dict(m1.state_dict(), strict=False)
    return m1, m2


def _assert_stage12_equal(left, right):
    for modality in ("text", "visual"):
        assert torch.allclose(left[f"H0_{modality}"], right[f"H0_{modality}"], atol=2e-6, rtol=2e-6)
        for a, b in zip(left[f"S_{modality}"], right[f"S_{modality}"], strict=True):
            assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
        for a, b in zip(left[f"G_{modality}"], right[f"G_{modality}"], strict=True):
            assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)


def test_m2_stage12_semantics_match_m1():
    m1, m2 = _paired_models()
    x, edge_index = _inputs()
    expected = m1._compute(x, edge_index, provenance_mode="adapter_off")
    actual = m2.forward_with_intervention(x, edge_index, return_details=True)["stage12"]
    for modality in ("text", "visual"):
        assert torch.allclose(expected[f"H0_{modality}"], actual[f"H0_{modality}"], atol=2e-6, rtol=2e-6)
        for a, b in zip(expected[f"S_{modality}"], actual[f"S_{modality}"], strict=True):
            assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)


def test_m2_stage12_provenance_match_m1():
    m1, m2 = _paired_models()
    x, edge_index = _inputs()
    expected = m1._compute(x, edge_index, provenance_mode="adapter_off")
    actual = m2.forward_with_intervention(x, edge_index, return_details=True)["stage12"]
    for modality in ("text", "visual"):
        for a, b in zip(expected[f"G_{modality}"], actual[f"G_{modality}"], strict=True):
            assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)


def test_delta_states_are_consecutive_differences():
    _, model = _paired_models()
    x, edge_index = _inputs()
    result = model.forward_with_intervention(x, edge_index, return_details=True)
    for modality in ("text", "visual"):
        for k, delta in enumerate(result["evidence"]["deltas"][modality], start=1):
            assert torch.equal(delta, result["stage12"][f"S_{modality}"][k] - result["stage12"][f"S_{modality}"][k - 1])


def test_evidence_key_value_shapes():
    _, model = _paired_models()
    x, edge_index = _inputs()
    evidence = model.forward_with_intervention(x, edge_index, return_details=True)["evidence"]
    assert evidence["keys"].shape == (6, 7, 128)
    assert evidence["values"].shape == (6, 7, 128)


def test_six_structural_tokens_plus_null():
    _, model = _paired_models()
    x, edge_index = _inputs()
    evidence = model.forward_with_intervention(x, edge_index, return_details=True)["evidence"]
    assert evidence["keys"][:, :6].shape == (6, 6, 128)
    assert evidence["keys"][:, 6].shape == (6, 128)


def test_null_value_is_exact_zero():
    _, model = _paired_models()
    x, edge_index = _inputs()
    evidence = model.forward_with_intervention(x, edge_index, return_details=True)["evidence"]
    assert torch.equal(evidence["values"][:, 6], torch.zeros_like(evidence["values"][:, 6]))


def test_provenance_specific_heads_zero_initialized():
    model = M2Model(_cfg(True), _data_info())
    for modality in ("text", "visual"):
        for name in ("key_provenance", "key_interaction", "value_interaction"):
            assert torch.count_nonzero(getattr(model, f"{name}_{modality}").weight) == 0


def test_m2_p_equals_m2_s_at_initialization():
    torch.manual_seed(123)
    m2s = M2Model(_cfg(False), _data_info()).eval()
    torch.manual_seed(123)
    m2p = M2Model(_cfg(True), _data_info()).eval()
    x, edge_index = _inputs()
    zs = m2s(x, edge_index)[0]
    zp = m2p(x, edge_index)[0]
    assert torch.allclose(zs, zp, atol=2e-6, rtol=2e-6)


def test_semantic_only_tokens_do_not_depend_on_G():
    _, model = _paired_models(conditioning=False)
    x, edge_index = _inputs()
    details = model.forward_with_intervention(x, edge_index, return_details=True)
    semantic, provenance = model._get_stage12(details["stage12"])
    changed = {m: [g + torch.randn_like(g) * 100 for g in provenance[m]] for m in model.MODALITIES}
    a = model._make_evidence(semantic, provenance, use_provenance=False)
    b = model._make_evidence(semantic, changed, use_provenance=False)
    assert torch.equal(a["keys"], b["keys"])
    assert torch.equal(a["values"], b["values"])


def test_provenance_tokens_change_when_G_changes():
    _, model = _paired_models(conditioning=True)
    for modality in model.MODALITIES:
        with torch.no_grad():
            getattr(model, f"key_provenance_{modality}").weight.normal_(0.0, 0.1)
            getattr(model, f"key_interaction_{modality}").weight.normal_(0.0, 0.1)
            getattr(model, f"value_interaction_{modality}").weight.normal_(0.0, 0.1)
    x, edge_index = _inputs()
    details = model.forward_with_intervention(x, edge_index, return_details=True)
    semantic, provenance = model._get_stage12(details["stage12"])
    changed = {m: [g + torch.randn_like(g) * 2 for g in provenance[m]] for m in model.MODALITIES}
    a = model._make_evidence(semantic, provenance, use_provenance=True)
    b = model._make_evidence(semantic, changed, use_provenance=True)
    assert not torch.allclose(a["keys"][:, :6], b["keys"][:, :6])
    assert not torch.allclose(a["values"][:, :6], b["values"][:, :6])


def test_query_shape():
    _, model = _paired_models()
    x, edge_index = _inputs()
    details = model.forward_with_intervention(x, edge_index, return_details=True)
    assert details["queries"].shape == (6, 2, 128)


def test_attention_weights_sum_to_one():
    _, model = _paired_models()
    queries = torch.randn(5, 2, 128)
    keys, values = torch.randn(5, 7, 128), torch.randn(5, 7, 128)
    _, attention = model._retrieve(queries, keys, values, return_attention=True)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(5, 2, 4), atol=1e-6)


def test_attention_mask_same_modality_only():
    _, model = _paired_models()
    queries = torch.randn(5, 2, 128)
    keys, values = torch.randn(5, 7, 128), torch.randn(5, 7, 128)
    _, attention = model._retrieve(queries, keys, values, same_modality_only=True, return_attention=True)
    assert torch.count_nonzero(attention[:, 0, :, 3:6]) == 0
    assert torch.count_nonzero(attention[:, 1, :, :3]) == 0
    assert torch.allclose(attention.sum(dim=-1), torch.ones(5, 2, 4), atol=1e-6)


def test_null_token_can_receive_attention():
    _, model = _paired_models()
    queries = torch.randn(5, 2, 128)
    keys, values = torch.randn(5, 7, 128), torch.randn(5, 7, 128)
    _, attention = model._retrieve(queries, keys, values, return_attention=True)
    assert bool((attention[..., 6] > 0).all())


def test_m2_does_not_use_legacy_gamma_readout():
    _, model = _paired_models()
    x, edge_index = _inputs()
    before = model(x, edge_index)[0]
    with torch.no_grad():
        model.gamma_logits_text.copy_(torch.tensor([9.0, -2.0, 4.0, 1.0]))
        model.gamma_logits_visual.copy_(torch.tensor([-3.0, 8.0, 0.0, 1.0]))
        model.fusion.weight.normal_(0.0, 100.0)
        model.fusion.bias.normal_(0.0, 100.0)
    after = model(x, edge_index)[0]
    assert torch.equal(before, after)


def test_m2_does_not_use_legacy_provenance_adapter():
    _, model = _paired_models()
    x, edge_index = _inputs()
    before = model(x, edge_index)[0]
    with torch.no_grad():
        model.provenance_adapter.weight.normal_(0.0, 100.0)
    after = model(x, edge_index)[0]
    assert torch.equal(before, after)


def test_legacy_readout_params_are_frozen():
    _, model = _paired_models()
    assert not model.gamma_logits_text.requires_grad
    assert not model.gamma_logits_visual.requires_grad
    assert all(not p.requires_grad for p in model.fusion.parameters())
    assert all(not p.requires_grad for p in model.provenance_adapter.parameters())
    assert model.legacy_frozen_parameter_count > 0


def test_provenance_off_preserves_H0_S_G():
    _assert_intervention_stage12("provenance_off")


def test_provenance_shuffle_preserves_H0_S_G():
    _assert_intervention_stage12("provenance_shuffle")


def test_order_mismatch_preserves_H0_S_G():
    _assert_intervention_stage12("order_mismatch")


def test_same_modality_only_preserves_H0_S_G():
    _assert_intervention_stage12("same_modality_only")


def _assert_intervention_stage12(intervention: str):
    _, model = _paired_models()
    x, edge_index = _inputs()
    full = model.forward_with_intervention(x, edge_index, return_details=True)["stage12"]
    changed = model.forward_with_intervention(x, edge_index, intervention=intervention, return_details=True)["stage12"]
    _assert_stage12_equal(full, changed)


def test_forward_shape_and_interface():
    _, model = _paired_models()
    x, edge_index = _inputs()
    output = model(x, edge_index)
    assert output[0].shape == (6, 128)
    assert output[1:] == (None, None, output[3], {}) or (output[1] is None and output[2] is None and output[3].ndim == 0 and output[4] == {})


def test_inference_matches_eval_forward():
    _, model = _paired_models()
    x, edge_index = _inputs()
    expected = model(x, edge_index)[0].detach().cpu()
    actual = model.inference(x, edge_index, device=torch.device("cpu"))
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_full_m2_provenance_parameters_receive_gradient():
    torch.manual_seed(941)
    model = M2Model(_cfg(True), _data_info())
    x, edge_index = _inputs()
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    head = torch.nn.Linear(128, 2)
    z = model(x, edge_index)[0]
    F.cross_entropy(head(z), labels).backward()
    for modality in model.MODALITIES:
        for name in ("key_provenance", "key_interaction", "value_interaction"):
            grad = getattr(model, f"{name}_{modality}").weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0


def test_semantic_only_stage3_provenance_conditioning_heads_receive_no_gradient():
    torch.manual_seed(942)
    model = M2Model(_cfg(False), _data_info())
    x, edge_index = _inputs()
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    head = torch.nn.Linear(128, 2)
    F.cross_entropy(head(model(x, edge_index)[0]), labels).backward()
    for modality in model.MODALITIES:
        for name in ("key_provenance", "key_interaction", "value_interaction"):
            assert getattr(model, f"{name}_{modality}").weight.grad is None
