from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.adaptive_evidence_assimilation_m21 import Model as M21Model
from src.models.provenance_evidence_m2 import Model as M2Model


def _cfg(provenance: bool = True, mode: str = "adaptive_vector"):
    return OmegaConf.create({"model": {
        "name": "adaptive_evidence_assimilation_m21", "hidden_dim": 128, "relation_dim": 64,
        "max_order": 3, "dropout": 0.0, "relation_dropout": 0.0, "transport_mode": "orthogonal",
        "transport_group_dim": 8, "transport_sweeps": 2, "transport_max_angle": math.pi / 2,
        "response_hidden_dim": 64, "compatibility_scale": 2.0, "anchor_beta": 0.1, "aggregation": "pna",
        "readout": "gpr", "relation_context": False, "edge_chunk_size": 2, "provenance_dim": 32,
        "provenance_dropout": 0.0, "provenance_composition": "distmult", "provenance_adapter": "residual",
        "evidence_dim": 128, "evidence_num_heads": 4, "evidence_dropout": 0.0,
        "provenance_conditioning": provenance, "assimilation_mode": mode, "assimilation_dropout": 0.0,
    }})


def _data_info():
    return {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_nodes": 6}


def _inputs():
    torch.manual_seed(17)
    x = torch.randn(6, 6)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 3, 4, 5], [1, 0, 2, 1, 4, 5, 3, 3]])
    return x, edge_index


def test_retrieval_matches_m2_exactly_with_shared_state():
    torch.manual_seed(123)
    m2 = M2Model(_cfg(), _data_info()).eval()
    torch.manual_seed(123)
    m21 = M21Model(_cfg(), _data_info()).eval()
    m21.load_state_dict(m2.state_dict(), strict=False)
    x, edge = _inputs()
    left = m2.forward_with_intervention(x, edge, return_details=True, return_attention=True)
    right = m21.forward_with_intervention(x, edge, return_details=True, return_attention=True)
    assert torch.equal(left["retrieval"], right["retrieval"])
    assert torch.equal(left["attention"], right["attention"])
    for key in ("keys", "values"):
        assert torch.equal(left["evidence"][key], right["evidence"][key])


def test_zero_initialized_adaptive_field_preserves_h0_and_paired_initialization():
    torch.manual_seed(221)
    semantic = M21Model(_cfg(False), _data_info()).eval()
    torch.manual_seed(221)
    provenance = M21Model(_cfg(True), _data_info()).eval()
    x, edge = _inputs()
    s = semantic.forward_with_intervention(x, edge, return_details=True)
    p = provenance.forward_with_intervention(x, edge, return_details=True)
    for modality in ("text", "visual"):
        assert torch.count_nonzero(s["assimilation_field"][modality]) == 0
        assert torch.equal(s[f"Z_{modality}"], s["stage12"][f"H0_{modality}"])
        assert torch.equal(p[f"Z_{modality}"], p["stage12"][f"H0_{modality}"])
    assert torch.allclose(s["z"], p["z"], atol=2e-6, rtol=2e-6)


def test_residual_norms_are_frozen_legacy_parameters():
    model = M21Model(_cfg(), _data_info())
    assert all(not p.requires_grad for p in model.text_residual_norm.parameters())
    assert all(not p.requires_grad for p in model.visual_residual_norm.parameters())
    assert "Inherited legacy M2 assimilation parameters" in model.analyze(*_inputs())["legacy_readout_frozen_statement"]


def test_assimilation_out_receives_gradient_at_zero_initialization():
    torch.manual_seed(941)
    model = M21Model(_cfg(), _data_info())
    x, edge = _inputs()
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    z = model(x, edge)[0]
    F.cross_entropy(torch.nn.Linear(128, 2)(z), labels).backward()
    for modality in ("text", "visual"):
        grad = getattr(model, f"assimilation_out_{modality}").weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_adaptive_injection_and_counterfactual_shapes():
    model = M21Model(_cfg(), _data_info()).eval()
    x, edge = _inputs()
    full = model.forward_with_intervention(x, edge, return_details=True)
    assert full["assimilation_field"]["text"].shape == (6, 128)
    off = model.forward_with_intervention(x, edge, return_details=True, assimilation_intervention="assimilation_off")
    assert torch.equal(off["Z_text"], off["stage12"]["H0_text"])
    uniform = model.forward_with_intervention(x, edge, return_details=True, assimilation_intervention="uniform_node")
    assert torch.allclose(uniform["assimilation_field"]["visual"], uniform["assimilation_field"]["visual"][:1].expand(6, -1))
    shuffled = model.forward_with_intervention(x, edge, return_details=True, assimilation_intervention="dimension_shuffle")
    assert torch.equal(shuffled["assimilation_field"]["text"], full["assimilation_field"]["text"])


def test_global_scalar_mode_is_zero_initialized_and_trainable():
    model = M21Model(_cfg(mode="global_scalar"), _data_info())
    x, edge = _inputs()
    details = model.forward_with_intervention(x, edge, return_details=True)
    assert torch.equal(details["Z_visual"], details["stage12"]["H0_visual"])
    assert model.assimilation_alpha_text.requires_grad and model.assimilation_alpha_visual.requires_grad


def test_m21_standard_forward_and_inference_contract():
    model = M21Model(_cfg(), _data_info()).eval()
    x, edge = _inputs()
    output = model(x, edge)
    assert output[0].shape == (6, 128)
    assert output[1] is None and output[2] is None and output[3].ndim == 0 and output[4] == {}
    assert torch.allclose(output[0].cpu(), model.inference(x, edge, device=torch.device("cpu")), atol=1e-6, rtol=1e-6)
