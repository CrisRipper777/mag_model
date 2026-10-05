from __future__ import annotations

import io
import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.models.mag_prior_context import (
    Model,
    _StructurePair,
    canonical_directed_edges,
    degree_and_mean,
)
from src.models.predictor import LinkPredictor
from src.tasks.prior_context_pilot import PilotTaskHeads, train_only_loss
from scripts.run_prior_context_pilot_v3 import (
    collect_model_diagnostics,
    parameter_groups,
    parameter_snapshots,
    step_diagnostics,
)


def _cfg(variant: str, width: int = 8):
    return OmegaConf.create({
        "model": {
            "variant": variant, "out_dim": 128, "dropout": 0.2,
            "width": width, "edge_chunk_size": 3,
        }
    })


def _model(variant: str = "R4", n: int = 6, width: int = 8):
    info = {"input_dim": 8, "text_dim": 4, "visual_dim": 4, "num_nodes": n, "num_classes": 3}
    return Model(_cfg(variant, width), info)


def _graph(device: torch.device | str = "cpu") -> torch.Tensor:
    # Bidirectional path 0--1--2--3--4--5.
    pairs = [(i, i + 1) for i in range(5)]
    pairs += [(j, i) for i, j in pairs]
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()


def test_edge_direction_degree_and_canonicalization():
    raw = torch.tensor([[0, 0, 1, 1, 2, 1], [1, 1, 0, 2, 1, 1]])
    clean = canonical_directed_edges(raw, 3)
    expected = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert torch.equal(clean, expected)
    state = torch.tensor([[0.0], [2.0], [6.0]])
    mean, degree = degree_and_mean(state, clean)
    assert torch.equal(degree, torch.tensor([1.0, 2.0, 1.0]))
    assert torch.equal(mean, torch.tensor([[2.0], [3.0], [2.0]]))


def test_edge_cache_invalidates_normal_inplace_mutation_and_rebuilds_inference_tensor():
    model = _model("R4")
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    first = model._canonical_edges(edges, 3)
    edges[0, 0] = 2
    second = model._canonical_edges(edges, 3)
    assert second is not first
    assert torch.equal(second, canonical_directed_edges(edges, 3))

    with torch.inference_mode():
        inference_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        first_inference = model._canonical_edges(inference_edges, 3)
        inference_edges[0, 0] = 2
        second_inference = model._canonical_edges(inference_edges, 3)
        assert not torch.equal(first_inference, second_inference)
        assert torch.equal(second_inference, canonical_directed_edges(inference_edges, 3))
        assert model._edge_cache_reference is None


def _set_scalar_gelu_pair(pair: _StructurePair) -> None:
    with torch.no_grad():
        for module in (pair.phi_s.net, pair.phi_d.net):
            module[0].weight.zero_()
            module[0].bias.zero_()
            module[0].weight[0, 0] = 1.0
            module[2].weight.zero_()
            if module[2].bias is not None:
                module[2].bias.zero_()
            module[2].weight[0, 0] = 1.0


def test_structure_nonlinearity_happens_before_neighbor_mean():
    edges = canonical_directed_edges(
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), 3
    )
    state = torch.zeros((3, 128))
    state[:, 0] = torch.tensor([0.0, 2.0, 6.0])
    _, degree = degree_and_mean(state, edges)
    pair = _StructurePair(4, 4)
    _set_scalar_gelu_pair(pair)
    s_value, d_value = pair(state, edges, degree, edge_chunk_size=2)
    gelu = torch.nn.functional.gelu
    # Node 1 has senders 0 and 2: mean(GELU(0), GELU(6)) differs from GELU(mean(0,6)).
    expected_s = (gelu(torch.tensor(0.0)) + gelu(torch.tensor(6.0))) / 2
    assert s_value[1, 0].item() == pytest.approx(expected_s.item(), abs=1e-7)
    assert not torch.isclose(s_value[1, 0], gelu(torch.tensor(3.0)), atol=1e-3)
    # Edge differences into node 1 are -2 and +4, and are transformed separately.
    expected_d = (gelu(torch.tensor(-2.0)) + gelu(torch.tensor(4.0))) / 2
    assert d_value[1, 0].item() == pytest.approx(expected_d.item(), abs=3e-7)
    assert not torch.isclose(d_value[1, 0], gelu(torch.tensor(1.0)), atol=1e-3)
    assert pair.phi_d.net[-1].bias is None


def test_equal_endpoints_make_difference_expert_exactly_zero():
    edges = _graph()
    state = torch.randn(6, 128)
    state[:] = state[0]
    _, degree = degree_and_mean(state, edges)
    pair = _StructurePair(8, 8)
    for training in (True, False):
        pair.train(training)
        _, difference = pair(state, edges, degree, edge_chunk_size=2)
        assert torch.equal(difference, torch.zeros_like(difference))


@pytest.mark.parametrize("training", [True, False])
def test_empty_graph_and_isolates_return_exact_center_prior(training: bool):
    model = _model("R4")
    model.train(training)
    x = torch.randn(4, 8)
    edges = torch.empty((2, 0), dtype=torch.long)
    z, _, _, _, extras = model(x, edges)
    assert torch.equal(z, extras["center"])
    assert torch.isfinite(z).all()


def test_candidate_embedding_contract_and_lp_pair_predictor():
    model = _model("R4")
    z, _, _, _, _ = model(torch.randn(6, 8), _graph())
    assert model.out_dim == 128 and z.shape == (6, 128)
    predictor = LinkPredictor(in_dim=128, hidden_dim=16, num_layers=2, dropout=0.2)
    score = predictor.score_pairs(z[:3], z[3:])
    assert score.shape == (3,) and torch.isfinite(score).all()


def test_r0_ignores_graph_r2_is_static_r3_uses_global_context_r5_shares_experts():
    x = torch.randn(6, 8)
    r0 = _model("R0").eval()
    z_empty = r0(x, torch.empty((2, 0), dtype=torch.long))[0]
    z_graph = r0(x, _graph())[0]
    assert torch.equal(z_empty, z_graph)

    r2 = _model("R2")
    assert len(r2.routers) == 0
    r2.eval(); r2._collect_diagnostics = True
    _, _, _, _, r2_extras = r2(x, _graph())
    assert all(torch.equal(value, torch.full_like(value, 0.5)) for value in r2_extras["route_weights"].values())
    r5 = _model("R5")
    assert r5._pair(0, "joint") is r5._pair(0, "text") is r5._pair(0, "image")
    assert r5.routers[0]["joint"] is not r5.routers[0]["text"]

    for variant in ("R3", "R4"):
        model = _model(variant).eval()
        captured = []
        handle = model.routers[0]["joint"].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        model(x, _graph())
        handle.remove()
        ctx = captured[0][:, 385:]
        if variant == "R3":
            assert torch.equal(ctx, ctx[:1].expand_as(ctx))
        else:
            assert not torch.equal(ctx, ctx[:1].expand_as(ctx))


@pytest.mark.parametrize("variant", [f"R{i}" for i in range(7)])
def test_fixed_sample_diagnostics_support_every_variant(variant: str):
    model = _model(variant).eval()
    diagnostics = collect_model_diagnostics(
        model, torch.randn(6, 8), _graph(), torch.tensor([1, 4])
    )
    assert diagnostics["representation_norms"]["z"]["mean"] >= 0
    assert diagnostics["context_mode"] == ("global_mean" if variant == "R3" else "per_node")
    if variant in {"R2", "R3", "R4", "R5"}:
        assert diagnostics["routes"]
        assert all(value["by_degree"] for value in diagnostics["routes"].values())
    else:
        assert "routes" not in diagnostics


def test_train_only_loss_ignores_nontrain_labels_and_r6_averages_three_heads():
    torch.manual_seed(19)
    z = torch.randn(6, 384)
    extras = {"source_states": {"text": z[:, :128], "image": z[:, 128:256]}}
    heads = PilotTaskHeads("R6", 4)
    train_idx = torch.tensor([0, 2, 4])
    labels_a = torch.tensor([0, 1, 1, 2, 3, 0])
    labels_b = labels_a.clone()
    labels_b[[1, 3, 5]] = torch.tensor([3, 0, 2])
    loss_a, parts_a = train_only_loss(z, extras, heads, labels_a, train_idx)
    loss_b, parts_b = train_only_loss(z, extras, heads, labels_b, train_idx)
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(parts_a["main_ce"], parts_b["main_ce"])
    assert torch.equal(parts_a["source_ce"], parts_b["source_ce"])
    all_logits, source_logits = heads(z, extras)
    assert torch.equal(all_logits, (heads.text(z[:, :128]) + heads.image(z[:, 128:256]) + heads.joint(z[:, 256:])) / 3)
    assert torch.equal(source_logits["text"], heads.text(z[:, :128]))
    loss_a.backward()
    assert all(head.weight.grad is not None for head in (heads.text, heads.image, heads.joint))


def test_r4_all_active_branches_update_and_checkpoint_roundtrip():
    torch.manual_seed(31)
    model = _model("R4")
    heads = PilotTaskHeads("R4", 4)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=0.01)
    x = torch.randn(6, 8)
    edges = _graph()
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    train_idx = torch.tensor([0, 1, 2, 3])
    for _ in range(3):
        model.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
        z, _, _, _, extras = model(x, edges)
        loss, _ = train_only_loss(z, extras, heads, labels, train_idx)
        loss.backward()
        optimizer.step()
    modules = model.diagnostic_modules()
    for name, module in modules.items():
        module_params = list(module.parameters())
        assert module_params, name
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in module_params), name
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module_params), name
    assert "layer1.joint.phiS" in modules and "layer2.image.router" in modules
    assert model.routers[0]["joint"][-1].weight.grad is not None
    assert torch.isfinite(model.routers[0]["joint"][0].weight.grad).all()
    assert model.routers[0]["joint"][0].weight.grad.abs().sum() > 0

    model.eval(); heads.eval()
    with torch.no_grad():
        expected_z = model(x, edges)[0]
        expected = heads(expected_z, model(x, edges)[4])[0].argmax(-1)
    checkpoint = {
        "model": model.state_dict(), "heads": heads.state_dict(),
        "optimizer": optimizer.state_dict(), "epoch": 3,
        "rng_state": {
            "torch_cpu": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(), "torch_cuda_all": [],
        },
        "config": {"variant": "R4", "out_dim": 128},
        "source_identity": {"source_sha256": "synthetic", "split_seed": 42},
    }
    buffer = io.BytesIO(); torch.save(checkpoint, buffer); buffer.seek(0)
    restored = _model("R4"); restored_heads = PilotTaskHeads("R4", 4)
    restored_optimizer = torch.optim.AdamW(
        list(restored.parameters()) + list(restored_heads.parameters()), lr=0.01
    )
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    assert {"model", "heads", "optimizer", "epoch", "rng_state", "config", "source_identity"}.issubset(loaded)
    restored.load_state_dict(loaded["model"]); restored_heads.load_state_dict(loaded["heads"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored.eval(); restored_heads.eval()
    with torch.no_grad():
        restored_z, _, _, _, restored_extras = restored(x, edges)
        actual = restored_heads(restored_z, restored_extras)[0].argmax(-1)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_z, restored_z)


def test_optimizer_step_diagnostics_recover_per_module_preclip_norms():
    torch.manual_seed(71)
    model = _model("R4")
    heads = PilotTaskHeads("R4", 4)
    groups = parameter_groups(model, heads)
    before = parameter_snapshots(groups)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=0.01)
    z, _, _, _, extras = model(torch.randn(6, 8), _graph())
    loss, _ = train_only_loss(
        z, extras, heads, torch.tensor([0, 1, 2, 3, 0, 1]), torch.tensor([0, 1, 2, 3])
    )
    loss.backward()
    direct_preclip = {}
    for name, parameters in groups.items():
        direct_preclip[name] = torch.sqrt(sum(
            (p.grad.detach().float().square().sum() for p in parameters if p.grad is not None),
            torch.zeros(()),
        ))
    total_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in list(model.parameters()) + list(heads.parameters()) if p.requires_grad],
        max_norm=0.01,
    )
    optimizer.step()
    diagnostics = step_diagnostics(groups, before, total_norm, 0.01)
    assert diagnostics["overall_gradient_norm_preclip"] == pytest.approx(float(total_norm), rel=1e-6)
    assert diagnostics["clip_scale"] < 1.0
    for name in groups:
        assert diagnostics["modules"][name]["gradient_norm_preclip"] == pytest.approx(
            float(direct_preclip[name]), rel=1e-5, abs=1e-7
        )
        assert diagnostics["modules"][name]["gradient_norm_postclip"] <= diagnostics["modules"][name]["gradient_norm_preclip"]


def test_r6_dgl_normalized_self_loop_graph_and_checkpoint_roundtrip():
    pytest.importorskip("dgl")
    torch.manual_seed(7)
    model = _model("R6").eval()
    heads = PilotTaskHeads("R6", 4)
    x = torch.randn(6, 8)
    edges = _graph()
    with torch.inference_mode():
        z1 = model(x, edges)[0]
        z2 = model(x, edges)[0]
    assert model.out_dim == 384 and z1.shape == (6, 384)
    assert torch.equal(z1, z2)

    # A topology tensor allocated in inference_mode has no usable _version.
    # The DGL cache must be bypassed, then reflect a later mutation.
    with torch.inference_mode():
        inference_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        model(torch.zeros((6, 8)), inference_edges)
        first_graph = model._get_dgl_graph(model._canonical_edges(inference_edges, 6), 6)
        first_pairs = set(zip(first_graph.edges()[0].tolist(), first_graph.edges()[1].tolist()))
        inference_edges[0, 0] = 2
        second_graph = model._get_dgl_graph(model._canonical_edges(inference_edges, 6), 6)
        second_pairs = set(zip(second_graph.edges()[0].tolist(), second_graph.edges()[1].tolist()))
        assert first_pairs != second_pairs
        assert model._dgl_graph is None and model._dgl_graph_key is None
    with torch.no_grad():
        prediction = heads(z1.clone(), model(x, edges)[4])[0].argmax(-1)
    restored = _model("R6").eval()
    restored.load_state_dict(model.state_dict())
    restored_heads = PilotTaskHeads("R6", 4).eval()
    restored_heads.load_state_dict(heads.state_dict())
    with torch.no_grad():
        z2 = restored(x, edges)[0]
        prediction2 = restored_heads(z2, restored(x, edges)[4])[0].argmax(-1)
    assert torch.equal(z1, z2)
    assert torch.equal(prediction, prediction2)
