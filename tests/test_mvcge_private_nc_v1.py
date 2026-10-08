from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.mvcge_private_nc_v1 import (
    MVCGEPrivateNC,
    _Expert,
    build_graph_operator,
    expected_load,
    pc_coefficients,
    sparse_pc_basis,
    sparse_top2,
)
from src.models.mag_collab_adapt import MAGCollabAdapt
from src.models.predictor import LinkPredictor
from src.tasks.collab_nc_pilot import CollabTaskHeads, train_only_loss


def _graph(n: int = 24) -> torch.Tensor:
    pairs = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 3) % n) for i in range(n)]
    return torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).T.contiguous()


def _inputs(n: int = 24):
    generator = torch.Generator().manual_seed(41)
    return torch.randn(n, 9, generator=generator), torch.randn(n, 5, generator=generator)


def test_dimensions_nc_and_synthetic_lp_interface():
    x_t, x_i = _inputs()
    edge_index = _graph(x_t.size(0))
    model = MVCGEPrivateNC(9, 5, "G1", init_seed=3).eval()
    model.set_graph(edge_index)
    with torch.no_grad():
        z, details = model(x_t, x_i, return_details=True)
    assert z.shape == (24, 256)
    assert details["readout_input_shape"] == (24, 514)
    assert [v.shape for v in details["own_semantics"]] == [(24, 128), (24, 128)]
    nc_head = torch.nn.Linear(256, 6)
    assert nc_head(z).shape == (24, 6)
    lp_decoder = LinkPredictor(256, 32, 2, 0.0)
    score = lp_decoder.score_pairs(z[:8], z[8:16])
    assert score.shape == (8,)
    score.sum().backward()
    assert lp_decoder.net[0].weight.grad is not None


def test_sparse_graph_pc_basis_matches_dense_and_includes_order_three():
    n = 6
    mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    op = build_graph_operator(edge, n, mask)
    dense_p = op.p.to_dense()
    values = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 7
    basis = sparse_pc_basis(op, values, max_order=3)
    dense_t = torch.diag(mask.float()) - dense_p
    expected = [values * mask[:, None]]
    for order in range(1, 4):
        expected.append(-dense_t @ expected[-1] / order)
    assert len(basis) == 4
    for left, right in zip(basis, expected):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
    assert float(basis[3].abs().sum()) > 0


def test_pc_matrix_full_rank_and_initial_expert_responses_match_polynomials():
    coeff = pc_coefficients(max_order=3, bank_size=3)
    assert torch.linalg.matrix_rank(coeff).item() == 4
    responses = torch.tensor([[1.0, -0.1, 0.0, 0.0], [1.0, -0.5, 0.0, 0.0], [1.0, -1.2, 0.9, -0.2], [1.0, 0.1, 0.0, 0.0]])
    model = MVCGEPrivateNC(9, 5, "G0", init_seed=7)
    t = torch.linspace(-0.5, 2.0, 31)
    for layer in model.layers:
        for index, expert in enumerate(layer.experts):
            gamma = expert.gamma().detach()[:, 0]
            actual = sum(gamma[n] * (-t).pow(n) / torch.lgamma(torch.tensor(float(n + 1))).exp() for n in range(4))
            target = sum(responses[index, n] * t.pow(n) for n in range(4))
            torch.testing.assert_close(actual, target, atol=2e-6, rtol=2e-6)


def test_shared_basis_filter_matches_projection_then_filter_forward_and_gradients():
    torch.manual_seed(5)
    n, d = 7, 4
    edge = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6], [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
    mask = torch.ones(n, dtype=torch.bool)
    op = build_graph_operator(edge, n, mask)
    coeff = pc_coefficients(3, 3)
    target = torch.tensor([1.0, -1.2, 0.9, -0.2], dtype=torch.float64)
    gamma_target = target * torch.tensor([1.0, -1.0, 2.0, -6.0], dtype=torch.float64)
    theta = torch.linalg.solve(coeff.T, gamma_target).float()
    expert = _Expert(d, 3, 3, coeff).eval()
    expert.initialize_theta(target)
    values = torch.randn(n, d, requires_grad=True)
    basis = sparse_pc_basis(op, values, 3)
    actual = expert(basis)
    loss = actual.square().sum()
    grad_actual = torch.autograd.grad(loss, (values, expert.w_in.weight, expert.theta), retain_graph=False)

    values_ref = values.detach().clone().requires_grad_(True)
    projected = values_ref @ expert.w_in.weight.T
    direct_basis = sparse_pc_basis(op, projected, 3)
    theta_ref = expert.theta
    filtered = theta_ref[0] * direct_basis[0]
    for bank in range(1, 4):
        filtered = filtered + theta_ref[bank] * sum(coeff[bank, order].float() * direct_basis[order] for order in range(4))
    expected = expert.w_out(expert.dropout(F.gelu(filtered + expert.bias)))
    grad_expected = torch.autograd.grad(expected.square().sum(), (values_ref, expert.w_in.weight, expert.theta))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for left, right in zip(grad_actual, grad_expected):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(theta, expert.theta[:, 0].detach(), atol=2e-6, rtol=2e-6)


def test_graph_loops_duplicates_isolates_mask_and_node_permutation():
    n = 8
    edges = torch.tensor([[0, 1, 0, 1, 2, 3, 3, 4], [1, 0, 1, 0, 3, 2, 4, 3]])
    mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.bool)
    op = build_graph_operator(edges, n, mask)
    dense = op.p.to_dense()
    torch.testing.assert_close(dense[5, 5], torch.tensor(1.0))
    torch.testing.assert_close(dense[0, 0], torch.tensor(0.5))
    assert torch.count_nonzero(dense[7]).item() == 0
    assert torch.count_nonzero(dense[:, 7]).item() == 0
    assert torch.count_nonzero(dense[5]).item() == 1  # one added self loop for an isolate
    assert torch.count_nonzero(dense[0]).item() == 2  # duplicate edge collapsed

    x_t, x_i = _inputs(n)
    masks = torch.stack((mask, mask), dim=1)
    model_a = MVCGEPrivateNC(9, 5, "G1", init_seed=17).eval()
    model_a.set_graph(edges, masks)
    with torch.no_grad():
        z_a, _ = model_a(x_t, x_i, masks)
    permutation = torch.tensor([4, 2, 6, 0, 5, 1, 3, 7])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n)
    perm_edges = inverse[edges]
    model_b = MVCGEPrivateNC(9, 5, "G1", init_seed=17).eval()
    model_b.set_graph(perm_edges, masks[permutation])
    with torch.no_grad():
        z_b, _ = model_b(x_t[permutation], x_i[permutation], masks[permutation])
    torch.testing.assert_close(z_b, z_a[permutation], atol=2e-5, rtol=2e-5)


def test_top2_has_exactly_two_unit_sum_gates_and_expected_load_has_gradients():
    logits = torch.randn(19, 4, requires_grad=True)
    gates, selected = sparse_top2(logits)
    assert (gates > 0).sum(-1).eq(2).all()
    torch.testing.assert_close(gates.sum(-1), torch.ones(19))
    assert selected.shape == (19, 2)
    sigma = torch.full_like(logits, 0.1, requires_grad=True)
    noisy = logits + torch.randn_like(logits) * sigma
    load = expected_load(logits, noisy, sigma, torch.ones(19, dtype=torch.bool))
    loss = (load - load.mean()).square().mean()
    grads = torch.autograd.grad(loss, (logits, sigma))
    assert torch.isfinite(load).all() and torch.isfinite(grads[0]).all() and torch.isfinite(grads[1]).all()
    assert grads[0].abs().sum() > 0 and grads[1].abs().sum() > 0


def test_task_loss_reaches_router_experts_adapter_and_private_ffn():
    torch.manual_seed(19)
    x_t, x_i = _inputs(48)
    edge = _graph(48)
    model = MVCGEPrivateNC(9, 5, "G1", init_seed=22)
    model.set_graph(edge)
    model.train()
    z, details = model(x_t, x_i, step=21)
    head = torch.nn.Linear(256, 5)
    labels = torch.randint(0, 5, (48,))
    loss = F.cross_entropy(head(z[:32]), labels[:32])
    router_grad = torch.autograd.grad(loss, model.routers[0].logit_head.weight, retain_graph=True)[0]
    assert torch.isfinite(router_grad).all() and router_grad.norm() > 0
    loss.backward()
    assert model.adapters[0][0].weight.grad is not None and model.adapters[0][0].weight.grad.norm() > 0
    active_expert_gradients = [
        float(parameter.grad.norm())
        for layer in model.layers
        for expert in layer.experts
        for parameter in expert.parameters()
        if parameter.grad is not None
    ]
    assert active_expert_gradients and max(active_expert_gradients) > 0
    assert model.layers[0].private[0][0].weight.grad is not None
    assert model.layers[0].private[0][0].weight.grad.norm() > 0
    assert details["balance_loss"].grad_fn is not None


def test_a0_reuses_the_historical_b1_encoder_and_s0_task_losses_on_synthetic_nc():
    torch.manual_seed(71)
    n = 20
    x_t, x_i = _inputs(n)
    x = torch.cat((x_t, x_i), dim=-1)
    edges = _graph(n)
    available = torch.ones(n, 2, dtype=torch.bool)
    model = MAGCollabAdapt(
        {"input_dim": 14, "num_nodes": n, "num_classes": 5, "text_dim": 9, "visual_dim": 5},
        variant="B1", dropout=0.2, init_seed=71, typed_value=False, fusion=False,
        out_dim=768, collect_diagnostics=False, optimize_cache=True,
    )
    heads = CollabTaskHeads("B1", 5, output_dim=768)
    model.set_epoch(21)
    z, _, _, _, extras = model(x, edges, available, compute_ssl=False)
    assert z.shape == (n, 768)
    labels = torch.randint(0, 5, (n,))
    train_ids = torch.arange(0, 12)
    loss, terms, _ = train_only_loss("B1", z, extras, heads, labels, train_ids, path_weight=0.05, rec_weight=0.01, pair_weight=0.01)
    assert torch.isfinite(loss)
    assert set(("main_ce", "path_ce", "rec", "pair")).issubset(terms)
    loss.backward()
    assert model.b1_updates[0][0].net[0].weight.grad is not None


def test_g0_g1_public_initialization_private_parameter_count_and_own_semantic_readout():
    g0 = MVCGEPrivateNC(9, 5, "G0", init_seed=123).eval()
    g1 = MVCGEPrivateNC(9, 5, "G1", init_seed=123).eval()
    state0, state1 = g0.state_dict(), g1.state_dict()
    public0 = {k: v for k, v in state0.items() if ".private." not in k}
    public1 = {k: v for k, v in state1.items() if ".private." not in k}
    assert public0.keys() == public1.keys()
    assert all(torch.equal(public0[k], public1[k]) for k in public0)
    assert sum(p.numel() for layer in g1.layers for ff in layer.private for p in ff.parameters()) == 33_408
    x_t, x_i = _inputs()
    g1.set_graph(_graph())
    with torch.no_grad():
        z, details = g1(x_t, x_i, return_details=True)
        own = details["own_semantics"]
        expected = torch.cat((g1.prior_norms[0](own[0]), g1.prior_norms[1](own[1])), dim=-1)
    torch.testing.assert_close(details["readout_input"][:, :256], expected)
    assert z.shape[-1] == 256


def test_context_uses_both_old_modal_states_before_first_layer_updates():
    x_t, x_i = _inputs()
    model = MVCGEPrivateNC(9, 5, "G1", init_seed=29).eval()
    model.set_graph(_graph())
    captured = {}
    def capture(_module, args):
        captured["input"] = args[0].detach().clone()
        return None

    hook = model.context_mlps[0][0].register_forward_pre_hook(capture)
    with torch.no_grad():
        adapters = tuple(model.adapters[m](v) for m, v in enumerate((x_t, x_i)))
        expected_states = tuple(model.layers[0].pre_norm[m](adapters[m]) for m in range(2))
        model(x_t, x_i)
    hook.remove()
    torch.testing.assert_close(captured["input"], torch.cat((*expected_states, torch.ones(24, 2)), dim=-1))


def test_eval_is_deterministic_and_does_not_mutate_model_state():
    x_t, x_i = _inputs()
    model = MVCGEPrivateNC(9, 5, "G1", init_seed=31).eval()
    model.set_graph(_graph())
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        z1, _ = model(x_t, x_i, stochastic_routing=False)
        z2, _ = model(x_t, x_i, stochastic_routing=False)
    assert torch.equal(z1, z2)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_optimizer_groups_cover_each_trainable_parameter_once_and_have_expected_rules():
    model = MVCGEPrivateNC(9, 5, "G1", init_seed=37)
    groups = model.optimizer_groups(0.001, 0.01)
    flattened = [id(p) for group in groups for p in group["params"]]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {id(p) for p in model.parameters() if p.requires_grad}
    assert any(g["name"] == "theta" and g["weight_decay"] == 0.001 for g in groups)
    assert any(g["name"] == "router_context" and g["lr"] == 0.0005 for g in groups)
    for name, parameter in model.named_parameters():
        if parameter.ndim == 1 or name.endswith("bias") or name.startswith("type_embedding"):
            containing = next(g for g in groups if any(id(p) == id(parameter) for p in g["params"]))
            assert containing["weight_decay"] == 0.0


def test_u1_has_no_router_context_and_uses_fixed_uniform_gates():
    x_t, x_i = _inputs()
    model = MVCGEPrivateNC(9, 5, "U1", init_seed=43).eval()
    model.set_graph(_graph())
    with torch.no_grad():
        _, details = model(x_t, x_i, return_details=True)
    assert model.routers is None and model.context_mlps is None
    for layer in details["layers"]:
        torch.testing.assert_close(layer["gates"], torch.full((24, 2, 4), 0.25))
        assert layer["balance_loss"] is None


def test_masked_missing_nodes_and_isolates_have_finite_outputs_and_zero_states():
    x_t, x_i = _inputs(10)
    mask = torch.ones(10, 2, dtype=torch.bool)
    mask[2, 0] = False
    mask[4] = False
    mask[9, 1] = False
    edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 6, 7], [1, 0, 2, 1, 3, 2, 4, 3, 7, 6]])
    model = MVCGEPrivateNC(9, 5, "G0", init_seed=47).eval()
    model.set_graph(edge, mask)
    with torch.no_grad():
        z, details = model(x_t, x_i, mask, return_details=True)
    assert torch.isfinite(z).all()
    assert torch.count_nonzero(details["states"][0][~mask[:, 0]]).item() == 0
    assert torch.count_nonzero(details["states"][1][~mask[:, 1]]).item() == 0
    for operator in model._graph_ops:
        assert torch.count_nonzero(operator.p.to_dense()[~operator.mask]).item() == 0


def test_training_noise_is_deterministic_only_with_explicit_generator_and_router_sigma_starts_at_point_one():
    x_t, x_i = _inputs()
    model = MVCGEPrivateNC(9, 5, "G1", init_seed=53).eval()
    model.set_graph(_graph())
    gen_a = torch.Generator().manual_seed(101)
    gen_b = torch.Generator().manual_seed(101)
    with torch.no_grad():
        _, a = model(x_t, x_i, stochastic_routing=True, noise_generator=gen_a)
        _, b = model(x_t, x_i, stochastic_routing=True, noise_generator=gen_b)
    torch.testing.assert_close(a["layers"][0]["sigma"], torch.full((24, 2, 4), 0.1), atol=1e-7, rtol=1e-7)
    assert torch.equal(a["layers"][0]["noisy_logits"], b["layers"][0]["noisy_logits"])


def test_g0_and_g1_task_heads_are_external_and_no_lp_task_is_part_of_encoder():
    model = MVCGEPrivateNC(9, 5, "G0", init_seed=59)
    assert not hasattr(model, "classifier")
    assert not hasattr(model, "lp_decoder")
    assert all("labels" not in name for name, _ in model.named_parameters())
