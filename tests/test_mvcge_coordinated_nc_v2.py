from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts.run_mvcge_coordinated_nc_v2 import (
    _restore_rng,
    _save_resume,
    build_modules,
    gradient_diagnostics,
    initialize_head,
    model_diagnostics,
    optimizer_for,
)
from src.models.factory import build_model
from src.models.mag_collab_adapt import MAGCollabAdapt
from src.models.mvcge_coordinated_nc_v2 import (
    MVCGECoordinatedNCV2,
    apply_r,
    build_row_mean_operator,
    expected_load,
    routing_eta,
    sparse_top2,
)
from src.models.predictor import LinkPredictor
from src.tasks.collab_nc_pilot import CollabTaskHeads, train_only_loss
from src.tasks.mvcge_coordinated_nc_v2 import (
    CoordinatedNCHeads,
    coordinated_loss,
    indexed_main_logits,
    labels_train_val_only,
)


def _edges(n: int = 9) -> torch.Tensor:
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def _data(n: int = 9, text_dim: int = 7, image_dim: int = 5):
    gen = torch.Generator().manual_seed(101)
    x = torch.randn(n, text_dim + image_dim, generator=gen)
    return x, _edges(n)


def test_row_mean_operator_direction_duplicates_loops_empty_and_isolates():
    edges = torch.tensor([[0, 1, 0, 0, 2, 2, 3], [1, 0, 1, 0, 2, 4, 2]])
    op, degree = build_row_mean_operator(edges, 6)
    dense = op.to_dense()
    expected = torch.zeros(6, 6)
    expected[0, 1] = 1.0
    expected[1, 0] = 1.0
    expected[2, 3] = 1.0
    expected[4, 2] = 1.0
    torch.testing.assert_close(dense, expected)
    torch.testing.assert_close(degree, torch.tensor([1., 1., 1., 0., 1., 0.]))
    assert op.shape == (6, 6)
    values = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    torch.testing.assert_close(apply_r(op, values), expected @ values)
    empty, empty_degree = build_row_mean_operator(torch.empty((2, 0), dtype=torch.long), 4)
    assert empty.shape == (4, 4) and empty._nnz() == 0
    assert torch.equal(empty_degree, torch.zeros(4))


def test_row_mean_sparse_matches_hand_computed_dense_nonregular_graph():
    edges = torch.tensor([[0, 0, 1, 2, 2, 3, 3], [1, 2, 0, 0, 3, 2, 4]])
    op, _ = build_row_mean_operator(edges, 6)
    dense = torch.zeros(6, 6)
    dense[1, 0] = 1
    dense[2, 0] = 0.5; dense[2, 3] = 0.5
    dense[0, 1] = 0.5; dense[0, 2] = 0.5
    dense[3, 2] = 1
    dense[4, 3] = 1
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(apply_r(op, x), dense @ x)


def test_pc_phi_identity_initial_equivalence_theta_gradients_and_signed_coefficients():
    n = 9
    x, edge = _data(n)
    m0 = MVCGECoordinatedNCV2(7, 5, "M0", init_seed=8).eval()
    p0 = MVCGECoordinatedNCV2(7, 5, "P0", init_seed=8).eval()
    m0.set_graph(edge, n); p0.set_graph(edge, n)
    with torch.no_grad():
        z_m = m0(x, edge)[0]
        z_p = p0(x, edge)[0]
    torch.testing.assert_close(z_m, z_p, atol=1e-6, rtol=1e-6)
    for layer in p0.layers:
        for expert in layer.experts:
            assert torch.equal(expert.theta0, torch.ones_like(expert.theta0))
            assert torch.equal(expert.theta1, torch.zeros_like(expert.theta1))
    torch.manual_seed(44)
    u = torch.randn(9, 128, requires_grad=True)
    op, _ = build_row_mean_operator(edge, n)
    neighbor = apply_r(op, u)
    phi1_a = neighbor + 0.5 * (apply_r(op, neighbor) - neighbor)
    phi1_b = 0.5 * (neighbor + apply_r(op, neighbor))
    torch.testing.assert_close(phi1_a, phi1_b)
    p0.train()
    _, _, _, balance, details = p0(x, edge, update=100)
    loss = details["g"][0].square().mean() + balance
    loss.backward()
    for expert in p0.layers[0].experts:
        assert expert.theta0.grad is not None and expert.theta1.grad is not None
        assert torch.isfinite(expert.theta0.grad).all() and torch.isfinite(expert.theta1.grad).all()
    with torch.no_grad():
        p0.layers[0].experts[0].theta0.fill_(-0.25)
        p0.layers[0].experts[0].theta1.fill_(-0.5)
    assert (p0.layers[0].experts[0].theta0 < 0).all()
    assert (p0.layers[0].experts[0].theta1 < 0).all()


def test_expert_has_independent_self_neighbor_maps_and_real_neighbor_response():
    model = MVCGECoordinatedNCV2(7, 5, "N0", init_seed=12).eval()
    expert = model.layers[0].experts[0]
    assert expert.w_self is not expert.w_nbr
    assert expert.w_self.weight.data_ptr() != expert.w_nbr.weight.data_ptr()
    u = torch.zeros(4, 128)
    n0 = torch.zeros_like(u)
    n1 = n0.clone(); n1[1] = 1
    with torch.no_grad():
        out0 = expert(u, n0)
        out1 = expert(u, n1)
    assert not torch.equal(out0[1], out1[1])


def test_context_and_two_modalities_update_synchronously_from_previous_states():
    x, edge = _data()
    model = MVCGECoordinatedNCV2(7, 5, "M0", init_seed=15).eval()
    model.set_graph(edge, x.size(0))
    captured = []
    hook = model.context.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        _, _, _, _, details = model(x, edge)
    hook.remove()
    assert len(captured) == 3  # c0, C(U^0), C(U^1)
    first_u = tuple(model.layers[0].pre_norm[m](details["h0"][m]) for m in range(2))
    torch.testing.assert_close(captured[1], torch.cat(first_u, dim=-1))
    second_u = tuple(model.layers[1].pre_norm[m](details["layers"][0]["next"][m]) for m in range(2))
    torch.testing.assert_close(captured[2], torch.cat(second_u, dim=-1))


def test_graph_cache_identity_version_size_device_and_all_true_mask_validation():
    x, edge_a = _data(n=9)
    model = MVCGECoordinatedNCV2(7, 5, "N0", init_seed=2).eval()
    model.set_graph(edge_a, 11)
    assert model._operator.shape == (11, 11)
    op_a = model._operator.to_dense().clone()
    edge_b = edge_a.clone()
    edge_b[0, 0], edge_b[1, 0] = 0, 8
    model.encode(torch.cat((x, torch.zeros(2, 12)), dim=0), edge_b, return_details=True)
    assert model._graph_source is edge_b
    assert model._operator.shape == (11, 11)
    assert not torch.equal(op_a, model._operator.to_dense())
    same_object = edge_b
    old = model._operator.to_dense().clone()
    with torch.no_grad():
        same_object[0, 0], same_object[1, 0] = 1, 8
    model.encode(torch.cat((x, torch.zeros(2, 12)), dim=0), same_object, return_details=True)
    assert model._graph_source is same_object
    assert not torch.equal(old, model._operator.to_dense())
    with pytest.raises(ValueError, match="all true"):
        model(x, edge_a, modality_available=torch.zeros((9, 2), dtype=torch.bool))
    if torch.cuda.is_available():
        model = model.cuda()
        edge_cuda = edge_a.cuda()
        x_cuda = x.cuda()
        model.encode(x_cuda, edge_cuda)
        assert model._operator.device.type == "cuda"


def test_router_schedule_dense_transition_sparse_eval_determinism_and_expected_load_gradients():
    assert [routing_eta(s) for s in (1, 50, 51, 75, 99, 100, 101)] == [0, 0, 0.02, 0.5, 0.98, 1, 1]
    logits = torch.randn(23, 4, generator=torch.Generator().manual_seed(3), requires_grad=True)
    gates, indices = sparse_top2(logits)
    assert (gates > 0).sum(-1).eq(2).all()
    torch.testing.assert_close(gates.sum(-1), torch.ones(23))
    sigma = torch.full_like(logits, 0.1, requires_grad=True)
    noisy = logits + torch.randn_like(logits) * sigma
    load = expected_load(logits, noisy, sigma)
    grad = torch.autograd.grad(load.square().sum(), (logits, sigma))
    assert torch.isfinite(load).all() and all(torch.isfinite(g).all() for g in grad)
    x, edge = _data()
    model = MVCGECoordinatedNCV2(7, 5, "M0", init_seed=18)
    model.set_graph(edge, x.size(0)); model.train()
    gen = torch.Generator().manual_seed(991)
    _, _, _, _, dense_details = model(x, edge, update=1, noise_generator=gen)
    assert all((layer["gates"][0] > 0).sum(-1).eq(4).all() for layer in dense_details["layers"])
    _, _, _, _, transition_details = model(x, edge, update=75, noise_generator=gen)
    assert all((layer["gates"][1] > 0).sum(-1).eq(4).all() for layer in transition_details["layers"])
    _, _, _, _, sparse_details = model(x, edge, update=100, noise_generator=gen)
    assert all((layer["gates"][0] > 0).sum(-1).eq(2).all() for layer in sparse_details["layers"])
    model.eval()
    with torch.no_grad():
        eval_a = model(x, edge, update=1)[0]
        eval_b = model(x, edge, update=99)[0]
    assert torch.equal(eval_a, eval_b)


def test_projection_stem_readout_dimensions_and_parameter_budget():
    model_budget = MVCGECoordinatedNCV2(768, 768, "P1", init_seed=5)
    assert sum(p.numel() for p in MVCGECoordinatedNCV2(768, 768, "N0", init_seed=5).parameters()) == 744_832
    assert sum(p.numel() for p in MVCGECoordinatedNCV2(768, 768, "M0", init_seed=5).parameters()) == 812_048
    assert sum(p.numel() for p in model_budget.parameters()) == 814_096
    model = MVCGECoordinatedNCV2(7, 5, "P1", init_seed=5)
    for stem in model.stems:
        torch.testing.assert_close(stem.weight, torch.eye(128))
        assert torch.count_nonzero(stem.bias) == 0
    x, edge = _data()
    model.eval(); model.set_graph(edge, x.size(0))
    with torch.no_grad():
        z, _, _, _, detail = model(x, edge)
    assert z.shape == (9, 640)
    torch.testing.assert_close(z, torch.cat((*detail["p"], detail["c0"], *detail["g"]), dim=-1))


def test_task_heads_train_validation_only_and_outside_labels_cannot_change_training_loss():
    labels = torch.arange(12) % 4
    train = torch.tensor([0, 1, 2, 3, 4, 5])
    val = torch.tensor([6, 7, 8])
    masked = labels_train_val_only(labels, train, val)
    changed = labels.clone(); changed[9:] = torch.tensor([3, 1, 2])
    masked_changed = labels_train_val_only(changed, train, val)
    assert torch.equal(masked, masked_changed)
    assert (masked[9:] == -1).all()
    head = CoordinatedNCHeads("M1", 4)
    z = torch.randn(12, 640, requires_grad=True)
    allowed = {"train": train, "validation": val}
    counters = {}
    logits = indexed_main_logits(head, z, train, split="train", allowed=allowed, counters=counters)
    assert logits.shape == (6, 4) and counters["train"] == 1
    with pytest.raises(ValueError, match="only train and validation"):
        indexed_main_logits(head, z, torch.tensor([9, 10, 11]), split="test", allowed=allowed, counters=counters)
    with pytest.raises(ValueError, match="do not match"):
        indexed_main_logits(head, z, torch.tensor([0, 1]), split="train", allowed=allowed, counters=counters)
    p = (torch.randn(12, 128, requires_grad=True), torch.randn(12, 128, requires_grad=True))
    labels_train = masked.index_select(0, train)
    main = head.main_logits(z.index_select(0, train))
    aux = head.semantic_logits(p, train)
    loss, _ = coordinated_loss("M1", main, labels_train, balance_loss=main.sum() * 0, semantic_logits=aux)
    grad = torch.autograd.grad(loss, (z, p[0], p[1]))
    assert grad[0][9:].abs().sum() == 0
    assert grad[1][6:].abs().sum() == 0 and grad[2][6:].abs().sum() == 0


def test_both_semantic_and_graph_paths_receive_task_gradients_and_adapter_is_near_identity():
    x, edge = _data()
    model = MVCGECoordinatedNCV2(7, 5, "M0", init_seed=42).train()
    model.set_graph(edge, x.size(0))
    head = CoordinatedNCHeads("M0", 4, init_seed=55).train()
    ids = torch.arange(6)
    labels = torch.arange(6) % 4
    z, _, _, balance, _ = model(x, edge, update=100, noise_generator=torch.Generator().manual_seed(3))
    logits = head.main_logits(z.index_select(0, ids))
    (F.cross_entropy(logits, labels) + balance).backward()
    required = [
        model.projectors[0].linear1.weight, model.projectors[1].linear1.weight,
        model.stems[0].weight, model.stems[1].weight, model.context[0].weight,
        model.layers[0].experts[0].w_self.weight, model.layers[0].experts[0].w_nbr.weight,
        model.layers[0].adapters[0].down.weight, model.layers[0].adapters[0].up.weight,
        model.layers[1].adapters[1].down.weight, model.delta_norms[0].weight,
    ]
    assert all(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in required)
    adapter = model.layers[0].adapters[0].eval()
    value = torch.randn(20, 128)
    with torch.no_grad():
        residual = adapter(value) - value
    assert residual.norm() / value.norm() < 0.1


def test_shared_initialization_is_stable_across_variants_and_optimizer_covers_once():
    models = {name: MVCGECoordinatedNCV2(7, 5, name, init_seed=32) for name in ("N0", "M0", "M1", "P0", "P1")}
    state = {name: model.state_dict() for name, model in models.items()}
    for variant in ("M0", "M1", "P0", "P1"):
        shared = set(state["N0"]) & set(state[variant])
        assert all(torch.equal(state["N0"][key], state[variant][key]) for key in shared if "router" not in key and "theta" not in key)
    head0 = initialize_head(777, 5, "M0", torch.device("cpu"))
    head1 = initialize_head(777, 5, "M1", torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(head0.main.state_dict().values(), head1.main.state_dict().values()))
    for name, model in models.items():
        groups = model.parameter_groups(0.001, 0.01)
        flat = [id(p) for group in groups for p in group["params"]]
        assert len(flat) == len(set(flat))
        assert set(flat) == {id(p) for p in model.parameters() if p.requires_grad}
    p_groups = models["P0"].parameter_groups(0.001, 0.01)
    theta_group = next(group for group in p_groups if group["name"] == "theta_decay")
    assert theta_group["lr"] == 0.001 and theta_group["weight_decay"] == 0.001
    router_group = next(group for group in p_groups if group["name"] == "router_decay")
    assert router_group["lr"] == 0.0005
    assert models["N0"].layers[0].router is None
    assert sum(p.numel() for p in CoordinatedNCHeads("M0", 20).parameters()) == 12_820
    assert sum(p.numel() for p in CoordinatedNCHeads("M1", 20).parameters()) == 17_980


def test_factory_five_tuple_and_synthetic_link_predictor_endpoint_product():
    cfg = OmegaConf.create({"model": {"name": "mvcge_coordinated_nc_v2", "variant": "M0"}, "seed": 27})
    info = {"text_dim": 7, "visual_dim": 5, "input_dim": 12, "num_nodes": 9, "num_classes": 4}
    model = build_model(cfg, info).eval()
    x, edge = _data()
    model.set_graph(edge, 9)
    with torch.no_grad():
        z, second, third, balance, extras = model(x, edge)
    assert z.shape == (9, 640) and second is None and third is None
    assert balance.ndim == 0 and "p" in extras
    decoder = LinkPredictor(640, 16, 1, 0.0)
    with torch.no_grad():
        decoder.net[0].weight.fill_(0.01); decoder.net[0].bias.fill_(0.2)
    left, right = z[:4], z[4:8]
    scores = decoder.score_pairs(left, right)
    expected = ((left * right) @ decoder.net[0].weight.t() + decoder.net[0].bias).flatten()
    torch.testing.assert_close(scores, expected)
    scores.sum().backward()
    assert decoder.net[0].weight.grad is not None


def test_diagnostics_leave_rng_gradients_and_model_mode_unchanged_and_vjp_sums():
    x, edge = _data()
    model = MVCGECoordinatedNCV2(7, 5, "M1", init_seed=55).train()
    model.set_graph(edge, x.size(0))
    head = CoordinatedNCHeads("M1", 4).train()
    noise = torch.Generator().manual_seed(77)
    z, _, _, balance, details = model(x, edge, update=100, noise_generator=noise)
    ids = torch.tensor([0, 1, 2, 3, 4])
    labels = torch.tensor([0, 1, 2, 3, 0])
    logits = head.main_logits(z.index_select(0, ids))
    aux = head.semantic_logits(details["p"], ids)
    rng_state = torch.random.get_rng_state().clone()
    noise_state = noise.get_state().clone()
    diagnostic = gradient_diagnostics(model, head, {"train_idx": ids}, z, details, logits, labels, aux)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(noise.get_state(), noise_state)
    assert model.training and head.training
    assert all(parameter.grad is None for parameter in list(model.parameters()) + list(head.parameters()))
    for row in diagnostic["shared_expert_message_exit_vjp"]:
        assert row["relative_error"] < 2e-5
    assert len(diagnostic["router_main_vs_weighted_balance"]) == 2
    assert "frontend_main_vs_weighted_semantic" in diagnostic
    data = {"x": x, "edges": edge, "n": x.size(0), "device": torch.device("cpu")}
    rng_state = torch.random.get_rng_state().clone()
    diag = model_diagnostics("M1", model, head, data, 33, 100)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert model.training and head.training
    assert diag["clean_routing"] and diag["noisy_routing"]
    assert all(parameter.grad is None for parameter in list(model.parameters()) + list(head.parameters()))


def test_best_checkpoint_replay_and_exact_rng_resume_state(tmp_path):
    x, edge = _data()
    data = {"dataset": "Movies", "x": x, "edges": edge, "n": 9, "text_dim": 7, "image_dim": 5, "classes": 4}
    model, head = build_modules(data, "M0", 63, torch.device("cpu"))
    optimizer = optimizer_for(model, head, "M0", 0.001, 0.001)
    model.set_graph(edge, 9); model.eval(); head.eval()
    with torch.no_grad():
        expected_z = model(x, edge)[0]
    noise = torch.Generator().manual_seed(1234)
    pair = torch.Generator().manual_seed(9876)
    checkpoint = tmp_path / "latest.pt"
    state = {"step": 4, "best_key": (3, 0.4, -3), "best_step": 3, "best_metrics": {"correct": 3}}
    _save_resume(checkpoint, model, head, optimizer, state, noise, pair, torch.device("cpu"))
    expected_torch = torch.rand(4)
    expected_noise = torch.randn(4, generator=noise)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    replay_model = MVCGECoordinatedNCV2(7, 5, "M0", init_seed=63)
    replay_model.set_graph(edge, 9); replay_model.load_state_dict(payload["model"]); replay_model.eval()
    with torch.no_grad():
        replay_z = replay_model(x, edge)[0]
    torch.testing.assert_close(expected_z, replay_z)
    replay_head = CoordinatedNCHeads("M0", 4); replay_head.load_state_dict(payload["head"])
    replay_optimizer = optimizer_for(replay_model, replay_head, "M0", 0.001, 0.001)
    replay_optimizer.load_state_dict(payload["optimizer"])
    replay_noise = torch.Generator().manual_seed(1); replay_pair = torch.Generator().manual_seed(2)
    torch.manual_seed(99)
    _restore_rng(payload, replay_noise, replay_pair, torch.device("cpu"))
    torch.testing.assert_close(torch.rand(4), expected_torch)
    torch.testing.assert_close(torch.randn(4, generator=replay_noise), expected_noise)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not visible in this Python process")
def test_cuda_synthetic_forward_backward_and_repeated_clean_eval():
    x, edge = _data()
    model = MVCGECoordinatedNCV2(7, 5, "P0", init_seed=81).cuda()
    x, edge = x.cuda(), edge.cuda()
    model.set_graph(edge, x.size(0))
    model.train()
    noise = torch.Generator(device="cuda").manual_seed(42)
    z, _, _, balance, details = model(x, edge, update=75, noise_generator=noise)
    loss = z.square().mean() + balance
    loss.backward()
    assert torch.isfinite(loss)
    assert model.layers[0].experts[0].theta0.grad is not None
    assert torch.isfinite(model.layers[0].experts[0].w_nbr.weight.grad).all()
    model.eval()
    with torch.no_grad():
        z1 = model(x, edge)[0]
        z2 = model(x, edge)[0]
    assert torch.equal(z1, z2)


def test_historical_a0_b1_and_fixed_task_terms_remain_available():
    x, edge = _data(n=12)
    info = {"input_dim": 12, "num_nodes": 12, "num_classes": 4, "text_dim": 7, "visual_dim": 5}
    model = MAGCollabAdapt(info, "B1", dropout=0.2, init_seed=71, typed_value=False, fusion=False, out_dim=768).train()
    head = CollabTaskHeads("B1", 4, output_dim=768)
    labels = torch.full((12,), -1, dtype=torch.long)
    train = torch.arange(0, 8)
    labels[train] = torch.arange(8) % 4
    available = torch.ones((12, 2), dtype=torch.bool)
    pair_generator = torch.Generator().manual_seed(808)
    z, _, _, _, extras = model(x, edge, available, pair_generator=pair_generator, compute_ssl=True)
    assert z.shape == (12, 768)
    loss, terms, _ = train_only_loss("B1", z, extras, head, labels, train, path_weight=0.05, rec_weight=0.01, pair_weight=0.01)
    torch.testing.assert_close(loss, terms["main_ce"] + 0.05 * terms["path_ce"] + 0.01 * terms["rec"] + 0.01 * terms["pair"])
    loss.backward()
    assert model.b1_updates[0][0].net[0].weight.grad is not None
