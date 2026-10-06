from __future__ import annotations

import inspect
import random

import torch
import torch.nn.functional as F

from src.models.mag_collab_nc import (
    MAGCollabNC,
    _CommonPath,
    _PrivatePath,
    _grouped_softmax,
    _grouped_softmax_with_null,
    parameter_groups_no_decay,
)
from src.tasks.collab_nc_pilot import CollabTaskHeads, train_only_loss
from src.models.predictor import LinkPredictor
from src.models.factory import build_model
from src.models.mag_prior_context import Model as LegacyPriorModel, canonical_directed_edges
from omegaconf import OmegaConf
from scripts.run_collab_nc_pilot_v1 import make_config


INFO = {"text_dim": 5, "visual_dim": 7, "input_dim": 12}


def _data():
    torch.manual_seed(11)
    x = torch.randn(8, 12)
    edge = torch.tensor([[0, 1, 0, 2, 3, 4, 5], [2, 2, 3, 4, 5, 6, 7]])
    available = torch.ones(8, 2, dtype=torch.bool)
    return x, edge, available


def test_softmax_axes_and_null_mass_match_dense_reference():
    logits = torch.tensor([[0.2, 1.0], [0.7, -0.3], [-0.1, 0.4], [0.5, 0.1]])
    group = torch.tensor([1, 1, 3, 3])
    valid = torch.tensor([True, True, True, False])
    got = _grouped_softmax(logits, group, valid, 5)
    expected = torch.zeros_like(got)
    expected[group == 1] = torch.softmax(logits[group == 1], dim=0)
    expected[2] = 1.0
    torch.testing.assert_close(got, expected)
    assert torch.allclose(got[group == 1].sum(0), torch.ones(2))
    assert torch.allclose(got[group == 3].sum(0), torch.tensor([1.0, 1.0]))

    dst = torch.tensor([1, 1, 1, 2])
    edge_logits = torch.zeros(4, 2)
    valid = torch.tensor([True, True, True, True])
    degree = torch.tensor([0.0, 3.0, 1.0, 0.0, 0.0])
    null = torch.where(degree[:, None] > 0, degree.clamp_min(1).log()[:, None] - torch.log(torch.tensor(9.0)), torch.zeros(5, 1)).expand(-1, 2)
    edge_mass, null_mass = _grouped_softmax_with_null(edge_logits, dst, valid, null, 5)
    torch.testing.assert_close(null_mass[1], torch.full((2,), 0.1))
    torch.testing.assert_close(null_mass[2], torch.full((2,), 0.1))
    torch.testing.assert_close(null_mass[[0, 3, 4]], torch.ones(3, 2))
    for node in (1, 2):
        torch.testing.assert_close(edge_mass[dst == node].sum(0) + null_mass[node], torch.ones(2))


def test_common_path_selects_sender_modality_per_edge_and_neighbor_axis_per_receiver():
    torch.manual_seed(12)
    module = _CommonPath().eval()
    states = (torch.randn(4, 64), torch.randn(4, 64))
    center = torch.randn(4, 128)
    raw = (torch.randn(4, 5), torch.randn(4, 7))
    available = torch.tensor([[True, False], [False, True], [True, True], [True, True]])
    edges = torch.tensor([[0, 1], [2, 2]])
    updates, stats = module(states, center, raw, available, edges, 1.0)
    pi = stats["modality_weights"]
    assert pi.shape == (2, 2, 2, 2)  # edge, receiving modality, head, sending modality
    torch.testing.assert_close(pi.sum(-1), torch.ones(2, 2, 2))
    assert torch.all(pi[0, ..., 0] == 1) and torch.all(pi[0, ..., 1] == 0)
    assert torch.all(pi[1, ..., 0] == 0) and torch.all(pi[1, ..., 1] == 1)
    alpha = stats["neighbor_weights"]
    null = stats["null_probability"]
    torch.testing.assert_close(alpha.sum(0) + null[2], torch.ones(2, 2))

    changed = (states[0], states[1].clone())
    changed[1][1] = changed[1][1] + 2.0
    changed_updates, _ = module(changed, center, raw, available, edges, 1.0)
    assert not torch.allclose(updates[0][2], changed_updates[0][2])
    empty, empty_stats = module(states, center, raw, available, torch.empty((2, 0), dtype=torch.long), 1.0)
    assert torch.equal(empty[0], torch.zeros_like(empty[0]))
    assert torch.equal(empty[1], torch.zeros_like(empty[1]))
    assert torch.equal(empty_stats["null_probability"], torch.ones_like(empty_stats["null_probability"]))


def test_private_path_null_and_difference_boundary():
    torch.manual_seed(13)
    private = _PrivatePath().eval()
    state = torch.randn(5, 64)
    raw = torch.randn(5, 6)
    available = torch.ones(5, dtype=torch.bool)
    edges = torch.empty((2, 0), dtype=torch.long)
    update, stats = private(state, raw, available, edges, 1.0)
    assert torch.equal(update, torch.zeros_like(update))
    assert torch.equal(stats["rho"], torch.zeros_like(stats["rho"]))
    assert torch.equal(private.phi_diff(torch.zeros(3, 64)), torch.zeros(3, 64))

    constant = torch.ones(5, 64)
    one_in = torch.tensor([[0], [1]])
    # With identical node states, the conditional value difference is zero.
    q = private.value(constant).view(5, 2, 32)
    mu = q[0].expand(5, -1, -1).reshape(5, 64)
    delta = q.reshape(5, 64) - mu
    torch.testing.assert_close(private.phi_diff(delta), torch.zeros_like(delta), atol=1e-7, rtol=0)
    assert one_in.shape == (2, 1)


def test_empty_graph_missing_modalities_and_source_private_independence():
    x, edge, available = _data()
    model = MAGCollabNC(INFO, "M1").eval()
    empty = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        z_empty, _, _, _, extras_empty = model(x, empty, available)
        z_real, _, _, _, _ = model(x, edge, available)
    torch.testing.assert_close(z_empty[:, 384:], torch.zeros_like(z_empty[:, 384:]), atol=0, rtol=0)
    torch.testing.assert_close(z_real[:, :384], z_empty[:, :384], atol=0, rtol=0)
    assert not torch.allclose(z_real, z_empty)

    changed = x.clone()
    changed[:, INFO["text_dim"]:] *= -3.0
    with torch.no_grad():
        _, _, _, _, before = model(x, edge, available)
        _, _, _, _, after = model(changed, edge, available)
    torch.testing.assert_close(before["source_states"]["text"], after["source_states"]["text"], atol=0, rtol=0)

    broken = x.clone()
    broken[0, :5] = float("nan")
    availability = available.clone()
    availability[0] = False
    availability[1, 1] = False
    availability[2, 0] = False
    with torch.no_grad():
        z, _, _, _, extras = model(broken, edge, availability)
    assert torch.isfinite(z).all()
    assert torch.equal(extras["availability"][0], torch.tensor([False, False]))
    assert torch.equal(extras["availability"][1], torch.tensor([True, False]))


def test_640_readout_uses_private_then_common_increment_order():
    x, edge, available = _data()
    model = MAGCollabNC(INFO, "M1").eval()
    with torch.no_grad():
        z, _, _, _, extras = model(x, edge, available)
    for index, block in enumerate(extras["graph_increments"]):
        torch.testing.assert_close(z[:, 384 + index * 64:384 + (index + 1) * 64], model.beta * block, atol=0, rtol=0)


def test_b0_keeps_identical_own_path_and_never_reads_graph():
    x, edge, available = _data()
    torch.manual_seed(23)
    ego = MAGCollabNC(INFO, "B0").eval()
    torch.manual_seed(23)
    full = MAGCollabNC(INFO, "M1").eval()
    with torch.no_grad():
        z0 = ego(x, edge, available)[0]
        z1 = full(x, edge, available)[0]
        z2 = ego(x, torch.flip(edge, (1,)), available)[0]
    torch.testing.assert_close(z0[:, :384], z1[:, :384], atol=0, rtol=0)
    torch.testing.assert_close(z0, z2, atol=0, rtol=0)
    assert z0.shape == (8, 640)
    assert not any("layers" in name for name, _ in ego.named_parameters())
    assert "labels" not in inspect.signature(ego.forward).parameters


def test_node_permutation_equivariance():
    x, edge, available = _data()
    model = MAGCollabNC(INFO, "M1").eval()
    permutation = torch.tensor([4, 1, 7, 0, 5, 2, 6, 3])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted_edges = inverse[edge]
    with torch.no_grad():
        z = model(x, edge, available)[0]
        z_permuted = model(x[permutation], permuted_edges, available[permutation])[0]
    torch.testing.assert_close(z_permuted, z[permutation], atol=1e-6, rtol=1e-6)


def test_task_loss_indexes_train_only_and_optimizer_updates_paths_ssl_and_heads():
    x, edge, available = _data()
    model = MAGCollabNC(INFO, "M0")
    heads = CollabTaskHeads("M0", 3)
    params = parameter_groups_no_decay(model, [heads])
    optimizer = torch.optim.AdamW(params, lr=0.003, weight_decay=1e-4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0])
    train = torch.tensor([0, 1, 2, 3, 4, 5])
    pair_rng = torch.Generator().manual_seed(45)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    before_head = {name: p.detach().clone() for name, p in heads.named_parameters()}
    for step in (1, 2):
        model.train()
        model.set_epoch(step)
        optimizer.zero_grad(set_to_none=True)
        z, _, _, _, extras = model(x, edge, available, pair_generator=pair_rng, compute_ssl=True)
        loss, terms, _ = train_only_loss("M0", z, extras, heads, labels, train)
        altered = labels.clone()
        altered[6:] = torch.tensor([2, 2])
        same, _, _ = train_only_loss("M0", z, extras, heads, altered, train)
        torch.testing.assert_close(loss, same, atol=0, rtol=0)
        assert all(torch.isfinite(value) for value in terms.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        optimizer.step()
    for key in ("layers.0.private.0.update.net.0.weight", "layers.1.common.ffn.net.0.weight", "pair_projection.weight", "decoder_text.weight"):
        assert not torch.equal(before[key], dict(model.named_parameters())[key])
    assert any(not torch.equal(before_head[name], p) for name, p in heads.named_parameters())
    for module in (
        model.layers[0].private[0].query, model.layers[1].private[1].value,
        model.layers[0].private[0].gamma, model.layers[0].common.query,
        model.layers[0].common.relation_same, model.layers[0].common.relation_cross,
        model.layers[0].common.edge_query, model.layers[0].common.edge_score,
        model.layers[1].common.null_score, model.layers[1].common.output,
        model.layers[1].common.ffn,
    ):
        parameters = list(module.parameters()) if isinstance(module, torch.nn.Module) else [module]
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert model.beta == 0.1 + 0.9 / 19.0 and model.tau == 2.0 - 1.0 / 19.0


def test_m0_m1_shared_initialization_eval_and_checkpoint_restore(tmp_path):
    x, edge, available = _data()
    torch.manual_seed(71)
    m0 = MAGCollabNC(INFO, "M0").eval()
    torch.manual_seed(71)
    m1 = MAGCollabNC(INFO, "M1").eval()
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert all(torch.equal(s0[key], s1[key]) for key in s1)
    with torch.no_grad():
        expected = m0(x, edge, available)[0]
        repeated = m0(x, edge, available)[0]
    assert torch.equal(expected, repeated)
    heads = CollabTaskHeads("M0", 3)
    optimizer = torch.optim.AdamW(parameter_groups_no_decay(m0, [heads]), lr=0.001)
    # Populate optimizer moments before save so restoration tests state, not just parameter copies.
    labels = torch.arange(8) % 3
    train = torch.arange(6)
    pair_rng = torch.Generator().manual_seed(6)
    m0.train()
    optimizer.zero_grad(set_to_none=True)
    z0, _, _, _, ex0 = m0(x, edge, available, pair_generator=pair_rng, compute_ssl=True)
    loss, _, _ = train_only_loss("M0", z0, ex0, heads, labels, train)
    loss.backward()
    optimizer.step()
    m0.eval()
    with torch.no_grad():
        expected = m0(x, edge, available)[0]
    path = tmp_path / "resume.pt"
    torch.save({"model": m0.state_dict(), "heads": heads.state_dict(), "optimizer": optimizer.state_dict(), "step": 17}, path)
    restored = MAGCollabNC(INFO, "M0").eval()
    restored_heads = CollabTaskHeads("M0", 3)
    restored_optimizer = torch.optim.AdamW(parameter_groups_no_decay(restored, [restored_heads]), lr=0.001)
    state = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_state_dict(state["model"])
    restored_heads.load_state_dict(state["heads"])
    restored_optimizer.load_state_dict(state["optimizer"])
    with torch.no_grad():
        actual = restored(x, edge, available)[0]
    assert torch.equal(expected, actual)
    assert state["step"] == 17


def test_lp_endpoint_product_shape_backpropagates():
    x, edge, available = _data()
    model = MAGCollabNC(INFO, "M1")
    z = model(x, edge, available)[0]
    predictor = LinkPredictor(model.out_dim, 16, 2, 0.2)
    src, dst = torch.tensor([0, 2]), torch.tensor([1, 3])
    scores = predictor.score_pairs(z[src], z[dst])
    scores.sum().backward()
    assert scores.shape == (2,)
    assert model.p_text.weight.grad is not None


def test_small_train_set_can_reduce_supervised_loss():
    x, edge, available = _data()
    model = MAGCollabNC(INFO, "B0")
    heads = CollabTaskHeads("B0", 3)
    optimizer = torch.optim.AdamW(parameter_groups_no_decay(model, [heads]), lr=0.01, weight_decay=1e-4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    idx = torch.arange(8)
    losses = []
    for _ in range(20):
        model.train(); heads.train()
        optimizer.zero_grad(set_to_none=True)
        z = model(x, edge, available)[0]
        loss, _, _ = train_only_loss("B0", z, {"path_states": {}, "loss_terms": {"rec": z.new_zeros(()), "pair": z.new_zeros(())}}, heads, labels, idx)
        losses.append(float(loss.detach()))
        loss.backward(); optimizer.step()
    assert losses[-1] < losses[0]


def test_new_model_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    x, edge, available = _data()
    model = MAGCollabNC(INFO, "M1").to(device)
    heads = CollabTaskHeads("M1", 3).to(device)
    labels = (torch.arange(8) % 3).to(device)
    train = torch.arange(6, device=device)
    model.eval()
    with torch.no_grad():
        z, _, _, _, extras = model(x.to(device), edge.to(device), available.to(device))
        repeated = model(x.to(device), edge.to(device), available.to(device))[0]
    torch.testing.assert_close(z, repeated, atol=1e-6, rtol=1e-6)
    assert torch.equal(z.argmax(-1), repeated.argmax(-1))
    model.train()
    loss, _, _ = train_only_loss("M1", z, extras, heads, labels, train)
    loss.backward()
    assert z.is_cuda and torch.isfinite(z).all() and torch.isfinite(loss)


def test_factory_constructor_and_test_off_configuration():
    cfg = make_config("Movies", "M1", 55, 0.001, 0.0001, "cpu")
    assert cfg.task.evaluate_test is False
    model = build_model(cfg, INFO)
    assert model.out_dim == 640
    assert "labels" not in inspect.signature(model.forward).parameters


def test_b2_legacy_r1_adapter_has_train_only_main_and_source_objective():
    x, edge, _ = _data()
    cfg = OmegaConf.create({"model": {"variant": "R1", "width": 144, "dropout": 0.2}})
    model = LegacyPriorModel(cfg, INFO)
    heads = CollabTaskHeads("B2", 3, output_dim=128)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 2, 2])
    train = torch.arange(6)
    clean_edges = canonical_directed_edges(edge, x.size(0))
    z, _, _, _, extras = model(x, clean_edges)
    loss, terms, logits = train_only_loss("B2", z, extras, heads, y, train)
    altered = y.clone()
    altered[6:] = torch.tensor([0, 1])
    altered_loss, _, _ = train_only_loss("B2", z, extras, heads, altered, train)
    torch.testing.assert_close(loss, altered_loss, atol=0, rtol=0)
    assert z.shape == (8, 128) and logits["main"].shape == (6, 3)
    assert torch.isfinite(terms["source_ce"])
    loss.backward()
    assert model.adapter_text[0].weight.grad is not None
    assert heads.legacy.source_text.weight.grad is not None


def test_interrupted_resume_matches_continuous_step(tmp_path):
    x, edge, available = _data()
    labels = torch.arange(8) % 3
    train = torch.arange(6)

    def build():
        torch.manual_seed(91)
        model = MAGCollabNC(INFO, "M0", init_seed=91)
        torch.manual_seed(100)
        heads = CollabTaskHeads("M0", 3)
        optimizer = torch.optim.AdamW(parameter_groups_no_decay(model, [heads]), lr=0.001, weight_decay=1e-4)
        return model, heads, optimizer

    def take_step(model, heads, optimizer, pair_rng, step):
        model.train(); heads.train(); model.set_epoch(step)
        optimizer.zero_grad(set_to_none=True)
        z, _, _, _, extras = model(x, edge, available, pair_generator=pair_rng, compute_ssl=True)
        loss, _, _ = train_only_loss("M0", z, extras, heads, labels, train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        optimizer.step()
        return float(loss.detach())

    random.seed(17); torch.manual_seed(18)
    model, heads, optimizer = build()
    pair_rng = torch.Generator().manual_seed(19)
    take_step(model, heads, optimizer, pair_rng, 1)
    saved_rng = {"python": random.getstate(), "torch": torch.get_rng_state(), "pair": pair_rng.get_state()}
    checkpoint = tmp_path / "resume.pt"
    torch.save({"model": model.state_dict(), "heads": heads.state_dict(), "optimizer": optimizer.state_dict(), "rng": saved_rng, "step": 1}, checkpoint)
    expected_loss = take_step(model, heads, optimizer, pair_rng, 2)
    expected_params = {**{f"m.{k}": v.detach().clone() for k, v in model.state_dict().items()}, **{f"h.{k}": v.detach().clone() for k, v in heads.state_dict().items()}}

    restored, restored_heads, restored_optimizer = build()
    restored_pair = torch.Generator().manual_seed(999)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(state["model"]); restored_heads.load_state_dict(state["heads"]); restored_optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["rng"]["python"]); torch.set_rng_state(state["rng"]["torch"]); restored_pair.set_state(state["rng"]["pair"])
    actual_loss = take_step(restored, restored_heads, restored_optimizer, restored_pair, 2)
    assert abs(expected_loss - actual_loss) < 1e-8
    actual_params = {**{f"m.{k}": v for k, v in restored.state_dict().items()}, **{f"h.{k}": v for k, v in restored_heads.state_dict().items()}}
    assert all(torch.equal(expected_params[name], actual_params[name]) for name in expected_params)
    assert state["step"] == 1
