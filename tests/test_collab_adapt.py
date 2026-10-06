from __future__ import annotations

import inspect
import copy

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.factory import build_model
from src.models.mag_collab_adapt import (
    MAGCollabAdapt,
    _CommonPath,
    _PrivatePath,
    _grouped_softmax,
    _grouped_softmax_with_null,
)
from src.models.mag_collab_nc import MAGCollabNC
from src.models.predictor import LinkPredictor
from src.tasks.collab_nc_pilot import CollabTaskHeads, train_only_loss


INFO = {"text_dim": 5, "visual_dim": 7, "input_dim": 12}


def graph():
    torch.manual_seed(901)
    x = torch.randn(8, 12)
    edge = torch.tensor([[0, 1, 3, 4, 5, 0], [2, 2, 2, 5, 6, 7]])
    available = torch.ones(8, 2, dtype=torch.bool)
    return x, edge, available


def test_masked_grouped_softmax_extreme_invalid_logit_has_finite_nonzero_valid_gradient():
    logits = torch.tensor([[0.3], [1.2], [1000.0]], requires_grad=True)
    group = torch.tensor([0, 0, 0])
    valid = torch.tensor([True, True, False])
    weights = _grouped_softmax(logits, group, valid, 2)
    torch.testing.assert_close(weights[:2].sum(0), torch.ones(1))
    assert weights[2].item() == 0.0
    (weights[0] * 1.0 + weights[1] * 3.0).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:2].abs().sum().item() > 0
    assert logits.grad[2].item() == 0.0

    edge_logits = torch.tensor([[0.3], [1.2], [1000.0]], requires_grad=True)
    null_logits = torch.tensor([[0.1], [0.5]], requires_grad=True)
    edge_weight, null_weight = _grouped_softmax_with_null(edge_logits, group, valid, null_logits, 2)
    torch.testing.assert_close(edge_weight[:2].sum(0) + null_weight[0], torch.ones(1))
    torch.testing.assert_close(null_weight[1], torch.ones(1))
    (edge_weight[0] + 3 * edge_weight[1] + 2 * null_weight[0]).sum().backward()
    assert torch.isfinite(edge_logits.grad).all() and torch.isfinite(null_logits.grad).all()
    assert edge_logits.grad[:2].abs().sum().item() > 0
    assert edge_logits.grad[2].item() == 0.0
    assert torch.equal(_grouped_softmax(edge_logits.detach()[:0], group[:0], valid[:0], 2), edge_logits.detach()[:0])
    all_invalid, isolated = _grouped_softmax_with_null(edge_logits.detach(), group, torch.zeros_like(valid), null_logits.detach(), 2)
    assert torch.equal(all_invalid, torch.zeros_like(all_invalid))
    torch.testing.assert_close(isolated, torch.ones_like(isolated))


def test_special_initialization_is_restored_for_m0_m1_s5_and_checkpoint_load_is_not_reset():
    models = (
        MAGCollabAdapt(INFO, "M0", init_seed=12),
        MAGCollabAdapt(INFO, "M1", init_seed=13),
        MAGCollabAdapt(INFO, "M0", init_seed=14, typed_value=True, fusion=True, out_dim=768),
    )
    for candidate in models:
        for layer in candidate.layers:
            for path in layer.private:
                assert torch.count_nonzero(path.gate.net[-1].weight) == 0
                assert torch.count_nonzero(path.gate.net[-1].bias) == 0
                assert torch.count_nonzero(path.null_score.net[-1].weight) == 0
                torch.testing.assert_close(path.null_score.net[-1].bias, torch.full_like(path.null_score.net[-1].bias, torch.log(torch.tensor(1 / 9))))
            for null in layer.common.null_score:
                assert torch.count_nonzero(null.net[-1].weight) == 0
                torch.testing.assert_close(null.net[-1].bias, torch.full_like(null.net[-1].bias, torch.log(torch.tensor(1 / 9))))
            torch.testing.assert_close(layer.common.relation_same, torch.eye(32).repeat(2, 1, 1))
            torch.testing.assert_close(layer.common.relation_cross, torch.eye(32).repeat(2, 1, 1))
            torch.testing.assert_close(layer.common.output[0], torch.eye(32).repeat(2, 1, 1))
            torch.testing.assert_close(layer.private[0].gamma, torch.ones(2))
    model = models[0]
    with torch.no_grad():
        model.layers[0].private[0].gate.net[-1].bias.fill_(0.37)
        model.layers[0].private[0].null_score.net[-1].bias.fill_(-2.0)
    restored = MAGCollabAdapt(INFO, "M0", init_seed=99)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.layers[0].private[0].gate.net[-1].bias, torch.full_like(restored.layers[0].private[0].gate.net[-1].bias, 0.37))
    torch.testing.assert_close(restored.layers[0].private[0].null_score.net[-1].bias, torch.full_like(restored.layers[0].private[0].null_score.net[-1].bias, -2.0))


def test_v_identity_initialization_matches_off_forward_and_adds_exact_active_parameter_delta():
    x, edge, available = graph()
    off = MAGCollabAdapt(INFO, "M0", init_seed=123, typed_value=False).eval()
    on = MAGCollabAdapt(INFO, "M0", init_seed=123, typed_value=True).eval()
    with torch.no_grad():
        z_off = off(x, edge, available)[0]
        z_on = on(x, edge, available)[0]
    torch.testing.assert_close(z_on, z_off, atol=1e-6, rtol=1e-5)
    base_parameters = sum(p.numel() for p in off.parameters())
    typed_parameters = sum(p.numel() for p in on.parameters())
    assert typed_parameters - base_parameters == 16_384
    for layer in on.layers:
        for mapping in layer.common.value_sender:
            torch.testing.assert_close(mapping.weight, layer.common.value_sender[0].weight)
        torch.testing.assert_close(layer.common.message_same, torch.eye(32).repeat(2, 1, 1))
        torch.testing.assert_close(layer.common.message_cross, torch.eye(32).repeat(2, 1, 1))


def test_fusion_zero_anchor_empty_graph_own_blocks_and_nonzero_gradient_path():
    x, edge, available = graph()
    model = MAGCollabAdapt(INFO, "M0", init_seed=234, fusion=True, out_dim=768).eval()
    empty = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        z_empty, _, _, _, extras_empty = model(x, empty, available)
        model.fusion_bias.fill_(4.0)
        z_empty_bias, _, _, _, _ = model(x, empty, available)
    torch.testing.assert_close(z_empty[:, :384], z_empty_bias[:, :384], atol=0, rtol=0)
    assert torch.count_nonzero(z_empty[:, 384:]) == 0
    assert torch.count_nonzero(z_empty_bias[:, 640:]) == 0
    assert all(torch.count_nonzero(block) == 0 for block in extras_empty["graph_increments"])

    model.train()
    z, _, _, _, extras = model(x, edge, available)
    assert extras["graph_increments"][0].norm() > 0
    loss = z[:, 640:].square().sum()
    loss.backward()
    for parameter in (model.fusion_e.weight, model.fusion_delta.weight, model.fusion_out.weight, model.fusion_bias):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.p_text.weight.grad is not None and model.p_text.weight.grad.abs().sum() > 0


def test_fusion_disabled_padding_and_softmax_axes_empty_sender_and_isolated_nodes():
    x, edge, available = graph()
    model = MAGCollabAdapt(INFO, "M0", init_seed=345, out_dim=768, fusion=False).eval()
    with torch.no_grad():
        z, _, _, _, extras = model(x, edge, available)
    assert z.shape == (8, 768) and torch.equal(z[:, 640:], torch.zeros_like(z[:, 640:]))
    path = _PrivatePath().eval()
    state = torch.randn(4, 64)
    empty_edge = torch.empty((2, 0), dtype=torch.long)
    message, stats = path(state, torch.randn(4, 5), torch.ones(4, dtype=torch.bool), empty_edge, 1.0)
    assert torch.equal(message, torch.zeros_like(message))

    common = _CommonPath().eval()
    states = (torch.randn(4, 64), torch.randn(4, 64))
    center = torch.randn(4, 128)
    raw = (torch.randn(4, 5), torch.randn(4, 7))
    no_available = torch.zeros(4, 2, dtype=torch.bool)
    result, diag = common(states, center, raw, no_available, torch.tensor([[0, 1], [2, 2]]), 1.0, collect_diagnostics=True)
    assert all(torch.isfinite(v).all() for v in result)
    assert torch.equal(diag["modality_weights"], torch.zeros_like(diag["modality_weights"]))
    assert torch.equal(diag["null_probability"], torch.ones_like(diag["null_probability"]))


def test_common_sender_and_receiver_softmaxes_use_distinct_axes_and_preserve_source_independence():
    torch.manual_seed(92)
    common = _CommonPath(typed_value=True).eval()
    state = (torch.randn(5, 64), torch.randn(5, 64))
    center = torch.randn(5, 128)
    raw = (torch.randn(5, 5), torch.randn(5, 7))
    available = torch.tensor([[True, False], [False, True], [True, True], [True, True], [True, True]])
    edge = torch.tensor([[0, 1, 2, 3], [4, 4, 4, 2]])
    updates, stats = common(state, center, raw, available, edge, 1.0, collect_diagnostics=True)
    pi, alpha, null = stats["modality_weights"], stats["neighbor_weights"], stats["null_probability"]
    assert pi.shape == (4, 2, 2, 2)
    torch.testing.assert_close(pi.sum(-1), torch.ones(4, 2, 2))
    assert torch.all(pi[0, ..., 0] == 1) and torch.all(pi[0, ..., 1] == 0)
    assert torch.all(pi[1, ..., 0] == 0) and torch.all(pi[1, ..., 1] == 1)
    for receiver, edge_rows in ((4, (0, 1, 2)), (2, (3,))):
        torch.testing.assert_close(alpha[list(edge_rows)].sum(0) + null[receiver], torch.ones(2, 2))
    for receiver in (0, 1, 3):
        assert torch.equal(null[receiver], torch.ones_like(null[receiver]))
        assert not torch.any(alpha[edge[1] == receiver])
    assert all(torch.isfinite(v).all() for v in updates)

    # Layer-one private source state must depend only on its original modality.
    x, graph_edges, mask = graph()
    model = MAGCollabAdapt(INFO, "M1", init_seed=93).eval()
    changed_image = x.clone(); changed_image[:, INFO["text_dim"]:] *= -2.3
    with torch.no_grad():
        source_a = model(x, graph_edges, mask)[4]["source_states"]["text"]
        source_b = model(changed_image, graph_edges, mask)[4]["source_states"]["text"]
    torch.testing.assert_close(source_a, source_b, atol=0, rtol=0)


def test_reference_old_model_and_optimized_model_match_forward_and_gradients():
    x, edge, available = graph()
    reference = MAGCollabNC(INFO, "M1", init_seed=444)
    optimized = MAGCollabAdapt(INFO, "M1", init_seed=777, optimize_cache=True)
    optimized.load_state_dict(reference.state_dict())
    reference.eval(); optimized.eval()
    ref_z = reference(x, edge, available)[0]
    opt_z = optimized(x, edge, available, optimize_cache=True)[0]
    torch.testing.assert_close(opt_z, ref_z, atol=1e-6, rtol=1e-5)
    probe = torch.randn_like(ref_z)
    (ref_z * probe).sum().backward()
    (opt_z * probe).sum().backward()
    ref_parameters, opt_parameters = dict(reference.named_parameters()), dict(optimized.named_parameters())
    assert ref_parameters.keys() == opt_parameters.keys()
    for name in ref_parameters:
        if ref_parameters[name].grad is not None:
            # Moving the identity-linear relation transform to node space
            # changes CPU reduction order slightly; observed close-call error
            # is below 2e-6 for elements around 3e-2.
            torch.testing.assert_close(opt_parameters[name].grad, ref_parameters[name].grad, atol=2e-6, rtol=1e-5, msg=name)


def test_b1_baseline_matches_frozen_same_modality_mean_reference():
    x, edge, available = graph()
    reference = MAGCollabNC(INFO, "B1", init_seed=333).eval()
    adapted = MAGCollabAdapt(INFO, "B1", init_seed=777).eval()
    adapted.load_state_dict(reference.state_dict())
    with torch.no_grad():
        ref_z = reference(x, edge, available)[0]
        new_z = adapted(x, edge, available)[0]
    assert ref_z.shape == new_z.shape == (8, 640)
    torch.testing.assert_close(new_z, ref_z, atol=1e-6, rtol=1e-5)


def test_cache_refreshes_for_same_shape_features_edge_mask_mutation_and_inference_mode():
    x, edge, available = graph()
    cached = MAGCollabAdapt(INFO, "M1", init_seed=555, optimize_cache=True).eval()
    fresh = MAGCollabAdapt(INFO, "M1", init_seed=666, optimize_cache=False).eval()
    fresh.load_state_dict(cached.state_dict())

    def compare(xv, ev, av):
        a = cached(xv, ev, av)[0]
        b = fresh(xv, ev, av, optimize_cache=False)[0]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    compare(x, edge, available)
    compare(x + 0.1, edge, available)  # same shape, different feature identity
    changed_edges = edge[:, torch.tensor([0, 1, 2, 3, 4])].clone()
    compare(x, changed_edges, available)
    changed_mask = available.clone(); changed_mask[0, 0] = False
    compare(x, edge, changed_mask)
    in_place_x, in_place_edge, in_place_mask = x.clone(), edge.clone(), available.clone()
    compare(in_place_x, in_place_edge, in_place_mask)
    in_place_x.add_(0.25)
    compare(in_place_x, in_place_edge, in_place_mask)
    in_place_edge[:, 0] = torch.tensor([2, 1])
    compare(in_place_x, in_place_edge, in_place_mask)
    in_place_mask[0, 0] = False
    compare(in_place_x, in_place_edge, in_place_mask)
    with torch.inference_mode():
        inferred = cached(x, edge, available)[0]
    assert torch.isfinite(inferred).all()
    differentiable = x.clone().requires_grad_(True)
    differentiable_out = cached(differentiable, edge, available)[0]
    differentiable_out.square().sum().backward()
    assert differentiable.grad is not None and torch.isfinite(differentiable.grad).all()


def test_factory_models_and_endpoint_product_head_use_new_readout_width():
    for variant, v, f in (("B1", False, False), ("B1", False, True), ("M0", False, False), ("M0", True, True)):
        cfg = OmegaConf.create({"seed": 10, "model": {"name": "mag_collab_adapt", "base_variant": variant, "out_dim": 768, "dropout": 0.2, "typed_value": v, "fusion": f, "collect_diagnostics": False, "optimize_cache": True}})
        model = build_model(cfg, INFO)
        x, edge, available = graph()
        result = model(x, edge, available)
        assert len(result) == 5 and result[0].shape == (8, 768)
        assert result[4]["path_dims"] == {"text_private": 64, "image_private": 64, "shared": 128}
        lp_head = LinkPredictor(768, 32, 2, 0.2)
        z = result[0]
        score = lp_head.score_pairs(z[edge[0]], z[edge[1]])
        assert score.shape == (edge.size(1),) and torch.isfinite(score).all()
    assert "labels" not in inspect.signature(MAGCollabAdapt.forward).parameters


def test_complete_checkpoint_resume_restores_optimizer_pair_dropout_rng_and_schedule():
    x, edge, available = graph()
    labels = torch.tensor([0, 1, 2, 0, 1, 2, -1, -1])
    train = torch.arange(6)

    def setup(seed):
        torch.manual_seed(seed)
        model = MAGCollabAdapt(INFO, "M0", init_seed=seed, typed_value=True, fusion=True, out_dim=768)
        torch.manual_seed(seed + 10_007)
        heads = CollabTaskHeads("M0", 3, output_dim=768)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=0.002, weight_decay=0.001)
        pair_rng = torch.Generator().manual_seed(seed + 31_415)
        return model, heads, optimizer, pair_rng

    def step(model, heads, optimizer, pair_rng, number):
        model.train(); heads.train(); model.set_epoch(number); optimizer.zero_grad(set_to_none=True)
        z, _, _, _, extras = model(x, edge, available, pair_generator=pair_rng, compute_ssl=True)
        # Loss receives only the explicit train IDs; sentinel labels at all
        # remaining nodes are never read by the task adapter.
        loss, _, _ = train_only_loss("M0", z, extras, heads, labels, train, rec_weight=0.01, pair_weight=0.01)
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0); optimizer.step()
        return float(loss.detach())

    torch.manual_seed(808)
    full, full_heads, full_opt, full_pair = setup(808)
    loss1 = step(full, full_heads, full_opt, full_pair, 1)
    checkpoint = {
        "model": copy.deepcopy(full.state_dict()), "heads": copy.deepcopy(full_heads.state_dict()), "optimizer": copy.deepcopy(full_opt.state_dict()),
        "torch_cpu_rng": torch.get_rng_state(), "pair_rng": full_pair.get_state(), "beta": full.beta, "tau": full.tau,
    }
    loss2 = step(full, full_heads, full_opt, full_pair, 2)
    expected_model = {k: v.detach().clone() for k, v in full.state_dict().items()}
    expected_heads = {k: v.detach().clone() for k, v in full_heads.state_dict().items()}
    expected_optimizer = copy.deepcopy(full_opt.state_dict())
    expected_pair = full_pair.get_state().clone()
    expected_rng = torch.get_rng_state().clone()

    resumed, resumed_heads, resumed_opt, resumed_pair = setup(909)
    resumed.load_state_dict(checkpoint["model"]); resumed_heads.load_state_dict(checkpoint["heads"])
    resumed_opt.load_state_dict(checkpoint["optimizer"])
    resumed_pair.set_state(checkpoint["pair_rng"]); torch.set_rng_state(checkpoint["torch_cpu_rng"])
    # Checkpoint stores the schedule, then next-step set_epoch advances it to
    # the same state as uninterrupted training.
    resumed.beta, resumed.tau = checkpoint["beta"], checkpoint["tau"]
    resumed_loss2 = step(resumed, resumed_heads, resumed_opt, resumed_pair, 2)
    torch.testing.assert_close(torch.tensor(resumed_loss2), torch.tensor(loss2), atol=0, rtol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(expected_pair, resumed_pair.get_state())
    assert expected_optimizer["param_groups"] == resumed_opt.state_dict()["param_groups"]
    for key, expected in expected_model.items():
        torch.testing.assert_close(resumed.state_dict()[key], expected, atol=0, rtol=0)
    for key, expected in expected_heads.items():
        torch.testing.assert_close(resumed_heads.state_dict()[key], expected, atol=0, rtol=0)
    assert full.beta == resumed.beta == 0.1 + 0.9 / 19.0
    assert full.tau == resumed.tau == 2.0 - 1.0 / 19.0
    for key in ("layers.0.private.0.query.weight", "layers.1.common.query.0.weight", "layers.0.common.message_same", "fusion_delta.weight"):
        assert not torch.equal(checkpoint["model"][key], full.state_dict()[key]), key


def test_eval_does_not_advance_pair_rng_and_source_losses_use_train_indices_only():
    x, edge, available = graph()
    model = MAGCollabAdapt(INFO, "M0", init_seed=77).eval()
    heads = CollabTaskHeads("M0", 3)
    pair_rng = torch.Generator().manual_seed(4545)
    before = pair_rng.get_state().clone()
    with torch.no_grad():
        z, _, _, _, extras = model(x, edge, available, pair_generator=pair_rng, compute_ssl=False)
    assert torch.equal(before, pair_rng.get_state())
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 1, 2])
    train = torch.arange(6)
    first, _, _ = train_only_loss("M0", z, extras, heads, labels, train, rec_weight=0, pair_weight=0)
    changed = labels.clone(); changed[6:] = torch.tensor([2, 1])
    second, _, _ = train_only_loss("M0", z, extras, heads, changed, train, rec_weight=0, pair_weight=0)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_cuda_reference_optimized_forward_backward_and_masked_softmax_when_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    x, edge, available = graph()
    x, edge, available = x.to(device), edge.to(device), available.to(device)
    reference = MAGCollabNC(INFO, "M1", init_seed=1001).to(device).eval()
    optimized = MAGCollabAdapt(INFO, "M1", init_seed=2002, optimize_cache=True).to(device).eval()
    optimized.load_state_dict(reference.state_dict())
    zr = reference(x, edge, available)[0]
    zo = optimized(x, edge, available)[0]
    torch.testing.assert_close(zo, zr, atol=1e-5, rtol=1e-4)
    probe = torch.randn_like(zr)
    (zr * probe).sum().backward(); (zo * probe).sum().backward()
    opt_params = dict(optimized.named_parameters())
    for name, parameter in reference.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(opt_params[name].grad, parameter.grad, atol=1e-5, rtol=1e-4, msg=name)

    extreme = torch.tensor([[0.2], [0.9], [1000.0]], device=device, requires_grad=True)
    mass = _grouped_softmax(extreme, torch.tensor([0, 0, 0], device=device), torch.tensor([True, True, False], device=device), 2)
    (mass[0].sum() + 3 * mass[1].sum()).backward()
    assert torch.isfinite(mass).all() and torch.isfinite(extreme.grad).all() and extreme.grad[2].item() == 0.0

    model = MAGCollabAdapt(INFO, "M0", init_seed=1003, typed_value=True, fusion=True, out_dim=768).to(device)
    heads = CollabTaskHeads("M0", 3, output_dim=768).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=0.002)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, -1, -1], device=device)
    train = torch.arange(6, device=device)
    pair_rng = torch.Generator(device=device).manual_seed(46)
    for step_num in (1, 2):
        model.train(); heads.train(); model.set_epoch(step_num); optimizer.zero_grad(set_to_none=True)
        z, _, _, _, extras = model(x, edge, available, pair_generator=pair_rng, compute_ssl=True)
        loss, _, _ = train_only_loss("M0", z, extras, heads, labels, train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        optimizer.step()
    for parameter in (model.layers[1].private[0].gate.net[0].weight, model.layers[0].common.query[0].weight, model.layers[0].common.message_same, model.fusion_delta.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
