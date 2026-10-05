from __future__ import annotations

import base64
import json
import logging
import torch
import zlib
from omegaconf import OmegaConf

from src.analysis.problem_validation import (bucket_metrics, confusion_metrics, correctness_pair,
    geometry_features, geometry_features_l2_neighbor_mean)
from src.analysis.problem_validation import compose_probe_config, fixed_split_path
from src.models.mag_probe import Model, resolve_capacity_widths
from src.tasks.common import build_optimizer
from src.data.types import MAGData
from src.tasks.nc import run_nc
from scripts.evaluate_locked_probe import ATTEMPT_CAP as EVAL_ATTEMPT_CAP
from scripts.evaluate_locked_probe import SEEDS as EVAL_SEEDS, eval_fingerprint
from scripts.run_problem_validation import runtime_job


def make_model(variant: str, text_dim: int = 3, image_dim: int = 2):
    cfg = OmegaConf.create({"seed": 42, "model": {"variant": variant, "out_dim": 128,
        "hidden_dim": 128, "dropout": 0.2}})
    return Model(cfg, {"text_dim": text_dim, "visual_dim": image_dim,
                       "num_classes": 3, "num_nodes": 5})


def test_modalities_use_ordered_slices_and_normalize_rows():
    model = make_model("J_FIXED")
    seen = {}
    model.adapter_t.register_forward_pre_hook(lambda _m, args: seen.setdefault("t", args[0].detach().clone()))
    model.adapter_i.register_forward_pre_hook(lambda _m, args: seen.setdefault("i", args[0].detach().clone()))
    x = torch.tensor([[3., 4., 0., 0., 2.], [0., 0., 0., 0., 0.],
                      [1., 0., 0., 0., 1.], [0., 1., 0., 2., 0.], [1., 1., 1., 1., 1.]])
    model.eval()
    model(x, torch.empty((2, 0), dtype=torch.long))
    assert torch.allclose(seen["t"], torch.nn.functional.normalize(x[:, :3], dim=1, eps=1e-8))
    assert torch.allclose(seen["i"], torch.nn.functional.normalize(x[:, 3:], dim=1, eps=1e-8))
    assert torch.equal(seen["t"][1], torch.zeros(3))
    assert model.adapter_t.bias is not None and model.adapter_i.bias is not None


def test_sparse_receiver_mean_orientation_dedup_selfloops_isolates_and_gradient():
    model = make_model("J_FIXED")
    edge = torch.tensor([[0, 1, 0, 2, 2, 3, 1],
                         [2, 2, 2, 2, 3, 2, 1]])
    p, degree, isolated = model._operator(edge, 5, torch.float32)
    assert p._nnz() == 4
    assert torch.equal(p.indices(), torch.tensor([[2, 2, 2, 3], [0, 1, 3, 2]]))
    assert torch.allclose(p.values(), torch.tensor([1/3, 1/3, 1/3, 1.]))
    assert torch.equal(degree, torch.tensor([0, 0, 3, 1, 0]))
    assert torch.equal(isolated, torch.tensor([True, True, False, False, True]))
    assert not torch.any(p.indices()[0] == p.indices()[1])
    features = torch.arange(15, dtype=torch.float32).reshape(5, 3).requires_grad_()
    actual = torch.sparse.mm(p, features)
    assert torch.allclose(actual[2], features[[0, 1, 3]].mean(0))
    assert torch.allclose(actual[3], features[2])
    assert torch.equal(actual[[0, 1, 4]], torch.zeros((3, 3)))
    actual.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_graph_change_rebuilds_topology_and_cache_is_not_serialized():
    model = make_model("J_FIXED").eval()
    x = torch.randn(5, 5)
    edge = torch.tensor([[0, 1], [2, 2]])
    model(x, edge)
    assert model.operator_rebuilds == 1
    state_keys = model.state_dict().keys()
    assert not any("operator_cache" in key or "sparse" in key for key in state_keys)
    model(x, edge)
    assert model.operator_rebuilds == 1
    edge[0, 0] = 3
    model(x, edge)
    assert model.operator_rebuilds == 2
    model(x, torch.tensor([[0], [4]]))
    assert model.operator_rebuilds == 3


def test_operator_cache_is_conservative_for_inference_tensors_and_works_in_no_grad():
    model = make_model("J_FIXED").eval()
    x = torch.randn(5, 5)
    regular_edge = torch.tensor([[0, 1], [2, 2]])
    with torch.no_grad():
        model(x, regular_edge)
        model(x, regular_edge)
        assert model.operator_rebuilds == 1
        regular_edge[0, 0] = 3
        model(x, regular_edge)
        assert model.operator_rebuilds == 2
    with torch.inference_mode():
        model(x, regular_edge)
        assert model.operator_rebuilds == 3
        model(x, regular_edge)
        assert model.operator_rebuilds == 4
        inference_edge = torch.tensor([[0, 1], [2, 2]])
        assert inference_edge.is_inference()
        first, *_ = model(x, inference_edge)
        assert model.operator_rebuilds == 5
        inference_edge[0, 0] = 3
        second, *_ = model(x, inference_edge)
        assert model.operator_rebuilds == 6
        assert not torch.equal(first, second)


def test_full_graph_forward_repeats_exactly_in_eval_and_active_weights_sum_to_one():
    torch.manual_seed(2)
    model = make_model("J_NODE").eval()
    x = torch.randn(5, 5)
    edge = torch.tensor([[0, 1, 1], [2, 2, 3]])
    z1, *_ = model(x, edge)
    z2, _, _, _, aux = model(x, edge)
    assert z1.shape == (5, 128) and torch.equal(z1, z2)
    weights = aux["router_weights"]
    mask = aux["activity_mask"]
    assert torch.allclose(weights.sum(1), torch.ones(5))
    assert torch.equal(mask[:, 1], aux["degree"] > 0)
    assert torch.equal(mask[:, 3], aux["degree"] > 0)
    assert torch.equal(weights[:, 1][aux["degree"] == 0], torch.zeros(3))
    assert torch.equal(weights[:, 3][aux["degree"] == 0], torch.zeros(3))
    assert torch.equal(aux["activity_mask"][:, 0], torch.ones(5, dtype=torch.bool))


def test_neighbor_only_variant_keeps_isolates_zero():
    model = make_model("J_N").eval()
    x = torch.randn(5, 5)
    edge = torch.tensor([[0, 1], [2, 2]])
    z, *_ = model(x, edge)
    assert torch.equal(z[[0, 1, 3, 4]], torch.zeros((4, 128)))


def test_explicit_context_override_intervenes_for_both_routed_variants_and_clears():
    x = torch.randn(6, 5)
    edge = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 3, 4, 5]])
    permutation = torch.tensor([2, 5, 1, 0, 4, 3])
    expected_mask = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1],
                                  [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    for variant in ("J_NODE", "J_NODE_SHUFFLE"):
        torch.manual_seed(31)
        model = make_model(variant).eval()
        # Model a trained router: its last layer is nonzero, unlike init-time uniform gates.
        torch.nn.init.normal_(model.router[-1].weight, std=0.08)
        torch.nn.init.normal_(model.router[-1].bias, std=0.04)
        seen_context, seen_phi = [], []
        model.router.register_forward_pre_hook(
            lambda _m, args: seen_context.append(args[0].detach().clone()))
        model.phi.register_forward_hook(
            lambda _m, _args, output: seen_phi.append(output.detach().clone()))

        model.set_context_permutation(None)
        z_default, _, _, _, aux_default = model(x, edge)
        default_context = seen_context[-1]

        model.set_context_permutation(torch.arange(x.size(0)))
        z_base, _, _, _, aux_base = model(x, edge)
        base_context = seen_context[-1]
        base_phi = seen_phi[-2:]
        assert torch.equal(aux_base["activity_mask"], expected_mask)

        model.set_context_permutation(permutation)
        z_intervened, _, _, _, aux_intervened = model(x, edge)
        assert torch.equal(seen_context[-1], base_context[permutation])
        assert all(torch.equal(a, b) for a, b in zip(base_phi, seen_phi[-2:]))
        assert torch.equal(aux_intervened["activity_mask"], aux_base["activity_mask"])
        assert torch.equal(aux_intervened["degree"], aux_base["degree"])
        assert not torch.allclose(aux_intervened["router_weights"], aux_base["router_weights"])
        assert not torch.allclose(z_intervened, z_base)

        model.set_context_permutation(None)
        z_restored, _, _, _, aux_restored = model(x, edge)
        assert torch.equal(seen_context[-1], default_context)
        assert torch.equal(z_restored, z_default)
        assert torch.equal(aux_restored["router_weights"], aux_default["router_weights"])


def test_no_override_routed_default_and_state_dict_round_trip_are_unchanged():
    torch.manual_seed(37)
    node = make_model("J_NODE").eval()
    torch.nn.init.normal_(node.router[-1].weight, std=0.05)
    shuffle = make_model("J_NODE_SHUFFLE").eval()
    shuffle.load_state_dict(node.state_dict(), strict=True)
    x = torch.randn(6, 5)
    edge = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 3, 4, 5]])
    captured = {"node": [], "shuffle": []}
    node.router.register_forward_pre_hook(lambda _m, a: captured["node"].append(a[0].detach().clone()))
    shuffle.router.register_forward_pre_hook(lambda _m, a: captured["shuffle"].append(a[0].detach().clone()))
    z_node, *_ = node(x, edge)
    z_shuffle, *_ = shuffle(x, edge)
    p = shuffle._shuffle_permutation(x.size(0))
    assert torch.equal(captured["node"][0][p], captured["shuffle"][0])
    assert not torch.equal(captured["node"][0], captured["shuffle"][0])
    assert z_node.shape == z_shuffle.shape
    assert set(node.state_dict()) == set(shuffle.state_dict())


def test_early_nonlinear_fusion_is_not_late_linear_fusion():
    torch.manual_seed(11)
    early, late = make_model("J_EARLY").eval(), make_model("J_E").eval()
    late.load_state_dict({**early.state_dict(), **{k: v for k, v in late.state_dict().items()
        if k.startswith("adapter_")}}, strict=False)
    x = torch.randn(5, 5)
    at = early.adapter_t(torch.nn.functional.normalize(x[:, :3], dim=1, eps=1e-8))
    ai = early.adapter_i(torch.nn.functional.normalize(x[:, 3:], dim=1, eps=1e-8))
    lhs = early.phi((at + ai) / 2)
    rhs = (early.phi(at) + early.phi(ai)) / 2
    assert not torch.allclose(lhs, rhs, atol=1e-7, rtol=1e-6)
    assert lhs.shape == rhs.shape == (5, 128)


def test_capacity_controls_match_complete_node_model_within_one_percent():
    widths = resolve_capacity_widths(num_classes=12, text_dim=512, image_dim=512)
    assert widths["wide_hidden_dim"] == 192
    assert widths["concat_bottleneck"] == 26
    for key in ("wide_relative_error", "concat_relative_error"):
        assert widths[key] <= 0.01
    node = make_model("J_NODE")
    wide = make_model("J_GLOBAL_WIDE")
    concat = make_model("J_CONCAT")
    head = lambda: torch.nn.Linear(128, 3)
    totals = [sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in head().parameters())
              for m in (node, wide, concat)]
    assert abs(totals[1] - totals[0]) / totals[0] <= 0.01
    assert abs(totals[2] - totals[0]) / totals[0] <= 0.01
    assert sum(p.numel() for p in node.router.parameters()) == 16676


def test_full_confusion_schema_counts_prediction_to_extra_valid_class():
    metrics = confusion_metrics(torch.tensor([0, 0]), torch.tensor([0, 11]), 12, [0])
    assert metrics["confusion_matrix"][0][11] == 1
    assert metrics["class_order"] == list(range(12))
    assert abs(metrics["macro_f1"] - (2 / 3)) < 1e-12


def test_training_seed_never_changes_fixed_split_and_test_is_disabled():
    configs = [compose_probe_config("Movies", seed, fixed_split_path("Movies"),
        3e-4, "J_FIXED", "cuda:0") for seed in (42, 43, 44)]
    assert len({cfg.dataset.nc_split_path for cfg in configs}) == 1
    assert configs[0].dataset.nc_split_path.endswith("Movies_nc_seed42_train0.6_val0.2.pt")
    for seed, cfg in zip((42, 43, 44), configs):
        assert cfg.dataset.auto_generate_splits is False
        assert cfg.task.evaluate_test is False
        assert cfg.task.scheduler is None
        assert cfg.num_runs == 1 and cfg.seed == seed
        assert cfg.task.lr == 3e-4


def test_composed_probe_uses_task_optimizer_defaults_and_model_overrides():
    cfg = compose_probe_config("Movies", 42, fixed_split_path("Movies"), 0.003, "J_NODE", "cpu")
    assert "lr" not in cfg.model and "weight_decay" not in cfg.model
    model = Model(cfg, {"text_dim": 3, "visual_dim": 2, "num_classes": 3, "num_nodes": 5})
    optimizer = build_optimizer(model.parameters(), cfg, model=model)
    assert optimizer.param_groups[0]["lr"] == 0.003
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    cfg.model.lr = 0.007
    cfg.model.weight_decay = 0.02
    override_optimizer = build_optimizer(model.parameters(), cfg, model=model)
    assert override_optimizer.param_groups[0]["lr"] == 0.007
    assert override_optimizer.param_groups[0]["weight_decay"] == 0.02


def test_normalized_neighbor_geometry_differs_from_raw_mean_for_unequal_norms():
    edge = torch.tensor([[1, 2], [0, 0]])
    text = torch.tensor([[0., 1.], [100., 0.], [0., 1.], [1., 1.]])
    image = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [1., -1.]])
    train, val = torch.tensor([0, 1, 2]), torch.tensor([3])
    raw = geometry_features(edge, text, image, train, val)
    normalized = geometry_features_l2_neighbor_mean(edge, text, image, train, val)
    assert abs(float(raw["text"]["cosine"][0]) - 0.01) < 1e-3
    assert abs(float(normalized["text"]["cosine"][0]) - 2 ** -0.5) < 1e-6
    assert normalized["text"]["cosine_thresholds"] != raw["text"]["cosine_thresholds"]
    assert torch.equal(normalized["text"]["isolated"], torch.tensor([False, True, True, True]))


def test_generic_nc_synthetic_integration_uses_external_head_seed_and_test_off(tmp_path, monkeypatch):
    cfg = compose_probe_config("Movies", 17, fixed_split_path("Movies"), 0.003, "J_NODE", "cpu")
    cfg.model.hidden_dim = 8
    cfg.model.dropout = 0.0
    cfg.task.epochs = 2
    cfg.task.patience = 2
    cfg.task.early_stop_min_epoch = 1
    cfg.task.early_stop_min_delta = 0.0
    cfg.task.eval_every = 1
    cfg.task.grad_clip = 1.0
    cfg.task.inference_batch_size = 32
    cfg.task.evaluate_test = False
    cfg.task.save_ckpt_path = str(tmp_path / "synthetic_nc.pt")

    x_t = torch.randn(8, 3)
    x_i = torch.randn(8, 2)
    data = MAGData(name="synthetic", source="synthetic", task="nc", x=torch.cat((x_t, x_i), 1),
        x_t=x_t, x_i=x_i, edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]),
        num_nodes=8, y=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        train_idx=torch.tensor([0, 1, 2, 3]), val_idx=torch.tensor([4, 5]),
        test_idx=torch.tensor([6, 7]), num_classes=3)
    epochs = []
    original_set_epoch = Model.set_epoch
    def tracked_set_epoch(self, epoch):
        epochs.append(int(epoch))
        original_set_epoch(self, epoch)
    monkeypatch.setattr(Model, "set_epoch", tracked_set_epoch)
    logger = logging.getLogger("test-generic-nc-smoke")
    logger.disabled = True

    first = run_nc(cfg, data, torch.device("cpu"), logger)
    checkpoint1 = torch.load(cfg.task.save_ckpt_path, map_location="cpu", weights_only=False)
    second = run_nc(cfg, data, torch.device("cpu"), logger)
    checkpoint2 = torch.load(cfg.task.save_ckpt_path, map_location="cpu", weights_only=False)
    assert epochs == [1, 2, 1, 2]
    assert set(first) == {"val_acc", "val_macro_f1"}
    assert all(torch.isfinite(torch.tensor(pair)).all() for pair in first.values())
    assert first == second
    assert set(checkpoint1["head_state"]) == {"weight", "bias"}
    assert all(torch.equal(checkpoint1["head_state"][k], checkpoint2["head_state"][k])
               for k in checkpoint1["head_state"])
    assert "test_acc" not in checkpoint2["metrics"] and "test_macro_f1" not in checkpoint2["metrics"]


def test_single_modality_model_only_instantiates_one_adapter():
    text = make_model("T_E")
    image = make_model("I_E")
    assert text.adapter_t is not None and text.adapter_i is None
    assert image.adapter_t is None and image.adapter_i is not None
    assert all(parameter.requires_grad for parameter in text.parameters())
    assert all(parameter.requires_grad for parameter in image.parameters())


def test_shuffle_generator_is_seed_epoch_namespace_deterministic_and_dropout_independent():
    model = make_model("J_NODE_SHUFFLE").train()
    model.set_run_seed(43)
    torch.manual_seed(10)
    model.set_epoch(4)
    first = model._shuffle_permutation(32)
    torch.manual_seed(999)
    second = model._shuffle_permutation(32)
    assert torch.equal(first, second)
    model.set_epoch(5)
    assert not torch.equal(first, model._shuffle_permutation(32))
    model.eval(); model.set_epoch(101)
    val1 = model._shuffle_permutation(32)
    model.set_epoch(202)
    val2 = model._shuffle_permutation(32)
    assert torch.equal(val1, val2)


def test_json_aggregate_primitives_recompute_pair_counts():
    y = torch.tensor([0, 0, 1, 1, 2])
    a = torch.tensor([0, 1, 1, 0, 2])
    b = torch.tensor([0, 0, 0, 1, 2])
    pair = correctness_pair(y, a, y, b)
    assert pair["both_correct"] == 2
    assert pair["a_only_correct"] == 1
    assert pair["b_only_correct"] == 2
    assert pair["both_wrong"] == 0
    assert abs(pair["accuracy_difference_a_minus_b"] + 0.2) < 1e-7


def test_bucket_pair_metrics_map_split_order_predictions_to_node_ids():
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    indices = torch.tensor([4, 1, 5])
    bucket = torch.tensor([0, 1, 0, 0, 1, 1])
    # Split order [node 4, node 1, node 5].
    pred_a = torch.tensor([0, 1, 1])
    pred_b = torch.tensor([1, 0, 1])
    result = bucket_metrics(y, pred_a, pred_b, 2, [0, 1], indices, bucket,
                            ["q0", "q1"], "training-only")
    q1 = next(row for row in result if row["bucket"] == "q1")
    assert q1["n"] == 3
    assert q1["a_correct"] == 3
    assert q1["b_correct"] == 1
    assert q1["paired_improvement"] == 2
    assert q1["paired_regression"] == 0
    encoded = q1["a_confusion_matrix"]
    assert encoded["shape"] == [2, 2] and encoded["class_order"] == [0, 1]
    decoded = torch.tensor(json.loads(zlib.decompress(base64.b64decode(encoded["data"]))))
    assert int(decoded.sum()) == q1["n"]


def test_reused_calibration_training_identity_and_locked_test_fingerprint():
    assert EVAL_SEEDS == (42, 43, 44)
    assert EVAL_ATTEMPT_CAP == 180
    common = {"dataset": "Movies", "variant": "J_FIXED", "seed": 42,
        "split_path": str(fixed_split_path("Movies")), "split_sha256": "split-hash",
        "lr": 1e-3, "code_commit": "source-commit", "source_hashes": {"model": "abc"}}
    calibration = runtime_job({**common, "job_id": "cal-job", "stage": "calibration"}, "cuda:1")
    primary = runtime_job({**common, "job_id": "P1-job", "stage": "main", "phase": "P1"}, "cuda:0")
    assert calibration["config_hash"] == primary["config_hash"]
    assert calibration["fingerprint"] == primary["fingerprint"]
    lock = {"protocol_lock_sha256": "lock"}
    slot = {"job_id": "P1-job", "split_sha256": "split-hash", "config_hash": primary["config_hash"]}
    a = eval_fingerprint(lock, slot, "checkpoint-a")
    b = eval_fingerprint(lock, slot, "checkpoint-b")
    assert a != b
    assert a == eval_fingerprint(lock, slot, "checkpoint-a")
