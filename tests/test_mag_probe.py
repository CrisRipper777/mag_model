from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.analysis.problem_validation import bucket_metrics, confusion_metrics, correctness_pair
from src.analysis.problem_validation import compose_probe_config, fixed_split_path
from src.models.mag_probe import Model, resolve_capacity_widths
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


def test_shuffle_permutation_changes_context_rows_not_receiver_paths_or_mask():
    torch.manual_seed(3)
    model = make_model("J_NODE_SHUFFLE").eval()
    x = torch.randn(5, 5)
    edge = torch.tensor([[0, 1, 2, 3], [2, 2, 3, 4]])
    captured = []
    model.router.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
    model.set_context_permutation(torch.arange(5))
    z_identity, *_ = model(x, edge)
    model.set_context_permutation(torch.tensor([2, 4, 1, 0, 3]))
    z_shuffled, _, _, _, aux = model(x, edge)
    assert len(captured) == 2
    assert not torch.equal(captured[0], captured[1])
    # The first 512 context columns are shuffled copies of complete rows.
    assert torch.equal(torch.sort(captured[0][:, :512], dim=0).values,
                       torch.sort(captured[1][:, :512], dim=0).values)
    # Zero output initialization yields the same uniform fusion of receiver-owned branches.
    assert torch.equal(z_identity, z_shuffled)
    assert torch.equal(aux["activity_mask"], torch.tensor(
        [[1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool))


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


def test_reused_calibration_training_identity_and_locked_test_fingerprint():
    assert EVAL_SEEDS == (42, 43, 44)
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
