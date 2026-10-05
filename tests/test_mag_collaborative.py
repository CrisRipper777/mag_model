from __future__ import annotations

import logging

import torch
from omegaconf import OmegaConf

from src.data.types import MAGData
from src.models.mag_collaborative import Model, _Interaction
from src.tasks.common import build_optimizer
from src.tasks.nc import run_nc
from src.utils.seeds import set_seed


def _model(variant: str = "P3", dropout: float = 0.0) -> Model:
    cfg = OmegaConf.create({"model": {"variant": variant, "out_dim": 128, "dropout": dropout}})
    return Model(cfg, {"input_dim": 7, "text_dim": 4, "visual_dim": 3, "num_classes": 3})


def test_biaxial_softmax_groups_sender_modality_and_receiver_incoming_edges():
    torch.manual_seed(101)
    interaction = _Interaction(dim=8, type_dim=3)
    states = (torch.randn(5, 8), torch.randn(5, 8))
    # Two senders arrive at node 2; two more arrive at node 3. Edges are directed.
    edges = torch.tensor([[0, 1, 0, 2, 2], [2, 2, 3, 3, 0]])
    available = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.bool)
    aggregate, stats = interaction(states, edges, available, torch.randn(2, 3))
    selector = stats["modality_weights"]
    valid = stats["neighbor_valid"]
    sender_valid = stats["sender_valid"]
    selector_sums = selector.sum(dim=-1)
    assert torch.allclose(selector_sums[sender_valid], torch.ones_like(selector_sums[sender_valid]))
    assert torch.equal(selector_sums[~sender_valid], torch.zeros_like(selector_sums[~sender_valid]))
    attention = stats["neighbor_weights"]
    for modality in range(2):
        for receiver in (0, 2, 3):
            edge_ids = (edges[1] == receiver) & valid[:, modality]
            if edge_ids.any():
                assert torch.allclose(attention[edge_ids, modality].sum(), torch.tensor(1.0))
    assert all(torch.isfinite(value).all() for value in aggregate)


def test_edge_modality_change_changes_relationship_selection_and_output():
    torch.manual_seed(117)
    interaction = _Interaction(dim=8, type_dim=3).eval()
    text, image = torch.randn(5, 8), torch.randn(5, 8)
    edges = torch.tensor([[0, 1, 0, 2], [2, 2, 3, 3]])
    available = torch.ones((5, 2), dtype=torch.bool)
    types = torch.randn(2, 3)
    baseline, stats_a = interaction((text, image), edges, available, types)
    changed_text = text.clone()
    changed_text[0] += torch.tensor([2., -1., 1., 0., 0.5, -0.5, 1.5, -2.])
    changed, stats_b = interaction((changed_text, image), edges, available, types)
    assert not torch.allclose(stats_a["modality_weights"][0], stats_b["modality_weights"][0])
    assert any(not torch.allclose(a, b) for a, b in zip(baseline, changed))


def test_graph_cache_invalidates_normal_mutations_and_does_not_reuse_inference_tensors():
    model = _model("P1").eval()
    normal_edge = torch.tensor([[0, 1], [2, 2]])
    with torch.no_grad():
        first = model._validated_edges(normal_edge, 4)
        assert model._validated_edges(normal_edge, 4) is first
        cached_version = model._edge_cache_signature[2]
        normal_edge[0, 0] = 3
        changed = model._validated_edges(normal_edge, 4)
        assert not torch.equal(first, changed)
        assert model._edge_cache_signature[2] != cached_version
    with torch.inference_mode():
        inference_edge = torch.tensor([[0, 1], [2, 2]])
        assert inference_edge.is_inference()
        first_inference = model._validated_edges(inference_edge, 4)
        assert model._edge_cache_reference is None
        inference_edge[0, 0] = 3
        second_inference = model._validated_edges(inference_edge, 4)
        assert not torch.equal(first_inference, second_inference)
        assert model._edge_cache_reference is None


def test_empty_neighborhood_and_all_unavailable_senders_are_finite_zero():
    torch.manual_seed(3)
    interaction = _Interaction(dim=8, type_dim=3)
    states = (torch.randn(4, 8), torch.randn(4, 8))
    no_edges = torch.empty((2, 0), dtype=torch.long)
    available = torch.zeros((4, 2), dtype=torch.bool)
    output, stats = interaction(states, no_edges, available, torch.randn(2, 3))
    assert all(torch.equal(value, torch.zeros_like(value)) for value in output)
    assert all(torch.isfinite(value).all() for value in output)
    assert stats["modality_weights"].shape == (0, 2, 2)

    edges = torch.tensor([[0, 1], [2, 2]])
    output, stats = interaction(states, edges, available, torch.randn(2, 3))
    assert all(torch.equal(value, torch.zeros_like(value)) for value in output)
    assert all(torch.isfinite(value).all() for value in output)
    assert torch.equal(stats["modality_weights"], torch.zeros_like(stats["modality_weights"]))

    model = _model("P3").eval()
    x = torch.randn(4, 7)
    x[1, 4:] = torch.nan
    x[2, :4] = torch.nan
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.bool)
    z = model(x, torch.tensor([[0, 1, 2], [1, 2, 3]]), mask)[0]
    assert torch.isfinite(z).all()


def test_same_modality_source_state_is_independent_of_other_modality_input():
    torch.manual_seed(10)
    model = _model("P3").eval()
    x = torch.randn(6, 7)
    edge = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 3, 4, 5]])
    first_z, _, _, _, first = model(x, edge)
    repeated_z = model(x, edge)[0]
    assert torch.equal(first_z, repeated_z)
    changed = x.clone()
    changed[:, 4:] += 4.0 * torch.randn_like(changed[:, 4:])
    _, _, _, _, second = model(changed, edge)
    assert torch.equal(first["source_states"]["text"], second["source_states"]["text"])
    assert not torch.equal(first["source_states"]["image"], second["source_states"]["image"])


def test_first_layer_same_modality_mean_is_uniform_and_excludes_unavailable_sources():
    state = torch.tensor([[1., 0.], [0., 3.], [9., 9.], [4., 4.]])
    edges = torch.tensor([[0, 1, 1, 2, 3, 3], [2, 2, 2, 3, 3, 3]])
    model = _model("P1")
    canonical = model._validated_edges(edges, 4)
    available = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    aggregate = model._same_modality_mean(state, canonical, available)
    assert torch.equal(aggregate[2], state[0])
    assert torch.equal(aggregate[3], state[2])
    assert torch.equal(aggregate[0], torch.zeros(2))
    assert torch.equal(aggregate[1], torch.zeros(2))


def test_every_enabled_p3_module_backpropagates_updates_and_checkpoint_restores(tmp_path):
    set_seed(29)
    model = _model("P3", dropout=0.0)
    classifier = torch.nn.Linear(128, 3)
    source_heads = torch.nn.ModuleDict({"text": torch.nn.Linear(128, 3), "image": torch.nn.Linear(128, 3)})
    cfg = OmegaConf.create({"model": {}, "task": {"optimizer": "adamw", "lr": 0.003, "weight_decay": 1e-4}})
    optimizer = build_optimizer(
        list(model.parameters()) + list(classifier.parameters()) + list(source_heads.parameters()), cfg, model
    )
    x = torch.randn(7, 7)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 2], [2, 2, 3, 4, 5, 6, 3, 4]])
    y = torch.tensor([0, 1, 2, 0, 1, 2, 1])
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for _ in range(2):
        model.train()
        model.set_epoch(_ + 1)
        optimizer.zero_grad(set_to_none=True)
        z, _, _, _, extras = model(x, edges)
        main_loss = torch.nn.functional.cross_entropy(classifier(z[:5]), y[:5])
        source_loss = sum(torch.nn.functional.cross_entropy(source_heads[name](extras["source_states"][name][:5]), y[:5])
                          for name in ("text", "image")) / 2
        (main_loss + 0.1 * source_loss).backward()
        assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        optimizer.step()
    changed_groups = {
        "input": any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(("input_text", "input_image"))),
        "layer1 experts": any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith("layers.0.")),
        "layer2 experts": any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith("layers.1.")),
        "interaction": any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith("interaction.")),
        "routers": any(not torch.equal(before[n], p) for n, p in model.named_parameters() if ".routers." in n),
        "fusion": any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith("fusion.")),
    }
    assert all(changed_groups.values()), changed_groups
    assert torch.linalg.vector_norm(next(model.layers[0].routers[0][0].parameters()).grad) > 0
    for module in (model.interaction.context, model.interaction.modality_score,
                   model.interaction.value_projection, model.interaction.query, model.interaction.key):
        assert all(parameter.grad is not None and torch.linalg.vector_norm(parameter.grad) > 0
                   for parameter in module.parameters())
        prefixes = tuple(name for name, child in model.named_modules() if child is module)
        prefix = prefixes[0] + "."
        assert any(not torch.equal(before[name], parameter)
                   for name, parameter in model.named_parameters() if name.startswith(prefix))
    for router in (*model.layers[0].routers, *model.layers[1].routers):
        assert any(parameter.grad is not None and torch.linalg.vector_norm(parameter.grad) > 0
                   for parameter in router.parameters())
        prefix = next(name for name, child in model.named_modules() if child is router) + "."
        assert any(not torch.equal(before[name], parameter)
                   for name, parameter in model.named_parameters() if name.startswith(prefix))

    model.eval()
    with torch.no_grad():
        expected = model(x, edges)[0][5:]
    payload = {
        "model_state": model.state_dict(),
        "head_state": classifier.state_dict(),
        "source_head_state": source_heads.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {"variant": "P3", "source_aux_weight": 0.1},
    }
    path = tmp_path / "full.pt"
    torch.save(payload, path)
    restored = _model("P3", dropout=0.0)
    restored_head = torch.nn.Linear(128, 3)
    restored_source = torch.nn.ModuleDict({"text": torch.nn.Linear(128, 3), "image": torch.nn.Linear(128, 3)})
    restored_optimizer = build_optimizer(
        list(restored.parameters()) + list(restored_head.parameters()) + list(restored_source.parameters()), cfg, restored
    )
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_state_dict(loaded["model_state"])
    restored_head.load_state_dict(loaded["head_state"])
    restored_source.load_state_dict(loaded["source_head_state"])
    restored_optimizer.load_state_dict(loaded["optimizer_state"])
    restored.eval()
    with torch.no_grad():
        actual = restored(x, edges)[0][5:]
    assert torch.equal(expected, actual)


def test_wide_capacity_matches_active_four_experts_and_static_retraining_has_no_router():
    wide = _model("P0")
    experts = _model("P1")
    static = _model("P4")
    for layer_idx in range(2):
        wide_count = sum(p.numel() for p in wide.layers[layer_idx].parameters())
        expert_count = sum(p.numel() for p in experts.layers[layer_idx].shared.parameters())
        expert_count += sum(p.numel() for p in experts.layers[layer_idx].specialists.parameters())
        assert abs(wide_count - expert_count) / expert_count <= 0.05
    assert not any("router" in name for name, _ in static.named_parameters())
    assert len(static.layers[0].specialists) == 3
    torch.manual_seed(118)
    static.train()
    classifier = torch.nn.Linear(128, 3)
    optimizer = torch.optim.AdamW(list(static.parameters()) + list(classifier.parameters()), lr=0.003)
    x = torch.randn(6, 7)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    before = [parameter.detach().clone() for parameter in static.layers[0].specialists.parameters()]
    optimizer.zero_grad(set_to_none=True)
    z, _, _, _, extras = static(x, edges)
    route = extras["diagnostics"]["layers"][0]["modality_0"]["router_mean"]
    assert torch.allclose(route.detach().cpu(), torch.full((3,), 1 / 3))
    torch.nn.functional.cross_entropy(classifier(z[:4]), torch.tensor([0, 1, 2, 1])).backward()
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, static.layers[0].specialists.parameters()))


def test_p0_p2_p6_train_the_wide_modules_and_share_the_same_fusion_shape():
    torch.manual_seed(319)
    fusion_shapes = []
    for variant in ("P0", "P2", "P6"):
        model = _model(variant, dropout=0.0)
        assert model.expert_kind == "wide"
        fusion_shapes.append({name: tuple(value.shape) for name, value in model.fusion.state_dict().items()})
        classifier = torch.nn.Linear(128, 3)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=0.003)
        before = [parameter.detach().clone() for parameter in model.layers[0].wide.parameters()]
        optimizer.zero_grad(set_to_none=True)
        z = model(torch.randn(6, 7), torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))[0]
        torch.nn.functional.cross_entropy(classifier(z[:4]), torch.tensor([0, 1, 2, 1])).backward()
        assert all(parameter.grad is not None for parameter in model.layers[0].wide.parameters())
        optimizer.step()
        assert any(not torch.equal(old, new) for old, new in zip(before, model.layers[0].wide.parameters()))
    assert fusion_shapes[0] == fusion_shapes[1] == fusion_shapes[2]


def test_seed_controls_parameter_initialization_without_changing_the_split():
    set_seed(321)
    first = _model("P3")
    set_seed(321)
    repeated = _model("P3")
    set_seed(322)
    changed = _model("P3")
    assert all(torch.equal(first.state_dict()[key], repeated.state_dict()[key]) for key in first.state_dict())
    assert any(not torch.equal(first.state_dict()[key], changed.state_dict()[key]) for key in first.state_dict())


def test_composed_pilot_config_uses_locked_split_task_optimizer_and_test_off():
    from scripts.run_mechanism_pilot_v2 import compose_config

    configs = [compose_config("Movies", "P6", seed, 0.003, "cpu") for seed in (42, 43, 44)]
    assert len({cfg.dataset.nc_split_path for cfg in configs}) == 1
    for cfg in configs:
        assert cfg.dataset.nc_split_path.endswith("Movies_nc_seed42_train0.6_val0.2.pt")
        assert cfg.dataset.auto_generate_splits is False
        assert cfg.task.evaluate_test is False
        assert cfg.task.lr == 0.003 and cfg.task.weight_decay == 1e-4
        assert cfg.task.source_aux_weight == 0.1
        assert cfg.task.eval_labels == list(range(20))
    cfg = configs[0]
    model = _model("P3")
    optimizer = build_optimizer(model.parameters(), cfg, model=model)
    assert optimizer.param_groups[0]["lr"] == 0.003
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_generic_nc_optional_source_heads_respect_explicit_labels_and_test_off(tmp_path):
    set_seed(17)
    cfg = OmegaConf.create({
        "seed": 17, "num_runs": 1,
        "model": {"name": "mag_collaborative", "variant": "P5", "out_dim": 128, "dropout": 0.0},
        "task": {
            "protocol_version": "pilot_v2_smoke", "training_mode": "full_graph",
            "optimizer": "adamw", "lr": 0.003, "weight_decay": 1e-4, "epochs": 2,
            "patience": 2, "early_stop_min_epoch": 1, "early_stop_min_delta": 0.0,
            "grad_clip": 1.0, "inference_batch_size": 32, "inference_mode": "full", "eval_every": 1,
            "evaluate_test": False, "eval_labels": [0, 1], "source_aux_weight": 0.1,
            "loss": {"aux_weight": 0.0}, "save_ckpt_path": str(tmp_path / "ckpt.pt"),
        },
    })
    data = MAGData(
        name="synthetic", source="test", task="nc", x=torch.randn(8, 7),
        x_t=torch.randn(8, 4), x_i=torch.randn(8, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]),
        y=torch.tensor([0, 1, 0, 1, 0, 1, -999, -999]), num_nodes=8,
        num_classes=2, train_idx=torch.tensor([0, 1, 2, 3]), val_idx=torch.tensor([4, 5]),
        test_idx=torch.tensor([6, 7]),
    )
    logger = logging.getLogger("mag-collaborative-smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    result = run_nc(cfg, data, torch.device("cpu"), logger)
    assert "test_acc" not in result and "test_macro_f1" not in result
    saved = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    assert set(saved["source_head_state"]) == {"text.weight", "text.bias", "image.weight", "image.bias"}
    assert saved["run_metadata"]["source_aux_weight"] == 0.1


def test_pilot_training_cell_smoke_saves_optimizer_and_train_validation_only_evidence(tmp_path, monkeypatch):
    import scripts.run_mechanism_pilot_v2 as pilot

    original_compose = pilot.compose_config

    def tiny_config(*args, **kwargs):
        cfg = original_compose(*args, **kwargs)
        cfg.task.epochs = 2
        cfg.task.patience = 1
        cfg.task.early_stop_min_epoch = 1
        return cfg

    monkeypatch.setattr(pilot, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pilot, "compose_config", tiny_config)
    original_optimizer = pilot.build_optimizer
    optimizer_instances = []

    def tracked_optimizer(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        optimizer_instances.append(optimizer)
        return optimizer

    monkeypatch.setattr(pilot, "build_optimizer", tracked_optimizer)
    epoch_calls = []
    original_set_epoch = Model.set_epoch

    def tracked_set_epoch(model, epoch):
        epoch_calls.append(int(epoch))
        original_set_epoch(model, epoch)

    monkeypatch.setattr(Model, "set_epoch", tracked_set_epoch)
    set_seed(202)
    data = MAGData(
        name="Movies", source="synthetic", task="nc", x=torch.randn(9, 7),
        x_t=torch.randn(9, 4), x_i=torch.randn(9, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]]),
        y=torch.tensor([0, 1, 2, 0, 1, 2, 1, 0, -999]), num_nodes=9,
        num_classes=20, train_idx=torch.tensor([0, 1, 2, 3, 4, 5]),
        val_idx=torch.tensor([6, 7]), test_idx=torch.tensor([8]), info={},
    )
    summary = pilot.run_one(
        "Movies", "P5", 42, 0.001, "micro", "cpu", data,
        {"synthetic_input": {"bytes": 1, "sha256": "synthetic"}},
    )
    assert summary["status"] == "succeeded"
    assert len(optimizer_instances) == 1
    assert epoch_calls == [1, 2]
    assert summary["test_inference"] is False
    assert summary["train"]["n"] == 6 and summary["validation"]["n"] == 2
    checkpoint = torch.load(summary["checkpoint_path"], map_location="cpu", weights_only=False)
    assert checkpoint["optimizer_state"]
    assert checkpoint["source_head_state"]
    predictions = torch.load(summary["prediction_path"], map_location="cpu", weights_only=False)
    assert set(predictions) == {"train_idx", "train_argmax", "val_idx", "val_argmax", "split_sha256"}
    assert predictions["train_argmax"].numel() == 6
    assert predictions["val_argmax"].numel() == 2
