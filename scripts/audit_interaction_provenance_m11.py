from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models.interaction_provenance_m1 import Model


DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
SHUFFLE_SEEDS = (3407, 3408, 3409)
CHECKPOINT_RELATIVE_PATHS = tuple(
    Path("outputs/model_design/m1/pilot") / dataset / f"seed{seed}" / "best_checkpoint.pt"
    for dataset in DATASETS
    for seed in SEEDS
)
TOLERANCE = 2e-6
EPSILON = 1e-12


def make_cfg(dataset: str, seed: int, checkpoint: Path):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=interaction_provenance_m1",
        f"seed={int(seed)}",
        "num_runs=1",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint.resolve()}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module: torch.nn.Module) -> str:
    return tensor_state_sha256(module.state_dict())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: torch.Tensor, quantiles=(0.5, 0.9)) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": 0.0, **{f"p{int(q * 100)}": 0.0 for q in quantiles}}
    q_values = torch.quantile(values, values.new_tensor(quantiles))
    return {
        "mean": float(values.mean().item()),
        **{
            f"p{int(q * 100)}": float(qv.item())
            for q, qv in zip(quantiles, q_values, strict=True)
        },
    }


def finite_tree(value: Any, path="root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not torch.isfinite(
        torch.tensor(float(value))
    ):
        raise AssertionError(f"Non-finite value at {path}: {value}")


def assert_semantic_invariance(
    reference: dict[str, Any], candidate: dict[str, Any], tolerance=TOLERANCE
) -> float:
    maximum = 0.0
    for modality in ("text", "visual"):
        for order, (left, right) in enumerate(
            zip(reference[f"S_{modality}"], candidate[f"S_{modality}"], strict=True)
        ):
            error = float((left - right).abs().max().item()) if left.numel() else 0.0
            maximum = max(maximum, error)
            if error > tolerance:
                raise AssertionError(
                    f"Semantic state changed for {modality} S{order}: {error} > {tolerance}"
                )
    return maximum


def interaction_atom_components(
    model: Model,
    relation: torch.Tensor,
    effect: torch.Tensor,
    modality: str,
) -> dict[str, torch.Tensor]:
    """Reconstruct r, e, x and the four audit atom definitions."""
    if modality not in {"text", "visual"}:
        raise ValueError(f"Unsupported modality: {modality}")
    relation_layer = getattr(model, f"{modality}_relation_to_provenance")
    relation_norm = getattr(model, f"{modality}_relation_provenance_norm")
    effect_layer = getattr(model, f"{modality}_effect_to_provenance")
    effect_norm = getattr(model, f"{modality}_effect_provenance_norm")
    cross_layer = getattr(model, f"{modality}_interaction_cross")
    atom_norm = getattr(model, f"{modality}_atom_norm")
    r = relation_norm(relation_layer(relation))
    e = effect_norm(effect_layer(effect))
    x = cross_layer(r * e)
    full = torch.tanh(atom_norm(r + e + x))
    relation_off = torch.tanh(atom_norm(e))
    effect_off = torch.tanh(atom_norm(r))
    cross_off = torch.tanh(atom_norm(r + e))
    # This call is the frozen model's own full-atom implementation. Comparing
    # it chunkwise makes the explicit reconstruction auditable without changing M1.
    model_full = model._interaction_atom(relation, effect, modality, "full")
    error = float((full - model_full).abs().max().item()) if full.numel() else 0.0
    if error > TOLERANCE:
        raise AssertionError(f"Full atom reconstruction error {error} > {TOLERANCE}")
    return {
        "r": r,
        "e": e,
        "x": x,
        "full": full,
        "relation_off_corrected": relation_off,
        "effect_off_corrected": effect_off,
        "cross_off": cross_off,
        "full_reconstruction_max_abs_error": full.new_tensor(error),
    }


def build_source_path_messages(
    previous: torch.Tensor,
    atom: torch.Tensor,
    source: torch.Tensor,
    source_history_mode: str = "full",
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply source-side history only; target combine state is handled separately."""
    if source_history_mode == "full":
        source_history = previous
    elif source_history_mode == "off":
        source_history = torch.ones_like(previous)
    elif source_history_mode == "shuffle":
        if permutation is None:
            raise ValueError("A permutation is required for source-history shuffle")
        source_history = previous[permutation]
    else:
        raise ValueError(f"Unsupported source history mode: {source_history_mode}")
    return source_history[source] * atom


def build_target_combine_input(
    previous: torch.Tensor, aggregate: torch.Tensor, target_memory_mode: str = "full"
) -> torch.Tensor:
    """Apply target-side memory only; source message construction is independent."""
    if target_memory_mode == "full":
        target_memory = previous
    elif target_memory_mode == "off":
        target_memory = torch.ones_like(previous)
    else:
        raise ValueError(f"Unsupported target memory mode: {target_memory_mode}")
    return torch.cat((target_memory, aggregate), dim=-1)


def realized_semantic_effect(
    model: Model,
    semantic_cache: dict[str, Any],
    modality: str,
    order: int,
    start: int,
    end: int,
) -> torch.Tensor:
    source = semantic_cache["source"][start:end]
    target = semantic_cache["target"][start:end]
    state = semantic_cache[f"S_{modality}"][order - 1]
    source_state = state[source]
    angles = semantic_cache[f"angles_{modality}"]
    if angles is not None:
        source_state = model._givens_transport(source_state, angles[start:end])
    discrepancy = source_state - state[target]
    conductance = semantic_cache[f"conductance_{modality}"][start:end]
    response = getattr(model, f"{modality}_response")
    return conductance[:, None] * response(discrepancy)


def realized_semantic_effect_at_indices(
    model: Model,
    semantic_cache: dict[str, Any],
    modality: str,
    order: int,
    edge_indices: torch.Tensor,
) -> torch.Tensor:
    source = semantic_cache["source"][edge_indices]
    target = semantic_cache["target"][edge_indices]
    state = semantic_cache[f"S_{modality}"][order - 1]
    source_state = state[source]
    angles = semantic_cache[f"angles_{modality}"]
    if angles is not None:
        source_state = model._givens_transport(source_state, angles[edge_indices])
    discrepancy = source_state - state[target]
    conductance = semantic_cache[f"conductance_{modality}"][edge_indices]
    return conductance[:, None] * getattr(model, f"{modality}_response")(discrepancy)


def provenance_step(
    model: Model,
    semantic_cache: dict[str, Any],
    previous: torch.Tensor,
    modality: str,
    order: int,
    *,
    atom_mode: str = "full",
    source_history_mode: str = "full",
    target_memory_mode: str = "full",
    permutation: torch.Tensor | None = None,
    sample_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """Analysis-only exact PNA + recurrent combine for one frozen provenance step."""
    source = semantic_cache["source"]
    target = semantic_cache["target"]
    relation = semantic_cache[f"R_{modality}"]
    degree = semantic_cache["degree"]
    scalers = semantic_cache["scalers"]
    chunk_size = int(model.edge_chunk_size)
    sampled_atoms: list[torch.Tensor] = []
    max_atom_error = 0.0

    def message_chunks():
        nonlocal max_atom_error
        for start in range(0, int(source.numel()), chunk_size):
            end = min(start + chunk_size, int(source.numel()))
            effect = realized_semantic_effect(model, semantic_cache, modality, order, start, end)
            components = interaction_atom_components(
                model, relation[start:end], effect, modality
            )
            max_atom_error = max(
                max_atom_error,
                float(components["full_reconstruction_max_abs_error"].item()),
            )
            if atom_mode == "full":
                atom = components["full"]
            elif atom_mode in {
                "relation_off_corrected",
                "effect_off_corrected",
                "cross_off",
            }:
                atom = components[atom_mode]
            else:
                raise ValueError(f"Unsupported atom mode: {atom_mode}")
            if sample_indices is not None:
                selected = sample_indices[(sample_indices >= start) & (sample_indices < end)] - start
                if selected.numel():
                    sampled_atoms.append(atom[selected])
            path = build_source_path_messages(
                previous,
                atom,
                source[start:end],
                source_history_mode,
                permutation,
            )
            yield path, target[start:end]

    aggregate = getattr(model, f"{modality}_provenance_aggregator")(
        message_chunks(), degree, scalers, previous.dtype, previous.device
    )
    combine_input = build_target_combine_input(previous, aggregate, target_memory_mode)
    combine = getattr(model, f"{modality}_provenance_combine")
    state_norm = getattr(model, f"{modality}_provenance_state_norm")
    next_state = state_norm(combine(combine_input))
    atom_samples = torch.cat(sampled_atoms, dim=0) if sampled_atoms else None
    return next_state, aggregate, atom_samples, max_atom_error


def _relative_difference(full: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
    return (full - counterfactual).norm(dim=-1) / (full.norm(dim=-1) + EPSILON)


def _val_metrics(classifier, z: torch.Tensor, data) -> dict[str, float]:
    """Evaluate only the original validation indices; never inspect test labels."""
    with torch.no_grad():
        idx = data.val_idx.to(z.device)
        prediction = classifier(z[idx]).argmax(dim=-1).cpu()
    target = data.y[data.val_idx].cpu()
    # num_classes is dataset metadata, so no held-out split labels are needed
    # to define the stable Macro-F1 class set.
    labels = list(range(int(data.num_classes)))
    return {
        "val_acc": float((prediction == target).float().mean().item()),
        "val_macro_f1": float(
            f1_score(target.numpy(), prediction.numpy(), labels=labels, average="macro", zero_division=0)
        ),
    }


def _summary_fields(prefix: str, values: torch.Tensor, quantiles=(0.5, 0.9)) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in distribution(values, quantiles).items()}


def _collect_component_sample_rows(
    dataset: str,
    seed: int,
    modality: str,
    order: int,
    relation: torch.Tensor,
    effect: torch.Tensor,
    sample_indices: torch.Tensor,
    model: Model,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    # The caller passes already sampled relation/effect tensors; the global
    # edge indices are retained only as stable row identifiers.
    sample_relation = relation
    sample_effect = effect
    components = interaction_atom_components(model, sample_relation, sample_effect, modality)
    r, e, x, full = (components[key] for key in ("r", "e", "x", "full"))
    corrected = {key: components[key] for key in (
        "relation_off_corrected", "effect_off_corrected", "cross_off"
    )}
    bias = getattr(model, f"{modality}_interaction_cross").bias
    bias_norm = float(bias.detach().norm().item()) if bias is not None else 0.0
    metrics = {
        "r_norm": r.norm(dim=-1),
        "e_norm": e.norm(dim=-1),
        "x_norm": x.norm(dim=-1),
        "cos_re": F.cosine_similarity(r, e, dim=-1, eps=EPSILON),
        "cos_rx": F.cosine_similarity(r, x, dim=-1, eps=EPSILON),
        "cos_ex": F.cosine_similarity(e, x, dim=-1, eps=EPSILON),
    }
    denominator = full.norm(dim=-1) + EPSILON
    for key, atom in corrected.items():
        metrics[f"{key}_relative_difference"] = (full - atom).norm(dim=-1) / denominator
    rows = []
    for sample_position, edge_index in enumerate(sample_indices.detach().cpu().tolist()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "modality": modality,
            "order": order,
            "sample_position": sample_position,
            "edge_index": edge_index,
            "interaction_cross_bias_norm": bias_norm,
        }
        for key, tensor in metrics.items():
            row[key] = float(tensor[sample_position].item())
        rows.append(row)
    return rows, {**components, **metrics}


def _model_data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }


def _check_semantics_from_cache(full: dict[str, Any], semantic_cache: dict[str, Any]) -> float:
    candidate = {f"S_{modality}": semantic_cache[f"S_{modality}"] for modality in ("text", "visual")}
    candidate = {**candidate, "S_text": tuple(candidate["S_text"]), "S_visual": tuple(candidate["S_visual"])}
    return assert_semantic_invariance(full, candidate)


def run_one(dataset: str, seed: int, checkpoint: Path, device: torch.device):
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
    cfg = make_cfg(dataset, seed, checkpoint)
    if str(cfg.task.name) != "nc" or bool(cfg.task.evaluate_test):
        raise AssertionError("M1.1 audit must run NC validation only with test evaluation disabled")
    data = load_mag_data(cfg, "nc", int(seed))
    model = Model(cfg, _model_data_info(data)).to(device)
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not {"model_state", "head_state"}.issubset(payload):
        raise AssertionError(f"Checkpoint lacks model_state/head_state: {checkpoint}")
    model.load_state_dict(payload["model_state"])
    classifier.load_state_dict(payload["head_state"])
    model.eval()
    classifier.eval()
    model_before = module_state_sha256(model)
    head_before = module_state_sha256(classifier)
    checkpoint_before = file_sha256(checkpoint)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        full = model._compute(x, edge_index, collect_diagnostics=False)
    if not torch.isfinite(full["z"]).all():
        raise AssertionError("Full M1 output contains NaN/Inf")
    full_metrics = _val_metrics(classifier, full["z"], data)
    semantic_check = _check_semantics_from_cache(full, full["semantic_cache"])
    semantic_cache = full["semantic_cache"]

    def checked_provenance_step(*step_args, **step_kwargs):
        nonlocal semantic_check
        result = provenance_step(*step_args, **step_kwargs)
        semantic_check = max(semantic_check, _check_semantics_from_cache(full, semantic_cache))
        if semantic_check > TOLERANCE:
            raise AssertionError(f"Semantic invariance error {semantic_check} > {TOLERANCE}")
        return result

    node_count = int(x.size(0))
    edge_count = int(semantic_cache["source"].numel())
    sample_count = min(edge_count, 8192)
    sample_indices = (
        torch.linspace(0, edge_count - 1, steps=sample_count, device=device).round().long()
        if sample_count
        else torch.empty(0, dtype=torch.long, device=device)
    )

    full_steps: dict[str, list[torch.Tensor]] = {"text": [], "visual": []}
    full_aggregates: dict[str, list[torch.Tensor | None]] = {"text": [None], "visual": [None]}
    sample_atoms: dict[str, list[torch.Tensor | None]] = {"text": [None], "visual": [None]}
    sample_component_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    max_atom_error = 0.0
    max_step_error = 0.0

    # Rebuild every full provenance step using the audit path and compare it
    # with the actual frozen model result, while caching the exact full PNA Q.
    for modality in ("text", "visual"):
        full_steps[modality].append(full[f"G_{modality}"][0])
        for order in range(1, 4):
            effect = realized_semantic_effect_at_indices(
                model, semantic_cache, modality, order, sample_indices
            )
            relation = semantic_cache[f"R_{modality}"]
            component_rows, component_values = _collect_component_sample_rows(
                dataset,
                seed,
                modality,
                order,
                relation[sample_indices],
                effect,
                sample_indices,
                model,
            )
            sample_component_rows.extend(component_rows)
            sample_atoms[modality].append(component_values["full"])
            for metric in (
                "r_norm", "e_norm", "x_norm", "cos_re", "cos_rx", "cos_ex",
                "relation_off_corrected_relative_difference",
                "effect_off_corrected_relative_difference", "cross_off_relative_difference",
            ):
                q = (0.1, 0.5, 0.9) if metric.startswith("cos_") else (0.5, 0.9)
                summary = distribution(component_values[metric], q)
                geometry_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "order": order,
                    "metric": metric,
                    **summary,
                    "interaction_cross_bias_norm": float(
                        getattr(model, f"{modality}_interaction_cross").bias.detach().norm().item()
                    ),
                    "sample_edges": sample_count,
                })
            previous = full[f"G_{modality}"][order - 1]
            with torch.no_grad():
                rebuilt, aggregate, _atoms, atom_error = checked_provenance_step(
                    model,
                    semantic_cache,
                    previous,
                    modality,
                    order,
                    sample_indices=sample_indices,
                )
            max_atom_error = max(max_atom_error, atom_error)
            step_error = float((rebuilt - full[f"G_{modality}"][order]).abs().max().item())
            max_step_error = max(max_step_error, step_error)
            if step_error > TOLERANCE:
                raise AssertionError(
                    f"Full {modality} provenance step k={order} reconstruction error {step_error}"
                )
            full_steps[modality].append(rebuilt)
            full_aggregates[modality].append(aggregate)
        if len(full_steps[modality]) != 4:
            raise AssertionError("Expected G0 through G3")

    semantic_check = max(semantic_check, _check_semantics_from_cache(full, semantic_cache))
    local_rows: list[dict[str, Any]] = []
    for modality in ("text", "visual"):
        for order in (2, 3):
            previous = full[f"G_{modality}"][order - 1]
            with torch.no_grad():
                source_off, _, _, atom_error = checked_provenance_step(
                    model, semantic_cache, previous, modality, order,
                    source_history_mode="off", sample_indices=None,
                )
                target_off = getattr(model, f"{modality}_provenance_state_norm")(
                    getattr(model, f"{modality}_provenance_combine")(
                        build_target_combine_input(previous, full_aggregates[modality][order], "off")
                    )
                )
            semantic_check = max(semantic_check, _check_semantics_from_cache(full, semantic_cache))
            if semantic_check > TOLERANCE:
                raise AssertionError(f"Semantic invariance error {semantic_check} > {TOLERANCE}")
            max_atom_error = max(max_atom_error, atom_error)
            shuffle_differences = []
            for shuffle_seed in SHUFFLE_SEEDS:
                permutation = model._history_shuffle_permutation(
                    semantic_cache["degree"], shuffle_seed
                )
                with torch.no_grad():
                    shuffled, _, _, atom_error = checked_provenance_step(
                        model,
                        semantic_cache,
                        previous,
                        modality,
                        order,
                        source_history_mode="shuffle",
                        target_memory_mode="full",
                        permutation=permutation,
                    )
                max_atom_error = max(max_atom_error, atom_error)
                shuffle_differences.append(_relative_difference(full[f"G_{modality}"][order], shuffled))
            shuffle_stack = torch.stack(shuffle_differences, dim=0)
            shuffle_mean_per_node = shuffle_stack.mean(dim=0)
            shuffle_sd_per_node = shuffle_stack.std(dim=0, unbiased=False)
            row = {
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "order": order,
                **_summary_fields("source_history_off_difference", _relative_difference(full[f"G_{modality}"][order], source_off)),
                **_summary_fields("target_memory_off_difference", _relative_difference(full[f"G_{modality}"][order], target_off)),
                **_summary_fields("source_history_shuffle_difference_mean", shuffle_mean_per_node),
                **_summary_fields("source_history_shuffle_difference_sd", shuffle_sd_per_node),
                "source_shuffle_seed_mean_global": float(
                    torch.stack([values.mean() for values in shuffle_differences]).mean().item()
                ),
                "source_shuffle_seed_population_sd_global": float(
                    torch.stack([values.mean() for values in shuffle_differences]).std(unbiased=False).item()
                ),
                "semantic_max_abs_error": semantic_check,
            }
            for shuffle_position, shuffle_seed in enumerate(SHUFFLE_SEEDS):
                row.update(_summary_fields(
                    f"source_history_shuffle_seed_{shuffle_seed}_difference",
                    shuffle_differences[shuffle_position],
                ))
            local_rows.append(row)

    recursive_rows: list[dict[str, Any]] = []
    recursive_states: dict[str, dict[str, list[torch.Tensor]]] = {}
    recursive_modes = (
        ("source_history_off", "off", "full"),
        ("target_memory_off", "full", "off"),
        ("both_history_off_legacy", "off", "off"),
    )
    for variant, source_mode, target_mode in recursive_modes:
        recursive_states[variant] = {}
        for modality in ("text", "visual"):
            states = [full[f"G_{modality}"][0]]
            for order in range(1, 4):
                previous = states[-1]
                effective_source_mode = source_mode if (variant != "both_history_off_legacy" or order > 1) else "full"
                effective_target_mode = target_mode if (variant != "both_history_off_legacy" or order > 1) else "full"
                with torch.no_grad():
                    current, _, _, atom_error = checked_provenance_step(
                        model,
                        semantic_cache,
                        previous,
                        modality,
                        order,
                        source_history_mode=effective_source_mode,
                        target_memory_mode=effective_target_mode,
                    )
                max_atom_error = max(max_atom_error, atom_error)
                if order == 1 and variant in {"source_history_off", "target_memory_off", "both_history_off_legacy"}:
                    step_error = float((current - full[f"G_{modality}"][1]).abs().max().item())
                    if step_error > TOLERANCE:
                        raise AssertionError(f"k=1 must remain Full for {variant}: {step_error}")
                states.append(current)
            recursive_states[variant][modality] = states
        metrics, rowset = _recursive_output_rows(
            dataset, seed, variant, recursive_states[variant], full, model, classifier, data
        )
        recursive_rows.extend(rowset)

    for shuffle_seed in SHUFFLE_SEEDS:
        variant = f"correct_source_history_shuffle_{shuffle_seed}"
        recursive_states[variant] = {}
        # One node permutation is shared between text and visual streams.
        permutation = model._history_shuffle_permutation(semantic_cache["degree"], shuffle_seed)
        for modality in ("text", "visual"):
            states = [full[f"G_{modality}"][0]]
            for order in range(1, 4):
                previous = states[-1]
                mode = "shuffle" if order > 1 else "full"
                with torch.no_grad():
                    current, _, _, atom_error = checked_provenance_step(
                        model,
                        semantic_cache,
                        previous,
                        modality,
                        order,
                        source_history_mode=mode,
                        target_memory_mode="full",
                        permutation=permutation,
                    )
                max_atom_error = max(max_atom_error, atom_error)
                if order == 1:
                    step_error = float((current - full[f"G_{modality}"][1]).abs().max().item())
                    if step_error > TOLERANCE:
                        raise AssertionError(f"k=1 must remain Full for shuffle: {step_error}")
                states.append(current)
            recursive_states[variant][modality] = states
        _, rowset = _recursive_output_rows(
            dataset, seed, variant, recursive_states[variant], full, model, classifier, data
        )
        recursive_rows.extend(rowset)

    corrected_rows: list[dict[str, Any]] = []
    corrected_modes = ("relation_off_corrected", "effect_off_corrected", "cross_off")
    corrected_states: dict[str, dict[str, list[torch.Tensor]]] = {}
    for variant in corrected_modes:
        corrected_states[variant] = {}
        for modality in ("text", "visual"):
            states = [full[f"G_{modality}"][0]]
            atom_relative_by_order: list[torch.Tensor | None] = [None]
            for order in range(1, 4):
                previous = states[-1]
                with torch.no_grad():
                    current, _, counterfactual_atom_samples, atom_error = checked_provenance_step(
                        model,
                        semantic_cache,
                        previous,
                        modality,
                        order,
                        atom_mode=variant,
                        sample_indices=sample_indices,
                    )
                max_atom_error = max(max_atom_error, atom_error)
                states.append(current)
                reference_atoms = sample_atoms[modality][order]
                if counterfactual_atom_samples is None or reference_atoms is None:
                    atom_relative_by_order.append(torch.empty(0, device=device))
                else:
                    atom_relative_by_order.append(_relative_difference(reference_atoms, counterfactual_atom_samples))
            corrected_states[variant][modality] = states
            for order in range(1, 4):
                atom_diff = atom_relative_by_order[order]
                state_diff = _relative_difference(full[f"G_{modality}"][order], states[order])
                corrected_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "modality": modality,
                    "order": order,
                    **_summary_fields("atom_relative_difference", atom_diff),
                    **_summary_fields("G_relative_difference", state_diff),
                    "sample_edges": sample_count,
                    "semantic_max_abs_error": semantic_check,
                })
        _metrics, rowset = _recursive_output_rows(
            dataset, seed, variant, corrected_states[variant], full, model, classifier, data
        )
        # Attach final recursive output metrics to the k=3 row for each stream.
        recursive_by_modality = {row["modality"]: row for row in rowset}
        for row in corrected_rows:
            if row["variant"] == variant and row["dataset"] == dataset and row["seed"] == seed and row["order"] == 3:
                rr = recursive_by_modality[row["modality"]]
                for key in (
                    "val_acc", "val_macro_f1", "delta_val_acc_vs_full", "delta_val_macro_f1_vs_full",
                    "provenance_residual_relative_difference_mean",
                    "provenance_residual_relative_difference_p50",
                    "provenance_residual_relative_difference_p90",
                ):
                    row[key] = rr[key]

    # Set stable output columns for recursive rows, including all three shuffle seeds.
    recursive_rows = _with_recursive_metric_fields(recursive_rows)
    corrected_rows = _with_corrected_metric_fields(corrected_rows)
    sample_component_rows = _with_component_fields(sample_component_rows)
    geometry_rows = _with_geometry_fields(geometry_rows)
    local_rows = _with_local_fields(local_rows)

    dynamics_rows = []
    node_sample_count = min(node_count, 8192)
    node_indices = (
        torch.linspace(0, node_count - 1, steps=node_sample_count, device=device).round().long()
        if node_sample_count
        else torch.empty(0, dtype=torch.long, device=device)
    )
    for modality in ("text", "visual"):
        states = full_steps[modality]
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "modality": modality,
            "state_sample_nodes": node_sample_count,
        }
        for order in (1, 2, 3):
            state = states[order][node_indices]
            diversity = model._diversity(state)
            row.update({f"G{order}_{key}": value for key, value in diversity.items()})
            variance = state.float().var(dim=0, unbiased=False)
            row[f"G{order}_channel_variance_mean"] = float(variance.mean().item())
            row[f"G{order}_channel_variance_p50"] = float(torch.quantile(variance, 0.5).item())
        for left_order, right_order in ((1, 2), (2, 3)):
            left, right = states[left_order], states[right_order]
            relative = (right - left).norm(dim=-1) / (left.norm(dim=-1) + EPSILON)
            cosine = F.cosine_similarity(left, right, dim=-1, eps=EPSILON)
            row.update(_summary_fields(f"G{left_order}_to_G{right_order}_relative_change", relative))
            row.update(_summary_fields(f"G{left_order}_to_G{right_order}_cosine", cosine, (0.1, 0.5, 0.9)))
        for left_order, right_order in ((1, 2), (2, 3), (1, 3)):
            cosine = F.cosine_similarity(
                states[left_order][node_indices], states[right_order][node_indices], dim=-1, eps=EPSILON
            )
            row.update(_summary_fields(f"cos_G{left_order}_G{right_order}", cosine, (0.1, 0.5, 0.9)))
        dynamics_rows.append(row)

    model_after = module_state_sha256(model)
    head_after = module_state_sha256(classifier)
    checkpoint_after = file_sha256(checkpoint)
    if model_before != model_after or head_before != head_after:
        raise AssertionError("Frozen model/head tensor state changed during the audit")
    if checkpoint_before != checkpoint_after:
        raise AssertionError("Checkpoint file hash changed during the audit")
    if semantic_check > TOLERANCE:
        raise AssertionError(f"Semantic invariance error {semantic_check} > {TOLERANCE}")
    finite_tree({
        "local_rows": local_rows,
        "recursive_rows": recursive_rows,
        "corrected_rows": corrected_rows,
        "geometry_rows": geometry_rows,
        "full_metrics": full_metrics,
    })
    run_metadata = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "model_tensor_sha256_before": model_before,
        "model_tensor_sha256_after": model_after,
        "head_tensor_sha256_before": head_before,
        "head_tensor_sha256_after": head_after,
        "full_validation": full_metrics,
        "checkpoint_metrics": payload.get("metrics", {}),
        "num_nodes": node_count,
        "num_directed_edges": edge_count,
        "sampled_edges_per_order": sample_count,
        "semantic_max_abs_error": semantic_check,
        "full_atom_reconstruction_max_abs_error": max_atom_error,
        "full_provenance_step_max_abs_error": max_step_error,
        "model_head_unchanged": model_before == model_after and head_before == head_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
    }
    print(json.dumps({"dataset": dataset, "seed": seed, "status": "complete", **run_metadata["full_validation"]}), flush=True)
    del model, classifier, data, x, edge_index, full, semantic_cache, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run_metadata, local_rows, recursive_rows, sample_component_rows, geometry_rows, corrected_rows, dynamics_rows


def _recursive_output_rows(dataset, seed, variant, states, full, model, classifier, data):
    joined = torch.cat((states["text"][3], states["visual"][3]), dim=-1)
    residual = model.provenance_adapter(joined)
    z = full["base_z"] + residual
    metrics = _val_metrics(classifier, z, data)
    full_residual = full["provenance_residual"]
    residual_difference = _relative_difference(full_residual, residual)
    rows = []
    for modality in ("text", "visual"):
        difference = _relative_difference(full[f"G_{modality}"][3], states[modality][3])
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            "modality": modality,
            **_summary_fields("G3_relative_difference", difference),
            **_summary_fields("provenance_residual_relative_difference", residual_difference),
            **metrics,
            "delta_val_acc_vs_full": metrics["val_acc"] - _val_metrics(classifier, full["z"], data)["val_acc"],
            "delta_val_macro_f1_vs_full": metrics["val_macro_f1"] - _val_metrics(classifier, full["z"], data)["val_macro_f1"],
        })
    return metrics, rows


def _with_recursive_metric_fields(rows):
    fields = (
        "G3_relative_difference_mean", "G3_relative_difference_p50", "G3_relative_difference_p90",
        "provenance_residual_relative_difference_mean", "provenance_residual_relative_difference_p50",
        "provenance_residual_relative_difference_p90", "val_acc", "val_macro_f1",
        "delta_val_acc_vs_full", "delta_val_macro_f1_vs_full",
    )
    return [{key: row.get(key, "") for key in ("dataset", "seed", "variant", "modality", *fields)} for row in rows]


def _with_corrected_metric_fields(rows):
    fields = (
        "atom_relative_difference_mean", "atom_relative_difference_p50", "atom_relative_difference_p90",
        "G_relative_difference_mean", "G_relative_difference_p50", "G_relative_difference_p90",
        "sample_edges", "semantic_max_abs_error", "val_acc", "val_macro_f1",
        "delta_val_acc_vs_full", "delta_val_macro_f1_vs_full",
        "provenance_residual_relative_difference_mean", "provenance_residual_relative_difference_p50",
        "provenance_residual_relative_difference_p90",
    )
    return [{key: row.get(key, "") for key in ("dataset", "seed", "variant", "modality", "order", *fields)} for row in rows]


def _with_component_fields(rows):
    fields = (
        "sample_position", "edge_index", "interaction_cross_bias_norm", "r_norm", "e_norm", "x_norm",
        "cos_re", "cos_rx", "cos_ex", "relation_off_corrected_relative_difference",
        "effect_off_corrected_relative_difference", "cross_off_relative_difference",
    )
    return [{key: row.get(key, "") for key in ("dataset", "seed", "modality", "order", *fields)} for row in rows]


def _with_geometry_fields(rows):
    fields = ("metric", "mean", "p10", "p50", "p90", "interaction_cross_bias_norm", "sample_edges")
    return [{key: row.get(key, "") for key in ("dataset", "seed", "modality", "order", *fields)} for row in rows]


def _with_local_fields(rows):
    fields = (
        "source_history_shuffle_seed_3407_difference_mean", "source_history_shuffle_seed_3407_difference_p50", "source_history_shuffle_seed_3407_difference_p90",
        "source_history_shuffle_seed_3408_difference_mean", "source_history_shuffle_seed_3408_difference_p50", "source_history_shuffle_seed_3408_difference_p90",
        "source_history_shuffle_seed_3409_difference_mean", "source_history_shuffle_seed_3409_difference_p50", "source_history_shuffle_seed_3409_difference_p90",
        "source_history_off_difference_mean", "source_history_off_difference_p50", "source_history_off_difference_p90",
        "target_memory_off_difference_mean", "target_memory_off_difference_p50", "target_memory_off_difference_p90",
        "source_history_shuffle_difference_mean_mean", "source_history_shuffle_difference_mean_p50",
        "source_history_shuffle_difference_mean_p90", "source_history_shuffle_difference_sd_mean",
        "source_history_shuffle_difference_sd_p50", "source_history_shuffle_difference_sd_p90",
        "source_shuffle_seed_mean_global", "source_shuffle_seed_population_sd_global", "semantic_max_abs_error",
    )
    return [{key: row.get(key, "") for key in ("dataset", "seed", "modality", "order", *fields)} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _branch_metadata() -> dict[str, Any]:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {
        "source_branch": "exp/interaction_provenance_m1",
        "source_sha": "fdc05a1af05a93a0c6d4c9d68b903f6cf018d811",
        "source_working_tree_status_at_start": "clean",
        "new_branch": git("branch", "--show-current"),
        "current_head_at_audit_start": git("rev-parse", "HEAD"),
        "current_working_tree_status_at_audit_start": git("status", "--porcelain"),
    }


def run(args) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    checkpoints = [ROOT / relative for relative in CHECKPOINT_RELATIVE_PATHS]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise SystemExit("STOP: required official M1 checkpoints are missing:\n" + "\n".join(missing))
    device = torch.device(args.device if str(args.device) != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(args.cuda_memory_fraction), device)
    start_hashes = {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in checkpoints}
    metadata: dict[str, Any] = {
        **_branch_metadata(),
        "protocol": {
            "audit": "M1.1 corrected interaction-provenance attribution",
            "training": False,
            "fine_tuning": False,
            "task": "original NC validation only",
            "test_evaluation": False,
            "LP": False,
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "semantic_invariance_tolerance": TOLERANCE,
            "max_component_geometry_edges_per_order": 8192,
        },
        "device": str(device),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_memory_fraction_cap": float(args.cuda_memory_fraction) if device.type == "cuda" else None,
        "checkpoint_hashes_at_start": start_hashes,
        "checkpoint_runs": [],
    }
    all_local, all_recursive, all_components, all_geometry, all_corrected, all_dynamics = [], [], [], [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            checkpoint = ROOT / "outputs/model_design/m1/pilot" / dataset / f"seed{seed}" / "best_checkpoint.pt"
            run_meta, local, recursive, components, geometry, corrected, dynamics = run_one(dataset, seed, checkpoint, device)
            metadata["checkpoint_runs"].append(run_meta)
            all_local.extend(local)
            all_recursive.extend(recursive)
            all_components.extend(components)
            all_geometry.extend(geometry)
            all_corrected.extend(corrected)
            all_dynamics.extend(dynamics)
            metadata["checkpoint_hashes_at_end"] = {
                p.relative_to(ROOT).as_posix(): file_sha256(p) for p in checkpoints
            }
            (output_dir / "source_metadata.json").parent.mkdir(parents=True, exist_ok=True)
            (output_dir / "source_metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    metadata["checkpoint_hashes_at_end"] = {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in checkpoints}
    metadata["all_checkpoints_unchanged"] = metadata["checkpoint_hashes_at_start"] == metadata["checkpoint_hashes_at_end"]
    metadata["all_model_head_states_unchanged"] = all(
        run["model_head_unchanged"] for run in metadata["checkpoint_runs"]
    )
    metadata["all_semantic_invariance_checks_passed"] = all(
        run["semantic_max_abs_error"] <= TOLERANCE for run in metadata["checkpoint_runs"]
    )
    metadata["maximum_full_atom_reconstruction_error"] = max(
        run["full_atom_reconstruction_max_abs_error"] for run in metadata["checkpoint_runs"]
    )
    metadata["maximum_full_provenance_step_error"] = max(
        run["full_provenance_step_max_abs_error"] for run in metadata["checkpoint_runs"]
    )
    if not metadata["all_checkpoints_unchanged"] or not metadata["all_model_head_states_unchanged"]:
        raise AssertionError("Checkpoint or tensor-state integrity check failed")
    if not metadata["all_semantic_invariance_checks_passed"]:
        raise AssertionError("One or more semantic invariance checks failed")
    (output_dir / "source_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "local_history_attribution.csv", all_local)
    write_csv(output_dir / "recursive_history_counterfactuals.csv", all_recursive)
    write_csv(output_dir / "corrected_atom_components.csv", all_components)
    write_csv(output_dir / "component_geometry.csv", all_geometry)
    write_csv(output_dir / "corrected_component_counterfactuals.csv", all_corrected)
    write_csv(output_dir / "provenance_dynamics_corrected.csv", all_dynamics)
    print(json.dumps({"audit": "complete", "output_dir": str(output_dir), "runs": len(metadata["checkpoint_runs"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Frozen corrected M1 provenance attribution audit")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or cuda:1")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.35)
    parser.add_argument("--output-dir", default="results/model_design/m11_audit")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
