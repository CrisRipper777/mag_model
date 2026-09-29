from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.analysis.problem_validation.p01plus_statistics import quintile_partition_indices
from src.analysis.problem_validation.p02_models import P02Model
from src.analysis.problem_validation.p02_training import IncomingCSR, safe_labels
from src.data import MAGData


def _check_gate_ranges(variant: str, scalar: torch.Tensor, feature: torch.Tensor, config: dict[str, Any]) -> None:
    if not torch.isfinite(scalar).all() or not torch.isfinite(feature).all():
        raise FloatingPointError(f"{variant}: non-finite relation gates")
    if scalar.numel() and (scalar.min() < 0 or scalar.max() > float(config["analysis"]["scalar_gate_scale"])):
        raise FloatingPointError(f"{variant}: scalar gate outside configured [0, scale] range")
    if variant == "conditional_feature":
        scale = float(config["analysis"]["feature_modulation_scale"])
        low, high = 1.0 - scale, 1.0 + scale
        if feature.numel() and (feature.min() < low - 1e-6 or feature.max() > high + 1e-6):
            raise FloatingPointError(f"{variant}: feature gate outside [{low}, {high}]")


@torch.no_grad()
def precompute_edge_states(
    model: P02Model,
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    config: dict[str, Any],
    *,
    identity_function: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute fixed relation a/g from H0 once, in physical-edge chunks."""
    model.eval()
    result: dict[str, dict[str, torch.Tensor]] = {}
    chunk_size = int(config["analysis"]["relation_edge_chunk_size"])
    src_all, dst_all = graph.src, graph.dst
    for modality_index, (name, embeddings) in enumerate((
        ("text", h_text), ("visual", h_visual)
    )):
        scalar_chunks: list[torch.Tensor] = []
        feature_chunks: list[torch.Tensor] = []
        similarity_chunks: list[torch.Tensor] = []
        for start in range(0, src_all.numel(), chunk_size):
            stop = min(start + chunk_size, src_all.numel())
            src, dst = src_all[start:stop], dst_all[start:stop]
            a, g = model.relation_function(
                modality_index,
                embeddings[dst],
                embeddings[src],
                identity_function=identity_function,
            )
            sim = F.cosine_similarity(embeddings[dst], embeddings[src], dim=-1)
            scalar_chunks.append(a.detach().float().cpu())
            feature_chunks.append(g.detach().float().cpu())
            similarity_chunks.append(sim.detach().float().cpu())
        scalar = torch.cat(scalar_chunks) if scalar_chunks else torch.empty((0, 1))
        feature = torch.cat(feature_chunks) if feature_chunks else torch.empty((0, embeddings.size(1)))
        similarity = torch.cat(similarity_chunks) if similarity_chunks else torch.empty(0)
        _check_gate_ranges(model.variant, scalar, feature, config)
        result[name] = {"a": scalar, "g": feature, "probe_similarity": similarity, "_embedding": embeddings.detach().float().cpu()}
    return result


@torch.no_grad()
def aggregate_with_edge_states(
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    states: dict[str, dict[str, torch.Tensor]],
    *,
    identity_function: bool = False,
    chunk_size: int = 32_768,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply frozen edge states to source features and use original degree means."""
    device = h_text.device
    outputs: list[torch.Tensor] = []
    for name, embeddings in (("text", h_text), ("visual", h_visual)):
        state = states[name]
        a = state["a"].to(device)
        g = state["g"].to(device)
        aggregate = embeddings.new_zeros(embeddings.shape)
        for start in range(0, graph.src.numel(), chunk_size):
            stop = min(start + chunk_size, graph.src.numel())
            src, dst = graph.src[start:stop], graph.dst[start:stop]
            feature_gate = torch.ones_like(g[start:stop]) if identity_function else g[start:stop]
            message = a[start:stop] * (feature_gate * embeddings[src])
            aggregate.index_add_(0, dst, message)
        aggregate = aggregate / graph.degree.clamp_min(1).to(aggregate.dtype).unsqueeze(-1)
        outputs.append(aggregate)
    return outputs[0], outputs[1]


@torch.no_grad()
def make_all_node_features(
    model: P02Model,
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    states: dict[str, dict[str, torch.Tensor]],
    *,
    identity_function: bool = False,
    chunk_size: int = 32_768,
) -> torch.Tensor:
    n_text, n_visual = aggregate_with_edge_states(
        graph,
        h_text,
        h_visual,
        states,
        identity_function=identity_function,
        chunk_size=chunk_size,
    )
    return torch.cat((h_text, h_visual, n_text, n_visual), dim=-1)


def norm_profile(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite diagnostic distribution")
    if values.numel() == 0:
        return {key: float("nan") for key in ("mean", "median", "p25", "p75", "p90", "p95", "max")}
    q = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75, 0.90, 0.95]))
    return {
        "mean": float(values.mean()),
        "median": float(q[1]),
        "p25": float(q[0]),
        "p75": float(q[2]),
        "p90": float(q[3]),
        "p95": float(q[4]),
        "max": float(values.max()),
    }


@torch.no_grad()
def context_rollout(
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    states: dict[str, dict[str, torch.Tensor]],
    *,
    max_order: int,
    chunk_size: int,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, Any]]:
    """Compose the H0-derived fixed relation states without re-encoding later orders."""
    contexts = {"text": [h_text], "visual": [h_visual]}
    stability: dict[str, Any] = {}
    device = h_text.device
    for order in range(1, max_order + 1):
        for name, embeddings in (("text", h_text), ("visual", h_visual)):
            previous = contexts[name][-1]
            state = states[name]
            a, g = state["a"].to(device), state["g"].to(device)
            aggregate = embeddings.new_zeros(embeddings.shape)
            for start in range(0, graph.src.numel(), chunk_size):
                stop = min(start + chunk_size, graph.src.numel())
                src, dst = graph.src[start:stop], graph.dst[start:stop]
                message = a[start:stop] * (g[start:stop] * previous[src])
                aggregate.index_add_(0, dst, message)
            current = aggregate / graph.degree.clamp_min(1).to(aggregate.dtype).unsqueeze(-1)
            if not torch.isfinite(current).all():
                raise FloatingPointError(f"context rollout {name} C{order} contains NaN/Inf; stop")
            norms = torch.linalg.vector_norm(current.float(), dim=-1)
            if not torch.isfinite(norms).all():
                raise FloatingPointError(f"context rollout {name} C{order} norm contains NaN/Inf; stop")
            contexts[name].append(current)
            stability[f"{name}_C{order}_norm"] = norm_profile(norms)
    return contexts, stability


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    chosen = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return chosen - masked.max(dim=-1).values


def exact_decomposition_audit(
    model: P02Model,
    all_features: torch.Tensor,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    target: torch.Tensor,
    neighbor: torch.Tensor,
    degree: torch.Tensor,
    edge_state: dict[str, dict[str, torch.Tensor]],
    *,
    seed: int,
    sample_count: int = 256,
) -> dict[str, Any]:
    """CPU float32 analytic-vs-explicit removal on >=100 random sampled validation edges."""
    if target.numel() < 100:
        raise ValueError("exact edge audit requires at least 100 sampled edges")
    rng = np.random.default_rng(seed)
    count = min(max(100, sample_count), int(target.numel()))
    chosen = torch.as_tensor(rng.choice(target.numel(), size=count, replace=False), dtype=torch.long)
    if "physical_edge_row" not in edge_state:
        raise ValueError("edge_state must include the P0.1-to-physical-graph row alignment")
    graph_rows = edge_state["physical_edge_row"][chosen].long()
    model_cpu = P02Model(
        model.variant,
        num_classes=model.classifier.out_features,
        semantic_dim=model.semantic_dim,
        relation_dim=model.relation_dim,
        relation_dropout=0.0,
        scalar_gate_scale=model.scalar_gate_scale,
        feature_modulation_scale=model.feature_modulation_scale,
        similarity_init_alpha=0.0,
        similarity_init_beta=0.0,
    ).cpu().float().eval()
    model_cpu.load_state_dict({k: v.detach().cpu().float() for k, v in model.state_dict().items()})
    t = target[chosen].cpu().long()
    n = neighbor[chosen].cpu().long()
    d = degree[chosen].cpu().float().clamp_min(1).unsqueeze(-1)
    features = all_features[t].detach().cpu().float()
    full = model_cpu.classifier(features)
    errors: dict[str, float] = {}
    for modality_index, name in enumerate(("text", "visual")):
        src_h = (h_text if name == "text" else h_visual).detach().cpu().float()[n]
        dst_h = (h_text if name == "text" else h_visual).detach().cpu().float()[t]
        a = edge_state[name]["a"][graph_rows].cpu().float()
        g = edge_state[name]["g"][graph_rows].cpu().float()
        message = a * (g * src_h)
        block_start = 2 * model.semantic_dim + modality_index * model.semantic_dim
        block = slice(block_start, block_start + model.semantic_dim)
        delta = (message / d) @ model_cpu.classifier.weight[:, block].T
        removed = features.clone()
        removed[:, block] -= message / d
        explicit = model_cpu.classifier(removed)
        errors[name] = float((explicit - (full - delta)).abs().max())
    joint_message_text = edge_state["text"]["a"][chosen].cpu().float() * (
        edge_state["text"]["g"][chosen].cpu().float() * h_text.detach().cpu().float()[n]
    )
    joint_message_visual = edge_state["visual"]["a"][chosen].cpu().float() * (
        edge_state["visual"]["g"][chosen].cpu().float() * h_visual.detach().cpu().float()[n]
    )
    removed_joint = features.clone()
    t_block = slice(2 * model.semantic_dim, 3 * model.semantic_dim)
    v_block = slice(3 * model.semantic_dim, 4 * model.semantic_dim)
    removed_joint[:, t_block] -= joint_message_text / d
    removed_joint[:, v_block] -= joint_message_visual / d
    delta_t = (joint_message_text / d) @ model_cpu.classifier.weight[:, t_block].T
    delta_v = (joint_message_visual / d) @ model_cpu.classifier.weight[:, v_block].T
    errors["joint"] = float(
        (model_cpu.classifier(removed_joint) - (full - delta_t - delta_v)).abs().max()
    )
    max_error = max(errors.values())
    if not math.isfinite(max_error) or max_error >= 1e-5:
        raise AssertionError(f"{model.variant}: exact edge decomposition max error {max_error:.8g} >= 1e-5")
    return {"sample_count": count, "seed": int(seed), "dtype": "cpu_float32", "max_abs_error": max_error, "by_scope": errors}


@torch.no_grad()
def compute_edge_utilities(
    data: MAGData,
    model: P02Model,
    graph: IncomingCSR,
    all_features: torch.Tensor,
    states: dict[str, dict[str, torch.Tensor]],
    p01_reference: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    """Model-specific exact removal utilities on the unchanged P0.1 edge rows."""
    target = p01_reference["target_node"].cpu().long()
    neighbor = p01_reference["neighbor_node"].cpu().long()
    degree = p01_reference["target_degree"].cpu().long()
    if not torch.equal(target, split["sampled_edge_target_node"]) or not torch.equal(
        neighbor, split["sampled_edge_neighbor_node"]
    ):
        raise RuntimeError("P0.2 reference population differs from the frozen P0.1 ordered edge population")
    device = all_features.device
    val = split["original_val_idx"].to(device)
    labels = safe_labels(data, target, split["original_val_idx"]).to(device)
    val_features = all_features[val]
    val_logits = model.classifier(val_features)
    val_position = torch.full((data.num_nodes,), -1, dtype=torch.long)
    val_position[val.cpu()] = torch.arange(val.numel(), dtype=torch.long)
    positions = val_position[target]
    if (positions < 0).any():
        raise RuntimeError("sampled relation targets are not all held-out original-val nodes")
    full_logits = val_logits[positions.to(device)]
    y = labels.to(device)
    result: dict[str, Any] = {
        "target_node": target,
        "neighbor_node": neighbor,
        "target_degree": degree,
        "analysis_target_nodes": split["analysis_target_nodes"].cpu().long(),
        "physical_edge_row": torch.cat([
            torch.searchsorted(graph.dst * graph.num_nodes + graph.src, target.to(graph.device) * graph.num_nodes + neighbor.to(graph.device)).cpu()
        ]),
        "full_logits_val_edges": full_logits.detach().cpu().float(),
        "target_label": labels.cpu().long(),
    }
    for modality_index, name in enumerate(("text", "visual")):
        state = states[name]
        graph_key = graph.dst * graph.num_nodes + graph.src
        sampled_key = target.to(device) * graph.num_nodes + neighbor.to(device)
        edge_rows = torch.searchsorted(graph_key, sampled_key)
        if (edge_rows >= graph_key.numel()).any() or not torch.equal(graph_key[edge_rows], sampled_key):
            raise RuntimeError("P0.1 sampled physical edge cannot be located in P0.2 graph")
        a = state["a"][edge_rows.cpu()].to(device)
        g = state["g"][edge_rows.cpu()].to(device)
        h_mod = states[name].get("_embedding")
        if h_mod is None:
            raise RuntimeError("internal error: frozen semantic embedding missing from edge state")
        msg = a * (g * h_mod[neighbor].to(device))
        block_start = 2 * model.semantic_dim + modality_index * model.semantic_dim
        block = slice(block_start, block_start + model.semantic_dim)
        delta = (msg / degree.to(device).clamp_min(1).unsqueeze(-1)) @ model.classifier.weight[:, block].T
        removed = full_logits - delta
        full_ce = F.cross_entropy(full_logits, y, reduction="none")
        utility_ce = F.cross_entropy(removed, y, reduction="none") - full_ce
        full_margin = _margin(full_logits, y)
        utility_margin = full_margin - _margin(removed, y)
        result[f"utility_ce_{name}"] = utility_ce.detach().cpu().float()
        result[f"utility_margin_{name}"] = utility_margin.detach().cpu().float()

    delta_text = None
    delta_visual = None
    for modality_index, name in enumerate(("text", "visual")):
        state = states[name]
        graph_key = graph.dst * graph.num_nodes + graph.src
        sampled_key = target.to(device) * graph.num_nodes + neighbor.to(device)
        edge_rows = torch.searchsorted(graph_key, sampled_key)
        a = state["a"][edge_rows.cpu()].to(device)
        g = state["g"][edge_rows.cpu()].to(device)
        h_mod = state["_embedding"]
        msg = a * (g * h_mod[neighbor].to(device))
        block = slice(2 * model.semantic_dim + modality_index * model.semantic_dim, 3 * model.semantic_dim + modality_index * model.semantic_dim)
        delta = (msg / degree.to(device).clamp_min(1).unsqueeze(-1)) @ model.classifier.weight[:, block].T
        if name == "text":
            delta_text = delta
        else:
            delta_visual = delta
    delta_joint = delta_text + delta_visual
    removed_joint = full_logits - delta_joint
    result["utility_ce_joint"] = (
        F.cross_entropy(removed_joint, y, reduction="none") - F.cross_entropy(full_logits, y, reduction="none")
    ).detach().cpu().float()
    result["utility_margin_joint"] = (
        _margin(full_logits, y) - _margin(removed_joint, y)
    ).detach().cpu().float()
    result["target_pred_correct"] = (full_logits.argmax(-1) == y).detach().cpu()
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite edge diagnostic field {key}")
    return result


def compatibility_masks(reference: dict[str, Any]) -> dict[str, Any]:
    """P0.1+ balanced-quintile counterexample and consistency masks."""
    result: dict[str, Any] = {}
    for name in ("text", "visual"):
        sim = reference[f"probe_sim_{name}"].cpu().numpy()
        utility = reference[f"utility_ce_{name}"].cpu().numpy()
        q = np.zeros(sim.size, dtype=np.int8)
        for number, indices in enumerate(quintile_partition_indices(sim), start=1):
            q[indices] = number
        low_b = (q == 1) & (utility > 0)
        high_h = (q == 5) & (utility < 0)
        conflict = low_b | high_h
        consistent = ((q == 1) & (utility < 0)) | ((q == 5) & (utility > 0))
        result[name] = {
            "quintile": q,
            "low_sim_beneficial": low_b,
            "high_sim_harmful": high_h,
            "conflict": conflict,
            "consistent": consistent,
        }
    ref_text = reference["utility_ce_text"].cpu().numpy()
    ref_visual = reference["utility_ce_visual"].cpu().numpy()
    ref_joint = reference["utility_ce_joint"].cpu().numpy()
    median = float(np.median(np.abs(ref_joint)))
    epsilon = 0.1 * median if median != 0.0 else 1e-8
    robust = (np.abs(ref_text) > epsilon) & (np.abs(ref_visual) > epsilon)
    disagree = robust & (np.sign(ref_text) != np.sign(ref_visual))
    agree = robust & (np.sign(ref_text) == np.sign(ref_visual))
    result["robust_tv_disagreement"] = disagree
    result["robust_tv_agreement"] = agree
    result["robust_epsilon"] = epsilon
    return result


def mechanism_diagnostics(
    model: P02Model,
    states: dict[str, dict[str, torch.Tensor]],
    reference: dict[str, Any],
    graph: IncomingCSR,
    split: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute sampled-edge transform activity and grouped distribution summaries."""
    if model.variant != "conditional_feature":
        raise ValueError("mechanism feature diagnostics are defined for conditional_feature")
    target = reference["target_node"].cpu().long()
    neighbor = reference["neighbor_node"].cpu().long()
    graph_key = graph.dst * graph.num_nodes + graph.src
    sampled_key = target.to(graph.device) * graph.num_nodes + neighbor.to(graph.device)
    edge_rows = torch.searchsorted(graph_key, sampled_key)
    if (edge_rows >= graph_key.numel()).any() or not torch.equal(graph_key[edge_rows], sampled_key):
        raise RuntimeError("cannot align mechanism factors to the P0.1 sample")
    masks = compatibility_masks(reference)
    arrays: dict[str, Any] = {"target_node": target, "neighbor_node": neighbor, "masks": masks}
    summaries: list[dict[str, Any]] = []
    for modality_index, name in enumerate(("text", "visual")):
        state = states[name]
        g = state["g"][edge_rows.cpu()].float()
        h = state["_embedding"][neighbor].float()
        transformed = g * h
        h_norm = torch.linalg.vector_norm(h, dim=-1)
        residual = torch.linalg.vector_norm((g - 1.0) * h, dim=-1) / (h_norm + 1e-12)
        cosine = F.cosine_similarity(transformed, h, dim=-1)
        non_collinearity = 1.0 - cosine.abs()
        channel_std = g.std(dim=-1, unbiased=False)
        values = {
            "transform_residual_ratio": residual,
            "non_collinearity": non_collinearity,
            "feature_gate_channel_std": channel_std,
        }
        arrays[name] = {key: value.detach().cpu() for key, value in values.items()}
        group_masks: dict[str, np.ndarray] = {"all_edges": np.ones(target.numel(), dtype=bool)}
        group_masks["compatibility_conflict"] = masks[name]["conflict"]
        group_masks["compatibility_consistent"] = masks[name]["consistent"]
        group_masks["robust_tv_disagreement"] = masks["robust_tv_disagreement"]
        group_masks["robust_tv_agreement"] = masks["robust_tv_agreement"]
        for group, mask in group_masks.items():
            row: dict[str, Any] = {
                "modality": name,
                "subset": group,
                "edge_count": int(mask.sum()),
                "robust_epsilon": float(masks["robust_epsilon"]),
            }
            for metric, values_tensor in values.items():
                profile = norm_profile(values_tensor[torch.as_tensor(mask)])
                for stat, val in profile.items():
                    row[f"{metric}_{stat}"] = val
            summaries.append(row)
    arrays["robust_epsilon"] = masks["robust_epsilon"]
    return arrays, {"rows": summaries, "robust_epsilon": masks["robust_epsilon"]}


def _cluster_bootstrap_mean(
    delta: np.ndarray,
    selected: np.ndarray,
    target_node: np.ndarray,
    analysis_target_nodes: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if not selected.any():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    target_pos = np.searchsorted(analysis_target_nodes, target_node)
    n_target = len(analysis_target_nodes)
    draws = rng.integers(0, n_target, size=(replicates, n_target))
    multiplicity = np.zeros((replicates, n_target), dtype=np.int32)
    rows = np.arange(replicates)[:, None]
    np.add.at(multiplicity, (np.broadcast_to(rows, draws.shape), draws), 1)
    weights = multiplicity[:, target_pos] * selected[None, :]
    denom = weights.sum(axis=1)
    numer = (weights * delta[None, :]).sum(axis=1)
    values = np.divide(numer, denom, out=np.full(replicates, np.nan), where=denom > 0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    ci = np.quantile(finite, [0.025, 0.975])
    return float(ci[0]), float(ci[1])


def conflict_delta_rows(
    reference: dict[str, Any],
    p02_v2: dict[str, Any],
    p02_v3: dict[str, Any],
    split: dict[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Paired V3-V2 utility differences with node-cluster bootstrap CIs."""
    masks = compatibility_masks(reference)
    target = reference["target_node"].cpu().numpy().astype(np.int64)
    analysis_targets = split["analysis_target_nodes"].cpu().numpy().astype(np.int64)
    if np.any(analysis_targets[1:] < analysis_targets[:-1]):
        raise ValueError("analysis_target_nodes must be sorted for deterministic bootstrap indexing")
    target_pos = np.searchsorted(analysis_targets, target)
    if (target_pos >= len(analysis_targets)).any() or not np.array_equal(analysis_targets[target_pos], target):
        raise ValueError("sampled target nodes fall outside the fixed bootstrap population")

    subset_rows: dict[str, tuple[str, np.ndarray]] = {}
    for ref_mod in ("text", "visual"):
        for label in ("low_sim_beneficial", "high_sim_harmful", "conflict", "consistent"):
            subset_rows[f"{label}_{ref_mod}"] = (ref_mod, masks[ref_mod][label])
    subset_rows["all_conflict_union"] = (
        "joint_union", masks["text"]["conflict"] | masks["visual"]["conflict"]
    )

    scopes = ("text", "visual", "joint")
    deltas = np.stack([
        p02_v3[f"utility_ce_{scope}"].cpu().numpy().astype(np.float64)
        - p02_v2[f"utility_ce_{scope}"].cpu().numpy().astype(np.float64)
        for scope in scopes
    ], axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    n_target = len(analysis_targets)
    draws = rng.integers(0, n_target, size=(bootstrap_replicates, n_target))
    multiplicity = np.zeros((bootstrap_replicates, n_target), dtype=np.int32)
    reps = np.broadcast_to(np.arange(bootstrap_replicates)[:, None], draws.shape)
    np.add.at(multiplicity, (reps, draws), 1)

    rows: list[dict[str, Any]] = []
    for subset_name, (reference_modality, selected) in subset_rows.items():
        if selected.any():
            selected_pos = target_pos[selected]
            node_count = np.bincount(selected_pos, minlength=n_target).astype(np.float64)
            cluster_delta = np.zeros((n_target, len(scopes)), dtype=np.float64)
            np.add.at(cluster_delta, selected_pos, deltas[selected])
            denom = multiplicity @ node_count
            numer = multiplicity @ cluster_delta
            samples = np.divide(
                numer, denom[:, None], out=np.full_like(numer, np.nan), where=denom[:, None] > 0
            )
        else:
            node_count = np.zeros(n_target, dtype=np.float64)
            cluster_delta = np.zeros((n_target, len(scopes)), dtype=np.float64)
            samples = np.full((bootstrap_replicates, len(scopes)), np.nan)
        for scope_index, utility_scope in enumerate(scopes):
            delta_selected = deltas[selected, scope_index]
            finite = samples[:, scope_index][np.isfinite(samples[:, scope_index])]
            ci_low, ci_high = (
                (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
                if finite.size else (float("nan"), float("nan"))
            )
            u2 = p02_v2[f"utility_ce_{utility_scope}"].cpu().numpy().astype(np.float64)
            u3 = p02_v3[f"utility_ce_{utility_scope}"].cpu().numpy().astype(np.float64)
            rows.append({
                "reference_modality": reference_modality,
                "subset": subset_name,
                "utility_scope": utility_scope,
                "edge_count": int(selected.sum()),
                "v2_mean_utility": float(u2[selected].mean()) if selected.any() else float("nan"),
                "v3_mean_utility": float(u3[selected].mean()) if selected.any() else float("nan"),
                "mean_delta_u_v3_minus_v2": float(delta_selected.mean()) if selected.any() else float("nan"),
                "median_delta_u_v3_minus_v2": float(np.median(delta_selected)) if selected.any() else float("nan"),
                "fraction_delta_u_positive": float(np.mean(delta_selected > 0)) if selected.any() else float("nan"),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "bootstrap_replicates": int(bootstrap_replicates),
                "bootstrap_seed": int(bootstrap_seed),
            })
    return rows
