from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from src.analysis.problem_validation.common import (
    BOOTSTRAP_REPLICATES,
    PROBE_DIM,
    classification_metrics,
    mean_neighbor_messages,
    set_run_seed,
)
from src.data import MAGData


class MessageProbe(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(4 * PROBE_DIM, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def build_message_features(
    h_text: torch.Tensor, h_visual: torch.Tensor, edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_text, isolated = mean_neighbor_messages(h_text, edge_index)
    n_visual, isolated_visual = mean_neighbor_messages(h_visual, edge_index)
    if not torch.equal(isolated, isolated_visual):
        raise AssertionError("modality message degree masks differ")
    features = torch.cat((h_text, h_visual, n_text, n_visual), dim=-1)
    return features, n_text, n_visual, isolated


def train_message_probe(
    data: MAGData,
    run_seed: int,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    physical_edge_index: torch.Tensor,
    probe_train_idx: torch.Tensor,
    probe_calib_idx: torch.Tensor,
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], MessageProbe, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train a linear P0.1 classifier while P0.0 semantic embeddings stay frozen."""
    assert data.y is not None and data.val_idx is not None
    run_dir.mkdir(parents=True, exist_ok=True)
    set_run_seed(run_seed)
    h_text_device = h_text.to(device)
    h_visual_device = h_visual.to(device)
    physical_edge_index = physical_edge_index.to(device)
    features, n_text, n_visual, isolated_mask = build_message_features(h_text_device, h_visual_device, physical_edge_index)
    train_idx = probe_train_idx.to(device)
    calib_idx = probe_calib_idx.to(device)
    train_labels = data.y[probe_train_idx].to(device)
    calib_labels = data.y[probe_calib_idx].to(device)
    model = MessageProbe(int(data.num_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = 30
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(features[train_idx])
        loss = F.cross_entropy(train_logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            calib_logits = model(features[calib_idx])
            calib = classification_metrics(calib_logits, calib_labels, int(data.num_classes))
        if calib["acc"] > best_acc + 1e-4:
            best_acc = calib["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 30
        elif epoch >= 30:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("P0.1 training did not produce a calibration-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(features[train_idx])
        calib_logits = model(features[calib_idx])
        val_idx = data.val_idx.to(device)
        val_labels = data.y[data.val_idx].to(device)
        val_logits = model(features[val_idx])
        train_metrics = classification_metrics(train_logits, train_labels, int(data.num_classes))
        calib_metrics = classification_metrics(calib_logits, calib_labels, int(data.num_classes))
        val_metrics = classification_metrics(val_logits, val_labels, int(data.num_classes))
        # Only original-val logits are populated; other nodes are never scored by the head.
        full_logits = torch.zeros((data.num_nodes, int(data.num_classes)), device=device, dtype=val_logits.dtype)
        full_logits[val_idx] = val_logits
    ckpt = {
        "classifier_state": {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()},
        "run_seed": run_seed,
        "best_epoch": best_epoch,
        "probe_train_acc": train_metrics["acc"],
        "probe_train_macro_f1": train_metrics["macro_f1"],
        "probe_calib_acc": calib_metrics["acc"],
        "probe_calib_macro_f1": calib_metrics["macro_f1"],
        "selection_metric": "probe_calib_accuracy",
        "semantic_projectors_frozen": True,
        "isolated_node_mask": isolated_mask.detach().cpu(),
    }
    torch.save(ckpt, run_dir / "message_probe.pt")
    metrics = {
        "best_epoch": best_epoch,
        "probe_train": train_metrics,
        "probe_calib": calib_metrics,
        "heldout_analysis_val": val_metrics,
        "checkpoint_selection": "probe_calib_accuracy_only",
    }
    return metrics, model, features.detach(), full_logits.detach(), isolated_mask.detach().cpu()


def analytic_message_contributions(
    model: MessageProbe,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    target_node: torch.Tensor,
    neighbor_node: torch.Tensor,
    target_degree: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = model.classifier.weight
    wt = weight[:, 2 * PROBE_DIM : 3 * PROBE_DIM]
    wv = weight[:, 3 * PROBE_DIM : 4 * PROBE_DIM]
    target_device = weight.device
    target_node = target_node.to(target_device)
    neighbor_node = neighbor_node.to(target_device)
    degree = target_degree.to(target_device).clamp_min(1).to(weight.dtype).unsqueeze(-1)
    delta_text = (h_text.to(target_device)[neighbor_node] @ wt.T) / degree
    delta_visual = (h_visual.to(target_device)[neighbor_node] @ wv.T) / degree
    return delta_text, delta_visual, delta_text + delta_visual


def check_decomposition_exactness(
    model: MessageProbe,
    features: torch.Tensor,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    target_node: torch.Tensor,
    neighbor_node: torch.Tensor,
    target_degree: torch.Tensor,
    sample_seed: int = 42,
    sample_count: int = 100,
) -> float:
    """CPU float32 explicit-vs-analytic message removal audit."""
    if target_node.numel() < sample_count:
        raise ValueError(f"decomposition audit needs at least {sample_count} sampled edges")
    rng = np.random.default_rng(sample_seed)
    choice = rng.choice(target_node.numel(), size=sample_count, replace=False)
    pick = torch.as_tensor(choice, dtype=torch.long)
    cpu_model = MessageProbe(model.classifier.out_features).cpu().float().eval()
    cpu_model.load_state_dict({k: v.detach().cpu().float() for k, v in model.state_dict().items()})
    f = features.detach().cpu().float()
    ht, hv = h_text.detach().cpu().float(), h_visual.detach().cpu().float()
    targets, neighbors = target_node[pick].cpu(), neighbor_node[pick].cpu()
    degrees = target_degree[pick].cpu().float().unsqueeze(-1)
    full = cpu_model(f[targets])
    delta_t, delta_v, delta_tv = analytic_message_contributions(
        cpu_model, ht, hv, targets, neighbors, degrees.squeeze(-1).long()
    )
    errors: list[torch.Tensor] = []
    for modality, delta in (("text", delta_t), ("visual", delta_v), ("joint", delta_tv)):
        removed = f[targets].clone()
        if modality in ("text", "joint"):
            removed[:, 2 * PROBE_DIM : 3 * PROBE_DIM] -= ht[neighbors] / degrees
        if modality in ("visual", "joint"):
            removed[:, 3 * PROBE_DIM : 4 * PROBE_DIM] -= hv[neighbors] / degrees
        explicit = cpu_model(removed)
        analytic = full - delta
        errors.append((explicit - analytic).abs().max())
    max_error = float(torch.stack(errors).max().item())
    if not math.isfinite(max_error) or max_error >= 1e-5:
        raise AssertionError(f"message decomposition max_abs_error={max_error:.8g} >= 1e-5")
    return max_error


def _similarities(
    raw_unit: torch.Tensor, probe_unit: torch.Tensor, targets: torch.Tensor, neighbors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = targets.to(raw_unit.device)
    neighbors = neighbors.to(raw_unit.device)
    raw = (raw_unit[targets] * raw_unit[neighbors]).sum(dim=-1)
    probe = (probe_unit[targets] * probe_unit[neighbors]).sum(dim=-1)
    return raw, probe


def remove_message_utility(
    full_logits: torch.Tensor, removed_logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Positive CE utility means removing the message raises loss: the message helped."""
    return F.cross_entropy(removed_logits, labels, reduction="none") - F.cross_entropy(
        full_logits, labels, reduction="none"
    )


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    chosen = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return chosen - masked.max(dim=-1).values


def compute_edge_rows(
    data: MAGData,
    model: MessageProbe,
    features: torch.Tensor,
    full_logits: torch.Tensor,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    population: dict[str, torch.Tensor],
    run_seed: int,
    device: torch.device,
    chunk_size: int = 65_536,
) -> dict[str, Any]:
    """Vectorized exact edge contributions and removal utilities, chunked by edges."""
    assert data.x_t is not None and data.x_i is not None and data.y is not None
    targets = population["target_node"].cpu().long()
    neighbors = population["neighbor_node"].cpu().long()
    degrees = population["target_degree"].cpu().long()
    n_edges = targets.numel()
    if n_edges == 0:
        raise ValueError("P0.1 analysis edge population is empty")
    row_chunks: dict[str, list[torch.Tensor]] = {}
    keys = (
        "raw_sim_text", "raw_sim_visual", "probe_sim_text", "probe_sim_visual",
        "utility_ce_text", "utility_ce_visual", "utility_ce_joint",
        "utility_margin_text", "utility_margin_visual", "utility_margin_joint",
        "full_ce", "full_margin", "text_visual_utility_sign_disagree",
        "target_pred_correct",
    )
    row_chunks = {key: [] for key in keys}
    raw_text_unit = F.normalize(data.x_t.to(device), p=2, dim=-1)
    raw_visual_unit = F.normalize(data.x_i.to(device), p=2, dim=-1)
    probe_text_unit = F.normalize(h_text.to(device), p=2, dim=-1)
    probe_visual_unit = F.normalize(h_visual.to(device), p=2, dim=-1)
    model.eval()
    with torch.no_grad():
        for start in range(0, n_edges, chunk_size):
            stop = min(start + chunk_size, n_edges)
            sl = slice(start, stop)
            t, n, d = targets[sl].to(device), neighbors[sl].to(device), degrees[sl].to(device)
            y = data.y[t.cpu()].to(device)
            base_logits = full_logits[t.to(full_logits.device)].to(device)
            delta_t, delta_v, delta_tv = analytic_message_contributions(model, h_text, h_visual, t, n, d)
            removed = (base_logits - delta_t, base_logits - delta_v, base_logits - delta_tv)
            full_ce = F.cross_entropy(base_logits, y, reduction="none")
            utilities = [remove_message_utility(base_logits, x, y) for x in removed]
            full_margin = _margin(base_logits, y)
            margins = [_margin(base_logits, y) - _margin(x, y) for x in removed]
            raw_t, probe_t = _similarities(raw_text_unit, probe_text_unit, t, n)
            raw_v, probe_v = _similarities(raw_visual_unit, probe_visual_unit, t, n)
            disagreement = torch.sign(utilities[0]) != torch.sign(utilities[1])
            values = {
                "raw_sim_text": raw_t,
                "raw_sim_visual": raw_v,
                "probe_sim_text": probe_t,
                "probe_sim_visual": probe_v,
                "utility_ce_text": utilities[0],
                "utility_ce_visual": utilities[1],
                "utility_ce_joint": utilities[2],
                "utility_margin_text": margins[0],
                "utility_margin_visual": margins[1],
                "utility_margin_joint": margins[2],
                "full_ce": full_ce,
                "full_margin": full_margin,
                "text_visual_utility_sign_disagree": disagreement,
                "target_pred_correct": base_logits.argmax(dim=-1) == y,
            }
            for key, value in values.items():
                row_chunks[key].append(value.detach().cpu())
    result: dict[str, Any] = {
        "dataset": data.name,
        "run_seed": int(run_seed),
        "target_node": targets,
        "neighbor_node": neighbors,
        "target_label": data.y[targets].cpu(),
        "target_degree": degrees,
        "analysis_target_nodes": population["target_nodes"].cpu().long(),
    }
    for key, chunks in row_chunks.items():
        result[key] = torch.cat(chunks)
    finite_keys = [key for key in keys if result[key].dtype != torch.bool]
    for key in finite_keys:
        if not bool(torch.isfinite(result[key]).all()):
            raise FloatingPointError(f"non-finite edge-analysis values in {data.name} {key}")
    result["isolated_node_count"] = int((population["target_nodes"].numel() - torch.unique(targets).numel()))
    result["sampled_relation_count"] = int(n_edges)
    return result


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _auroc(score: np.ndarray, utility: np.ndarray) -> float:
    label = utility > 0
    if np.unique(label).size < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def _group_partition(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    groups_sorted = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, len(values)]))
    groups = np.empty(len(values), dtype=np.int64)
    groups[order] = groups_sorted
    return order, groups, int(starts.size)


def _weighted_ranks(groups: np.ndarray, n_groups: int, weights: np.ndarray) -> np.ndarray:
    sorted_group_weights = np.bincount(groups, weights=weights, minlength=n_groups)
    cum = np.cumsum(sorted_group_weights)
    group_rank = cum - (sorted_group_weights - 1.0) / 2.0
    return group_rank[groups]


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    total = w.sum()
    if total <= 1:
        return float("nan")
    mx, my = np.dot(w, x) / total, np.dot(w, y) / total
    dx, dy = x - mx, y - my
    vx, vy = np.dot(w, dx * dx), np.dot(w, dy * dy)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.dot(w, dx * dy) / math.sqrt(vx * vy))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float, order: np.ndarray) -> float:
    sorted_w = weights[order]
    cumulative = np.cumsum(sorted_w)
    total = cumulative[-1]
    index = min(int(np.searchsorted(cumulative, q * total, side="left")), len(order) - 1)
    return float(values[order[index]])


def node_bootstrap_cis(
    arrays: dict[str, np.ndarray],
    target_nodes: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 42,
    bootstrap_target_nodes: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Node-cluster bootstrap CIs for CE correlations, quantile rates, and disagreement."""
    unique_nodes = np.unique(target_nodes if bootstrap_target_nodes is None else bootstrap_target_nodes)
    edge_node_group = np.searchsorted(unique_nodes, target_nodes)
    n_nodes = len(unique_nodes)
    if n_nodes < 2 or len(target_nodes) == 0:
        return {}
    x_names = ["raw_sim_text", "raw_sim_visual", "probe_sim_text", "probe_sim_visual"]
    utility_names = ["utility_ce_text", "utility_ce_visual"]
    group_info = {name: _group_partition(arrays[name]) for name in x_names + utility_names}
    rng = np.random.default_rng(seed)
    stats: dict[str, list[float]] = {}
    highlow_specs = [
        (x, u, "high_sim_harmful", "<")
        for x, u in (("raw_sim_text", "utility_ce_text"), ("raw_sim_visual", "utility_ce_visual"),
                     ("probe_sim_text", "utility_ce_text"), ("probe_sim_visual", "utility_ce_visual"))
    ] + [
        (x, u, "low_sim_beneficial", ">")
        for x, u in (("raw_sim_text", "utility_ce_text"), ("raw_sim_visual", "utility_ce_visual"),
                     ("probe_sim_text", "utility_ce_text"), ("probe_sim_visual", "utility_ce_visual"))
    ]
    for _ in range(replicates):
        node_draw = rng.integers(0, n_nodes, size=n_nodes)
        node_multiplicity = np.bincount(node_draw, minlength=n_nodes)
        edge_w = node_multiplicity[edge_node_group].astype(np.float64)
        total = edge_w.sum()
        if total <= 1:
            continue
        rank_cache: dict[str, np.ndarray] = {}
        for name, (order, groups, n_groups) in group_info.items():
            rank_cache[name] = _weighted_ranks(groups, n_groups, edge_w)
        for x, u in (("raw_sim_text", "utility_ce_text"), ("raw_sim_visual", "utility_ce_visual"),
                     ("probe_sim_text", "utility_ce_text"), ("probe_sim_visual", "utility_ce_visual")):
            val = _weighted_corr(rank_cache[x], rank_cache[u], edge_w)
            stats.setdefault(f"{x}.spearman_ce", []).append(val)
        for x, u, metric, direction in highlow_specs:
            order = group_info[x][0]
            q = 0.8 if metric == "high_sim_harmful" else 0.2
            threshold = _weighted_quantile(arrays[x], edge_w, q, order)
            selected = arrays[x] >= threshold if q == 0.8 else arrays[x] <= threshold
            outcome = arrays[u] < 0 if direction == "<" else arrays[u] > 0
            denom = edge_w[selected].sum()
            val = float(np.dot(edge_w[selected], outcome[selected]) / denom) if denom else float("nan")
            stats.setdefault(f"{x}.{metric}", []).append(val)
        u_t, u_v = arrays["utility_ce_text"], arrays["utility_ce_visual"]
        exact = np.sign(u_t) != np.sign(u_v)
        stats.setdefault("utility_ce_text_visual_exact_disagreement", []).append(float(np.dot(edge_w, exact) / total))
        eps = 0.1 * float(np.median(np.abs(arrays["utility_ce_joint"])))
        if eps == 0:
            eps = 1e-8
        robust = (np.abs(u_t) > eps) & (np.abs(u_v) > eps)
        robust_n = edge_w[robust].sum()
        stats.setdefault("utility_ce_text_visual_robust_disagreement", []).append(
            float(np.dot(edge_w[robust], exact[robust]) / robust_n) if robust_n else float("nan")
        )
        stats.setdefault("utility_ce_text_visual_robust_coverage", []).append(float(robust_n / total))
    output = {}
    for name, values in stats.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            output[name] = {"ci95_low": float(np.quantile(finite, 0.025)), "ci95_high": float(np.quantile(finite, 0.975))}
        else:
            output[name] = {"ci95_low": float("nan"), "ci95_high": float("nan")}
    return output


def summarize_edge_rows(edge_rows: dict[str, Any], bootstrap_replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    arrays = {k: v.detach().cpu().numpy() for k, v in edge_rows.items() if isinstance(v, torch.Tensor) and v.ndim == 1}
    statistics: dict[str, Any] = {}
    for modality in ("text", "visual"):
        utility_ce = arrays[f"utility_ce_{modality}"]
        utility_margin = arrays[f"utility_margin_{modality}"]
        stats_mod: dict[str, Any] = {}
        for space in ("raw", "probe"):
            sim = arrays[f"{space}_sim_{modality}"]
            for utility_name, utility in (("ce", utility_ce), ("margin", utility_margin)):
                stats_mod[f"{space}_spearman_{utility_name}"] = _safe_spearman(sim, utility)
                if utility_name == "ce":
                    stats_mod[f"{space}_auroc_beneficial"] = _auroc(sim, utility)
                    q20, q80 = np.quantile(sim, [0.2, 0.8])
                    low = sim <= q20
                    high = sim >= q80
                    stats_mod[f"{space}_low_sim_beneficial_rate"] = float(np.mean(utility[low] > 0)) if low.any() else float("nan")
                    stats_mod[f"{space}_high_sim_harmful_rate"] = float(np.mean(utility[high] < 0)) if high.any() else float("nan")
                    stats_mod[f"{space}_low_sim_threshold"] = float(q20)
                    stats_mod[f"{space}_high_sim_threshold"] = float(q80)
        statistics[modality] = stats_mod
    ut, uv, uj = arrays["utility_ce_text"], arrays["utility_ce_visual"], arrays["utility_ce_joint"]
    exact_mask = np.sign(ut) != np.sign(uv)
    eps = 0.1 * float(np.median(np.abs(uj)))
    if eps == 0:
        eps = 1e-8
    robust_mask = (np.abs(ut) > eps) & (np.abs(uv) > eps)
    statistics["text_visual"] = {
        "exact_disagreement_rate": float(exact_mask.mean()),
        "robust_disagreement_rate": float(exact_mask[robust_mask].mean()) if robust_mask.any() else float("nan"),
        "robust_edge_coverage": float(robust_mask.mean()),
        "robust_epsilon": eps,
        "ce_margin_sign_agreement_text": float(np.mean(np.sign(ut) == np.sign(arrays["utility_margin_text"]))),
        "ce_margin_sign_agreement_visual": float(np.mean(np.sign(uv) == np.sign(arrays["utility_margin_visual"]))),
        "ce_margin_spearman_text": _safe_spearman(ut, arrays["utility_margin_text"]),
        "ce_margin_spearman_visual": _safe_spearman(uv, arrays["utility_margin_visual"]),
    }
    statistics["bootstrap"] = node_bootstrap_cis(
        arrays, arrays["target_node"], bootstrap_replicates, seed=42,
        bootstrap_target_nodes=arrays["analysis_target_nodes"],
    )
    return statistics


def analyze_and_save_edges(
    data: MAGData,
    model: MessageProbe,
    features: torch.Tensor,
    full_logits: torch.Tensor,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    population: dict[str, torch.Tensor],
    run_seed: int,
    run_dir: Path,
    device: torch.device,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    isolated_node_mask: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    err = check_decomposition_exactness(
        model, features, h_text, h_visual,
        population["target_node"], population["neighbor_node"], population["target_degree"],
    )
    rows = compute_edge_rows(data, model, features, full_logits, h_text, h_visual, population, run_seed, device)
    stats = summarize_edge_rows(rows, bootstrap_replicates=bootstrap_replicates)
    stats["bootstrap_metadata"] = {"resampling_unit": "target_node", "replicates": int(bootstrap_replicates), "seed": 42, "confidence_level": 0.95}
    rows["statistics"] = stats
    rows["bootstrap_replicates"] = int(bootstrap_replicates)
    rows["bootstrap_seed"] = 42
    rows["decomposition_max_abs_error"] = err
    if isolated_node_mask is not None:
        rows["isolated_node_mask"] = isolated_node_mask.detach().cpu().bool()
    if not bool(torch.isfinite(rows["utility_ce_text"]).all()):
        raise FloatingPointError("non-finite CE utility detected")
    torch.save(rows, run_dir / "edge_analysis.pt")
    return rows, stats
