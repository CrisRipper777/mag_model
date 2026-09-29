from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

BOOTSTRAP_METRICS = (
    "overall_beneficial_rate",
    "overall_harmful_rate",
    "q1_beneficial_rate",
    "q5_beneficial_rate",
    "q1_harmful_rate",
    "q5_harmful_rate",
    "q1_beneficial_lift",
    "q5_beneficial_lift",
    "q1_harmful_lift",
    "q5_harmful_lift",
    "beneficial_stratification_range",
    "harmful_stratification_range",
)


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _validate(similarity: Any, utility: Any) -> tuple[np.ndarray, np.ndarray]:
    sim = _array(similarity, "similarity").astype(np.float64, copy=False)
    util = _array(utility, "utility").astype(np.float64, copy=False)
    if sim.shape != util.shape:
        raise ValueError("similarity and utility must have the same shape")
    if sim.size == 0:
        raise ValueError("the sampled relation population cannot be empty")
    if not np.isfinite(sim).all() or not np.isfinite(util).all():
        raise ValueError("similarity and utility must contain only finite values")
    return sim, util


def quintile_partition_indices(similarity: Any) -> tuple[np.ndarray, ...]:
    """Partition all relations into balanced quintiles using stable global ordering.

    This matches the original P0.1 diagnostic-figure rule: stable mergesort over
    the complete per-run relation population followed by ``numpy.array_split``.
    When similarity ties cross a boundary, original artifact row order breaks
    the tie deterministically.
    """
    sim = _array(similarity, "similarity").astype(np.float64, copy=False)
    if sim.size == 0:
        raise ValueError("the sampled relation population cannot be empty")
    if not np.isfinite(sim).all():
        raise ValueError("similarity must contain only finite values")
    order = np.argsort(sim, kind="mergesort")
    return tuple(np.asarray(group, dtype=np.int64) for group in np.array_split(order, 5))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0.0 else float("nan")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(y == y[0]):
        return float("nan")
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _profile_from_weights(
    similarity: np.ndarray,
    utility: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute base/conditional rates for an integer-weighted edge population.

    Integer weights represent bootstrap copies of an original relation. Their
    repeated ranks are kept adjacent in stable artifact-row order, which makes
    weighted partitions equivalent to explicitly repeating rows and applying
    ``array_split``.
    """
    if order is None:
        order = np.argsort(similarity, kind="mergesort")
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0:
        return {name: float("nan") for name in (
            "overall_beneficial_rate", "overall_harmful_rate", "overall_zero_rate",
            *(f"q{q}_{kind}_rate" for q in range(1, 6) for kind in ("beneficial", "harmful", "zero")),
            *(f"q{q}_{kind}_{suffix}" for q in (1, 5) for kind in ("beneficial", "harmful") for suffix in ("lift", "ratio")),
            "beneficial_stratification_range", "harmful_stratification_range",
            "beneficial_quantile_spearman", "harmful_quantile_spearman",
        )}

    beneficial = utility > 0.0
    harmful = utility < 0.0
    zero = utility == 0.0
    weighted = weights.astype(np.float64, copy=False)
    overall_b = float(np.dot(weighted, beneficial) / total)
    overall_h = float(np.dot(weighted, harmful) / total)
    overall_z = float(np.dot(weighted, zero) / total)

    sorted_weights = weighted[order]
    cumulative_end = np.cumsum(sorted_weights, dtype=np.float64)
    cumulative_start = cumulative_end - sorted_weights
    population_size = int(round(total))
    base, extra = divmod(population_size, 5)
    sizes = [base + int(index < extra) for index in range(5)]

    result: dict[str, float] = {
        "overall_beneficial_rate": overall_b,
        "overall_harmful_rate": overall_h,
        "overall_zero_rate": overall_z,
    }
    q_b: list[float] = []
    q_h: list[float] = []
    lower = 0
    for q, size in enumerate(sizes, start=1):
        upper = lower + size
        overlap = np.maximum(
            0.0,
            np.minimum(cumulative_end, float(upper)) - np.maximum(cumulative_start, float(lower)),
        )
        denom = float(size)
        b_rate = float(np.dot(overlap, beneficial[order]) / denom) if denom else float("nan")
        h_rate = float(np.dot(overlap, harmful[order]) / denom) if denom else float("nan")
        z_rate = float(np.dot(overlap, zero[order]) / denom) if denom else float("nan")
        result[f"q{q}_beneficial_rate"] = b_rate
        result[f"q{q}_harmful_rate"] = h_rate
        result[f"q{q}_zero_rate"] = z_rate
        q_b.append(b_rate)
        q_h.append(h_rate)
        lower = upper

    for q, rate in ((1, q_b[0]), (5, q_b[4])):
        result[f"q{q}_beneficial_lift"] = float(rate - overall_b)
        result[f"q{q}_beneficial_ratio"] = _safe_ratio(rate, overall_b)
    for q, rate in ((1, q_h[0]), (5, q_h[4])):
        result[f"q{q}_harmful_lift"] = float(rate - overall_h)
        result[f"q{q}_harmful_ratio"] = _safe_ratio(rate, overall_h)

    result["beneficial_stratification_range"] = float(np.nanmax(q_b) - np.nanmin(q_b))
    result["harmful_stratification_range"] = float(np.nanmax(q_h) - np.nanmin(q_h))
    q_index = np.arange(1, 6, dtype=np.float64)
    result["beneficial_quantile_spearman"] = _safe_spearman(q_index, np.asarray(q_b))
    result["harmful_quantile_spearman"] = _safe_spearman(q_index, np.asarray(q_h))
    return result


def summarize_utility_by_similarity(similarity: Any, utility: Any) -> dict[str, float]:
    """Summarize exact-sign utility rates over one fixed sampled relation set."""
    sim, util = _validate(similarity, utility)
    return _profile_from_weights(
        sim,
        util,
        np.ones(sim.size, dtype=np.int64),
        order=np.argsort(sim, kind="mergesort"),
    )


def node_bootstrap_metric_samples(
    similarity: Any,
    utilities: dict[str, Any],
    target_node: Any,
    analysis_target_nodes: Any,
    *,
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Target-node cluster bootstrap; each selected target carries all its edges.

    The quantile partition and each lift are recomputed within each replicate.
    This avoids subtracting independently bootstrapped confidence intervals.
    """
    sim = _array(similarity, "similarity").astype(np.float64, copy=False)
    nodes = _array(target_node, "target_node").astype(np.int64, copy=False)
    target_population = _array(analysis_target_nodes, "analysis_target_nodes").astype(np.int64, copy=False)
    if sim.size != nodes.size:
        raise ValueError("similarity and target_node must have the same length")
    if sim.size == 0 or target_population.size == 0:
        raise ValueError("bootstrap population cannot be empty")
    if len(np.unique(target_population)) != target_population.size:
        raise ValueError("analysis_target_nodes must be unique")
    if np.any(target_population[1:] < target_population[:-1]):
        raise ValueError("analysis_target_nodes must be sorted for deterministic cluster indexing")
    if not np.isfinite(sim).all():
        raise ValueError("similarity must contain only finite values")
    edge_node_index = np.searchsorted(target_population, nodes)
    if (edge_node_index >= target_population.size).any() or not np.array_equal(target_population[edge_node_index], nodes):
        raise ValueError("every sampled relation target must belong to analysis_target_nodes")
    utility_arrays: dict[str, np.ndarray] = {}
    for name, values in utilities.items():
        _, util = _validate(sim, values)
        utility_arrays[name] = util
    orders = {name: np.argsort(sim, kind="mergesort") for name in utility_arrays}
    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in BOOTSTRAP_METRICS} for name in utility_arrays
    }
    n_targets = target_population.size
    for _ in range(replicates):
        draws = rng.integers(0, n_targets, size=n_targets)
        multiplicity = np.bincount(draws, minlength=n_targets)
        edge_weights = multiplicity[edge_node_index]
        for name, util in utility_arrays.items():
            profile = _profile_from_weights(sim, util, edge_weights, orders[name])
            for metric in BOOTSTRAP_METRICS:
                values[name][metric].append(profile[metric])
    return {
        name: {metric: np.asarray(samples, dtype=np.float64) for metric, samples in metrics.items()}
        for name, metrics in values.items()
    }


def percentile_ci(samples: Any) -> tuple[float, float]:
    array = _array(samples, "bootstrap samples").astype(np.float64, copy=False)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def node_bootstrap_confidence_intervals(
    similarity: Any,
    utilities: dict[str, Any],
    target_node: Any,
    analysis_target_nodes: Any,
    *,
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float]]]:
    samples = node_bootstrap_metric_samples(
        similarity,
        utilities,
        target_node,
        analysis_target_nodes,
        replicates=replicates,
        seed=seed,
    )
    return {
        name: {metric: percentile_ci(values) for metric, values in metrics.items()}
        for name, metrics in samples.items()
    }
