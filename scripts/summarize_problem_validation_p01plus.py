from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.problem_validation.common import DATASETS, OUTPUT_ROOT  # noqa: E402
from src.analysis.problem_validation.p01plus_statistics import (  # noqa: E402
    BOOTSTRAP_METRICS,
    node_bootstrap_metric_samples,
    summarize_utility_by_similarity,
)

SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
SIMILARITY_SPACES = ("raw", "probe")
UTILITY_KINDS = ("ce", "margin")
ARTIFACT_FIELDS = (
    "target_node",
    "analysis_target_nodes",
    "raw_sim_text",
    "raw_sim_visual",
    "probe_sim_text",
    "probe_sim_visual",
    "utility_ce_text",
    "utility_ce_visual",
    "utility_margin_text",
    "utility_margin_visual",
)


def load_existing_edge_artifacts(
    input_root: Path,
    *,
    datasets: tuple[str, ...] = DATASETS,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load only the named P0.1 edge_analysis.pt files; never glob other artifacts."""
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in datasets:
        for seed in seeds:
            artifact_path = input_root / dataset / f"seed{seed}" / "edge_analysis.pt"
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Required existing P0.1 artifact is missing: {artifact_path}")
            payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected a tensor dictionary in {artifact_path}")
            missing = set(ARTIFACT_FIELDS) - payload.keys()
            if missing:
                raise KeyError(f"{artifact_path} is missing required P0.1 arrays: {sorted(missing)}")
            if payload.get("dataset") != dataset or int(payload.get("run_seed", -1)) != seed:
                raise ValueError(f"Artifact identity mismatch in {artifact_path}")
            # Select only the inputs specified for this audit. In particular, do not
            # access target_label, test indices, metrics.json, model states, or embeddings.
            loaded[(dataset, seed)] = {key: payload[key] for key in ARTIFACT_FIELDS}
            loaded[(dataset, seed)]["sampled_relation_count"] = int(payload["sampled_relation_count"])
            loaded[(dataset, seed)]["robust_text_visual_disagreement_rate"] = float(
                payload["statistics"]["text_visual"]["robust_disagreement_rate"]
            )
    return loaded


def _to_numpy(payload: dict[str, Any], name: str) -> np.ndarray:
    value = payload[name]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def analyze_run(
    dataset: str,
    seed: int,
    payload: dict[str, Any],
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 42,
) -> list[dict[str, Any]]:
    edge_target = _to_numpy(payload, "target_node").astype(np.int64, copy=False)
    target_population = _to_numpy(payload, "analysis_target_nodes").astype(np.int64, copy=False)
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        utilities: dict[str, np.ndarray] = {}
        for kind in UTILITY_KINDS:
            utilities[kind] = _to_numpy(payload, f"utility_{kind}_{modality}").astype(np.float64, copy=False)
        for space in SIMILARITY_SPACES:
            similarity = _to_numpy(payload, f"{space}_sim_{modality}").astype(np.float64, copy=False)
            point_estimates = {
                kind: summarize_utility_by_similarity(similarity, utility)
                for kind, utility in utilities.items()
            }
            bootstrap = node_bootstrap_metric_samples(
                similarity,
                utilities,
                edge_target,
                target_population,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "modality": modality.title(),
                "similarity_space": space,
                "sampled_relation_count": int(payload["sampled_relation_count"]),
                "robust_text_visual_disagreement_rate": float(payload["robust_text_visual_disagreement_rate"]),
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_resampling_unit": "target_node; all sampled relations retained",
            }
            for kind, estimates in point_estimates.items():
                prefix = "" if kind == "ce" else "margin_"
                for metric, value in estimates.items():
                    row[f"{prefix}{metric}"] = value
                # Explicit names make the low/high endpoints easy to consume.
                aliases = {
                    "low_beneficial_lift": "q1_beneficial_lift",
                    "high_beneficial_lift": "q5_beneficial_lift",
                    "low_harmful_lift": "q1_harmful_lift",
                    "high_harmful_lift": "q5_harmful_lift",
                    "low_beneficial_ratio": "q1_beneficial_ratio",
                    "high_beneficial_ratio": "q5_beneficial_ratio",
                    "low_harmful_ratio": "q1_harmful_ratio",
                    "high_harmful_ratio": "q5_harmful_ratio",
                }
                for alias, metric in aliases.items():
                    row[f"{prefix}{alias}"] = estimates[metric]
                for metric in BOOTSTRAP_METRICS:
                    low, high = _percentile_ci(bootstrap[kind][metric])
                    row[f"{prefix}{metric}_ci95_low"] = low
                    row[f"{prefix}{metric}_ci95_high"] = high
            rows.append(row)
    return rows


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ("dataset", "modality", "similarity_space")
    identifiers = set(group_keys) | {"seed", "bootstrap_replicates", "bootstrap_seed", "bootstrap_resampling_unit"}
    numeric_metrics = [
        key for key, value in rows[0].items()
        if key not in identifiers and isinstance(value, (int, float, np.number)) and "_ci95_" not in key
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in group_keys)
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: (DATASETS.index(k[0]), MODALITIES.index(k[1].lower()), SIMILARITY_SPACES.index(k[2]))):
        members = groups[key]
        aggregate: dict[str, Any] = dict(zip(group_keys, key, strict=True))
        aggregate["available_seeds"] = len({member["seed"] for member in members})
        for metric in numeric_metrics:
            vals = np.asarray([float(member[metric]) for member in members], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            aggregate[f"{metric}_mean"] = float(vals.mean()) if vals.size else float("nan")
            aggregate[f"{metric}_population_std"] = float(vals.std(ddof=0)) if vals.size else float("nan")
        result.append(aggregate)
    return result


def make_dataset_plot(dataset: str, per_seed: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = [row for row in per_seed if row["dataset"] == dataset and row["similarity_space"] == "probe"]
    colors = {"beneficial": "#2171A5", "harmful": "#B34A3C"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    q = np.arange(1, 6)
    for ax, modality in zip(axes, MODALITIES, strict=True):
        seeds = [row for row in rows if row["modality"] == modality]
        for outcome, color in colors.items():
            metric = f"q{{i}}_{outcome}_rate"
            values = np.asarray([[row[metric.format(i=i)] for i in q] for row in seeds], dtype=np.float64)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=0)
            ax.plot(q, mean, color=color, marker="o", linewidth=2, label=f"P(U {('>' if outcome == 'beneficial' else '<')} 0 | Qq)")
            ax.fill_between(q, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.16, linewidth=0)
            base = np.asarray([row[f"overall_{outcome}_rate"] for row in seeds], dtype=np.float64)
            ax.axhline(base.mean(), color=color, linestyle="--", linewidth=1.3, label=f"overall P(U {('>' if outcome == 'beneficial' else '<')} 0)")
        ax.set(title=modality, xlabel="Probe-similarity quintile", xticks=q, ylim=(0, 1))
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=7.7, loc="best")
    axes[0].set_ylabel("Conditional relation proportion")
    fig.suptitle(f"{dataset}: CE utility by probe-similarity quintile (mean ± seed SD)")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"p01plus_{dataset}_probe_quintile_utility.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _finite(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_agg(row: dict[str, Any], metric: str, digits: int = 3) -> str:
    mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_population_std")
    if mean is None or not math.isfinite(float(mean)):
        return "NA"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def _ci_sign_counts(rows: list[dict[str, Any]], metric: str) -> tuple[int, int, int]:
    pos = neg = mixed = 0
    for row in rows:
        low = row.get(f"{metric}_ci95_low")
        high = row.get(f"{metric}_ci95_high")
        if low is None or high is None or not math.isfinite(float(low)) or not math.isfinite(float(high)):
            mixed += 1
        elif float(low) > 0:
            pos += 1
        elif float(high) < 0:
            neg += 1
        else:
            mixed += 1
    return pos, neg, mixed


def _sign_summary(rows: list[dict[str, Any]], metric: str) -> str:
    pos, neg, mixed = _ci_sign_counts(rows, metric)
    return f"bootstrap 95% CI positive/negative/spans 0: {pos}/{neg}/{mixed} seeds"


def generate_report(
    per_seed: list[dict[str, Any]],
    cross: list[dict[str, Any]],
    report_path: Path,
    *,
    source_commit: str,
    analysis_code_commit: str,
    plot_paths: list[Path],
) -> None:
    cross_by_key = {(r["dataset"], r["modality"], r["similarity_space"]): r for r in cross}
    seed_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in per_seed:
        seed_by_key.setdefault((row["dataset"], row["modality"], row["similarity_space"]), []).append(row)
    lines = [
        "# P0.1+ Utility Base-Rate and Enrichment Audit", "",
        "## 1. Scope", "",
        f"- source branch / commit: `exp/problem_validation` / `{source_commit}`",
        f"- analysis branch / code commit: `exp/problem_validation_p01plus` / `{analysis_code_commit}`",
        "- analysis uses only the 15 existing `edge_analysis.pt` files; no training or model selection was run.",
        "- no test files, test labels, or test metrics were accessed; no test evaluation was performed.",
        "- each artifact's exact existing sampled validation relation population was reused. No relation population was regenerated.",
        "- all rates describe the fixed sampled validation relation population (at most 64 sampled relations per target), not every edge in each graph.",
        "- utility definitions are unchanged: `U > 0` beneficial, `U < 0` harmful, and exact floating-point `U == 0` zero; no epsilon was applied.",
        "- quintiles use the complete dataset × seed × modality relation set. Stable mergesort followed by balanced `array_split` matches the original P0.1 quantile-figure rule; exact ties crossing boundaries are broken by artifact row order.",
        "- uncertainty intervals are 1,000 target-node bootstrap replicates (seed 42); every selected target carries all of its sampled relations. Three-seed means ± population SD are reported separately.",
        "",
        "## 2. Why This Supplement Is Needed", "",
        "A conditional beneficial rate alone establishes only that beneficial relations exist in a similarity region. It does not establish enrichment: if the overall beneficial base rate is higher, the region is depleted relative to the population. This audit therefore reports `P(U > 0 | Qq) - P(U > 0)` (lift), and analogously for harmful utility, beside both conditional and overall rates.",
        "",
        "## 3. Overall Utility Base Rates", "",
        "Mean ± population SD over seeds 42, 43, and 44. CE is primary; margin is included as a robustness view. Similarity space does not affect these overall rates, so each dataset × modality appears once.",
        "",
        "| Dataset | Modality | CE beneficial | CE harmful | CE zero | Margin beneficial | Margin harmful | Margin zero |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for modality in MODALITIES:
            row = cross_by_key[(dataset, modality.title(), "probe")]
            lines.append(
                f"| {dataset} | {modality.title()} | "
                + " | ".join(_fmt_agg(row, metric) for metric in (
                    "overall_beneficial_rate", "overall_harmful_rate", "overall_zero_rate",
                    "margin_overall_beneficial_rate", "margin_overall_harmful_rate", "margin_overall_zero_rate",
                )) + " |"
            )

    lines.extend(["", "## 4. Probe-Similarity Quintile Utility Profiles", "",
                  "Cells are mean ± population SD across seeds; rates use CE utility. Each table reports the conditional beneficial and harmful rates within that quintile.", ""])
    for dataset in DATASETS:
        lines.extend([f"### {dataset}", "", "| Modality | Outcome | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---:|---:|---:|---:|---:|"])
        for modality in MODALITIES:
            row = cross_by_key[(dataset, modality.title(), "probe")]
            for outcome in ("beneficial", "harmful"):
                values = " | ".join(_fmt_agg(row, f"q{q}_{outcome}_rate") for q in range(1, 6))
                lines.append(f"| {modality.title()} | {outcome} | {values} |")
        lines.append("")

    lines.extend(["## 5. Enrichment Relative to Base Rate", "",
                  "Lift is a conditional rate minus its overall base rate. Positive lift is enrichment; negative lift is depletion. Values are mean ± population SD over the three seeds. Run-specific target-node bootstrap intervals remain separate in the per-seed CSV.", "",
                  "| Dataset | Modality | Low-sim beneficial lift (Q1) | High-sim harmful lift (Q5) | High-sim beneficial lift (Q5) | Low-sim harmful lift (Q1) |", "|---|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for modality in MODALITIES:
            row = cross_by_key[(dataset, modality.title(), "probe")]
            lines.append(f"| {dataset} | {modality.title()} | " + " | ".join(
                _fmt_agg(row, metric) for metric in (
                    "q1_beneficial_lift", "q5_harmful_lift", "q5_beneficial_lift", "q1_harmful_lift",
                )
            ) + " |")

    lines.extend(["", "## 6. Similarity Stratification Strength", "",
                  "Range is `max_q P(sign | Qq) - min_q P(sign | Qq)`; quantile Spearman is descriptive over five ordered bins and is not tested for significance. The report uses probe similarity; raw values are available in the cross-dataset CSV.", "",
                  "| Dataset | Modality | Beneficial range | Harmful range | Beneficial quantile Spearman | Harmful quantile Spearman |", "|---|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for modality in MODALITIES:
            row = cross_by_key[(dataset, modality.title(), "probe")]
            lines.append(f"| {dataset} | {modality.title()} | " + " | ".join(
                _fmt_agg(row, metric) for metric in (
                    "beneficial_stratification_range", "harmful_stratification_range",
                    "beneficial_quantile_spearman", "harmful_quantile_spearman",
                )
            ) + " |")

    lines.extend(["", "## 7. Raw vs Task-Aware Similarity", "",
                  "The per-seed and cross-dataset CSVs retain both raw semantic and frozen-probe similarity, for Text and Visual separately. Compare their beneficial/harmful ranges, Q1/Q5 lifts, and five-bin Spearman values there. Differences are dataset- and modality-specific; this audit does not force raw and probe similarity to agree.", "",
                  "## 8. Dataset-Specific Interpretation", ""])
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        for modality in MODALITIES:
            crossrow = cross_by_key[(dataset, modality.title(), "probe")]
            runrows = seed_by_key[(dataset, modality.title(), "probe")]
            low_b = crossrow["q1_beneficial_lift_mean"]
            high_h = crossrow["q5_harmful_lift_mean"]
            b_range = crossrow["beneficial_stratification_range_mean"]
            h_range = crossrow["harmful_stratification_range_mean"]
            b_rho = crossrow["beneficial_quantile_spearman_mean"]
            h_rho = crossrow["harmful_quantile_spearman_mean"]
            base = crossrow["overall_beneficial_rate_mean"]
            low_rate = crossrow["q1_beneficial_rate_mean"]
            lines.append(
                f"- {modality.title()}: probe Q1 beneficial `{_fmt(low_rate)}` vs overall beneficial base `{_fmt(base)}` "
                f"(lift `{_fmt(low_b)}`); Q5 harmful lift `{_fmt(high_h)}`. Beneficial/harmful ranges are "
                f"`{_fmt(b_range)}` / `{_fmt(h_range)}`, with quantile Spearman `{_fmt(b_rho)}` / `{_fmt(h_rho)}`. "
                f"Node-bootstrap Q1 beneficial lift: {_sign_summary(runrows, 'q1_beneficial_lift')}; "
                f"Q5 harmful lift: {_sign_summary(runrows, 'q5_harmful_lift')}."
            )
        lines.append("")

    probe_rows = [row for row in per_seed if row["similarity_space"] == "probe"]
    q1_nonzero = sum(float(row["q1_beneficial_rate"]) > 0.0 for row in probe_rows)
    q1_lifts = [cross_by_key[(dataset, modality.title(), "probe")]["q1_beneficial_lift_mean"] for dataset in DATASETS for modality in MODALITIES]
    q5_harm_lifts = [cross_by_key[(dataset, modality.title(), "probe")]["q5_harmful_lift_mean"] for dataset in DATASETS for modality in MODALITIES]
    q1_pos, q1_neg = sum(value > 0.0 for value in q1_lifts), sum(value < 0.0 for value in q1_lifts)
    q5h_pos, q5h_neg = sum(value > 0.0 for value in q5_harm_lifts), sum(value < 0.0 for value in q5_harm_lifts)
    q1_ci_pos = sum(_ci_sign_counts(seed_by_key[(dataset, modality.title(), "probe")], "q1_beneficial_lift")[0] == 3 for dataset in DATASETS for modality in MODALITIES)
    q5h_ci_pos = sum(_ci_sign_counts(seed_by_key[(dataset, modality.title(), "probe")], "q5_harmful_lift")[0] == 3 for dataset in DATASETS for modality in MODALITIES)
    lines.extend(["## 9. Revised Interpretation of P0.1", "",
                  f"1. **Does low-sim beneficial existence remain true?** Yes: `P(U > 0 | Q1) > 0` in {q1_nonzero}/{len(probe_rows)} probe-similarity dataset × seed × modality rows. This is counterexample existence in the fixed sampled population.",
                  f"2. **Is low-sim beneficial enriched or depleted?** Across 10 dataset × modality mean lifts, Q1 beneficial lift is positive in {q1_pos} and negative in {q1_neg}; all-three-seed node-bootstrap CIs are positive in {q1_ci_pos}/10 comparisons. Interpret each row in Section 5 and its run-specific intervals in the CSV.",
                  f"3. **Are high-sim harmful relations enriched?** Q5 harmful lift is positive in {q5h_pos} and negative in {q5h_neg} of 10 dataset × modality mean comparisons; all-three-seed node-bootstrap CIs are positive in {q5h_ci_pos}/10. A nonzero Q5 harmful rate by itself is not enrichment.",
                  "4. **Where is similarity informative?** Use the sign-rate ranges and five-bin monotonic profiles in Section 6. These are descriptive properties, not significance claims.",
                  "5. **Where is similarity insufficient?** Flat or seed/modality-inconsistent profiles mean similarity does not consistently separate beneficial from harmful sampled relations, even when some endpoint lift is present.",
                  "", "### Requested dataset regime labels (descriptive only)", "",
                  "These relative labels use observed probe-CE profiles across the five datasets. `Strong` is assigned only to a dataset with the largest non-weak mean sign-rate range composite among datasets whose beneficial-quantile Spearman direction is shared by both modalities and all three seeds. `Weak` is the dataset with the smallest mean composite range. Remaining datasets are `intermediate`. If no dataset meets the consistency condition for `Strong`, none is labeled strong. No threshold test or significance claim is implied.", ""])
    # Assign the regime from observed per-seed ranges and directions, not dataset names.
    regime_scores: list[tuple[str, float, bool]] = []
    for dataset in DATASETS:
        ds_rows = [r for r in per_seed if r["dataset"] == dataset and r["similarity_space"] == "probe"]
        ranges = [float(r[metric]) for r in ds_rows for metric in ("beneficial_stratification_range", "harmful_stratification_range")]
        score = float(np.mean(ranges))
        directions: list[int] = []
        for modality in MODALITIES:
            for row in seed_by_key[(dataset, modality.title(), "probe")]:
                rho = row["beneficial_quantile_spearman"]
                directions.append(int(np.sign(rho)) if math.isfinite(float(rho)) else 0)
        nonzero = [direction for direction in directions if direction != 0]
        consistent = bool(nonzero) and len(nonzero) == len(directions) and all(d == nonzero[0] for d in nonzero)
        regime_scores.append((dataset, score, consistent))
    ranked = sorted(regime_scores, key=lambda item: item[1])
    weak_name = ranked[0][0]
    strong_candidates = [item for item in ranked[1:] if item[2]]
    strong_name = max(strong_candidates, key=lambda item: item[1])[0] if strong_candidates else None
    for dataset, score, consistent in regime_scores:
        if dataset == weak_name:
            label = "weak similarity regime"
        elif dataset == strong_name:
            label = "strong similarity-sign regime"
        else:
            label = "intermediate similarity regime"
        lines.append(f"- {dataset}: **{label}** (mean probe sign-rate range composite `{score:.3f}`; monotonic direction shared across modality × seed: `{str(consistent).lower()}`).")

    lines.extend(["", "## 10. Implications for P0.2", "",
                  "This audit can prioritize controls for a later, separately specified P0.2 comparison: datasets with little or inconsistent quantile separation are useful tests of a scalar similarity-reliability baseline versus a relation-conditioned alternative, while datasets with clearer separation test whether a learned method adds value beyond similarity alone. Freeze the dataset-specific interpretation and evaluation protocol before any P0.2 implementation. No P0.2 work was started here.",
                  "",
                  "## 11. What This Still Does Not Prove", "",
                  "This audit does not prove that a scalar learned relation is insufficient; vector relation state is necessary; semantic transformation is necessary; MoE, basis routing, or FiLM is necessary; or context heterogeneity is established. It reports only how the existing sampled validation relation utilities are distributed across observed similarity quintiles.",
                  "",
                  "## Figures", ""])
    for path in plot_paths:
        lines.append(f"- [{path.name}](../../results/problem_validation/p01plus/plots/{path.name})")
    lines.extend(["", "## Result Files", "",
                  "- `results/problem_validation/p01plus/p01plus_per_seed.csv`: all dataset × seed × modality × similarity-space rows; includes CE and margin profiles and run-specific bootstrap intervals.",
                  "- `results/problem_validation/p01plus/p01plus_cross_dataset.csv`: mean ± population SD across seeds.",
                  "- Node-bootstrap intervals are per seed and are not combined with three-seed SD.", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def summarize(
    input_root: Path = OUTPUT_ROOT,
    output_root: Path = ROOT / "results/problem_validation/p01plus",
    *,
    source_commit: str,
    analysis_code_commit: str,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 42,
) -> tuple[Path, Path, Path]:
    artifacts = load_existing_edge_artifacts(input_root)
    if len(artifacts) != len(DATASETS) * len(SEEDS):
        raise AssertionError(f"Expected 15 edge artifacts, got {len(artifacts)}")
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            rows.extend(analyze_run(
                dataset,
                seed,
                artifacts[(dataset, seed)],
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ))
    if len(rows) != 5 * 3 * 2 * 2:
        raise AssertionError(f"Expected 60 per-seed result rows, got {len(rows)}")
    cross = aggregate_seeds(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_root / "p01plus_per_seed.csv"
    cross_path = output_root / "p01plus_cross_dataset.csv"
    write_csv(per_seed_path, rows)
    write_csv(cross_path, cross)
    plot_dir = output_root / "plots"
    plot_paths = [make_dataset_plot(dataset, rows, plot_dir) for dataset in DATASETS]
    report_path = ROOT / "docs/problem_validation/p01plus_report.md"
    generate_report(
        rows,
        cross,
        report_path,
        source_commit=source_commit,
        analysis_code_commit=analysis_code_commit,
        plot_paths=plot_paths,
    )
    return per_seed_path, cross_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P0.1 utility base rates and similarity-quintile enrichment.")
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/problem_validation/p01plus")
    parser.add_argument("--source-commit", default="c56f6589450ab0c7abf3960f0e7fb2996db383d6")
    parser.add_argument("--analysis-code-commit", default="unknown (supply the implementation commit)")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    # The bootstrap uses many short vector operations; large BLAS pools make
    # these much slower through oversubscription. Keep this deterministic CPU
    # analysis single-threaded at the BLAS layer.
    with threadpool_limits(limits=1, user_api="blas"):
        paths = summarize(
            args.input_root,
            args.output_root,
            source_commit=args.source_commit,
            analysis_code_commit=args.analysis_code_commit,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
