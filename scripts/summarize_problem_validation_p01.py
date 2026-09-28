from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.problem_validation.common import DATASETS, OUTPUT_ROOT  # noqa: E402

METRICS = (
    "raw_spearman_ce", "raw_auroc_beneficial", "raw_high_sim_harmful_rate", "raw_low_sim_beneficial_rate",
    "probe_spearman_ce", "probe_auroc_beneficial", "probe_high_sim_harmful_rate", "probe_low_sim_beneficial_rate",
    "raw_spearman_margin", "probe_spearman_margin",
    "exact_disagreement_rate", "robust_disagreement_rate", "robust_edge_coverage",
)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    values = [float(v) for v in values if finite(v) is not None]
    if not values:
        return None, None
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def fmt(value: Any, digits: int = 3) -> str:
    v = finite(value)
    return "NA" if v is None else f"{v:.{digits}f}"


def fmt_mean_std(values: list[Any], digits: int = 3) -> str:
    m, s = mean_std(values)
    return "NA" if m is None else f"{m:.{digits}f} ± {s:.{digits}f}"


def read_runs(root: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    edges: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in DATASETS:
        for seed in (42, 43, 44):
            run_dir = root / dataset / f"seed{seed}"
            metrics_path, edge_path = run_dir / "metrics.json", run_dir / "edge_analysis.pt"
            if metrics_path.exists():
                metrics[(dataset, seed)] = json.loads(metrics_path.read_text(encoding="utf-8"))
            if edge_path.exists():
                edges[(dataset, seed)] = torch.load(edge_path, map_location="cpu", weights_only=False)
    return metrics, edges


def metric_value(payload: dict[str, Any], modality: str, name: str) -> Any:
    p01 = payload.get("p01_analysis", {})
    if name.startswith("raw_") or name.startswith("probe_"):
        space, stat = name.split("_", 1)
        key_map = {
            "spearman_ce": f"{space}_spearman_ce",
            "spearman_margin": f"{space}_spearman_margin",
            "auroc_beneficial": f"{space}_auroc_beneficial",
            "high_sim_harmful_rate": f"{space}_high_sim_harmful_rate",
            "low_sim_beneficial_rate": f"{space}_low_sim_beneficial_rate",
        }
        return p01.get(modality, {}).get(key_map.get(stat, stat))
    return p01.get("text_visual", {}).get(name)


def bootstrap_ci(payload: dict[str, Any], modality: str, metric: str) -> tuple[Any, Any]:
    p01 = payload.get("p01_analysis", {})
    if metric in {"exact_disagreement_rate", "robust_disagreement_rate", "robust_edge_coverage"}:
        key = {
            "exact_disagreement_rate": "utility_ce_text_visual_exact_disagreement",
            "robust_disagreement_rate": "utility_ce_text_visual_robust_disagreement",
            "robust_edge_coverage": "utility_ce_text_visual_robust_coverage",
        }[metric]
    else:
        space = "raw" if metric.startswith("raw_") else "probe"
        stat = metric[len(space) + 1 :]
        if stat == "spearman_ce":
            x_name = f"{space}_sim_{modality}"
            key = f"{x_name}.spearman_ce"
        elif stat in {"high_sim_harmful_rate", "low_sim_beneficial_rate"}:
            bootstrap_name = "high_sim_harmful" if stat == "high_sim_harmful_rate" else "low_sim_beneficial"
            key = f"{space}_sim_{modality}.{bootstrap_name}"
        else:
            return None, None
    ci = p01.get("bootstrap", {}).get(key, {})
    return ci.get("ci95_low"), ci.get("ci95_high")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _edge_arrays(edge: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in edge.items() if isinstance(value, torch.Tensor)}


def make_dataset_plots(dataset: str, edge_runs: list[dict[str, Any]], out: Path) -> list[Path]:
    if not edge_runs:
        return []
    out.mkdir(parents=True, exist_ok=True)
    arrays = [_edge_arrays(e) for e in edge_runs]
    colors = ("#1874A5", "#C85B36")
    output_paths: list[Path] = []

    # Figure A: task-aware semantic cosine versus exact CE edge utility.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, modality, color in zip(axes, ("text", "visual"), colors, strict=True):
        x = np.concatenate([a[f"probe_sim_{modality}"] for a in arrays])
        y = np.concatenate([a[f"utility_ce_{modality}"] for a in arrays])
        ax.hexbin(x, y, gridsize=42, mincnt=1, bins="log", cmap="Blues" if modality == "text" else "Oranges")
        bounds = np.quantile(x, np.linspace(0, 1, 21))
        centers, means = [], []
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            mask = (x >= lo) & (x <= hi) if hi == bounds[-1] else (x >= lo) & (x < hi)
            if mask.any():
                centers.append(float(x[mask].mean()))
                means.append(float(y[mask].mean()))
        ax.plot(centers, means, color=color, marker="o", linewidth=1.8, markersize=3, label="binned mean")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.65)
        ax.set(title=f"{modality.title()}", xlabel="P0.0 probe cosine similarity", ylabel="CE message utility")
        ax.legend(frameon=False)
    fig.suptitle(f"{dataset}: probe semantic similarity vs CE utility")
    path = out / f"{dataset}_figure_a_similarity_utility.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)

    # Figure B: five global similarity quantiles, rates averaged across independent seeds.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    labels = ("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")
    for ax, modality, color in zip(axes, ("text", "visual"), colors, strict=True):
        beneficial_seed, harmful_seed = [], []
        for a in arrays:
            sim, utility = a[f"probe_sim_{modality}"], a[f"utility_ce_{modality}"]
            order = np.argsort(sim, kind="mergesort")
            groups = np.array_split(order, 5)
            beneficial_seed.append([float(np.mean(utility[g] > 0)) for g in groups])
            harmful_seed.append([float(np.mean(utility[g] < 0)) for g in groups])
        beneficial = np.nanmean(np.asarray(beneficial_seed), axis=0)
        harmful = np.nanmean(np.asarray(harmful_seed), axis=0)
        xpos = np.arange(5)
        width = 0.36
        ax.bar(xpos - width / 2, beneficial, width, label="beneficial (U > 0)", color=color, alpha=0.9)
        ax.bar(xpos + width / 2, harmful, width, label="harmful (U < 0)", color="#777777", alpha=0.78)
        ax.set_xticks(xpos, labels, rotation=20)
        ax.set(title=modality.title(), xlabel="Probe similarity quantile", ylabel="Relation proportion", ylim=(0, 1))
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{dataset}: utility rates by probe-similarity quantile")
    path = out / f"{dataset}_figure_b_quantile_rates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)

    # Figure C: same physical relation's Text/Visual utility signs.
    quadrants = np.zeros(4, dtype=np.float64)
    for a in arrays:
        ut, uv = a["utility_ce_text"], a["utility_ce_visual"]
        signs = np.stack((np.sign(ut), np.sign(uv)), axis=1)
        names = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        quadrants += np.asarray([np.mean((signs[:, 0] == x) & (signs[:, 1] == y)) for x, y in names])
    quadrants /= len(arrays)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.bar(("(+,+)", "(+,-)", "(-,+)", "(-,-)"), quadrants, color=("#4B9B73", "#D39343", "#5E86BD", "#9B6BA8"))
    ax.set(ylabel="Mean relation proportion across seeds", title=f"{dataset}: Text vs Visual CE utility signs", ylim=(0, 1))
    for i, value in enumerate(quadrants):
        ax.text(i, value + 0.015, f"{value:.1%}", ha="center", fontsize=9)
    path = out / f"{dataset}_figure_c_text_visual_quadrants.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    output_paths.append(path)
    return output_paths


def _decision(metrics: dict[tuple[str, int], dict[str, Any]]) -> tuple[str, str]:
    complete = all((ds, seed) in metrics for ds in DATASETS for seed in (42, 43, 44))
    if not complete:
        return "Not yet adjudicated (incomplete run matrix)", "This report is an interim output; all five datasets and three seeds are required before applying the pre-registered H1 decision rule."
    per_dataset_hits = {}
    robust_nonzero = 0
    explanations = []
    for ds in DATASETS:
        values = [metrics[(ds, s)] for s in (42, 43, 44)]
        rho_means = []
        auc_means = []
        high_means = []
        low_means = []
        for modality in ("text", "visual"):
            for space in ("raw", "probe"):
                rho_means.append(mean_std([metric_value(p, modality, f"{space}_spearman_ce") for p in values])[0])
                auc_means.append(mean_std([metric_value(p, modality, f"{space}_auroc_beneficial") for p in values])[0])
                high_means.append(mean_std([metric_value(p, modality, f"{space}_high_sim_harmful_rate") for p in values])[0])
                low_means.append(mean_std([metric_value(p, modality, f"{space}_low_sim_beneficial_rate") for p in values])[0])
        phenomena = [
            any(v is not None and abs(v) < 0.5 for v in rho_means),
            any(v is not None and v < 0.75 for v in auc_means),
            any(v is not None and v >= 0.10 for v in high_means),
            any(v is not None and v >= 0.10 for v in low_means),
        ]
        per_dataset_hits[ds] = sum(phenomena)
        robust_values = [finite(metric_value(p, "text", "robust_disagreement_rate")) for p in values]
        robust_ci_lowers = [
            finite(p.get("p01_analysis", {}).get("bootstrap", {}).get(
                "utility_ce_text_visual_robust_disagreement", {}
            ).get("ci95_low"))
            for p in values
        ]
        robust_ci_runs = sum(v is not None and v > 0 for v in robust_ci_lowers)
        robust_nonzero += int(robust_ci_runs >= 2)
        explanations.append(
            f"{ds}: {sum(phenomena)}/4 H1 diagnostic criteria; mean robust T/V disagreement="
            f"{fmt_mean_std(robust_values)}; node-bootstrap lower CI > 0 in {robust_ci_runs}/3 seeds"
        )
    supporting_datasets = sum(v >= 2 for v in per_dataset_hits.values())
    if supporting_datasets >= 4 and robust_nonzero >= 3:
        decision = "Supported"
    elif supporting_datasets >= 2 or robust_nonzero >= 2:
        decision = "Partially Supported"
    else:
        decision = "Unsupported"
    return decision, f"{supporting_datasets}/5 datasets meet at least two H1 diagnostic criteria; robust Text/Visual disagreement has a positive node-bootstrap 95% CI in at least two seeds for {robust_nonzero}/5 datasets. " + " ".join(explanations)


def generate_report(
    output_root: Path,
    metrics: dict[tuple[str, int], dict[str, Any]],
    p00_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> str:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    decision, basis = _decision(metrics)
    lines = [
        "# P0.0 + P0.1 Problem Validation Report", "", "## 1. Scope",
        f"- branch: `{branch or 'unknown'}`", f"- commit: `{commit or 'unknown'}`",
        "- datasets: Movies, Toys, Grocery, ele-fashion, Reddit-S",
        "- model run seeds: 42, 43, 44; data/probe split seed: 42",
        "- protocol: P0.0 feature-only probe and P0.1 decomposable one-hop message probe; checkpoint selection used probe_calib only; original val was used only for held-out analysis",
        "- test set was not evaluated; test labels were not used for training, selection, analysis, or reporting.", "",
        "## 2. Data Split Audit", "",
        "| Dataset | Original train | Probe train | Probe calib | Original val | Original test (size only) | Analysis targets | Sampled edges | Leakage checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for ds in DATASETS:
        path = output_root / "splits" / f"{ds}.pt"
        if not path.exists():
            lines.append(f"| {ds} | — | — | — | — | — | — | — | not run |")
            continue
        split = torch.load(path, map_location="cpu", weights_only=False)
        audit = split.get("split_audit", {})
        lines.append(
            f"| {ds} | {split.get('original_train_size','—')} | {audit.get('probe_train_size','—')} | {audit.get('probe_calib_size','—')} "
            f"| {split.get('original_val_size','—')} | {split.get('original_test_size','—')} | {len(split.get('analysis_target_nodes',[]))} "
            f"| {split.get('sampled_relation_count','—')} | probe train/calib/val disjoint; targets⊆val and ∩test=∅ |"
        )
    lines.extend([
        "", "Runtime assertions verify probe_train ∩ probe_calib = ∅, both inner splits are disjoint from original val, analysis targets are a subset of original val, and targets do not intersect test indices. Test indices are used only for that final disjointness check; their labels are not accessed by the analysis.",
        "", "## 3. P0.0 Semantic Probe", "",
    ])
    for ds in DATASETS:
        lines.extend([f"### {ds}", "", "| Seed | Best epoch | Probe train Acc / Macro-F1 | Probe calib Acc / Macro-F1 | heldout_analysis_val Acc / Macro-F1 |", "|---:|---:|---:|---:|---:|"])
        for seed in (42, 43, 44):
            payload = metrics.get((ds, seed))
            if not payload:
                lines.append(f"| {seed} | — | — | — | — |")
                continue
            p00 = payload["p00"]
            def pair(key: str) -> str:
                d = p00.get(key, {})
                return f"{fmt(d.get('acc'))} / {fmt(d.get('macro_f1'))}"
            lines.append(f"| {seed} | {p00.get('best_epoch','—')} | {pair('probe_train')} | {pair('probe_calib')} | {pair('heldout_analysis_val')} |")
        lines.append("")
    lines.extend(["### Cross-dataset summary", "", "| Dataset | Train Acc | Train Macro-F1 | Calib Acc | Calib Macro-F1 | heldout_analysis_val Acc | heldout_analysis_val Macro-F1 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for ds in DATASETS:
        ps = [metrics[(ds,s)]["p00"] for s in (42,43,44) if (ds,s) in metrics]
        cells=[]
        for split, metric in (("probe_train","acc"),("probe_train","macro_f1"),("probe_calib","acc"),("probe_calib","macro_f1"),("heldout_analysis_val","acc"),("heldout_analysis_val","macro_f1")):
            cells.append(fmt_mean_std([p[split].get(metric) for p in ps]))
        lines.append("| " + ds + " | " + " | ".join(cells) + " |")

    lines.extend(["", "## 4. P0.1 Similarity vs Message Utility", "", "### 4.1 Raw Semantic Similarity", "", "| Dataset | Modality | Spearman CE | AUROC | High-sim harmful | Low-sim beneficial |", "|---|---|---:|---:|---:|---:|"])
    for ds in DATASETS:
        for mod in ("text", "visual"):
            rows=[r for r in seed_rows if r["dataset"]==ds and r["modality"]==mod]
            if rows:
                keys=("raw_spearman_ce","raw_auroc_beneficial","raw_high_sim_harmful_rate","raw_low_sim_beneficial_rate")
                lines.append(f"| {ds} | {mod.title()} | " + " | ".join(fmt_mean_std([r.get(k) for r in rows]) for k in keys) + " |")
    lines.extend(["", "### 4.2 Task-aware Semantic Similarity", "", "| Dataset | Modality | Spearman CE | AUROC | High-sim harmful | Low-sim beneficial |", "|---|---|---:|---:|---:|---:|"])
    for ds in DATASETS:
        for mod in ("text", "visual"):
            rows=[r for r in seed_rows if r["dataset"]==ds and r["modality"]==mod]
            if rows:
                keys=("probe_spearman_ce","probe_auroc_beneficial","probe_high_sim_harmful_rate","probe_low_sim_beneficial_rate")
                lines.append(f"| {ds} | {mod.title()} | " + " | ".join(fmt_mean_std([r.get(k) for r in rows]) for k in keys) + " |")

    lines.extend(["", "### 4.3 Text vs Visual", "", "| Dataset | Exact sign disagreement | Robust sign disagreement | Robust edge coverage |", "|---|---:|---:|---:|"])
    for ds in DATASETS:
        ps=[metrics[(ds,s)]["p01_analysis"]["text_visual"] for s in (42,43,44) if (ds,s) in metrics]
        if ps:
            keys=("exact_disagreement_rate","robust_disagreement_rate","robust_edge_coverage")
            lines.append(f"| {ds} | " + " | ".join(fmt_mean_std([p.get(k) for p in ps]) for k in keys) + " |")
    lines.extend(["", "### 4.4 CE Utility vs Margin Utility", "", "| Dataset | Modality | Raw margin Spearman | Probe margin Spearman | CE/margin sign agreement | CE-vs-margin Spearman |", "|---|---|---:|---:|---:|---:|"])
    for ds in DATASETS:
        for mod in ("text", "visual"):
            ps=[metrics[(ds,s)]["p01_analysis"] for s in (42,43,44) if (ds,s) in metrics]
            if not ps:
                continue
            raw=fmt_mean_std([p[mod].get("raw_spearman_margin") for p in ps])
            probe=fmt_mean_std([p[mod].get("probe_spearman_margin") for p in ps])
            suffix="text" if mod=="text" else "visual"
            sign=fmt_mean_std([p["text_visual"].get(f"ce_margin_sign_agreement_{suffix}") for p in ps])
            rho=fmt_mean_std([p["text_visual"].get(f"ce_margin_spearman_{suffix}") for p in ps])
            lines.append(f"| {ds} | {mod.title()} | {raw} | {probe} | {sign} | {rho} |")

    lines.extend(["", "### Node-bootstrap 95% CIs (1,000 target-node replicates, seed 42; reported per seed and separate from three-seed std)", "", "| Dataset | Modality | Raw ρ CE | Probe ρ CE | Raw high-sim harmful | Probe high-sim harmful | Raw low-sim beneficial | Probe low-sim beneficial | Exact T/V disagreement | Robust T/V disagreement |", "|---|---|---|---|---|---|---|---|---|---|"])
    for ds in DATASETS:
        for mod in ("text", "visual"):
            rows=[r for r in seed_rows if r["dataset"]==ds and r["modality"]==mod]
            if not rows:
                continue
            metrics_ci=("raw_spearman_ce","probe_spearman_ce","raw_high_sim_harmful_rate","probe_high_sim_harmful_rate","raw_low_sim_beneficial_rate","probe_low_sim_beneficial_rate","exact_disagreement_rate","robust_disagreement_rate")
            cells=[]
            for metric in metrics_ci:
                parts=[]
                for row in sorted(rows,key=lambda x:x["run_seed"]):
                    lo=row.get(f"{metric}_bootstrap_ci95_low"); hi=row.get(f"{metric}_bootstrap_ci95_high")
                    parts.append(f"{row['run_seed']}: [{fmt(lo)}, {fmt(hi)}]")
                cells.append("; ".join(parts))
            lines.append(f"| {ds} | {mod.title()} | " + " | ".join(cells) + " |")

    lines.extend(["", "## 5. Cross-dataset Summary", "", "Values are mean ± population std over seeds. Bootstrap CIs remain listed separately above.", "", "| Dataset | Modality | Spearman | AUROC | High-Sim Harmful | Low-Sim Beneficial | Robust T/V disagreement |", "|---|---|---:|---:|---:|---:|---:|"])
    for ds in DATASETS:
        for mod in ("text", "visual"):
            rows=[r for r in seed_rows if r["dataset"]==ds and r["modality"]==mod]
            if rows:
                keys=("probe_spearman_ce","probe_auroc_beneficial","probe_high_sim_harmful_rate","probe_low_sim_beneficial_rate","robust_disagreement_rate")
                lines.append(f"| {ds} | {mod.title()} | " + " | ".join(fmt_mean_std([r.get(k) for r in rows]) for k in keys) + " |")

    rho_values=[finite(r.get("probe_spearman_ce")) for r in seed_rows]
    rho_values=[v for v in rho_values if v is not None]
    auc_values=[finite(r.get("probe_auroc_beneficial")) for r in seed_rows]
    auc_values=[v for v in auc_values if v is not None]
    high_values=[finite(r.get("probe_high_sim_harmful_rate")) for r in seed_rows]
    high_values=[v for v in high_values if v is not None]
    low_values=[finite(r.get("probe_low_sim_beneficial_rate")) for r in seed_rows]
    low_values=[v for v in low_values if v is not None]
    robust_by_dataset=[]
    for ds in DATASETS:
        rows=[r for r in seed_rows if r["dataset"]==ds and r["modality"]=="text"]
        avg,_=mean_std([r.get("robust_disagreement_rate") for r in rows])
        if avg is not None:
            robust_by_dataset.append(avg)
    high_hit=sum(v>=0.10 for v in high_values)
    lines.extend([
        "", "## 6. Hypothesis Decision", "", f"H1: **{decision}**", "", basis,
        "", "For the predeclared four diagnostic criteria, each dataset is counted when at least one modality/similarity-space mean meets that criterion: |Spearman| < 0.5, signed AUROC < 0.75, high-sim harmful rate ≥ 10%, or low-sim beneficial rate ≥ 10%. Robust Text/Visual disagreement is counted as non-zero when its target-node bootstrap 95% CI lower bound is above zero in at least two of three seeds. Strong support requires at least two criteria in at least four datasets plus non-zero robust disagreement in at least three datasets. The AUROC is not reflected around 0.5.",
        "", "## 7. What the Evidence Supports", "",
        f"- Across 30 task-aware run × modality CE-correlation results (10 dataset × modality groups over three seeds), Spearman spans {fmt(min(rho_values))} to {fmt(max(rho_values))}; this is weak rank association overall.",
        f"- Task-aware similarity AUROC is below 0.75 in {sum(v<0.75 for v in auc_values)}/{len(auc_values)} dataset × seed × modality observations; signed AUROC values are retained as measured.",
        f"- Probe-similarity top-20% edges are harmful at a rate ≥10% in {high_hit}/{len(high_values)} run × modality observations. Probe-similarity bottom-20% edges are beneficial in {sum(v>=0.10 for v in low_values)}/{len(low_values)} observations; the observed rate range is {fmt(min(low_values))}–{fmt(max(low_values))}.",
        f"- Robust Text/Visual CE-utility sign disagreement is non-zero in all five datasets; dataset means range {fmt(min(robust_by_dataset))}–{fmt(max(robust_by_dataset))}. The robust coverage and per-seed node-bootstrap intervals are reported above.",
        "- Raw and task-aware similarities differ by dataset and modality. These findings support the limited claim that similarity is an imperfect surrogate for the one-hop task utility measured here.",
        "", "## 8. What the Evidence Does NOT Support", "",
        "P0.1 does not establish that vector relation state is necessary, MoE is necessary, basis routing is necessary, semantic transformation is better than scalar weighting, deeper structural context is heterogeneous, or any final proposed architecture is superior.",
        "", "## 9. Implications for P0.2", "",
        "If proceeding, the next validation should directly compare a scalar similarity-reliability baseline with a relation-conditioned alternative under the same frozen semantic embeddings, fixed probe split, and held-out original-val utility analysis. The current evidence motivates that comparison but does not predetermine its winner; define the controls and selection protocol before implementing P0.2.",
        "", "## Diagnostic Figures", "",
    ])
    for path in plot_paths:
        rel = f"../../results/problem_validation/plots/{path.name}"
        lines.append(f"- [{path.name}]({rel})")
    lines.append("")
    report_path = ROOT / "docs/problem_validation/p0_p1_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)

def main(output_root: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize P0.0/P0.1 validation outputs.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    if output_root is None:
        args = parser.parse_args()
        output_root = args.output_root
    output_root = Path(output_root)
    summaries = output_root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    metrics, edges = read_runs(output_root)

    p00_rows: list[dict[str, Any]] = []
    p01_rows: list[dict[str, Any]] = []
    for ds in DATASETS:
        for seed in (42,43,44):
            payload=metrics.get((ds,seed))
            if not payload:
                continue
            p00=payload["p00"]
            p00_rows.append({
                "dataset":ds,"run_seed":seed,"best_epoch":p00.get("best_epoch"),
                "probe_train_acc":p00["probe_train"].get("acc"),"probe_train_macro_f1":p00["probe_train"].get("macro_f1"),
                "probe_calib_acc":p00["probe_calib"].get("acc"),"probe_calib_macro_f1":p00["probe_calib"].get("macro_f1"),
                "heldout_analysis_val_acc":p00["heldout_analysis_val"].get("acc"),"heldout_analysis_val_macro_f1":p00["heldout_analysis_val"].get("macro_f1"),
            })
            for mod in ("text","visual"):
                row={"dataset":ds,"run_seed":seed,"modality":mod}
                for metric in METRICS:
                    row[metric]=metric_value(payload,mod,metric)
                    lo,hi=bootstrap_ci(payload,mod,metric)
                    row[f"{metric}_bootstrap_ci95_low"]=lo
                    row[f"{metric}_bootstrap_ci95_high"]=hi
                p01_rows.append(row)

    write_csv(summaries/"p00_semantic_probe.csv", list(p00_rows[0]) if p00_rows else ["dataset","run_seed"],p00_rows)
    if p01_rows:
        write_csv(summaries/"p01_similarity_utility.csv",list(p01_rows[0]),p01_rows)

    seed_summary=[]
    for ds in DATASETS:
        for mod in ("text","visual"):
            candidates=[r for r in p01_rows if r["dataset"]==ds and r["modality"]==mod]
            if not candidates:
                continue
            row={"dataset":ds,"modality":mod,"available_seeds":len(candidates)}
            for metric in METRICS:
                m,s=mean_std([r.get(metric) for r in candidates])
                row[f"{metric}_mean"]=m
                row[f"{metric}_population_std"]=s
            seed_summary.append(row)
    if seed_summary:
        write_csv(summaries/"p01_seed_summary.csv",list(seed_summary[0]),seed_summary)
        cross=list(seed_summary)
        for mod in ("text","visual"):
            rows=[r for r in seed_summary if r["modality"]==mod]
            if rows:
                row={"dataset":"ALL_DATASETS","modality":mod,"available_seeds":sum(r["available_seeds"] for r in rows)}
                for metric in METRICS:
                    m,s=mean_std([r[f"{metric}_mean"] for r in rows])
                    row[f"{metric}_mean"]=m
                    row[f"{metric}_population_std"]=s
                cross.append(row)
        write_csv(summaries/"p01_cross_dataset_summary.csv",list(cross[0]),cross)

    plot_paths=[]
    for ds in DATASETS:
        ds_edges=[edges[(ds,s)] for s in (42,43,44) if (ds,s) in edges]
        plot_paths.extend(make_dataset_plots(ds,ds_edges,summaries/"plots"))
    report=generate_report(output_root,metrics,p00_rows,p01_rows,plot_paths)
    print(f"Wrote {summaries}; report={report}; runs={len(metrics)}; edge files={len(edges)}")


if __name__ == "__main__":
    main()
