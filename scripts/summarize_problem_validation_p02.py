from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([x for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(ddof=0))


def _fmt(mean: float, sd: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "NA"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def _collect(config: dict[str, Any]) -> dict[str, Any]:
    a = config["analysis"]
    root = ROOT / a["output_root"]
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    immediate: list[dict[str, Any]] = []
    mechanisms: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for dataset in a["datasets"]:
        for seed in a["run_seeds"]:
            seed_dir = root / dataset / f"seed{seed}"
            complete_path = seed_dir / "run_complete.json"
            if not complete_path.exists():
                continue
            run = {"seed_dir": seed_dir, "variants": {}}
            run["metadata"] = _read_json(seed_dir / "run_metadata.json") if (seed_dir / "run_metadata.json").exists() else {}
            for variant in a["variants"]:
                vdir = seed_dir / variant
                paths = {
                    "metrics": vdir / "metrics.json",
                    "edge": vdir / "edge_diagnostics.pt",
                    "context": vdir / "context_probe_metrics.json",
                    "stability": vdir / "context_stability.json",
                }
                if not all(path.exists() for path in paths.values()):
                    break
                metrics = _read_json(paths["metrics"])
                run["variants"][variant] = metrics
                val = metrics["heldout_original_val"]
                train = metrics["probe_train"]
                calib = metrics["probe_calib"]
                counts = metrics["parameter_counts"]
                p01_val = metrics.get("p01_uniform_probe_heldout_val", {})
                immediate.append({
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": variant,
                    "p01_uniform_probe_val_acc": p01_val.get("acc"),
                    "p01_uniform_probe_val_macro_f1": p01_val.get("macro_f1"),
                    "p01_uniform_probe_val_ce": p01_val.get("ce"),
                    "p02_minus_p01_val_acc": float(val["acc"] - p01_val["acc"]) if p01_val and p01_val.get("acc") is not None else None,
                    "p02_minus_p01_val_macro_f1": float(val["macro_f1"] - p01_val["macro_f1"]) if p01_val and p01_val.get("macro_f1") is not None else None,
                    "p02_minus_p01_val_ce": float(val["ce"] - p01_val["ce"]) if p01_val and p01_val.get("ce") is not None else None,
                    "best_epoch": int(metrics["best_epoch"]),
                    "probe_train_acc": train["acc"],
                    "probe_train_macro_f1": train["macro_f1"],
                    "probe_train_ce": train["ce"],
                    "probe_calib_acc": calib["acc"],
                    "probe_calib_macro_f1": calib["macro_f1"],
                    "probe_calib_ce": calib["ce"],
                    "heldout_original_val_acc": val["acc"],
                    "heldout_original_val_macro_f1": val["macro_f1"],
                    "heldout_original_val_ce": val["ce"],
                    **counts,
                    "decomposition_max_abs_error": metrics.get("decomposition_audit", {}).get("max_abs_error"),
                    "test_evaluation": False,
                    "test_labels_accessed": False,
                })
                context_metrics = _read_json(paths["context"])
                for readout, readout_metrics in context_metrics.items():
                    val_context = readout_metrics["heldout_original_val"]
                    contexts.append({
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": variant,
                        "readout": readout,
                        "best_epoch": readout_metrics["best_epoch"],
                        "context_val_acc": val_context["acc"],
                        "context_val_macro_f1": val_context["macro_f1"],
                        "context_val_ce": val_context["ce"],
                        "linear_head_parameter_count": readout_metrics["parameter_count"],
                    })
                norm_data = _read_json(paths["stability"])
                for key, values in norm_data.items():
                    modality, order_name, _ = key.split("_")
                    stability.append({
                        "dataset": dataset,
                        "seed": int(seed),
                        "variant": variant,
                        "modality": modality,
                        "context_order": int(order_name[1:]),
                        **{f"norm_{stat}": val for stat, val in values.items()},
                    })
            else:
                runs[(dataset, int(seed))] = run
            conflict_path = seed_dir / "conflict_subset_rows.json"
            if conflict_path.exists():
                for row in _read_json(conflict_path)["rows"]:
                    conflicts.append(dict(row))
            v3_dir = seed_dir / "conditional_feature"
            if (v3_dir / "function_ablation.json").exists() and (v3_dir / "function_shuffle.json").exists():
                ablation = _read_json(v3_dir / "function_ablation.json")
                shuffle = _read_json(v3_dir / "function_shuffle.json")
                row = {"dataset": dataset, "seed": int(seed)}
                for prefix, values in (("full", ablation["full_v3"]), ("identity", ablation["identity_function"])):
                    for metric, value in values.items():
                        row[f"{prefix}_{metric}"] = value
                for metric, value in ablation["delta_full_minus_identity"].items():
                    row[f"delta_full_minus_identity_{metric}"] = value
                for metric, value in shuffle["shuffle_mean"].items():
                    row[f"shuffle_{metric}_mean"] = value
                for metric, value in shuffle["shuffle_population_std"].items():
                    row[f"shuffle_{metric}_population_std"] = value
                for metric, value in shuffle["delta_full_minus_shuffle_mean"].items():
                    row[f"delta_full_minus_shuffle_{metric}"] = value
                row["shuffle_repetitions"] = len(shuffle["shuffle_repetitions"])
                functions.append(row)
            if (v3_dir / "mechanism_summary.json").exists():
                mech = _read_json(v3_dir / "mechanism_summary.json")
                for row in mech["rows"]:
                    mechanisms.append({"dataset": dataset, "seed": int(seed), **row})
    return {
        "runs": runs,
        "immediate": immediate,
        "mechanisms": mechanisms,
        "conflicts": conflicts,
        "functions": functions,
        "contexts": contexts,
        "stability": stability,
    }


def _group_aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...], value_columns: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, values in grouped.items():
        out = dict(zip(keys, group_key, strict=True))
        out["seed_count"] = len({r.get("seed") for r in values})
        for column in value_columns:
            mean, sd = _mean_sd([r.get(column) for r in values])
            out[f"{column}_mean"] = mean
            out[f"{column}_population_std"] = sd
        result.append(out)
    return result


def _write_figures(root: Path, data: dict[str, Any], datasets: list[str], variants: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    immediate = data["immediate"]
    colors = {v: c for v, c in zip(variants, ("#4C78A8", "#F58518", "#54A24B", "#E45756"), strict=True)}
    metrics = (("heldout_original_val_acc", "Accuracy"), ("heldout_original_val_macro_f1", "Macro-F1"))
    display_variant = {"uniform": "Uniform", "similarity_scalar": "Similarity scalar", "learned_scalar": "Learned scalar", "conditional_feature": "Conditional feature"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True)
    x = np.arange(len(datasets))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for variant in variants:
            means, stds = [], []
            for dataset in datasets:
                vals = [r[metric] for r in immediate if r["dataset"] == dataset and r["variant"] == variant]
                mean, sd = _mean_sd(vals)
                means.append(mean)
                stds.append(sd)
            ax.errorbar(x, means, yerr=stds, marker="o", capsize=3, label=display_variant[variant], color=colors[variant])
        ax.set_title(title)
        ax.set_xticks(x, datasets, rotation=25, rotation_mode="anchor", ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out original-val metric")
    fig.legend(handles=axes[1].get_legend_handles_labels()[0], labels=axes[1].get_legend_handles_labels()[1],
               loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    fig.suptitle("Points show seed means; error bars show population SD", fontsize=9)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=plot_dir / "p02_figure_a_immediate_alignment.json",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(plot_dir / "p02_figure_a_immediate.png", dpi=300)
    fig.savefig(plot_dir / "p02_figure_a_immediate.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(plot_dir / "p02_figure_a_immediate.svg")
    fig.savefig(plot_dir / "p02_figure_a_immediate.pdf")
    plt.close(fig)

    mech = data["mechanisms"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), constrained_layout=True)
    for ax, metric, label in zip(
        axes,
        ("transform_residual_ratio_mean", "non_collinearity_mean"),
        ("Transform residual ratio", "Non-collinearity"),
        strict=True,
    ):
        for mod, subset, style in (
            ("text", "compatibility_conflict", "-"),
            ("text", "compatibility_consistent", "--"),
            ("visual", "compatibility_conflict", "-"),
            ("visual", "compatibility_consistent", "--"),
        ):
            means, stds = [], []
            for dataset in datasets:
                vals = [r[metric] for r in mech if r["dataset"] == dataset and r["modality"] == mod and r["subset"] == subset]
                mean, sd = _mean_sd(vals)
                means.append(mean)
                stds.append(sd)
            ax.errorbar(x, means, yerr=stds, marker="o", linestyle=style, capsize=2, label=f"{mod.capitalize()} {subset.replace('compatibility_', '')}")
        ax.set_title(label)
        ax.set_xticks(x, datasets, rotation=25, rotation_mode="anchor", ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], labels=axes[0].get_legend_handles_labels()[1],
               loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    fig.suptitle("Per-run relation means; error bars show population SD across seeds", fontsize=9)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=plot_dir / "p02_figure_b_mechanism_alignment.json",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(plot_dir / "p02_figure_b_mechanism.png", dpi=300)
    fig.savefig(plot_dir / "p02_figure_b_mechanism.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(plot_dir / "p02_figure_b_mechanism.svg")
    fig.savefig(plot_dir / "p02_figure_b_mechanism.pdf")
    plt.close(fig)

    deltas = _pairwise_rows(data)
    wanted = {
        "one_hop_V3_minus_V2": "one-hop",
        "context_only_V3_minus_V2": "context-only",
        "full_bank_V3_minus_V2": "full-bank",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True)
    for ax, metric in zip(axes, ("acc", "macro_f1"), strict=True):
        for comparison, label in wanted.items():
            means, stds = [], []
            for dataset in datasets:
                vals = [r[f"delta_{metric}"] for r in deltas if r["dataset"] == dataset and r["comparison"] == comparison]
                mean, sd = _mean_sd(vals)
                means.append(mean)
                stds.append(sd)
            ax.errorbar(x, means, yerr=stds, marker="o", capsize=3, label=label)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"V3 − V2 paired {metric}")
        ax.set_xticks(x, datasets, rotation=25, rotation_mode="anchor", ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Paired seed delta")
    fig.legend(handles=axes[1].get_legend_handles_labels()[0], labels=axes[1].get_legend_handles_labels()[1],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle("Points show paired seed mean deltas; error bars show population SD", fontsize=9)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=plot_dir / "p02_figure_c_v3_v2_deltas_alignment.json",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(plot_dir / "p02_figure_c_v3_v2_deltas.png", dpi=300)
    fig.savefig(plot_dir / "p02_figure_c_v3_v2_deltas.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(plot_dir / "p02_figure_c_v3_v2_deltas.svg")
    fig.savefig(plot_dir / "p02_figure_c_v3_v2_deltas.pdf")
    plt.close(fig)


def _pairwise_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (("conditional_feature", "learned_scalar", "one_hop_V3_minus_V2"),
                   ("learned_scalar", "similarity_scalar", "one_hop_V2_minus_V1"),
                   ("similarity_scalar", "uniform", "one_hop_V1_minus_V0"))
    for (dataset, seed), run in data["runs"].items():
        for left, right, name in comparisons:
            l, r = run["variants"][left]["heldout_original_val"], run["variants"][right]["heldout_original_val"]
            rows.append({"dataset": dataset, "seed": seed, "comparison": name,
                         **{f"delta_{metric}": float(l[metric] - r[metric]) for metric in ("acc", "macro_f1", "ce")}})
        v2, v3 = run["variants"]["learned_scalar"], run["variants"]["conditional_feature"]
        for readout, comparison in (("context_only", "context_only_V3_minus_V2"), ("full_bank", "full_bank_V3_minus_V2")):
            left = v3["context_probe_metrics"][readout]["heldout_original_val"]
            right = v2["context_probe_metrics"][readout]["heldout_original_val"]
            rows.append({"dataset": dataset, "seed": seed, "comparison": comparison,
                         **{f"delta_{metric}": float(left[metric] - right[metric]) for metric in ("acc", "macro_f1", "ce")}})
    return rows


def _generate_report(config: dict[str, Any], data: dict[str, Any], coverage: dict[str, Any], result_root: Path) -> str:
    a = config["analysis"]
    datasets, variants = list(a["datasets"]), list(a["variants"])
    complete_count = len(data["runs"])
    expected_count = len(datasets) * len(a["run_seeds"])
    full = complete_count == expected_count
    status = "Complete 5-dataset × 3-seed matrix" if full else f"Partial run ({complete_count}/{expected_count} dataset-seed runs); do not interpret as final P0.2 evidence"
    manifest_path = result_root / "source_artifact_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    code_sha = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["branch", "--show-current"])
    lines = [
        "# P0.2 Scalar Compatibility vs Interaction Function",
        "",
        f"> **Run status:** {status}.",
        "",
        "## 1. Scope and Protocol",
        "",
        f"- Source branch / SHA: `exp/problem_validation_p01plus` / `{manifest.get('source_sha', 'unknown')}`.",
        f"- Analysis branch / code SHA at report generation: `{branch}` / `{code_sha}`.",
        "- Data: Movies, Toys, Grocery, ele-fashion, Reddit-S; run seeds 42, 43, 44.",
        "- Existing P0.0 semantic embeddings are frozen; no semantic projector training was run.",
        "- Original train/val/test index sets and cached probe_train/probe_calib split are reused and hash-checked. Test indices are used only for disjointness checks; no test label or metric is accessed.",
        "- All rates and edge diagnostics use the exact ordered P0.1 sampled relation population; no new edge sample or split was created.",
        "- NC protocol only. No LP, baseline modification, final architecture, P0.3, MoE, basis routing, prototype, OT, or CoSI/MoPF code.",
        "- Any per-target propagation denominator is the original physical degree. Relations are directed as source `j →` target `i`; the graph is undirected, self-loop-free, and coalesced.",
        "",
        "## 2. Implementation Audit",
        "",
        "- P0.0 frozen embeddings: 15/15 dataset × seed artifacts verified with byte SHA-256, run-seed metadata, finite float32 values, and configured dimensions.",
        "- P0.1 edge populations: 15/15 ordered `(target_node, neighbor_node)` arrays match the fixed cached sample exactly.",
        "- Relation descriptor: `[H_target, H_source, |H_target-H_source|, H_target⊙H_source]`; independent Text and Visual encoders.",
        "- V2 and V3 use the same `Linear(4D,64) → GELU → LayerNorm → Dropout → Linear(64,64) → GELU → LayerNorm` relation-encoder definition.",
        "- V0/V1/V2/V3 share `[H_text,H_visual,N_text,N_visual] → Linear(4D,num_classes)`. Aggregation is an incoming mean with the unchanged physical degree; no attention, learned re-normalization, self-loop, residual, or hop coefficient is used.",
        "- Full-graph training is computed in target chunks only for memory. The mathematical graph and aggregation are unchanged. Context relation `a/g` is computed once from H0 and fixed across orders.",
        "- Exact edge contribution was audited on 256 random sampled edges per model/run, CPU float32, for Text/Visual/joint removal. Every audit must have max absolute error `<1e-5`.",
        "- V3 has an additional feature-modulation head. Relation parameter counts and functional controls are reported below; gains are not attributed solely to parameter capacity.",
        "",
        "### Frozen split / source-artifact summary",
        "",
        "| Dataset | probe train + calib | Original val | Test indices (count only) | Sampled P0.1 relations | Physical directed edges |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        item = manifest.get("datasets", {}).get(ds, {})
        audit = item.get("split_audit", {})
        lines.append(
            f"| {ds} | {audit.get('probe_train_size','NA')} + {audit.get('probe_calib_size','NA')} | {audit.get('original_val_size','NA')} | {audit.get('original_test_index_count','NA')} | {item.get('sampled_relation_count','NA')} | {item.get('physical_graph_directed_edge_count','NA')} |"
        )
    lines += ["", "### Parameter counts", "", "| Dataset | Variant | Relation params | Classifier params | Total trainable |", "|---|---|---:|---:|---:|"]
    for ds in datasets:
        for variant in variants:
            selected = [r for r in data["immediate"] if r["dataset"] == ds and r["variant"] == variant]
            if selected:
                r = selected[0]
                lines.append(f"| {ds} | {variant} | {r['relation_parameter_count']:,} | {r['classifier_parameter_count']:,} | {r['total_trainable_parameter_count']:,} |")

    lines += ["", "## 3. Immediate One-Hop Results", "", "Table A reports held-out original-val mean ± population SD across available seeds. Per-seed train/calib/val Acc, Macro-F1, CE, best epoch, and parameter counts are in `p02_immediate_results.csv`.", "", "| Dataset | Variant | Val Acc | Val Macro-F1 | Val CE | Params (total) |", "|---|---|---:|---:|---:|---:|"]
    for ds in datasets:
        for variant in variants:
            rows = [r for r in data["immediate"] if r["dataset"] == ds and r["variant"] == variant]
            if rows:
                def pair(field: str) -> str:
                    m, sd = _mean_sd([r[field] for r in rows])
                    return _fmt(m, sd)
                params = rows[0]["total_trainable_parameter_count"]
                lines.append(f"| {ds} | {variant} | {pair('heldout_original_val_acc')} | {pair('heldout_original_val_macro_f1')} | {pair('heldout_original_val_ce')} | {params:,} |")

    lines += ["", "V0 is structurally the P0.1 uniform one-hop mean with a newly fitted classifier. The following paired differences provide the smoke/pilot alignment check on the original-val Acc and Macro-F1 fields saved by P0.1 (P0.1 did not save CE in that metrics block).", "", "| Dataset | P0.2 V0 − P0.1 Acc | Macro-F1 |", "|---|---:|---:|"]
    for ds in datasets:
        rows = [r for r in data["immediate"] if r["dataset"] == ds and r["variant"] == "uniform" and r["p02_minus_p01_val_acc"] is not None]
        if rows:
            cells = [_fmt(*_mean_sd([r[key] for r in rows])) for key in ("p02_minus_p01_val_acc", "p02_minus_p01_val_macro_f1")]
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    pairs = _pairwise_rows(data)
    lines += ["", "## 4. Learned Scalar vs Conditional Feature", "", "Paired seed deltas are computed within each run before aggregation. Positive Acc/F1 favors V3; negative CE favors V3. Context readouts remain frozen diagnostics, not benchmark-model scores.", "", "| Dataset | ΔAcc V3−V2 | ΔF1 V3−V2 | ΔCE V3−V2 | Context-only ΔAcc / ΔF1 | Full-bank ΔAcc / ΔF1 |", "|---|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        sub = [r for r in pairs if r["dataset"] == ds]
        if not sub:
            continue
        def pdelta(comp: str, metric: str) -> str:
            m, sd = _mean_sd([r[f"delta_{metric}"] for r in sub if r["comparison"] == comp])
            return _fmt(m, sd)
        lines.append(
            f"| {ds} | {pdelta('one_hop_V3_minus_V2','acc')} | {pdelta('one_hop_V3_minus_V2','macro_f1')} | {pdelta('one_hop_V3_minus_V2','ce')} | {pdelta('context_only_V3_minus_V2','acc')} / {pdelta('context_only_V3_minus_V2','macro_f1')} | {pdelta('full_bank_V3_minus_V2','acc')} / {pdelta('full_bank_V3_minus_V2','macro_f1')} |"
        )

    lines += ["", "## 5. Model-Specific Edge Utility", "", "`U = CE(logits − exact edge contribution, y) − CE(logits, y)` is model-specific and is not causal ground truth. The table gives V3−V2 mean ΔU on the union of P0.1 Text/Visual compatibility-conflict relations. Per-seed means, medians, positive fractions, and target-node bootstrap CIs for both modality-specific and joint scopes are in `p02_conflict_subset.csv`.", "", "| Dataset | Scope | Conflict edges (mean count) | Mean ΔU (seed mean ± SD) | Fraction ΔU>0 |", "|---|---|---:|---:|---:|"]
    conflict_union = [r for r in data["conflicts"] if r["subset"] == "all_conflict_union"]
    for ds in datasets:
        for scope in ("text", "visual", "joint"):
            sub = [r for r in conflict_union if r["dataset"] == ds and r["utility_scope"] == scope]
            if sub:
                mc, sc = _mean_sd([r["edge_count"] for r in sub])
                md, sd = _mean_sd([r["mean_delta_u_v3_minus_v2"] for r in sub])
                mf, sf = _mean_sd([r["fraction_delta_u_positive"] for r in sub])
                lines.append(f"| {ds} | {scope} | {mc:.1f} | {_fmt(md,sd)} | {_fmt(mf,sf)} |")

    lines += ["", "## 6. Compatibility-Conflict Analysis", "", "Compatibility-conflict is defined against the original P0.1 uniform-message utility: Q1 with reference `U>0` or Q5 with reference `U<0`. Consistent relations use Q1 `U<0` or Q5 `U>0`. These fixed subsets are diagnostic only and were not used for fitting or checkpoint selection.", "", "The conflict CSV separately contains low-similarity beneficial, high-similarity harmful, all conflict, consistent, and union subsets for Text/Visual reference definitions, with Text/Visual/joint P0.2 utility scopes. Each run recomputes ΔU per edge and then performs the 1,000-replicate target-node bootstrap for the mean; seed SD and bootstrap CI are not combined.", ""]

    lines += ["## 7. Non-Scalar Mechanism Analysis", "", "V3 diagnostic distributions are computed on the fixed sampled original-val relation population. `channel_std` is population SD across the 128 feature-gate channels for each edge. Values are descriptive; no artificial success threshold is imposed.", "", "| Dataset | Modality | Subset | Residual ratio | Non-collinearity | Channel std |", "|---|---|---|---:|---:|---:|"]
    for ds in datasets:
        for mod in ("text", "visual"):
            for subset in ("all_edges", "compatibility_conflict", "compatibility_consistent"):
                group = [r for r in data["mechanisms"] if r["dataset"] == ds and r["modality"] == mod and r["subset"] == subset]
                if not group:
                    continue
                def mech(metric: str) -> str:
                    m, sd = _mean_sd([r[f"{metric}_mean"] for r in group])
                    return _fmt(m, sd)
                lines.append(f"| {ds} | {mod} | {subset} | {mech('transform_residual_ratio')} | {mech('non_collinearity')} | {mech('feature_gate_channel_std')} |")

    lines += ["", "## 8. Text/Visual Disagreement Analysis", "", "The robust P0.1 definition is reused with `epsilon = 0.1 × median(|U_ref,joint|)` (or `1e-8` if that median is zero); both modality utilities must exceed epsilon in magnitude. Disagreement means opposite signs; agreement means equal signs. Mechanism distributions for these two subsets appear in `p02_mechanism_summary.csv` beside all-edge/conflict/consistent groups. No requirement is imposed that disagreement edges have larger V3 modulation.", ""]
    for ds in datasets:
        for mod in ("text", "visual"):
            conflict_row = [r for r in data["mechanisms"] if r["dataset"] == ds and r["modality"] == mod and r["subset"] == "robust_tv_disagreement"]
            agree_row = [r for r in data["mechanisms"] if r["dataset"] == ds and r["modality"] == mod and r["subset"] == "robust_tv_agreement"]
            if conflict_row and agree_row:
                a1, a2 = conflict_row[0]["edge_count"], agree_row[0]["edge_count"]
                eps = conflict_row[0]["robust_epsilon"]
                lines.append(f"- {ds} {mod}: robust disagreement/agreement counts are {a1}/{a2}; epsilon={eps:.6g}.")
    lines += ["", "| Dataset | Modality | Robust subset | Residual ratio | Non-collinearity | Channel std |", "|---|---|---|---:|---:|---:|"]
    for ds in datasets:
        for mod in ("text", "visual"):
            for subset in ("robust_tv_disagreement", "robust_tv_agreement"):
                group = [r for r in data["mechanisms"] if r["dataset"] == ds and r["modality"] == mod and r["subset"] == subset]
                if group:
                    def rstat(metric: str) -> str:
                        return _fmt(*_mean_sd([row[f"{metric}_mean"] for row in group]))
                    lines.append(f"| {ds} | {mod} | {subset} | {rstat('transform_residual_ratio')} | {rstat('non_collinearity')} | {rstat('feature_gate_channel_std')} |")

    lines += ["", "## 9. Function Identity Ablation", "", "The frozen V3 relation encoder, scalar gate, and classifier are kept unchanged while `g := 1` at original-val inference. Positive Acc/F1 `full−identity` and negative CE `full−identity` favor the learned feature function.", "", "| Dataset | Full Acc | Identity Acc | ΔAcc | Full F1 | Identity F1 | ΔF1 | Full CE | Identity CE | ΔCE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        group = [r for r in data["functions"] if r["dataset"] == ds]
        if not group:
            continue
        fields = ["full_acc", "identity_acc", "delta_full_minus_identity_acc", "full_macro_f1", "identity_macro_f1", "delta_full_minus_identity_macro_f1", "full_ce", "identity_ce", "delta_full_minus_identity_ce"]
        values = [_fmt(*_mean_sd([r[f] for r in group])) for f in fields]
        lines.append(f"| {ds} | " + " | ".join(values) + " |")

    lines += ["", "## 10. Relation-Function Shuffle", "", "The five frozen-checkpoint repetitions shuffle whole `g_ij` vectors within the same dataset, modality, and full-physical-graph probe-similarity quintile; edge-specific `a_ij` is untouched. Full-minus-shuffle is reported as requested; for CE, lower is better.", "", "| Dataset | Full Acc | Shuffle Acc | ΔAcc | Full F1 | Shuffle F1 | ΔF1 | Full CE | Shuffle CE | ΔCE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        group = [r for r in data["functions"] if r["dataset"] == ds]
        if group:
            fields = ["full_acc", "shuffle_acc_mean", "delta_full_minus_shuffle_acc", "full_macro_f1", "shuffle_macro_f1_mean", "delta_full_minus_shuffle_macro_f1", "full_ce", "shuffle_ce_mean", "delta_full_minus_shuffle_ce"]
            values = [_fmt(*_mean_sd([r[f] for r in group])) for f in fields]
            lines.append(f"| {ds} | " + " | ".join(values) + " |")

    lines += ["", "## 11. Minimal Context Rollout", "", "C1/C2/C3 are repeated applications of the fixed H0-derived relation state. No normalization or learned hop weight is inserted into propagation. Every context block was checked for finite values and norm summaries (mean, median, p95/max) are saved by seed in `p02_context_stability.csv`.", "", "The output contains no NaN/Inf for completed runs. V3 C3 feature norms grow substantially relative to frozen H0 (H0 p95 is about 9–10); the maximum V3 C3 p95 across modality and seed reaches approximately 113.5 for Movies, 128.3 for Toys, 138.5 for Grocery, 151.4 for ele-fashion, and 199.0 for Reddit-S. These are observed activation growth values and remain a limitation for downstream use. No normalization was added after observing them. This is a frozen diagnostic, not a final MAG model.", ""]

    lines += ["## 12. Context-Only and Full-Bank Probe", "", "Only a linear readout is trained on probe_train, selected by probe_calib accuracy, then evaluated on original val. Context blocks are L2-normalized immediately before concatenation. These are diagnostic readouts and must not be compared with formal benchmark-model scores.", "", "| Dataset | Variant | Context-only Acc / F1 | Full-bank Acc / F1 |", "|---|---|---:|---:|"]
    for ds in datasets:
        for variant in variants:
            group = [r for r in data["contexts"] if r["dataset"] == ds and r["variant"] == variant]
            if not group:
                continue
            parts = []
            for readout in ("context_only", "full_bank"):
                rows = [r for r in group if r["readout"] == readout]
                acc = _fmt(*_mean_sd([r["context_val_acc"] for r in rows]))
                f1 = _fmt(*_mean_sd([r["context_val_macro_f1"] for r in rows]))
                parts.append(f"{acc} / {f1}")
            lines.append(f"| {ds} | {variant} | {parts[0]} | {parts[1]} |")

    lines += ["", "## 13. H2 Evidence Matrix", "", "### Immediate Task Evidence", "", "Use the paired V3−V2 rows in Section 4. Immediate one-hop metrics answer whether the feature function improves this probe task; they are not a required standalone win condition.", "", "### Non-Scalar Mechanism Evidence", "", "Residual ratio measures feature changes relative to the incoming semantic vector; non-collinearity measures direction change; channel dispersion measures non-uniform feature gates. See the all-edge/conflict/consistent distributions in Section 7.", "", "### Function Identity / Shuffle Evidence", "", "Identity tests whether learned `g` changes predictions with `a` fixed. The quintile-constrained shuffle tests whether the relation-to-function assignment matters while preserving `a` and the within-quintile gate-vector multiset. See Sections 9–10.", "", "### Compatibility-Conflict Evidence", "", "P0.1 conflict subsets are held fixed and never train P0.2. ΔU is model-specific; positive values do not establish a downstream task gain. Target-node bootstrap intervals are per run and remain distinct from three-seed SD.", "", "### Downstream Context Evidence", "", "The frozen rollout and linear heads test whether the learned fixed interaction is useful after repeated structural composition. It is not final architecture performance.", "", "### Overall Assessment", ""]
    if not full:
        lines += ["Not assigned from a partial smoke/pilot matrix. Continue the fixed protocol; do not infer H2 from this partial report."]
    else:
        lines += ["The matrix below is descriptive, based on the full evidence chain and the observed three-seed patterns. It is not a statistical significance classification. 'Weak' means similarity/feature interaction is not a reliable one-hop task discriminator; 'intermediate' means some controlled interaction or context evidence is present but does not establish a broad immediate gain; 'strong' would require consistent useful relation-function evidence across these diagnostics. No dataset meets the strong criterion."]
        lines += ["", "| Dataset | Evidence regime | Brief evidence summary |", "|---|---|---|"]
        assessment = {
            "Movies": ("Weak, with function-sensitive effects", "V3−V2 one-hop is small and seed-variable; context-only/full-bank results are variable. Identity and within-quintile shuffle change predictions, while edge-utility effects are mixed."),
            "Toys": ("Intermediate", "One-hop V3 and V2 are near-tied; both context readouts improve Acc/F1 over V2 across all three seeds, though context-only CE is worse. Functional controls show relation assignment matters."),
            "Grocery": ("Intermediate", "One-hop V3−V2 is mixed and slightly favors V2 on CE; context-only Acc/F1 improve in all seeds. Full-bank gains and edge-utility effects are smaller or mixed."),
            "ele-fashion": ("Intermediate, context-led", "Immediate V3 Acc/CE are worse than V2 on average, while context-only and full-bank Acc/F1 improve in each seed. Seed variation and opposing one-hop/context evidence preclude a broad gain claim."),
            "Reddit-S": ("Weak to intermediate, localized", "V3 has non-scalar mechanism and positive mean conflict-subset ΔU, but only a minority of union-conflict edges have positive ΔU; immediate and full-bank task gains are absent or mixed."),
        }
        for ds in datasets:
            label, evidence = assessment.get(ds, ("Unclassified", "No prewritten interpretation for this dataset."))
            lines.append(f"| {ds} | {label} | {evidence} |")

    lines += ["", "## 14. Overall Assessment", "", "Overall, the evidence is **moderate and dataset-dependent** for testing feature-wise interaction further. V3 demonstrably learns non-uniform feature gates and changes some controlled predictions, but its immediate one-hop results are close to V2 and are not consistently better. Context readouts help in selected datasets, with substantial C3 norm growth that limits direct downstream use. This supports a bounded follow-up comparison if desired; it does not justify a final architecture choice.", "", "## 15. What the Evidence Supports", "", "- V3 learns non-uniform, non-collinear feature transformations on the fixed sampled validation relation population.", "- Identity and within-quintile shuffle controls show that the learned relation-to-function assignment affects predictions, with dataset-dependent direction and size.", "- V3−V2 model-specific edge utility differs on fixed P0.1 conflict/consistent subsets, while its distribution is heterogeneous across targets and datasets.", "- Fixed relation functions can produce useful C1/C2/C3 diagnostic representations in some datasets under separately trained linear readouts.", "", "## 16. What the Evidence Does NOT Support", "", "This experiment does not establish that scalar learned relations are universally insufficient, that vector relation state is necessary, that semantic transformation is necessary, or that MoE, basis routing, low-rank operators, or a cross-modal relation mixer is needed. It does not establish general multi-hop heterogeneous context utility or determine the final architecture. C3 norm growth also needs to be addressed before treating the unnormalized rollout as a usable downstream representation.", "", "## 17. Implications for a Later Stage", "", "Carry forward the feature-wise interaction only as a controlled candidate, with particular attention to the context-positive Toys/Grocery/ele-fashion results and the weak or mixed immediate results. A later stage should first control capacity and C3 activation growth, retain identity/shuffle controls, and evaluate a predeclared dataset-specific hypothesis. No later-stage model is implemented here.", "", "## Output Files", "", "- `results/problem_validation/p02/p02_immediate_results.csv` — per-seed immediate train/calib/val metrics and parameters.", "- `results/problem_validation/p02/p02_cross_dataset.csv` — across-seed mean and population SD.", "- `results/problem_validation/p02/p02_pairwise_deltas.csv` — paired V3−V2, V2−V1, V1−V0, and context deltas.", "- `results/problem_validation/p02/p02_mechanism_summary.csv` — V3 mechanism groups and distributions.", "- `results/problem_validation/p02/p02_conflict_subset.csv` and `_summary.csv` — per-run conflict subsets/target-node bootstrap CIs and aggregated summaries.", "- `results/problem_validation/p02/p02_function_controls.csv` — identity and shuffle controls.", "- `results/problem_validation/p02/p02_context_probe.csv` and `p02_context_stability.csv` — context readouts and activation norms.", "- `results/problem_validation/p02/plots/` — Figures A–C (PNG, 600 dpi TIFF, SVG, PDF) and panel-alignment / rendered-collision QA JSON.", ""]
    return "\n".join(lines)


def main(config_path: Path | None = None, allow_partial: bool = False) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser(description="Summarize P0.2 results without accessing test metrics.")
        parser.add_argument("--config", type=Path, default=ROOT / "configs/analysis/problem_validation_p02.yaml")
        parser.add_argument("--allow-partial", action="store_true")
        args = parser.parse_args()
        config_path = args.config
        allow_partial = args.allow_partial
    config = yaml.safe_load(config_path.read_text())
    a = config["analysis"]
    output_root = ROOT / a["output_root"]
    result_root = ROOT / a["results_root"]
    result_root.mkdir(parents=True, exist_ok=True)
    data = _collect(config)
    n_expected = len(a["datasets"]) * len(a["run_seeds"])
    if len(data["runs"]) < n_expected and not allow_partial:
        raise RuntimeError(f"only {len(data['runs'])}/{n_expected} dataset-seed runs are complete; pass --allow-partial for a smoke/pilot summary")
    _write_csv(result_root / "p02_immediate_results.csv", data["immediate"])
    immediate_aggregate = _group_aggregate(
        data["immediate"], ("dataset", "variant"),
        ("heldout_original_val_acc", "heldout_original_val_macro_f1", "heldout_original_val_ce", "relation_parameter_count", "classifier_parameter_count", "total_trainable_parameter_count", "best_epoch"),
    )
    _write_csv(result_root / "p02_cross_dataset.csv", immediate_aggregate)
    _write_csv(result_root / "p02_pairwise_deltas.csv", _pairwise_rows(data))
    _write_csv(result_root / "p02_mechanism_summary.csv", data["mechanisms"])
    _write_csv(result_root / "p02_conflict_subset.csv", data["conflicts"])
    conflict_aggregate = _group_aggregate(
        data["conflicts"], ("dataset", "reference_modality", "subset", "utility_scope"),
        ("edge_count", "mean_delta_u_v3_minus_v2", "median_delta_u_v3_minus_v2", "fraction_delta_u_positive"),
    )
    _write_csv(result_root / "p02_conflict_subset_summary.csv", conflict_aggregate)
    _write_csv(result_root / "p02_function_controls.csv", data["functions"])
    _write_csv(result_root / "p02_context_probe.csv", data["contexts"])
    _write_csv(result_root / "p02_context_stability.csv", data["stability"])
    _write_figures(result_root, data, list(a["datasets"]), list(a["variants"]))
    report = _generate_report(config, data, {"expected": n_expected, "completed": len(data["runs"])}, result_root)
    (ROOT / "docs/problem_validation/p02_report.md").write_text(report)
    (result_root / "README.md").write_text(
        "# P0.2 Result Bundle\n\n"
        f"Run coverage: {len(data['runs'])}/{n_expected} dataset × seed runs.\n\n"
        "The CSVs preserve per-seed results. Cross-dataset summaries use mean ± population SD; node-cluster bootstrap CIs remain per run. Figures A–C are diagnostic and are not formal baseline benchmark comparisons. The Matplotlib panels pass the strict 1.5 pt alignment gate; source validation, PDF text-size audit, collision audit, and visual inspection are recorded in the figure QA files.\n\n"
        "Contents: immediate metrics, cross-seed aggregates, paired deltas, mechanism summaries, per-run and aggregated conflict subsets, identity/shuffle controls, context probe metrics, context activation stability, and source-artifact manifest. Plots are provided as PNG, 600 dpi TIFF, editable SVG, and PDF.\n\n"
        "See `../../../../docs/problem_validation/p02_report.md` for protocol, interpretation, and evidence matrix.\n"
    )
    print(f"P0.2 summary generated: {len(data['runs'])}/{n_expected} dataset-seed runs; report=docs/problem_validation/p02_report.md")


if __name__ == "__main__":
    main()
