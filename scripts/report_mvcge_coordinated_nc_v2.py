from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "mvcge_coordinated_nc_v2"
DATASETS = ("Movies", "Grocery", "Reddit-S")
MODELS = ("A0", "N0", "M0", "M1", "P0", "P1")
SEEDS = (113, 114, 115)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def _f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report() -> None:
    audit = _read_json(REPORT / "audit_result.json")
    calibration_audit = _read_json(REPORT / "calibration_audit.json")
    if audit.get("status") != "passed" or calibration_audit.get("status") != "passed":
        raise RuntimeError("both calibration and confirmation audits must pass before reporting")
    registry = _read_json(REPORT / "run_registry.json")["attempts"]
    selection = _read_json(REPORT / "selection_lock.json")["selections"]
    profile = _read_json(REPORT / "data_profile.json")
    protocol_lock = _read_json(ROOT / "docs" / "mvcge_coordinated_nc_v2" / "protocol_lock.json")
    results = [row for row in registry.values() if row.get("status") == "success"]
    confirmation = [row for row in results if row["phase"] == "confirmation"]
    calibration = [row for row in results if row["phase"] == "calibration"]
    if len(calibration) != 162 or len(confirmation) != 54:
        raise RuntimeError(f"cannot report incomplete matrix: calibration={len(calibration)}, confirmation={len(confirmation)}")
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in confirmation:
        by_key[(row["dataset"], row["model"], int(row["training_seed"]))] = row

    summary_path = REPORT / "summary.csv"
    columns = [
        "phase", "dataset", "model", "seed", "config_index", "lr", "weight_decay", "train_accuracy", "train_macro_f1",
        "validation_accuracy", "validation_macro_f1", "best_step", "updates", "stop_reason", "total_parameters",
        "median_stable_step_ms_update_ge_110", "optimizer_update_seconds", "validation_wall_seconds", "diagnostic_seconds",
        "total_cell_wall_seconds", "peak_allocated_gib", "peak_reserved_gib", "selected_lr_boundary", "selected_weight_decay_boundary",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in sorted(results, key=lambda item: (item["phase"], DATASETS.index(item["dataset"]), MODELS.index(item["model"]), int(item["training_seed"]), int(item.get("config_index", -1)))):
            selected = selection[f"{row['dataset']}/{row['model']}"] if row["phase"] == "confirmation" else None
            writer.writerow({
                "phase": row["phase"], "dataset": row["dataset"], "model": row["model"], "seed": row["training_seed"],
                "config_index": row["config_index"], "lr": row["lr"], "weight_decay": row["weight_decay"],
                "train_accuracy": row["train"]["accuracy"], "train_macro_f1": row["train"]["macro_f1"],
                "validation_accuracy": row["validation"]["accuracy"], "validation_macro_f1": row["validation"]["macro_f1"],
                "best_step": row["best_step"], "updates": row["updates"], "stop_reason": row["stop_reason"],
                "total_parameters": row["parameter_counts_total"], "median_stable_step_ms_update_ge_110": row.get("median_stable_non_diagnostic_step_ms_update_ge_110"),
                "optimizer_update_seconds": row["optimizer_update_seconds"], "validation_wall_seconds": row["validation_wall_seconds"],
                "diagnostic_seconds": row["diagnostic_seconds"], "total_cell_wall_seconds": row["total_cell_wall_seconds"],
                "peak_allocated_gib": None if row["peak_allocated_bytes"] is None else row["peak_allocated_bytes"] / 1024**3,
                "peak_reserved_gib": None if row["peak_reserved_bytes"] is None else row["peak_reserved_bytes"] / 1024**3,
                "selected_lr_boundary": None if selected is None else selected["lr"] in {0.0003, 0.003},
                "selected_weight_decay_boundary": None if selected is None else selected["weight_decay"] in {0.0001, 0.01},
            })

    primary_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            runs = [by_key[(dataset, model, seed)] for seed in SEEDS]
            acc_mean, acc_sd = _mean_sd([float(run["validation"]["accuracy"]) for run in runs])
            f1_mean, f1_sd = _mean_sd([float(run["validation"]["macro_f1"]) for run in runs])
            train_acc_mean, train_acc_sd = _mean_sd([float(run["train"]["accuracy"]) for run in runs])
            train_f1_mean, train_f1_sd = _mean_sd([float(run["train"]["macro_f1"]) for run in runs])
            selected = selection[f"{dataset}/{model}"]
            primary_rows.append({
                "dataset": dataset, "model": model, "validation_accuracy_mean": acc_mean, "validation_accuracy_population_sd": acc_sd,
                "validation_macro_f1_mean": f1_mean, "validation_macro_f1_population_sd": f1_sd,
                "train_accuracy_mean": train_acc_mean, "train_accuracy_population_sd": train_acc_sd,
                "train_macro_f1_mean": train_f1_mean, "train_macro_f1_population_sd": train_f1_sd,
                "validation_accuracy_per_seed": [run["validation"]["accuracy"] for run in runs],
                "validation_macro_f1_per_seed": [run["validation"]["macro_f1"] for run in runs],
                "train_accuracy_per_seed": [run["train"]["accuracy"] for run in runs],
                "train_macro_f1_per_seed": [run["train"]["macro_f1"] for run in runs],
                "selected": selected,
                "best_steps": [run["best_step"] for run in runs], "updates": [run["updates"] for run in runs],
                "max_update_fraction": sum(int(run["updates"]) >= 1000 for run in runs) / 3,
                "train_minus_validation_accuracy_mean": train_acc_mean - acc_mean,
                "encoder_parameters": runs[0]["active_encoder_parameters"], "task_head_parameters": runs[0]["active_task_head_parameters"],
                "total_parameters": runs[0]["parameter_counts_total"],
            })

    comparisons = [(name, left, right) for name, left, right in (
        ("M0_vs_A0", "M0", "A0"), ("M1_vs_A0", "M1", "A0"), ("P0_vs_A0", "P0", "A0"), ("P1_vs_A0", "P1", "A0"),
        ("M0_vs_N0", "M0", "N0"), ("P0_vs_M0", "P0", "M0"), ("M1_vs_M0", "M1", "M0"), ("P1_vs_P0", "P1", "P0"),
    )]
    paired: dict[str, Any] = {}
    for comparison, left, right in comparisons:
        per_graph = {}
        all_acc, all_f1 = [], []
        for dataset in DATASETS:
            seed_rows = []
            for seed in SEEDS:
                lrun, rrun = by_key[(dataset, left, seed)], by_key[(dataset, right, seed)]
                row = {
                    "seed": seed,
                    "validation_accuracy_delta": float(lrun["validation"]["accuracy"] - rrun["validation"]["accuracy"]),
                    "validation_macro_f1_delta": float(lrun["validation"]["macro_f1"] - rrun["validation"]["macro_f1"]),
                }
                seed_rows.append(row); all_acc.append(row["validation_accuracy_delta"]); all_f1.append(row["validation_macro_f1_delta"])
            acc_mean, acc_sd = _mean_sd([row["validation_accuracy_delta"] for row in seed_rows])
            f1_mean, f1_sd = _mean_sd([row["validation_macro_f1_delta"] for row in seed_rows])
            per_graph[dataset] = {"per_seed": seed_rows, "validation_accuracy_delta_mean": acc_mean, "validation_accuracy_delta_population_sd": acc_sd, "validation_macro_f1_delta_mean": f1_mean, "validation_macro_f1_delta_population_sd": f1_sd}
        paired[comparison] = {"left": left, "right": right, "per_graph": per_graph, "equal_graph_mean_validation_accuracy_delta": float(np.mean([per_graph[d]["validation_accuracy_delta_mean"] for d in DATASETS])), "equal_graph_mean_validation_macro_f1_delta": float(np.mean([per_graph[d]["validation_macro_f1_delta_mean"] for d in DATASETS])), "pooled_same_seed_validation_accuracy_delta_mean": float(np.mean(all_acc)), "pooled_same_seed_validation_macro_f1_delta_mean": float(np.mean(all_f1))}
    interactions = {}
    for dataset in DATASETS:
        rows = []
        for seed in SEEDS:
            delta_f1 = (by_key[(dataset, "P1", seed)]["validation"]["macro_f1"] - by_key[(dataset, "P0", seed)]["validation"]["macro_f1"]) - (by_key[(dataset, "M1", seed)]["validation"]["macro_f1"] - by_key[(dataset, "M0", seed)]["validation"]["macro_f1"])
            delta_acc = (by_key[(dataset, "P1", seed)]["validation"]["accuracy"] - by_key[(dataset, "P0", seed)]["validation"]["accuracy"]) - (by_key[(dataset, "M1", seed)]["validation"]["accuracy"] - by_key[(dataset, "M0", seed)]["validation"]["accuracy"])
            rows.append({"seed": seed, "validation_accuracy_descriptive_interaction": delta_acc, "validation_macro_f1_descriptive_interaction": delta_f1})
        interactions[dataset] = {"per_seed": rows, "mean_accuracy": float(np.mean([row["validation_accuracy_descriptive_interaction"] for row in rows])), "mean_macro_f1": float(np.mean([row["validation_macro_f1_descriptive_interaction"] for row in rows]))}
    resource = {}
    for model in MODELS:
        runs = [run for run in confirmation if run["model"] == model]
        resource[model] = {
            "runs": len(runs), "max_update_count": sum(int(run["updates"]) >= 1000 for run in runs),
            "median_stable_update_ms": float(np.median([run["median_stable_non_diagnostic_step_ms_update_ge_110"] for run in runs])),
            "median_optimizer_update_seconds": float(np.median([run["optimizer_update_seconds"] for run in runs])),
            "total_validation_seconds": float(sum(run["validation_wall_seconds"] for run in runs)),
            "total_diagnostic_seconds": float(sum(run["diagnostic_seconds"] for run in runs)),
            "total_cell_wall_seconds": float(sum(run["total_cell_wall_seconds"] for run in runs)),
            "peak_allocated_gib_max": float(max((run["peak_allocated_bytes"] or 0) for run in runs) / 1024**3),
            "peak_reserved_gib_max": float(max((run["peak_reserved_bytes"] or 0) for run in runs) / 1024**3),
            "diagnostic_fraction_mean": float(np.mean([run["diagnostic_fraction_of_update_and_diagnostic_time"] for run in runs])),
            "gpu_start_snapshots": [run["gpu_snapshot_start"] for run in runs],
            "gpu_end_snapshots": [run["gpu_snapshot_end"] for run in runs],
            "gpu_compute_processes_start": [run.get("gpu_compute_processes_start", []) for run in runs],
            "gpu_compute_processes_end": [run.get("gpu_compute_processes_end", []) for run in runs],
            "runner_pids": [run.get("runner_pid") for run in runs],
        }
    summary = {
        "study": "mvcge_coordinated_nc_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_cell_counts": {"calibration_success": len(calibration), "confirmation_success": len(confirmation), "total_success": len(results)},
        "primary_confirmation": primary_rows, "paired_differences": paired, "descriptive_auxiliary_by_PC_interaction": interactions,
        "resource_summary_confirmation": resource, "all_classes_macro_f1": True, "macro_f1_zero_division": 0,
        "test_evaluations": 0, "real_lp_runs": 0, "additional_datasets": 0,
        "source_commit": protocol_lock["training_source_commit"],
        "source_commits_present_in_audited_results": sorted({row["source_commit"] for row in confirmation}),
        "implementation_versions": audit.get("implementation_versions", []),
        "protocol_amendment": protocol_lock.get("amendment"),
    }
    write_json(REPORT / "summary.json", summary)
    write_json(REPORT / "paired_differences.json", {"comparisons": paired, "descriptive_interactions": interactions})

    lines = [
        "# Coordinated graph experts: results", "",
        f"The frozen matrix completed {len(calibration)} calibration and {len(confirmation)} confirmation cells ({len(results)} total). Results below use only confirmation seeds 113-115; calibration performance is reported separately in `summary.csv` and the selection lock.",
        "", "## Protocol amendment and implementation provenance", "",
        "Calibration and its 18-entry selection lock were completed under `protocol_lock_v1.json`. The first confirmation pass completed all training updates, but 36 routed-model cells could not finalize because the gradient summary omitted unused parameter slots before comparing vectors. The 18 A0/N0 cells were unaffected and remain under the original lock. The diagnostic now fills unused slots with zeros in parameter order; the 36 routed cells were rerun as retained retry attempts under the amended lock. The amendment changes diagnostic aggregation only; model computation, losses, optimizer updates, seeds, data, splits, and selected recipes are unchanged.",
        "", "The first retry dispatch hit a retry-ID parsing bug and exited before constructing a model; no run started under the superseded `protocol_lock_v2.json`, which is preserved. All 36 routed retries then completed under `protocol_lock_v3.json`. The first audit invocation resolved unconfigured placeholders and marked the audit incomplete; the audit-only correction in the final lock keys expected cells from the selection lock. The audit records each successful cell's source commit and lock hash in `audit_result.json`; the final 54-cell set therefore has 18 valid v1 records and 36 valid v3 records. No training was run under the audit-only final lock.",
        "", "Validation is the repeatedly used development split, not an independent benchmark. Differences are descriptive across three seeds; no significance, generalization, or causal claim is made.",
        "", "## Confirmation results", "",
        "Acc and macro-F1 are mean ± population SD over the three training seeds. Train means describe fit only.",
        "", "| Graph | Model | Val Acc | Val macro-F1 | Train Acc | Train macro-F1 | Selected LR / WD | Best updates | Max-budget share | Parameters (encoder + head) |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in primary_rows:
        chosen = row["selected"]
        lines.append(f"| {row['dataset']} | {row['model']} | {_fmt(row['validation_accuracy_mean'],row['validation_accuracy_population_sd'])} | {_fmt(row['validation_macro_f1_mean'],row['validation_macro_f1_population_sd'])} | {_fmt(row['train_accuracy_mean'],row['train_accuracy_population_sd'])} | {_fmt(row['train_macro_f1_mean'],row['train_macro_f1_population_sd'])} | {chosen['lr']:g} / {chosen['weight_decay']:g} | {','.join(str(s) for s in row['best_steps'])} | {row['max_update_fraction']:.2f} | {row['encoder_parameters']:,} + {row['task_head_parameters']:,} = {row['total_parameters']:,} |")
    calibration_winners = {}
    for key, chosen in selection.items():
        calibration_winners[key] = next(row for row in calibration if row["dataset"] == chosen["dataset"] and row["model"] == chosen["model"] and int(row["config_index"]) == int(chosen["config_index"]))
    lines += ["", "## Calibration selections", "", "Each recipe was selected only from its nine seed-112 cells by validation correct count, macro-F1, then lower config index.", "", "| Graph | Model | Config | LR / WD | Validation correct | Validation Acc | Validation macro-F1 | Best update |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, chosen in selection.items():
        run = calibration_winners[key]
        lines.append(f"| {chosen['dataset']} | {chosen['model']} | {chosen['config_index']} | {chosen['lr']:g} / {chosen['weight_decay']:g} | {chosen['validation_correct']} / {run['validation']['count']} | {run['validation']['accuracy']:.4f} | {run['validation']['macro_f1']:.4f} | {chosen['best_step']} |")
    lines += ["", "## Paired validation differences", "", "Each cell is the mean same-seed difference over seeds 113-115; rows preserve graph-specific changes. Positive values favor the left model. The equal-graph summary weights each graph equally.", "", "| Contrast | Graph | Δ Acc | Δ macro-F1 |", "|---|---|---:|---:|"]
    for contrast, value in paired.items():
        for dataset in DATASETS:
            graph = value["per_graph"][dataset]
            lines.append(f"| {contrast} | {dataset} | {graph['validation_accuracy_delta_mean']:+.4f} | {graph['validation_macro_f1_delta_mean']:+.4f} |")
        lines.append(f"| {contrast} | Equal-graph mean | {value['equal_graph_mean_validation_accuracy_delta']:+.4f} | {value['equal_graph_mean_validation_macro_f1_delta']:+.4f} |")
    lines += ["", "`(P1-P0)-(M1-M0)` is a descriptive interaction under independently selected recipes, not a fixed-recipe causal interaction. Its per-graph and per-seed values are in `paired_differences.json`.", "", "## Optimization and cost", "", "| Model | Confirmation cells at 1000 updates | Median stable update (ms) | Median optimizer-update time (s) | Validation time total (s) | Diagnostic time total (s) | Max allocated / reserved (GiB) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model, row in resource.items():
        lines.append(f"| {model} | {row['max_update_count']}/{row['runs']} | {row['median_stable_update_ms']:.3f} | {row['median_optimizer_update_seconds']:.1f} | {row['total_validation_seconds']:.1f} | {row['total_diagnostic_seconds']:.1f} | {row['peak_allocated_gib_max']:.2f} / {row['peak_reserved_gib_max']:.2f} |")
    boundary = [f"{key} (LR={value['lr']:g}, WD={value['weight_decay']:g})" for key, value in selection.items() if value["lr"] in {0.0003, 0.003} or value["weight_decay"] in {0.0001, 0.01}]
    lines += ["", f"Selections at an LR or WD grid boundary: {len(boundary)}/18. " + (", ".join(boundary) if boundary else "None."), "A selection at a grid boundary leaves optimization unresolved; a run that reaches 1000 updates is also unresolved. See the per-run columns in `summary.csv`.", "", "## Interpretation limits", "", "A0 is the full historical B1 reference, including its padded 768D output and historical task losses; it is not parameter-matched. N0 has no router parameters. M0-N0 also changes parameter count, P0-M0 changes neighbor propagation and coefficients, and M1/P1 were tuned independently. These are comparisons between complete recipes, not isolated causal interventions.", "", "This stage evaluates only full-modal transductive NC on Movies, Grocery, and Reddit-S. The LP decoder was checked only with synthetic endpoint embeddings; no real LP score, test result, or additional-graph result was produced.", "", "The audit is an implementer self-audit. The report's metrics were independently recomputed from saved train/validation IDs, targets, and argmax predictions by the audit script, but the review is not independent peer review.", "", "Raw checkpoints, prediction arrays, and per-update logs remain under ignored `outputs/mvcge_coordinated_nc_v2/`; this repository commit contains concise metrics and hashes only.", ""]
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    completed_at_boundary = [f"{r['dataset']}/{r['model']}/seed{r['training_seed']}" for r in confirmation if r["updates"] >= 1000 or r["lr"] in {0.0003, 0.003} or r["weight_decay"] in {0.0001, 0.01}]
    review = [
        "# Implementer review", "", "## Checks completed", "",
        "- Calibration: all 162 required cells succeeded; fixed-grid selection rules were applied after prediction recomputation.",
        "- Confirmation: all 54 required cells succeeded using the immutable 18-entry selection lock; 36 routed cells were retried under a diagnostic-only protocol amendment after the initial attempts failed in result finalization.",
        "- Audit: train/validation predictions, metrics, confusions, IDs, both permitted source/protocol lineages, output files, and runtime test/LP counters were checked.",
        "- No test evaluator, real LP workload, or dataset expansion was run.",
        "", "## Items requiring reviewer attention", "",
        "- Validation is repeatedly used during development and checkpoint selection; it is not independent test evidence.",
        "- N0/M0, M0/P0, and auxiliary-head contrasts are complete-recipe comparisons with their stated capacity/propagation differences.",
        f"- Runs with boundary selections or the update limit ({len(completed_at_boundary)} matching run records) retain unresolved optimization or recipe-search uncertainty. See `summary.csv` for exact per-run flags.",
        "- Diagnostic cost fractions, any trim decisions, and GPU co-load snapshots are stored per run under the concise summary and ignored raw evidence.",
        "- This is implementer self-review; an independent review remains pending.",
        "", "## Handoff", "", "The experiment is complete. No follow-on training was started. Await external review before any new experiment or model choice.", "",
    ]
    (REPORT / "review.md").write_text("\n".join(review), encoding="utf-8")

    calibration_audit_records = {row["attempt_id"]: row for row in calibration_audit["results"]}
    confirmation_audit_records = {row["attempt_id"]: row for row in audit["results"]}
    manifest = {
        "study": "mvcge_coordinated_nc_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_source_commit": summary["source_commit"], "protocol_lock_sha256": hashlib.sha256((ROOT / "docs" / "mvcge_coordinated_nc_v2" / "protocol_lock.json").read_bytes()).hexdigest(),
        "source_commits_present_in_audited_results": summary["source_commits_present_in_audited_results"],
        "implementation_versions": audit.get("implementation_versions", []),
        "calibration_audit_sha256": hashlib.sha256((REPORT / "calibration_audit.json").read_bytes()).hexdigest(),
        "confirmation_audit_sha256": hashlib.sha256((REPORT / "audit_result.json").read_bytes()).hexdigest(),
        "summary_csv_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "report_sha256_after_write": None,
        "artifact_evidence": [],
    }
    manifest["artifact_evidence"] = [
        {"attempt_id": run["attempt_id"], "prediction": {"path": run["prediction_file"], "sha256": run["prediction_sha256"]},
         "best_checkpoint": {"path": run["best_checkpoint"], "sha256": run["best_checkpoint_sha256"]},
         "last_checkpoint": {"path": run["last_checkpoint"], "sha256": run["last_checkpoint_sha256"]},
         "step_log": {"path": run["step_log"], "sha256": run["step_log_sha256"]},
         "diagnostics": {"path": run["diagnostics_file"], "sha256": run["diagnostics_sha256"]},
         "resolved_config": {"path": run["resolved_config"], "sha256": run["resolved_config_sha256"]},
         "audit_metrics": (calibration_audit_records if run["phase"] == "calibration" else confirmation_audit_records)[run["attempt_id"]]}
        for run in sorted(results, key=lambda item: (item["phase"], item["dataset"], item["model"], int(item["training_seed"]), int(item.get("config_index", -1))))
    ]
    manifest["report_sha256_after_write"] = hashlib.sha256((REPORT / "report.md").read_bytes()).hexdigest()
    write_json(REPORT / "manifest.json", manifest)
    print(f"Wrote report for {len(calibration)} calibration and {len(confirmation)} confirmation cells.")


if __name__ == "__main__":
    build_report()
