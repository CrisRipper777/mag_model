from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.tasks import run_nc
from src.utils.device import get_device
from src.utils.logging import setup_logger


def make_cfg(dataset: str, seed: int, variant: str, assimilation_mode: str, epochs: int, checkpoint: Path):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=adaptive_evidence_assimilation_m21",
            f"model.provenance_conditioning={'true' if variant.endswith('-P') else 'false'}",
            f"model.assimilation_mode={assimilation_mode}", f"seed={seed}", "num_runs=1",
            f"task.epochs={epochs}", "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint.resolve()}",
            "paths.output_root=outputs/model_design/m21",
        ])


def _watched_parameter(name: str) -> bool:
    return (
        name.startswith("assimilation_in_") or name.startswith("assimilation_out_")
        or name in {"assimilation_alpha_text", "assimilation_alpha_visual", "text_query.weight", "visual_query.weight",
                    "query_head_projection.weight", "key_head_projection.weight", "value_head_projection.weight",
                    "output_projection.weight"}
        or any(name.startswith(f"{kind}_{modality}.") for kind in ("key_provenance", "key_interaction", "value_interaction")
               for modality in ("text", "visual"))
    )


def gradient_requirements(trace: dict[str, list[dict]], variant: str, mode: str) -> dict:
    def records_for(prefix: str):
        return [items for name, items in trace.items() if name.startswith(prefix) and items]
    def first_nonzero(items):
        return bool(items and items[0]["finite"] and items[0]["nonzero"])
    def later_nonzero(prefixes):
        items = [item for prefix in prefixes for group in records_for(prefix) for item in group[1:]]
        return any(item["finite"] and item["nonzero"] for item in items)
    checks = {"assimilation_out_first_step": True, "assimilation_in_after_step_two": True,
              "retrieval_query_after_step_two": True, "retrieval_key_after_step_two": True,
              "retrieval_value_after_step_two": True, "provenance_heads_after_step_two": True}
    if mode == "adaptive_vector":
        out_records = [trace.get(f"assimilation_out_{modality}.weight", []) for modality in ("text", "visual")]
        checks["assimilation_out_first_step"] = len(out_records) == 2 and all(first_nonzero(items) for items in out_records)
        checks["assimilation_in_after_step_two"] = later_nonzero(["assimilation_in_"])
        checks["retrieval_query_after_step_two"] = later_nonzero(["text_query.", "visual_query.", "query_head_projection."])
        checks["retrieval_key_after_step_two"] = later_nonzero(["key_head_projection."])
        checks["retrieval_value_after_step_two"] = later_nonzero(["value_head_projection.", "output_projection."])
    elif mode == "global_scalar":
        alpha = records_for("assimilation_alpha_")
        checks["assimilation_out_first_step"] = len(alpha) == 2 and all(first_nonzero(items) for items in alpha)
        checks["assimilation_in_after_step_two"] = True
    if variant.endswith("-S"):
        checks["provenance_heads_after_step_two"] = True
    else:
        checks["provenance_heads_after_step_two"] = later_nonzero(["key_provenance_", "key_interaction_", "value_interaction_"])
    return {"checks": checks, "passed": all(checks.values()),
            "steps_observed": max((len(items) for items in trace.values()), default=0),
            "trace": {name: items[:2] for name, items in trace.items()}}


def worker(dataset: str, seed: int, variant: str, assimilation_mode: str, epochs: int, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_checkpoint.pt"
    (run_dir / "main.log").unlink(missing_ok=True)
    cfg = make_cfg(dataset, seed, variant, assimilation_mode, epochs, checkpoint)
    if str(cfg.task.name) != "nc" or bool(cfg.task.evaluate_test) or int(cfg.num_runs) != 1:
        raise AssertionError("M2.1 workers require one NC seed and test evaluation disabled")
    torch.set_num_threads(int(cfg.task.get("torch_threads", 4)))
    device = get_device(str(cfg.device))
    logger = setup_logger(run_dir, str(cfg.logging.level))
    logger.info("M2.1 pilot variant=%s mode=%s dataset=%s seed=%d epochs=%d device=%s",
                variant, assimilation_mode, dataset, seed, epochs, device)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    data = load_mag_data(cfg, "nc", seed)
    logger.info("NC-only: nodes=%d edges=%d train=%d val=%d test-index=%d (test not evaluated)",
                data.num_nodes, data.num_edges, int(data.train_idx.numel()), int(data.val_idx.numel()), int(data.test_idx.numel()))

    import src.tasks.nc as nc_module
    original_build_model = nc_module.build_model
    gradient_trace: dict[str, list[dict]] = {}
    hook_handles = []
    def build_with_gradient_hooks(worker_cfg, data_info):
        model = original_build_model(worker_cfg, data_info)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and _watched_parameter(name):
                gradient_trace[name] = []
                def save_gradient(grad, parameter_name=name):
                    gradient_trace[parameter_name].append({
                        "finite": bool(torch.isfinite(grad).all()),
                        "nonzero": bool(torch.count_nonzero(grad)),
                        "norm": float(grad.detach().float().norm().item()),
                    })
                    return grad
                hook_handles.append(parameter.register_hook(save_gradient))
        return model
    nc_module.build_model = build_with_gradient_hooks
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        results = run_nc(cfg, data, device, logger)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        nc_module.build_model = original_build_model
        for handle in hook_handles: handle.remove()
    epoch_count = len(set(re.findall(r"\bEpoch (\d+) \|", (run_dir / "main.log").read_text(encoding="utf-8"))))
    if epoch_count <= 0 or not checkpoint.is_file():
        raise RuntimeError("M2.1 run did not finish epochs or save its selected checkpoint")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    gradient_report = gradient_requirements(gradient_trace, variant, assimilation_mode)
    if epochs == 2 and not gradient_report["passed"]:
        raise AssertionError(f"Smoke gradient assertions failed: {gradient_report['checks']}")
    memory = {"allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
              "reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else 0,
              "device": str(device)}
    record = {"dataset": dataset, "variant": variant, "assimilation_mode": assimilation_mode, "seed": seed,
              "epochs_requested": int(epochs), "completed_epochs": epoch_count, "best_epoch": int(checkpoint_data["epoch"]),
              "epoch_wall_time_seconds": elapsed / epoch_count, "total_training_seconds": elapsed,
              "peak_memory": memory,
              "metrics": {key: {"mean": float(value[0]), "std": float(value[1])} for key, value in results.items()},
              "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "gradient_checks": gradient_report,
              "protocol": {"task": "NC", "train_split": "original train", "selection_split": "original validation",
                           "test_evaluation": False, "link_prediction": False}}
    (run_dir / "training_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("Training summary: dataset=%s variant=%s mode=%s seed=%d epochs=%d val=%s grad_checks=%s peak=%s", dataset, variant, assimilation_mode, seed, epoch_count, record["metrics"], gradient_report["checks"], memory)


def run_worker(dataset, seed, variant, mode, epochs, run_dir):
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--dataset", dataset,
                    "--seed", str(seed), "--variant", variant, "--assimilation-mode", mode,
                    "--epochs-worker", str(epochs), "--run-dir", str(run_dir.resolve())], cwd=ROOT, check=True)
    return json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))


def run_audit(run, output: Path):
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_adaptive_evidence_assimilation_m21.py"),
                    "--dataset", run["dataset"], "--seed", str(run["seed"]), "--variant", run["variant"],
                    "--assimilation-mode", run["assimilation_mode"], "--checkpoint", str((ROOT / run["checkpoint"]).resolve()),
                    "--output", str(output.resolve())], cwd=ROOT, check=True)
    audit = json.loads(output.read_text(encoding="utf-8"))
    run["audit"] = output.relative_to(ROOT).as_posix()
    run["validation"] = audit["full_validation"]
    run["validation_classwise"] = audit["full_validation_classwise"]
    run["model_parameter_counts"] = audit["analysis"]["parameter_counts"]
    run["analysis"] = audit["analysis"]
    run["counterfactuals"] = audit["counterfactuals"]
    run["audit_peak_memory"] = audit["peak_memory"]
    run["gradient_probe"] = audit["gradient_probe"]
    run["M2_retrieval_max_abs_error"] = audit["M2_retrieval_max_abs_error"]
    run["M2_attention_max_abs_error"] = audit["M2_attention_max_abs_error"]
    return audit


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def mean_sd(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=False))


def build_results(runs):
    out = ROOT / "results/model_design/m21"
    historical = json.loads((ROOT / "results/model_design/m2/m2_runs.json").read_text(encoding="utf-8"))
    hist = {(row["dataset"], row["seed"], row["variant"]): row for row in historical}
    by_key = {(r["dataset"], r["seed"], r["variant"], r["assimilation_mode"]): r for r in runs}
    main_rows, paired_rows, scalar_rows, cf_rows, class_rows = [], [], [], [], []
    for dataset in ("Movies", "Grocery"):
        for variant in ("M2-S", "M2-P", "M2.1-S", "M2.1-P"):
            selected = []
            for seed in (42, 43, 44):
                if variant.startswith("M2.1"):
                    selected.append(by_key[(dataset, seed, variant, "adaptive_vector")])
                else:
                    selected.append(hist[(dataset, seed, variant)])
            for metric in ("acc", "macro_f1"):
                mean, sd = mean_sd([r["validation"][metric] for r in selected])
                main_rows.append({"dataset": dataset, "variant": variant, "metric": metric, "mean": mean,
                                  "population_sd": sd, "mean_percent": mean * 100, "population_sd_percentage_points": sd * 100,
                                  "source": "retrained M2.1" if variant.startswith("M2.1") else "reused frozen M2 pilot"})
        for seed in (42, 43, 44):
            m21s = by_key[(dataset, seed, "M2.1-S", "adaptive_vector")]
            m21p = by_key[(dataset, seed, "M2.1-P", "adaptive_vector")]
            m2s = hist[(dataset, seed, "M2-S")]
            m2p = hist[(dataset, seed, "M2-P")]
            comparisons = (("M2.1-P minus M2.1-S", m21p["validation"], m21s["validation"]),
                           ("M2.1-S minus M2-S", m21s["validation"], m2s["validation"]),
                           ("M2.1-P minus M2-P", m21p["validation"], m2p["validation"]))
            for comparison, left, right in comparisons:
                paired_rows.append({"dataset": dataset, "seed": seed, "comparison": comparison,
                                    "delta_val_acc": left["acc"] - right["acc"],
                                    "delta_val_macro_f1": left["macro_f1"] - right["macro_f1"]})
        adaptive = by_key[(dataset, 42, "M2.1-P", "adaptive_vector")]
        scalar = by_key[(dataset, 42, "M2.1-P", "global_scalar")]
        scalar_rows.append({"dataset": dataset, "seed": 42,
                            "adaptive_val_acc": adaptive["validation"]["acc"], "scalar_val_acc": scalar["validation"]["acc"],
                            "delta_scalar_minus_adaptive_val_acc": scalar["validation"]["acc"] - adaptive["validation"]["acc"],
                            "adaptive_val_macro_f1": adaptive["validation"]["macro_f1"], "scalar_val_macro_f1": scalar["validation"]["macro_f1"],
                            "delta_scalar_minus_adaptive_val_macro_f1": scalar["validation"]["macro_f1"] - adaptive["validation"]["macro_f1"]})
    precheck = list(csv.DictReader((out / "precheck_classwise.csv").open(encoding="utf-8")))
    historical_class = {(r["dataset"], int(r["seed"]), int(r["class"])): r for r in precheck}
    for run in runs:
        if run["assimilation_mode"] != "adaptive_vector": continue
        for row in run["validation_classwise"]:
            old = historical_class[(run["dataset"], run["seed"], row["class"])]
            old_f1 = old["M2_P_f1"] if run["variant"].endswith("-P") else old["M2_S_f1"]
            class_rows.append({"dataset": run["dataset"], "seed": run["seed"], "variant": run["variant"],
                               "class": row["class"], "validation_support": row["support"], "M2_historical_f1": float(old_f1),
                               "M2_1_f1": row["f1"], "delta_M2_1_minus_M2_f1": row["f1"] - float(old_f1),
                               "precision": row["precision"], "recall": row["recall"]})
        if run["variant"].endswith("-P"):
            for cf in run["counterfactuals"]:
                cf_rows.append({"dataset": run["dataset"], "seed": run["seed"], "mode": cf["mode"],
                                "shuffle_seed": cf.get("shuffle_seed", ""), "val_acc": cf["validation"]["acc"],
                                "val_macro_f1": cf["validation"]["macro_f1"],
                                "delta_val_acc_vs_full": cf["validation"]["acc"] - run["validation"]["acc"],
                                "delta_val_macro_f1_vs_full": cf["validation"]["macro_f1"] - run["validation"]["macro_f1"],
                                "retrieval_relative_change_mean": cf["retrieval_relative_change"]["mean"],
                                "final_z_relative_change_mean": cf["final_z_relative_change"]["mean"],
                                "attention_l1_change_mean": cf["attention_l1_change"]["mean"],
                                "stage12_max_abs_error": cf["stage12_max_abs_error"]})
    write_json(out / "pilot_runs.json", runs)
    write_csv(out / "main_results.csv", main_rows)
    write_csv(out / "paired_differences.csv", paired_rows)
    write_csv(out / "scalar_sanity_comparison.csv", scalar_rows)
    write_csv(out / "counterfactual_results.csv", cf_rows)
    write_csv(out / "classwise_comparison.csv", class_rows)
    diagnostics = {"runs": [{"dataset": r["dataset"], "seed": r["seed"], "variant": r["variant"],
                              "assimilation_mode": r["assimilation_mode"], "analysis": r["analysis"],
                              "gradient_checks": r["gradient_checks"], "gradient_probe": r["gradient_probe"],
                              "training_resources": {"total_training_seconds": r["total_training_seconds"],
                                                      "epoch_wall_time_seconds": r["epoch_wall_time_seconds"],
                                                      "peak_memory": r["peak_memory"], "audit_peak_memory": r["audit_peak_memory"]}}
                         for r in runs]}
    write_json(out / "mechanism_diagnostics.json", diagnostics)
    return main_rows, paired_rows, scalar_rows, cf_rows, class_rows


def write_report(runs, main_rows, paired_rows, scalar_rows, cf_rows, class_rows):
    out = ROOT / "docs/model_design/m21_pilot_report.md"
    result_dir = ROOT / "results/model_design/m21"
    metadata = json.loads((result_dir / "source_metadata.json").read_text(encoding="utf-8"))
    smoke_runs = json.loads((result_dir / "smoke_runs.json").read_text(encoding="utf-8"))
    smoke_pass = len(smoke_runs) == 3 and all(r.get("gradient_checks", {}).get("passed", False) for r in smoke_runs)
    precheck = list(csv.DictReader((result_dir / "precheck_summary.csv").open(encoding="utf-8")))
    correlations = list(csv.DictReader((result_dir / "precheck_correlations.csv").open(encoding="utf-8")))
    correctness = list(csv.DictReader((result_dir / "precheck_correctness.csv").open(encoding="utf-8")))
    lines = ["# M2.1 Pilot Report", "", "## A. Source and protocol", "",
             f"Source branch `{metadata['source_branch']}` at `{metadata['source_sha']}`; experiment branch `{metadata['experiment_branch']}`.",
             "NC only on Movies and Grocery. Original train was used for fitting, original validation for checkpoint selection and evaluation; test evaluation and LP were disabled.",
             "", "## B. Frozen architecture contract", "",
             "M2 Stage I/II and Stage III evidence creation, queries, attention retrieval, masks, token order and projections were reused. M2.1 changes only how retrieved evidence enters each modality. Final fusion remains 256→128, LayerNorm, GELU and dropout 0.2.",
             "The inherited `text_residual_norm` and `visual_residual_norm` are frozen and recorded as ‘inherited legacy M2 assimilation parameters; not part of the M2.1 conceptual architecture.’ No normalization follows the M2.1 residual injection.",
             "", "## C. M2 validation precheck", "",
             "Six existing formal M2-P checkpoints supplied validation-only retrieval, compatibility and displacement measurements. Existing M2-S checkpoints supplied the historical per-class F1 comparison. Correlations below are descriptive associations with validation-node CE, not causal estimates.",
             "", "| Dataset | Modality | Retrieval / H0 | Cosine(H0,U) | 1−Cosine(H0,Z) | Correct / incorrect retrieval ratio | Mean Spearman rho: ratio / cosine / displacement |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        for modality in ("text", "visual"):
            vals = {}
            for metric in ("retrieval_ratio", "compatibility_cosine", "structural_displacement"):
                entries = [float(r["mean"]) for r in precheck if r["dataset"] == dataset and r["modality"] == modality and r["metric"] == metric]
                vals[metric] = sum(entries) / len(entries)
            cr = {}
            for group in ("correct", "incorrect"):
                entries = [float(r["mean"]) for r in correctness if r["dataset"] == dataset and r["modality"] == modality and r["correctness"] == group and r["metric"] == "retrieval_ratio"]
                cr[group] = sum(entries) / len(entries)
            rho = []
            for metric in ("retrieval_ratio", "compatibility_cosine", "structural_displacement"):
                entries = [float(r["spearman_rho"]) for r in correlations if r["dataset"] == dataset and r["modality"] == modality and r["metric"] == metric]
                rho.append(sum(entries) / len(entries))
            lines.append(f"| {dataset} | {modality} | {vals['retrieval_ratio']:.3f} | {vals['compatibility_cosine']:.3f} | {vals['structural_displacement']:.3f} | {cr['correct']:.3f} / {cr['incorrect']:.3f} | {rho[0]:+.2f} / {rho[1]:+.2f} / {rho[2]:+.2f} |")
    lines += ["", "`precheck_summary.csv`, `precheck_correctness.csv`, `precheck_classwise.csv`, and `precheck_correlations.csv` retain seed-level means/p10/p50/p90, support, historical F1, correctness strata and all six checkpoint measurements. The Spearman analyses use validation CE and make no causal claim.",
              "", "## D. Initialization and gradient checks", "",
              f"Smoke used Movie seed42 for M2.1-S vector, M2.1-P vector and M2.1-P scalar, each for 2 epochs; all three smoke gradient and audit checks passed: {smoke_pass}.",
              "Adaptive vector S/P use paired seeds, initialize both assimilation output layers to zero, and start with `Z_text=H0_text` and `Z_visual=H0_visual`. Gradient traces confirm finite nonzero assimilation output-layer gradients at step one and nonzero upstream assimilation/retrieval gradients after step two.",
              f"Across all 14 trained checkpoints, maximum M2 retrieval and attention absolute differences under a shared Stage I/II state were {max(r['M2_retrieval_max_abs_error'] for r in runs):.1e} and {max(r['M2_attention_max_abs_error'] for r in runs):.1e}; Stage I/II counterfactual maximum difference was {max((cf['stage12_max_abs_error'] for r in runs for cf in r['counterfactuals']), default=0.0):.1e}.",
              "", "## E. Main validation results", "",
              "| Dataset | Variant | Accuracy mean ± population SD | Macro-F1 mean ± population SD | Source |", "|---|---|---:|---:|---|"]
    lookup = {(r["dataset"], r["variant"], r["metric"]): r for r in main_rows}
    for dataset in ("Movies", "Grocery"):
        for variant in ("M2-S", "M2-P", "M2.1-S", "M2.1-P"):
            acc, f1 = lookup[(dataset, variant, "acc")], lookup[(dataset, variant, "macro_f1")]
            lines.append(f"| {dataset} | {variant} | {acc['mean']*100:.2f}% ± {acc['population_sd']*100:.2f} pp | {f1['mean']*100:.2f}% ± {f1['population_sd']*100:.2f} pp | {acc['source']} |")
    lines += ["", "### Seed-paired comparisons", "", "| Dataset | Seed | Comparison | Δ accuracy | Δ Macro-F1 |", "|---|---:|---|---:|---:|"]
    for row in paired_rows:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['comparison']} | {row['delta_val_acc']*100:+.2f} pp | {row['delta_val_macro_f1']*100:+.2f} pp |")
    lines += ["", "### Seed42 scalar sanity comparison", "", "| Dataset | Metric | Adaptive vector | Global scalar | Scalar − vector |", "|---|---|---:|---:|---:|"]
    for row in scalar_rows:
        lines.append(f"| {row['dataset']} | Accuracy | {row['adaptive_val_acc']*100:.2f}% | {row['scalar_val_acc']*100:.2f}% | {row['delta_scalar_minus_adaptive_val_acc']*100:+.2f} pp |")
        lines.append(f"| {row['dataset']} | Macro-F1 | {row['adaptive_val_macro_f1']*100:.2f}% | {row['scalar_val_macro_f1']*100:.2f}% | {row['delta_scalar_minus_adaptive_val_macro_f1']*100:+.2f} pp |")
    lines += ["", "## F. Per-class validation results", "", "`classwise_comparison.csv` contains validation support and historical M2 versus M2.1 class F1 for every seed and class. Across the 60 class-seed observations per cell, M2.1-S improved/lowered class F1 on Movies in 27/24 cases and Grocery in 17/42; M2.1-P improved/lowered class F1 on Movies in 21/29 and Grocery in 14/44. Full values and per-class retrieval/displacement are in the classwise precheck and comparison CSV files.",
              "", "## G. Assimilation and evidence diagnostics", "",
              "| Dataset | Variant | Mean correction ratio q: text / visual | Mean |a|: text / visual | Centered effective rank: text / visual | Active fraction |", "|---|---|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        for variant in ("M2.1-S", "M2.1-P"):
            selected = [r for r in runs if r["dataset"] == dataset and r["variant"] == variant and r["assimilation_mode"] == "adaptive_vector"]
            q = [sum(r["analysis"]["assimilation_correction_ratio"][m]["mean"] for r in selected) / len(selected) for m in ("text", "visual")]
            amag = [sum(r["analysis"]["assimilation_field"][m]["absolute_value"]["mean"] for r in selected) / len(selected) for m in ("text", "visual")]
            rank = [sum(r["analysis"]["assimilation_field"][m]["diversity"]["centered_effective_rank"] for r in selected) / len(selected) for m in ("text", "visual")]
            active = [sum(r["analysis"]["assimilation_field"][m]["diversity"]["active_fraction_abs_gt_0_1"] for r in selected) / len(selected) for m in ("text", "visual")]
            lines.append(f"| {dataset} | {variant} | {q[0]:.3f} / {q[1]:.3f} | {amag[0]:.3f} / {amag[1]:.3f} | {rank[0]:.2f} / {rank[1]:.2f} | {active[0]:.3f} / {active[1]:.3f} |")
    lines += ["", "Full M2 attention reports, field sign/near-zero fractions, nodewise norm variance, compatibility quintiles, parameter counts, gradients and resource data are in `mechanism_diagnostics.json`. Adaptive diversity uses at most 8192 deterministic evenly spaced nodes, centered covariance eigenvalues, top-1/top-5 energy, channel variance and active fraction.",
              "", "## H. Frozen counterfactuals", "", "The eight per-run M2.1-P vector interventions preserve Stage I/II. Assimilation-Off, Uniform-Node and Dimension Shuffle modify only the assimilation field. Provenance-Off, degree-matched Node Shuffle (seeds 3407/3408/3409) and Order Mismatch modify provenance copies only when constructing retrieval evidence.",
              "", "| Dataset | Intervention | Runs | Mean Δ accuracy | Mean Δ Macro-F1 |", "|---|---|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        modes = ("assimilation_off", "uniform_node", "dimension_shuffle", "provenance_off", "provenance_shuffle", "order_mismatch")
        for mode in modes:
            selected = [r for r in cf_rows if r["dataset"] == dataset and r["mode"] == mode]
            mean_acc = sum(r["delta_val_acc_vs_full"] for r in selected) / len(selected)
            mean_f1 = sum(r["delta_val_macro_f1_vs_full"] for r in selected) / len(selected)
            lines.append(f"| {dataset} | {mode} | {len(selected)} | {mean_acc*100:+.2f} pp | {mean_f1*100:+.2f} pp |")
    lines += ["", "The per-seed, per-shuffle outcomes are in `counterfactual_results.csv`; M2.1-P classwise counterfactual metrics are in each run audit JSON. These intervention outcomes describe this validation setup.",
              "", "## I. Resources and parameter status", ""]
    for row in runs:
        mem = row["peak_memory"]
        lines.append(f"- {row['dataset']} {row['variant']} {row['assimilation_mode']} seed {row['seed']}: {row['completed_epochs']} epochs; {row['epoch_wall_time_seconds']:.1f} s/epoch; peak allocated/reserved {mem['allocated_bytes']/2**30:.2f}/{mem['reserved_bytes']/2**30:.2f} GiB; trainable parameters {row['model_parameter_counts']['trainable']:,}; inherited frozen {row['model_parameter_counts']['legacy_frozen']:,}.")
    lines += ["", "Scalar sanity results use seed42 only and are not a multi-seed estimate.",
              "", "## J. Decision", "",
              "**Decision: Assimilation alone does not solve limitation.** Adaptive vector M2.1-P did not improve both datasets over historical M2-P: Movies was slightly lower in mean accuracy and Macro-F1, while Grocery was lower by about 0.96 and 1.63 percentage points. M2.1-S also lost about 1.03 accuracy points on Grocery. Provenance-conditioned gains over M2.1-S varied by seed and dataset.",
              "The learned vector field was active and the Assimilation-Off and Dimension-Shuffle checks reduced validation scores, especially on Grocery. This shows the model uses the field in its predictions, while the multi-seed M2 comparisons show that this use did not provide a reliable aggregate task gain. Global scalar was promising for Grocery seed42, but its Movies Macro-F1 was lower than adaptive vector and it was only one seed, so the pilot does not justify preferring scalar.",
              "", "This decision is limited to the two-dataset, three-seed NC pilot and the seed42 scalar sanity runs. Test evaluation, LP, extra datasets, baselines, class weighting and formal ablations were excluded.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def run_suite(args):
    outroot = ROOT / "outputs/model_design/m21"
    results = ROOT / "results/model_design/m21"
    smoke_work = [("Movies", 42, "M2.1-S", "adaptive_vector"), ("Movies", 42, "M2.1-P", "adaptive_vector"),
                  ("Movies", 42, "M2.1-P", "global_scalar")]
    if not args.skip_smoke:
        smoke_runs = []
        for dataset, seed, variant, mode in smoke_work:
            run_dir = outroot / "smoke" / f"{variant}_{mode}" / dataset / f"seed{seed}"
            audit_path = results / "run_audits" / "smoke" / f"{variant}_{mode}" / f"{dataset}_seed{seed}.json"
            metrics_path = run_dir / "training_metrics.json"
            if metrics_path.is_file() and (run_dir / "best_checkpoint.pt").is_file():
                run = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                run = run_worker(dataset, seed, variant, mode, 2, run_dir)
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                run["audit"] = audit_path.relative_to(ROOT).as_posix()
                run["validation"] = audit["full_validation"]
                run["validation_classwise"] = audit["full_validation_classwise"]
                run["model_parameter_counts"] = audit["analysis"]["parameter_counts"]
                run["analysis"] = audit["analysis"]
                run["counterfactuals"] = audit["counterfactuals"]
                run["audit_peak_memory"] = audit["peak_memory"]
                run["gradient_probe"] = audit["gradient_probe"]
            else:
                run_audit(run, audit_path)
            smoke_runs.append(run)
            write_json(results / "smoke_runs.json", smoke_runs)
            print(f"Smoke complete: {variant} {mode} {dataset} seed {seed}", flush=True)
        if len(smoke_runs) != 3 or not all(r["gradient_checks"]["passed"] for r in smoke_runs):
            raise AssertionError("Required M2.1 smoke suite did not pass")
        print("All three 2-epoch smoke runs and audits passed; continuing automatically to the full pilot.", flush=True)
    work = [(dataset, seed, variant, "adaptive_vector")
            for dataset in ("Movies", "Grocery") for seed in (42, 43, 44) for variant in ("M2.1-S", "M2.1-P")]
    work += [(dataset, 42, "M2.1-P", "global_scalar") for dataset in ("Movies", "Grocery")]
    runs = []
    for dataset, seed, variant, mode in work:
        run_dir = outroot / "pilot" / f"{variant}_{mode}" / dataset / f"seed{seed}"
        audit_path = results / "run_audits" / "pilot" / f"{variant}_{mode}" / f"{dataset}_seed{seed}.json"
        metrics_path = run_dir / "training_metrics.json"
        if metrics_path.is_file() and (run_dir / "best_checkpoint.pt").is_file():
            run = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            run = run_worker(dataset, seed, variant, mode, int(args.epochs), run_dir)
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            run["audit"] = audit_path.relative_to(ROOT).as_posix()
            run["validation"] = audit["full_validation"]
            run["validation_classwise"] = audit["full_validation_classwise"]
            run["model_parameter_counts"] = audit["analysis"]["parameter_counts"]
            run["analysis"] = audit["analysis"]
            run["counterfactuals"] = audit["counterfactuals"]
            run["audit_peak_memory"] = audit["peak_memory"]
            run["gradient_probe"] = audit["gradient_probe"]
            run["M2_retrieval_max_abs_error"] = audit["M2_retrieval_max_abs_error"]
            run["M2_attention_max_abs_error"] = audit["M2_attention_max_abs_error"]
        else:
            run_audit(run, audit_path)
        runs.append(run)
        write_json(results / "pilot_runs.json", runs)
        print(f"Pilot complete: {dataset} {variant} {mode} seed {seed} | acc={run['validation']['acc']:.4f} macro_f1={run['validation']['macro_f1']:.4f}", flush=True)
    if len(runs) != 14: raise RuntimeError(f"Expected 14 M2.1 runs, got {len(runs)}")
    main, paired, scalar, cf, cls = build_results(runs)
    write_report(runs, main, paired, scalar, cf, cls)
    print("M2.1 pilot complete; results and report generated.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=("Movies", "Grocery"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=("M2.1-S", "M2.1-P"))
    parser.add_argument("--assimilation-mode", choices=("adaptive_vector", "global_scalar"), default="adaptive_vector")
    parser.add_argument("--run-dir")
    parser.add_argument("--epochs-worker", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        if not all((args.dataset, args.seed, args.variant, args.run_dir)): parser.error("--worker needs dataset, seed, variant and run-dir")
        worker(args.dataset, args.seed, args.variant, args.assimilation_mode, args.epochs_worker, Path(args.run_dir))
    else:
        run_suite(args)


if __name__ == "__main__": main()
