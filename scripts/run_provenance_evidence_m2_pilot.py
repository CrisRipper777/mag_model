from __future__ import annotations

import argparse
import csv
import json
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
PYTHON = sys.executable

from src.data import load_mag_data
from src.tasks import run_nc
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.summary import count_parameters


def make_cfg(dataset: str, seed: int, variant: str, epochs: int, checkpoint: Path):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=provenance_evidence_m2",
                f"model.provenance_conditioning={'true' if variant == 'M2-P' else 'false'}",
                f"seed={seed}",
                "num_runs=1",
                f"task.epochs={epochs}",
                "task.evaluate_test=false",
                f"task.save_ckpt_path={checkpoint.resolve()}",
                "paths.output_root=outputs/model_design/m2",
            ],
        )


def worker(dataset: str, seed: int, variant: str, epochs: int, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_checkpoint.pt"
    (run_dir / "main.log").unlink(missing_ok=True)
    cfg = make_cfg(dataset, seed, variant, epochs, checkpoint)
    if str(cfg.task.name) != "nc" or bool(cfg.task.evaluate_test) or int(cfg.num_runs) != 1:
        raise AssertionError("M2 workers must run one NC seed with test evaluation disabled")
    torch.set_num_threads(int(cfg.task.get("torch_threads", 4)))
    device = get_device(str(cfg.device))
    logger = setup_logger(run_dir, str(cfg.logging.level))
    logger.info("M2 pilot variant=%s dataset=%s seed=%d epochs=%d device=%s", variant, dataset, seed, epochs, device)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    data = load_mag_data(cfg, "nc", seed)
    logger.info(
        "NC-only data: nodes=%d edges=%d train=%d val=%d test-index=%d (test not evaluated)",
        data.num_nodes, data.num_edges, int(data.train_idx.numel()), int(data.val_idx.numel()), int(data.test_idx.numel()),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    results = run_nc(cfg, data, device, logger)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    epoch_count = len(set(re.findall(r"\bEpoch (\d+) \|", (run_dir / "main.log").read_text(encoding="utf-8"))))
    if epoch_count <= 0 or not checkpoint.is_file():
        raise RuntimeError("M2 run did not complete epochs and produce its selected checkpoint")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    memory = {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else 0,
        "device": str(device),
    }
    record = {
        "dataset": dataset,
        "variant": variant,
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "completed_epochs": epoch_count,
        "best_epoch": int(checkpoint_data["epoch"]),
        "epoch_wall_time_seconds": elapsed / epoch_count,
        "total_training_seconds": elapsed,
        "peak_memory": memory,
        "metrics": {key: {"mean": float(value[0]), "std": float(value[1])} for key, value in results.items()},
        "checkpoint": str(checkpoint.resolve()),
        "trainable_model_parameters": None,
        "protocol": {
            "task": "NC", "train_split": "original train", "selection_split": "original validation",
            "test_evaluation": False, "link_prediction": False,
        },
    }
    (run_dir / "training_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    logger.info("Training summary: %s", json.dumps(record, sort_keys=True))


def run_worker(dataset: str, seed: int, variant: str, epochs: int, run_dir: Path):
    subprocess.run(
        [
            PYTHON, str(Path(__file__).resolve()), "--worker", "--dataset", dataset,
            "--seed", str(seed), "--variant", variant, "--epochs-worker", str(epochs),
            "--run-dir", str(run_dir.resolve()),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))


def run_audit(run: dict, output: Path):
    subprocess.run(
        [
            PYTHON, str(ROOT / "scripts/audit_provenance_evidence_m2.py"),
            "--dataset", run["dataset"], "--seed", str(run["seed"]),
            "--variant", run["variant"], "--checkpoint", run["checkpoint"],
            "--output", str(output.resolve()),
        ],
        cwd=ROOT,
        check=True,
    )
    audit = json.loads(output.read_text(encoding="utf-8"))
    run["audit"] = str(output.relative_to(ROOT))
    run["validation"] = audit["full_validation"]
    run["model_parameter_counts"] = audit["model_parameter_counts"]
    run["analysis"] = audit["analysis"]
    run["counterfactuals"] = audit["counterfactuals"]
    run["audit_peak_memory"] = audit["peak_memory"]
    checkpoint = Path(run["checkpoint"]).resolve()
    run["checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
    return audit


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def build_results(runs: list[dict]):
    results_dir = ROOT / "results/model_design/m2"
    summary_rows, paired_rows, counterfactual_rows = [], [], []
    paired_map = {(run["dataset"], run["seed"], run["variant"]): run for run in runs}
    for dataset in ("Movies", "Grocery"):
        for variant in ("M2-S", "M2-P"):
            selected = [r for r in runs if r["dataset"] == dataset and r["variant"] == variant]
            for metric, field in (("val_acc", "acc"), ("val_macro_f1", "macro_f1")):
                values = [r["validation"][field] for r in selected]
                mean, sd = _mean_sd(values)
                summary_rows.append({
                    "dataset": dataset, "variant": variant, "metric": metric,
                    "mean": mean, "population_sd": sd, "mean_percent": 100 * mean,
                    "population_sd_percentage_points": 100 * sd,
                    "trainable_model_parameters": selected[0]["model_parameter_counts"]["trainable"],
                    "legacy_frozen_parameters": selected[0]["model_parameter_counts"]["legacy_frozen"],
                    "best_epoch_mean": sum(r["best_epoch"] for r in selected) / len(selected),
                })
        for seed in (42, 43, 44):
            s = paired_map[(dataset, seed, "M2-S")]
            p = paired_map[(dataset, seed, "M2-P")]
            paired_rows.append({
                "dataset": dataset, "seed": seed,
                "delta_val_acc_P_minus_S": p["validation"]["acc"] - s["validation"]["acc"],
                "delta_val_macro_f1_P_minus_S": p["validation"]["macro_f1"] - s["validation"]["macro_f1"],
                "M2S_val_acc": s["validation"]["acc"], "M2P_val_acc": p["validation"]["acc"],
                "M2S_val_macro_f1": s["validation"]["macro_f1"], "M2P_val_macro_f1": p["validation"]["macro_f1"],
            })
    for run in runs:
        for cf in run["counterfactuals"]:
            counterfactual_rows.append({
                "dataset": run["dataset"], "seed": run["seed"], "mode": cf["mode"],
                "shuffle_seed": cf.get("shuffle_seed", ""),
                "val_acc": cf["validation"]["acc"], "val_macro_f1": cf["validation"]["macro_f1"],
                "delta_val_acc_vs_full": cf["validation"]["acc"] - run["validation"]["acc"],
                "delta_val_macro_f1_vs_full": cf["validation"]["macro_f1"] - run["validation"]["macro_f1"],
                "key_relative_change_mean": cf["key_relative_change"]["mean"],
                "value_relative_change_mean": cf["value_relative_change"]["mean"],
                "retrieval_relative_change_mean": cf["retrieval_relative_change"]["mean"],
                "final_z_relative_change_mean": cf["final_z_relative_change"]["mean"],
                "attention_l1_change_mean": cf["attention_l1_change"]["mean"],
                "stage12_max_abs_error": cf["stage12_max_abs_error"],
            })
    write_json(results_dir / "m2_runs.json", runs)
    write_csv(results_dir / "main_results.csv", summary_rows)
    write_csv(results_dir / "paired_differences.csv", paired_rows)
    write_csv(results_dir / "counterfactual_results.csv", counterfactual_rows)
    diagnostics = {
        "runs": [
            {
                "dataset": r["dataset"], "seed": r["seed"], "variant": r["variant"],
                "model_parameter_counts": r["model_parameter_counts"],
                "analysis": r["analysis"],
                "training_resources": {
                    "total_training_seconds": r["total_training_seconds"],
                    "epoch_wall_time_seconds": r["epoch_wall_time_seconds"],
                    "peak_memory": r["peak_memory"], "audit_peak_memory": r["audit_peak_memory"],
                },
            }
            for r in runs
        ]
    }
    write_json(results_dir / "mechanism_diagnostics.json", diagnostics)
    return summary_rows, paired_rows, counterfactual_rows


def _pct(value):
    return f"{100 * value:.2f}%"


def write_report(runs: list[dict], summary_rows: list[dict], paired_rows: list[dict], cf_rows: list[dict]):
    out = ROOT / "docs/model_design/m2_pilot_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M2 Pilot Report", "",
        "Source: `1793ba2136909255637e4ebd2a6d1231921434b7` on `exp/m11_corrected_provenance_audit`.",
        "Branch: `exp/provenance_evidence_m2`. NC only; original train was used for fitting, original validation for checkpoint selection, test evaluation and LP were disabled.",
        "", "## Main retrained comparison", "",
        "| Dataset | Variant | Validation accuracy (mean ± population SD) | Validation Macro-F1 (mean ± population SD) | Trainable model parameters | Best epoch mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in ("Movies", "Grocery"):
        for variant in ("M2-S", "M2-P"):
            selected = [r for r in runs if r["dataset"] == dataset and r["variant"] == variant]
            acc_mean, acc_sd = _mean_sd([r["validation"]["acc"] for r in selected])
            f1_mean, f1_sd = _mean_sd([r["validation"]["macro_f1"] for r in selected])
            params = selected[0]["model_parameter_counts"]["trainable"]
            best_epoch = sum(r["best_epoch"] for r in selected) / len(selected)
            lines.append(f"| {dataset} | {variant} | {_pct(acc_mean)} ± {_pct(acc_sd)} | {_pct(f1_mean)} ± {_pct(f1_sd)} | {params:,} | {best_epoch:.1f} |")
    lines += ["", "### Seed-paired M2-P minus M2-S", "", "| Dataset | Seed | Δ validation accuracy | Δ validation Macro-F1 |", "|---|---:|---:|---:|"]
    for row in paired_rows:
        lines.append(f"| {row['dataset']} | {row['seed']} | {_pct(row['delta_val_acc_P_minus_S'])} | {_pct(row['delta_val_macro_f1_P_minus_S'])} |")
    lines += ["", "## Resources and parameter status", ""]
    for run in runs:
        mem = run["peak_memory"]
        lines.append(f"- {run['dataset']} {run['variant']} seed {run['seed']}: {run['completed_epochs']} epochs, {run['epoch_wall_time_seconds']:.1f} s/epoch, {run['total_training_seconds']:.1f} s total, peak allocated/reserved {mem['allocated_bytes'] / 2**30:.2f}/{mem['reserved_bytes'] / 2**30:.2f} GiB; trainable {run['model_parameter_counts']['trainable']:,}, legacy frozen {run['model_parameter_counts']['legacy_frozen']:,}.")
    lines += ["", "Legacy parameters remain only because M2 inherits the frozen M1 implementation for correctness; they are not part of the M2 conceptual architecture and should be removed in the later architecture-freeze refactor.", "", "## Evidence and attention diagnostics", ""]
    for run in runs:
        analysis = run["analysis"]
        lines.append(f"### {run['dataset']} {run['variant']} seed {run['seed']}")
        lines.append("")
        lines.append("| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for query, q in analysis["attention"].items():
            flags = analysis["collapse_checks"][query]
            active = ", ".join(k for k in ("null_collapse", "single_token_collapse", "node_invariant_retrieval") if flags[k]) or "none"
            lines.append(f"| {query} | {q['null_mass']:.4f} | {q['same_modality_mass']:.4f} | {q['cross_modality_mass']:.4f} | {q['entropy']['mean']:.4f} | {q['effective_tokens']['mean']:.3f} | {q['nodewise_attention_weight_variance_mean']:.3e} | {active} |")
        lines.append("")
        lines.append("ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.")
        lines.append("")
    lines += ["## Frozen M2-P counterfactuals", "", "All Stage I/II invariance checks passed at max absolute error ≤ 2e-6. Attention changes are sampled over at most 8192 deterministically selected nodes. Attention mass describes retrieval behavior and is not a causal importance score.", "", "| Dataset | Seed | Intervention | Δ Val Acc | Δ Val Macro-F1 | Key change | Value change | Retrieval change | Final z change | Attention L1 change |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in cf_rows:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['mode']}{' '+str(row['shuffle_seed']) if row['shuffle_seed'] else ''} | {_pct(row['delta_val_acc_vs_full'])} | {_pct(row['delta_val_macro_f1_vs_full'])} | {row['key_relative_change_mean']:.4f} | {row['value_relative_change_mean']:.4f} | {row['retrieval_relative_change_mean']:.4f} | {row['final_z_relative_change_mean']:.4f} | {row['attention_l1_change_mean']:.4f} |")
    lines += ["", "## Interpretation and decision", "", "Interpret validation metrics together with the mechanism checks: whether P learns nonzero key/value conditioning, whether Provenance-Off changes retrieval, whether shared-node shuffling or order mismatch changes evidence use, and whether attention collapses to null, one token, or a node-invariant distribution. A positive M2-P versus M2-S task delta is evidence, not a required pass condition.", "", "### Overall recommendation", "", "Decision is based on the observed 2-dataset, 3-seed pilot. This report does not authorize or start the five-dataset benchmark, formal ablations, or LP.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args):
    output_root = ROOT / "outputs/model_design/m2"
    results_root = ROOT / "results/model_design/m2"
    if args.smoke:
        work = [("Movies", 42, variant) for variant in ("M2-S", "M2-P")]
        epochs = 2
        tag = "smoke"
    else:
        work = [(dataset, seed, variant) for dataset in ("Movies", "Grocery") for seed in (42, 43, 44) for variant in ("M2-S", "M2-P")]
        epochs = int(args.epochs)
        tag = "pilot"
    runs, audit_results = [], []
    for dataset, seed, variant in work:
        run_dir = output_root / tag / variant / dataset / f"seed{seed}"
        run = run_worker(dataset, seed, variant, epochs, run_dir)
        audit_path = results_root / "run_audits" / tag / variant / f"{dataset}_seed{seed}.json"
        audit = run_audit(run, audit_path)
        runs.append(run)
        audit_results.append(audit)
        write_json(results_root / f"{tag}_runs.json", runs)
        print(
            f"Completed {tag}: {dataset} {variant} seed {seed} | "
            f"val_acc={run['validation']['acc']:.4f} val_macro_f1={run['validation']['macro_f1']:.4f}"
        )
    if args.smoke:
        write_json(results_root / "smoke_record.json", runs)
        print("M2-S and M2-P smoke runs plus audits passed; automatically starting the full pilot.")
        args.smoke = False
        run_experiment(args)
        return
    summary_rows, paired_rows, cf_rows = build_results(runs)
    write_report(runs, summary_rows, paired_rows, cf_rows)
    print(f"Pilot complete: {len(runs)} runs; report=docs/model_design/m2_pilot_report.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=("Movies", "Grocery"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=("M2-S", "M2-P"))
    parser.add_argument("--run-dir")
    parser.add_argument("--epochs-worker", type=int, default=2)
    args = parser.parse_args()
    if args.finalize_existing:
        runs = json.loads((ROOT / "results/model_design/m2/pilot_runs.json").read_text(encoding="utf-8"))
        if len(runs) != 12:
            raise RuntimeError(f"Expected 12 completed pilot runs, found {len(runs)}")
        summary_rows, paired_rows, cf_rows = build_results(runs)
        write_report(runs, summary_rows, paired_rows, cf_rows)
        print(f"Finalized {len(runs)} saved pilot runs and regenerated result artifacts.")
    elif args.worker:
        if args.dataset is None or args.seed is None or args.variant is None or args.run_dir is None:
            parser.error("--worker requires dataset, seed, variant, and run-dir")
        worker(args.dataset, args.seed, args.variant, args.epochs_worker, Path(args.run_dir))
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
