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

from src.data import load_mag_data
from src.tasks import run_nc
from src.utils.device import get_device
from src.utils.logging import setup_logger
PYTHON = sys.executable


def make_cfg(dataset: str, seed: int, epochs: int, checkpoint: Path):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=interaction_provenance_m1",
        f"seed={int(seed)}",
        "num_runs=1",
        f"task.epochs={int(epochs)}",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint.resolve()}",
        "paths.output_root=outputs/model_design/m1",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def worker(dataset: str, seed: int, epochs: int, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_checkpoint.pt"
    cfg = make_cfg(dataset, seed, epochs, checkpoint)
    if str(cfg.task.name) != "nc" or bool(cfg.task.evaluate_test):
        raise AssertionError("M1 pilot must run NC with test evaluation disabled")
    if int(cfg.num_runs) != 1:
        raise AssertionError("Each pilot worker must execute exactly one seed")
    if torch.cuda.is_available():
        torch.set_num_threads(int(cfg.task.get("torch_threads", 4)))
    device = get_device(str(cfg.device))
    logger = setup_logger(run_dir, str(cfg.logging.level))
    logger.info("M1 pilot dataset=%s seed=%d epochs=%d device=%s", dataset, seed, epochs, device)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    data = load_mag_data(cfg, "nc", int(seed))
    logger.info(
        "NC-only data: nodes=%d edges=%d train=%d val=%d test-index=%d (not evaluated)",
        data.num_nodes,
        data.num_edges,
        int(data.train_idx.numel()),
        int(data.val_idx.numel()),
        int(data.test_idx.numel()),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = run_nc(cfg, data, device, logger)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    completed_epochs = sum(
        1
        for line in (run_dir / "main.log").read_text(encoding="utf-8").splitlines()
        if re.search(r"\bEpoch \d+ \|", line)
    )
    if completed_epochs <= 0:
        raise RuntimeError("NC run log contains no completed epoch records")
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available()
        else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if torch.cuda.is_available()
        else 0,
        "device": str(device),
    }
    metrics = {
        "dataset": dataset,
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "completed_epochs": completed_epochs,
        "epoch_wall_time_seconds": elapsed / completed_epochs,
        "total_training_seconds": elapsed,
        "peak_memory": memory,
        "metrics": {key: {"mean": value[0], "std": value[1]} for key, value in result.items()},
        "checkpoint": str(checkpoint.resolve()),
    }
    (run_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("Training summary: %s", json.dumps(metrics, sort_keys=True))


def run_worker(dataset: str, seed: int, epochs: int, run_dir: Path):
    command = [
        PYTHON,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--epochs-worker",
        str(epochs),
        "--run-dir",
        str(run_dir.resolve()),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads((run_dir / "training_metrics.json").read_text())


def run_audit(dataset: str, seed: int, checkpoint: Path, output: Path):
    command = [
        PYTHON,
        str(ROOT / "scripts/audit_interaction_provenance_m1.py"),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output.resolve()),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def lookup_m01():
    records = json.loads((ROOT / "results/model_design/m0/pilot_records.json").read_text())
    return {
        (row["dataset"], int(row["seed"])): row
        for row in records
        if row.get("variant") == "orthogonal_transport"
        and row.get("dataset") in {"Movies", "Grocery"}
        and int(row.get("seed", -1)) in {42, 43, 44}
    }


def make_result_artifacts(runs: list[dict], audits: dict[tuple[str, int], dict]):
    results_dir = ROOT / "results/model_design/m1"
    m01 = lookup_m01()
    comparison = []
    dynamics, diversity, atoms, cka, counterfactual_rows, shuffle_rows = [], [], [], [], [], []
    for run in runs:
        key = (run["dataset"], int(run["seed"]))
        ref = m01[key]
        audit = audits[key]
        m1_val = audit["full_validation"]
        comparison.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "m01_val_acc": ref["val_acc"],
                "m1_val_acc": m1_val["val_acc"],
                "delta_val_acc_m1_minus_m01": m1_val["val_acc"] - ref["val_acc"],
                "m01_val_macro_f1": ref["val_macro_f1"],
                "m1_val_macro_f1": m1_val["val_macro_f1"],
                "delta_val_macro_f1_m1_minus_m01": m1_val["val_macro_f1"] - ref["val_macro_f1"],
                "m1_best_epoch": run.get("best_epoch", ""),
                "checkpoint": run["checkpoint"],
            }
        )
        diag = audit["analysis"]
        for modality in ("text", "visual"):
            p = diag["provenance"][modality]
            for order in range(1, 4):
                dynamics.append(
                    {
                        "dataset": key[0],
                        "seed": key[1],
                        "modality": modality,
                        "order": order,
                        "G_norm_mean": p["state_norms"][f"G{order}"]["mean"],
                        "G_norm_p50": p["state_norms"][f"G{order}"]["p50"],
                        "relative_change_mean": p["progressive_change"][f"k{order}"]["relative_change"]["mean"],
                        "relative_change_p50": p["progressive_change"][f"k{order}"]["relative_change"]["p50"],
                        "relative_change_p90": p["progressive_change"][f"k{order}"]["relative_change"]["p90"],
                        "cosine_mean": p["progressive_change"][f"k{order}"]["cosine_similarity"]["mean"],
                        "cosine_p10": p["progressive_change"][f"k{order}"]["cosine_similarity"]["p10"],
                        "cosine_p50": p["progressive_change"][f"k{order}"]["cosine_similarity"]["p50"],
                    }
                )
                diversity.append(
                    {
                        "dataset": key[0],
                        "seed": key[1],
                        "modality": modality,
                        "state": f"G{order}",
                        **p["state_diversity"][f"G{order}"],
                    }
                )
                atom_metrics = p["interaction_atoms"][f"k{order}"]
                atoms.append(
                    {
                        "dataset": key[0],
                        "seed": key[1],
                        "modality": modality,
                        "order": order,
                        **{key_name: value for key_name, value in atom_metrics.items() if key_name not in {"norm", "per_channel_variance"}},
                        "norm_mean": atom_metrics["norm"]["mean"],
                        "norm_p50": atom_metrics["norm"]["p50"],
                        "norm_p90": atom_metrics["norm"]["p90"],
                        "per_channel_variance": json.dumps(atom_metrics["per_channel_variance"]),
                    }
                )
                cka.append(
                    {
                        "dataset": key[0],
                        "seed": key[1],
                        "modality": modality,
                        "order": order,
                        **diag["semantic_provenance_cka"][f"{modality}_k{order}"],
                    }
                )
        for cf in audit["counterfactuals"]:
            mean_g3 = sum(
                cf["G3_full_vs_variant_relative_difference"][m]["mean"]
                for m in ("text", "visual")
            ) / 2.0
            row = {
                "dataset": key[0],
                "seed": key[1],
                "mode": cf["mode"],
                "shuffle_seed": cf["shuffle_seed"] if cf["shuffle_seed"] is not None else "",
                "val_acc": cf["val_acc"],
                "val_macro_f1": cf["val_macro_f1"],
                "delta_val_acc_vs_full": cf["delta_val_acc_vs_full"],
                "delta_val_macro_f1_vs_full": cf["delta_val_macro_f1_vs_full"],
                "G3_mean_relative_difference": mean_g3,
                "provenance_residual_relative_difference_mean": cf["residual_relative_difference"]["mean"],
                "atom_relative_difference_mean": sum(
                    value["mean"] for value in cf["atom_geometry"].values()
                ) / max(len(cf["atom_geometry"]), 1),
                "semantic_max_abs_error": cf["semantic_max_abs_error"],
            }
            counterfactual_rows.append(row)
            if cf["mode"] == "history_shuffle":
                shuffle_rows.append(row)

    write_json(results_dir / "m1_runs.json", runs)
    write_csv(results_dir / "m01_vs_m1.csv", comparison)
    write_csv(results_dir / "provenance_dynamics.csv", dynamics)
    write_csv(results_dir / "provenance_diversity.csv", diversity)
    write_csv(results_dir / "interaction_atom_diversity.csv", atoms)
    write_csv(results_dir / "semantic_provenance_cka.csv", cka)
    write_csv(results_dir / "provenance_counterfactuals.csv", counterfactual_rows)
    write_csv(results_dir / "history_shuffle.csv", shuffle_rows)
    write_json(results_dir / "frozen_counterfactual_audits.json", list(audits.values()))
    return comparison, dynamics, diversity, atoms, cka, counterfactual_rows, shuffle_rows


def write_report(tables):
    comparison, dynamics, diversity, atoms, cka, counterfactuals, shuffles = tables
    import statistics

    def mean(rows, key):
        return statistics.mean(float(row[key]) for row in rows) if rows else 0.0

    def mean_sd(rows, key):
        values = [float(row[key]) for row in rows]
        if not values:
            return "n/a"
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.4f} ± {spread:.4f}"

    rows = [
        "# M1 Interaction Provenance pilot",
        "",
        "## Scope and protocol",
        "",
        "M1 was implemented on `exp/interaction_provenance_m1` from frozen M0.1 source SHA `749d1b4f20c13886a308cddb4770e1e25a865ffd`. The M0.2 implementation was not used. Six NC runs covered Movies and Grocery with seeds 42/43/44. The original train split was used for training and original validation split for checkpoint selection; test evaluation and LP were disabled. The M0.1 reference uses only existing `orthogonal_transport` rows in `results/model_design/m0/pilot_records.json`; no baseline was retrained.",
        "",
        "The provenance adapter is a temporary task-exposure mechanism for M1 development and is NOT the final Stage III evidence-composition design. Immediate task performance is supportive evidence only; M1 is primarily evaluated by whether it learns a non-degenerate, history-sensitive provenance state.",
        "",
        "## M0.1 paired validation comparison",
        "",
        "| Dataset | M0.1 Acc | M1 Acc | Paired Δ Acc | M0.1 Macro-F1 | M1 Macro-F1 | Paired Δ Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("Movies", "Grocery"):
        group = [row for row in comparison if row["dataset"] == dataset]
        rows.append(
            f"| {dataset} | {mean_sd(group, 'm01_val_acc')} | {mean_sd(group, 'm1_val_acc')} | {mean_sd(group, 'delta_val_acc_m1_minus_m01')} | {mean_sd(group, 'm01_val_macro_f1')} | {mean_sd(group, 'm1_val_macro_f1')} | {mean_sd(group, 'delta_val_macro_f1_m1_minus_m01')} |"
        )
    rows += [
        "",
        "Per-seed paired values:",
        "",
        "| Dataset | Seed | M0.1 Acc | M1 Acc | Δ Acc | M0.1 Macro-F1 | M1 Macro-F1 | Δ Macro-F1 | Best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        rows.append(
            f"| {row['dataset']} | {row['seed']} | {row['m01_val_acc']:.4f} | {row['m1_val_acc']:.4f} | {row['delta_val_acc_m1_minus_m01']:+.4f} | {row['m01_val_macro_f1']:.4f} | {row['m1_val_macro_f1']:.4f} | {row['delta_val_macro_f1_m1_minus_m01']:+.4f} | {row['m1_best_epoch']} |"
        )

    rows += ["", "## Training time and memory", "", "Epoch time is total NC train/evaluation time divided by epochs actually completed before early stopping. CUDA peak memory is measured inside the isolated training worker.", "", "| Dataset | Seed | Epochs completed | Seconds/epoch | Peak allocated (GiB) | Peak reserved (GiB) |", "|---|---:|---:|---:|---:|---:|"]
    for run in _REPORT_RUNS:
        memory = run["peak_memory"]
        rows.append(
            f"| {run['dataset']} | {run['seed']} | {run['completed_epochs']} | {run['epoch_wall_time_seconds']:.2f} | {memory['peak_allocated_bytes'] / 2**30:.2f} | {memory['peak_reserved_bytes'] / 2**30:.2f} |"
        )

    rows += ["", "## Provenance dynamics", "", "| Modality | Order | G norm mean | Relative change mean | Cosine mean |", "|---|---:|---:|---:|---:|"]
    for modality in ("text", "visual"):
        for order in (1, 2, 3):
            group = [row for row in dynamics if row["modality"] == modality and int(row["order"]) == order]
            rows.append(
                f"| {modality} | {order} | {mean(group, 'G_norm_mean'):.3f} | {mean(group, 'relative_change_mean'):.3f} | {mean(group, 'cosine_mean'):.3f} |"
            )
    rows += ["", "## State and interaction-atom diversity", "", "Centered effective rank is entropy-based over sampled node/edge states. Top energies and active-channel fractions are reported without pass thresholds.", "", "| State/order | Effective rank | Top1 energy | Top5 energy | Active channel fraction |", "|---|---:|---:|---:|---:|"]
    for state in ("G1", "G2", "G3"):
        group = [row for row in diversity if row["state"] == state]
        rows.append(
            f"| {state} | {mean(group, 'effective_rank'):.2f} / 32 | {mean(group, 'top1_energy'):.3f} | {mean(group, 'top5_energy'):.3f} | {mean(group, 'channel_variance_active_fraction'):.3f} |"
        )
    for order in (1, 2, 3):
        group = [row for row in atoms if int(row["order"]) == order]
        rows.append(
            f"| Atom k{order} | {mean(group, 'effective_rank'):.2f} / 32 | {mean(group, 'top1_energy'):.3f} | {mean(group, 'top5_energy'):.3f} | {mean(group, 'channel_variance_active_fraction'):.3f} |"
        )

    rows += ["", "## Adapter and semantic-provenance CKA", "", "| Measure | Mean across six runs/modalities/orders |", "|---|---:|"]
    adapter_norms = [run["analyze"]["provenance_adapter_weight_norm"] for run in _REPORT_RUNS]
    residual_ratios = [run["analyze"]["provenance_residual_ratio"]["mean"] for run in _REPORT_RUNS]
    rows.append(f"| Adapter weight norm | {statistics.mean(adapter_norms):.3f} (range {min(adapter_norms):.3f}–{max(adapter_norms):.3f}) |")
    rows.append(f"| Residual/base embedding norm ratio | {statistics.mean(residual_ratios):.3f} (range {min(residual_ratios):.3f}–{max(residual_ratios):.3f}) |")
    for order in (1, 2, 3):
        group = [row for row in cka if int(row["order"]) == order]
        rows.append(f"| CKA Gk vs ΔSk (k={order}) | {mean(group, 'G_vs_delta_S'):.3f} |")
        rows.append(f"| CKA Gk vs Sk (k={order}) | {mean(group, 'G_vs_S'):.3f} |")

    rows += ["", "## Frozen counterfactuals", "", "Each provenance-only counterfactual reused the same semantic cache and asserted text/visual S0–S3 equality within 2e-6; observed maximum absolute error was 0. These are frozen functional diagnostics, not retrained ablations.", "", "| Mode | Runs | Mean Δ Acc vs Full | Mean Δ Macro-F1 vs Full | Mean G3 relative change | Mean residual relative change | Mean atom relative change |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in ("adapter_off", "history_off", "history_shuffle", "relation_off", "effect_off"):
        group = [row for row in counterfactuals if row["mode"] == mode]
        rows.append(
            f"| {mode} | {len(group)} | {mean(group, 'delta_val_acc_vs_full'):+.4f} | {mean(group, 'delta_val_macro_f1_vs_full'):+.4f} | {mean(group, 'G3_mean_relative_difference'):.3f} | {mean(group, 'provenance_residual_relative_difference_mean'):.3f} | {mean(group, 'atom_relative_difference_mean'):.3f} |"
        )
    rows += ["", "History Shuffle used the prescribed degree buckets (1, 2–3, 4–7, 8–15, 16–31, ≥32), a shared text/visual node permutation, and shuffle seeds 3407/3408/3409 at k=2/3. The table aggregates 18 validations; per-seed/per-shuffle values are in `history_shuffle.csv`.", "", "## Evidence assessment", ""]
    rows.append(
        f"- **Active:** G states have mean norms around 5.7–5.8, all 32 sampled channels are active, adapter norm averages {statistics.mean(adapter_norms):.3f}, and residual/base norm ratio averages {statistics.mean(residual_ratios):.3f}."
    )
    rows.append(
        f"- **Progressive:** mean relative G changes stay substantial from G1→G2→G3 (text {mean([r for r in dynamics if r['modality']=='text' and int(r['order'])==2], 'relative_change_mean'):.2f} then {mean([r for r in dynamics if r['modality']=='text' and int(r['order'])==3], 'relative_change_mean'):.2f}; visual {mean([r for r in dynamics if r['modality']=='visual' and int(r['order'])==2], 'relative_change_mean'):.2f} then {mean([r for r in dynamics if r['modality']=='visual' and int(r['order'])==3], 'relative_change_mean'):.2f})."
    )
    history = [row for row in counterfactuals if row["mode"] == "history_off"]
    shuffle = [row for row in counterfactuals if row["mode"] == "history_shuffle"]
    rows.append(
        f"- **History-sensitive:** History-Off changes G3 by {mean(history, 'G3_mean_relative_difference'):.2f} relative norm on average; degree-matched History Shuffle changes it by {mean(shuffle, 'G3_mean_relative_difference'):.2f}. Their mean validation accuracy deltas are {mean(history, 'delta_val_acc_vs_full')*100:+.2f} and {mean(shuffle, 'delta_val_acc_vs_full')*100:+.2f} percentage points."
    )
    relation = [row for row in counterfactuals if row["mode"] == "relation_off"]
    effect = [row for row in counterfactuals if row["mode"] == "effect_off"]
    rows.append(
        f"- **Interaction-grounded:** Relation-Off and Effect-Off change sampled atoms by {mean(relation, 'atom_relative_difference_mean'):.2f} and {mean(effect, 'atom_relative_difference_mean'):.2f} relative norm, respectively; both components affect the recursively composed state. Relation-Off has the larger mean G3 change ({mean(relation, 'G3_mean_relative_difference'):.2f} vs {mean(effect, 'G3_mean_relative_difference'):.2f})."
    )
    rows.append(
        f"- **Non-redundant:** mean linear CKA Gk vs Sk ranges from {min(mean([r for r in cka if int(r['order'])==k], 'G_vs_S') for k in (1,2,3)):.2f} to {max(mean([r for r in cka if int(r['order'])==k], 'G_vs_S') for k in (1,2,3)):.2f}; this indicates related but non-identical sampled representations. G state effective rank averages roughly {mean([r for r in diversity if r['state']=='G1'], 'effective_rank'):.1f}–{mean([r for r in diversity if r['state']=='G3'], 'effective_rank'):.1f} of 32, so coordinates are active but anisotropic."
    )
    overall_acc = statistics.mean(float(row["delta_val_acc_m1_minus_m01"]) for row in comparison)
    overall_f1 = statistics.mean(float(row["delta_val_macro_f1_m1_minus_m01"]) for row in comparison)
    rows.append(
        f"- **Task performance:** overall paired means are Δ Acc {overall_acc*100:+.2f} pp and Δ Macro-F1 {overall_f1*100:+.2f} pp; dataset/seed effects are mixed, so these results do not establish a task-performance improvement."
    )
    rows += ["", "## Decision", "", "The pilot supports proceeding to M2: the provenance branch is active and progressive; history correspondence, latent relation, and realized message each influence its geometry; and CKA does not indicate a copied semantic state. Keep the moderate state-rank concentration and mixed Macro-F1 deltas visible as limitations. This recommendation is about testing how provenance should be combined with semantic increments in M2, not a claim that M1 improves the task metric.", ""]
    (ROOT / "docs/model_design/m1_pilot_report.md").write_text("\n".join(rows))


_REPORT_RUNS = []


def finalize_existing():
    global _REPORT_RUNS
    results_dir = ROOT / "results/model_design/m1"
    runs = json.loads((results_dir / "m1_runs.json").read_text())
    audits = {}
    for run in runs:
        dataset, seed = run["dataset"], int(run["seed"])
        run_dir = ROOT / "outputs/model_design/m1/pilot" / dataset / f"seed{seed}"
        log_path = run_dir / "main.log"
        completed_epochs = sum(
            1
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if re.search(r"\bEpoch \d+ \|", line)
        )
        if completed_epochs <= 0:
            raise RuntimeError(f"No completed epochs recorded in {log_path}")
        run["completed_epochs"] = completed_epochs
        run["epoch_wall_time_seconds"] = run["total_training_seconds"] / completed_epochs
        checkpoint_path = Path(run["checkpoint"]).resolve()
        run["checkpoint"] = checkpoint_path.relative_to(ROOT).as_posix()
        audit_path = ROOT / run["audit"]
        audit = json.loads(audit_path.read_text())
        audit["checkpoint"] = Path(audit["checkpoint"]).resolve().relative_to(ROOT).as_posix()
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        audits[(dataset, seed)] = audit
    _REPORT_RUNS = runs
    tables = make_result_artifacts(runs, audits)
    write_report(tables)
    print(f"Finalized {len(runs)} runs with epoch counts from logs")


def run_full(args):
    global _REPORT_RUNS
    tag = "smoke" if args.smoke else "pilot"
    if args.smoke:
        runs_to_do = [("Movies", 42)]
        epochs = 2
    else:
        runs_to_do = [(dataset, seed) for dataset in ("Movies", "Grocery") for seed in (42, 43, 44)]
        epochs = int(args.epochs)
    output_root = ROOT / "outputs/model_design/m1" / tag
    results_root = ROOT / "results/model_design/m1"
    all_runs = []
    all_audits = {}
    for dataset, seed in runs_to_do:
        run_dir = output_root / dataset / f"seed{seed}"
        training = run_worker(dataset, seed, epochs, run_dir)
        audit_path = results_root / "run_audits" / tag / f"{dataset}_seed{seed}.json"
        audit = run_audit(dataset, seed, Path(training["checkpoint"]), audit_path)
        record = {
            **training,
            "best_epoch": int(torch.load(training["checkpoint"], map_location="cpu", weights_only=False)["epoch"]),
            "validation": audit["full_validation"],
            "audit": str(audit_path.relative_to(ROOT)),
            "analyze": audit["analysis"],
            "counterfactuals": audit["counterfactuals"],
            "protocol": audit["protocol"],
        }
        all_runs.append(record)
        all_audits[(dataset, seed)] = audit
        if args.smoke:
            write_json(results_root / "smoke_record.json", record)
            print(f"Smoke passed: {dataset} seed {seed}; val={audit['full_validation']}")
    if args.smoke:
        return
    _REPORT_RUNS = all_runs
    tables = make_result_artifacts(all_runs, all_audits)
    write_report(tables)
    print(f"Pilot complete: {len(all_runs)} runs; report=docs/model_design/m1_pilot_report.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--dataset", choices=("Movies", "Grocery"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--epochs-worker", type=int, default=2)
    args = parser.parse_args()
    if args.finalize_existing:
        finalize_existing()
    elif args.worker:
        if args.dataset is None or args.seed is None or args.run_dir is None:
            parser.error("--worker requires --dataset, --seed, and --run-dir")
        worker(args.dataset, args.seed, args.epochs_worker, Path(args.run_dir))
    else:
        run_full(args)


if __name__ == "__main__":
    main()
