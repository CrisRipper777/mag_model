from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.utils.device import get_device

OUTPUT = ROOT / "outputs" / "model_design" / "m02"
RESULTS = ROOT / "results" / "model_design" / "m02"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)


def _overrides(dataset: str, seed: int, checkpoint: Path, epochs: int | None = None):
    values = [
        f"dataset={dataset}",
        "task=nc",
        "model=relation_transport_m02",
        f"seed={seed}",
        "num_runs=1",
        "device=cuda:0",
        "task.inference_mode=full",
        "task.evaluate_test=false",
        "task.training_mode=full_graph",
        f"task.save_ckpt_path={checkpoint}",
    ]
    if epochs is not None:
        values.append(f"task.epochs={epochs}")
    return values


def _compose(dataset: str, seed: int, checkpoint: Path, epochs: int | None = None):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            config_name="config",
            overrides=_overrides(dataset, seed, checkpoint, epochs),
        )


def _gpu_process_memory(pid: int) -> int | None:
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if query.returncode:
        return None
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == str(pid):
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


def _train(dataset: str, seed: int, run_dir: Path, checkpoint: Path, epochs: int | None):
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "src/main.py",
        *_overrides(dataset, seed, checkpoint, epochs),
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    peak_mb = 0
    with (run_dir / "training_console.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            used = _gpu_process_memory(proc.pid)
            if used is not None:
                peak_mb = max(peak_mb, used)
            time.sleep(0.5)
        return_code = proc.wait()
    if return_code:
        tail = (run_dir / "training_console.log").read_text(errors="replace")[-8000:]
        raise RuntimeError(f"Training failed for {dataset}/seed{seed}:\n{tail}")
    return peak_mb or None


def _metric(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, num_classes: int):
    prediction = logits[indices.to(logits.device)].argmax(dim=-1).detach().cpu().numpy()
    target = labels[indices].detach().cpu().numpy()
    return {
        "acc": float((prediction == target).mean()),
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=list(range(int(num_classes))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    q = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p10": float(q[0].item()),
        "p50": float(q[1].item()),
        "p90": float(q[2].item()),
    }


@torch.no_grad()
def _progressive(model, x: torch.Tensor, edge_index: torch.Tensor):
    model.eval()
    result = model._compute(x, edge_index)
    rows = []
    for modality in ("text", "visual"):
        states = result[f"S_{modality}"]
        for order in (1, 2, 3):
            previous, current, initial = states[order - 1], states[order], states[0]
            relative = (current - previous).norm(dim=-1) / (previous.norm(dim=-1) + 1e-12)
            adjacent = torch.nn.functional.cosine_similarity(
                previous, current, dim=-1, eps=1e-8
            )
            novelty = 1.0 - torch.nn.functional.cosine_similarity(
                initial, current, dim=-1, eps=1e-8
            )
            rows.append(
                {
                    "modality": modality,
                    "order": order,
                    "relative_delta": _distribution(relative),
                    "adjacent_cosine": _distribution(adjacent),
                    "cumulative_novelty": _distribution(novelty),
                }
            )
    return rows


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(torch.isfinite(torch.tensor(float(value))).item())
    return True


def _diagnose(dataset: str, seed: int, checkpoint: Path, peak_memory_mb: int | None):
    cfg = _compose(dataset, seed, checkpoint)
    data = load_mag_data(cfg, str(cfg.task.name), seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = build_model(cfg, data_info)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes))
    head.load_state_dict(saved["head_state"], strict=True)
    device = get_device("cuda:0")
    model.to(device).eval()
    head.to(device).eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    val_idx, labels = data.val_idx.to(device), data.y.to(device)
    with torch.no_grad():
        full_z, *_ = model(x, edge_index)
        full_metrics = _metric(head(full_z), labels, val_idx, int(data.num_classes))
        off_z, *_ = model.forward_without_relation_context(x, edge_index)
        off_metrics = _metric(head(off_z), labels, val_idx, int(data.num_classes))
        diagnostics = model.analyze(x, edge_index)
        progressive = _progressive(model, x, edge_index)
    if not all(
        torch.isfinite(parameter).all()
        for module in (model, head)
        for parameter in module.parameters()
    ):
        raise FloatingPointError(f"Non-finite M0.2 model/head parameters in {checkpoint}")
    if not _finite_tree((full_metrics, off_metrics, diagnostics, progressive)):
        raise FloatingPointError(f"Non-finite M0.2 diagnostics in {checkpoint}")
    return {
        "dataset": dataset,
        "seed": seed,
        "best_epoch": int(saved["epoch"]),
        "val_acc": float(saved["metrics"]["val_acc"]),
        "val_macro_f1": float(saved["metrics"]["val_macro_f1"]),
        "validation_fixed_class_metrics": full_metrics,
        "context_off_validation_fixed_class_metrics": off_metrics,
        "context_off_delta_acc": full_metrics["acc"] - off_metrics["acc"],
        "context_off_delta_macro_f1": full_metrics["macro_f1"] - off_metrics["macro_f1"],
        "params": (
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            + sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
        ),
        "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
        "peak_gpu_process_memory_mb": peak_memory_mb,
        "diagnostics": diagnostics,
        "progressive_dynamics": progressive,
    }


def _read_m01_references():
    source = ROOT / "results" / "model_design" / "m0" / "pilot_records.json"
    records = json.loads(source.read_text())
    return {
        (row["dataset"], int(row["seed"])): row
        for row in records
        if row["variant"] == "orthogonal_transport"
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_m01_vs_m02(records: list[dict], references: dict):
    rows = []
    for record in records:
        reference = references[(record["dataset"], record["seed"])]
        rows.append(
            {
                "row_type": "paired_seed",
                "dataset": record["dataset"],
                "seed": record["seed"],
                "m01_val_acc": reference["val_acc"],
                "m01_val_macro_f1": reference["val_macro_f1"],
                "m02_val_acc": record["val_acc"],
                "m02_val_macro_f1": record["val_macro_f1"],
                "paired_delta_acc": record["val_acc"] - reference["val_acc"],
                "paired_delta_macro_f1": record["val_macro_f1"] - reference["val_macro_f1"],
            }
        )
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        aggregate = {"row_type": "mean_population_sd", "dataset": dataset, "seed": ""}
        for column in (
            "m01_val_acc",
            "m01_val_macro_f1",
            "m02_val_acc",
            "m02_val_macro_f1",
            "paired_delta_acc",
            "paired_delta_macro_f1",
        ):
            values = [float(row[column]) for row in subset]
            mean = sum(values) / len(values)
            sd = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
            aggregate[column] = f"{mean:.6f} ± {sd:.6f}"
        rows.append(aggregate)
    return rows


def _result_rows(records: list[dict]):
    context = []
    counterfactual = []
    progressive = []
    for record in records:
        for modality in ("text", "visual"):
            for diagnostic, key in (
                ("context_norm", "relation_context_norm"),
                ("relative_context_residual", "contextual_relation_residual"),
                ("abs_conductance_change", "context_conductance_abs_change"),
                ("mean_abs_angle_change", "context_angle_mean_abs_change"),
            ):
                row = {
                    "dataset": record["dataset"],
                    "seed": record["seed"],
                    "modality": modality,
                    "diagnostic": diagnostic,
                }
                row.update(record["diagnostics"][key][modality])
                context.append(row)

        full = record["validation_fixed_class_metrics"]
        context_off = record["context_off_validation_fixed_class_metrics"]
        counterfactual.append(
            {
                "dataset": record["dataset"],
                "seed": record["seed"],
                "full_context_acc": full["acc"],
                "full_context_macro_f1": full["macro_f1"],
                "context_off_acc": context_off["acc"],
                "context_off_macro_f1": context_off["macro_f1"],
                "delta_acc_full_minus_off": record["context_off_delta_acc"],
                "delta_macro_f1_full_minus_off": record["context_off_delta_macro_f1"],
            }
        )

        for item in record["progressive_dynamics"]:
            for metric_key in ("relative_delta", "adjacent_cosine", "cumulative_novelty"):
                row = {
                    "dataset": record["dataset"],
                    "seed": record["seed"],
                    "modality": item["modality"],
                    "order": item["order"],
                    "metric": metric_key,
                }
                row.update(item[metric_key])
                progressive.append(row)
    return context, counterfactual, progressive


def _run(smoke: bool = False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    runs = [("Movies", 42)] if smoke else [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    for number, (dataset, seed) in enumerate(runs, start=1):
        tag = "smoke" if smoke else "pilot"
        run_dir = OUTPUT / tag / "runs" / dataset / f"seed{seed}"
        checkpoint = OUTPUT / tag / "checkpoints" / f"{dataset}_seed{seed}.pt"
        record_path = RESULTS / tag / f"{dataset}_seed{seed}.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.exists() and checkpoint.exists():
            record = json.loads(record_path.read_text())
            print(f"[{number}/{len(runs)}] reuse {dataset} seed={seed}", flush=True)
        else:
            epochs = 2 if smoke else None
            print(f"[{number}/{len(runs)}] train {dataset} seed={seed} ({epochs or 'full'} epochs)", flush=True)
            peak_memory_mb = _train(dataset, seed, run_dir, checkpoint, epochs)
            record = _diagnose(dataset, seed, checkpoint, peak_memory_mb)
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        records.append(record)
        if smoke:
            summary = {
                "dataset": dataset,
                "seed": seed,
                "checkpoint": record["checkpoint"],
                "finite_diagnostics": _finite_tree(record["diagnostics"]),
                "context_off": record["context_off_validation_fixed_class_metrics"],
                "peak_gpu_process_memory_mb": record["peak_gpu_process_memory_mb"],
            }
            (RESULTS / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"    val_acc={record['val_acc']:.6f} val_macro_f1={record['val_macro_f1']:.6f} "
            f"context_off_acc={record['context_off_validation_fixed_class_metrics']['acc']:.6f}",
            flush=True,
        )

    if smoke:
        return
    (RESULTS / "m02_runs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    refs = _read_m01_references()
    _write_csv(
        RESULTS / "m01_vs_m02.csv",
        _flatten_m01_vs_m02(records, refs),
        ["row_type", "dataset", "seed", "m01_val_acc", "m01_val_macro_f1", "m02_val_acc", "m02_val_macro_f1", "paired_delta_acc", "paired_delta_macro_f1"],
    )
    context, counterfactual, progressive = _result_rows(records)
    _write_csv(
        RESULTS / "context_diagnostics.csv",
        context,
        ["dataset", "seed", "modality", "diagnostic", "mean", "std", "p10", "p50", "p90"],
    )
    _write_csv(
        RESULTS / "context_counterfactual.csv",
        counterfactual,
        ["dataset", "seed", "full_context_acc", "full_context_macro_f1", "context_off_acc", "context_off_macro_f1", "delta_acc_full_minus_off", "delta_macro_f1_full_minus_off"],
    )
    _write_csv(
        RESULTS / "progressive_dynamics.csv",
        progressive,
        ["dataset", "seed", "modality", "order", "metric", "mean", "p10", "p50", "p90"],
    )
    print(f"Wrote M0.2 pilot records and summaries to {RESULTS}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the M0.2 Movies/Grocery NC pilot")
    parser.add_argument("--smoke", action="store_true", help="Run Movies seed 42 for two epochs only")
    args = parser.parse_args()
    _run(smoke=args.smoke)


if __name__ == "__main__":
    main()
