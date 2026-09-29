from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.utils.device import get_device
from src.utils.summary import count_parameters


OUTPUT = ROOT / "outputs" / "model_design" / "m0"
RESULTS = ROOT / "results" / "model_design" / "m0"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = ("identity_transport", "orthogonal_transport", "multi_order_bank_gpr")


def _overrides(dataset: str, variant: str, seed: int, checkpoint: Path | None = None):
    if variant == "multi_order_bank_gpr":
        model_name = "multi_order_bank"
        transport = []
        model_extra = ["model.readout=gpr"]
    else:
        model_name = "relation_transport_m0"
        mode = "identity" if variant == "identity_transport" else "orthogonal"
        transport = [f"model.transport_mode={mode}"]
        model_extra = []
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        f"model={model_name}",
        f"seed={seed}",
        "num_runs=1",
        "device=cuda:0",
        "task.inference_mode=full",
        "task.evaluate_test=false",
        "task.training_mode=full_graph",
        *model_extra,
        *transport,
    ]
    if checkpoint is not None:
        overrides.append(f"task.save_ckpt_path={checkpoint}")
    return overrides


def _compose(dataset: str, variant: str, seed: int, checkpoint: Path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=_overrides(dataset, variant, seed, checkpoint),
        )
    return cfg


def _is_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(_is_finite(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_is_finite(value) for value in obj)
    if isinstance(obj, float):
        return torch.isfinite(torch.tensor(obj)).item()
    return True


def _train(dataset: str, variant: str, seed: int, run_dir: Path, checkpoint: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "src/main.py", *_overrides(dataset, variant, seed, checkpoint)]
    command.append(f"hydra.run.dir={run_dir / 'hydra'}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    with (run_dir / "training_console.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        tail = (run_dir / "training_console.log").read_text(errors="replace")[-6000:]
        raise RuntimeError(f"Training failed for {dataset}/{variant}/seed{seed}:\n{tail}")


def _diagnose(dataset: str, variant: str, seed: int, checkpoint: Path):
    cfg = _compose(dataset, variant, seed, checkpoint)
    data = load_mag_data(cfg, str(cfg.task.name), seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = build_model(cfg, data_info)
    parameter_count = count_parameters(model)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"])
    device = get_device("cuda:0")
    model.to(device).eval()
    if variant == "multi_order_bank_gpr":
        diagnostics = {
            "gamma_text": model.gamma_text.detach().cpu().tolist(),
            "gamma_visual": model.gamma_visual.detach().cpu().tolist(),
        }
    else:
        diagnostics = model.analyze(data.x.to(device), data.edge_index.to(device))
    if not _is_finite(diagnostics):
        raise FloatingPointError(f"Non-finite diagnostics in {checkpoint}")
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "best_epoch": int(saved["epoch"]),
        "val_acc": float(saved["metrics"]["val_acc"]),
        "val_macro_f1": float(saved["metrics"]["val_macro_f1"]),
        "params": int(parameter_count),
        "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
        "diagnostics": diagnostics,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    runs = [(dataset, variant, seed) for dataset in DATASETS for variant in VARIANTS for seed in SEEDS]
    for number, (dataset, variant, seed) in enumerate(runs, start=1):
        run_dir = OUTPUT / "runs" / dataset / variant / f"seed{seed}"
        checkpoint = OUTPUT / "checkpoints" / f"{dataset}_{variant}_seed{seed}.pt"
        record_path = RESULTS / "runs" / dataset / variant / f"seed{seed}" / "pilot_record.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.exists() and checkpoint.exists():
            record = json.loads(record_path.read_text())
            print(f"[{number}/{len(runs)}] reuse {dataset} {variant} seed={seed}", flush=True)
        else:
            print(f"[{number}/{len(runs)}] train {dataset} {variant} seed={seed}", flush=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.exists():
                _train(dataset, variant, seed, run_dir, checkpoint)
            else:
                print(f"    diagnose existing checkpoint {checkpoint.name}", flush=True)
            record = _diagnose(dataset, variant, seed, checkpoint)
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        records.append(record)
        (RESULTS / "pilot_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            f"    val_acc={record['val_acc']:.6f} "
            f"val_macro_f1={record['val_macro_f1']:.6f} "
            f"best_epoch={record['best_epoch']} params={record['params']}",
            flush=True,
        )
    print(f"Wrote {RESULTS / 'pilot_records.json'}", flush=True)


if __name__ == "__main__":
    main()
