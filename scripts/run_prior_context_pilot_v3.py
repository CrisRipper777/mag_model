from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models.mag_prior_context import Model, canonical_directed_edges
from src.tasks.common import build_optimizer
from src.tasks.prior_context_pilot import PilotTaskHeads, train_only_loss
from src.utils.seeds import set_seed

DATASETS = ("Movies", "Grocery")
VARIANTS = tuple(f"R{i}" for i in range(7))
STAGE_A = ("R0", "R1", "R4")
STAGE_B = ("R2", "R3", "R5", "R6")
LR_GRID = (0.0003, 0.001, 0.003)
TRAIN_SEEDS = (52, 53, 54)
V1_LOCK = ROOT / "reports/problem_validation_v1/protocol_lock.json"
V2_MANIFEST = ROOT / "reports/mechanism_pilot_v2/manifest.json"
STUDY_ROOT = ROOT / "docs/prior_context_pilot_v3"
REPORT_ROOT = ROOT / "reports/prior_context_pilot_v3"
OUTPUT_ROOT = ROOT / "outputs/prior_context_pilot_v3"
PROTOCOL_LOCK = STUDY_ROOT / "protocol_lock.json"
BASE_COMMIT = "1bc436b3eab2298fee7994c468375ac06e8707c0"
SOURCE_FILES = (
    "src/models/mag_prior_context.py",
    "src/tasks/prior_context_pilot.py",
    "src/models/factory.py",
    "src/tasks/common.py",
    "src/data/loaders.py",
    "src/data/graph_utils.py",
    "src/data/splits.py",
    "src/data/types.py",
    "configs/config.yaml",
    "configs/task/nc.yaml",
    "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml",
    "configs/model/mag_prior_context.yaml",
    "scripts/run_prior_context_pilot_v3.py",
    "scripts/audit_prior_context_pilot_v3.py",
    "tests/test_mag_prior_context.py",
    "docs/prior_context_pilot_v3/design.md",
    "docs/prior_context_pilot_v3/protocol.md",
    "docs/prior_context_pilot_v3/matrix.json",
    "docs/prior_context_pilot_v3/capacity.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def v1_lock() -> dict[str, Any]:
    raw = V1_LOCK.read_bytes()
    lock = json.loads(raw)
    v2 = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != v2["v1_protocol_lock_sha256"]:
        raise RuntimeError("v1 protocol lock changed relative to the committed v2 manifest")
    if lock.get("study") != "mag_problem_validation_v1":
        raise RuntimeError("unexpected v1 lock study identity")
    return lock


def input_hashes_for(dataset: str) -> dict[str, dict[str, Any]]:
    schema = v1_lock()["schemas"][dataset]
    actual: dict[str, dict[str, Any]] = {}
    for path_text, expected in schema["dataset_input_hashes"].items():
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"locked input is missing: {path}")
        actual_item = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual_item != expected:
            raise RuntimeError(f"locked input hash mismatch for {path}: expected={expected}, actual={actual_item}")
        actual[path_text] = actual_item
    split_path = next(Path(path) for path in actual if "/MAGB_split/" in path)
    if actual[str(split_path)]["sha256"] != schema["official_split_sha256"]:
        raise RuntimeError(f"official seed42 split identity mismatch for {dataset}")
    return actual


def compose_config(dataset: str, variant: str, seed: int, lr: float, device: str, width: int | None):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=mag_prior_context",
            f"seed={int(seed)}", f"device={device}",
        ])
    OmegaConf.set_struct(cfg, False)
    schema = v1_lock()["schemas"][dataset]
    split_path = next(path for path in schema["dataset_input_hashes"] if "/MAGB_split/" in path)
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.model.variant = str(variant)
    if width is not None:
        cfg.model.width = int(width)
    cfg.model.dropout = 0.2
    cfg.task.optimizer = "adamw"
    cfg.task.lr = float(lr)
    cfg.task.weight_decay = 1e-4
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 0.0
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.scheduler = None
    cfg.task.evaluate_test = False
    cfg.task.eval_labels = list(schema["macro_f1_label_set"])
    cfg.task.source_aux_weight = 0.1
    cfg.task.loss.aux_weight = 0.0
    cfg.dataset.nc_split_path = split_path
    cfg.dataset.auto_generate_splits = False
    cfg.dataset.make_undirected = True
    cfg.dataset.add_self_loops = False
    return cfg


def load_locked_dataset(dataset: str):
    hashes = input_hashes_for(dataset)
    cfg = compose_config(dataset, "R4", 42, LR_GRID[0], "cpu", None)
    data = load_mag_data(cfg, "nc", 42)
    schema = v1_lock()["schemas"][dataset]
    expected_sizes = schema["split_sizes"]
    if int(data.num_classes) != int(schema["num_classes"]):
        raise RuntimeError(f"class count changed for {dataset}")
    if int(data.train_idx.numel()) != int(expected_sizes[0]) or int(data.val_idx.numel()) != int(expected_sizes[1]):
        raise RuntimeError(f"train/validation split size changed for {dataset}")
    if data.x_t is None or data.x_i is None:
        raise RuntimeError(f"expected the official two-modality loader for {dataset}")
    edges = canonical_directed_edges(data.edge_index, data.num_nodes)
    return data, edges, hashes, {
        "input_dim": int(data.input_dim), "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
    }


def n_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _model_for_count(dataset: str, variant: str, info: dict[str, int], width: int | None = None) -> Model:
    cfg = compose_config(dataset, variant, 52, LR_GRID[0], "cpu", width)
    return Model(cfg, info)


def choose_capacity_widths(dataset: str, info: dict[str, int]) -> dict[str, Any]:
    reference = _model_for_count(dataset, "R4", info)
    target = n_parameters(reference)
    ref_modules = reference.diagnostic_modules()
    ref_counts = {name: n_parameters(module) for name, module in ref_modules.items()}
    adapters = ref_counts["adapter_text"] + ref_counts["adapter_image"]
    common = target - sum(
        count for name, count in ref_counts.items()
        if ".phiS" in name or ".phiD" in name or ".router" in name
    )

    def counts(width: int) -> dict[str, int]:
        return {
            "R0": adapters + (385 * width + 128),
            "R1": common + (6 * (385 * width + 128)),
            "R2": common + (6 * (514 * width + 128)),
        }

    selected: dict[str, int] = {}
    targets: dict[str, int] = {}
    for variant in ("R0", "R1", "R2"):
        width = min(range(1, 4097), key=lambda w: (abs(counts(w)[variant] - target), w))
        selected[variant] = width
        targets[variant] = counts(width)[variant]
        check = _model_for_count(dataset, variant, info, width)
        if n_parameters(check) != targets[variant]:
            raise AssertionError(f"analytical and instantiated {variant} counts differ: {n_parameters(check)} != {targets[variant]}")
        if abs(targets[variant] - target) / target > 0.05:
            raise RuntimeError(f"{variant} width {width} is outside the frozen ±5% of R4 on {dataset}")
    return {"r4_target": target, "widths": selected, "analytical_counts": targets, "r4_modules": ref_counts}


def build_plan() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    capacity_rows = [["dataset", "scheme", "width", "active_backbone_parameters", "r4_target", "delta_percent", "task_head_parameters", "total_trainable_parameters"]]
    for dataset in DATASETS:
        data, edges, hashes, info = load_locked_dataset(dataset)
        diagnostic_pool = torch.cat((data.train_idx.cpu().long(), data.val_idx.cpu().long()))
        sample_size = min(512, int(diagnostic_pool.numel()))
        sample_generator = torch.Generator(device="cpu").manual_seed(20261005)
        sample_offsets = torch.randperm(diagnostic_pool.numel(), generator=sample_generator)[:sample_size]
        diagnostic_sample_node_ids = diagnostic_pool.index_select(0, sample_offsets).sort().values.tolist()
        capacity = choose_capacity_widths(dataset, info)
        widths = capacity["widths"]
        rows = {}
        for variant in VARIANTS:
            width = widths.get(variant)
            model = _model_for_count(dataset, variant, info, width)
            active = n_parameters(model)
            modules = {name: n_parameters(module) for name, module in model.diagnostic_modules().items()}
            if sum(modules.values()) != active:
                raise AssertionError(f"diagnostic module count mismatch for {dataset}/{variant}")
            heads = PilotTaskHeads(variant, info["num_classes"], model.out_dim)
            head_count = n_parameters(heads)
            target = capacity["r4_target"]
            delta = (active - target) / target * 100.0
            if variant in {"R0", "R1", "R2", "R3", "R4", "R5"} and abs(delta) > 5.0:
                raise RuntimeError(f"capacity target violated: {dataset}/{variant} {delta:.3f}%")
            rows[variant] = {
                "width": width, "active_backbone_parameters": active,
                "task_head_parameters": head_count, "total_trainable_parameters": active + head_count,
                "r4_target": target, "delta_percent": delta, "backbone_modules": modules,
            }
            capacity_rows.append([
                dataset, variant, "" if width is None else width, active, target,
                f"{delta:.6f}", head_count, active + head_count,
            ])
        datasets[dataset] = {
            "input_hashes": hashes, "data_info": info,
            "fixed_split_path": next(path for path in hashes if "/MAGB_split/" in path),
            "train_nodes": int(data.train_idx.numel()), "validation_nodes": int(data.val_idx.numel()),
            "diagnostic_sample_seed": 20261005,
            "diagnostic_sample_node_ids": diagnostic_sample_node_ids,
            "canonical_directed_edges": int(edges.size(1)),
            "macro_f1_label_set": list(v1_lock()["schemas"][dataset]["macro_f1_label_set"]),
            "capacity": capacity, "schemes": rows,
        }
    matrix = {
        "study": "prior_context_mag_pilot_v3",
        "scope": {"datasets": list(DATASETS), "task": "NC train/validation", "test_inference": False, "lp": False},
        "stage_a": {"schemes": list(STAGE_A), "calibration_runs": 18, "confirmation_runs": 12, "maximum_runs": 30},
        "stage_b": {"schemes": list(STAGE_B), "calibration_runs": 24, "confirmation_runs": 16, "maximum_runs": 40, "status_before_gate": "pending_gate"},
        "lr_grid": list(LR_GRID), "training_seeds": list(TRAIN_SEEDS), "split_seed": 42,
        "selection": "max seed52 validation accuracy, then fixed-label Macro-F1, then lower LR; exact epoch-score tie retains earlier epoch",
        "gate": {
            "comparators": ["R0", "R1"],
            "accuracy_path": {"equal_graph_mean_delta_at_least": 0.005, "macro_f1_mean_delta_at_least": 0.0, "positive_accuracy_pairs_at_least": 3},
            "macro_f1_path": {"equal_graph_mean_delta_at_least": 0.01, "accuracy_mean_delta_at_least": 0.0, "positive_macro_f1_pairs_at_least": 3},
            "per_graph_mean_floor": {"accuracy_delta": -0.002, "macro_f1_delta": -0.01},
            "resource_comparator": "R1", "max_mean_step_time_ratio": 3.0, "max_mean_peak_allocated_ratio": 3.0,
            "confirmation_seeds": [53, 54], "gate_data_points_per_comparator": 4,
        },
        "training": {"optimizer": "AdamW", "weight_decay": 0.0001, "dropout": 0.2, "epochs": 300, "patience": 30, "min_epoch": 30, "grad_clip": 1.0, "scheduler": None, "source_aux_weight": 0.1},
        "datasets": datasets,
        "edge_policy": "loader undirected, no self loops; model unique directed pairs, removes loops, adds reciprocal pairs; no sampling/truncation",
    }
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(STUDY_ROOT / "matrix.json", matrix)
    capacity_buffer = io.StringIO()
    csv.writer(capacity_buffer, lineterminator="\n").writerows(capacity_rows)
    (STUDY_ROOT / "capacity.csv").write_text(capacity_buffer.getvalue(), encoding="utf-8")
    return matrix


def source_file_manifest() -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"source identity file missing: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return files


def combined_source_sha(files: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def freeze_lock() -> dict[str, Any]:
    matrix_path = STUDY_ROOT / "matrix.json"
    capacity_path = STUDY_ROOT / "capacity.csv"
    if not matrix_path.is_file() or not capacity_path.is_file():
        raise RuntimeError("run --phase plan before freezing protocol inputs/capacity")
    lock_old = v1_lock()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    inputs = {ds: matrix["datasets"][ds]["input_hashes"] for ds in DATASETS}
    source_files = source_file_manifest()
    source_sha = combined_source_sha(source_files)
    source_commit = git_value("rev-parse", "HEAD")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, source_commit], cwd=ROOT, check=True)
    lock = {
        "study": "prior_context_mag_pilot_v3",
        "frozen_utc": utc_now(),
        "branch": "codex/mag-prior-context-pilot-v3",
        "baseline_commit": BASE_COMMIT,
        "training_source_commit": source_commit,
        "training_source_sha256": source_sha,
        "training_source_files": source_files,
        "v1_protocol_lock_path": str(V1_LOCK),
        "v1_protocol_lock_sha256": sha256_file(V1_LOCK),
        "v1_protocol_lock_internal_sha256": lock_old["protocol_lock_sha256"],
        "v2_manifest_sha256": sha256_file(V2_MANIFEST),
        "design_sha256": sha256_file(STUDY_ROOT / "design.md"),
        "protocol_sha256": sha256_file(STUDY_ROOT / "protocol.md"),
        "matrix_sha256": sha256_file(matrix_path),
        "capacity_sha256": sha256_file(capacity_path),
        "fixed_inputs": inputs,
        "fixed_split_seed": 42,
        "training_seeds": list(TRAIN_SEEDS),
        "macro_f1_label_sets": {ds: matrix["datasets"][ds]["macro_f1_label_set"] for ds in DATASETS},
        "test_evaluation": False,
        "test_prediction": False,
        "lp": False,
    }
    write_json(PROTOCOL_LOCK, lock)
    return lock


def verify_frozen_identity() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    matrix_path = STUDY_ROOT / "matrix.json"
    capacity_path = STUDY_ROOT / "capacity.csv"
    if lock["baseline_commit"] != BASE_COMMIT:
        raise RuntimeError("pilot baseline commit differs from frozen protocol")
    if sha256_file(V1_LOCK) != lock["v1_protocol_lock_sha256"]:
        raise RuntimeError("v1 evidence lock changed after v3 freeze")
    if sha256_file(STUDY_ROOT / "protocol.md") != lock["protocol_sha256"]:
        raise RuntimeError("frozen v3 protocol has changed")
    if sha256_file(STUDY_ROOT / "design.md") != lock["design_sha256"]:
        raise RuntimeError("frozen v3 design has changed")
    if sha256_file(matrix_path) != lock["matrix_sha256"] or sha256_file(capacity_path) != lock["capacity_sha256"]:
        raise RuntimeError("frozen matrix/capacity has changed")
    actual_source = source_file_manifest()
    if actual_source != lock["training_source_files"] or combined_source_sha(actual_source) != lock["training_source_sha256"]:
        raise RuntimeError("training source files differ from the frozen source identity")
    source_commit = lock["training_source_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT, check=True)
    if git_value("branch", "--show-current") != lock["branch"]:
        raise RuntimeError(f"must run on {lock['branch']}")
    for dataset in DATASETS:
        current = input_hashes_for(dataset)
        if current != lock["fixed_inputs"][dataset]:
            raise RuntimeError(f"input hashes differ from the v3 lock for {dataset}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    return lock, matrix


def environment_record(device: str) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("dgl", "torch-geometric", "numpy", "scikit-learn", "hydra-core", "omegaconf"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    device_name = None
    if device.startswith("cuda"):
        device_name = torch.cuda.get_device_name(torch.device(device))
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda, "packages": packages,
        "device": device, "device_name": device_name,
    }


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def module_state_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda_all": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def count_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int, macro_labels: list[int]) -> tuple[dict[str, Any], torch.Tensor]:
    prediction = logits.argmax(dim=-1)
    flat = target.long() * int(num_classes) + prediction.long()
    confusion = torch.bincount(flat, minlength=int(num_classes) ** 2).reshape(num_classes, num_classes)
    diagonal = confusion.diag().to(torch.float64)
    denom = 2 * diagonal + confusion.sum(0) - diagonal + confusion.sum(1) - diagonal
    per_class = torch.where(denom > 0, 2 * diagonal / denom, torch.zeros_like(denom))
    f1 = per_class[torch.as_tensor(macro_labels, dtype=torch.long, device=logits.device)].mean()
    metrics = {
        "n": int(target.numel()),
        "accuracy": int((prediction == target).sum().item()) / max(int(target.numel()), 1),
        "macro_f1": float(f1.item()),
        "cross_entropy": float(nn.functional.cross_entropy(logits, target).item()),
        "confusion_matrix": confusion.detach().cpu().tolist(),
        "class_order": list(range(int(num_classes))),
        "macro_f1_label_set": list(macro_labels),
    }
    return metrics, prediction.detach().cpu()


def split_extras(extras: dict[str, Any], index: torch.Tensor) -> dict[str, Any]:
    source = extras["source_states"]
    return {"source_states": {key: value.index_select(0, index) for key, value in source.items()}}


@torch.no_grad()
def evaluate_train_validation(
    model: Model,
    heads: PilotTaskHeads,
    x: torch.Tensor,
    edges: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    macro_labels: list[int],
) -> dict[str, Any]:
    model.eval()
    heads.eval()
    model._collect_diagnostics = False
    z, _, _, _, extras = model(x, edges)
    train_z = z.index_select(0, train_idx)
    val_z = z.index_select(0, val_idx)
    train_logits, _ = heads(train_z, split_extras(extras, train_idx))
    val_logits, _ = heads(val_z, split_extras(extras, val_idx))
    train_metrics, train_prediction = count_metrics(train_logits, train_y, num_classes, macro_labels)
    val_metrics, val_prediction = count_metrics(val_logits, val_y, num_classes, macro_labels)
    result = {
        "train": train_metrics, "validation": val_metrics,
        "train_predictions": train_prediction, "validation_predictions": val_prediction,
        "model_diagnostics": {},
    }
    del z, extras, train_logits, val_logits
    model._collect_diagnostics = False
    return result


@torch.no_grad()
def collect_model_diagnostics(
    model: Model, x: torch.Tensor, edges: torch.Tensor, sample_idx: torch.Tensor
) -> dict[str, Any]:
    model.eval()
    model._collect_diagnostics = True
    z, _, _, _, extras = model(x, edges)
    index = sample_idx.to(device=z.device, dtype=torch.long)
    routes = {
        name: value.index_select(0, index)
        for name, value in extras.get("route_weights", {}).items()
    }
    structures = {
        name: (left.index_select(0, index), right.index_select(0, index))
        for name, (left, right) in extras.get("structure_outputs", {}).items()
    }
    correction = extras.get("correction")
    diagnostics = model._summary(
        text=extras["source_states"]["text"].index_select(0, index),
        image=extras["source_states"]["image"].index_select(0, index),
        center=extras.get("center", extras.get("joint_state", z)).index_select(0, index),
        correction=None if correction is None else correction.index_select(0, index),
        z=z.index_select(0, index),
        routes=routes,
        structures=structures,
        degree=None if extras.get("degree") is None else extras["degree"].index_select(0, index),
    )
    diagnostics = json_ready(diagnostics)
    del z
    del extras
    model._collect_diagnostics = False
    return diagnostics


def parameter_groups(model: Model, heads: PilotTaskHeads) -> dict[str, list[nn.Parameter]]:
    groups = {
        name: [p for p in module.parameters() if p.requires_grad]
        for name, module in model.diagnostic_modules().items()
    }
    for name, module in heads.named_children():
        groups[f"task_head.{name}"] = [p for p in module.parameters() if p.requires_grad]
    flat = [p for values in groups.values() for p in values]
    if len({id(p) for p in flat}) != len(flat):
        raise RuntimeError("diagnostic parameter groups overlap")
    expected = [p for p in list(model.parameters()) + list(heads.parameters()) if p.requires_grad]
    if {id(p) for p in flat} != {id(p) for p in expected}:
        raise RuntimeError("diagnostic parameter groups do not cover every optimized parameter")
    return groups


def parameter_snapshots(groups: dict[str, list[nn.Parameter]]) -> dict[str, list[torch.Tensor]]:
    return {name: [parameter.detach().cpu().clone() for parameter in values] for name, values in groups.items()}


def step_diagnostics(
    groups: dict[str, list[nn.Parameter]],
    before: dict[str, list[torch.Tensor]],
    total_preclip_norm: torch.Tensor,
    clip_limit: float,
) -> dict[str, Any]:
    # clip_grad_norm_ applies one global scale to every gradient. Reconstruct
    # per-module pre-clip norms from the retained post-clip gradients only at
    # registered diagnostic epochs; this keeps normal optimizer steps free of
    # per-module reduction kernels.
    norm_value = total_preclip_norm.detach().float().cpu()
    clip_scale = torch.clamp(torch.as_tensor(clip_limit) / (norm_value + 1e-6), max=1.0)
    safe_scale = clip_scale.clamp_min(1e-12)
    modules: dict[str, Any] = {}
    for name, values in groups.items():
        if not values:
            modules[name] = None
            continue
        parameter_sq = torch.zeros((), dtype=torch.float32)
        update_sq = torch.zeros((), dtype=torch.float32)
        postclip_grad_sq = values[0].new_zeros((), dtype=torch.float32)
        for parameter, previous in zip(values, before[name]):
            current = parameter.detach().float().cpu()
            parameter_sq = parameter_sq + current.square().sum()
            update_sq = update_sq + (current - previous.float()).square().sum()
            if parameter.grad is not None:
                postclip_grad_sq = postclip_grad_sq + parameter.grad.detach().float().square().sum()
        parameter_norm = torch.sqrt(parameter_sq)
        update_norm = torch.sqrt(update_sq)
        gradient_norm_postclip = torch.sqrt(postclip_grad_sq).detach().float().cpu()
        modules[name] = {
            "parameter_norm": parameter_norm,
            "gradient_norm_preclip": gradient_norm_postclip / safe_scale,
            "gradient_norm_postclip": gradient_norm_postclip,
            "update_norm": update_norm,
            "relative_update": update_norm / parameter_norm.clamp_min(1e-12),
        }
    return json_ready({
        "overall_gradient_norm_preclip": norm_value,
        "clip_scale": clip_scale,
        "modules": modules,
    })


def calibration_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    if len(succeeded) != len(LR_GRID):
        raise RuntimeError("cannot select LR until all seed-52 calibration cells succeeded")
    chosen = max(succeeded, key=lambda row: (
        row["validation"]["accuracy"], row["validation"]["macro_f1"], -float(row["lr"])
    ))
    return {
        "selected_lr": float(chosen["lr"]),
        "selected_run_id": chosen["run_id"],
        "selected_checkpoint_path": chosen["checkpoint_path"],
        "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
        "selection_metric": {
            "validation_accuracy": chosen["validation"]["accuracy"],
            "validation_macro_f1": chosen["validation"]["macro_f1"],
        },
        "candidates": [
            {"lr": float(row["lr"]), "run_id": row["run_id"],
             "validation_accuracy": row["validation"]["accuracy"],
             "validation_macro_f1": row["validation"]["macro_f1"],
             "checkpoint_sha256": row["checkpoint_sha256"]}
            for row in sorted(succeeded, key=lambda row: float(row["lr"]))
        ],
        "rule": "max validation accuracy, then fixed-label Macro-F1, then lower LR",
    }


def expected_config_dict(cfg) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


def read_prior_success(run_dir: Path, source: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "succeeded":
        return None
    if summary.get("source", {}).get("training_source_sha256") != source["training_source_sha256"] or summary.get("input_hashes") != inputs:
        raise RuntimeError(f"existing successful run identity differs; preserving it and refusing implicit retraining: {run_dir}")
    if summary.get("protocol_lock_sha256") != sha256_file(PROTOCOL_LOCK):
        raise RuntimeError(f"existing successful run protocol lock differs; preserving it: {run_dir}")
    for field in ("checkpoint", "predictions", "epoch_log"):
        artifact = summary["raw_artifacts"][field]
        path = Path(artifact["path"])
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"existing successful raw evidence failed hash verification: {path}")
    return summary


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_one(
    dataset: str,
    variant: str,
    phase: str,
    seed: int,
    lr: float,
    device_text: str,
    data,
    edge_index_cpu: torch.Tensor,
    input_hashes: dict[str, Any],
    info: dict[str, int],
    width: int | None,
    calibration_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lock, matrix = verify_frozen_identity()
    if phase not in {"calibration", "confirmation"}:
        raise ValueError(f"invalid formal phase {phase!r}")
    if variant not in VARIANTS or dataset not in DATASETS or seed not in TRAIN_SEEDS:
        raise ValueError("formal run cell is outside the frozen matrix")
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device unavailable: {device_text}")
    source = {
        "training_source_commit": lock["training_source_commit"],
        "training_source_sha256": lock["training_source_sha256"],
        "execution_commit": git_value("rev-parse", "HEAD"),
    }
    cell_id = f"{phase}-{dataset}-{variant}-s{seed}-lr{lr:.7g}"
    parent = OUTPUT_ROOT / dataset / phase / variant
    parent.mkdir(parents=True, exist_ok=True)
    prior = sorted(parent.glob(f"{cell_id}*"))
    for run_dir in prior:
        existing = read_prior_success(run_dir, source, input_hashes)
        if existing is not None:
            existing["reused_existing_success"] = True
            return existing
    attempt_no = len(prior) + 1
    run_id = cell_id if attempt_no == 1 else f"{cell_id}-attempt{attempt_no:02d}"
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempt_path = run_dir / "attempt.json"
    epoch_log_path = run_dir / "epochs.jsonl"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    prediction_path = run_dir / "train_val_argmax.pt"
    summary_path = run_dir / "summary.json"
    cfg = compose_config(dataset, variant, seed, lr, device_text, width)
    cli_phase = "A" if variant in STAGE_A else "B"
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", cli_phase, "--device", device_text]
    attempt = {
        "run_id": run_id, "cell_id": cell_id, "phase": phase, "dataset": dataset,
        "variant": variant, "seed": int(seed), "lr": float(lr), "status": "running",
        "attempt_number": attempt_no, "started_utc": utc_now(), "command": command,
        "device": device_text, "source": source, "input_hashes": input_hashes,
        "calibration_reference": calibration_reference,
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "resolved_config": expected_config_dict(cfg),
        "environment": environment_record(device_text),
        "diagnostic_sample_seed": matrix["datasets"][dataset]["diagnostic_sample_seed"],
        "diagnostic_sample_node_ids": matrix["datasets"][dataset]["diagnostic_sample_node_ids"],
        "test_evaluation": False, "test_prediction": False,
    }
    write_json(attempt_path, attempt)
    epoch_log_path.write_text("", encoding="utf-8")
    wall_started = time.perf_counter()
    try:
        set_seed(int(seed))
        if device_text.startswith("cuda"):
            device = torch.device(device_text)
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
        model = Model(cfg, info).to(device)
        heads = PilotTaskHeads(variant, int(info["num_classes"]), int(model.out_dim)).to(device)
        planned = matrix["datasets"][dataset]["schemes"][variant]
        if n_parameters(model) != int(planned["active_backbone_parameters"]):
            raise RuntimeError(f"active backbone count changed for {dataset}/{variant}")
        if n_parameters(heads) != int(planned["task_head_parameters"]):
            raise RuntimeError(f"task head count changed for {dataset}/{variant}")
        optimizer_params = list(model.parameters()) + list(heads.parameters())
        if len({id(p) for p in optimizer_params}) != len(optimizer_params):
            raise RuntimeError("duplicate optimizer parameters")
        groups = parameter_groups(model, heads)
        optimizer = build_optimizer(optimizer_params, cfg, model=model)
        x = data.x.to(device)
        edges = edge_index_cpu.to(device)
        train_idx_cpu = data.train_idx.detach().cpu().long()
        val_idx_cpu = data.val_idx.detach().cpu().long()
        train_idx = train_idx_cpu.to(device)
        val_idx = val_idx_cpu.to(device)
        diagnostic_sample_idx = torch.as_tensor(
            matrix["datasets"][dataset]["diagnostic_sample_node_ids"],
            dtype=torch.long, device=device,
        )
        # The loader returns the full label vector, but only train and validation
        # IDs are indexed. Test labels are never copied, predicted, or evaluated.
        train_y = data.y.index_select(0, train_idx_cpu).to(device)
        val_y = data.y.index_select(0, val_idx_cpu).to(device)
        macro_labels = list(matrix["datasets"][dataset]["macro_f1_label_set"])

        best_score: tuple[float, float] | None = None
        best_epoch = 0
        patience_left = int(cfg.task.patience)
        best_package: dict[str, Any] | None = None
        best_eval: dict[str, Any] | None = None
        step_times_ms: list[dict[str, Any]] = []
        max_normal_allocated = 0
        max_normal_reserved = 0
        max_diagnostic_allocated = 0
        max_diagnostic_increment = 0
        diagnostic_seconds = 0.0
        snapshot_seconds = 0.0
        diagnostic_epochs: list[int] = []
        parameters = [parameter for parameter in optimizer_params if parameter.requires_grad]
        clip_limit = float(cfg.task.grad_clip)

        for epoch in range(1, int(cfg.task.epochs) + 1):
            model.train(); heads.train()
            model.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            snap_start = time.perf_counter()
            before = parameter_snapshots(groups)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            snapshot_seconds += time.perf_counter() - snap_start

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            z, _, _, model_aux, extras = model(x, edges)
            loss, loss_parts = train_only_loss(
                z, extras, heads, train_y, train_idx, source_aux_weight=0.1,
                labels_are_train_only=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_preclip_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=clip_limit, error_if_nonfinite=True
            )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_finished = time.perf_counter()
            step_ms = max(0.0, step_finished - step_start) * 1000.0
            main_ce_step = float(loss_parts["main_ce"].detach().item())
            source_ce_step = float(loss_parts["source_ce"].detach().item())
            del z, extras, model_aux, loss, loss_parts

            evaluation = evaluate_train_validation(
                model, heads, x, edges, train_idx, val_idx, train_y, val_y,
                int(info["num_classes"]), macro_labels,
            )
            train_metrics = evaluation["train"]
            val_metrics = evaluation["validation"]
            score = (float(val_metrics["accuracy"]), float(val_metrics["macro_f1"]))
            improved = best_score is None or score > best_score
            selected_diag = epoch in {1, 10, 30} or improved
            if device.type == "cuda":
                max_normal_allocated = max(max_normal_allocated, int(torch.cuda.max_memory_allocated(device)))
                max_normal_reserved = max(max_normal_reserved, int(torch.cuda.max_memory_reserved(device)))
                epoch_peak_allocated = int(torch.cuda.max_memory_allocated(device))
                epoch_peak_reserved = int(torch.cuda.max_memory_reserved(device))
            else:
                epoch_peak_allocated = None
                epoch_peak_reserved = None
            step_diag = None
            model_diag: dict[str, Any] = {}
            if selected_diag:
                diag_start = time.perf_counter()
                step_diag = step_diagnostics(groups, before, total_preclip_norm, clip_limit)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline_allocated = torch.cuda.memory_allocated(device)
                else:
                    baseline_allocated = 0
                model_diag = collect_model_diagnostics(model, x, edges, diagnostic_sample_idx)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    diagnostic_peak = int(torch.cuda.max_memory_allocated(device))
                    max_diagnostic_allocated = max(max_diagnostic_allocated, diagnostic_peak)
                    max_diagnostic_increment = max(max_diagnostic_increment, max(diagnostic_peak - baseline_allocated, 0))
                    torch.cuda.reset_peak_memory_stats(device)
                diagnostic_seconds += time.perf_counter() - diag_start
                diagnostic_epochs.append(epoch)
                step_diag["epoch"] = epoch
            step_times_ms.append({"epoch": epoch, "step_ms": step_ms, "diagnostic_epoch": selected_diag})

            if improved:
                best_score = score
                best_epoch = epoch
                patience_left = int(cfg.task.patience)
                if step_diag is None:
                    raise AssertionError("best epoch must carry optimizer-step diagnostics")
                best_package = {
                    "model_state": module_state_cpu(model),
                    "task_heads_state": module_state_cpu(heads),
                    "optimizer_state": cpu_tree(optimizer.state_dict()),
                    "rng_state": capture_rng_state(),
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": val_metrics,
                    "train_main_ce_step": main_ce_step,
                    "train_source_ce_step": source_ce_step,
                    "optimizer_step_diagnostics": step_diag,
                    "model_diagnostics": model_diag,
                    "diagnostic_epochs": list(diagnostic_epochs),
                }
                best_eval = {
                    "train_predictions": evaluation["train_predictions"].clone(),
                    "validation_predictions": evaluation["validation_predictions"].clone(),
                }
            elif epoch >= int(cfg.task.early_stop_min_epoch):
                patience_left -= 1

            event = {
                "epoch": epoch, "train_main_ce_step": main_ce_step,
                "train_source_ce_step": source_ce_step,
                "train": train_metrics, "validation": val_metrics,
                "step_ms_synchronized": step_ms,
                "normal_peak_cuda_allocated_bytes_epoch": epoch_peak_allocated,
                "normal_peak_cuda_reserved_bytes_epoch": epoch_peak_reserved,
                "is_best": bool(improved), "patience_remaining": patience_left,
            }
            if selected_diag:
                event["optimizer_step_diagnostics"] = step_diag
                event["model_diagnostics"] = model_diag
            append_jsonl(epoch_log_path, event)
            del evaluation, before, step_diag, model_diag
            if epoch >= int(cfg.task.early_stop_min_epoch) and patience_left <= 0:
                break

        if best_package is None or best_eval is None:
            raise RuntimeError("run did not produce a validation-selected checkpoint")
        best_package["diagnostic_epochs"] = list(diagnostic_epochs)
        best_package.update({
            "study": "prior_context_mag_pilot_v3", "dataset": dataset, "variant": variant,
            "seed": int(seed), "best_epoch": int(best_epoch), "lr": float(lr), "config": expected_config_dict(cfg),
            "source": source, "input_hashes": input_hashes,
            "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
            "fixed_split_seed": 42,
            "diagnostic_sample_seed": matrix["datasets"][dataset]["diagnostic_sample_seed"],
            "diagnostic_sample_node_ids": matrix["datasets"][dataset]["diagnostic_sample_node_ids"],
            "parameter_counts": {
                "active_backbone": n_parameters(model), "task_heads": n_parameters(heads),
                "total_trainable": n_parameters(model) + n_parameters(heads),
                "backbone_modules": {name: n_parameters(module) for name, module in model.diagnostic_modules().items()},
            },
            "test_evaluation": False, "test_prediction": False,
        })
        torch.save(best_package, checkpoint_path)
        torch.save({
            "train_idx": train_idx_cpu,
            "train_argmax": best_eval["train_predictions"],
            "validation_idx": val_idx_cpu,
            "validation_argmax": best_eval["validation_predictions"],
            "prediction_splits": ["train", "validation"],
        }, prediction_path)
        selected_step_times = [
            item["step_ms"] for item in step_times_ms
            if 2 <= item["epoch"] <= 30 and not item["diagnostic_epoch"]
        ]
        if not selected_step_times:
            raise RuntimeError("no non-diagnostic epoch2-30 timing samples were collected")
        peak_allocated = max_normal_allocated if device.type == "cuda" else None
        peak_reserved = max_normal_reserved if device.type == "cuda" else None
        diag_peak_allocated = max_diagnostic_allocated if device.type == "cuda" else None
        elapsed = time.perf_counter() - wall_started
        summary = {
            "study": "prior_context_mag_pilot_v3", "run_id": run_id, "cell_id": cell_id,
            "phase": phase, "dataset": dataset, "variant": variant, "seed": int(seed),
            "lr": float(lr), "status": "succeeded", "started_utc": attempt["started_utc"],
            "finished_utc": utc_now(), "elapsed_seconds": elapsed,
            "device": device_text, "environment": environment_record(device_text),
            "resolved_config": expected_config_dict(cfg),
            "source": source, "input_hashes": input_hashes,
            "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
            "fixed_split_seed": 42, "train_nodes": int(train_idx_cpu.numel()),
            "validation_nodes": int(val_idx_cpu.numel()), "num_nodes": int(data.num_nodes),
            "canonical_directed_edges": int(edge_index_cpu.size(1)),
            "macro_f1_label_set": macro_labels, "best_epoch": int(best_epoch),
            "epochs_run": len(step_times_ms), "train": best_package["train"],
            "validation": best_package["validation"],
            "train_main_ce_step_best": best_package["train_main_ce_step"],
            "source_train_cross_entropy_at_best": best_package["train_source_ce_step"],
            "best_optimizer_step_ms": next((item["step_ms"] for item in step_times_ms if item["epoch"] == best_epoch), None),
            "median_non_diagnostic_epoch2_30_step_ms": statistics.median(selected_step_times),
            "step_timing_samples_epoch2_30": [item for item in step_times_ms if 2 <= item["epoch"] <= 30],
            "optimizer": {"name": "AdamW", "lr": float(lr), "weight_decay": 1e-4, "grad_clip": clip_limit},
            "parameter_counts": best_package["parameter_counts"],
            "peak_cuda_allocated_bytes_normal_run": peak_allocated,
            "peak_cuda_reserved_bytes_normal_run": peak_reserved,
            "peak_cuda_allocated_bytes_diagnostic_forward": diag_peak_allocated,
            "diagnostic_forward_increment_bytes": max_diagnostic_increment if device.type == "cuda" else None,
            "diagnostic_forward_seconds": diagnostic_seconds,
            "parameter_snapshot_diagnostic_seconds": snapshot_seconds,
            "diagnostic_epochs": best_package["diagnostic_epochs"],
            "diagnostic_sample_seed": best_package["diagnostic_sample_seed"],
            "diagnostic_sample_node_ids": best_package["diagnostic_sample_node_ids"],
            "best_optimizer_step_diagnostics": best_package["optimizer_step_diagnostics"],
            "best_model_diagnostics": best_package["model_diagnostics"],
            "test_evaluation": False, "test_prediction": False,
            "calibration_reference": calibration_reference,
        }
        artifacts = {
            "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
            "predictions": {"path": str(prediction_path), "bytes": prediction_path.stat().st_size, "sha256": sha256_file(prediction_path)},
            "epoch_log": {"path": str(epoch_log_path), "bytes": epoch_log_path.stat().st_size, "sha256": sha256_file(epoch_log_path)},
        }
        summary["raw_artifacts"] = artifacts
        summary["checkpoint_path"] = str(checkpoint_path)
        summary["checkpoint_sha256"] = artifacts["checkpoint"]["sha256"]
        summary["prediction_path"] = str(prediction_path)
        summary["prediction_sha256"] = artifacts["predictions"]["sha256"]
        summary["epoch_log_path"] = str(epoch_log_path)
        summary["epoch_log_sha256"] = artifacts["epoch_log"]["sha256"]
        write_json(summary_path, summary)
        attempt.update({
            "status": "succeeded", "finished_utc": summary["finished_utc"],
            "summary_path": str(summary_path), "raw_artifacts": artifacts,
        })
        write_json(attempt_path, attempt)
        return summary
    except Exception as exc:
        attempt.update({
            "status": "failed", "finished_utc": utc_now(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        write_json(attempt_path, attempt)
        raise


def compact_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"], "cell_id": row["cell_id"], "phase": row["phase"],
        "dataset": row["dataset"], "variant": row["variant"], "seed": row["seed"],
        "lr": row["lr"], "status": row["status"], "best_epoch": row["best_epoch"],
        "train_accuracy": row["train"]["accuracy"], "train_macro_f1": row["train"]["macro_f1"],
        "train_cross_entropy": row["train"]["cross_entropy"],
        "validation_accuracy": row["validation"]["accuracy"],
        "validation_macro_f1": row["validation"]["macro_f1"],
        "validation_cross_entropy": row["validation"]["cross_entropy"],
        "checkpoint_path": row["checkpoint_path"], "checkpoint_sha256": row["checkpoint_sha256"],
        "median_non_diagnostic_epoch2_30_step_ms": row["median_non_diagnostic_epoch2_30_step_ms"],
        "peak_cuda_allocated_bytes_normal_run": row["peak_cuda_allocated_bytes_normal_run"],
        "peak_cuda_reserved_bytes_normal_run": row["peak_cuda_reserved_bytes_normal_run"],
        "raw_artifacts": row["raw_artifacts"],
    }


def summary_for_cell(dataset: str, variant: str, phase: str, seed: int, lr: float) -> dict[str, Any]:
    cell_id = f"{phase}-{dataset}-{variant}-s{seed}-lr{lr:.7g}"
    parent = OUTPUT_ROOT / dataset / phase / variant
    summaries = []
    for path in parent.glob(f"{cell_id}*/summary.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("cell_id") == cell_id and row.get("status") == "succeeded":
            summaries.append(row)
    if len(summaries) != 1:
        raise RuntimeError(f"expected exactly one successful attempt for {cell_id}; found {len(summaries)}")
    # Reuse path validates all three raw hashes.
    read_prior_success(Path(summaries[0]["checkpoint_path"]).parent, summaries[0]["source"], summaries[0]["input_hashes"])
    return summaries[0]


def new_selection() -> dict[str, Any]:
    return {"study": "prior_context_mag_pilot_v3", "calibration": {}, "selected_lr": {}, "confirmations": {}, "stage_b": {"status": "pending_gate"}}


def load_selection() -> dict[str, Any]:
    path = REPORT_ROOT / "selection.json"
    if not path.is_file():
        return new_selection()
    return json.loads(path.read_text(encoding="utf-8"))


def save_selection(selection: dict[str, Any]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_ROOT / "selection.json", selection)


def run_stage(
    schemes: tuple[str, ...],
    phase_name: str,
    device: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    for dataset in DATASETS:
        data, edge_cpu, inputs, info = load_locked_dataset(dataset)
        widths = json.loads((STUDY_ROOT / "matrix.json").read_text(encoding="utf-8"))["datasets"][dataset]["capacity"]["widths"]
        for variant in schemes:
            print(f"[{phase_name}] {dataset}/{variant}: seed52 LR calibration", flush=True)
            rows = []
            for lr in LR_GRID:
                row = run_one(
                    dataset, variant, "calibration", 52, lr, device, data,
                    edge_cpu, inputs, info, widths.get(variant),
                )
                rows.append(row)
                print(
                    f"  calibration lr={lr:g} best={row['best_epoch']} "
                    f"val={row['validation']['accuracy']:.5f}/{row['validation']['macro_f1']:.5f} "
                    f"({row['run_id']})", flush=True,
                )
            choice = calibration_choice(rows)
            selection.setdefault("calibration", {}).setdefault(dataset, {})[variant] = [compact_run(row) for row in rows]
            selection.setdefault("selected_lr", {}).setdefault(dataset, {})[variant] = choice
            selection.setdefault("confirmations", {}).setdefault(dataset, {})[variant] = {}
            save_selection(selection)
            print(f"[{phase_name}] {dataset}/{variant}: selected lr={choice['selected_lr']:g}; seed43/54 confirmation", flush=True)
            for seed in (53, 54):
                row = run_one(
                    dataset, variant, "confirmation", seed, choice["selected_lr"], device,
                    data, edge_cpu, inputs, info, widths.get(variant),
                    calibration_reference={
                        "selected_lr": choice["selected_lr"],
                        "seed52_run_id": choice["selected_run_id"],
                        "seed52_checkpoint_sha256": choice["selected_checkpoint_sha256"],
                    },
                )
                selection["confirmations"][dataset][variant][str(seed)] = compact_run(row)
                save_selection(selection)
                print(
                    f"  seed={seed} best={row['best_epoch']} "
                    f"val={row['validation']['accuracy']:.5f}/{row['validation']['macro_f1']:.5f} "
                    f"({row['run_id']})", flush=True,
                )
    return selection


def compute_gate(selection: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    metric_fields = {"accuracy": "validation_accuracy", "macro_f1": "validation_macro_f1"}
    for baseline in ("R0", "R1"):
        pairs = []
        by_dataset: dict[str, Any] = {}
        for dataset in DATASETS:
            ds_pairs = []
            for seed in (53, 54):
                candidate = selection["confirmations"][dataset]["R4"][str(seed)]
                control = selection["confirmations"][dataset][baseline][str(seed)]
                pair = {
                    "dataset": dataset, "seed": seed,
                    "delta_accuracy": candidate["validation_accuracy"] - control["validation_accuracy"],
                    "delta_macro_f1": candidate["validation_macro_f1"] - control["validation_macro_f1"],
                }
                pairs.append(pair); ds_pairs.append(pair)
            by_dataset[dataset] = {
                "mean_delta_accuracy": statistics.mean(row["delta_accuracy"] for row in ds_pairs),
                "mean_delta_macro_f1": statistics.mean(row["delta_macro_f1"] for row in ds_pairs),
            }
        mean_acc = statistics.mean(row["delta_accuracy"] for row in pairs)
        mean_f1 = statistics.mean(row["delta_macro_f1"] for row in pairs)
        positive_acc = sum(row["delta_accuracy"] > 0 for row in pairs)
        positive_f1 = sum(row["delta_macro_f1"] > 0 for row in pairs)
        acc_path = mean_acc >= 0.005 and mean_f1 >= 0.0 and positive_acc >= 3
        f1_path = mean_f1 >= 0.01 and mean_acc >= 0.0 and positive_f1 >= 3
        per_graph = all(
            by_dataset[dataset]["mean_delta_accuracy"] >= -0.002
            and by_dataset[dataset]["mean_delta_macro_f1"] >= -0.01
            for dataset in DATASETS
        )
        comparisons[baseline] = {
            "pairs": pairs, "by_dataset": by_dataset,
            "mean_delta_accuracy": mean_acc, "mean_delta_macro_f1": mean_f1,
            "positive_accuracy_pairs": positive_acc, "positive_macro_f1_pairs": positive_f1,
            "accuracy_path_pass": acc_path, "macro_f1_path_pass": f1_path,
            "per_graph_floor_pass": per_graph,
            "metric_gate_pass": (acc_path or f1_path) and per_graph,
        }

    by_dataset_step_ratios = {}
    by_dataset_memory_ratios = {}
    for dataset in DATASETS:
        r4_runs = [selection["confirmations"][dataset]["R4"][str(seed)] for seed in (53, 54)]
        r1_runs = [selection["confirmations"][dataset]["R1"][str(seed)] for seed in (53, 54)]
        r4_step = statistics.mean(row["median_non_diagnostic_epoch2_30_step_ms"] for row in r4_runs)
        r1_step = statistics.mean(row["median_non_diagnostic_epoch2_30_step_ms"] for row in r1_runs)
        r4_mem = statistics.mean(row["peak_cuda_allocated_bytes_normal_run"] for row in r4_runs)
        r1_mem = statistics.mean(row["peak_cuda_allocated_bytes_normal_run"] for row in r1_runs)
        by_dataset_step_ratios[dataset] = {"r4_mean_ms": r4_step, "r1_mean_ms": r1_step, "ratio": r4_step / max(r1_step, 1e-12)}
        by_dataset_memory_ratios[dataset] = {"r4_mean_bytes": r4_mem, "r1_mean_bytes": r1_mem, "ratio": r4_mem / max(r1_mem, 1.0)}
    mean_step_ratio = statistics.mean(row["ratio"] for row in by_dataset_step_ratios.values())
    mean_memory_ratio = statistics.mean(row["ratio"] for row in by_dataset_memory_ratios.values())
    resource_pass = mean_step_ratio <= 3.0 and mean_memory_ratio <= 3.0
    metric_pass = all(comparisons[base]["metric_gate_pass"] for base in ("R0", "R1"))
    return {
        "uses_confirmation_seeds_only": [53, 54],
        "comparisons": comparisons,
        "step_time_ratios_by_dataset": by_dataset_step_ratios,
        "peak_allocated_ratios_by_dataset": by_dataset_memory_ratios,
        "mean_step_time_ratio": mean_step_ratio,
        "mean_peak_allocated_ratio": mean_memory_ratio,
        "resource_pass": resource_pass,
        "metric_pass": metric_pass,
        "passed": metric_pass and resource_pass,
    }


def run_phase_a(device: str) -> dict[str, Any]:
    selection = load_selection()
    selection["stage_a_started_utc"] = utc_now()
    selection = run_stage(STAGE_A, "A", device, selection)
    gate = compute_gate(selection)
    selection["stage_a_gate"] = gate
    selection["stage_a_finished_utc"] = utc_now()
    selection["stage_b"]["status"] = "ready" if gate["passed"] else "skipped_gate"
    if not gate["passed"]:
        selection["stage_b"]["reason"] = "R4 did not satisfy the frozen validation and/or resource gate against both R0 and R1."
    save_selection(selection)
    return selection


def run_phase_b(device: str) -> dict[str, Any]:
    selection = load_selection()
    if not selection.get("stage_a_gate"):
        raise RuntimeError("Stage A selection/gate evidence is missing; run --phase A first")
    if not selection["stage_a_gate"]["passed"]:
        selection["stage_b"] = {
            "status": "skipped_gate", "reason": "Stage A gate failed; frozen rule forbids Stage B training.",
            "decision_utc": utc_now(),
        }
        save_selection(selection)
        print("Stage B skipped: the frozen Stage A gate did not pass.", flush=True)
        return selection
    selection["stage_b"]["status"] = "running"
    selection["stage_b"]["started_utc"] = utc_now()
    save_selection(selection)
    selection = run_stage(STAGE_B, "B", device, selection)
    selection["stage_b"].update({"status": "succeeded", "finished_utc": utc_now()})
    save_selection(selection)
    return selection


def all_attempt_records() -> list[dict[str, Any]]:
    records = []
    if OUTPUT_ROOT.exists():
        for attempt_path in sorted(OUTPUT_ROOT.rglob("attempt.json")):
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            row = {"attempt": attempt, "attempt_path": str(attempt_path)}
            summary_path = attempt_path.parent / "summary.json"
            if summary_path.is_file():
                row["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
                row["summary_path"] = str(summary_path)
            records.append(row)
    return records


def expected_matrix_cells(selection: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    gate_pass = bool(selection.get("stage_a_gate", {}).get("passed", False))
    for dataset in DATASETS:
        for variant in STAGE_A:
            for lr in LR_GRID:
                cells.append({"phase": "calibration", "dataset": dataset, "variant": variant,
                              "seed": 52, "lr": lr, "status": "planned"})
            choice = selection.get("selected_lr", {}).get(dataset, {}).get(variant)
            for seed in (53, 54):
                cells.append({"phase": "confirmation", "dataset": dataset, "variant": variant,
                              "seed": seed, "lr": None if choice is None else choice["selected_lr"],
                              "status": "planned" if choice is not None else "blocked_missing_calibration"})
        for variant in STAGE_B:
            stage_b_status = "planned" if gate_pass else "skipped_gate"
            for lr in LR_GRID:
                cells.append({"phase": "calibration", "dataset": dataset, "variant": variant,
                              "seed": 52, "lr": lr, "status": stage_b_status})
            choice = selection.get("selected_lr", {}).get(dataset, {}).get(variant)
            for seed in (53, 54):
                cells.append({"phase": "confirmation", "dataset": dataset, "variant": variant,
                              "seed": seed, "lr": None if choice is None else choice["selected_lr"],
                              "status": stage_b_status if choice is None else "planned"})
    attempts = all_attempt_records()
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        attempt = row["attempt"]
        by_cell.setdefault(attempt["cell_id"], []).append(row)
    for cell in cells:
        if cell["lr"] is None:
            continue
        cid = f'{cell["phase"]}-{cell["dataset"]}-{cell["variant"]}-s{cell["seed"]}-lr{cell["lr"]:.7g}'
        rows = by_cell.get(cid, [])
        if rows:
            cell["attempt_ids"] = [item["attempt"]["run_id"] for item in rows]
            statuses = [item["attempt"]["status"] for item in rows]
            if any(status == "succeeded" for status in statuses):
                cell["status"] = "succeeded"
            elif "running" in statuses:
                cell["status"] = "interrupted_or_running"
            else:
                cell["status"] = "failed"
    return cells


def build_result_rows() -> list[dict[str, Any]]:
    rows = []
    for entry in all_attempt_records():
        summary = entry.get("summary")
        if not summary or summary.get("status") != "succeeded":
            continue
        # One successful attempt per planned cell; refuse hidden duplicated training.
        rows.append({
            "run_id": summary["run_id"], "cell_id": summary["cell_id"],
            "phase": summary["phase"], "dataset": summary["dataset"],
            "variant": summary["variant"], "seed": summary["seed"], "lr": summary["lr"],
            "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
            "train": summary["train"], "validation": summary["validation"],
            "train_main_ce_step_best": summary["train_main_ce_step_best"],
            "train_source_cross_entropy_at_best": summary["source_train_cross_entropy_at_best"],
            "parameter_counts": summary["parameter_counts"],
            "median_non_diagnostic_epoch2_30_step_ms": summary["median_non_diagnostic_epoch2_30_step_ms"],
            "step_timing_samples_epoch2_30": summary["step_timing_samples_epoch2_30"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_cuda_allocated_bytes_normal_run": summary["peak_cuda_allocated_bytes_normal_run"],
            "peak_cuda_reserved_bytes_normal_run": summary["peak_cuda_reserved_bytes_normal_run"],
            "peak_cuda_allocated_bytes_diagnostic_forward": summary["peak_cuda_allocated_bytes_diagnostic_forward"],
            "diagnostic_forward_increment_bytes": summary["diagnostic_forward_increment_bytes"],
            "diagnostic_forward_seconds": summary["diagnostic_forward_seconds"],
            "parameter_snapshot_diagnostic_seconds": summary["parameter_snapshot_diagnostic_seconds"],
            "best_optimizer_step_diagnostics": summary["best_optimizer_step_diagnostics"],
            "best_model_diagnostics": summary["best_model_diagnostics"],
            "diagnostic_sample_seed": summary["diagnostic_sample_seed"],
            "diagnostic_sample_node_ids": summary["diagnostic_sample_node_ids"],
            "macro_f1_label_set": summary["macro_f1_label_set"],
            "source": summary["source"], "input_hashes": summary["input_hashes"],
            "protocol_lock_sha256": summary["protocol_lock_sha256"],
            "resolved_config": summary["resolved_config"],
            "calibration_reference": summary.get("calibration_reference"),
            "checkpoint_path": summary["checkpoint_path"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "prediction_path": summary["prediction_path"],
            "prediction_sha256": summary["prediction_sha256"],
            "epoch_log_path": summary["epoch_log_path"],
            "epoch_log_sha256": summary["epoch_log_sha256"],
            "raw_artifacts": summary["raw_artifacts"],
            "test_evaluation": False, "test_prediction": False,
        })
    return sorted(rows, key=lambda row: (row["dataset"], row["variant"], row["phase"], row["seed"], row["lr"]))


def pair_differences(rows: list[dict[str, Any]], candidate: str, comparator: str) -> dict[str, Any]:
    by_key = {(row["dataset"], row["variant"], row["seed"]): row for row in rows if row["phase"] == "confirmation"}
    pairs = []
    for dataset in DATASETS:
        for seed in (53, 54):
            left = by_key[(dataset, candidate, seed)]
            right = by_key[(dataset, comparator, seed)]
            pairs.append({
                "dataset": dataset, "seed": seed,
                "delta_accuracy": left["validation"]["accuracy"] - right["validation"]["accuracy"],
                "delta_macro_f1": left["validation"]["macro_f1"] - right["validation"]["macro_f1"],
            })
    return {
        "candidate_minus_comparator": f"{candidate}-{comparator}",
        "pairs": pairs,
        "mean_delta_accuracy": statistics.mean(item["delta_accuracy"] for item in pairs),
        "mean_delta_macro_f1": statistics.mean(item["delta_macro_f1"] for item in pairs),
        "positive_accuracy_pairs": sum(item["delta_accuracy"] > 0 for item in pairs),
        "positive_macro_f1_pairs": sum(item["delta_macro_f1"] > 0 for item in pairs),
    }


def summarize_metrics(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def format_mean_std(values: list[float], scale: float = 100.0, digits: int = 2) -> str:
    mean, std = summarize_metrics(values)
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def generate_report(selection: dict[str, Any], rows: list[dict[str, Any]], matrix: dict[str, Any]) -> str:
    gate = selection.get("stage_a_gate")
    stage_b_status = selection.get("stage_b", {}).get("status", "pending_gate")
    n_success = len(rows)
    attempts = all_attempt_records()
    n_failed = sum(entry["attempt"].get("status") == "failed" for entry in attempts)
    n_incomplete = sum(entry["attempt"].get("status") == "running" for entry in attempts)
    micro_path = OUTPUT_ROOT / "micro_attempts.json"
    micro_attempts = json.loads(micro_path.read_text(encoding="utf-8")).get("attempts", []) if micro_path.is_file() else []
    micro_passed = sum(item.get("status") == "passed" for item in micro_attempts)
    micro_failed = sum(item.get("status", "").startswith("failed") for item in micro_attempts)
    lines = [
        "# Prior-context MAG pilot v3 report", "",
        "## Decision and scope", "",
    ]
    if gate is None:
        lines.append("Stage A is incomplete, so the frozen gate has not been evaluated. Stage B remains unrun. No model promotion decision is made from incomplete evidence.")
    elif gate["passed"]:
        lines.append("Stage A passed the pre-registered validation and resource gate against both R0 and R1. Stage B was therefore run under the already frozen matrix. This engineering gate is a budget rule, not a significance test.")
    else:
        lines.append("Stage A did not pass the pre-registered validation and/or resource gate against both R0 and R1. Stage B was skipped exactly as specified; the matrix, loss, and selection rules were not expanded after seeing validation results.")
    lines.extend([
        "The v2 frozen evidence and protocol remain unchanged. V2 already supplied unpropagated h0 to its final FusionMLP; v3 explicitly adds a coefficient-one center representation and source supervision, without claiming lossless preservation. This run covers Movies/Grocery train/validation only. No test predictions or metrics, LP, or five-NC extension were run.",
        f"Formal successful cells: {n_success}; failed attempts: {n_failed}; incomplete/running attempts: {n_incomplete}; Stage B status: `{stage_b_status}`.",
        f"Synthetic GPU smoke attempts: {micro_passed} passed and {micro_failed} GPU-visibility environment failures recorded; these did not use real graph data.",
        "Movies and Grocery are previously studied development graphs; seed52 is LR calibration, while only seeds53/54 enter gate comparisons. The seed summaries below are descriptive and do not imply statistical significance.",
        "",
    ])
    if gate:
        lines += ["### Stage A gate", "", "| R4 comparator | Mean Δ Acc (pp) | Mean Δ Macro-F1 (pp) | Positive pairs Acc/F1 | Metric path | Per-graph floor |", "|---|---:|---:|---:|---|---|"]
        for comparator, item in gate["comparisons"].items():
            path = "Acc" if item["accuracy_path_pass"] else ("Macro-F1" if item["macro_f1_path_pass"] else "fail")
            lines.append(f"| R4−{comparator} | {item['mean_delta_accuracy']*100:+.3f} | {item['mean_delta_macro_f1']*100:+.3f} | {item['positive_accuracy_pairs']}/4 / {item['positive_macro_f1_pairs']}/4 | {path} | {'pass' if item['per_graph_floor_pass'] else 'fail'} |")
        lines += [
            "", f"Confirmation-seed mean R4/R1 step-time ratio: {gate['mean_step_time_ratio']:.3f}; allocated-memory ratio: {gate['mean_peak_allocated_ratio']:.3f}; resource gate: {'pass' if gate['resource_pass'] else 'fail'}.",
            "Both R4 baselines must pass a metric path and the per-graph floor; resource ratios are compared against R1 and equally average the two graph ratios.", "",
        ]
    lines += [
        "## Validation and train metrics", "",
        "Mean ± population standard deviation (ddof=0), percentage units. Each selected row includes seed52 calibration checkpoint reused plus seeds53/54 confirmation.",
        "",
        "| Dataset | Scheme | Val Acc | Val Macro-F1 | Train Acc | Train Macro-F1 | Train−Val Acc (pp) | Train−Val F1 (pp) | LR seeds52–54 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    available_schemes = [v for v in VARIANTS if any(r["dataset"] == "Movies" and r["variant"] == v for r in rows)]
    for dataset in DATASETS:
        for variant in available_schemes:
            matched = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant and r["seed"] in (52, 53, 54)]
            if len(matched) != 3:
                continue
            val_acc = [r["validation"]["accuracy"] for r in matched]
            val_f1 = [r["validation"]["macro_f1"] for r in matched]
            train_acc = [r["train"]["accuracy"] for r in matched]
            train_f1 = [r["train"]["macro_f1"] for r in matched]
            gap_acc = statistics.mean(train_acc) - statistics.mean(val_acc)
            gap_f1 = statistics.mean(train_f1) - statistics.mean(val_f1)
            lr_text = ", ".join(f"s{r['seed']}={r['lr']:g}" for r in sorted(matched, key=lambda r:r["seed"]))
            lines.append(
                f"| {dataset} | {variant} | {format_mean_std(val_acc)} | {format_mean_std(val_f1)} | {format_mean_std(train_acc)} | {format_mean_std(train_f1)} | {gap_acc*100:+.2f} | {gap_f1*100:+.2f} | {lr_text} |"
            )
    lines += ["", "### Seed-level records", "", "| Dataset | Scheme | Seed | LR | Train Acc/F1 (%) | Train CE step | Val Acc/F1 (%) | Val CE | Best epoch | Wall (min) | Step median (ms) | Peak allocated (GiB) | Peak reserved (GiB) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        train = row["train"]; val = row["validation"]
        alloc = row["peak_cuda_allocated_bytes_normal_run"]
        alloc_text = "n/a" if alloc is None else f"{alloc / 1024**3:.2f}"
        reserved = row["peak_cuda_reserved_bytes_normal_run"]
        reserved_text = "n/a" if reserved is None else f"{reserved / 1024**3:.2f}"
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['seed']} | {row['lr']:g} | {train['accuracy']*100:.2f}/{train['macro_f1']*100:.2f} | {row['train_main_ce_step_best']:.4f}/{row['train_source_cross_entropy_at_best']:.4f} | {val['accuracy']*100:.2f}/{val['macro_f1']*100:.2f} | {val['cross_entropy']:.4f} | {row['best_epoch']} | {row['elapsed_seconds']/60:.2f} | {row['median_non_diagnostic_epoch2_30_step_ms']:.3f} | {alloc_text} | {reserved_text} |"
        )
    lines += ["", "## Matched confirmation differences", "", "Matched seed53/54 pairs; values are candidate minus comparator. These descriptive differences use 4 cells (2 graphs × 2 seeds).", "", "| Comparison | Mean Δ Acc (pp) | Mean Δ Macro-F1 (pp) | Positive Acc pairs | Positive F1 pairs |", "|---|---:|---:|---:|---:|"]
    comparisons = []
    if gate:
        for base in ("R0", "R1"):
            comparisons.append(pair_differences(rows, "R4", base))
    if stage_b_status == "succeeded":
        for base in STAGE_B:
            comparisons.append(pair_differences(rows, "R4", base))
    for item in comparisons:
        lines.append(f"| {item['candidate_minus_comparator']} | {item['mean_delta_accuracy']*100:+.3f} | {item['mean_delta_macro_f1']*100:+.3f} | {item['positive_accuracy_pairs']}/4 | {item['positive_macro_f1_pairs']}/4 |")
    lines += ["", "## Capacity, optimization cost, and diagnostics", "", "Backbone parameter widths were chosen by active trainable backbone parameter count before results. Main/source heads are listed separately. Parameter matching does not imply matched compute, optimization, or function family.", "", "| Dataset | Scheme | Width | Active backbone params | NC task heads | Total trainable | Δ vs R4 |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            cap = matrix["datasets"][dataset]["schemes"][variant]
            lines.append(f"| {dataset} | {variant} | {cap['width'] if cap['width'] is not None else 'fixed'} | {cap['active_backbone_parameters']:,} | {cap['task_head_parameters']:,} | {cap['total_trainable_parameters']:,} | {cap['delta_percent']:+.3f}% |")
    lines += ["", "Selected-run route, S/D, representation, and module update summaries are retained in `diagnostics_summary.json` and raw epoch logs. Router values are health diagnostics only, not true roles, information fractions, causal utility, or evidence of gain. The best-step optimizer records include pre/post-clip gradients, parameter norms, actual relative updates, global pre-clip norm, and clip scale.", "", "## Interpretation and limitations", ""]
    if gate and not gate["passed"]:
        lines.append("R4 did not clear the frozen minimum validation/resource bar against both the topology-free MLP and wide ordinary graph control. No extra mechanism family was trained in Stage B.")
    elif stage_b_status == "succeeded":
        lines.append("Stage A cleared its engineering gate. Stage B contrasts are interpreted only against their named matched controls; an R4 advantage does not show every component is necessary, and failure of a component contrast lowers that component's priority.")
    else:
        lines.append("There is no complete gate result from which to decide whether the candidate merits follow-up. Incomplete evidence is not a positive finding.")
    lines += [
        "S/D encode transformed sender values and transformed edge differences before aggregation; they do not establish homophily/heterophily roles. J/T/V are processing paths, not identifiable shared/unique information. The center path is an explicit coefficient-one route through learned 128-D adapters and can still be counteracted by the correction.",
        "Potential overfitting, optimizer interactions, parameter-count versus compute mismatch, fixed split reuse, earlier exposure of the development graphs, and small descriptive seed count limit interpretation. No causal mechanism is claimed. R6 is a paper-style reference, not a complete author implementation reproduction.",
        "No test labels were indexed, no test predictions/logits were produced, no LP or other NC datasets were run, and no post-result hyperparameter/module search was made.",
        "",
        "## Evidence locations", "",
        "- Frozen protocol and lock: `docs/prior_context_pilot_v3/`.",
        "- Aggregated results, selection, registry, and hashes: `reports/prior_context_pilot_v3/`.",
        "- Ignored raw checkpoints, per-epoch train/validation logs, and train/validation argmax only: `outputs/prior_context_pilot_v3/`.",
    ]
    return "\n".join(lines) + "\n"


def write_final_artifacts(selection: dict[str, Any]) -> dict[str, Any]:
    lock, matrix = verify_frozen_identity()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_result_rows()
    attempts = all_attempt_records()
    cells = expected_matrix_cells(selection)
    cell_status_counts: dict[str, int] = {}
    for cell in cells:
        cell_status_counts[cell["status"]] = cell_status_counts.get(cell["status"], 0) + 1
    success_by_cell = [row["cell_id"] for row in rows]
    if len(set(success_by_cell)) != len(success_by_cell):
        raise RuntimeError("more than one successful training attempt exists for a formal cell")

    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        event_path = Path(row["epoch_log_path"])
        selected_epochs = {1, 10, 30, row["best_epoch"]}
        events = []
        for line in event_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["epoch"] in selected_epochs and "optimizer_step_diagnostics" in event:
                events.append({
                    "epoch": event["epoch"], "is_best": event["is_best"],
                    "optimizer_step_diagnostics": event["optimizer_step_diagnostics"],
                    "model_diagnostics": event["model_diagnostics"],
                })
        diagnostics.append({
            "run_id": row["run_id"], "dataset": row["dataset"], "variant": row["variant"],
            "seed": row["seed"], "best_epoch": row["best_epoch"],
            "diagnostic_sample_seed": row["diagnostic_sample_seed"],
            "diagnostic_sample_node_ids": row["diagnostic_sample_node_ids"],
            "selected_epoch_diagnostics": events,
            "best_optimizer_step_diagnostics": row["best_optimizer_step_diagnostics"],
            "best_model_diagnostics": row["best_model_diagnostics"],
        })
    write_json(REPORT_ROOT / "diagnostics_summary.json", {
        "scope": "train/validation-only; module norms from actual optimizer steps; representation/router summaries use unlabeled forward on pre-registered train/validation diagnostic nodes",
        "runs": diagnostics,
    })

    with (REPORT_ROOT / "run_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")

    attempt_rows = []
    raw_files: list[dict[str, Any]] = []
    for entry in attempts:
        attempt_path = Path(entry["attempt_path"])
        attempt_row = {"attempt": entry["attempt"], "attempt_path": str(attempt_path)}
        raw_files.append({"kind": "attempt_record", "path": str(attempt_path), "bytes": attempt_path.stat().st_size, "sha256": sha256_file(attempt_path)})
        if entry.get("summary_path"):
            summary_path = Path(entry["summary_path"])
            attempt_row["summary_path"] = str(summary_path)
            attempt_row["summary_sha256"] = sha256_file(summary_path)
            raw_files.append({"kind": "summary", "path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)})
        if entry.get("summary"):
            attempt_row["result"] = compact_run(entry["summary"])
        attempt_rows.append(attempt_row)
        for key, artifact in entry.get("summary", {}).get("raw_artifacts", {}).items():
            raw_files.append({"kind": key, **artifact})
        epoch_path = attempt_path.parent / "epochs.jsonl"
        if epoch_path.is_file() and not any(item["path"] == str(epoch_path) for item in raw_files):
            raw_files.append({"kind": "epoch_log_or_partial", "path": str(epoch_path), "bytes": epoch_path.stat().st_size, "sha256": sha256_file(epoch_path)})
    micro_path = OUTPUT_ROOT / "micro_attempts.json"
    if micro_path.is_file():
        raw_files.append({"kind": "micro_attempts", "path": str(micro_path), "bytes": micro_path.stat().st_size, "sha256": sha256_file(micro_path)})

    run_registry = {
        "study": "prior_context_mag_pilot_v3",
        "training_source_commit": lock["training_source_commit"],
        "training_source_sha256": lock["training_source_sha256"],
        "execution_commit": git_value("rev-parse", "HEAD"),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "stage_a_gate": selection.get("stage_a_gate"),
        "stage_b_status": selection.get("stage_b", {}).get("status"),
        "planned_cells": cells,
        "attempts": attempt_rows,
        "micro_attempts_path": str(micro_path),
        "micro_attempts_sha256": sha256_file(micro_path) if micro_path.is_file() else None,
        "selection": selection,
        "counts": {
            "successful_attempts": sum(row["attempt"].get("status") == "succeeded" for row in attempt_rows),
            "failed_attempts": sum(row["attempt"].get("status") == "failed" for row in attempt_rows),
            "interrupted_attempts": sum(row["attempt"].get("status") == "running" for row in attempt_rows),
            "successful_cells": len(set(success_by_cell)),
            "planned_cell_statuses": cell_status_counts,
            "seed52_selected_checkpoints_reused_as_calibration": sum(
                len(selection.get("selected_lr", {}).get(dataset, {})) for dataset in DATASETS
            ),
        },
    }
    write_json(REPORT_ROOT / "run_registry.json", run_registry)

    report_text = generate_report(selection, rows, matrix)
    (REPORT_ROOT / "report.md").write_text(report_text, encoding="utf-8")

    report_files = [
        "report.md", "selection.json", "run_registry.json", "run_results.jsonl", "diagnostics_summary.json",
    ]
    file_hashes = {
        f"reports/prior_context_pilot_v3/{name}": {
            "bytes": (REPORT_ROOT / name).stat().st_size,
            "sha256": sha256_file(REPORT_ROOT / name),
        }
        for name in report_files
    }
    for relative in ("review.md", "protocol_audit.md", "audit_result.json"):
        path = REPORT_ROOT / relative
        if path.is_file():
            file_hashes[f"reports/prior_context_pilot_v3/{relative}"] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "study": "prior_context_mag_pilot_v3",
        "branch": git_value("branch", "--show-current"),
        "baseline_commit": BASE_COMMIT,
        "training_source_commit": lock["training_source_commit"],
        "training_source_sha256": lock["training_source_sha256"],
        "execution_commit": git_value("rev-parse", "HEAD"),
        "frozen_protocol_lock_path": str(PROTOCOL_LOCK),
        "frozen_protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "v1_evidence_root": str(ROOT / "outputs/problem_validation_v1"),
        "v1_protocol_lock_path": str(V1_LOCK),
        "v1_protocol_lock_sha256": lock["v1_protocol_lock_sha256"],
        "v1_training_source_commit": v1_lock()["training_source_commit"],
        "v1_evaluation_code_commit": v1_lock()["evaluation_code_commit"],
        "v2_report_manifest_sha256": sha256_file(V2_MANIFEST),
        "design_sha256": lock["design_sha256"], "protocol_sha256": lock["protocol_sha256"],
        "matrix_sha256": lock["matrix_sha256"], "capacity_sha256": lock["capacity_sha256"],
        "fixed_inputs": lock["fixed_inputs"],
        "output_root": str(OUTPUT_ROOT),
        "stage_a_gate": selection.get("stage_a_gate"),
        "stage_b_status": selection.get("stage_b", {}).get("status"),
        "formal_attempt_limit": 70,
        "counts": run_registry["counts"],
        "raw_files": raw_files,
        "report_files": file_hashes,
        "test_prediction": False, "test_evaluation": False, "lp": False,
    }
    write_json(REPORT_ROOT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged, train/validation-only MAG prior-context pilot v3")
    parser.add_argument("--phase", choices=("plan", "freeze", "A", "B", "finalize"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.phase == "plan":
        matrix = build_plan()
        print(json.dumps({
            "matrix": str(STUDY_ROOT / "matrix.json"),
            "capacity": str(STUDY_ROOT / "capacity.csv"),
            "dataset_capacity": {ds: matrix["datasets"][ds]["capacity"] for ds in DATASETS},
        }, ensure_ascii=False, indent=2), flush=True)
        return
    if args.phase == "freeze":
        lock = freeze_lock()
        print(json.dumps({
            "training_source_commit": lock["training_source_commit"],
            "training_source_sha256": lock["training_source_sha256"],
            "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
            "path": str(PROTOCOL_LOCK),
        }, ensure_ascii=False, indent=2), flush=True)
        return
    if args.phase in {"A", "B"} and not args.device.startswith("cuda"):
        raise ValueError("formal cells require the available CUDA device; CPU is reserved for synthetic checks")
    if args.phase == "A":
        selection = run_phase_a(args.device)
        write_final_artifacts(selection)
        print(f"Stage A gate: {'PASS' if selection['stage_a_gate']['passed'] else 'FAIL'}", flush=True)
        return
    if args.phase == "B":
        selection = run_phase_b(args.device)
        write_final_artifacts(selection)
        print(f"Stage B status: {selection['stage_b']['status']}", flush=True)
        return
    selection = load_selection()
    manifest = write_final_artifacts(selection)
    print(json.dumps({"manifest": str(REPORT_ROOT / "manifest.json"), "counts": manifest["counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
