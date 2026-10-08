from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.data.loaders import resolve_path
from src.models.factory import build_model
from src.models.mag_collab_adapt import MAGCollabAdapt, parameter_groups_no_decay
from src.models.mvcge_coordinated_nc_v2 import MVCGECoordinatedNCV2
from src.tasks.collab_nc_pilot import CollabTaskHeads, train_only_loss
from src.tasks.mvcge_coordinated_nc_v2 import CoordinatedNCHeads, coordinated_loss, indexed_main_logits, labels_train_val_only

STUDY = "mvcge_coordinated_nc_v2"
SELECTED_GPU_UUID = "GPU-28bf6c52-0671-4e90-e88b-4e47457b7cce"
DATASETS = ("Movies", "Grocery", "Reddit-S")
MODELS = ("A0", "N0", "M0", "M1", "P0", "P1")
NEW_MODELS = MODELS[1:]
LRS = (0.0003, 0.001, 0.003)
WDS = (0.0001, 0.001, 0.01)
CALIBRATION_SEED = 112
CONFIRMATION_SEEDS = (113, 114, 115)
SPLIT_SEED = 42
MAX_UPDATES = 1000
MIN_UPDATES = 200
EVAL_FIRST = 21
EVAL_INTERVAL = 5
PATIENCE = 150
GRAD_DIAGNOSTIC_STEPS = (1, 50, 100, 200)
DOC_ROOT = ROOT / "docs" / STUDY
REPORT_ROOT = ROOT / "reports" / STUDY
OUTPUT_ROOT = ROOT / "outputs" / STUDY
MATRIX_PATH = DOC_ROOT / "matrix.json"
LOCK_PATH = DOC_ROOT / "protocol_lock.json"
REGISTRY_PATH = REPORT_ROOT / "run_registry.json"
RESULTS_PATH = REPORT_ROOT / "run_results.jsonl"
SELECTION_PATH = REPORT_ROOT / "selection_lock.json"
DATA_CACHE: dict[str, dict[str, Any]] = {}

SOURCE_FILES = (
    "src/models/mvcge_coordinated_nc_v2.py",
    "src/tasks/mvcge_coordinated_nc_v2.py",
    "scripts/run_mvcge_coordinated_nc_v2.py",
    "scripts/audit_mvcge_coordinated_nc_v2.py",
    "scripts/report_mvcge_coordinated_nc_v2.py",
    "tests/test_mvcge_coordinated_nc_v2.py",
    "configs/model/mvcge_coordinated_nc_v2.yaml",
    "src/models/mag_collab_adapt.py",
    "src/models/predictor.py",
    "src/tasks/collab_nc_pilot.py",
    "src/tasks/prior_context_pilot.py",
    "src/data/__init__.py",
    "src/data/loaders.py",
    "src/data/graph_utils.py",
    "src/data/splits.py",
    "src/data/types.py",
    "src/models/factory.py",
    "configs/config.yaml",
    "configs/task/nc.yaml",
    "configs/model/mag_collab_adapt.yaml",
    "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml",
    "configs/dataset/Reddit-S.yaml",
    "docs/mvcge_coordinated_nc_v2/model_spec.md",
    "docs/mvcge_coordinated_nc_v2/protocol.md",
    "docs/mvcge_coordinated_nc_v2/matrix.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_snapshot() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except Exception as exc:
        return [{"error": repr(exc)}]
    rows = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 6:
            rows.append({"index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory_used_mib": int(fields[3]), "memory_total_mib": int(fields[4]), "utilization_percent": int(fields[5])})
    return rows


def gpu_process_snapshot() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except Exception as exc:
        return [{"error": repr(exc)}]
    rows = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 4:
            try:
                rows.append({"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": fields[2], "memory_used_mib": int(fields[3])})
            except ValueError:
                rows.append({"raw": line})
    return rows


def validate_compute_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("formal and real-graph smoke runs require the authorized CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if SELECTED_GPU_UUID not in [value.strip() for value in visible.split(",") if value.strip()]:
        raise RuntimeError(f"pin CUDA_VISIBLE_DEVICES to the previously checked free card {SELECTED_GPU_UUID}")
    rows = gpu_snapshot()
    selected = next((row for row in rows if row.get("uuid") == SELECTED_GPU_UUID), None)
    if selected is None:
        raise RuntimeError("the authorized GPU UUID is not visible in nvidia-smi")
    if selected["memory_used_mib"] > 1024:
        raise RuntimeError(f"the selected GPU is no longer idle enough to begin this run: {selected}")


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_matrix() -> dict[str, Any]:
    calibration = []
    for dataset in DATASETS:
        for model in MODELS:
            for lr in LRS:
                for wd in WDS:
                    calibration.append({
                        "phase": "calibration", "dataset": dataset, "model": model,
                        "seed": CALIBRATION_SEED, "config_index": len([row for row in calibration if row["dataset"] == dataset and row["model"] == model]),
                        "lr": lr, "weight_decay": wd,
                    })
    confirmation = [{"phase": "confirmation", "dataset": d, "model": m, "seed": s, "selection_key": f"{d}/{m}"} for d in DATASETS for m in MODELS for s in CONFIRMATION_SEEDS]
    return {
        "study": STUDY, "created_utc": utc_now(), "datasets": list(DATASETS), "models": list(MODELS),
        "lr_grid": list(LRS), "weight_decay_grid": list(WDS), "calibration_seed": CALIBRATION_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS), "split_seed": SPLIT_SEED,
        "max_updates": MAX_UPDATES, "min_updates": MIN_UPDATES, "warmup_updates": [1, 20],
        "validation_first_update": EVAL_FIRST, "validation_interval": EVAL_INTERVAL,
        "early_stop_patience_updates": PATIENCE,
        "selection_rule": "validation correct node count descending, fixed-C macro-F1 descending, lower config_index first",
        "candidate_order": "dataset, model, LR outer ascending, weight decay inner ascending",
        "calibration_count": len(calibration), "confirmation_count": len(confirmation),
        "calibration_candidates": calibration, "confirmation_slots": confirmation,
    }


def _base_cfg(dataset: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "model=mvcge_coordinated_nc_v2", f"seed={SPLIT_SEED}"])


def _raw_graph_profile(path: Path) -> dict[str, Any]:
    import dgl
    graphs, _ = dgl.load_graphs(str(path))
    graph = graphs[0]
    src, dst = graph.edges()
    src, dst = src.long().cpu(), dst.long().cpu()
    keys = torch.unique(dst * int(graph.num_nodes()) + src)
    reverse = torch.unique(src * int(graph.num_nodes()) + dst)
    return {
        "raw_directed_edge_rows": int(src.numel()), "raw_self_loop_rows": int((src == dst).sum()),
        "raw_unique_directed_coordinates": int(keys.numel()),
        "reverse_edge_coverage": float(torch.isin(keys, reverse).float().mean()) if keys.numel() else 1.0,
        "num_nodes": int(graph.num_nodes()), "edge_weights_exposed": bool(graph.edata),
    }


def load_dataset(dataset: str, device: torch.device, *, profile: bool = False) -> dict[str, Any]:
    if dataset in DATA_CACHE and DATA_CACHE[dataset]["device"] == device:
        return DATA_CACHE[dataset]
    cfg = _base_cfg(dataset)
    mag_data = load_mag_data(cfg, "nc", SPLIT_SEED)
    train_idx = mag_data.train_idx.long().contiguous()
    val_idx = mag_data.val_idx.long().contiguous()
    test_size = int(mag_data.test_idx.numel())
    task_labels = labels_train_val_only(mag_data.y, train_idx, val_idx)
    edge_cpu = mag_data.edge_index.long().contiguous()
    src_cpu, dst_cpu = edge_cpu
    keep = src_cpu != dst_cpu
    directed_keys = torch.unique(dst_cpu[keep] * int(mag_data.num_nodes) + src_cpu[keep])
    reverse_keys = torch.unique(src_cpu[keep] * int(mag_data.num_nodes) + dst_cpu[keep])
    input_paths = {
        "graph": str(resolve_path(cfg.dataset.graph_path)),
        "text_features": str(resolve_path(cfg.dataset.text_feat_path)),
        "image_features": str(resolve_path(cfg.dataset.image_feat_path)),
        "nc_split": str(resolve_path(cfg.dataset.nc_split_path)),
    }
    values: dict[str, Any] = {
        "dataset": dataset, "device": device,
        "x": mag_data.x.float().contiguous().to(device),
        "edges": edge_cpu.to(device), "labels": task_labels.to(device),
        "train_idx": train_idx.to(device), "val_idx": val_idx.to(device),
        "n": int(mag_data.num_nodes), "m": int(edge_cpu.size(1)), "classes": int(mag_data.num_classes),
        "text_dim": int(mag_data.x_t.size(1)), "image_dim": int(mag_data.x_i.size(1)),
        "train_size": int(train_idx.numel()), "val_size": int(val_idx.numel()), "test_size": test_size,
        "paths": input_paths,
        "split_hashes": {"train_ids": tensor_sha256(train_idx), "validation_ids": tensor_sha256(val_idx)},
        "canonical_graph": {
            "loader_edge_rows": int(edge_cpu.size(1)), "self_loop_rows": int((src_cpu == dst_cpu).sum()),
            "duplicate_nonself_directed_rows": int(keep.sum() - directed_keys.numel()),
            "unique_nonself_directed_pairs": int(directed_keys.numel()),
            "reverse_edge_coverage": float(torch.isin(directed_keys, reverse_keys).float().mean()) if directed_keys.numel() else 1.0,
            "weights_exposed_by_MAGData": False,
        },
    }
    if profile:
        values["file_hashes"] = {key: sha256_file(Path(path)) for key, path in input_paths.items()}
        values["raw_graph"] = _raw_graph_profile(Path(input_paths["graph"]))
        values["feature_profile"] = {
            "text_finite": bool(torch.isfinite(mag_data.x_t).all()), "image_finite": bool(torch.isfinite(mag_data.x_i).all()),
            "text_exact_zero_rows": int((mag_data.x_t == 0).all(-1).sum()), "image_exact_zero_rows": int((mag_data.x_i == 0).all(-1).sum()),
        }
    # The on-disk loader materializes full labels and split IDs; neither test tensor is retained here.
    del mag_data.y, mag_data.test_idx, mag_data, task_labels
    DATA_CACHE[dataset] = values
    return values


def prepare() -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matrix = make_matrix()
    write_json(MATRIX_PATH, matrix)
    profile = {}
    for dataset in DATASETS:
        data = load_dataset(dataset, torch.device("cpu"), profile=True)
        profile[dataset] = {
            "num_nodes": data["n"], "loader_edge_rows": data["m"], "text_dim": data["text_dim"], "image_dim": data["image_dim"],
            "num_classes": data["classes"], "train_size": data["train_size"], "validation_size": data["val_size"], "test_size_metadata_only": data["test_size"],
            "train_validation_id_hashes": data["split_hashes"], "input_file_hashes": data["file_hashes"],
            "input_paths": data["paths"], "canonical_graph": data["canonical_graph"], "raw_graph": data["raw_graph"], "feature_profile": data["feature_profile"],
        }
    write_json(REPORT_ROOT / "data_profile.json", {
        "study": STUDY, "split_seed": SPLIT_SEED,
        "loader_note": "MAGData materializes stored graph labels and split objects. The runner copies only frozen train/validation labels into a sentinel-filled task tensor, retains train/validation IDs only, and drops test IDs/labels immediately; the disk loader itself necessarily reads the stored object.",
        "datasets": profile,
    })
    env = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda, "gpu_snapshot": gpu_snapshot(), "selected_gpu_uuid": SELECTED_GPU_UUID,
        "gpu_compute_processes": gpu_process_snapshot(),
        "cuda_visible_devices_at_intake": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {name: package_version(name) for name in ("torch-geometric", "dgl", "hydra-core", "numpy", "scikit-learn")},
        "git_head_at_intake": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "cwd": str(ROOT), "created_utc": utc_now(),
    }
    write_json(REPORT_ROOT / "environment.json", env)
    status = subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    preserved = [line for line in status.splitlines() if "CLAUDE.md" in line or "docs/V3_design.md" in line or "docs/mechanism_pilot_v2_design.md" in line or "docs/plan.md" in line or "reports/collab_adapt_nc_v2/s_calibration_selection.json" in line]
    intake = [
        f"# Intake: {STUDY}", "", f"- Starting HEAD: `{git('rev-parse', 'deb15363debd7260234964861446ec0d06938b11')}`.",
        f"- Working branch: `{git('branch', '--show-current')}`.",
        "- Existing source branch commit is the v1 experiment parent; no prior v1 outcomes are used in this v2 matrix.",
        "- Data source is the existing MAGB loader; text features precede image features. Features are float32, frozen, finite, and receive no extra normalization. Loader exposes no availability metadata or edge weights.",
        "- The runner retains only train/validation IDs and labels. The generic loader materializes full stored label/split objects; those are dropped before model/task execution. Full graph features and topology remain transductive inputs.",
        "- New R is constructed from the loader's undirected deduplicated edge list, drops loops/duplicates, uses receiver-row means, and keeps isolated rows zero.",
        "- No test evaluator, real LP task, or additional dataset is invoked; LP is tested only against a synthetic graph.",
        "- GPU intake snapshot and runtime versions are in `reports/mvcge_coordinated_nc_v2/environment.json`.",
        "- User-owned original workspace changes preserved and excluded from stage commits: " + (", ".join(f"`{line}`" for line in preserved) if preserved else "none observed"),
        "", "| Dataset | Nodes | Loader edge rows | Text/image dim | Classes | Train/val/test sizes* | Raw graph |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in profile.items():
        raw = item["raw_graph"]
        intake.append(f"| {name} | {item['num_nodes']:,} | {item['loader_edge_rows']:,} | {item['text_dim']}/{item['image_dim']} | {item['num_classes']} | {item['train_size']:,}/{item['validation_size']:,}/{item['test_size_metadata_only']:,} | {raw['raw_directed_edge_rows']:,} raw directed rows; reverse coverage {raw['reverse_edge_coverage']:.4f}; {raw['raw_self_loop_rows']} loops |")
    intake.extend(["", "*Test size is metadata only; test IDs and labels are not copied into the task runner. Input and split file hashes are recorded in `data_profile.json`."])
    (REPORT_ROOT / "intake.md").write_text("\n".join(intake) + "\n", encoding="utf-8")
    print(f"Prepared {len(matrix['calibration_candidates'])} calibration and {len(matrix['confirmation_slots'])} confirmation slots.")


def freeze_protocol() -> None:
    if not MATRIX_PATH.is_file():
        raise FileNotFoundError("run --prepare before --freeze")
    dirty = []
    for relative in SOURCE_FILES:
        status = subprocess.run(["git", "status", "--porcelain", "--", relative], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        if status:
            dirty.append(relative)
    if dirty:
        raise RuntimeError("commit the frozen training source package before creating protocol_lock.json: " + ", ".join(dirty))
    training_source_commit = git("rev-parse", "HEAD")
    hashes = {path: sha256_file(ROOT / path) for path in SOURCE_FILES if (ROOT / path).is_file()}
    data_profile = json.loads((REPORT_ROOT / "data_profile.json").read_text(encoding="utf-8"))
    lock = {
        "study": STUDY, "created_utc": utc_now(), "starting_head": "deb15363debd7260234964861446ec0d06938b11",
        "training_source_commit": training_source_commit,
        "training_source_tree": git("rev-parse", f"{training_source_commit}^{{tree}}"),
        "protocol_sha256": sha256_file(DOC_ROOT / "protocol.md"), "model_spec_sha256": sha256_file(DOC_ROOT / "model_spec.md"),
        "matrix_sha256": sha256_file(MATRIX_PATH), "source_hashes": hashes,
        "dataset_input_hashes": {name: item["input_file_hashes"] for name, item in data_profile["datasets"].items()},
        "train_validation_split_hashes": {name: item["train_validation_id_hashes"] for name, item in data_profile["datasets"].items()},
        "assertions": {"calibration_cells": 162, "confirmation_cells": 54, "total_cells": 216, "test_evaluations": 0, "real_lp_runs": 0, "dataset_count": 3},
    }
    baseline_lock_path = DOC_ROOT / "protocol_lock_v1.json"
    previous_lock_path = DOC_ROOT / "protocol_lock_v2.json" if (DOC_ROOT / "protocol_lock_v2.json").is_file() else baseline_lock_path
    if previous_lock_path.is_file():
        previous_lock = json.loads(previous_lock_path.read_text(encoding="utf-8"))
        lock["amendment"] = {
            "revision": int(previous_lock.get("amendment", {}).get("revision", 1)) + 1,
            "supersedes_protocol_lock": previous_lock_path.name,
            "supersedes_protocol_lock_sha256": sha256_file(previous_lock_path),
            "baseline_protocol_lock": baseline_lock_path.name,
            "baseline_protocol_lock_sha256": sha256_file(baseline_lock_path),
            "calibration_selection_lock_sha256": sha256_file(REPORT_ROOT / "selection_lock.json"),
            "calibration_audit_sha256": sha256_file(REPORT_ROOT / "calibration_audit.json"),
            "change_scope": "diagnostic gradient-vector aggregation and retry-attempt ID dispatch only; no model, forward, loss, optimizer, data, split, or calibration-selection changes",
            "initial_confirmation_attempts": {"successful_unaffected_A0_N0": 18, "failed_routed_diagnostic_aggregation": 36},
            "superseded_retry_dispatch": {"protocol_lock": "protocol_lock_v2.json", "training_attempts_started": 0, "reason": "base attempt IDs were parsed as retry suffixes; failure occurred before model construction"},
            "rerun_policy": "preserve the original failed attempts and retry the 36 routed confirmation slots under this source revision",
        }
    write_json(LOCK_PATH, lock)
    print(f"Protocol lock written: {LOCK_PATH}")


def ensure_registry() -> dict[str, Any]:
    if REGISTRY_PATH.is_file():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    attempts = {}
    for cell in make_matrix()["calibration_candidates"]:
        run_id = attempt_id("calibration", cell["dataset"], cell["model"], cell["seed"], cell["config_index"])
        attempts[run_id] = {**cell, "attempt_id": run_id, "status": "not_started", "attempt_number": 0}
    for cell in make_matrix()["confirmation_slots"]:
        run_id = attempt_id("confirmation", cell["dataset"], cell["model"], cell["seed"], None)
        attempts[run_id] = {**cell, "attempt_id": run_id, "status": "not_started", "attempt_number": 0}
    return {"study": STUDY, "created_utc": utc_now(), "attempts": attempts}


def save_registry(registry: dict[str, Any]) -> None:
    write_json(REGISTRY_PATH, registry)


def attempt_id(phase: str, dataset: str, model: str, seed: int, config_index: int | None) -> str:
    base = f"{phase}_{dataset.replace('-', '_')}_{model}_seed{seed}"
    return base if config_index is None else f"{base}_cfg{int(config_index):02d}"


def _select_attempt(registry: dict[str, Any], base: str) -> tuple[str, dict[str, Any] | None]:
    related = [(key, value) for key, value in registry["attempts"].items() if key == base or key.startswith(base + "_retry")]
    if not related:
        return base, None
    related.sort(key=lambda pair: (0 if pair[0] == base else int(pair[0].rsplit("retry", 1)[-1])))
    return related[-1]


def _next_retry_id(attempt_ids, base: str) -> str:
    prefix = base + "_retry"
    retry_numbers = []
    for key in attempt_ids:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix.isdecimal():
            retry_numbers.append(int(suffix))
    return f"{base}_retry{max(retry_numbers, default=0) + 1}"


def initialize_head(seed: int, classes: int, model: str, device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed) + 10_007)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 10_007)
    if model == "A0":
        head = CollabTaskHeads("B1", classes, output_dim=768)
    else:
        head = CoordinatedNCHeads(model, classes, input_dim=640, init_seed=int(seed) + 10_007)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return head.to(device)


def build_modules(data: dict[str, Any], model_name: str, seed: int, device: torch.device):
    seed_all(seed)
    info = {"input_dim": data["text_dim"] + data["image_dim"], "num_nodes": data["n"], "num_classes": data["classes"], "text_dim": data["text_dim"], "visual_dim": data["image_dim"]}
    if model_name == "A0":
        model = MAGCollabAdapt(info, "B1", dropout=0.2, init_seed=seed, typed_value=False, fusion=False, out_dim=768, collect_diagnostics=False, optimize_cache=True).to(device)
        head = initialize_head(seed, data["classes"], model_name, device)
        model.private_seed = int(seed)
    else:
        cfg = _base_cfg(data["dataset"])
        cfg.model.variant = model_name
        cfg.seed = int(seed)
        model = build_model(cfg, info).to(device)
        model.set_graph(data["edges"], data["n"])
        head = initialize_head(seed, data["classes"], model_name, device)
    # Initialization is name-derived; restart stochastic training streams after all constructors.
    seed_all(seed)
    return model, head


def optimizer_for(model: nn.Module, head: nn.Module, model_name: str, lr: float, wd: float):
    if model_name == "A0":
        groups = parameter_groups_no_decay(model, [head])
        for index, group in enumerate(groups):
            group.update({"lr": float(lr), "weight_decay": 0.0 if group.get("weight_decay") == 0.0 else float(wd), "name": "A0_no_decay" if group.get("weight_decay") == 0.0 else "A0_decay"})
    else:
        groups = model.parameter_groups(lr, wd)
        norm_ids = {id(p) for module in head.modules() if isinstance(module, nn.LayerNorm) for p in module.parameters(recurse=False)}
        decay, no_decay = [], []
        for name, param in head.named_parameters():
            if name.endswith("bias") or id(param) in norm_ids:
                no_decay.append(param)
            else:
                decay.append(param)
        if decay:
            groups.append({"params": decay, "lr": float(lr), "weight_decay": float(wd), "name": "task_head_decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0, "name": "task_head_no_decay"})
    flat = [id(parameter) for group in groups for parameter in group["params"]]
    expected = {id(parameter) for module in (model, head) for parameter in module.parameters() if parameter.requires_grad}
    if len(flat) != len(set(flat)) or set(flat) != expected:
        raise RuntimeError("optimizer groups do not cover every trainable parameter exactly once")
    return torch.optim.AdamW(groups, lr=float(lr), betas=(0.9, 0.999), eps=1e-8)


def update_lr(optimizer, target_lrs, step):
    factor = 0.1 + 0.9 * (int(step) - 1) / 19.0 if step <= 20 else 1.0
    for group, target in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target * factor


def validate_eval_ids(data: dict[str, Any], ids: torch.Tensor, split: str) -> torch.Tensor:
    if split not in {"train", "validation"}:
        raise ValueError("the experiment evaluator only permits train and validation")
    idx = ids.to(device=data["x"].device, dtype=torch.long).flatten()
    expected = data["train_idx"] if split == "train" else data["val_idx"]
    if idx.numel() != expected.numel() or not torch.equal(torch.sort(idx).values, torch.sort(expected).values):
        raise ValueError(f"IDs do not exactly match frozen {split} IDs")
    return idx


@torch.no_grad()
def eval_ids(model_name: str, model: nn.Module, head: nn.Module, data: dict[str, Any], ids: torch.Tensor, split: str, counters: dict[str, int], *, step: int) -> dict[str, Any]:
    idx = validate_eval_ids(data, ids, split)
    model.eval(); head.eval()
    if model_name == "A0":
        model._mvcge_step = int(step)
        model.set_epoch(int(step))
        z, _, _, _, extras = model(data["x"], data["edges"], torch.ones((data["n"], 2), dtype=torch.bool, device=data["x"].device), compute_ssl=False)
        subset = z.index_select(0, idx)
        task_extras = {"path_states": {key: value.index_select(0, idx) for key, value in extras["path_states"].items()}}
        logits, _ = head.forward_new(subset, task_extras)
        counters[split] = counters.get(split, 0) + 1
    else:
        z, _, _, _, _ = model(data["x"], data["edges"], update=step, modality_available=torch.ones((data["n"], 2), dtype=torch.bool, device=data["x"].device))
        logits = indexed_main_logits(head, z, idx, split=split, allowed={"train": data["train_idx"], "validation": data["val_idx"]}, counters=counters)
    targets = data["labels"].index_select(0, idx)
    pred = logits.argmax(-1)
    matrix = confusion_matrix(targets.detach().cpu().numpy(), pred.detach().cpu().numpy(), labels=np.arange(data["classes"]))
    f1 = f1_score(targets.detach().cpu().numpy(), pred.detach().cpu().numpy(), labels=np.arange(data["classes"]), average=None, zero_division=0)
    return {
        "node_ids": idx.detach().cpu().numpy().astype(np.int64), "targets": targets.detach().cpu().numpy().astype(np.int64),
        "argmax": pred.detach().cpu().numpy().astype(np.int64), "count": int(idx.numel()), "correct": int(pred.eq(targets).sum()),
        "accuracy": float(pred.eq(targets).float().mean()), "macro_f1": float(f1.mean()), "class_f1": f1.tolist(),
        "class_support": matrix.sum(1).tolist(), "confusion_matrix": matrix.tolist(),
    }


def _grad_norm(grads) -> torch.Tensor:
    present = [grad.detach().float().reshape(-1) for grad in grads if grad is not None]
    return torch.cat(present).norm() if present else torch.tensor(0.0)


def _grad_pair(left, right) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError(f"gradient tuples must align by parameter: {len(left)} != {len(right)}")
    aligned_left, aligned_right = [], []
    for grad_left, grad_right in zip(left, right):
        if grad_left is None and grad_right is None:
            continue
        reference = grad_left if grad_left is not None else grad_right
        assert reference is not None
        if grad_left is not None and grad_left.shape != reference.shape:
            raise ValueError("left gradient shape differs for the same parameter")
        if grad_right is not None and grad_right.shape != reference.shape:
            raise ValueError("right gradient shape differs for the same parameter")
        aligned_left.append((torch.zeros_like(reference) if grad_left is None else grad_left).detach().float().reshape(-1))
        aligned_right.append((torch.zeros_like(reference) if grad_right is None else grad_right).detach().float().reshape(-1))
    a = torch.cat(aligned_left) if aligned_left else torch.empty((0,))
    b = torch.cat(aligned_right) if aligned_right else torch.empty((0,))
    if a.numel() == 0 or b.numel() == 0:
        return {"left_norm": float(_grad_norm(left)), "right_norm": float(_grad_norm(right)), "norm_ratio": None, "cosine": None}
    an, bn = a.norm(), b.norm()
    cosine = None if an.item() == 0 or bn.item() == 0 else float(torch.dot(a, b) / (an * bn))
    return {"left_norm": float(an), "right_norm": float(bn), "norm_ratio": None if an.item() == 0 else float(bn / an), "cosine": cosine}


def gradient_diagnostics(model: MVCGECoordinatedNCV2, head: CoordinatedNCHeads, data: dict[str, Any], z: torch.Tensor, details: dict[str, Any], logits: torch.Tensor, labels: torch.Tensor, semantic_logits=None) -> dict[str, Any]:
    main_ce = F.cross_entropy(logits, labels)
    result: dict[str, Any] = {}
    expert_parameter_sets = [[p for expert in layer.experts for p in expert.parameters() if p.requires_grad] for layer in model.layers]
    direct_all = []
    message_exit = []
    for layer_idx, params in enumerate(expert_parameter_sets):
        direct = torch.autograd.grad(main_ce, params, retain_graph=True, allow_unused=True)
        direct_all.append(direct)
        layer_details = details["layers"][layer_idx]
        parts = []
        for modality in range(2):
            message = layer_details["mixed"][modality]
            value_exit = torch.autograd.grad(main_ce, message, retain_graph=True, allow_unused=False)[0]
            parts.append(torch.autograd.grad(message, params, grad_outputs=value_exit, retain_graph=True, allow_unused=True))
        combined = [None if a is None and b is None else (torch.zeros_like(a) if a is None else a) + (torch.zeros_like(b) if b is None else b) for a, b in zip(parts[0], parts[1])]
        errors = []
        for g1, g2, gd in zip(parts[0], parts[1], direct):
            reference = next((g for g in (g1, g2, gd) if g is not None), None)
            if reference is None:
                errors.append(torch.tensor(0.0, device=z.device))
                continue
            g1v = torch.zeros_like(reference) if g1 is None else g1
            g2v = torch.zeros_like(reference) if g2 is None else g2
            gdv = torch.zeros_like(reference) if gd is None else gd
            errors.append((g1v + g2v - gdv).detach().float().norm())
        direct_norm = float(_grad_norm(direct))
        error_norm = float(torch.stack(errors).norm()) if errors else 0.0
        message_exit.append({"text_norm": float(_grad_norm(parts[0])), "image_norm": float(_grad_norm(parts[1])), "sum_norm": float(_grad_norm(combined)), "direct_norm": direct_norm, "absolute_reconstruction_error": error_norm, "relative_error": error_norm / max(direct_norm, 1e-12)})
    result["shared_expert_message_exit_vjp"] = message_exit
    router_rows = []
    for layer in model.layers:
        if layer.router is None:
            continue
        params = list(layer.router.parameters())
        g_main = torch.autograd.grad(main_ce, params, retain_graph=True, allow_unused=True)
        g_balance = torch.autograd.grad(details["balance_loss"], params, retain_graph=True, allow_unused=True)
        router_rows.append(_grad_pair(g_main, g_balance))
    result["router_main_vs_weighted_balance"] = router_rows
    if semantic_logits is not None:
        aux_loss = 0.5 * (F.cross_entropy(semantic_logits[0], labels) + F.cross_entropy(semantic_logits[1], labels)) * 0.05
        params = list(model.projectors.parameters())
        g_main = torch.autograd.grad(main_ce, params, retain_graph=True, allow_unused=True)
        g_aux = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)
        result["frontend_main_vs_weighted_semantic"] = _grad_pair(g_main, g_aux)
    result["main_ce"] = float(main_ce.detach())
    return result


def _metric_record(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in ("count", "correct", "accuracy", "macro_f1", "class_f1", "class_support", "confusion_matrix")}


def write_prediction_file(path: Path, train: dict[str, Any], val: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            train_ids=train["node_ids"], train_targets=train["targets"], train_argmax=train["argmax"],
            validation_ids=val["node_ids"], validation_targets=val["targets"], validation_argmax=val["argmax"],
        )
        stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)
    return sha256_file(path)


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float()
    return {
        "mean": float(values.mean()), "std_population": float(values.std(unbiased=False)),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def _route_summary(routes: list[dict[str, torch.Tensor]], n_nodes: int) -> dict[str, Any]:
    rows = []
    combo_counts: dict[str, int] = {}
    all_slots = torch.zeros((4,), dtype=torch.long)
    all_selected = torch.zeros((4,), dtype=torch.long)
    all_weights, all_entropy, all_sigma = [], [], []
    for modality, route in enumerate(routes):
        gates = route["gates"].detach().float().cpu()
        selected = route["top2_indices"].detach().cpu()
        token_counts = torch.bincount(selected.reshape(-1), minlength=4)
        selected_once = torch.bincount(selected.reshape(-1), minlength=4)  # slot count; coverage below is per node
        coverage = torch.stack([(selected == expert).any(-1).sum() for expert in range(4)])
        all_slots += token_counts
        all_selected += coverage
        sorted_pairs = selected.sort(-1).values
        packed = sorted_pairs[:, 0] * 4 + sorted_pairs[:, 1]
        pair_hist = torch.bincount(packed, minlength=16)
        pairs = {f"{int(i // 4)}-{int(i % 4)}": int(pair_hist[i]) for i in torch.nonzero(pair_hist, as_tuple=False).flatten().tolist()}
        combo_counts[str(modality)] = pairs
        entropy = -(gates * gates.clamp_min(1e-12).log()).sum(-1)
        sigma = route["sigma"].detach().float().cpu() if "sigma" in route else torch.zeros_like(gates)
        row = {
            "modality_index": modality, "expert_token_coverage_count_denominator_N": [int(v) for v in coverage],
            "expert_token_coverage_fraction_denominator_N": [float(v / max(n_nodes, 1)) for v in coverage],
            "top2_slot_count_denominator_2N": [int(v) for v in token_counts],
            "top2_slot_share_denominator_2N": [float(v / max(2 * n_nodes, 1)) for v in token_counts],
            "unordered_top2_pair_counts_denominator_N": pairs,
            "mean_gate_weight_by_expert": gates.mean(0).tolist(), "mean_gate_entropy": float(entropy.mean()),
            "mean_sigma_by_expert": sigma.mean(0).tolist(),
        }
        rows.append(row)
        all_weights.append(gates); all_entropy.append(entropy); all_sigma.append(sigma)
    return {
        "per_modality": rows,
        "combined_tokens_2N": {
            "expert_token_coverage_count_denominator_2N": [int(v) for v in all_selected],
            "top2_slot_count_denominator_4N": [int(v) for v in all_slots],
            "mean_gate_weight_by_expert": torch.cat(all_weights, 0).mean(0).tolist(),
            "mean_gate_entropy": float(torch.cat(all_entropy, 0).mean()),
            "mean_sigma_by_expert": torch.cat(all_sigma, 0).mean(0).tolist(),
        },
        "pair_counts_per_modality": combo_counts,
    }


@torch.no_grad()
def _model_diagnostics_impl(model_name: str, model: nn.Module, head: nn.Module, data: dict[str, Any], seed: int, best_step: int) -> dict[str, Any]:
    model.eval(); head.eval()
    started = time.perf_counter()
    if model_name == "A0":
        model.set_epoch(best_step); model._mvcge_step = best_step
        z, _, _, _, extras = model(data["x"], data["edges"], torch.ones((data["n"], 2), dtype=torch.bool, device=data["x"].device), compute_ssl=False, collect_diagnostics=True)
        return {"historical_encoder_diagnostics": json_safe(extras.get("diagnostics", {})), "diagnostic_seconds": time.perf_counter() - started}
    z, _, _, _, details = model(data["x"], data["edges"], update=best_step, modality_available=torch.ones((data["n"], 2), dtype=torch.bool, device=data["x"].device))
    clean_route = []
    for layer in details["layers"]:
        clean_route.append(layer["routes"])
    clean_stats = [_route_summary(routes, data["n"]) for routes in clean_route] if model.routed else []
    noise_generator = torch.Generator(device=data["x"].device).manual_seed(seed + 1_712_141)
    _, noisy_details = model.encode(data["x"], data["edges"], update=best_step, noise_generator=noise_generator, modality_available=torch.ones((data["n"], 2), dtype=torch.bool, device=data["x"].device), return_details=True, diagnostic_noise=model.routed)
    noisy_stats = [_route_summary(layer["routes"], data["n"]) for layer in noisy_details["layers"]] if model.routed else []
    feature_stats: dict[str, Any] = {
        "p_text_norm": _summary(details["p"][0].norm(dim=-1)), "p_image_norm": _summary(details["p"][1].norm(dim=-1)),
        "h0_text_norm": _summary(details["h0"][0].norm(dim=-1)), "h0_image_norm": _summary(details["h0"][1].norm(dim=-1)),
        "c0_norm": _summary(details["c0"].norm(dim=-1)), "g_text_norm": _summary(details["g"][0].norm(dim=-1)), "g_image_norm": _summary(details["g"][1].norm(dim=-1)),
        "delta_text_norm": _summary(details["delta"][0].norm(dim=-1)), "delta_image_norm": _summary(details["delta"][1].norm(dim=-1)),
        "delta_text_channel_variance_mean": float(details["delta"][0].var(dim=-1, unbiased=False).mean()),
        "delta_image_channel_variance_mean": float(details["delta"][1].var(dim=-1, unbiased=False).mean()),
        "delta_text_layernorm_inverse_std": _summary((details["delta"][0].var(dim=-1, unbiased=False) + 1e-5).rsqrt()),
        "delta_image_layernorm_inverse_std": _summary((details["delta"][1].var(dim=-1, unbiased=False) + 1e-5).rsqrt()),
        "per_layer": [],
    }
    for layer in details["layers"]:
        feature_stats["per_layer"].append({
            "text_u_norm": _summary(layer["u"][0].norm(dim=-1)), "image_u_norm": _summary(layer["u"][1].norm(dim=-1)),
            "text_neighbor_norm": _summary(layer["neighbor"][0].norm(dim=-1)), "image_neighbor_norm": _summary(layer["neighbor"][1].norm(dim=-1)),
            "text_mixed_graph_update_norm": _summary(layer["mixed"][0].norm(dim=-1)), "image_mixed_graph_update_norm": _summary(layer["mixed"][1].norm(dim=-1)),
            "text_adapter_residual_norm": _summary(layer["adapter_residual"][0].norm(dim=-1)), "image_adapter_residual_norm": _summary(layer["adapter_residual"][1].norm(dim=-1)),
        })
    result = {"branch_norms_and_variance": feature_stats, "clean_routing": clean_stats, "noisy_routing": noisy_stats, "diagnostic_seconds": time.perf_counter() - started}
    if model.pc:
        xgrid = torch.linspace(-1.0, 1.0, 101, device=data["x"].device)
        poly = []
        for li, layer in enumerate(model.layers):
            experts = []
            for ei, expert in enumerate(layer.experts):
                t0, t1 = expert.theta0.detach(), expert.theta1.detach()
                alpha1, alpha2 = t0 + 0.5 * t1, 0.5 * t1
                response = alpha1[:, None] * xgrid + alpha2[:, None] * xgrid.square()
                experts.append({"expert": ei, "theta0": t0.cpu().tolist(), "theta1": t1.cpu().tolist(), "alpha1": alpha1.cpu().tolist(), "alpha2": alpha2.cpu().tolist(), "response_mean_over_channels": response.mean(0).cpu().tolist(), "response_min_over_channels": response.min(0).values.cpu().tolist(), "response_max_over_channels": response.max(0).values.cpu().tolist()})
            poly.append({"layer": li, "experts": experts})
        result["pc_coefficients_and_response"] = {"x_grid": xgrid.cpu().tolist(), "layers": poly, "response_scope": "neighbor polynomial only; not a full operator norm guarantee"}
    return result


def model_diagnostics(model_name: str, model: nn.Module, head: nn.Module, data: dict[str, Any], seed: int, best_step: int) -> dict[str, Any]:
    model_training, head_training = model.training, head.training
    try:
        return _model_diagnostics_impl(model_name, model, head, data, seed, best_step)
    finally:
        model.train(model_training)
        head.train(head_training)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _restore_rng(state: dict[str, Any], noise_generator: torch.Generator, pair_generator: torch.Generator, device: torch.device) -> None:
    torch.random.set_rng_state(state["torch_rng_cpu"])
    if device.type == "cuda" and state.get("torch_rng_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_rng_cuda"])
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    noise_generator.set_state(state["noise_rng"]); pair_generator.set_state(state["pair_rng"])


def _save_resume(path: Path, model, head, optimizer, state: dict[str, Any], noise_generator, pair_generator, device):
    payload = {
        "model": model.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(),
        "noise_rng": noise_generator.get_state(), "pair_rng": pair_generator.get_state(),
        "torch_rng_cpu": torch.random.get_rng_state(), "torch_rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), **state,
    }
    _atomic_torch_save(path, payload)


def _best_key(metrics: dict[str, Any], step: int) -> tuple[int, float, int]:
    return int(metrics["correct"]), float(metrics["macro_f1"]), -int(step)


def _train_forward(model_name, model, head, data, step, noise_generator, pair_generator):
    if model_name == "A0":
        model._mvcge_step = step; model.set_epoch(step)
        z, _, _, _, extras = model(data["x"], data["edges"], torch.ones((data["n"], 2), dtype=torch.bool, device=data["x"].device), pair_generator=pair_generator, compute_ssl=True)
        total, terms, _ = train_only_loss("B1", z, extras, head, data["labels"], data["train_idx"], source_weight=0.1, path_weight=0.05, rec_weight=0.01, pair_weight=0.01)
        return total, terms["main_ce"], total - terms["main_ce"], z, extras, terms
    z, _, _, balance, details = model(data["x"], data["edges"], update=step, noise_generator=noise_generator, modality_available=torch.ones((data["n"], 2), dtype=torch.bool, device=data["x"].device))
    ids = data["train_idx"]
    labels = data["labels"].index_select(0, ids)
    logits = head.main_logits(z.index_select(0, ids))
    aux_logits = head.semantic_logits(details["p"], ids) if model.variant in {"M1", "P1"} else None
    total, terms = coordinated_loss(model_name, logits, labels, balance_loss=balance, semantic_logits=aux_logits)
    return total, terms["main_ce"], balance, z, details, (logits, labels, aux_logits, terms)


def _attempt_source_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES if (ROOT / relative).is_file()}


def run_cell(phase: str, data: dict[str, Any], model_name: str, seed: int, config_index: int | None, lr: float, wd: float, device: torch.device, *, retry_failed: bool = False) -> dict[str, Any]:
    registry = ensure_registry()
    base_id = attempt_id(phase, data["dataset"], model_name, seed, config_index)
    run_id, previous = _select_attempt(registry, base_id)
    if previous and previous.get("status") == "success":
        return previous
    if previous and previous.get("status") == "failed" and not retry_failed:
        return previous
    if previous and previous.get("status") == "failed" and retry_failed:
        run_id = _next_retry_id(registry["attempts"], base_id)
        previous = None
    output_dir = OUTPUT_ROOT / phase / data["dataset"] / model_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = output_dir / "latest.pt", output_dir / "best.pt"
    step_path, prediction_path = output_dir / "steps.jsonl", output_dir / "train_validation_argmax.npz"
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    source_hashes = _attempt_source_hashes()
    source_commit = lock["training_source_commit"]
    current_head = git("rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, current_head], cwd=ROOT).returncode != 0:
        raise RuntimeError("current HEAD does not contain the frozen training source commit")
    gpu_start_snapshot = gpu_snapshot() if device.type == "cuda" else []
    gpu_processes_start = gpu_process_snapshot() if device.type == "cuda" else []
    if source_hashes != lock["source_hashes"]:
        raise RuntimeError("frozen source hash mismatch; formal run refused")
    input_hashes = json.loads((REPORT_ROOT / "data_profile.json").read_text(encoding="utf-8"))["datasets"][data["dataset"]]["input_file_hashes"]
    if phase == "confirmation":
        if not SELECTION_PATH.is_file():
            raise RuntimeError("confirmation cannot start before selection_lock.json exists")
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        selected = selection["selections"][f"{data['dataset']}/{model_name}"]
        if (float(lr), float(wd)) != (float(selected["lr"]), float(selected["weight_decay"])):
            raise RuntimeError("confirmation configuration differs from immutable selection lock")
    model, head = build_modules(data, model_name, seed, device)
    optimizer = optimizer_for(model, head, model_name, lr, wd)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 31_415)
    pair_generator = torch.Generator(device=device).manual_seed(seed + 8_675_309)
    target_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    all_trainable = [p for module in (model, head) for p in module.parameters() if p.requires_grad]
    if len({id(p) for p in all_trainable}) != len(all_trainable):
        raise RuntimeError("model/task modules share trainable parameter objects")
    counters = {"train": 0, "validation": 0}
    start_step, best_key, best_step, best_metrics, best_train_loss, best_train_terms = 1, None, 0, None, None, None
    prior_diagnostics: list[dict[str, Any]] = []
    step_times_ms: list[float] = []
    prior_elapsed_seconds, validation_seconds = 0.0, 0.0
    prior_peak_allocated, prior_peak_reserved = 0, 0
    resume_completed = False
    gradient_diag_seconds = 0.0
    skipped_update_200_diagnostic = False
    run_started = time.perf_counter()
    if previous and previous.get("status") in {"running", "interrupted"} and latest_path.is_file():
        resume = torch.load(latest_path, map_location=device, weights_only=False)
        expected_provenance = {
            "source_commit": source_commit, "source_hashes": source_hashes,
            "protocol_lock_sha256": sha256_file(LOCK_PATH), "input_hashes": input_hashes,
            "train_ids_sha256": data["split_hashes"]["train_ids"], "validation_ids_sha256": data["split_hashes"]["validation_ids"],
        }
        for key, value in expected_provenance.items():
            if resume.get(key) != value:
                raise RuntimeError(f"resume checkpoint provenance mismatch: {key}")
        model.load_state_dict(resume["model"]); head.load_state_dict(resume["head"]); optimizer.load_state_dict(resume["optimizer"])
        _restore_rng(resume, noise_generator, pair_generator, device)
        start_step = int(resume["step"]) + 1
        best_key, best_step, best_metrics = resume["best_key"], int(resume["best_step"]), resume["best_metrics"]
        best_train_loss, best_train_terms = resume.get("best_train_loss"), resume.get("best_train_terms")
        prior_diagnostics = resume.get("gradient_diagnostics", [])
        gradient_diag_seconds = float(resume.get("gradient_diagnostic_seconds", 0.0))
        prior_elapsed_seconds = float(resume.get("elapsed_seconds_so_far", 0.0))
        validation_seconds = float(resume.get("validation_seconds_so_far", 0.0))
        counters["validation"] = int(resume.get("validation_calls", 0))
        prior_peak_allocated = int(resume.get("peak_allocated_bytes_so_far", 0))
        prior_peak_reserved = int(resume.get("peak_reserved_bytes_so_far", 0))
        resume_completed = bool(resume.get("training_complete", False))
        if resume_completed:
            start_step = int(resume["step"])
        if step_path.is_file():
            kept = [line for line in step_path.read_text(encoding="utf-8").splitlines() if line.strip() and int(json.loads(line)["update"]) <= int(resume["step"])]
            step_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            step_times_ms = [float(json.loads(line)["optimizer_update_ms"]) for line in kept]
    else:
        (output_dir / "resolved_config.json").write_text(json.dumps({"phase": phase, "dataset": data["dataset"], "model": model_name, "seed": seed, "config_index": config_index, "lr": lr, "weight_decay": wd, "split_seed": SPLIT_SEED, "max_updates": MAX_UPDATES, "source_commit": source_commit, "protocol_lock_sha256": sha256_file(LOCK_PATH)}, indent=2) + "\n", encoding="utf-8")
        registry["attempts"][run_id] = {
            "attempt_id": run_id, "phase": phase, "dataset": data["dataset"], "model": model_name, "training_seed": seed,
            "split_seed": SPLIT_SEED, "config_index": config_index, "lr": lr, "weight_decay": wd,
            "status": "running", "attempt_number": 0 if "retry" not in run_id else int(run_id.rsplit("retry", 1)[-1]),
            "started_utc": utc_now(), "source_commit": source_commit, "source_hashes": source_hashes,
            "protocol_lock_sha256": sha256_file(LOCK_PATH), "train_ids_sha256": data["split_hashes"]["train_ids"],
            "validation_ids_sha256": data["split_hashes"]["validation_ids"], "test_task_head_calls": 0,
            "test_metrics_computed": False, "real_lp_runs": 0, "output_dir": str(output_dir),
            "selected_gpu_uuid": SELECTED_GPU_UUID if device.type == "cuda" else None,
            "runner_pid": os.getpid(),
            "gpu_snapshot_start": gpu_start_snapshot, "gpu_compute_processes_start": gpu_processes_start,
        }
        save_registry(registry)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stop_reason = str(resume.get("stop_reason", "max_updates_reached")) if resume_completed else "max_updates_reached"
    try:
        for step in range(start_step, MAX_UPDATES + 1):
            if resume_completed:
                step = int(resume["step"])
                break
            model.train(); head.train()
            update_lr(optimizer, target_lrs, step)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
            else:
                wall_start = time.perf_counter()
            total_loss, main_ce, weighted_balance, z, details, train_parts = _train_forward(model_name, model, head, data, step, noise_generator, pair_generator)
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"nonfinite total loss at update {step}")
            grad_diag = None
            if phase == "confirmation" and model_name in NEW_MODELS and step in GRAD_DIAGNOSTIC_STEPS:
                stable_seconds_before_200 = sum(step_times_ms) / 1000.0 - gradient_diag_seconds
                ratio_before_200 = gradient_diag_seconds / max(stable_seconds_before_200, 1e-12)
                if step == 200 and ratio_before_200 >= 0.10:
                    skipped_update_200_diagnostic = True
                else:
                    grad_diag_started = time.perf_counter()
                    logits, labels, aux_logits, _ = train_parts
                    grad_diag = gradient_diagnostics(model, head, data, z, details, logits, labels, aux_logits)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    gradient_diag_seconds += time.perf_counter() - grad_diag_started
                    grad_diag.update({"update": step, "mode": "train_dropout_noisy_routing"})
                    prior_diagnostics.append(grad_diag)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, 5.0, error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                event_end.record(); torch.cuda.synchronize(device)
                step_ms = float(event_start.elapsed_time(event_end))
            else:
                step_ms = 1000.0 * (time.perf_counter() - wall_start)
            step_times_ms.append(step_ms)
            row: dict[str, Any] = {
                "update": step, "main_ce": float(main_ce.detach()),
                "weighted_balance": None if model_name == "A0" else float(weighted_balance.detach()),
                "a0_non_main_loss": float(weighted_balance.detach()) if model_name == "A0" else None,
                "total_loss": float(total_loss.detach()), "pre_clip_global_grad_norm": float(grad_norm.detach()),
                "lr_by_group": {str(group.get("name", i)): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)},
                "optimizer_update_ms": step_ms, "diagnostic": grad_diag,
            }
            if model_name == "A0" and train_parts is not None:
                row["a0_loss_terms"] = {name: float(value.detach()) for name, value in train_parts.items()}
            should_eval = step >= EVAL_FIRST and ((step - EVAL_FIRST) % EVAL_INTERVAL == 0 or step == MAX_UPDATES)
            improved = False
            if should_eval:
                validation_started = time.perf_counter()
                val = eval_ids(model_name, model, head, data, data["val_idx"], "validation", counters, step=step)
                validation_seconds += time.perf_counter() - validation_started
                current_key = _best_key(val, step)
                improved = best_key is None or current_key > tuple(best_key)
                row["validation"] = {key: val[key] for key in ("count", "correct", "accuracy", "macro_f1")}
                if improved:
                    best_key, best_step, best_metrics = current_key, step, _metric_record(val)
                    best_train_loss = float(main_ce.detach())
                    best_train_terms = {"total": float(total_loss.detach()), "main_ce": float(main_ce.detach()), "weighted_balance": float(weighted_balance.detach())}
                    if model_name == "A0" and train_parts is not None:
                        best_train_terms = {"total": float(total_loss.detach()), **{name: float(value.detach()) for name, value in train_parts.items()}}
                    _atomic_torch_save(best_path, {
                        "model": model.state_dict(), "head": head.state_dict(), "best_step": step, "best_key": best_key,
                        "validation": best_metrics, "source_commit": source_commit, "source_hashes": source_hashes,
                        "protocol_lock_sha256": sha256_file(LOCK_PATH), "input_hashes": input_hashes,
                    })
            append_jsonl(step_path, row)
            if step % 10 == 0 or improved or step == MAX_UPDATES:
                _save_resume(latest_path, model, head, optimizer, {
                    "step": step, "best_key": best_key, "best_step": best_step, "best_metrics": best_metrics,
                    "best_train_loss": best_train_loss, "best_train_terms": best_train_terms,
                    "gradient_diagnostics": prior_diagnostics, "validation_calls": counters["validation"],
                    "gradient_diagnostic_seconds": gradient_diag_seconds,
                    "elapsed_seconds_so_far": prior_elapsed_seconds + time.perf_counter() - run_started,
                    "validation_seconds_so_far": validation_seconds,
                    "peak_allocated_bytes_so_far": max(prior_peak_allocated, int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
                    "peak_reserved_bytes_so_far": max(prior_peak_reserved, int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0),
                    "source_commit": source_commit, "source_hashes": source_hashes,
                    "protocol_lock_sha256": sha256_file(LOCK_PATH), "input_hashes": input_hashes,
                    "train_ids_sha256": data["split_hashes"]["train_ids"], "validation_ids_sha256": data["split_hashes"]["validation_ids"],
                    "phase": phase, "dataset": data["dataset"], "model_name": model_name, "training_seed": seed,
                    "config_index": config_index, "lr": lr, "weight_decay": wd,
                }, noise_generator, pair_generator, device)
            if step >= MIN_UPDATES and best_step > 0 and step - best_step >= PATIENCE:
                stop_reason = "early_stop_patience_150_optimizer_updates"
                break
        else:
            step = MAX_UPDATES
        if best_key is None or not best_path.is_file():
            raise RuntimeError("cell ended without an eligible validation checkpoint")
        _save_resume(latest_path, model, head, optimizer, {
            "step": step, "best_key": best_key, "best_step": best_step, "best_metrics": best_metrics,
            "best_train_loss": best_train_loss, "best_train_terms": best_train_terms,
            "gradient_diagnostics": prior_diagnostics, "validation_calls": counters["validation"],
            "gradient_diagnostic_seconds": gradient_diag_seconds,
            "elapsed_seconds_so_far": prior_elapsed_seconds + time.perf_counter() - run_started,
            "validation_seconds_so_far": validation_seconds,
            "peak_allocated_bytes_so_far": max(prior_peak_allocated, int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
            "peak_reserved_bytes_so_far": max(prior_peak_reserved, int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0),
            "source_commit": source_commit, "source_hashes": source_hashes,
            "protocol_lock_sha256": sha256_file(LOCK_PATH), "input_hashes": input_hashes,
            "train_ids_sha256": data["split_hashes"]["train_ids"], "validation_ids_sha256": data["split_hashes"]["validation_ids"],
            "phase": phase, "dataset": data["dataset"], "model_name": model_name, "training_seed": seed,
            "config_index": config_index, "lr": lr, "weight_decay": wd,
            "training_complete": True, "stop_reason": stop_reason,
        }, noise_generator, pair_generator, device)
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model"]); head.load_state_dict(best_payload["head"])
        validation_started = time.perf_counter()
        train_metrics = eval_ids(model_name, model, head, data, data["train_idx"], "train", counters, step=best_step)
        val_metrics = eval_ids(model_name, model, head, data, data["val_idx"], "validation", counters, step=best_step)
        validation_seconds += time.perf_counter() - validation_started
        prediction_hash = write_prediction_file(prediction_path, train_metrics, val_metrics)
        diagnostics_started = time.perf_counter()
        diag = model_diagnostics(model_name, model, head, data, seed, best_step)
        if phase == "confirmation" and model_name in NEW_MODELS:
            model.eval(); head.eval()
            z_eval, _, _, _, detail_eval = model(data["x"], data["edges"], update=best_step, modality_available=torch.ones((data["n"], 2), dtype=torch.bool, device=device))
            train_ids = data["train_idx"]
            train_labels = data["labels"].index_select(0, train_ids)
            logits = head.main_logits(z_eval.index_select(0, train_ids))
            semantic = head.semantic_logits(detail_eval["p"], train_ids) if model_name in {"M1", "P1"} else None
            with torch.enable_grad():
                # The clean best checkpoint gradient record is a diagnostic forward, not an optimizer update.
                z_eval.requires_grad_(True)
                diag_best = gradient_diagnostics(model, head, data, z_eval, detail_eval, logits, train_labels, semantic)
            diag_best.update({"update": best_step, "mode": "best_eval_clean_top2"})
            prior_diagnostics.append(diag_best)
        best_diagnostic_seconds = time.perf_counter() - diagnostics_started
        diag["best_checkpoint_and_route_diagnostic_seconds"] = best_diagnostic_seconds
        diag["gradient_diagnostic_seconds"] = gradient_diag_seconds
        diag["diagnostic_seconds"] = best_diagnostic_seconds + gradient_diag_seconds
        non_diagnostic_update_seconds = max(sum(step_times_ms) / 1000.0 - gradient_diag_seconds, 0.0)
        diagnostic_fraction = diag["diagnostic_seconds"] / max(non_diagnostic_update_seconds + diag["diagnostic_seconds"], 1e-12)
        trimmed_updates: list[int] = []
        if diagnostic_fraction > 0.10:
            retained = []
            for diagnostic in prior_diagnostics:
                if diagnostic.get("mode") == "best_eval_clean_top2" or diagnostic.get("update") == 100:
                    retained.append(diagnostic)
                elif diagnostic.get("update") is not None:
                    trimmed_updates.append(int(diagnostic["update"]))
            prior_diagnostics = retained
            if step_path.is_file():
                filtered_rows = []
                for line in step_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if value.get("update") in {1, 50, 200}:
                        value["diagnostic"] = None
                    filtered_rows.append(canonical_json(value))
                step_path.write_text("\n".join(filtered_rows) + "\n", encoding="utf-8")
        diag["gradient_diagnostics"] = prior_diagnostics
        diag["diagnostic_budget"] = {
            "threshold_fraction": 0.10, "observed_fraction_of_update_plus_diagnostic_time": diagnostic_fraction,
            "skipped_update_200_diagnostic_before_running": skipped_update_200_diagnostic,
            "trimmed_from_saved_step_and_diagnostic_records": sorted(set(trimmed_updates)),
            "retained_core_points": [row.get("update") for row in prior_diagnostics],
        }
        write_json(output_dir / "diagnostics.json", diag)
        if step >= MAX_UPDATES and best_step >= MAX_UPDATES - EVAL_INTERVAL:
            stop_reason = "max_updates_validation_still_improving_budget_may_be_insufficient"
        elif step >= MAX_UPDATES and stop_reason != "early_stop_patience_150_optimizer_updates":
            stop_reason = "max_updates_reached"
        elapsed = prior_elapsed_seconds + time.perf_counter() - run_started
        current_peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        current_peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        non_diagnostic_steps = [ms for update, ms in enumerate(step_times_ms, 1) if not (phase == "confirmation" and model_name in NEW_MODELS and update in GRAD_DIAGNOSTIC_STEPS)]
        stable_steps = [ms for update, ms in enumerate(step_times_ms, 1) if update >= 110 and not (phase == "confirmation" and model_name in NEW_MODELS and update in GRAD_DIAGNOSTIC_STEPS)]
        result = {
            "attempt_id": run_id, "phase": phase, "dataset": data["dataset"], "model": model_name,
            "training_seed": seed, "split_seed": SPLIT_SEED, "config_index": config_index, "lr": lr, "weight_decay": wd,
            "status": "success", "source_commit": source_commit, "source_hashes": source_hashes,
            "protocol_lock_sha256": sha256_file(LOCK_PATH), "train_ids_sha256": data["split_hashes"]["train_ids"],
            "validation_ids_sha256": data["split_hashes"]["validation_ids"], "best_step": best_step, "updates": int(step),
            "stop_reason": stop_reason, "validation_best_key": list(best_key), "best_step_training_loss": best_train_terms,
            "train": _metric_record(train_metrics), "validation": _metric_record(val_metrics),
            "active_encoder_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "active_task_head_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
            "optimizer_groups": [{"name": group["name"], "lr_target": target_lrs[i], "weight_decay": group["weight_decay"], "parameter_count": sum(p.numel() for p in group["params"])} for i, group in enumerate(optimizer.param_groups)],
            "median_step_ms": float(statistics.median(step_times_ms)),
            "median_non_diagnostic_step_ms": float(statistics.median(non_diagnostic_steps)),
            "median_stable_non_diagnostic_step_ms_update_ge_110": float(statistics.median(stable_steps)),
            "optimizer_update_seconds": sum(step_times_ms) / 1000.0,
            "non_diagnostic_optimizer_update_seconds": sum(non_diagnostic_steps) / 1000.0,
            "validation_wall_seconds": validation_seconds,
            "total_cell_wall_seconds": elapsed,
            "peak_allocated_bytes": max(prior_peak_allocated, current_peak_allocated) if device.type == "cuda" else None,
            "peak_reserved_bytes": max(prior_peak_reserved, current_peak_reserved) if device.type == "cuda" else None,
            "train_evaluator_calls": counters["train"], "validation_evaluator_calls": counters["validation"],
            "test_task_head_calls": 0, "test_metrics_computed": False, "real_lp_runs": 0,
            "selected_gpu_uuid": SELECTED_GPU_UUID if device.type == "cuda" else None,
            "runner_pid": os.getpid(),
            "gpu_snapshot_start": gpu_start_snapshot,
            "gpu_snapshot_end": gpu_snapshot() if device.type == "cuda" else [],
            "gpu_compute_processes_start": gpu_processes_start,
            "gpu_compute_processes_end": gpu_process_snapshot() if device.type == "cuda" else [],
            "prediction_file": str(prediction_path), "prediction_sha256": prediction_hash,
            "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint": str(latest_path), "last_checkpoint_sha256": sha256_file(latest_path),
            "step_log": str(step_path), "step_log_sha256": sha256_file(step_path),
            "diagnostics_file": str(output_dir / "diagnostics.json"), "diagnostics_sha256": sha256_file(output_dir / "diagnostics.json"),
            "resolved_config": str(output_dir / "resolved_config.json"), "resolved_config_sha256": sha256_file(output_dir / "resolved_config.json"),
            "parameter_counts_total": sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
            "diagnostic_seconds": float(diag.get("diagnostic_seconds", 0.0)),
            "diagnostic_fraction_of_update_and_diagnostic_time": diagnostic_fraction,
            "diagnostic_budget": diag["diagnostic_budget"],
        }
        registry = ensure_registry()
        entry = dict(registry["attempts"].get(run_id, {})); entry.update(result); entry["ended_utc"] = utc_now()
        registry["attempts"][run_id] = entry; save_registry(registry); append_jsonl(RESULTS_PATH, result)
        return entry
    except Exception as exc:
        failed = {
            "attempt_id": run_id, "phase": phase, "dataset": data["dataset"], "model": model_name, "training_seed": seed,
            "config_index": config_index, "lr": lr, "weight_decay": wd, "status": "failed", "failed_utc": utc_now(),
            "error": repr(exc), "traceback": traceback.format_exc(limit=30), "last_completed_update": int(locals().get("step", 0)),
            "source_commit": source_commit, "source_hashes": source_hashes, "test_task_head_calls": 0, "real_lp_runs": 0,
            "output_dir": str(output_dir), "selected_gpu_uuid": SELECTED_GPU_UUID if device.type == "cuda" else None,
            "runner_pid": os.getpid(),
            "gpu_snapshot_start": gpu_start_snapshot, "gpu_snapshot_failure": gpu_snapshot() if device.type == "cuda" else [],
            "gpu_compute_processes_start": gpu_processes_start,
            "gpu_compute_processes_failure": gpu_process_snapshot() if device.type == "cuda" else [],
        }
        registry = ensure_registry(); registry["attempts"][run_id] = failed; save_registry(registry); append_jsonl(RESULTS_PATH, failed)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return failed
    finally:
        del model, head, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_phase(phase: str, device: torch.device, *, retry_failed: bool = False) -> None:
    validate_compute_device(device)
    if not LOCK_PATH.is_file():
        raise RuntimeError("protocol lock missing; run --freeze, commit frozen sources, then train")
    if git("rev-parse", "HEAD") == "deb15363debd7260234964861446ec0d06938b11":
        raise RuntimeError("frozen training source commit is not present")
    source_commit = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["training_source_commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, git("rev-parse", "HEAD")], cwd=ROOT).returncode != 0:
        raise RuntimeError("frozen training source commit is not in current branch history")
    registry = ensure_registry(); save_registry(registry)
    if phase == "calibration":
        cells = make_matrix()["calibration_candidates"]
    elif phase == "confirmation":
        if not SELECTION_PATH.is_file():
            raise RuntimeError("run the independent calibration audit and create selection_lock.json first")
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selections"]
        cells = [{"phase": phase, "dataset": d, "model": m, "seed": seed, "config_index": selection[f"{d}/{m}"]["config_index"], "lr": selection[f"{d}/{m}"]["lr"], "weight_decay": selection[f"{d}/{m}"]["weight_decay"]} for d in DATASETS for m in MODELS for seed in CONFIRMATION_SEEDS]
    else:
        raise ValueError(phase)
    locked_data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["dataset_input_hashes"]
    verified_datasets: set[str] = set()
    for index, cell in enumerate(cells, 1):
        data = load_dataset(cell["dataset"], device)
        if cell["dataset"] not in verified_datasets:
            actual_hashes = {key: sha256_file(Path(path)) for key, path in data["paths"].items()}
            if actual_hashes != locked_data[cell["dataset"]]:
                raise RuntimeError(f"input file hashes differ from the frozen protocol for {cell['dataset']}")
            if data["split_hashes"] != json.loads((REPORT_ROOT / "data_profile.json").read_text(encoding="utf-8"))["datasets"][cell["dataset"]]["train_validation_id_hashes"]:
                raise RuntimeError(f"train/validation ID hashes differ from the frozen profile for {cell['dataset']}")
            verified_datasets.add(cell["dataset"])
        print(f"[{phase} {index}/{len(cells)}] {cell['dataset']} {cell['model']} seed={cell['seed']} config={cell.get('config_index')} lr={cell['lr']} wd={cell['weight_decay']}", flush=True)
        result = run_cell(phase, data, cell["model"], cell["seed"], cell.get("config_index"), cell["lr"], cell["weight_decay"], device, retry_failed=retry_failed)
        print(f"  {result['status']} updates={result.get('updates', result.get('last_completed_update'))} best={result.get('best_step')} error={result.get('error')}", flush=True)
    DATA_CACHE.clear()
    if "data" in locals():
        del data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_smoke(device: torch.device) -> None:
    rows = []
    gpu_before = gpu_snapshot()
    for dataset in DATASETS:
        data = load_dataset(dataset, device)
        for model_name in ("M0", "P0"):
            seed = 129
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model, head = build_modules(data, model_name, seed, device)
            optimizer = optimizer_for(model, head, model_name, 0.001, 0.001)
            noise = torch.Generator(device=device).manual_seed(31_415 + seed)
            pair = torch.Generator(device=device).manual_seed(8_675_309 + seed)
            model.train(); head.train()
            times, losses = [], []
            for actual_update, simulated_schedule in enumerate((1, 50, 75, 100, 101), 1):
                optimizer.zero_grad(set_to_none=True)
                start = time.perf_counter()
                total, main, bal, _, details, _ = _train_forward(model_name, model, head, data, simulated_schedule, noise, pair)
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for module in (model, head) for p in module.parameters()], 5.0)
                optimizer.step()
                if device.type == "cuda": torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                if not torch.isfinite(total) or not torch.isfinite(grad_norm):
                    raise FloatingPointError("smoke loss/gradient is nonfinite")
                losses.append(float(total.detach())); times.append(elapsed)
            key_grad = float(model.layers[0].experts[0].w_nbr.weight.grad.norm())
            row = {"dataset": dataset, "model": model_name, "actual_optimizer_updates": 5, "simulated_schedule_steps": [1, 50, 75, 100, 101], "losses": losses, "median_step_seconds": float(statistics.median(times)), "neighbor_weight_gradient_norm_after_final_smoke_update": key_grad, "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model, head, optimizer; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    DATA_CACHE.clear()
    if "data" in locals():
        del data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    write_json(REPORT_ROOT / "smoke_summary.json", {"study": STUDY, "created_utc": utc_now(), "gpu_snapshot_before": gpu_before, "gpu_snapshot_after": gpu_snapshot(), "gpu_compute_processes_after": gpu_process_snapshot(), "selected_device": str(device), "selected_gpu_uuid": SELECTED_GPU_UUID if device.type == "cuda" else None, "runs": rows, "scope": "six real-graph five-update smoke cells; no validation/test evaluation or formal score"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.freeze:
        freeze_protocol()
    if args.smoke:
        device = torch.device(args.device)
        validate_compute_device(device)
        run_smoke(device)
    if args.calibration:
        run_phase("calibration", torch.device(args.device), retry_failed=args.retry_failed)
    if args.confirmation:
        run_phase("confirmation", torch.device(args.device), retry_failed=args.retry_failed)
    if not any((args.prepare, args.freeze, args.smoke, args.calibration, args.confirmation)):
        parser.error("select at least one action")


if __name__ == "__main__":
    main()
