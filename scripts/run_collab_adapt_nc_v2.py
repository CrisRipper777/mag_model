from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
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
from src.models.mag_collab_adapt import MAGCollabAdapt, parameter_groups_no_decay
from src.models.mag_collab_nc import MAGCollabNC as FrozenV1Model
from src.tasks.collab_nc_pilot import CollabTaskHeads, indexed_logits, train_only_loss

STUDY = "mag_collab_adapt_nc_v2"
DATASETS = ("Movies", "Grocery", "Reddit-S")
R_MODELS = ("R-M0", "R-M1")
S_MODELS = ("S0", "S1", "S2", "S3", "S4", "S5")
R_LR = (0.0003, 0.001, 0.003)
R_WD = (0.0001, 0.001)
S_LR = (0.0003, 0.001, 0.003)
S_WD = (0.001, 0.05)
R_CAL_SEED, R_CONFIRM = 62, (63, 64, 65)
S_CAL_SEED, S_CONFIRM = 72, (73, 74, 75)
SPLIT_SEED = 42
DIAGNOSTIC_STEPS = (1, 10, 30, 100)
V1_LOCK = ROOT / "docs/collab_nc_pilot_v1/protocol_lock.json"
HISTORICAL_LOCK = ROOT / "reports/problem_validation_v1/protocol_lock.json"
DOC_ROOT = ROOT / "docs/collab_adapt_nc_v2"
REPORT_ROOT = ROOT / "reports/collab_adapt_nc_v2"
OUTPUT_ROOT = ROOT / "outputs/collab_adapt_nc_v2"
PROTOCOL_LOCK = DOC_ROOT / "protocol_lock.json"
REGISTRY_PATH = REPORT_ROOT / "run_registry.json"
RESULTS_PATH = REPORT_ROOT / "run_results.jsonl"

SOURCE_FILES = (
    "src/models/mag_collab_adapt.py", "src/models/mag_collab_nc.py",
    "src/models/factory.py", "src/models/predictor.py",
    "src/tasks/collab_nc_pilot.py", "src/tasks/prior_context_pilot.py",
    "src/data/loaders.py", "src/data/splits.py", "src/data/graph_utils.py",
    "src/data/types.py", "src/data/__init__.py", "configs/config.yaml",
    "configs/task/nc.yaml", "configs/model/mag_collab_adapt.yaml",
    "configs/model/mag_collab_nc.yaml", "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml", "configs/dataset/Reddit-S.yaml",
    "scripts/run_collab_adapt_nc_v2.py", "scripts/audit_collab_adapt_nc_v2.py",
    "tests/test_collab_adapt.py", "docs/collab_adapt_nc_v2/protocol.md",
    "docs/collab_adapt_nc_v2/model_spec.md", "docs/collab_adapt_nc_v2/matrix.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return safe_json(value.detach().cpu().tolist())
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe_json(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(safe_json(value)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gpu_snapshot() -> dict[str, Any]:
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        rows = []
        for fields in csv.reader(proc.stdout.splitlines(), skipinitialspace=True):
            if len(fields) == 5:
                rows.append({"physical_index": int(fields[0]), "name": fields[1], "memory_used_mib": int(fields[2]), "memory_total_mib": int(fields[3]), "utilization_percent": int(fields[4])})
        return {"status": "observed", "time_utc": utc_now(), "devices": rows}
    except Exception as exc:
        return {"status": "unavailable", "error": repr(exc), "time_utc": utc_now()}


def historical_v1() -> dict[str, Any]:
    return json.loads(V1_LOCK.read_text(encoding="utf-8"))


def base_config(dataset: str, seed: int, device: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "model=mag_collab_adapt", f"seed={seed}", f"device={device}"])
    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.task.evaluate_test = False
    cfg.task.eval_labels = list(historical_v1()["old_schema"][dataset]["macro_f1_label_set"])
    cfg.dataset.nc_split_path = historical_v1()["datasets"][dataset]["split_path"]
    cfg.dataset.auto_generate_splits = False
    cfg.dataset.make_undirected = True
    cfg.dataset.add_self_loops = False
    cfg.model.base_variant = "M0"
    cfg.model.out_dim = 640
    if bool(cfg.task.evaluate_test):
        raise RuntimeError("task.evaluate_test must stay false")
    return cfg


def verify_locked_inputs(dataset: str, prior_lock: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    old = historical_v1()["datasets"][dataset]
    found: dict[str, dict[str, Any]] = {}
    for raw, expected in old["input_hashes"].items():
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise RuntimeError(f"locked input changed: {path}")
        found[str(path)] = actual
    split = found.get(old["split_path"])
    if split is None or split["sha256"] != old["split_sha256"]:
        raise RuntimeError(f"split seed42 identity mismatch: {dataset}")
    if prior_lock is not None and found != prior_lock["inputs"][dataset]:
        raise RuntimeError(f"input hashes no longer match the v2 lock: {dataset}")
    return found


def load_cpu_data(dataset: str):
    cfg = base_config(dataset, S_CAL_SEED, "cpu")
    data = load_mag_data(cfg, "nc", SPLIT_SEED)
    return cfg, data


def data_profile(dataset: str, data, lock_v1: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    prior = lock_v1["datasets"][dataset]
    if int(data.num_nodes) != prior["num_nodes"] or int(data.edge_index.size(1)) != prior["loader_edge_count"]:
        raise RuntimeError(f"loaded graph identity differs from frozen v1 graph: {dataset}")
    if int(data.x_t.size(1)) != prior["text_dim"] or int(data.x_i.size(1)) != prior["image_dim"]:
        raise RuntimeError(f"feature widths differ from frozen v1 graph: {dataset}")
    available, availability = FrozenV1Model.infer_availability(data.x_t, data.x_i)
    counts = [int(available[:, m].sum()) for m in range(2)]
    if availability != prior["availability"] or counts != prior["available_counts"]:
        raise RuntimeError(f"availability profile differs from frozen v1 input: {dataset}")
    return {
        "dataset": dataset, "inputs": inputs, "num_nodes": int(data.num_nodes),
        "loader_edge_count": int(data.edge_index.size(1)), "loader_self_loops": int((data.edge_index[0] == data.edge_index[1]).sum()),
        "loader_edge_tensor_sha256": tensor_sha256(data.edge_index),
        "text_dim": int(data.x_t.size(1)), "image_dim": int(data.x_i.size(1)),
        "num_classes": int(data.num_classes), "train_count": int(data.train_idx.numel()),
        "validation_count": int(data.val_idx.numel()), "available_counts": counts,
        "train_ids_sha256": tensor_sha256(data.train_idx), "validation_ids_sha256": tensor_sha256(data.val_idx),
        "availability": availability, "split_seed": SPLIT_SEED,
        "split_path": prior["split_path"], "split_sha256": prior["split_sha256"],
        "diagnostic_node_ids": prior["diagnostic_node_ids"],
        "diagnostic_node_ids_sha256": prior["diagnostic_node_ids_sha256"],
        "edge_semantics": "loader edge_index used exactly as loaded; no reverse, deduplication, self-loop, sampling or truncation",
    }


def matrix_rows() -> list[dict[str, Any]]:
    rows = []
    for stage, models, seed, lrs, wds in (("R", R_MODELS, R_CAL_SEED, R_LR, R_WD), ("S", S_MODELS, S_CAL_SEED, S_LR, S_WD)):
        for dataset in DATASETS:
            for model_id in models:
                for config_index, (lr, wd) in enumerate((lr, wd) for lr in lrs for wd in wds):
                    rows.append({"stage": stage, "phase": "calibration", "dataset": dataset, "model_id": model_id, "training_seed": seed, "config_index": config_index, "lr": lr, "weight_decay": wd})
                for run_seed in (R_CONFIRM if stage == "R" else S_CONFIRM):
                    rows.append({"stage": stage, "phase": "confirmation", "dataset": dataset, "model_id": model_id, "training_seed": run_seed, "selection_locked": True})
    return rows


def phase_prepare() -> None:
    if not git("status", "--porcelain", "--", *SOURCE_FILES) == "":
        raise RuntimeError("training source files must be committed before prepare")
    if git("branch", "--show-current") != "codex/mag-collab-adapt-nc-v2":
        raise RuntimeError("prepare must run on codex/mag-collab-adapt-nc-v2")
    if not Path("docs/collab_adapt_nc_v2/protocol.md").is_file() or not Path("docs/collab_adapt_nc_v2/matrix.json").is_file():
        raise FileNotFoundError("review/commit the protocol and matrix before prepare")
    prior = historical_v1()
    if sha256_file(HISTORICAL_LOCK) != prior.get("v1_protocol_lock_sha256", sha256_file(HISTORICAL_LOCK)):
        raise RuntimeError("historical problem-validation lock changed")
    v1_lock = json.loads(V1_LOCK.read_text(encoding="utf-8"))
    inputs, profiles = {}, {}
    for dataset in DATASETS:
        inputs[dataset] = verify_locked_inputs(dataset)
        _, data = load_cpu_data(dataset)
        profiles[dataset] = data_profile(dataset, data, v1_lock, inputs[dataset])
        del data
        gc.collect()
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}
    source_commit = git("rev-parse", "HEAD")
    lock = {
        "study": STUDY, "schema_version": 1, "created_utc": utc_now(),
        "branch": git("branch", "--show-current"), "training_source_commit": source_commit,
        "training_source_hashes": source_hashes,
        "prior_v1_training_source_commit": v1_lock["training_source_commit"],
        "prior_v1_training_source_hashes": v1_lock["source_hashes"],
        "prior_v1_protocol_lock": str(V1_LOCK.resolve()), "prior_v1_protocol_lock_sha256": sha256_file(V1_LOCK),
        "historical_problem_validation_lock": str(HISTORICAL_LOCK.resolve()), "historical_problem_validation_lock_sha256": sha256_file(HISTORICAL_LOCK),
        "inputs": inputs, "datasets": profiles,
        "matrix": {"R_calibration": 36, "R_confirmation": 18, "S_calibration": 108, "S_confirmation": 54, "formal_total": 216, "smoke_runs": 6, "smoke_steps": 18},
        "environment": {
            "environment_name": "yhf_env", "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "dgl": pkg_version("dgl"), "numpy": np.__version__, "sklearn": pkg_version("scikit-learn"),
            "physical_gpu_selected": 1, "cuda_visible_devices_expected": "1", "process_device_expected": "cuda:0",
            "visible_gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_snapshot": gpu_snapshot(), "torch_threads": torch.get_num_threads(),
            "amp": False, "dtype": "float32", "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "split_seed": SPLIT_SEED, "evaluation_policy": "train/validation only; no task-head test forward, test metrics, test selection, real LP, or excluded NC datasets",
        "formal_matrix": matrix_rows(), "smoke_policy": "R-M0 and S5 each run exactly three train-only steps on every dataset, all excluded from selection",
        "seed_policy": {"backbone": "training seed with deterministic name-derived local Xavier seeds", "task_heads": "training seed + 10007", "pair_rng": "training seed + 31415"},
    }
    write_json(PROTOCOL_LOCK, lock)
    write_json(REPORT_ROOT / "environment_profile.json", lock["environment"])
    write_json(REPORT_ROOT / "data_profile.json", profiles)
    write_json(REPORT_ROOT / "run_registry.json", {"study": STUDY, "source_commit": source_commit, "attempts": {}})
    write_json(REPORT_ROOT / "run_results_header.json", {"study": STUDY, "source_commit": source_commit, "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK)})
    print(f"prepared locked inputs for {len(profiles)} datasets; source={source_commit}; lock={sha256_file(PROTOCOL_LOCK)}")


def verify_frozen_source() -> dict[str, Any]:
    if not PROTOCOL_LOCK.is_file():
        raise FileNotFoundError("run prepare after the training source commit")
    lock = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    if lock["study"] != STUDY:
        raise RuntimeError("unexpected protocol-lock study ID")
    if subprocess.run(["git", "merge-base", "--is-ancestor", lock["training_source_commit"], "HEAD"], cwd=ROOT).returncode != 0:
        raise RuntimeError("current HEAD does not descend from the frozen training-source commit")
    for relative, expected in lock["training_source_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"frozen training source changed: {relative}")
    for dataset in DATASETS:
        verify_locked_inputs(dataset, lock)
    return lock


def configure_model_id(cfg, model_id: str, *, stage: str):
    if stage == "R":
        cfg.model.base_variant = "M0" if model_id == "R-M0" else "M1"
        cfg.model.typed_value = False
        cfg.model.fusion = False
        cfg.model.out_dim = 640
    else:
        mapping = {
            "S0": ("B1", False, False), "S1": ("B1", False, True),
            "S2": ("M0", False, False), "S3": ("M0", True, False),
            "S4": ("M0", False, True), "S5": ("M0", True, True),
        }
        base, typed_value, fusion = mapping[model_id]
        cfg.model.base_variant = base
        cfg.model.typed_value = typed_value
        cfg.model.fusion = fusion
        cfg.model.out_dim = 768
    cfg.model.collect_diagnostics = False
    cfg.model.optimize_cache = True
    return cfg


def build_modules(model_id: str, stage: str, dataset: str, info: dict[str, int], seed: int, device: torch.device):
    cfg = configure_model_id(base_config(dataset, seed, str(device)), model_id, stage=stage)
    torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = MAGCollabAdapt(
        info, str(cfg.model.base_variant), dropout=0.2, init_seed=seed,
        typed_value=bool(cfg.model.typed_value), fusion=bool(cfg.model.fusion),
        out_dim=int(cfg.model.out_dim), collect_diagnostics=False, optimize_cache=True,
    )
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed) + 10_007)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 10_007)
    heads = CollabTaskHeads(str(cfg.model.base_variant), 20, output_dim=int(cfg.model.out_dim))
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    model.to(device); heads.to(device)
    return cfg, model, heads


def load_run_data(dataset: str, device: torch.device, lock: dict[str, Any]) -> dict[str, Any]:
    verify_locked_inputs(dataset, lock)
    cfg = base_config(dataset, S_CAL_SEED, "cpu")
    data = load_mag_data(cfg, "nc", SPLIT_SEED)
    profile = lock["datasets"][dataset]
    if int(data.num_nodes) != profile["num_nodes"] or int(data.edge_index.size(1)) != profile["loader_edge_count"]:
        raise RuntimeError(f"graph shape changed after prepare for {dataset}")
    if tensor_sha256(data.edge_index) != profile["loader_edge_tensor_sha256"]:
        raise RuntimeError(f"loader edge order/content changed after prepare for {dataset}")
    if tensor_sha256(data.train_idx) != profile["train_ids_sha256"] or tensor_sha256(data.val_idx) != profile["validation_ids_sha256"]:
        raise RuntimeError(f"seed42 train/validation indices changed after prepare for {dataset}")
    x = data.x.contiguous().to(device=device, dtype=torch.float32)
    avail_cpu, availability = FrozenV1Model.infer_availability(data.x_t, data.x_i)
    if availability != profile["availability"] or [int(avail_cpu[:, 0].sum()), int(avail_cpu[:, 1].sum())] != profile["available_counts"]:
        raise RuntimeError(f"modality availability changed after prepare for {dataset}")
    train = data.train_idx.to(device=device, dtype=torch.long)
    val = data.val_idx.to(device=device, dtype=torch.long)
    # Test labels/IDs are deliberately never copied or indexed by this runner.
    labels = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
    labels[train] = data.y.index_select(0, data.train_idx).to(device)
    labels[val] = data.y.index_select(0, data.val_idx).to(device)
    diag_train = torch.tensor(profile["diagnostic_node_ids"]["train"], dtype=torch.long, device=device)
    diag_val = torch.tensor(profile["diagnostic_node_ids"]["validation"], dtype=torch.long, device=device)
    return {
        "x": x, "edges": data.edge_index.to(device=device, dtype=torch.long).contiguous(),
        "available": avail_cpu.to(device), "train_idx": train, "val_idx": val,
        "y": labels, "diagnostic_train": diag_train, "diagnostic_val": diag_val,
        "diagnostic_nodes": torch.cat((diag_train, diag_val)),
        "info": {"input_dim": int(data.input_dim), "text_dim": int(data.x_t.size(1)), "visual_dim": int(data.x_i.size(1)), "num_nodes": int(data.num_nodes)},
        "num_nodes": int(data.num_nodes), "num_edges": int(data.edge_index.size(1)),
    }


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model: nn.Module, heads: nn.Module, lr: float, wd: float):
    params = parameter_groups_no_decay(model, [heads])
    for group in params:
        group["weight_decay"] = 0.0 if group.get("weight_decay") == 0.0 else float(wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)


def parameters_by_group(model: nn.Module, heads: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {
        "projection_and_center": [], "initial_stems": [], "same_modality_graph": [],
        "private_graph": [], "common_graph": [], "typed_value": [], "fusion": [],
        "ssl": [], "main_head": [], "path_heads": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith(("p_text", "p_image", "p_norm", "e_mlp", "e_norm")):
            groups["projection_and_center"].append(parameter)
        elif name.startswith(("common_adapter", "common_stem", "private_stem")):
            groups["initial_stems"].append(parameter)
        elif name.startswith(("b1_norms", "b1_updates")):
            groups["same_modality_graph"].append(parameter)
        elif name.startswith("layers.") and ".private." in name:
            groups["private_graph"].append(parameter)
        elif name.startswith("layers.") and ".common." in name:
            if any(token in name for token in ("value_sender", "message_same", "message_cross")):
                groups["typed_value"].append(parameter)
            else:
                groups["common_graph"].append(parameter)
        elif name.startswith("fusion_"):
            groups["fusion"].append(parameter)
        elif name.startswith(("decoder_", "pair_projection")):
            groups["ssl"].append(parameter)
        else:
            groups["initial_stems"].append(parameter)
    for name, parameter in heads.named_parameters():
        groups["main_head" if name.startswith("main.") else "path_heads"].append(parameter)
    return {name: params for name, params in groups.items() if params}


def parameters_norm(params: list[nn.Parameter]) -> torch.Tensor:
    values = [p.detach().float().square().sum() for p in params]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0)


def grad_norm(params: list[nn.Parameter]) -> torch.Tensor:
    values = [p.grad.detach().float().square().sum() for p in params if p.grad is not None]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0)


def flatten_params(params: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().float().reshape(-1) for p in params]).clone() if params else torch.empty(0)


def optimizer_groups(model: nn.Module, heads: nn.Module, lr: float, wd: float):
    return make_optimizer(model, heads, lr, wd)


def make_attempt_id(stage: str, phase: str, dataset: str, model_id: str, seed: int, lr: float, wd: float, config_index: int, retry: int = 0) -> str:
    name = f"{stage.lower()}_{phase}_{dataset}_{model_id}_seed{seed}_cfg{config_index:02d}_lr{lr:.7g}_wd{wd:.7g}"
    name = name.replace(".", "p")
    return name if retry == 0 else f"{name}_retry{retry}"


def read_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {"study": STUDY, "attempts": {}}


def save_registry(registry: dict[str, Any]) -> None:
    write_json(REGISTRY_PATH, registry)


def read_results() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    return [json.loads(line) for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def rng_state(pair_rng: torch.Generator) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "pair_rng": pair_rng.get_state()}


def restore_rng(state: dict[str, Any], pair_rng: torch.Generator) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    pair_rng.set_state(state["pair_rng"])


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def checkpoint_payload(model, heads, optimizer, pair_rng, *, step, best_key, best_step, patience_left, patience_reference, metadata):
    return {
        "model": model.state_dict(), "task_heads": heads.state_dict(), "optimizer": optimizer.state_dict(),
        "step": int(step), "best_key": best_key, "best_step": int(best_step),
        "patience_left": int(patience_left), "patience_reference": patience_reference,
        "rng": rng_state(pair_rng), "resolved_run_metadata": metadata,
        "model_schedule": {"beta": float(model.beta), "tau": float(model.tau)},
    }


def stage_spec(stage: str, model_id: str) -> dict[str, Any]:
    if stage == "R":
        return {"base_variant": "M0" if model_id == "R-M0" else "M1", "typed_value": False, "fusion": False, "out_dim": 640, "max_steps": 300, "min_stop": 60, "patience": 40, "best_policy": "correct_macro_f1_earlier", "loss_weights": {"path": 0.05, "rec": 0.01 if model_id == "R-M0" else 0.0, "pair": 0.01 if model_id == "R-M0" else 0.0}, "grid_lr": R_LR, "grid_wd": R_WD}
    base, typed, fusion = {"S0": ("B1", False, False), "S1": ("B1", False, True), "S2": ("M0", False, False), "S3": ("M0", True, False), "S4": ("M0", False, True), "S5": ("M0", True, True)}[model_id]
    return {"base_variant": base, "typed_value": typed, "fusion": fusion, "out_dim": 768, "max_steps": 400, "min_stop": 20, "patience": 60, "best_policy": "U_acc_f1_earlier", "loss_weights": {"path": 0.05, "rec": 0.01 if base == "M0" or base == "B1" else 0.0, "pair": 0.01 if base == "M0" or base == "B1" else 0.0}, "grid_lr": S_LR, "grid_wd": S_WD}


def metric_block(labels: torch.Tensor, logits: torch.Tensor, class_ids: list[int], *, include_confusion: bool = False) -> dict[str, Any]:
    pred = logits.detach().argmax(-1)
    y_np, p_np = labels.detach().cpu().long().numpy(), pred.cpu().long().numpy()
    result: dict[str, Any] = {
        "count": int(labels.numel()), "correct": int((pred == labels).sum().item()),
        "accuracy": float((pred == labels).float().mean().item()) if labels.numel() else None,
        "macro_f1": float(f1_score(y_np, p_np, labels=class_ids, average="macro", zero_division=0)) if labels.numel() else None,
        "cross_entropy": float(F.cross_entropy(logits, labels).detach().cpu()) if labels.numel() else None,
    }
    if include_confusion:
        result["confusion_matrix"] = confusion_matrix(y_np, p_np, labels=class_ids).astype(int).tolist()
        probabilities = torch.softmax(logits.detach().float(), dim=-1)
        one_hot = F.one_hot(labels.long(), num_classes=len(class_ids)).to(probabilities.dtype)
        result["brier_sum_over_classes"] = float((probabilities - one_hot).square().sum(-1).mean().cpu())
        max_conf = probabilities.max(-1).values
        wrong = pred != labels
        result["wrong_prediction_mean_max_confidence"] = float(max_conf[wrong].mean().cpu()) if bool(wrong.any()) else None
        result["nll"] = result["cross_entropy"]
    result["predictions"] = p_np.astype(np.int64)
    return result


def evaluate(stage: str, model_id: str, model, heads, tensors: dict[str, Any], step: int, classes: list[int], *, collect_diagnostics: bool = False, with_grad: bool = False):
    spec = stage_spec(stage, model_id)
    model.eval(); heads.eval(); model.set_epoch(step)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], compute_ssl=False, diagnostic_node_ids=tensors["diagnostic_nodes"] if collect_diagnostics else None, collect_diagnostics=collect_diagnostics)
        result, predictions = {}, {}
        for part, idx in (("train", tensors["train_idx"]), ("validation", tensors["val_idx"])):
            logits, masks = indexed_logits(spec["base_variant"], z, extras, heads, idx)
            y = tensors["y"].index_select(0, idx)
            result[part] = {}
            for head_name, values in logits.items():
                mask = masks.get(head_name, torch.ones_like(y, dtype=torch.bool))
                if not bool(mask.any()):
                    result[part][head_name] = None
                    continue
                metrics = metric_block(y[mask], values[mask], classes, include_confusion=collect_diagnostics)
                predictions[f"{part}_{head_name}"] = metrics.pop("predictions")
                result[part][head_name] = metrics
        result["diagnostics"] = extras.get("diagnostics", {})
        result["availability"] = extras["availability"]
        result["z"] = z if collect_diagnostics else None
        result["extras"] = extras if collect_diagnostics else None
    return result, predictions


def validation_utility(metrics: dict[str, Any]) -> float:
    return 0.5 * (float(metrics["accuracy"]) + float(metrics["macro_f1"]))


def prediction_npz_payload(tensors, predictions):
    result = {"train_ids": tensors["train_idx"].detach().cpu().numpy(), "validation_ids": tensors["val_idx"].detach().cpu().numpy()}
    for key, pred in predictions.items():
        result[f"{key}_pred"] = np.asarray(pred, dtype=np.int64)
    result["train_available"] = tensors["available"].index_select(0, tensors["train_idx"]).detach().cpu().numpy().astype(np.uint8)
    result["validation_available"] = tensors["available"].index_select(0, tensors["val_idx"]).detach().cpu().numpy().astype(np.uint8)
    return result


def ablation_eval(stage: str, model_id: str, model, heads, tensors, step: int, classes: list[int]) -> dict[str, Any]:
    spec = stage_spec(stage, model_id)
    model.eval(); heads.eval(); model.set_epoch(step)
    with torch.no_grad():
        z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], compute_ssl=False)
        variants: dict[str, torch.Tensor] = {"full": z}
        if spec["fusion"]:
            no_r = z.clone(); no_r[:, 640:] = 0
            variants["mask_r"] = no_r
        if stage == "S" and model_id == "S5":
            for label, spans in (("mask_private", ((384, 512),)), ("mask_common", ((512, 640),)), ("mask_all_graph", ((384, 640),))):
                masked = z.clone()
                for start, stop in spans:
                    masked[:, start:stop] = 0
                variants[label] = masked
        output = {}
        for label, z_variant in variants.items():
            output[label] = {}
            for part, idx in (("train", tensors["train_idx"]), ("validation", tensors["val_idx"])):
                y = tensors["y"].index_select(0, idx)
                logits = heads.main(z_variant.index_select(0, idx))
                metrics = metric_block(y, logits, classes, include_confusion=True)
                metrics.pop("predictions", None)
                output[label][part] = metrics
        return output


def module_diagnostic(stage: str, model_id: str, model, heads, tensors, step: int, classes: list[int], groups: dict[str, list[nn.Parameter]], group_before: dict[str, torch.Tensor] | None, group_pre_clip: dict[str, torch.Tensor] | None, global_pre_clip: torch.Tensor | None) -> dict[str, Any]:
    measured, _ = evaluate(stage, model_id, model, heads, tensors, step, classes, collect_diagnostics=True, with_grad=True)
    base = stage_spec(stage, model_id)["base_variant"]
    objective, _, _ = train_only_loss(base, measured["z"], measured["extras"], heads, tensors["y"], tensors["train_idx"], path_weight=stage_spec(stage, model_id)["loss_weights"]["path"], rec_weight=stage_spec(stage, model_id)["loss_weights"]["rec"], pair_weight=stage_spec(stage, model_id)["loss_weights"]["pair"])
    e_grad, delta_grad = torch.autograd.grad(objective, (measured["extras"]["diagnostic_tensors"]["e"], measured["extras"]["diagnostic_tensors"]["delta"]), allow_unused=True)
    diag_groups = {}
    for name, params in groups.items():
        after = flatten_params(params)
        before = group_before.get(name) if group_before is not None else None
        update_norm = float((after - before).norm().cpu()) if before is not None else None
        diag_groups[name] = {
            "parameter_norm": float(after.norm().cpu()) if after.numel() else None,
            "post_clip_gradient_norm": float(grad_norm(params).cpu()) if params else None,
            "pre_clip_gradient_norm": float(group_pre_clip[name].cpu()) if group_pre_clip is not None and name in group_pre_clip else None,
            "update_norm": update_norm,
            "update_sampled": before is not None,
        }
    main_weight = heads.main.weight.detach()
    readout = {
        "p_text": float(main_weight[:, 0:128].norm().cpu()),
        "p_image": float(main_weight[:, 128:256].norm().cpu()),
        "e": float(main_weight[:, 256:384].norm().cpu()),
        "private_text": float(main_weight[:, 384:448].norm().cpu()),
        "private_image": float(main_weight[:, 448:512].norm().cpu()),
        "common_text": float(main_weight[:, 512:576].norm().cpu()),
        "common_image": float(main_weight[:, 576:640].norm().cpu()),
        "fusion_r": float(main_weight[:, 640:768].norm().cpu()) if model.out_dim == 768 else None,
        "inactive_padding": float(main_weight[:, 640:768].norm().cpu()) if model.out_dim == 768 and not model.fusion_enabled else 0.0,
    }
    typed = []
    if model.typed_value:
        for layer in model.layers:
            typed.append({"sender_value_weight_norms": [float(m.weight.detach().norm().cpu()) for m in layer.common.value_sender], "message_same_norm": float(layer.common.message_same.detach().norm().cpu()), "message_cross_norm": float(layer.common.message_cross.detach().norm().cpu())})
    fusion = None
    if model.fusion_enabled:
        nodes = tensors["diagnostic_nodes"]
        fusion = {
            "r_sample_mean_norm": float(measured["extras"]["fusion_residual"].index_select(0, nodes).norm(dim=-1).mean().detach().cpu()),
            "W_e_norm": float(model.fusion_e.weight.detach().norm().cpu()),
            "W_delta_norm": float(model.fusion_delta.weight.detach().norm().cpu()),
            "W_o_norm": float(model.fusion_out.weight.detach().norm().cpu()),
            "bias_norm": float(model.fusion_bias.detach().norm().cpu()),
            "train_only_eval_objective_grad_e_norm": float(e_grad.detach().norm().cpu()) if e_grad is not None else None,
            "train_only_eval_objective_grad_delta_norm": float(delta_grad.detach().norm().cpu()) if delta_grad is not None else None,
        }
    return {
        "step": int(step), "beta": float(model.beta), "tau": float(model.tau),
        "global_pre_clip_gradient_norm": float(global_pre_clip.detach().cpu()) if global_pre_clip is not None else None,
        "global_post_clip_gradient_norm": float(torch.stack([grad_norm(p) for p in groups.values()]).norm().cpu()) if groups else None,
        "groups": diag_groups, "backbone": measured["diagnostics"], "readout_block_weight_norms": readout,
        "typed_value": typed, "fusion": fusion,
        "diagnostic_gradient_objective": "eval-mode train main+path CE only; train labels only; autograd.grad at e/delta; no optimizer update, SSL RNG, or parameter .grad accumulation",
    }


def _read_prediction_ids(tensors: dict[str, Any]) -> dict[str, list[int]]:
    return {"train": tensors["train_idx"].detach().cpu().tolist(), "validation": tensors["val_idx"].detach().cpu().tolist()}


def _selection_compare_key(stage: str, row: dict[str, Any]):
    if stage == "R":
        return (int(row["best_val_correct"]), float(row["best_val_macro_f1"]), -int(row["best_step"]), -int(row["config_index"]))
    return (float(row["best_utility"]), float(row["best_val_accuracy"]), float(row["best_val_macro_f1"]), -int(row["best_step"]), -int(row["config_index"]))


def run_cell(*, stage: str, phase: str, dataset: str, model_id: str, seed: int, lr: float, wd: float, config_index: int, tensors: dict[str, Any], lock: dict[str, Any], device: torch.device, retry_failed: bool = False, no_resume: bool = False) -> dict[str, Any]:
    spec = stage_spec(stage, model_id)
    registry = read_registry()
    base_id = make_attempt_id(stage, phase, dataset, model_id, seed, lr, wd, config_index)
    related = [key for key in registry["attempts"] if key == base_id or key.startswith(base_id + "_retry")]
    related.sort(key=lambda key: (0 if "_retry" not in key else int(key.rsplit("_retry", 1)[-1])))
    previous = registry["attempts"].get(related[-1]) if related else None
    if previous and previous.get("status") == "success":
        return previous
    resume_same = bool(previous and previous.get("status") in {"running", "interrupted"} and not no_resume and Path(previous.get("output_dir", ""), "latest.pt").is_file())
    if previous and previous.get("status") in {"failed", "invalid"} and not retry_failed:
        return previous
    if previous and previous.get("status") in {"running", "interrupted"} and not resume_same and not retry_failed:
        stopped = dict(previous); stopped.update({"status": "interrupted", "ended_utc": utc_now(), "error": "latest checkpoint missing or resume disabled"})
        registry["attempts"][related[-1]] = stopped; save_registry(registry); append_jsonl(RESULTS_PATH, stopped)
        return stopped
    last_id = related[-1] if related else ""
    retry_index = ((int(last_id.rsplit("_retry", 1)[1]) + 1) if "_retry" in last_id else 1) if related and retry_failed and not resume_same else 0
    run_id = related[-1] if resume_same else make_attempt_id(stage, phase, dataset, model_id, seed, lr, wd, config_index, retry_index)
    output_dir = Path(previous["output_dir"]) if resume_same else OUTPUT_ROOT / stage / dataset / model_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, model, heads = build_modules(model_id, stage, dataset, tensors["info"], seed, device)
    optimizer = optimizer_groups(model, heads, lr, wd)
    pair_rng = torch.Generator(device=device).manual_seed(int(seed) + 31_415)
    classes = list(historical_v1()["old_schema"][dataset]["macro_f1_label_set"])
    groups = parameters_by_group(model, heads)
    all_params = [p for p in list(model.parameters()) + list(heads.parameters()) if p.requires_grad]
    max_steps, min_stop, patience = spec["max_steps"], spec["min_stop"], spec["patience"]
    base_variant = spec["base_variant"]
    metadata = {
        "attempt_id": run_id, "stage": stage, "phase": phase, "model_id": model_id,
        "base_variant": base_variant, "typed_value": spec["typed_value"], "fusion": spec["fusion"],
        "output_dim": spec["out_dim"], "dataset": dataset, "training_seed": seed, "split_seed": SPLIT_SEED,
        "lr": lr, "weight_decay": wd, "config_index": config_index, "test_evaluation": False,
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK), "training_source_commit": lock["training_source_commit"],
        "input_hashes": lock["inputs"][dataset], "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "loss_weights": spec["loss_weights"], "device": str(device),
    }
    log_path, diag_path = output_dir / "steps.jsonl", output_dir / "diagnostics.jsonl"
    latest_path, best_path = output_dir / "latest.pt", output_dir / "best.pt"
    attempt = dict(previous or {}) if resume_same else {
        "attempt_id": run_id, "stage": stage, "phase": phase, "dataset": dataset, "model_id": model_id,
        "base_variant": base_variant, "typed_value": spec["typed_value"], "fusion": spec["fusion"],
        "training_seed": seed, "split_seed": SPLIT_SEED, "lr": lr, "weight_decay": wd,
        "config_index": config_index, "status": "running", "started_utc": utc_now(), "device": str(device),
        "source_commit": lock["training_source_commit"], "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "input_hashes": lock["inputs"][dataset], "output_dir": str(output_dir), "retry_of": related[-1] if related else None,
    }
    if resume_same:
        attempt["resumed_utc"] = utc_now(); attempt["resume_count"] = int(attempt.get("resume_count", 0)) + 1
    registry["attempts"][run_id] = attempt; save_registry(registry)
    start_step, best_key, best_step, best_correct = 1, None, 0, -1
    patience_left, patience_reference = patience, None
    if resume_same:
        current = torch.load(latest_path, map_location=device, weights_only=False)
        saved_best = torch.load(best_path, map_location=device, weights_only=False) if best_path.is_file() else None
        if saved_best is not None and int(saved_best["step"]) > int(current["step"]):
            current = saved_best
        model.load_state_dict(current["model"]); heads.load_state_dict(current["task_heads"]); optimizer.load_state_dict(current["optimizer"])
        start_step = int(current["step"]) + 1
        best_key, best_step = current["best_key"], int(current["best_step"])
        patience_left, patience_reference = int(current["patience_left"]), current.get("patience_reference")
        best_correct = int(best_key[0]) if stage == "R" and best_key else -1
        restore_rng(current["rng"], pair_rng)
        if log_path.exists():
            logrows = [json.loads(row) for row in log_path.read_text(encoding="utf-8").splitlines() if row.strip()]
            orphaned = [row for row in logrows if int(row["step"]) > int(current["step"])]
            if orphaned:
                for row in orphaned:
                    append_jsonl(output_dir / "orphaned_steps.jsonl", {**row, "status": "not_in_latest_checkpoint"})
                log_path.write_text("".join(canonical_json(row) + "\n" for row in logrows if int(row["step"]) <= int(current["step"])), encoding="utf-8")
        if diag_path.exists():
            diagrows = [json.loads(row) for row in diag_path.read_text(encoding="utf-8").splitlines() if row.strip()]
            diag_path.write_text("".join(canonical_json(row) + "\n" for row in diagrows if int(row["step"]) <= int(current["step"])), encoding="utf-8")

    started = time.perf_counter()
    step_times: list[float] = []
    normal_peak_alloc = normal_peak_reserved = 0
    diagnostic_peak_alloc = diagnostic_peak_reserved = 0
    successful_step = max_steps
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(start_step, max_steps + 1):
            model.train(); heads.train(); model.set_epoch(step)
            sample_step = step in DIAGNOSTIC_STEPS
            before = {name: flatten_params(params) for name, params in groups.items()} if sample_step else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
            else:
                wall_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], pair_generator=pair_rng, compute_ssl=bool(spec["loss_weights"]["rec"] or spec["loss_weights"]["pair"]), collect_diagnostics=False)
            loss, loss_terms, _ = train_only_loss(base_variant, z, extras, heads, tensors["y"], tensors["train_idx"], path_weight=spec["loss_weights"]["path"], rec_weight=spec["loss_weights"]["rec"], pair_weight=spec["loss_weights"]["pair"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite train objective at step={step}")
            loss.backward()
            group_pre = {name: grad_norm(params).detach() for name, params in groups.items()} if sample_step else None
            global_pre = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            if not torch.isfinite(global_pre):
                raise FloatingPointError(f"nonfinite pre-clip global gradient at step={step}")
            optimizer.step()
            if device.type == "cuda":
                event_end.record(); torch.cuda.synchronize(device)
                step_ms = float(event_start.elapsed_time(event_end))
            else:
                step_ms = float((time.perf_counter() - wall_start) * 1000.0)
            step_times.append(step_ms)
            train_loss_values = {key: float(val.detach().cpu()) for key, val in loss_terms.items()}
            train_loss_values["total"] = float(loss.detach().cpu())
            eval_result, eval_predictions = evaluate(stage, model_id, model, heads, tensors, step, classes, collect_diagnostics=False, with_grad=False)
            val_main = eval_result["validation"]["main"]
            if val_main is None:
                raise RuntimeError("validation main head unexpectedly empty")
            eligible = step >= (20 if stage == "S" else 20 if base_variant != "B0" else 1)
            if stage == "R":
                key = (int(val_main["correct"]), float(val_main["macro_f1"]), -int(step))
                better = bool(eligible and (best_key is None or key > tuple(best_key)))
                count_improved = bool(eligible and int(val_main["correct"]) > best_correct)
                if eligible:
                    patience_left = patience if best_key is None or count_improved else patience_left - 1
                    best_correct = max(best_correct, int(val_main["correct"]))
            else:
                utility = validation_utility(val_main)
                key = (utility, float(val_main["accuracy"]), float(val_main["macro_f1"]), -int(step))
                better = bool(eligible and (best_key is None or key > tuple(best_key)))
                if eligible:
                    if patience_reference is None:
                        patience_reference = utility; patience_left = patience
                    elif utility >= float(patience_reference) + 0.0005:
                        patience_reference = utility; patience_left = patience
                    else:
                        patience_left -= 1
            if better:
                best_key, best_step = key, step
            if device.type == "cuda":
                normal_peak_alloc = max(normal_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
                normal_peak_reserved = max(normal_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            need_diag = sample_step or better
            diagnostic = None
            if need_diag:
                postclip_by_group = {name: grad_norm(params).detach() for name, params in groups.items()}
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                diagnostic = module_diagnostic(stage, model_id, model, heads, tensors, step, classes, groups, before if sample_step else None, group_pre, global_pre if sample_step else None)
                if device.type == "cuda":
                    diagnostic_peak_alloc = max(diagnostic_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
                    diagnostic_peak_reserved = max(diagnostic_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
                diagnostic["global_post_clip_gradient_norm"] = float(torch.stack(list(postclip_by_group.values())).norm().cpu()) if postclip_by_group else 0.0
                diagnostic["reason"] = [*( [f"step_{step}"] if sample_step else []), *( ["best_checkpoint"] if better else [])]
                diagnostic["clip_coefficient"] = float(min(1.0, 1.0 / (float(global_pre.detach().cpu()) + 1e-6))) if sample_step else None
                append_jsonl(diag_path, diagnostic)
            if better:
                atomic_save(best_path, checkpoint_payload(model, heads, optimizer, pair_rng, step=step, best_key=best_key, best_step=best_step, patience_left=patience_left, patience_reference=patience_reference, metadata=metadata))
            record = {
                "step": step, "loss": train_loss_values,
                "train": {head: {k: v for k, v in values.items() if k != "confusion_matrix"} if values is not None else None for head, values in eval_result["train"].items()},
                "validation": {head: {k: v for k, v in values.items() if k != "confusion_matrix"} if values is not None else None for head, values in eval_result["validation"].items()},
                "eligible": eligible, "best": better, "best_step": best_step, "best_key": best_key,
                "patience_left": patience_left, "patience_reference_utility": patience_reference,
                "beta": float(model.beta), "tau": float(model.tau), "optimizer_step_cuda_event_ms" if device.type == "cuda" else "optimizer_step_wall_ms": step_ms,
            }
            append_jsonl(log_path, record)
            if better or step % 10 == 0 or step == max_steps:
                atomic_save(latest_path, checkpoint_payload(model, heads, optimizer, pair_rng, step=step, best_key=best_key, best_step=best_step, patience_left=patience_left, patience_reference=patience_reference, metadata=metadata))
            if step >= min_stop and eligible and patience_left <= 0:
                successful_step = step
                atomic_save(latest_path, checkpoint_payload(model, heads, optimizer, pair_rng, step=step, best_key=best_key, best_step=best_step, patience_left=patience_left, patience_reference=patience_reference, metadata=metadata))
                break
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if not best_path.is_file():
            raise RuntimeError("training finished without an eligible best checkpoint")
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"]); heads.load_state_dict(best_state["task_heads"])
        best_eval, predictions = evaluate(stage, model_id, model, heads, tensors, int(best_state["step"]), classes, collect_diagnostics=True, with_grad=False)
        # A small probability-aware evaluation: logits are transient and never persisted.
        best_metrics = {part: best_eval[part] for part in ("train", "validation")}
        pred_path = output_dir / "train_validation_argmax.npz"
        np.savez_compressed(pred_path, **prediction_npz_payload(tensors, predictions))
        ablations = ablation_eval(stage, model_id, model, heads, tensors, int(best_state["step"]), classes) if phase == "confirmation" and stage == "S" and (model_id in {"S1", "S4", "S5"}) else {}
        elapsed = time.perf_counter() - started
        params_backbone = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_heads = sum(p.numel() for p in heads.parameters() if p.requires_grad)
        padding_count = 20 * 128 if stage == "S" and not spec["fusion"] else 0
        effective_params_heads = params_heads - padding_count
        success = {
            **attempt, "status": "success", "ended_utc": utc_now(), "best_step": int(best_state["step"]),
            "best_key": best_state["best_key"], "best_val_correct": int(best_state["best_key"][0]) if stage == "R" else int(best_eval["validation"]["main"]["correct"]),
            "best_val_accuracy": float(best_eval["validation"]["main"]["accuracy"]),
            "best_val_macro_f1": float(best_eval["validation"]["main"]["macro_f1"]),
            "best_utility": validation_utility(best_eval["validation"]["main"]),
            "best_metrics": best_metrics, "best_model_diagnostics": best_eval["diagnostics"],
            "best_ablation_metrics": ablations, "steps_completed": int(successful_step),
            "wall_seconds_train_validation_checkpoint_diagnostic": elapsed,
            "optimizer_step_ms_median_all": float(np.median(step_times)) if step_times else None,
            "optimizer_step_ms_median_steps2_30": float(np.median(step_times[1:min(len(step_times), 30)])) if len(step_times) > 1 else None,
            "peak_normal_allocated_bytes": normal_peak_alloc, "peak_normal_reserved_bytes": normal_peak_reserved,
            "peak_diagnostic_allocated_bytes": diagnostic_peak_alloc, "peak_diagnostic_reserved_bytes": diagnostic_peak_reserved,
            "active_backbone_parameters": params_backbone, "active_task_head_parameters": effective_params_heads,
            "total_active_parameters": params_backbone + effective_params_heads, "total_instantiated_trainable_parameters": params_backbone + params_heads,
            "inactive_padding_head_weight_count": padding_count,
            "test_evaluation": False, "test_head_forward_events": 0, "evaluated_partitions": ["train", "validation"],
            "latest_checkpoint": str(latest_path.resolve()), "best_checkpoint": str(best_path.resolve()),
            "prediction_file": str(pred_path.resolve()), "step_log": str(log_path.resolve()), "diagnostic_log": str(diag_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path), "latest_checkpoint_sha256": sha256_file(latest_path),
            "prediction_sha256": sha256_file(pred_path), "step_log_sha256": sha256_file(log_path), "diagnostic_log_sha256": sha256_file(diag_path) if diag_path.exists() else None,
            "best_checkpoint_bytes": best_path.stat().st_size, "prediction_bytes": pred_path.stat().st_size,
        }
        registry = read_registry(); registry["attempts"][run_id] = success; save_registry(registry); append_jsonl(RESULTS_PATH, success)
        return success
    except BaseException as exc:
        state = "invalid" if isinstance(exc, FloatingPointError) else "failed"
        failure = {
            **attempt, "status": state, "ended_utc": utc_now(), "error": repr(exc), "traceback": traceback.format_exc(),
            "latest_checkpoint": str(latest_path.resolve()) if latest_path.is_file() else None,
            "latest_checkpoint_sha256": sha256_file(latest_path) if latest_path.is_file() else None,
            "best_checkpoint": str(best_path.resolve()) if best_path.is_file() else None,
            "best_checkpoint_sha256": sha256_file(best_path) if best_path.is_file() else None,
            "step_log": str(log_path.resolve()) if log_path.is_file() else None,
            "step_log_sha256": sha256_file(log_path) if log_path.is_file() else None,
            "diagnostic_log": str(diag_path.resolve()) if diag_path.is_file() else None,
            "diagnostic_log_sha256": sha256_file(diag_path) if diag_path.is_file() else None,
            "test_evaluation": False, "test_head_forward_events": 0,
        }
        registry = read_registry(); registry["attempts"][run_id] = failure; save_registry(registry); append_jsonl(RESULTS_PATH, failure)
        raise


def assert_smoke_done() -> None:
    path = REPORT_ROOT / "smoke_summary.json"
    if not path.is_file():
        raise RuntimeError("formal training is locked until all six train-only smoke runs pass")
    result = json.loads(path.read_text(encoding="utf-8"))
    if len(result["runs"]) != 6 or any(row["status"] != "success" or row["steps"] != 3 for row in result["runs"]):
        raise RuntimeError("smoke record is incomplete/failed")


def assert_profile_done() -> None:
    path = REPORT_ROOT / "optimization_profile.json"
    if not path.is_file():
        raise RuntimeError("run the matched reference/optimized profile before formal training")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if len(profile.get("runs", [])) != 6 or any(row.get("status") != "success" for row in profile["runs"]):
        raise RuntimeError("matched optimization profile is incomplete")


def phase_smoke(device: torch.device) -> None:
    lock = verify_frozen_source()
    rows = []
    for dataset in DATASETS:
        tensors = load_run_data(dataset, device, lock)
        for model_id, stage in (("R-M0", "R"), ("S5", "S")):
            seed = 20_261_006
            started = time.perf_counter(); step_times = []
            row = {"dataset": dataset, "model_id": model_id, "seed": seed, "planned_steps": 3, "status": "running", "test_evaluation": False, "selection_eligible": False, "train_labels_only": True}
            try:
                if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
                _, model, heads = build_modules(model_id, stage, dataset, tensors["info"], seed, device)
                optimizer = optimizer_groups(model, heads, 0.001, 0.001)
                pair_rng = torch.Generator(device=device).manual_seed(seed + 31_415)
                for step in (1, 2, 3):
                    model.train(); heads.train(); model.set_epoch(step); optimizer.zero_grad(set_to_none=True)
                    t0 = time.perf_counter()
                    z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], pair_generator=pair_rng, compute_ssl=True, collect_diagnostics=False)
                    loss, _, _ = train_only_loss("M0", z, extras, heads, tensors["y"], tensors["train_idx"], path_weight=0.05, rec_weight=0.01, pair_weight=0.01)
                    if not torch.isfinite(loss): raise FloatingPointError(f"smoke loss is nonfinite at step {step}")
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
                    if not torch.isfinite(norm): raise FloatingPointError(f"smoke gradient is nonfinite at step {step}")
                    optimizer.step(); step_times.append((time.perf_counter() - t0) * 1000.0)
                row.update({"status": "success", "steps": 3, "loss_last": float(loss.detach().cpu()), "last_pre_clip_gradient_norm": float(norm.detach().cpu()), "step_median_ms": float(np.median(step_times)), "wall_seconds": time.perf_counter() - started, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None})
            except Exception as exc:
                row.update({"status": "failed", "steps": len(step_times), "error": repr(exc), "wall_seconds": time.perf_counter() - started})
            finally:
                rows.append(row)
                if "model" in locals(): del model
                if "heads" in locals(): del heads
                if "optimizer" in locals(): del optimizer
                if "pair_rng" in locals(): del pair_rng
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            print(f"smoke {dataset}/{model_id}: {row['status']} {row.get('steps', 0)}/3 steps", flush=True)
            write_json(REPORT_ROOT / "smoke_summary.json", {"runs": rows, "all_train_only": True, "selection_eligible": False, "test_evaluation": False})
        del tensors; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


def phase_profile(device: torch.device) -> None:
    """Matched short train-only timings for uncached reference vs optimized path."""
    lock = verify_frozen_source()
    rows = []
    for dataset in DATASETS:
        tensors = load_run_data(dataset, device, lock)
        for optimized in (False, True):
            seed = 20_261_007
            path_name = "optimized" if optimized else "reference_no_cache_edge_space_relation"
            row = {"dataset": dataset, "path": path_name, "warmup_steps": 2, "timed_steps": 3, "diagnostic_flag": False, "same_model_seed": seed, "status": "running", "train_labels_only": True, "selection_eligible": False, "test_evaluation": False}
            try:
                if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
                _, model, heads = build_modules("R-M0", "R", dataset, tensors["info"], seed, device)
                model.optimize_cache = optimized
                optimizer = optimizer_groups(model, heads, 0.001, 0.001)
                pair_rng = torch.Generator(device=device).manual_seed(seed + 31_415)
                samples = []
                for step in range(1, 6):
                    model.train(); heads.train(); model.set_epoch(step)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                    else:
                        start_wall = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], pair_generator=pair_rng, compute_ssl=True, collect_diagnostics=False)
                    loss, _, _ = train_only_loss("M0", z, extras, heads, tensors["y"], tensors["train_idx"], path_weight=0.05, rec_weight=0.01, pair_weight=0.01)
                    if not torch.isfinite(loss): raise FloatingPointError(f"{dataset} {path_name} loss is nonfinite at step {step}")
                    loss.backward(); gradient_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
                    if not torch.isfinite(gradient_norm): raise FloatingPointError(f"{dataset} {path_name} gradient is nonfinite at step {step}")
                    optimizer.step()
                    if device.type == "cuda":
                        end.record(); torch.cuda.synchronize(device); elapsed = float(start.elapsed_time(end))
                    else:
                        elapsed = float((time.perf_counter() - start_wall) * 1000)
                    if step > 2: samples.append(elapsed)
                row.update({"status": "success", "step_median_ms": float(np.median(samples)), "step_samples_ms": samples, "last_pre_clip_gradient_norm": float(gradient_norm.detach().cpu()), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None})
            except Exception as exc:
                row.update({"status": "failed", "error": repr(exc)})
            finally:
                rows.append(row)
                if "model" in locals(): del model
                if "heads" in locals(): del heads
                if "optimizer" in locals(): del optimizer
                if "pair_rng" in locals(): del pair_rng
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
                write_json(REPORT_ROOT / "optimization_profile.json", {"runs": rows, "comparison": {}, "diagnostics": False, "selection_eligible": False, "test_evaluation": False})
        del tensors; gc.collect()
        print(f"profile {dataset}: matched reference/optimized steps measured", flush=True)
    table = {}
    for dataset in DATASETS:
        ref = next(row for row in rows if row["dataset"] == dataset and row["path"].startswith("reference"))
        opt = next(row for row in rows if row["dataset"] == dataset and row["path"] == "optimized")
        if ref["status"] == opt["status"] == "success":
            table[dataset] = {"reference_step_median_ms": ref["step_median_ms"], "optimized_step_median_ms": opt["step_median_ms"], "reference_over_optimized_ratio": ref["step_median_ms"] / opt["step_median_ms"] if opt["step_median_ms"] else None, "reference_peak_allocated_bytes": ref["peak_allocated_bytes"], "optimized_peak_allocated_bytes": opt["peak_allocated_bytes"]}
    write_json(REPORT_ROOT / "optimization_profile.json", {"runs": rows, "comparison": table, "warmup_and_timing_boundary": "both implementations: 2 warmup updates excluded, then 3 CUDA-event (or same CPU wall) timings cover forward, train-only loss, backward, one global clip and optimizer.step; diagnostics, validation and checkpointing excluded", "diagnostics": False, "selection_eligible": False, "test_evaluation": False})


def grid(stage: str):
    spec_lrs, spec_wds = (R_LR, R_WD) if stage == "R" else (S_LR, S_WD)
    return [(i, lr, wd) for i, (lr, wd) in enumerate((lr, wd) for lr in spec_lrs for wd in spec_wds)]


def phase_calibrate(stage: str, device: torch.device, retry_failed: bool, no_resume: bool) -> None:
    lock = verify_frozen_source(); assert_smoke_done(); assert_profile_done()
    selection_file = REPORT_ROOT / ("r_selection_lock.json" if stage == "R" else "s_selection_lock.json")
    if selection_file.exists(): raise RuntimeError(f"{stage} calibration is frozen by {selection_file.name}")
    for dataset in DATASETS:
        tensors = load_run_data(dataset, device, lock)
        for model_id in R_MODELS if stage == "R" else S_MODELS:
            for cfg_index, lr, wd in grid(stage):
                try:
                    row = run_cell(stage=stage, phase="calibration", dataset=dataset, model_id=model_id, seed=R_CAL_SEED if stage == "R" else S_CAL_SEED, lr=lr, wd=wd, config_index=cfg_index, tensors=tensors, lock=lock, device=device, retry_failed=retry_failed, no_resume=no_resume)
                    print(f"{row.get('status')} calibration {stage} {dataset}/{model_id}/cfg{cfg_index:02d}", flush=True)
                except Exception as exc:
                    print(f"failed calibration {stage} {dataset}/{model_id}/cfg{cfg_index:02d}: {exc!r}", flush=True)
        del tensors; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


def phase_lock_selection(stage: str) -> None:
    lock = verify_frozen_source()
    target = REPORT_ROOT / ("r_selection_lock.json" if stage == "R" else "s_selection_lock.json")
    if target.exists(): raise RuntimeError("selection lock already exists and is immutable")
    rows = [r for r in read_results() if r.get("stage") == stage and r.get("phase") == "calibration" and r.get("status") == "success"]
    selected, groups = {}, (R_MODELS if stage == "R" else S_MODELS)
    for dataset in DATASETS:
        for model_id in groups:
            candidates = [r for r in rows if r["dataset"] == dataset and r["model_id"] == model_id]
            by_config: dict[int, dict[str, Any]] = {}
            for row in candidates:
                index = int(row["config_index"])
                if index not in by_config or row["ended_utc"] > by_config[index]["ended_utc"]:
                    by_config[index] = row
            if len(by_config) != 6:
                raise RuntimeError(f"cannot freeze {dataset}/{model_id}: {len(by_config)}/6 successful calibration configs")
            winner = max(by_config.values(), key=lambda r: _selection_compare_key(stage, r))
            selected[f"{dataset}/{model_id}"] = {
                "lr": winner["lr"], "weight_decay": winner["weight_decay"], "config_index": winner["config_index"],
                "calibration_attempt_id": winner["attempt_id"], "best_step": winner["best_step"],
                "best_utility": winner["best_utility"], "best_val_accuracy": winner["best_val_accuracy"],
                "best_val_macro_f1": winner["best_val_macro_f1"],
                "best_val_correct": winner["best_val_correct"], "selection_key": _selection_compare_key(stage, winner),
                "candidate_attempt_ids": [by_config[index]["attempt_id"] for index in sorted(by_config)],
            }
    payload = {
        "study": STUDY, "stage": stage, "created_utc": utc_now(), "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "training_source_commit": lock["training_source_commit"],
        "rule": "R=(val correct desc, unrounded fixed-20 Macro-F1 desc, best step asc, config index asc); S=(unrounded U desc, Acc desc, Macro-F1 desc, best step asc, config index asc)",
        "selection_uses_calibration_only": True, "selected": selected,
    }
    write_json(target, payload)
    write_json(REPORT_ROOT / ("r_calibration_selection.json" if stage == "R" else "s_calibration_selection.json"), {"successful_calibrations": rows, "selected": selected})
    print(f"wrote {target}; selected {len(selected)}/6 dataset-model cells")


def phase_confirm(stage: str, device: torch.device, selection_sha: str, retry_failed: bool, no_resume: bool) -> None:
    lock = verify_frozen_source(); assert_smoke_done(); assert_profile_done()
    selection_path = REPORT_ROOT / ("r_selection_lock.json" if stage == "R" else "s_selection_lock.json")
    if not selection_path.is_file(): raise FileNotFoundError(selection_path)
    actual_sha = sha256_file(selection_path)
    if not selection_sha or selection_sha != actual_sha: raise RuntimeError("confirmation requires the explicit committed selection-lock SHA256")
    if not git("ls-files", "--error-unmatch", str(selection_path.relative_to(ROOT))):
        raise RuntimeError("selection lock is not tracked")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["protocol_lock_sha256"] != sha256_file(PROTOCOL_LOCK): raise RuntimeError("selection lock protocol mismatch")
    models, seeds = (R_MODELS, R_CONFIRM) if stage == "R" else (S_MODELS, S_CONFIRM)
    for dataset in DATASETS:
        tensors = load_run_data(dataset, device, lock)
        for model_id in models:
            chosen = selection["selected"][f"{dataset}/{model_id}"]
            for seed in seeds:
                try:
                    row = run_cell(stage=stage, phase="confirmation", dataset=dataset, model_id=model_id, seed=seed, lr=float(chosen["lr"]), wd=float(chosen["weight_decay"]), config_index=int(chosen["config_index"]), tensors=tensors, lock=lock, device=device, retry_failed=retry_failed, no_resume=no_resume)
                    print(f"{row.get('status')} confirmation {stage} {dataset}/{model_id}/seed{seed}", flush=True)
                except Exception as exc:
                    print(f"failed confirmation {stage} {dataset}/{model_id}/seed{seed}: {exc!r}", flush=True)
        del tensors; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


def latest_successes(stage: str, phase: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in read_results():
        if row.get("stage") != stage or row.get("phase") != phase or row.get("status") != "success":
            continue
        key = (row["dataset"], row["model_id"], row["training_seed"], row.get("config_index"))
        if key not in grouped or row.get("ended_utc", "") > grouped[key].get("ended_utc", ""):
            grouped[key] = row
    return list(grouped.values())


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    return (None, None) if not values else (float(np.mean(values)), float(np.std(values, ddof=0)))


def _main_metric(row: dict[str, Any], partition: str) -> dict[str, Any]:
    return row["best_metrics"][partition]["main"]


def phase_report() -> None:
    lock = verify_frozen_source()
    r_cal, r_conf = latest_successes("R", "calibration"), latest_successes("R", "confirmation")
    s_cal, s_conf = latest_successes("S", "calibration"), latest_successes("S", "confirmation")
    r_selection_path, s_selection_path = REPORT_ROOT / "r_selection_lock.json", REPORT_ROOT / "s_selection_lock.json"
    if not r_selection_path.is_file() or not s_selection_path.is_file():
        raise FileNotFoundError("report requires committed R and S selection locks")
    r_selection, s_selection = (json.loads(path.read_text(encoding="utf-8")) for path in (r_selection_path, s_selection_path))
    report_summary: dict[str, Any] = {
        "study": STUDY, "training_source_commit": lock["training_source_commit"],
        "test_evaluation": False, "real_lp_runs": 0, "excluded_nc_formal_runs": 0,
        "formal_attempt_status_counts": {}, "successful_unique_cells": {
            "R_calibration": len(r_cal), "R_confirmation": len(r_conf), "S_calibration": len(s_cal), "S_confirmation": len(s_conf),
        },
        "R_confirmation": {}, "S_confirmation": {},
    }
    registry = read_registry()
    for row in registry["attempts"].values():
        if row.get("phase") in {"calibration", "confirmation"}:
            status_key = f"{row.get('stage')}_{row.get('phase')}_{row.get('status')}"
            report_summary["formal_attempt_status_counts"][status_key] = report_summary["formal_attempt_status_counts"].get(status_key, 0) + 1

    lines = [
        "# MAG collaboration adaptation NC v2",
        "",
        "本轮是修复既有 MAG 协作骨干并检验 typed same/cross value（V）与保留自身语义的非线性融合（F）的 Movies、Grocery、Reddit-S train/validation 原型。只使用 split seed42 和冻结 20 类 Macro-F1 schema。没有测试集任务头前向/指标/选择，没有真实 LP，也没有 Toys 或 ele-fashion 正式训练。三个开发图的 validation 已在既往工作中被使用；seeds 只作描述性比较，不作显著性宣称。",
        "",
        f"- 修复复验 R：calibration {len(r_cal)}/36，confirmation {len(r_conf)}/18。",
        f"- 组合/对照 S：calibration {len(s_cal)}/108，confirmation {len(s_conf)}/54。",
        f"- 所有 attempt 的状态计数：`{json.dumps(report_summary['formal_attempt_status_counts'], ensure_ascii=False, sort_keys=True)}`。重试单独登记；以下估计每个格只取最近一次成功 attempt。",
        "- 全矩阵按照冻结规则执行；seed72校准不并入S确认均值。U 只用于本阶段预先声明的开发选择，Acc 与 Macro-F1 仍分别报告。",
        "",
        "## R：初始化/softmax/梯度诊断修复后的复验",
        "",
        "v1旧 M0/M1 与本轮R在同数据集、同训练 seed 上配对。R 是从头训练，且初始化与计算修复同时变化；差异不能唯一归因于初始化。旧 v1 原报告、锁与 checkpoint 保持原样。",
        "",
        "| Dataset | Model | old val Acc/F1 (seed63/64/65) | R val Acc/F1 (seed63/64/65) | mean Δ Acc/F1 (pp) | R选中LR/WD | best steps |",
        "|---|---|---|---|---:|---|---|",
    ]
    old_rows = [json.loads(line) for line in (ROOT / "reports/collab_nc_pilot_v1/run_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    old_conf = [r for r in old_rows if r.get("phase") == "confirmation" and r.get("status") == "success" and r.get("variant") in {"M0", "M1"}]
    r_pairs = []
    for dataset in DATASETS:
        for model_id in R_MODELS:
            old_variant = model_id[-2:]
            old_map = {(r["dataset"], r["training_seed"]): r for r in old_conf if r["dataset"] == dataset and r["variant"] == old_variant}
            new_group = sorted([r for r in r_conf if r["dataset"] == dataset and r["model_id"] == model_id], key=lambda r: r["training_seed"])
            old_disp, new_disp, deltas = [], [], []
            for row in new_group:
                old = old_map.get((dataset, row["training_seed"]))
                if old is None:
                    continue
                om, nm = old["best_metrics"]["validation"]["main"], _main_metric(row, "validation")
                old_disp.append(f"{100*om['accuracy']:.2f}/{100*om['macro_f1']:.2f}")
                new_disp.append(f"{100*nm['accuracy']:.2f}/{100*nm['macro_f1']:.2f}")
                deltas.append({"dataset": dataset, "model_id": model_id, "seed": row["training_seed"], "delta_accuracy_pp": 100*(nm["accuracy"]-om["accuracy"]), "delta_macro_f1_pp": 100*(nm["macro_f1"]-om["macro_f1"])})
            sel = r_selection["selected"].get(f"{dataset}/{model_id}", {})
            da, _ = mean_sd([d["delta_accuracy_pp"] for d in deltas]); df, _ = mean_sd([d["delta_macro_f1_pp"] for d in deltas])
            lines.append(f"| {dataset} | {model_id} | {', '.join(old_disp) or '缺失'} | {', '.join(new_disp) or '缺失'} | {da:+.2f}/{df:+.2f} | {sel.get('lr')}/{sel.get('weight_decay')} | {','.join(str(r['best_step']) for r in new_group)} |")
            r_pairs.extend(deltas)
    report_summary["R_confirmation"] = {"runs": r_conf, "old_new_matched_differences": r_pairs}

    lines += ["", "## S：六方案验证结果", "", "确认表为 seeds73/74/75 的均值 ± population SD（ddof=0）；seed72 calibration 不纳入。Acc/F1/U 单位为百分比。train−val 显示过拟合差距的描述量。", "", "| Dataset | Model | Val Acc | Val Macro-F1 | Val U | Train Acc/F1 | Train−val Acc/F1 pp | LR/WD | best step | step median ms | peak normal alloc GiB | active backbone/head params |"]
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for dataset in DATASETS:
        for model_id in S_MODELS:
            group = sorted([r for r in s_conf if r["dataset"] == dataset and r["model_id"] == model_id], key=lambda r: r["training_seed"])
            selection = s_selection["selected"].get(f"{dataset}/{model_id}", {})
            val_a = [float(_main_metric(r, "validation")["accuracy"]) for r in group]
            val_f = [float(_main_metric(r, "validation")["macro_f1"]) for r in group]
            val_u = [0.5*(a+f) for a, f in zip(val_a,val_f)]
            tr_a = [float(_main_metric(r,"train")["accuracy"]) for r in group]
            tr_f = [float(_main_metric(r,"train")["macro_f1"]) for r in group]
            ma, sa = mean_sd(val_a); mf, sf = mean_sd(val_f); mu, su = mean_sd(val_u)
            mta, _ = mean_sd(tr_a); mtf, _ = mean_sd(tr_f)
            acc_gap, f1_gap = (mta-ma, mtf-mf) if ma is not None else (None, None)
            med, _ = mean_sd([float(r["optimizer_step_ms_median_steps2_30"]) for r in group if r.get("optimizer_step_ms_median_steps2_30") is not None])
            alloc, _ = mean_sd([r["peak_normal_allocated_bytes"]/1024**3 for r in group])
            params_b = group[0]["active_backbone_parameters"] if group else None
            params_h = group[0]["active_task_head_parameters"] if group else None
            lines.append(f"| {dataset} | {model_id} | {100*ma:.2f} ± {100*sa:.2f} | {100*mf:.2f} ± {100*sf:.2f} | {100*mu:.2f} ± {100*su:.2f} | {100*mta:.2f}/{100*mtf:.2f} | {100*acc_gap:+.2f}/{100*f1_gap:+.2f} | {selection.get('lr')}/{selection.get('weight_decay')} | {','.join(str(r['best_step']) for r in group)} | {med:.2f} | {alloc:.2f} | {params_b}/{params_h} |" if group else f"| {dataset} | {model_id} | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |")
            report_summary["S_confirmation"].setdefault(dataset, {})[model_id] = {"n": len(group), "val_acc_mean": ma, "val_acc_sd_ddof0": sa, "val_macro_f1_mean": mf, "val_macro_f1_sd_ddof0": sf, "val_U_mean": mu, "val_U_sd_ddof0": su, "train_acc_mean": mta, "train_macro_f1_mean": mtf, "selected_lr": selection.get("lr"), "selected_weight_decay": selection.get("weight_decay"), "best_steps": [r["best_step"] for r in group]}

    lines += ["", "## S逐seed确认数据", "", "| Dataset | Model | Seed | LR/WD | train Acc/F1 | val Acc/F1/U | train/val CE | best | wall min | median step ms | peak normal/diagnostic alloc GiB |"]
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(s_conf, key=lambda r: (DATASETS.index(r["dataset"]), S_MODELS.index(r["model_id"]), r["training_seed"])):
        tm, vm = _main_metric(row,"train"), _main_metric(row,"validation")
        lines.append(f"| {row['dataset']} | {row['model_id']} | {row['training_seed']} | {row['lr']}/{row['weight_decay']} | {100*tm['accuracy']:.2f}/{100*tm['macro_f1']:.2f} | {100*vm['accuracy']:.2f}/{100*vm['macro_f1']:.2f}/{100*validation_utility(vm):.2f} | {tm['cross_entropy']:.4f}/{vm['cross_entropy']:.4f} | {row['best_step']} | {row['wall_seconds_train_validation_checkpoint_diagnostic']/60:.2f} | {row['optimizer_step_ms_median_steps2_30']:.2f} | {row['peak_normal_allocated_bytes']/1024**3:.2f}/{row['peak_diagnostic_allocated_bytes']/1024**3:.2f} |")
    report_summary["S_confirmation"]["runs"] = s_conf

    contrasts = (("S5-S0","S5","S0"),("S5-S1","S5","S1"),("S5-S2","S5","S2"),("S1-S0","S1","S0"),("S3-S2","S3","S2"),("S4-S2","S4","S2"))
    paired: dict[str, Any] = {"confirmation": {}, "calibration_interaction_by_common_config": {}}
    cell_map = {(r["dataset"],r["model_id"],r["training_seed"]):r for r in s_conf}
    lines += ["", "## 配对差值与组合交互", "", "确认配对按同图同训练 seed 做差，单位百分点；先列每图三seed均值，再给三图等权均值。描述性区间，不宣称显著性。", "", "| Contrast | Dataset | ΔAcc pp | ΔMacro-F1 pp | ΔU pp | seed-level ΔAcc / ΔF1 pp |"]
    lines.append("|---|---|---:|---:|---:|---|")
    for label, a_id, b_id in contrasts:
        paired["confirmation"][label] = {}
        all_a=[]; all_f=[]; all_u=[]
        for dataset in DATASETS:
            diffs=[]
            for seed in S_CONFIRM:
                arow,brow=cell_map.get((dataset,a_id,seed)),cell_map.get((dataset,b_id,seed))
                if not arow or not brow: continue
                am,bm=_main_metric(arow,"validation"),_main_metric(brow,"validation")
                d={"seed":seed,"delta_accuracy_pp":100*(am["accuracy"]-bm["accuracy"]),"delta_macro_f1_pp":100*(am["macro_f1"]-bm["macro_f1"]),"delta_utility_pp":100*(validation_utility(am)-validation_utility(bm))}
                diffs.append(d); all_a.append(d["delta_accuracy_pp"]); all_f.append(d["delta_macro_f1_pp"]); all_u.append(d["delta_utility_pp"])
            ma,_=mean_sd([d["delta_accuracy_pp"] for d in diffs]); mf,_=mean_sd([d["delta_macro_f1_pp"] for d in diffs]); mu,_=mean_sd([d["delta_utility_pp"] for d in diffs])
            seed_deltas = ", ".join("{:+.2f}/{:+.2f}".format(d["delta_accuracy_pp"], d["delta_macro_f1_pp"]) for d in diffs)
            lines.append(f"| {label} | {dataset} | {ma:+.2f} | {mf:+.2f} | {mu:+.2f} | {seed_deltas} |")
            paired["confirmation"][label][dataset]=diffs
        ma,_=mean_sd(all_a); mf,_=mean_sd(all_f); mu,_=mean_sd(all_u)
        lines.append(f"| {label} | equal-graph mean | {ma:+.2f} | {mf:+.2f} | {mu:+.2f} | n={len(all_a)} pairs |")
    interactions=[]
    for dataset in DATASETS:
        for seed in S_CONFIRM:
            required=[cell_map.get((dataset,k,seed)) for k in ("S5","S3","S4","S2")]
            if not all(required): continue
            vals=[_main_metric(r,"validation") for r in required]
            interactions.append({"dataset":dataset,"seed":seed,"delta_accuracy_pp":100*(vals[0]["accuracy"]-vals[1]["accuracy"]-vals[2]["accuracy"]+vals[3]["accuracy"]),"delta_macro_f1_pp":100*(vals[0]["macro_f1"]-vals[1]["macro_f1"]-vals[2]["macro_f1"]+vals[3]["macro_f1"]),"delta_utility_pp":100*(validation_utility(vals[0])-validation_utility(vals[1])-validation_utility(vals[2])+validation_utility(vals[3]))})
    for dataset in DATASETS:
        subset=[r for r in interactions if r["dataset"]==dataset]
        ma,_=mean_sd([r["delta_accuracy_pp"] for r in subset]); mf,_=mean_sd([r["delta_macro_f1_pp"] for r in subset]); mu,_=mean_sd([r["delta_utility_pp"] for r in subset])
        seed_deltas = ", ".join("{:+.2f}/{:+.2f}".format(r["delta_accuracy_pp"], r["delta_macro_f1_pp"]) for r in subset)
        lines.append(f"| I=S5−S3−S4+S2 | {dataset} | {ma:+.2f} | {mf:+.2f} | {mu:+.2f} | {seed_deltas} |")
    im_a,_=mean_sd([r["delta_accuracy_pp"] for r in interactions]); im_f,_=mean_sd([r["delta_macro_f1_pp"] for r in interactions]); im_u,_=mean_sd([r["delta_utility_pp"] for r in interactions])
    lines.append(f"| I=S5−S3−S4+S2 | equal-graph mean | {im_a:+.2f} | {im_f:+.2f} | {im_u:+.2f} | n={len(interactions)} pairs |")
    cal_map={(r["dataset"],r["model_id"],int(r["config_index"])):r for r in s_cal}
    cal_interactions=[]
    for dataset in DATASETS:
        for config_index,_,_ in grid("S"):
            rr=[cal_map.get((dataset,k,config_index)) for k in ("S5","S3","S4","S2")]
            if all(rr):
                vv=[_main_metric(r,"validation") for r in rr]
                cal_interactions.append({"dataset":dataset,"config_index":config_index,"delta_accuracy_pp":100*(vv[0]["accuracy"]-vv[1]["accuracy"]-vv[2]["accuracy"]+vv[3]["accuracy"]),"delta_macro_f1_pp":100*(vv[0]["macro_f1"]-vv[1]["macro_f1"]-vv[2]["macro_f1"]+vv[3]["macro_f1"])})
    paired["confirmation"]["S5-S3-S4+S2_interaction"] = interactions
    paired["calibration_interaction_by_common_config"] = cal_interactions
    write_json(REPORT_ROOT / "paired_differences.json", paired)
    interaction_payload={"confirmation_selected_recipe_descriptive_interaction":interactions,"calibration_seed72_common_config_interaction":cal_interactions,"note":"Confirmation may use different selected LR/WD; these are descriptive interactions of selected recipes, not unique causal module synergy. Calibration interaction pairs identical config_index without affecting confirmation selection."}
    write_json(REPORT_ROOT / "interaction_summary.json", interaction_payload)

    lines += [
        "", "## 冻结 checkpoint 概率指标与同头屏蔽诊断", "",
        "确认 checkpoint 的 NLL（等同CE）、固定20类多类 Brier（先对每节点的 sum_c(p_c−onehot_c)^2，再对节点求平均）以及错误预测平均 max confidence 都从 train/validation 即时 logits 计算，logits未保存。S1/S4/S5 屏蔽 r、S5 屏蔽 private/common/all graph blocks 使用同一个已训练 main head；这是冻结表示的屏蔽诊断，不是重训消融或因果贡献。",
        "",
        "| Dataset | Model | Part | NLL | Brier sum-class | wrong mean max-confidence | same-head mask ΔAcc pp: r/private/common/all-graph |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in sorted([r for r in s_conf if r["model_id"] in {"S1","S4","S5"}], key=lambda r:(DATASETS.index(r["dataset"]),S_MODELS.index(r["model_id"]),r["training_seed"])):
        for part in ("train","validation"):
            main=_main_metric(row,part); ab=row.get("best_ablation_metrics",{})
            base_acc=main["accuracy"]
            dd=[]
            for name in ("mask_r","mask_private","mask_common","mask_all_graph"):
                mm=ab.get(name,{}).get(part)
                dd.append("" if mm is None else f"{100*(mm['accuracy']-base_acc):+.2f}")
            lines.append(f"| {row['dataset']} | {row['model_id']} s{row['training_seed']} | {part} | {main.get('nll',main['cross_entropy']):.4f} | {main.get('brier_sum_over_classes',float('nan')):.4f} | {main.get('wrong_prediction_mean_max_confidence')} | {'/'.join(dd)} |")

    lines += [
        "", "## Reference 与优化路径成本", "",
        "下表来自额外 train-only 计算 profile，不属于 formal matrix。每图/reference 与 optimized 使用相同 R-M0/head seed、两步 warmup后测三步；diagnostics=false，CUDA event 同一计时边界（forward/loss/backward/clip/step）。该值只描述此次固定硬件与图上的实现开销，不外推通用加速比例。",
        "",
        "| Dataset | reference median step ms | optimized median step ms | observed ratio | reference/optimized peak alloc GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    perf_profile_path = REPORT_ROOT / "optimization_profile.json"
    perf_profile = json.loads(perf_profile_path.read_text(encoding="utf-8")) if perf_profile_path.is_file() else {"comparison": {}}
    for dataset in DATASETS:
        row = perf_profile.get("comparison", {}).get(dataset)
        if row:
            lines.append(f"| {dataset} | {row['reference_step_median_ms']:.2f} | {row['optimized_step_median_ms']:.2f} | {row['reference_over_optimized_ratio']:.2f}× | {row['reference_peak_allocated_bytes']/1024**3:.2f}/{row['optimized_peak_allocated_bytes']/1024**3:.2f} |")
        else:
            lines.append(f"| {dataset} | missing | missing | missing | missing |")

    lines += [
        "", "## 成本与结构解释", "",
        "active backbone 参数不含 task head；active task-head 参数单列。F-off 的统一768维主头最后128列是固定零 padding，其权重数量单独记作 inactive；它不是容量配平。Typed V 相对对应 M0 V-off 的 backbone 增加16,384个参数；F增加65,664个参数。参数预算不代表计算相同。训练 step 计时只覆盖 forward/loss/backward/clip/optimizer.step；全程 wall time含 train/validation、定期诊断与checkpoint。普通峰值显存与诊断峰值分别记录。",
        "",
        "V在本实现中只加入模态类型与 same/cross 模态关系条件的公共 value 内容变换；原 Q/K/R_attention、π、α与NULL语义保持。same/cross不指真实KG关系类型。F是差分非线性 residual 适配，不是 RoleMAG 原样复现。S3/S4相对S2隔离各自开关；S1控制一般融合的基线收益。S5交互量受LR/WD各自选择影响，只能解释为已选训练配方的描述性交互。",
        "",
        "权重、NULL、范数、梯度和屏蔽变化仅描述此冻结计算系统，不证明真实语义角色、因果效用或信息占比。是否受过拟合、共享φ、竞争gate或优化因素影响，本轮没有因果结论。结果若有seed分歧或接近，应按逐seed表阅读；不会据此增加网格或追加模块。",
        "",
        f"- Protocol: `{DOC_ROOT / 'protocol.md'}`；source/input lock: `{PROTOCOL_LOCK}`。",
        f"- Matched compute profile: `{perf_profile_path}`。",
        f"- Registry/results: `{REPORT_ROOT / 'run_registry.json'}`, `{RESULTS_PATH}`。",
        f"- Ignored raw checkpoints/logs/train-validation argmax: `{OUTPUT_ROOT.resolve()}`。",
        "- 代码与结果检查属于本实现者自审；不是独立审查。",
        "",
    ]
    report_summary["R_confirmation"]["selected"] = r_selection["selected"]
    report_summary["S_confirmation"]["selected"] = s_selection["selected"]
    report_summary["S_confirmation"]["confirmation_rows"] = len(s_conf)
    write_json(REPORT_ROOT / "summary.json", report_summary)
    (REPORT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report for R={len(r_conf)}/18 and S={len(s_conf)}/54 confirmations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/validation-only MAG collaboration adaptation R/S pilot; no test or LP phase exists.")
    parser.add_argument("phase", choices=("prepare", "smoke", "profile", "calibrate-R", "lock-R", "confirm-R", "calibrate-S", "lock-S", "confirm-S", "report"))
    parser.add_argument("--device", default="cuda:0", help="with CUDA_VISIBLE_DEVICES=1, cuda:0 is physical RTX 3090 #1")
    parser.add_argument("--selection-sha256", default=None, help="required exact committed stage selection-lock SHA for confirmation")
    parser.add_argument("--retry-failed", action="store_true", help="create a distinct retry attempt from initialization after a failed cell")
    parser.add_argument("--no-resume", action="store_true", help="do not resume an interrupted cell; requires an explicit retry for a new attempt")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.phase not in {"prepare", "lock-R", "lock-S", "report"}:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the registered smoke/formal phases require the selected CUDA GPU")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("select the authorized idle physical GPU 1 with CUDA_VISIBLE_DEVICES=1; do not change other processes")
    if args.phase == "prepare": phase_prepare()
    elif args.phase == "smoke": phase_smoke(device)
    elif args.phase == "profile": phase_profile(device)
    elif args.phase == "calibrate-R": phase_calibrate("R", device, args.retry_failed, args.no_resume)
    elif args.phase == "lock-R": phase_lock_selection("R")
    elif args.phase == "confirm-R": phase_confirm("R", device, args.selection_sha256 or "", args.retry_failed, args.no_resume)
    elif args.phase == "calibrate-S": phase_calibrate("S", device, args.retry_failed, args.no_resume)
    elif args.phase == "lock-S": phase_lock_selection("S")
    elif args.phase == "confirm-S": phase_confirm("S", device, args.selection_sha256 or "", args.retry_failed, args.no_resume)
    elif args.phase == "report": phase_report()


if __name__ == "__main__":
    main()
