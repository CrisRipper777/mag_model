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
import shutil
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
from src.models.mag_collab_nc import MAGCollabNC, parameter_groups_no_decay
from src.models.mag_prior_context import Model as LegacyPriorModel, canonical_directed_edges
from src.tasks.collab_nc_pilot import CollabTaskHeads, indexed_logits, train_only_loss

STUDY = "mag_collab_nc_pilot_v1"
DATASETS = ("Movies", "Grocery", "Reddit-S")
VARIANTS = ("B0", "B1", "B2", "M0", "M1")
CAL_LR = (0.0003, 0.001, 0.003)
CAL_WD = (0.0001, 0.001)
CONFIRM_SEEDS = (63, 64, 65)
CALIBRATION_SEED = 62
DIAGNOSTIC_SEED = 20261006
STEP_LIMIT = 300
MIN_STEP = 60
PATIENCE = 40
LOCK_PATH = ROOT / "docs/collab_nc_pilot_v1/protocol_lock.json"
DOC_ROOT = ROOT / "docs/collab_nc_pilot_v1"
REPORT_ROOT = ROOT / "reports/collab_nc_pilot_v1"
OUTPUT_ROOT = ROOT / "outputs/collab_nc_pilot_v1"
V1_LOCK = ROOT / "reports/problem_validation_v1/protocol_lock.json"
TRAINING_FILES = (
    "src/models/mag_collab_nc.py",
    "src/models/mag_prior_context.py",
    "src/models/factory.py",
    "src/models/predictor.py",
    "src/tasks/collab_nc_pilot.py",
    "src/tasks/prior_context_pilot.py",
    "src/data/loaders.py",
    "src/data/splits.py",
    "src/data/graph_utils.py",
    "src/data/types.py",
    "src/data/__init__.py",
    "configs/config.yaml",
    "configs/task/nc.yaml",
    "configs/model/mag_collab_nc.yaml",
    "configs/model/mag_prior_context.yaml",
    "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml",
    "configs/dataset/Reddit-S.yaml",
    "scripts/run_collab_nc_pilot_v1.py",
    "scripts/audit_collab_nc_pilot_v1.py",
    "tests/test_collab_nc_pilot.py",
    "docs/collab_nc_pilot_v1/protocol.md",
    "docs/collab_nc_pilot_v1/model_spec.md",
    "docs/collab_nc_pilot_v1/matrix.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.ndarray,)):
        return safe_json(value.tolist())
    if isinstance(value, torch.Tensor):
        return safe_json(value.detach().cpu().tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe_json(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(safe_json(payload)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gpu_inventory_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        rows = []
        for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(fields) != 5:
                continue
            rows.append({"index": int(fields[0]), "name": fields[1], "memory_used_mib": int(fields[2]), "memory_total_mib": int(fields[3]), "utilization_percent": int(fields[4])})
        return {"status": "observed", "queried_utc": utc_now(), "devices": rows}
    except Exception as exc:
        return {"status": "unavailable", "queried_utc": utc_now(), "error": repr(exc)}


def locked_v1() -> dict[str, Any]:
    return json.loads(V1_LOCK.read_text(encoding="utf-8"))


def verify_inputs(dataset: str, lock: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    old = locked_v1()
    schema = old["schemas"][dataset]
    actual: dict[str, dict[str, Any]] = {}
    for raw_path, expected in schema["dataset_input_hashes"].items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"locked input is missing: {path}")
        got = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if got != expected:
            raise RuntimeError(f"locked input mismatch for {path}: expected={expected}, actual={got}")
        actual[str(path)] = got
    split_file = next(path for path in actual if "/MAGB_split/" in path)
    if actual[split_file]["sha256"] != schema["official_split_sha256"]:
        raise RuntimeError(f"seed42 split hash mismatch for {dataset}")
    if lock is not None and actual != lock["datasets"][dataset]["input_hashes"]:
        raise RuntimeError(f"current input identities differ from v1.1 lock for {dataset}")
    return actual


def compose_loader_config(dataset: str):
    schema = locked_v1()["schemas"][dataset]
    split_path = next(p for p in schema["dataset_input_hashes"] if "/MAGB_split/" in p)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "model=mag_collab_nc", "device=cpu"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.evaluate_test = False
    cfg.task.eval_labels = list(schema["macro_f1_label_set"])
    cfg.dataset.nc_split_path = split_path
    cfg.dataset.auto_generate_splits = False
    cfg.dataset.make_undirected = True
    cfg.dataset.add_self_loops = False
    return cfg


def split_diagnostic_ids(data) -> dict[str, list[int]]:
    result = {}
    for offset, (name, ids) in enumerate((("train", data.train_idx), ("validation", data.val_idx))):
        gen = torch.Generator(device="cpu").manual_seed(DIAGNOSTIC_SEED + offset)
        count = min(512, int(ids.numel()))
        selected = ids.cpu().index_select(0, torch.randperm(ids.numel(), generator=gen)[:count]).sort().values
        result[name] = selected.tolist()
    return result


def load_profile(dataset: str, old_lock: dict[str, Any]) -> dict[str, Any]:
    input_hashes = verify_inputs(dataset)
    cfg = compose_loader_config(dataset)
    data = load_mag_data(cfg, "nc", 42)
    schema = old_lock["schemas"][dataset]
    expected_sizes = schema["split_sizes"]
    if int(data.num_classes) != int(schema["num_classes"]) or int(data.num_classes) != 20:
        raise RuntimeError(f"locked class schema changed for {dataset}")
    if [int(data.train_idx.numel()), int(data.val_idx.numel())] != [int(expected_sizes[0]), int(expected_sizes[1])]:
        raise RuntimeError(f"train/validation split sizes changed for {dataset}")
    if data.x_t is None or data.x_i is None:
        raise RuntimeError(f"expected two separated modalities for {dataset}")
    available, availability_stats = MAGCollabNC.infer_availability(data.x_t, data.x_i)
    src, dst = data.edge_index
    keys = src.long() * int(data.num_nodes) + dst.long()
    unique = torch.unique(keys)
    split_hash = next(value["sha256"] for path, value in input_hashes.items() if "/MAGB_split/" in path)
    diag = split_diagnostic_ids(data)
    all_diag = sorted(diag["train"] + diag["validation"])
    b2_edges = canonical_directed_edges(data.edge_index, data.num_nodes)
    loader_edge_set = set(keys.tolist())
    b2_edge_set = set((b2_edges[0] * int(data.num_nodes) + b2_edges[1]).tolist())
    diff_count = len(loader_edge_set.symmetric_difference(b2_edge_set))
    return {
        "dataset": dataset,
        "input_hashes": input_hashes,
        "split_path": str(data.info["nc_split_path"]),
        "split_sha256": split_hash,
        "num_nodes": int(data.num_nodes),
        "loader_edge_count": int(data.edge_index.size(1)),
        "loader_duplicate_directed_edges": int(data.edge_index.size(1) - unique.numel()),
        "loader_self_loops": int((src == dst).sum()),
        "text_dim": int(data.x_t.size(1)),
        "image_dim": int(data.x_i.size(1)),
        "train_count": int(data.train_idx.numel()),
        "validation_count": int(data.val_idx.numel()),
        "num_classes": 20,
        "availability": availability_stats,
        "available_counts": [int(available[:, 0].sum()), int(available[:, 1].sum())],
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "diagnostic_node_ids": diag,
        "diagnostic_node_ids_sha256": hashlib.sha256(canonical_json(diag).encode()).hexdigest(),
        "b2_legacy_canonical_edge_count": int(b2_edges.size(1)),
        "b2_edge_set_symmetric_difference_count": diff_count,
        "b2_edge_hash": hashlib.sha256(b2_edges.t().contiguous().numpy().tobytes()).hexdigest(),
    }


def planned_matrix() -> list[dict[str, Any]]:
    rows = []
    configs = [(lr, wd) for lr in CAL_LR for wd in CAL_WD]
    for dataset in DATASETS:
        for variant in VARIANTS:
            for config_idx, (lr, wd) in enumerate(configs):
                rows.append({"phase": "calibration", "dataset": dataset, "variant": variant, "training_seed": CALIBRATION_SEED, "lr": lr, "weight_decay": wd, "config_index": config_idx})
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in CONFIRM_SEEDS:
                rows.append({"phase": "confirmation", "dataset": dataset, "variant": variant, "training_seed": seed, "selection_locked_lr_wd": True})
    return rows


def phase_prepare() -> None:
    if not (ROOT / "docs/collab_nc_pilot_v1/protocol.md").is_file():
        raise FileNotFoundError("write and review docs/collab_nc_pilot_v1/protocol.md before prepare")
    old = locked_v1()
    dataset_profiles = {dataset: load_profile(dataset, old) for dataset in DATASETS}
    source_hashes = {}
    for relative in TRAINING_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"training source is missing: {relative}")
        source_hashes[relative] = sha256_file(path)
    lock = {
        "study": STUDY,
        "schema_version": 1,
        "branch": git("branch", "--show-current"),
        "base_commit": "0718faefd6674417e57cc37c769c22d5f75a262a",
        "training_source_commit": git("rev-parse", "HEAD"),
        "created_utc": utc_now(),
        "source_hashes": source_hashes,
        "v1_protocol_lock_sha256": sha256_file(V1_LOCK),
        "old_schema": {dataset: {"num_classes": 20, "macro_f1_label_set": old["schemas"][dataset]["macro_f1_label_set"], "split_sizes": old["schemas"][dataset]["split_sizes"]} for dataset in DATASETS},
        "datasets": dataset_profiles,
        "matrix": {"calibration_cells": 90, "confirmation_cells": 45, "standard_total": 135, "lr_order": list(CAL_LR), "weight_decay_order_per_lr": list(CAL_WD), "calibration_seed": CALIBRATION_SEED, "confirmation_seeds": list(CONFIRM_SEEDS)},
        "optimization": {"optimizer": "AdamW", "dropout": 0.2, "max_steps": STEP_LIMIT, "min_early_stop_step": MIN_STEP, "patience": PATIENCE, "grad_clip": 1.0, "dtype": "float32", "validation_every_step": True, "test_evaluation": False, "scheduler": None, "weight_decay_exempt": "all bias and LayerNorm parameters", "selection": ["validation correct count descending", "unrounded fixed-label Macro-F1 descending", "best step ascending", "configuration enumeration ascending"]},
        "seed_mapping": {"model_module_initialization": "BLAKE2b(seed:name) local CPU generators; shared named modules use identical init across variants", "task_head_initialization_seed": "training_seed + 10007", "pair_sampling_seed": "training_seed + 31415", "split_seed": 42, "diagnostic_seed": DIAGNOSTIC_SEED},
        "schedule": {"beta": "0.1 + 0.9*min((s-1)/19,1)", "tau_graph": "2 - min((s-1)/19,1)", "step_origin": 1},
        "losses": {"B0": "main", "B1": "main + .05 path + .01 rec + .01 pair", "B2": "main + .1 mean(legacy source text/image CE)", "M0": "main + .05 path + .01 rec + .01 pair", "M1": "main + .05 path"},
        "availability": "no explicit availability metadata in MAGB loader; mark a row unavailable if it contains non-finite values or is all zero; report both counts and do not interpret zero-row heuristic as authoritative",
        "graph_policy": "M0/B0/B1 use loader edge_index exactly as returned (undirected, no self-loops, no de-duplication or sampling); B2 retains legacy canonicalization and its edge set is recorded per graph",
        "evaluation_policy": "train and validation indices only; labels are selected only by explicit train_idx or val_idx; no test inference, metric, or selection",
        "device_profile": {"selected_physical_gpu": 1, "process_visible_gpu": 0, "gpu_name": "NVIDIA GeForce RTX 3090", "gpu_inventory_snapshot": gpu_inventory_snapshot(), "environment": "yhf_env", "torch": torch.__version__, "torch_cuda": torch.version.cuda, "dgl": package_version("dgl"), "numpy": np.__version__, "sklearn": package_version("scikit-learn"), "python": platform.python_version()},
        "search_extension": {"allowed": True, "only_before_selection_lock": True, "per_graph_max_extra": 10, "study_max_extra": 30, "candidate_lr": [0.0001, 0.01], "requires_pre_registered_evidence": True, "selected": []},
    }
    write_json(LOCK_PATH, lock)
    write_json(DOC_ROOT / "matrix.json", {"standard_runs": planned_matrix(), "planned_standard_count": 135, "extension_count": 0})
    write_json(REPORT_ROOT / "environment_profile.json", lock["device_profile"])
    write_json(REPORT_ROOT / "data_profile.json", dataset_profiles)
    print(f"protocol lock written: {LOCK_PATH}")


def phase_smoke(args) -> None:
    lock = verify_frozen_source()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    smoke_rows = []
    for dataset in DATASETS:
        tensors = load_run_data(dataset, device, lock)
        seed = 2_026_1006
        model, heads = make_modules("M0", dataset, tensors["info"], seed, device)
        optimizer = build_optimizer(model, heads, 0.001, 0.0001)
        pair_rng = torch.Generator(device=device).manual_seed(seed + 31_415)
        losses = []
        step_ms = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(1, 4):
            model.train(); heads.train(); model.set_epoch(step)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()
            z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], pair_generator=pair_rng, compute_ssl=True, diagnostic_node_ids=tensors["diagnostic_nodes"])
            loss, terms, _ = train_only_loss("M0", z, extras, heads, tensors["y"], tensors["train_idx"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
            optimizer.step()
            if device.type == "cuda":
                end_event.record(); torch.cuda.synchronize(device)
                elapsed_ms = float(start_event.elapsed_time(end_event))
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            step_ms.append(elapsed_ms)
            losses.append({"step": step, "total": float(loss.detach().cpu()), **{name: float(value.detach().cpu()) for name, value in terms.items()}})
        row = {
            "dataset": dataset, "variant": "M0", "steps": 3, "seed": seed,
            "train_indices_only_for_labels": True, "validation_evaluated": False, "test_evaluated": False,
            "losses": losses, "cuda_event_step_ms": step_ms,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "num_nodes": lock["datasets"][dataset]["num_nodes"], "num_edges": lock["datasets"][dataset]["loader_edge_count"],
            "created_utc": utc_now(),
        }
        smoke_rows.append(row)
        path = OUTPUT_ROOT / "smoke" / f"{dataset}_M0_3step.json"
        write_json(path, row)
        print(f"smoke {dataset}: step_ms={step_ms} peak_alloc={row['peak_allocated_bytes']}", flush=True)
    write_json(REPORT_ROOT / "smoke_summary.json", {"rows": smoke_rows, "used_for_selection": False, "test_evaluation": False})


def verify_frozen_source() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["study"] != STUDY:
        raise RuntimeError("unexpected protocol lock identity")
    if sha256_file(V1_LOCK) != lock["v1_protocol_lock_sha256"]:
        raise RuntimeError("historical v1 protocol lock changed")
    actual = {relative: sha256_file(ROOT / relative) for relative in lock["source_hashes"]}
    if actual != lock["source_hashes"]:
        changed = [key for key in actual if actual[key] != lock["source_hashes"].get(key)]
        raise RuntimeError(f"frozen training source hash mismatch: {changed}")
    for dataset in DATASETS:
        verify_inputs(dataset, lock)
    return lock


def make_config(dataset: str, variant: str, seed: int, lr: float, wd: float, device: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        model_config = "mag_prior_context" if variant == "B2" else "mag_collab_nc"
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", f"model={model_config}", f"seed={seed}", f"device={device}"])
    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.task.evaluate_test = False
    cfg.task.lr = float(lr)
    cfg.task.weight_decay = float(wd)
    cfg.task.epochs = STEP_LIMIT
    cfg.task.patience = PATIENCE
    cfg.task.early_stop_min_epoch = MIN_STEP
    cfg.task.early_stop_min_delta = 0.0
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.scheduler = None
    cfg.task.loss.aux_weight = 0.0
    cfg.dataset.nc_split_path = next(p for p in locked_v1()["schemas"][dataset]["dataset_input_hashes"] if "/MAGB_split/" in p)
    cfg.dataset.auto_generate_splits = False
    cfg.dataset.make_undirected = True
    cfg.dataset.add_self_loops = False
    cfg.task.eval_labels = list(locked_v1()["schemas"][dataset]["macro_f1_label_set"])
    if variant == "B2":
        cfg.model.variant = "R1"
        cfg.model.width = 144
        cfg.model.dropout = 0.2
    else:
        cfg.model.variant = variant
        cfg.model.dropout = 0.2
        cfg.model.out_dim = 640
    if bool(cfg.task.evaluate_test):
        raise RuntimeError("task.evaluate_test must remain false for this pilot")
    return cfg


def load_run_data(dataset: str, device: torch.device, lock: dict[str, Any]):
    cfg = make_config(dataset, "M1", CALIBRATION_SEED, CAL_LR[0], CAL_WD[0], "cpu")
    data = load_mag_data(cfg, "nc", 42)
    profile = lock["datasets"][dataset]
    if int(data.num_nodes) != profile["num_nodes"] or int(data.edge_index.size(1)) != profile["loader_edge_count"]:
        raise RuntimeError(f"locked graph size changed for {dataset}")
    x = data.x.contiguous().to(device=device, dtype=torch.float32)
    x_t_cpu, x_i_cpu = data.x_t, data.x_i
    availability_cpu, fallback = MAGCollabNC.infer_availability(x_t_cpu, x_i_cpu)
    if fallback != profile["availability"] or [int(availability_cpu[:, 0].sum()), int(availability_cpu[:, 1].sum())] != profile["available_counts"]:
        raise RuntimeError(f"locked availability profile changed for {dataset}")
    available = availability_cpu.to(device)
    edges = data.edge_index.to(device=device, dtype=torch.long).contiguous()
    train_idx = data.train_idx.to(device=device, dtype=torch.long)
    val_idx = data.val_idx.to(device=device, dtype=torch.long)
    # Copy only the labels explicitly permitted for train and validation evaluation.
    y_safe = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
    y_safe[train_idx] = data.y.index_select(0, data.train_idx).to(device)
    y_safe[val_idx] = data.y.index_select(0, data.val_idx).to(device)
    train_diag = torch.tensor(profile["diagnostic_node_ids"]["train"], dtype=torch.long, device=device)
    val_diag = torch.tensor(profile["diagnostic_node_ids"]["validation"], dtype=torch.long, device=device)
    diagnostic = torch.cat((train_diag, val_diag))
    info = {"input_dim": int(data.input_dim), "text_dim": int(data.x_t.size(1)), "visual_dim": int(data.x_i.size(1)), "num_nodes": int(data.num_nodes)}
    b2_edges = canonical_directed_edges(data.edge_index, data.num_nodes).to(device)
    return {"x": x, "edges": edges, "b2_edges": b2_edges, "available": available, "train_idx": train_idx, "val_idx": val_idx, "y": y_safe, "diagnostic_nodes": diagnostic, "diagnostic_train": train_diag, "diagnostic_val": val_diag, "info": info, "fallback": fallback}


def make_modules(variant: str, dataset: str, dataset_info: dict[str, Any], seed: int, device: torch.device):
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if variant == "B2":
        cfg = make_config(dataset, "B2", seed, CAL_LR[0], CAL_WD[0], "cpu")
        model = LegacyPriorModel(cfg, dataset_info)
    else:
        model = MAGCollabNC(dataset_info, variant, dropout=0.2, init_seed=seed)
    model_rng_cpu = torch.random.get_rng_state()
    model_rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed) + 10_007)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 10_007)
    heads = CollabTaskHeads(variant, 20, output_dim=int(getattr(model, "out_dim", 640)))
    torch.random.set_rng_state(model_rng_cpu)
    if model_rng_cuda is not None:
        torch.cuda.set_rng_state_all(model_rng_cuda)
    model.to(device)
    heads.to(device)
    return model, heads


def training_groups(variant: str, model: nn.Module, heads: nn.Module):
    groups: dict[str, list[nn.Parameter]] = {}
    if variant == "B2":
        groups["legacy_backbone"] = list(model.parameters())
        groups["task_main"] = list(heads.legacy.main.parameters())
        groups["task_paths"] = list(heads.legacy.source_text.parameters()) + list(heads.legacy.source_image.parameters())
        return groups
    groups["projection_and_own_fusion"] = []
    for name in ("p_text", "p_image", "p_norm", "e_mlp", "e_norm"):
        groups["projection_and_own_fusion"].extend(getattr(model, name).parameters())
    if variant in {"B1", "M0", "M1"}:
        groups["initial_common_private"] = []
        for name in ("common_adapter", "common_stem", "common_stem_norm", "private_stem", "private_stem_norm"):
            groups["initial_common_private"].extend(getattr(model, name).parameters())
        if variant == "B1":
            groups["mean_graph"] = list(model.b1_norms.parameters()) + list(model.b1_updates.parameters())
        else:
            groups["private_paths"] = list(model.layers[0].private.parameters()) + list(model.layers[1].private.parameters())
            groups["common_paths"] = list(model.layers[0].common.parameters()) + list(model.layers[1].common.parameters())
        groups["ssl"] = list(model.decoder_text.parameters()) + list(model.decoder_image.parameters()) + list(model.pair_projection.parameters()) if variant in {"B1", "M0"} else []
    groups["task_main"] = list(heads.main.parameters())
    groups["task_paths"] = list(heads.path_heads.parameters()) if variant != "B0" else []
    return groups


def grad_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
    terms = [parameter.grad.detach().float().pow(2).sum() for parameter in parameters if parameter.grad is not None]
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).sum().sqrt()


def param_vector(parameters: list[nn.Parameter]) -> torch.Tensor:
    if not parameters:
        return torch.empty((0,))
    return torch.cat([parameter.detach().float().reshape(-1) for parameter in parameters]).clone()


def build_optimizer(model: nn.Module, heads: nn.Module, lr: float, wd: float):
    params = parameter_groups_no_decay(model, [heads])
    for group in params:
        if group.get("weight_decay") == 0.0:
            group["weight_decay"] = 0.0
        else:
            group["weight_decay"] = float(wd)
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(wd))


def metric_block(labels: torch.Tensor, logits: torch.Tensor, class_ids: list[int]) -> dict[str, Any]:
    y = labels.detach().cpu().long().numpy()
    pred = logits.detach().argmax(-1).cpu().long().numpy()
    ce = float(F.cross_entropy(logits, labels).detach().cpu())
    correct = int((logits.detach().argmax(-1) == labels).sum().item())
    count = int(labels.numel())
    return {
        "count": count,
        "correct": correct,
        "accuracy": correct / max(count, 1),
        "macro_f1": float(f1_score(y, pred, labels=class_ids, average="macro", zero_division=0)),
        "cross_entropy": ce,
        "predictions": pred,
    }


def evaluate_variant(variant: str, model, heads, tensors: dict[str, Any], step: int, class_ids: list[int]):
    model.eval()
    heads.eval()
    if hasattr(model, "set_epoch"):
        model.set_epoch(step)
    with torch.no_grad():
        if variant == "B2":
            z, _, _, _, extras = model(tensors["x"], tensors["b2_edges"])
        else:
            z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], diagnostic_node_ids=tensors["diagnostic_nodes"])
        result = {}
        saved = {}
        for partition in ("train", "validation"):
            idx = tensors["train_idx"] if partition == "train" else tensors["val_idx"]
            logits, masks = indexed_logits(variant, z, extras, heads, idx)
            y = tensors["y"].index_select(0, idx)
            part = {}
            for name, values in logits.items():
                mask = masks.get(name, torch.ones_like(y, dtype=torch.bool))
                masked_logits = values[mask]
                masked_y = y[mask]
                if masked_y.numel() == 0:
                    part[name] = None
                    continue
                part[name] = metric_block(masked_y, masked_logits, class_ids)
                saved[f"{partition}_{name}"] = part[name].pop("predictions")
            result[partition] = part
        result["diagnostics"] = extras.get("diagnostics", {})
        result["availability"] = extras.get("availability")
    return result, saved


def rng_state(pair_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "pair_generator": pair_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], pair_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    pair_generator.set_state(state["pair_generator"])


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def checkpoint_payload(model, heads, optimizer, pair_rng, *, step, best_key, best_step, patience_left, run_id, phase, metadata):
    return {
        "model": model.state_dict(),
        "task_heads": heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "best_key": best_key,
        "best_step": int(best_step),
        "patience_left": int(patience_left),
        "run_id": run_id,
        "phase": phase,
        "resolved_run_metadata": metadata,
        "rng": rng_state(pair_rng),
    }


def prediction_payload(tensors, eval_result, saved_preds, variant: str):
    fields: dict[str, np.ndarray] = {
        "train_ids": tensors["train_idx"].detach().cpu().numpy(),
        "validation_ids": tensors["val_idx"].detach().cpu().numpy(),
    }
    for key, value in saved_preds.items():
        fields[key + "_pred"] = np.asarray(value, dtype=np.int64)
    fields["train_available"] = tensors["available"].index_select(0, tensors["train_idx"]).detach().cpu().numpy().astype(np.uint8)
    fields["validation_available"] = tensors["available"].index_select(0, tensors["val_idx"]).detach().cpu().numpy().astype(np.uint8)
    return fields


def run_identifier(phase: str, dataset: str, variant: str, seed: int, lr: float, wd: float, config_index: int | None = None) -> str:
    lr_s = f"{lr:.7g}".replace(".", "p")
    wd_s = f"{wd:.7g}".replace(".", "p")
    middle = "" if config_index is None else f"_cfg{config_index:02d}"
    return f"{phase}_{dataset}_{variant}_seed{seed}{middle}_lr{lr_s}_wd{wd_s}"


def load_registry() -> dict[str, Any]:
    path = REPORT_ROOT / "run_registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"study": STUDY, "attempts": {}}


def save_registry(registry: dict[str, Any]) -> None:
    write_json(REPORT_ROOT / "run_registry.json", registry)


def result_index() -> dict[str, dict[str, Any]]:
    result_path = REPORT_ROOT / "run_results.jsonl"
    found = {}
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            found[row["attempt_id"]] = row
    return found


def save_diagnostic_row(path: Path, payload: dict[str, Any]) -> None:
    append_jsonl(path, payload)


def run_cell(
    *, phase: str, dataset: str, variant: str, seed: int, lr: float, wd: float, config_index: int | None,
    tensors: dict[str, Any], lock: dict[str, Any], device: torch.device, resume: bool = True, retry_failed: bool = False,
) -> dict[str, Any]:
    base_id = run_identifier(phase, dataset, variant, seed, lr, wd, config_index)
    registry = load_registry()
    attempts = registry["attempts"]
    prior_ids = [key for key in attempts if key == base_id or key.startswith(base_id + "_retry")]
    prior_ids.sort(key=lambda key: (0 if key == base_id else int(key.rsplit("_retry", 1)[1])))
    previous = attempts[prior_ids[-1]] if prior_ids else None
    prior_results = result_index()
    for key in reversed(prior_ids):
        if attempts[key].get("status") == "success":
            # A crash between the atomic registry update and JSONL append must
            # not turn an already completed cell into a KeyError.
            return prior_results.get(key, attempts[key])
    if previous and previous.get("status") == "failed" and not retry_failed:
        return prior_results.get(prior_ids[-1], previous)
    if previous and previous.get("status") == "invalid":
        return prior_results.get(prior_ids[-1], previous)
    resume_same_attempt = False
    if previous and previous.get("status") == "running":
        prior_latest = Path(previous.get("output_dir", "")) / "latest.pt"
        if prior_latest.is_file() and resume:
            run_id = prior_ids[-1]
            output_dir = Path(previous["output_dir"])
            resume_same_attempt = True
        else:
            previous["status"] = "interrupted"
            previous["ended_utc"] = utc_now()
            previous["error"] = "process stopped before a resumable latest checkpoint was available"
            registry["attempts"][prior_ids[-1]] = previous
            save_registry(registry)
            append_jsonl(REPORT_ROOT / "run_results.jsonl", previous)
            prior_results[prior_ids[-1]] = previous
    if not resume_same_attempt:
        if previous and previous.get("status") in {"interrupted", "failed"} and not retry_failed:
            return prior_results.get(prior_ids[-1], previous)
        run_id = base_id if not prior_ids else f"{base_id}_retry{len(prior_ids)}"
        output_dir = OUTPUT_ROOT / dataset / variant / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if previous and retry_failed and not resume_same_attempt:
        old_dir = Path(previous["output_dir"])
        for filename in ("latest.pt", "best.pt"):
            old_checkpoint = old_dir / filename
            if old_checkpoint.exists():
                shutil.copy2(old_checkpoint, output_dir / filename)
    if resume_same_attempt:
        attempt = dict(previous)
        attempt["resumed_utc"] = utc_now()
        attempt["resume_count"] = int(previous.get("resume_count", 0)) + 1
        attempt["status"] = "running"
    else:
        attempt = {
        "attempt_id": run_id, "phase": phase, "dataset": dataset, "variant": variant,
        "training_seed": seed, "lr": lr, "weight_decay": wd, "config_index": config_index,
        "status": "running", "started_utc": utc_now(), "device": str(device),
        "source_commit": lock["training_source_commit"], "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "input_hashes": lock["datasets"][dataset]["input_hashes"], "output_dir": str(output_dir),
        "retry_of": None if previous is None else previous.get("attempt_id"),
        }
    registry["attempts"][run_id] = attempt
    save_registry(registry)
    log_path = output_dir / "steps.jsonl"
    diag_path = output_dir / "diagnostics.jsonl"
    if log_path.exists() and not (output_dir / "latest.pt").exists():
        log_path.unlink()
    classes = list(lock["old_schema"][dataset]["macro_f1_label_set"])
    data_info = tensors["info"]
    resolved_cfg = make_config(dataset, variant, seed, lr, wd, str(device))
    resolved_cfg_dict = OmegaConf.to_container(resolved_cfg, resolve=True)
    model, heads = make_modules(variant, dataset, data_info, seed, device)
    optimizer = build_optimizer(model, heads, lr, wd)
    pair_rng = torch.Generator(device=device).manual_seed(int(seed) + 31_415)
    all_params = [p for p in list(model.parameters()) + list(heads.parameters()) if p.requires_grad]
    groups = training_groups(variant, model, heads)
    start_step = 1
    best_key = None
    best_step = 0
    best_correct = -1
    patience_left = PATIENCE
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    checkpoint_metadata = {
        "attempt_id": run_id, "phase": phase, "dataset": dataset, "variant": variant,
        "training_seed": seed, "split_seed": 42, "lr": lr, "weight_decay": wd,
        "config_index": config_index, "evaluate_test": False,
        "protocol_lock_sha256": sha256_file(LOCK_PATH), "training_source_commit": lock["training_source_commit"],
        "input_hashes": lock["datasets"][dataset]["input_hashes"], "resolved_config": resolved_cfg_dict,
    }
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        best_state = torch.load(best_path, map_location=device, weights_only=False) if best_path.exists() else None
        # A best checkpoint is written before its step record/latest checkpoint.
        # If interruption lands in that small window, use the newer complete
        # checkpoint (which also carries optimizer and RNG state) as the resume
        # point instead of resuming from an older optimizer state.
        if best_state is not None and int(best_state["step"]) > int(state["step"]):
            state = best_state
        model.load_state_dict(state["model"])
        heads.load_state_dict(state["task_heads"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_key = state["best_key"]
        best_step = int(state["best_step"])
        patience_left = int(state["patience_left"])
        best_correct = int(state["best_key"][0]) if state.get("best_key") else -1
        restore_rng_state(state["rng"], pair_rng)
        # Keep records after the latest atomic checkpoint as interruption evidence,
        # then remove them from the resumed canonical step trace to avoid duplicates.
        if log_path.exists():
            log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            orphan_rows = [row for row in log_rows if int(row["step"]) > int(state["step"])]
            if orphan_rows:
                orphan_path = output_dir / "orphaned_uncheckpointed_steps.jsonl"
                for row in orphan_rows:
                    append_jsonl(orphan_path, {**row, "status": "after_latest_checkpoint_not_restored"})
                with log_path.open("w", encoding="utf-8") as stream:
                    for row in log_rows:
                        if int(row["step"]) <= int(state["step"]):
                            stream.write(canonical_json(row) + "\n")
        if diag_path.exists():
            diag_rows = [json.loads(line) for line in diag_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            retained = [row for row in diag_rows if int(row["step"]) <= int(state["step"])]
            if len(retained) != len(diag_rows):
                with diag_path.open("w", encoding="utf-8") as stream:
                    for row in retained:
                        stream.write(canonical_json(row) + "\n")
    started = time.perf_counter()
    train_step_ms: list[float] = []
    peak_allocated = 0
    peak_reserved = 0
    diagnostic_peak_allocated = 0
    diagnostic_peak_reserved = 0
    last_eval_result = None
    last_predictions = None
    last_step_loss: dict[str, float] = {}
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(start_step, STEP_LIMIT + 1):
            model.train()
            heads.train()
            if hasattr(model, "set_epoch"):
                model.set_epoch(step)
            param_before = {name: param_vector(params) for name, params in groups.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                ev_start, ev_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev_start.record()
            else:
                t_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if variant == "B2":
                z, _, _, _, extras = model(tensors["x"], tensors["b2_edges"])
            else:
                z, _, _, _, extras = model(tensors["x"], tensors["edges"], tensors["available"], pair_generator=pair_rng, compute_ssl=True, diagnostic_node_ids=tensors["diagnostic_nodes"])
            total_loss, loss_terms, _ = train_only_loss(variant, z, extras, heads, tensors["y"], tensors["train_idx"])
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite objective at step {step}")
            total_loss.backward()
            global_pre_clip = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            if not torch.isfinite(global_pre_clip):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            clip_coef = min(1.0, 1.0 / (float(global_pre_clip.detach().cpu()) + 1e-6))
            optimizer.step()
            if device.type == "cuda":
                ev_end.record()
                torch.cuda.synchronize(device)
                step_ms = float(ev_start.elapsed_time(ev_end))
            else:
                step_ms = (time.perf_counter() - t_start) * 1000.0
            train_step_ms.append(step_ms)
            last_step_loss = {name: float(value.detach().cpu()) for name, value in loss_terms.items()}
            last_step_loss["total"] = float(total_loss.detach().cpu())

            eval_result, predictions = evaluate_variant(variant, model, heads, tensors, step, classes)
            if device.type == "cuda":
                peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            val_main = eval_result["validation"]["main"]
            assert val_main is not None
            eligible = step >= (20 if variant in {"B1", "M0", "M1"} else 1)
            key = (int(val_main["correct"]), float(val_main["macro_f1"]), -int(step))
            better = bool(eligible and (best_key is None or key > best_key))
            count_improved = bool(eligible and int(val_main["correct"]) > best_correct)
            if eligible:
                if best_key is None:
                    patience_left = PATIENCE
                elif count_improved:
                    patience_left = PATIENCE
                else:
                    patience_left -= 1
                best_correct = max(best_correct, int(val_main["correct"]))
            if better:
                best_key, best_step = key, step
                payload = checkpoint_payload(model, heads, optimizer, pair_rng, step=step, best_key=best_key, best_step=best_step, patience_left=patience_left, run_id=run_id, phase=phase, metadata=checkpoint_metadata)
                atomic_torch_save(best_path, payload)
                last_eval_result, last_predictions = eval_result, predictions
            reasons = []
            if step in (1, 10, 30):
                reasons.append(f"step_{step}")
            if better:
                reasons.append("best_checkpoint")
            if reasons:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                grad_norms = {name: grad_norm(params) for name, params in groups.items()}
                diagnostics = {}
                for name, params in groups.items():
                    before = param_before[name]
                    after = param_vector(params)
                    update = after - before
                    diagnostics[name] = {
                        "pre_parameter_norm": float(before.norm().cpu()) if before.numel() else None,
                        "post_parameter_norm": float(after.norm().cpu()) if after.numel() else None,
                        "pre_clip_gradient_norm": float((grad_norms[name] / max(clip_coef, 1e-12)).cpu()) if params else None,
                        "post_clip_gradient_norm": float((grad_norms[name] * clip_coef).cpu()) if params else None,
                        "update_norm": float(update.norm().cpu()) if update.numel() else None,
                        "relative_update_norm": float(update.norm().cpu() / before.norm().clamp_min(1e-12)) if update.numel() else None,
                    }
                diag_row = {"step": step, "reason": reasons, "beta": float(getattr(model, "beta", 1.0)), "tau_graph": float(getattr(model, "tau", 1.0)), "grad_global_preclip_norm": float(global_pre_clip.detach().cpu()), "clip_coefficient": clip_coef, "groups": diagnostics, "model": eval_result.get("diagnostics", {})}
                if variant != "B2":
                    main_weight = heads.main.weight.detach()
                    diag_row["main_readout_block_norms"] = {
                        "p_text": float(main_weight[:, 0:128].norm().cpu()),
                        "p_image": float(main_weight[:, 128:256].norm().cpu()),
                        "e": float(main_weight[:, 256:384].norm().cpu()),
                        "private_text_increment": float(main_weight[:, 384:448].norm().cpu()),
                        "private_image_increment": float(main_weight[:, 448:512].norm().cpu()),
                        "common_text_increment": float(main_weight[:, 512:576].norm().cpu()),
                        "common_image_increment": float(main_weight[:, 576:640].norm().cpu()),
                    }
                else:
                    diag_row["main_readout_block_norms"] = {"legacy_128": float(heads.legacy.main.weight.detach().norm().cpu())}
                save_diagnostic_row(diag_path, diag_row)
                if device.type == "cuda":
                    diagnostic_peak_allocated = max(diagnostic_peak_allocated, int(torch.cuda.max_memory_allocated(device)))
                    diagnostic_peak_reserved = max(diagnostic_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            record = {
                "step": step, "loss": last_step_loss,
                "train": {key: {name: metric.get(name) for name in ("count", "correct", "accuracy", "macro_f1", "cross_entropy")} if metric is not None else None for key, metric in eval_result["train"].items()},
                "validation": {key: {name: metric.get(name) for name in ("count", "correct", "accuracy", "macro_f1", "cross_entropy")} if metric is not None else None for key, metric in eval_result["validation"].items()},
                "eligible": eligible, "best": better, "best_step": best_step, "patience_left": patience_left,
                "beta": float(getattr(model, "beta", 1.0)), "tau_graph": float(getattr(model, "tau", 1.0)),
                "train_step_ms": step_ms,
            }
            append_jsonl(log_path, record)
            if step % 10 == 0 or better or step == STEP_LIMIT:
                payload = checkpoint_payload(model, heads, optimizer, pair_rng, step=step, best_key=best_key, best_step=best_step, patience_left=patience_left, run_id=run_id, phase=phase, metadata=checkpoint_metadata)
                atomic_torch_save(latest_path, payload)
            if step >= MIN_STEP and eligible and patience_left <= 0:
                latest_payload = checkpoint_payload(model, heads, optimizer, pair_rng, step=step, best_key=best_key, best_step=best_step, patience_left=patience_left, run_id=run_id, phase=phase, metadata=checkpoint_metadata)
                atomic_torch_save(latest_path, latest_payload)
                break

        if not best_path.exists():
            raise RuntimeError("run ended without an eligible best checkpoint")
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
        heads.load_state_dict(best_state["task_heads"])
        best_eval, best_predictions = evaluate_variant(variant, model, heads, tensors, best_state["step"], classes)
        pred_path = output_dir / "train_validation_argmax.npz"
        np.savez_compressed(pred_path, **prediction_payload(tensors, best_eval, best_predictions, variant))
        elapsed = time.perf_counter() - started
        successful = {
            **attempt,
            "status": "success", "ended_utc": utc_now(), "best_step": int(best_state["step"]),
            "best_val_correct": int(best_state["best_key"][0]), "best_val_macro_f1": float(best_state["best_key"][1]),
            "best_metrics": {partition: {name: {key: value for key, value in metric.items() if key != "evaluation_indices"} if metric is not None else None for name, metric in row.items()} for partition, row in best_eval.items() if partition in ("train", "validation")},
            "steps_completed": step, "wall_seconds_train_validation_checkpoint": elapsed,
            "optimizer_step_ms_median_steps2_30": float(np.median(train_step_ms[1:min(len(train_step_ms), 30)])) if len(train_step_ms) > 1 else None,
            "optimizer_step_ms_median_all": float(np.median(train_step_ms)) if train_step_ms else None,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "diagnostic_peak_allocated_bytes": diagnostic_peak_allocated, "diagnostic_peak_reserved_bytes": diagnostic_peak_reserved,
            "active_backbone_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "active_task_head_parameters": sum(p.numel() for p in heads.parameters() if p.requires_grad),
            "effective_task_head_parameters": sum(p.numel() for p in heads.parameters() if p.requires_grad) - (256 * 20 if variant == "B0" else 0),
            "zero_pad_main_head_parameters": 256 * 20 if variant == "B0" else 0,
            "latest_checkpoint": str(latest_path.resolve()), "best_checkpoint": str(best_path.resolve()),
            "prediction_file": str(pred_path.resolve()), "step_log": str(log_path.resolve()), "diagnostic_log": str(diag_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path), "prediction_sha256": sha256_file(pred_path),
            "step_log_sha256": sha256_file(log_path), "diagnostic_log_sha256": sha256_file(diag_path) if diag_path.exists() else None,
            "best_checkpoint_bytes": best_path.stat().st_size, "prediction_bytes": pred_path.stat().st_size,
            "test_evaluation": False, "evaluation_partitions": ["train", "validation"],
        }
        registry = load_registry()
        registry["attempts"][run_id] = successful
        save_registry(registry)
        append_jsonl(REPORT_ROOT / "run_results.jsonl", successful)
        return successful
    except BaseException as exc:
        state = "invalid" if isinstance(exc, FloatingPointError) else "failed"
        failure = {**attempt, "status": state, "ended_utc": utc_now(), "error": repr(exc), "traceback": traceback.format_exc(), "latest_checkpoint": str(latest_path.resolve()) if latest_path.exists() else None}
        registry = load_registry()
        registry["attempts"][run_id] = failure
        save_registry(registry)
        append_jsonl(REPORT_ROOT / "run_results.jsonl", failure)
        raise


def read_results() -> list[dict[str, Any]]:
    path = REPORT_ROOT / "run_results.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique = {}
    for row in rows:
        unique[row["attempt_id"]] = row
    return list(unique.values())


def phase_calibrate(args) -> None:
    lock = verify_frozen_source()
    if (REPORT_ROOT / "selection_lock.json").exists():
        raise RuntimeError("calibration is frozen after selection_lock.json; no more search cells are permitted")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    tensors_by_dataset = {}
    configs = [(lr, wd) for lr in CAL_LR for wd in CAL_WD]
    matrix = [{"dataset": d, "variant": v, "lr": lr, "wd": wd, "config_index": i} for d in DATASETS for v in VARIANTS for i, (lr, wd) in enumerate(configs)]
    amendment_path = REPORT_ROOT / "search_amendment.json"
    amendments = []
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment.get("study") != STUDY or amendment.get("protocol_lock_sha256") != sha256_file(LOCK_PATH):
            raise RuntimeError("search amendment does not bind to this protocol lock")
        if amendment.get("dataset") not in DATASETS or amendment.get("lr") not in (0.0001, 0.01):
            raise RuntimeError("search amendment must select one permitted graph and boundary LR")
        if not amendment.get("evidence") or not amendment.get("reason"):
            raise RuntimeError("search amendment requires explicit boundary optimization evidence and reason")
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(amendment_path.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
        if tracked.returncode != 0:
            raise RuntimeError("commit search_amendment.json before running extended calibration")
        amendments = [amendment]
        dataset = amendment["dataset"]
        for variant in VARIANTS:
            for index, wd in enumerate(CAL_WD, start=6):
                matrix.append({"dataset": dataset, "variant": variant, "lr": amendment["lr"], "wd": wd, "config_index": index, "extension": amendment})
    # Keep the enumeration explicitly in LR-major, WD-minor order.
    for entry in matrix:
        dataset, variant = entry["dataset"], entry["variant"]
        if dataset not in tensors_by_dataset:
            tensors_by_dataset[dataset] = load_run_data(dataset, device, lock)
        try:
            outcome = run_cell(phase="calibration", dataset=dataset, variant=variant, seed=CALIBRATION_SEED, lr=entry["lr"], wd=entry["wd"], config_index=entry["config_index"], tensors=tensors_by_dataset[dataset], lock=lock, device=device, resume=not args.no_resume, retry_failed=args.retry_failed)
            print(f"{outcome['status']} {entry['dataset']}/{variant} cfg{entry['config_index']} lr={entry['lr']} wd={entry['wd']}", flush=True)
        except Exception as exc:
            print(f"failed {entry['dataset']}/{variant} cfg{entry['config_index']}: {exc}", flush=True)
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()


def phase_lock_selection() -> None:
    lock = verify_frozen_source()
    rows = [r for r in read_results() if r.get("phase") == "calibration"]
    amendment_path = REPORT_ROOT / "search_amendment.json"
    amendments = [json.loads(amendment_path.read_text(encoding="utf-8"))] if amendment_path.exists() else []
    expected_by_dataset = {dataset: set(range(6)) for dataset in DATASETS}
    for amendment in amendments:
        expected_by_dataset[amendment["dataset"]].update((6, 7))
    selected, groups = {}, {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            all_candidates = [r for r in rows if r.get("dataset") == dataset and r.get("variant") == variant]
            successful = [r for r in all_candidates if r.get("status") == "success"]
            by_config: dict[int, dict[str, Any]] = {}
            for candidate in successful:
                index = int(candidate["config_index"])
                if index not in by_config or candidate.get("ended_utc", "") > by_config[index].get("ended_utc", ""):
                    by_config[index] = candidate
            candidates = [by_config[index] for index in sorted(by_config)]
            expected_indices = expected_by_dataset[dataset]
            if set(by_config) != expected_indices:
                groups[f"{dataset}/{variant}"] = {"status": "unselectable", "successful_config_indices": sorted(by_config), "required_config_indices": sorted(expected_indices), "attempts": [{"attempt_id": r["attempt_id"], "config_index": r.get("config_index"), "status": r.get("status")} for r in all_candidates]}
                continue
            winner = max(candidates, key=lambda r: (r["best_val_correct"], r["best_val_macro_f1"], -r["best_step"], -r["config_index"]))
            selected[f"{dataset}/{variant}"] = {
                "status": "selected", "lr": winner["lr"], "weight_decay": winner["weight_decay"],
                "config_index": winner["config_index"], "calibration_attempt_id": winner["attempt_id"],
                "best_step": winner["best_step"], "best_val_correct": winner["best_val_correct"],
                "best_val_macro_f1": winner["best_val_macro_f1"], "best_checkpoint": winner["best_checkpoint"],
                "best_checkpoint_sha256": winner["best_checkpoint_sha256"],
                "all_candidates": [{"attempt_id": r["attempt_id"], "config_index": r.get("config_index"), "lr": r.get("lr"), "weight_decay": r.get("weight_decay"), "status": r.get("status"), "best_val_correct": r.get("best_val_correct"), "best_val_macro_f1": r.get("best_val_macro_f1"), "best_step": r.get("best_step")} for r in sorted(all_candidates, key=lambda r: (r.get("config_index", 99), r["attempt_id"]))],
            }
    selection = {
        "study": STUDY, "created_utc": utc_now(), "training_source_commit": lock["training_source_commit"],
        "protocol_lock_sha256": sha256_file(LOCK_PATH), "rule": "val correct count desc, unrounded fixed-label Macro-F1 desc, best step asc, config enumeration asc",
        "selected": selected, "unselectable": groups,
        "standard_calibration_successes": sum(1 for r in rows if r.get("status") == "success"),
        "standard_calibration_expected": 90,
        "search_amendments": [{**item, "sha256": sha256_file(amendment_path)} for item in amendments],
    }
    write_json(REPORT_ROOT / "selection_lock.json", selection)
    write_json(REPORT_ROOT / "calibration_summary.json", {"rows": rows, "selection": selected, "unselectable": groups})
    print(f"selection lock written; selected {len(selected)}/15 graph/model cells")


def phase_confirm(args) -> None:
    lock = verify_frozen_source()
    selection_path = REPORT_ROOT / "selection_lock.json"
    if not selection_path.exists():
        raise FileNotFoundError("confirmation requires reports/collab_nc_pilot_v1/selection_lock.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["protocol_lock_sha256"] != sha256_file(LOCK_PATH):
        raise RuntimeError("selection lock was made from a different protocol lock")
    expected_selection_sha = args.selection_sha256
    actual_selection_sha = sha256_file(selection_path)
    if not expected_selection_sha:
        raise RuntimeError("confirmation requires an explicit --selection-sha256 frozen commit guard")
    if expected_selection_sha and expected_selection_sha != actual_selection_sha:
        raise RuntimeError("selection lock SHA does not match --selection-sha256")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(selection_path.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode != 0:
        raise RuntimeError("commit the frozen selection lock before launching confirmation")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    tensors_by_dataset = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            selected = selection["selected"].get(f"{dataset}/{variant}")
            if selected is None:
                continue
            if dataset not in tensors_by_dataset:
                tensors_by_dataset[dataset] = load_run_data(dataset, device, lock)
            for seed in CONFIRM_SEEDS:
                try:
                    outcome = run_cell(phase="confirmation", dataset=dataset, variant=variant, seed=seed, lr=selected["lr"], wd=selected["weight_decay"], config_index=selected["config_index"], tensors=tensors_by_dataset[dataset], lock=lock, device=device, resume=not args.no_resume, retry_failed=args.retry_failed)
                    print(f"{outcome['status']} confirmation {dataset}/{variant}/seed{seed}", flush=True)
                except Exception as exc:
                    print(f"failed confirmation {dataset}/{variant}/seed{seed}: {exc}", flush=True)
                    gc.collect()
                    if device.type == "cuda": torch.cuda.empty_cache()


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=0))


def phase_report() -> None:
    rows = read_results()
    selections = json.loads((REPORT_ROOT / "selection_lock.json").read_text(encoding="utf-8")) if (REPORT_ROOT / "selection_lock.json").exists() else {"selected": {}}
    confirmations = [row for row in rows if row.get("phase") == "confirmation" and row.get("status") == "success"]
    calibrations = [row for row in rows if row.get("phase") == "calibration"]
    standard_calibrations = [row for row in calibrations if int(row.get("config_index", -1)) < 6]
    extended_calibrations = [row for row in calibrations if int(row.get("config_index", -1)) >= 6]
    extension_text = "无" if not selections.get("search_amendments") else "; ".join(f"{item['dataset']} 加 LR={item['lr']} × 两个 WD" for item in selections["search_amendments"])
    lines = ["# MAG 协同 NC pilot v1", "", "本轮只覆盖 Movies、Grocery、Reddit-S 的 train/validation。test 推理与指标、真实 LP、Toys/ele-fashion 扩展均未运行。三个图的 validation 已被此前多轮开发使用，seed62 是校准，63–65 是本轮描述性确认；不作显著性或因果宣称。", "", f"- 标准校准：{sum(r.get('status') == 'success' for r in standard_calibrations)}/90 成功；标准失败尝试 {sum(r.get('status') == 'failed' for r in standard_calibrations)}。", f"- 扩展校准：{sum(r.get('status') == 'success' for r in extended_calibrations)} 成功；失败尝试 {sum(r.get('status') == 'failed' for r in extended_calibrations)}。", f"- Confirmation：{len(confirmations)}/45 成功；失败 {sum(r.get('phase') == 'confirmation' and r.get('status') == 'failed' for r in rows)}。", f"- LR 范围扩展：{extension_text}。", f"- 协议锁：`{LOCK_PATH}`；training source commit：`{json.loads(LOCK_PATH.read_text())['training_source_commit']}`。", "", "## Confirmation train / validation 表", "", "Acc 与 Macro-F1 为百分比。均值±SD 使用 seed63/64/65 且 ddof=0；seed62 校准 checkpoint 不并入确认均值。train−val 是均值差，用于描述拟合差距。", "", "| Dataset | Model | Val Acc | Val Macro-F1 | Train Acc | Train Macro-F1 | Train−val Acc/F1 | LR | WD | best steps | step median ms | peak alloc GiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    aggregate = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = sorted([r for r in confirmations if r["dataset"] == dataset and r["variant"] == variant], key=lambda r: r["training_seed"])
            aggregate[f"{dataset}/{variant}"] = group
            if not group:
                lines.append(f"| {dataset} | {variant} | — | — | — | — | — | — | — | — | — |")
                continue
            val_acc = [r["best_metrics"]["validation"]["main"]["accuracy"] * 100 for r in group]
            val_f1 = [r["best_metrics"]["validation"]["main"]["macro_f1"] * 100 for r in group]
            train_acc = [r["best_metrics"]["train"]["main"]["accuracy"] * 100 for r in group]
            train_f1 = [r["best_metrics"]["train"]["main"]["macro_f1"] * 100 for r in group]
            a_m, a_s = mean_sd(val_acc); f_m, f_s = mean_sd(val_f1); t_m, _ = mean_sd(train_acc); tf_m, _ = mean_sd(train_f1)
            gap_acc, gap_f1 = t_m - a_m, tf_m - f_m
            sel = selections["selected"].get(f"{dataset}/{variant}", {})
            times = [r["optimizer_step_ms_median_steps2_30"] for r in group if r.get("optimizer_step_ms_median_steps2_30") is not None]
            mem = [r["peak_allocated_bytes"] / (1024 ** 3) for r in group]
            lines.append(f"| {dataset} | {variant} | {a_m:.2f} ± {a_s:.2f} | {f_m:.2f} ± {f_s:.2f} | {t_m:.2f} | {tf_m:.2f} | {gap_acc:+.2f}/{gap_f1:+.2f} | {sel.get('lr','—')} | {sel.get('weight_decay','—')} | {','.join(str(r['best_step']) for r in group)} | {np.mean(times):.2f} | {np.mean(mem):.3f} |")
    lines += ["", "## Seed-level confirmation", "", "| Dataset | Model | Seed | LR | WD | Train Acc/F1 | Val Acc/F1 | Train/Val CE | Best step | Wall min | Median step ms | Peak alloc/reserved GiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(confirmations, key=lambda x: (DATASETS.index(x["dataset"]), VARIANTS.index(x["variant"]), x["training_seed"])):
        tr = r["best_metrics"]["train"]["main"]; va = r["best_metrics"]["validation"]["main"]
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['training_seed']} | {r['lr']} | {r['weight_decay']} | {tr['accuracy']*100:.2f}/{tr['macro_f1']*100:.2f} | {va['accuracy']*100:.2f}/{va['macro_f1']*100:.2f} | {tr['cross_entropy']:.4f}/{va['cross_entropy']:.4f} | {r['best_step']} | {r['wall_seconds_train_validation_checkpoint']/60:.2f} | {r['optimizer_step_ms_median_steps2_30']:.2f} | {r['peak_allocated_bytes']/1024**3:.3f}/{r['peak_reserved_bytes']/1024**3:.3f} |")
    lines += ["", "## Path-head metrics at each selected checkpoint", "", "来源头只在对应原始 availability 子集评估；表中给出分母。B0 无 path head。", "", "| Dataset | Model | Seed | Partition | Head | n | Acc | Macro-F1 | CE |", "|---|---|---:|---|---|---:|---:|---:|---:|"]
    for r in sorted(confirmations, key=lambda x: (DATASETS.index(x["dataset"]), VARIANTS.index(x["variant"]), x["training_seed"])):
        for partition in ("train", "validation"):
            for head_name, metric in r["best_metrics"][partition].items():
                if head_name == "main" or metric is None:
                    continue
                lines.append(f"| {r['dataset']} | {r['variant']} | {r['training_seed']} | {partition} | {head_name} | {metric['count']} | {metric['accuracy']*100:.2f} | {metric['macro_f1']*100:.2f} | {metric['cross_entropy']:.4f} |")
    lines += ["", "## Active parameters and runtime", "", "参数按活动 backbone 与 NC task heads 分开；B0 的 640-D 主头含零填充槽，表中同时列完整 head 参数与有效 head 参数，不声称容量配平。", "", "| Dataset | Model | Active backbone | Task heads total/effective | Zero-pad head params | Total active | Wall min mean | Normal peak alloc/reserved GiB | Diagnostic peak alloc GiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in confirmations if r["dataset"] == dataset and r["variant"] == variant]
            if not group:
                continue
            backbone = int(np.mean([r["active_backbone_parameters"] for r in group]))
            heads_count = int(np.mean([r["active_task_head_parameters"] for r in group]))
            effective_heads = int(np.mean([r["effective_task_head_parameters"] for r in group]))
            zero_pad = int(np.mean([r["zero_pad_main_head_parameters"] for r in group]))
            wall = np.mean([r["wall_seconds_train_validation_checkpoint"] / 60 for r in group])
            alloc = np.mean([r["peak_allocated_bytes"] / 1024**3 for r in group])
            reserved = np.mean([r["peak_reserved_bytes"] / 1024**3 for r in group])
            diag_alloc = np.mean([r.get("diagnostic_peak_allocated_bytes", 0) / 1024**3 for r in group])
            lines.append(f"| {dataset} | {variant} | {backbone} | {heads_count}/{effective_heads} | {zero_pad} | {backbone+heads_count} | {wall:.2f} | {alloc:.3f}/{reserved:.3f} | {diag_alloc:.3f} |")
    lines += ["", "## Calibration cells", "", "每个候选配置都保留在 `run_results.jsonl` 与 registry；下表列出全部候选的验证正确数/F1/best step，便于复核配置选择。", "", "| Dataset | Model | Config | LR | WD | Val correct | Val Macro-F1 | Best step | Status |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(calibrations, key=lambda x: (DATASETS.index(x["dataset"]), VARIANTS.index(x["variant"]), x.get("config_index", 0))):
        lines.append(f"| {r['dataset']} | {r['variant']} | {r.get('config_index','—')} | {r.get('lr','—')} | {r.get('weight_decay','—')} | {r.get('best_val_correct','—')} | {r.get('best_val_macro_f1','—')} | {r.get('best_step','—')} | {r['status']} |")
    lines += ["", "## Matched descriptive deltas", "", "同数据集、同 confirmation training seed 的候选差，单位百分点；先对三个 seed 求均值，再对三个图等权平均。差值是描述性比较，不作显著性或因果结论。", "", "| Model contrast | Dataset | Mean Δ Acc | Mean Δ Macro-F1 | Seed-level Acc deltas | Seed-level F1 deltas |", "|---|---|---:|---:|---|---|"]
    contrasts = (("M0", "B1"), ("M1", "B1"), ("M0", "B2"), ("M1", "B2"), ("M0", "M1"))
    pair_data = {}
    for left, right in contrasts:
        graph_rows = []
        for dataset in DATASETS:
            left_rows = {r["training_seed"]: r for r in aggregate[f"{dataset}/{left}"]}
            right_rows = {r["training_seed"]: r for r in aggregate[f"{dataset}/{right}"]}
            common = sorted(set(left_rows) & set(right_rows))
            da = [(left_rows[s]["best_metrics"]["validation"]["main"]["accuracy"] - right_rows[s]["best_metrics"]["validation"]["main"]["accuracy"]) * 100 for s in common]
            df = [(left_rows[s]["best_metrics"]["validation"]["main"]["macro_f1"] - right_rows[s]["best_metrics"]["validation"]["main"]["macro_f1"]) * 100 for s in common]
            if da and df:
                graph_rows.append((dataset, float(np.mean(da)), float(np.mean(df)), da, df))
                lines.append(f"| {left}−{right} | {dataset} | {np.mean(da):+.2f} | {np.mean(df):+.2f} | {','.join(f'{x:+.2f}' for x in da)} | {','.join(f'{x:+.2f}' for x in df)} |")
        pair_data[f"{left}-{right}"] = graph_rows
        if graph_rows:
            lines.append(f"| {left}−{right} | Equal-graph mean | {np.mean([r[1] for r in graph_rows]):+.2f} | {np.mean([r[2] for r in graph_rows]):+.2f} | — | — |")
    lines += ["", "## Descriptive reading", ""]
    if len(confirmations) == 45:
        for contrast in ("M0-B1", "M1-B1", "M0-B2", "M1-B2", "M0-M1"):
            values = pair_data.get(contrast, [])
            if values:
                delta_acc = float(np.mean([row[1] for row in values]))
                delta_f1 = float(np.mean([row[2] for row in values]))
                word = "higher" if delta_acc > 0 and delta_f1 > 0 else "mixed" if (delta_acc >= 0) != (delta_f1 >= 0) else "lower"
                lines.append(f"- {contrast}: equal-graph means are {delta_acc:+.2f} pp Acc and {delta_f1:+.2f} pp Macro-F1 ({word} on the two metrics in these three fitted seeds). This is descriptive, not a significance or causal result.")
    else:
        lines.append("Confirmation is incomplete; model ordering is not treated as a completed comparison.")
    lines += ["", "M0−M1 is the task-only versus weak pre-graph reconstruction/pair objective comparison for this implementation. B1/B2 comparisons include their specified operator and, for B2, legacy edge canonicalization. Path-head scores and router values describe trained computations; they do not establish semantic roles, causal utility, or information proportions. Train−validation gaps can describe fit/generalization differences but cannot by themselves identify overfitting, shared-embedding effects, gate competition, or another optimizer cause.", ""]
    lines += ["", "## Scope, cost, and interpretation", "", "所有正式训练使用全图 train-only supervision；全图特征和拓扑按既有传导协议参与 forward。指标只索引固定 train/validation。B2 保持旧 R1 的图 canonicalization，M0/B1 用 loader 原边，因此 B2 对比需连同 edge-set 差异解释。参数量不代表计算量匹配。", "", "路径头、路由/NULL/π与状态范数是计算诊断，不是因果效用、信息占比或真实语义角色。M0/M1 的差异可描述弱图前 SSL 是否与任务目标共存；B0/B1/B2 的差异也不能单独定位某一结构因素。过拟合、共享表示、gate竞争和优化因素没有因果判定。", "", "## Evidence", "", f"- Registry: `{(REPORT_ROOT / 'run_registry.json').resolve()}`", f"- 逐单元结果: `{(REPORT_ROOT / 'run_results.jsonl').resolve()}`", f"- Protocol lock: `{LOCK_PATH.resolve()}`", f"- Raw checkpoints, logs, train/validation argmax: `{OUTPUT_ROOT.resolve()}`", "- 本文由冻结 train/validation argmax 和指标重新汇总；审查性质为自审，不是独立审查。", ""]
    write_json(REPORT_ROOT / "summary.json", {"study": STUDY, "calibration_attempts": calibrations, "confirmation_attempts": confirmations, "pair_deltas": pair_data, "confirmation_expected": 45, "test_evaluation": False, "lp_runs": 0})
    (REPORT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report written: {REPORT_ROOT / 'report.md'}")


def phase_audit() -> None:
    lock = verify_frozen_source()
    registry = load_registry()
    results = read_results()
    issue_list = []
    input_hash_checks = {}
    for dataset in DATASETS:
        try:
            input_hash_checks[dataset] = verify_inputs(dataset, lock)
        except Exception as exc:
            issue_list.append(f"input hash: {dataset}: {exc}")
    prediction_checks = []
    confusion = {}
    data_cache = {}
    selection_path = REPORT_ROOT / "selection_lock.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
    amendments_path = REPORT_ROOT / "search_amendment.json"
    amendments = [json.loads(amendments_path.read_text(encoding="utf-8"))] if amendments_path.exists() else []
    calibration_rows = [row for row in results if row.get("phase") == "calibration"]
    confirmation_rows = [row for row in results if row.get("phase") == "confirmation"]
    selection_checks = {}
    if selection is None:
        issue_list.append("selection lock is missing")
    else:
        required_indices = {dataset: set(range(6)) for dataset in DATASETS}
        for amendment in amendments:
            required_indices[amendment["dataset"]].update((6, 7))
        for dataset in DATASETS:
            for variant in VARIANTS:
                key = f"{dataset}/{variant}"
                cells = [r for r in calibration_rows if r.get("dataset") == dataset and r.get("variant") == variant]
                successful = [r for r in cells if r.get("status") == "success"]
                by_index = {}
                for row in successful:
                    index = int(row["config_index"])
                    if index not in by_index or row.get("ended_utc", "") > by_index[index].get("ended_utc", ""):
                        by_index[index] = row
                chosen = selection.get("selected", {}).get(key)
                valid_indices = set(by_index) == required_indices[dataset]
                expected = max(
                    by_index.values(),
                    key=lambda row: (row["best_val_correct"], row["best_val_macro_f1"], -row["best_step"], -row["config_index"]),
                ) if valid_indices and by_index else None
                matches = bool(
                    expected is not None and chosen is not None
                    and chosen.get("calibration_attempt_id") == expected.get("attempt_id")
                    and int(chosen.get("config_index", -1)) == int(expected["config_index"])
                    and float(chosen.get("lr", -1)) == float(expected["lr"])
                    and float(chosen.get("weight_decay", -1)) == float(expected["weight_decay"])
                    and chosen.get("best_checkpoint_sha256") == expected.get("best_checkpoint_sha256")
                )
                selection_checks[key] = {
                    "required_config_indices": sorted(required_indices[dataset]),
                    "successful_config_indices": sorted(by_index),
                    "selection_matches_recomputed_winner": matches,
                    "expected_attempt_id": expected.get("attempt_id") if expected else None,
                    "locked_attempt_id": chosen.get("calibration_attempt_id") if chosen else None,
                }
                if expected is None:
                    if chosen is not None:
                        issue_list.append(f"selection exists without a complete successful calibration grid: {key}")
                elif not matches:
                    issue_list.append(f"selection lock winner mismatch: {key}")
    if len(calibration_rows) < 90:
        issue_list.append(f"calibration result rows below standard matrix size: {len(calibration_rows)}/90")
    if len(confirmation_rows) < 45:
        issue_list.append(f"confirmation result rows below standard matrix size: {len(confirmation_rows)}/45")
    for phase_name, phase_rows, expected_count in (("calibration", calibration_rows, 90), ("confirmation", confirmation_rows, 45)):
        successful_cell_keys = {
            (row.get("dataset"), row.get("variant"), row.get("training_seed"), row.get("config_index") if phase_name == "calibration" else row.get("lr"), row.get("weight_decay") if phase_name == "calibration" else row.get("config_index"))
            for row in phase_rows if row.get("status") == "success"
        }
        if len(successful_cell_keys) > expected_count and not amendments:
            issue_list.append(f"{phase_name} successes exceed standard matrix without an amendment")
    for row in results:
        if row.get("status") != "success":
            continue
        path = Path(row["prediction_file"])
        if not path.is_file() or sha256_file(path) != row.get("prediction_sha256"):
            issue_list.append(f"prediction hash mismatch: {row['attempt_id']}")
            continue
        dataset = row["dataset"]
        if dataset not in data_cache:
            cfg = compose_loader_config(dataset)
            data_cache[dataset] = load_mag_data(cfg, "nc", 42)
        data = data_cache[dataset]
        schema = locked_v1()["schemas"][dataset]
        class_ids = list(schema["macro_f1_label_set"])
        for artifact_field, digest_field in (("best_checkpoint", "best_checkpoint_sha256"), ("prediction_file", "prediction_sha256"), ("step_log", "step_log_sha256"), ("diagnostic_log", "diagnostic_log_sha256")):
            artifact_path = row.get(artifact_field)
            if artifact_path and (not Path(artifact_path).is_file() or (row.get(digest_field) and sha256_file(Path(artifact_path)) != row[digest_field])):
                issue_list.append(f"{artifact_field} hash mismatch: {row['attempt_id']}")
        with np.load(path, allow_pickle=False) as pred:
            if any("test" in field.lower() for field in pred.files):
                issue_list.append(f"test-named prediction field exists: {row['attempt_id']}")
            for partition, ids_tensor in (("train", data.train_idx), ("validation", data.val_idx)):
                ids = pred[f"{partition}_ids"]
                if not np.array_equal(ids, ids_tensor.cpu().numpy()):
                    issue_list.append(f"evaluation node IDs differ: {row['attempt_id']}/{partition}")
                    continue
                labels = data.y.index_select(0, ids_tensor).cpu().numpy()
                main_pred = pred[f"{partition}_main_pred"]
                matrix = confusion_matrix(labels, main_pred, labels=class_ids)
                acc = float((labels == main_pred).mean())
                f1 = float(f1_score(labels, main_pred, labels=class_ids, average="macro", zero_division=0))
                expected = row["best_metrics"][partition]["main"]
                if abs(acc - expected["accuracy"]) > 1e-12 or abs(f1 - expected["macro_f1"]) > 1e-12:
                    issue_list.append(f"main metric mismatch: {row['attempt_id']}/{partition}")
                confusion[f"{row['attempt_id']}/{partition}/main"] = matrix.tolist()
                prediction_checks.append({"attempt_id": row["attempt_id"], "partition": partition, "head": "main", "count": int(ids.size), "accuracy": acc, "macro_f1": f1, "confusion_matrix": matrix.tolist()})
                for head_name in ("text_private", "image_private", "shared", "text", "image"):
                    field = f"{partition}_{head_name}_pred"
                    if field not in pred.files:
                        continue
                    if head_name == "text_private": mask = pred[f"{partition}_available"][:, 0].astype(bool)
                    elif head_name == "image_private": mask = pred[f"{partition}_available"][:, 1].astype(bool)
                    elif head_name in {"text", "image"}: mask = pred[f"{partition}_available"][:, 0 if head_name == "text" else 1].astype(bool)
                    else: mask = pred[f"{partition}_available"].astype(bool).any(1)
                    if not mask.any():
                        continue
                    p = pred[field][mask]
                    yy = labels[mask]
                    matrix_path = confusion_matrix(yy, p, labels=class_ids)
                    path_f1 = float(f1_score(yy, p, labels=class_ids, average="macro", zero_division=0))
                    metric_name = "text" if head_name == "text" else "image" if head_name == "image" else head_name
                    expected_path = row["best_metrics"][partition].get(metric_name)
                    if expected_path is not None and (abs(float((yy == p).mean()) - expected_path["accuracy"]) > 1e-12 or abs(path_f1 - expected_path["macro_f1"]) > 1e-12):
                        issue_list.append(f"path metric mismatch: {row['attempt_id']}/{partition}/{head_name}")
                    prediction_checks.append({"attempt_id": row["attempt_id"], "partition": partition, "head": head_name, "count": int(mask.sum()), "accuracy": float((yy == p).mean()), "macro_f1": path_f1, "confusion_matrix": matrix_path.tolist()})
        if row.get("evaluation_partitions") != ["train", "validation"] or row.get("test_evaluation") is not False:
            issue_list.append(f"non-train/validation evaluation declaration: {row['attempt_id']}")
        step_log = Path(row.get("step_log", ""))
        if step_log.is_file():
            for line_number, line in enumerate(step_log.read_text(encoding="utf-8").splitlines(), start=1):
                event = json.loads(line)
                if set(event).intersection({"test", "test_metrics", "test_predictions"}):
                    issue_list.append(f"test evaluation event found: {row['attempt_id']} step-log line {line_number}")
                if not {"train", "validation"}.issubset(event):
                    issue_list.append(f"unexpected step metric scope: {row['attempt_id']} line {line_number}")
    counts = {status: sum(1 for value in registry["attempts"].values() if value.get("status") == status) for status in ("success", "failed", "running", "invalid", "interrupted")}
    for attempt_id, attempt in registry["attempts"].items():
        if attempt.get("status") == "success" and attempt_id not in {row.get("attempt_id") for row in results}:
            issue_list.append(f"registry success is absent from results JSONL: {attempt_id}")
        if attempt.get("status") == "success" and (attempt.get("test_evaluation") is not False or attempt.get("evaluation_partitions") != ["train", "validation"]):
            issue_list.append(f"registry success has unexpected evaluation scope: {attempt_id}")
    unique_successful_cells = {}
    for phase_name, phase_rows in (("calibration", calibration_rows), ("confirmation", confirmation_rows)):
        unique_successful_cells[phase_name] = len({
            (row.get("dataset"), row.get("variant"), row.get("training_seed"), row.get("config_index") if phase_name == "calibration" else row.get("lr"), row.get("weight_decay") if phase_name == "calibration" else row.get("config_index"))
            for row in phase_rows if row.get("status") == "success"
        })
    audit = {
        "study": STUDY, "audit_type": "self-audit from saved train/validation argmax; not independent review", "created_utc": utc_now(),
        "training_source_commit": lock["training_source_commit"], "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "input_hash_checks": input_hash_checks, "attempt_status_counts": counts,
        "matrix_counts": {"calibration_attempt_rows": len(calibration_rows), "calibration_successful_unique_cells": unique_successful_cells["calibration"], "confirmation_attempt_rows": len(confirmation_rows), "confirmation_successful_unique_cells": unique_successful_cells["confirmation"], "calibration_expected_standard": 90, "confirmation_expected_standard": 45},
        "selection_recomputation": selection_checks,
        "prediction_metric_checks": prediction_checks, "confusion_matrices": confusion,
        "evaluation_partitions_observed": ["train", "validation"], "test_prediction_or_metric_artifacts": [],
        "issues": issue_list, "passed": len(issue_list) == 0,
    }
    write_json(REPORT_ROOT / "audit_result.json", audit)
    review_lines = [
        "# MAG collaboration NC pilot v1 self-review", "",
        "This is an implementation self-review and saved-prediction audit, not an independent review.", "",
        f"- Audit status: {'PASS' if audit['passed'] else 'ISSUES FOUND'} ({len(issue_list)} issues).",
        f"- Calibration: {len(calibration_rows)} attempt rows, {unique_successful_cells['calibration']} unique successful cells; standard target 90.",
        f"- Confirmation: {len(confirmation_rows)} attempt rows, {unique_successful_cells['confirmation']} unique successful cells; standard target 45.",
        f"- Recomputed selection cells: {sum(row['selection_matches_recomputed_winner'] for row in selection_checks.values())}/{len(selection_checks)} matching cells.",
        f"- Saved train/validation prediction metric checks: {len(prediction_checks)}; test prediction/metric artifacts: {len(audit['test_prediction_or_metric_artifacts'])}.",
        "- Inputs are re-hashed against the frozen v1 lock; train/validation argmax and aggregate confusion matrices are recomputed from the locked schema.",
        "- No real LP, test evaluation, or additional NC datasets are part of this campaign.",
        "", "## Audit issues", "",
    ]
    review_lines.extend([f"- {issue}" for issue in issue_list] or ["- None."])
    review_lines += ["", "## Test record", "", "The source test command and the actual run result are recorded in `test_record.json` after the frozen implementation tests complete.", ""]
    (REPORT_ROOT / "review.md").write_text("\n".join(review_lines), encoding="utf-8")
    print(f"audit {'passed' if audit['passed'] else 'failed'}; issues={len(issue_list)}; argmax rows={len(prediction_checks)}")


def phase_manifest() -> None:
    lock = verify_frozen_source()
    rows = read_results()
    paths = []
    for name, path in (("protocol_lock", LOCK_PATH), ("matrix", DOC_ROOT / "matrix.json"), ("environment_profile", REPORT_ROOT / "environment_profile.json"), ("data_profile", REPORT_ROOT / "data_profile.json"), ("smoke_summary", REPORT_ROOT / "smoke_summary.json"), ("calibration_summary", REPORT_ROOT / "calibration_summary.json"), ("search_amendment", REPORT_ROOT / "search_amendment.json"), ("selection_lock", REPORT_ROOT / "selection_lock.json"), ("report", REPORT_ROOT / "report.md"), ("review", REPORT_ROOT / "review.md"), ("registry", REPORT_ROOT / "run_registry.json"), ("results", REPORT_ROOT / "run_results.jsonl"), ("summary", REPORT_ROOT / "summary.json"), ("audit", REPORT_ROOT / "audit_result.json"), ("test_record", REPORT_ROOT / "test_record.json")):
        if path.exists():
            paths.append({"name": name, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    per_run = []
    for row in rows:
        item = {"attempt_id": row["attempt_id"], "status": row["status"]}
        for field in ("best_checkpoint", "latest_checkpoint", "prediction_file", "step_log", "diagnostic_log"):
            raw = row.get(field)
            if raw and Path(raw).is_file():
                path = Path(raw)
                item[field] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        per_run.append(item)
    manifest = {
        "study": STUDY, "created_utc": utc_now(), "branch": git("branch", "--show-current"),
        "training_source_commit": lock["training_source_commit"], "final_source_commit_at_manifest": git("rev-parse", "HEAD"),
        "training_source_hashes": lock["source_hashes"], "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "v1_protocol_lock_sha256": lock["v1_protocol_lock_sha256"], "datasets": lock["datasets"],
        "artifacts": paths, "runs": per_run,
        "raw_evidence_root": str(OUTPUT_ROOT.resolve()), "test_evaluation": False, "lp_runs": 0,
        "smoke_raw_artifacts": [
            {"dataset": item["dataset"], "path": str((OUTPUT_ROOT / "smoke" / f"{item['dataset']}_M0_3step.json").resolve()), "bytes": (OUTPUT_ROOT / "smoke" / f"{item['dataset']}_M0_3step.json").stat().st_size, "sha256": sha256_file(OUTPUT_ROOT / "smoke" / f"{item['dataset']}_M0_3step.json")}
            for item in (json.loads((REPORT_ROOT / "smoke_summary.json").read_text(encoding="utf-8")).get("rows", []) if (REPORT_ROOT / "smoke_summary.json").is_file() else [])
        ],
    }
    write_json(REPORT_ROOT / "manifest.json", manifest)
    print(f"manifest written: {REPORT_ROOT / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-input train/validation MAG collaboration NC pilot (no test path).")
    parser.add_argument("phase", choices=("prepare", "smoke", "calibrate", "lock-selection", "confirm", "report", "audit", "manifest"))
    parser.add_argument("--device", default="cuda:0", help="CUDA device; run with CUDA_VISIBLE_DEVICES=1 to use physical GPU 1")
    parser.add_argument("--selection-sha256", default=None, help="optional confirmation guard for the committed selection lock")
    parser.add_argument("--no-resume", action="store_true", help="do not restore latest checkpoints")
    parser.add_argument("--retry-failed", action="store_true", help="make a new registered retry attempt for prior failed cells")
    args = parser.parse_args()
    if args.phase == "prepare":
        phase_prepare()
    elif args.phase == "smoke":
        phase_smoke(args)
    elif args.phase == "calibrate":
        phase_calibrate(args)
    elif args.phase == "lock-selection":
        phase_lock_selection()
    elif args.phase == "confirm":
        phase_confirm(args)
    elif args.phase == "report":
        phase_report()
    elif args.phase == "audit":
        phase_audit()
    elif args.phase == "manifest":
        phase_manifest()


if __name__ == "__main__":
    main()
