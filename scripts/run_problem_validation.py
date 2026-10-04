from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.analysis.problem_validation import (  # noqa: E402
    compose_probe_config, confusion_metrics, dataset_input_paths, fixed_split_path,
    load_raw_edges, sha256_file, sha256_json, tensor_sha256, validate_loaded_data,
)
from src.data import load_mag_data  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.models.mag_probe import resolve_capacity_widths  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "Reddit-S", "ele-fashion")
SEEDS = (42, 43, 44)
CAL_MODELS = ("J_FIXED", "J_GLOBAL", "J_NODE")
CAL_LRS = (3e-4, 1e-3, 3e-3)
P1_MODELS = ("T_E", "I_E", "T_M", "I_M", "J_E", "J_EARLY", "J_N", "J_FIXED", "J_GLOBAL")
P2_MODELS = ("J_NODE", "J_NODE_SHUFFLE", "J_GLOBAL_WIDE", "J_CONCAT")
P3_MODELS = ("T_E", "I_E", "J_E", "J_FIXED", "J_GLOBAL", "J_NODE", "J_GLOBAL_WIDE", "J_CONCAT")
REPORTS = ROOT / "reports/problem_validation_v1"
RAW = ROOT / "outputs/problem_validation_v1"
REGISTRY_PATH = REPORTS / "run_registry.json"
RESULTS_PATH = REPORTS / "run_results.jsonl"
ATTEMPT_CAP = 180


class GPUUnavailable(RuntimeError):
    pass


def select_idle_gpu(requested: str) -> str:
    if not requested.startswith("cuda"):
        return requested
    if not torch.cuda.is_available():
        raise GPUUnavailable("CUDA is not visible in this process")
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        raise GPUUnavailable(f"nvidia-smi availability check failed: {exc}") from exc
    devices = []
    for line in raw.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            idx, used, total, util = map(int, parts)
            devices.append({"index": idx, "used_mib": used, "total_mib": total, "util_pct": util})
    idle = [d for d in devices if d["util_pct"] <= 10 and d["used_mib"] <= max(512, int(d["total_mib"] * 0.05))]
    if not idle:
        raise GPUUnavailable(f"no idle GPU found; current query: {raw.strip()}")
    requested_id = int(requested.split(":", 1)[1]) if ":" in requested else 0
    selected = next((d for d in idle if d["index"] == requested_id), None)
    if selected is None:
        selected = min(idle, key=lambda d: (d["used_mib"], d["util_pct"], d["index"]))
    return f"cuda:{selected['index']}"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes() -> dict[str, str]:
    paths = ["src/models/mag_probe.py", "src/analysis/problem_validation.py",
             "src/data/loaders.py", "src/data/splits.py", "src/data/graph_utils.py", "src/data/types.py",
             "src/tasks/nc.py", "src/models/factory.py", "src/utils/seeds.py",
             "scripts/run_problem_validation.py", "scripts/evaluate_locked_probe.py",
             "configs/model/mag_probe.yaml", "configs/config.yaml", "configs/task/nc.yaml"]
    paths.extend(f"configs/dataset/{ds}.yaml" for ds in DATASETS)
    return {path: sha256_file(ROOT / path) for path in paths if (ROOT / path).is_file()}


def config_for(dataset: str, seed: int, variant: str, lr: float,
               device: str) -> Any:
    return compose_probe_config(dataset, seed, fixed_split_path(dataset), lr, variant, device)


def read_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry has not been created: {REGISTRY_PATH}; run --dry-run first")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def get_data(cfg, cache: dict[str, Any] | None = None):
    key = str(cfg.dataset.name)
    if cache is not None and key in cache:
        return cache[key]
    data = load_mag_data(cfg, "nc", int(cfg.seed))
    if cache is not None:
        cache[key] = data
    return data


def audit_dataset(dataset: str) -> dict[str, Any]:
    split = fixed_split_path(dataset)
    if not split.is_file():
        raise FileNotFoundError(f"Required fixed split is missing: {split}")
    cfg = config_for(dataset, 42, "J_FIXED", 1e-3, "cpu")
    paths = dataset_input_paths(cfg)
    split_files: dict[str, str] = {}
    if dataset == "ele-fashion":
        split_files[str(paths["node_split_path"])] = sha256_file(paths["node_split_path"])
    else:
        for seed in SEEDS:
            p = Path(f"/hdd1/DataInHere/YHF/data/MAGB_split/{dataset}_nc_seed{seed}_train0.6_val0.2.pt")
            if not p.is_file():
                raise FileNotFoundError(f"Required pre-existing split is missing: {p}")
            split_files[str(p)] = sha256_file(p)
        if str(split.resolve()) not in split_files:
            raise AssertionError("the training split must explicitly use the seed-42 file")
    data = get_data(cfg)
    raw_edges = load_raw_edges(cfg)
    report = validate_loaded_data(data, raw_edges)
    report["fixed_training_split_path"] = str(split.resolve())
    report["fixed_training_split_sha256"] = sha256_file(split)
    report["available_split_sha256_by_training_seed_filename"] = split_files
    report["input_file_hashes"] = {
        str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in paths.values()
    }
    report["auto_generate_splits"] = False
    report["split_and_training_seed_separate"] = True
    if dataset == "ele-fashion":
        mapping = torch.load(paths["node_mapping_path"], map_location="cpu", weights_only=False)
        ids = list(mapping.values()) if isinstance(mapping, dict) else []
        report["official_node_mapping"] = {
            "entries": len(mapping), "id_values_unique": len(set(ids)) == len(ids),
            "id_values_are_0_to_N_minus_1": sorted(ids) == list(range(data.num_nodes)),
            "sha256": sha256_file(paths["node_mapping_path"]),
            "mapping_preserved": True,
        }
        report["ele_fashion_schema_note"] = (
            "Configured label schema is 0..11. Current official labels contain the 11 IDs "
            "listed in macro_f1_label_set; class ID 5 is absent. Labels and official node mapping "
            "are unchanged; no category remapping is performed."
        )
        expected = [58659, 9777, 29330]
        actual = [report["splits"][name]["n"] for name in ("train", "val", "test")]
        report["official_split_sizes_match_expected"] = actual == expected
    widths = resolve_capacity_widths(data.num_classes, data.x_t.size(1), data.x_i.size(1))
    report["capacity_match_widths"] = widths
    return report


def write_protocol_audit(audits: list[dict[str, Any]], source_commit: str) -> None:
    lines = ["# Protocol audit", "", f"Source commit before training: `{source_commit}`.",
             "", "## Code and task protocol", "",
             "The inherited NC task path uses full-graph training for graph encoders, an external linear head, validation-accuracy checkpoint selection, and optional test evaluation. The probe is an explicit new model/config and does not change legacy defaults. Every probe uses full-graph training, one optimizer step per epoch, train-index cross-entropy, and sets test evaluation off.",
             "", "The loader uses MAGB `torch.cat([text,image])` and MM-Graph slices `[text,image]`. MAGB graph preprocessing removes self-loops and makes graphs undirected; the new operator independently removes self-loops and duplicate receiver/sender arcs and computes receiver means. The model cache holds topology only.",
             "", "## Dataset and split checks", ""]
    for item in audits:
        lines += [f"### {item['dataset']}", "",
                  f"- Nodes/classes: {item['num_nodes']} / {item['num_classes_configured']} configured; observed supervised Macro-F1 labels `{item['macro_f1_label_set']}`.",
                  f"- Feature order: `{item['feature_order']}`; text `{item['features']['text']['shape']}`, image `{item['features']['image']['shape']}`; nonfinite rows text/image `{item['features']['text']['nonfinite_rows']}/{item['features']['image']['nonfinite_rows']}`; zero rows `{item['features']['text']['zero_rows']}/{item['features']['image']['zero_rows']}`.",
                  f"- Split sizes train/val/test: `{item['splits']['train']['n']}/{item['splits']['val']['n']}/{item['splits']['test']['n']}`; mutually exclusive and label-range checks passed.",
                  f"- Explicit split `{item['fixed_training_split_path']}` SHA-256 `{item['fixed_training_split_sha256']}`; split generation disabled.",
                  f"- Raw graph checks: `{item['raw_graph']}`; processed graph checks: `{item['processed_graph']}`.",
                  f"- Capacity matching: wide h={item['capacity_match_widths']['wide_hidden_dim']} ({item['capacity_match_widths']['wide_relative_error']:.4%}), concat b={item['capacity_match_widths']['concat_bottleneck']} ({item['capacity_match_widths']['concat_relative_error']:.4%})."]
        if item["dataset"] == "ele-fashion":
            lines += [f"- Official split sizes match 58,659/9,777/29,330: `{item['official_split_sizes_match_expected']}`.",
                      f"- Category schema: {item['ele_fashion_schema_note']}",
                      f"- Official node mapping checks: `{item['official_node_mapping']}`."]
        lines.append("")
    lines += ["## Review boundaries", "",
              "Raw file and processed-edge hashes are in `run_registry.json`. Val/test Macro-F1 uses the observed supervised label set fixed before prediction and always derives counts from the complete configured confusion matrix. No label-informed candidate construction or validation/test label enters graph structure.",
              ""]
    (REPORTS / "protocol_audit.md").write_text("\n".join(lines), encoding="utf-8")


def build_slots() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration = []
    for dataset in ("Movies", "Grocery"):
        for variant in CAL_MODELS:
            for lr in CAL_LRS:
                calibration.append({"job_id": f"cal-{dataset}-{variant}-s42-lr{lr:g}",
                    "stage": "calibration", "dataset": dataset, "variant": variant,
                    "seed": 42, "split_path": str(fixed_split_path(dataset).resolve()), "lr": lr,
                    "status": "pending", "evaluate_test": False})
    primary = []
    def add(phase: str, datasets: tuple[str, ...], variants: tuple[str, ...]):
        for dataset in datasets:
            for variant in variants:
                for seed in SEEDS:
                    primary.append({"job_id": f"{phase}-{dataset}-{variant}-s{seed}",
                        "stage": "main", "phase": phase, "dataset": dataset,
                        "variant": variant, "seed": seed,
                        "split_path": str(fixed_split_path(dataset).resolve()),
                        "lr": None, "status": "pending", "evaluate_test": False,
                        "reuses_training_job_id": None})
    add("P1", ("Movies", "Grocery"), P1_MODELS)
    add("P2", ("Movies", "Grocery"), P2_MODELS)
    add("P3", ("Toys", "Reddit-S", "ele-fashion"), P3_MODELS)
    return calibration, primary


def initial_registry(force: bool = False) -> dict[str, Any]:
    if REGISTRY_PATH.exists() and not force:
        existing = read_registry()
        if source_hashes() != existing.get("source_hashes"):
            raise RuntimeError("source hashes changed since registration; refusing to mix code identities")
        return existing
    if force and REGISTRY_PATH.exists():
        existing = read_registry()
        if int(existing.get("attempt_count", 0)) > 0:
            raise RuntimeError("refusing to replace a registry after a training attempt has started")
    REPORTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    commit = git_value("rev-parse", "HEAD")
    initial_status = git_value("status", "--short", "--branch")
    hashes = source_hashes()
    audits = [audit_dataset(ds) for ds in DATASETS]
    write_protocol_audit(audits, commit)
    calibration, primary = build_slots()
    data_by_name = {x["dataset"]: x for x in audits}
    for slot in calibration + primary:
        slot["split_sha256"] = data_by_name[slot["dataset"]]["fixed_training_split_sha256"]
        slot["dataset_input_hashes"] = data_by_name[slot["dataset"]]["input_file_hashes"]
        slot["code_commit"] = commit
        slot["source_hashes"] = hashes
        if slot["lr"] is not None:
            cfg = config_for(slot["dataset"], slot["seed"], slot["variant"], slot["lr"], "cuda:0")
            # Register exactly the configuration fingerprint used at runtime.
            slot["config_hash"] = _training_config_hash(cfg)
            widths = resolve_capacity_widths(data_by_name[slot["dataset"]]["num_classes_configured"],
                data_by_name[slot["dataset"]]["features"]["text"]["shape"][1],
                data_by_name[slot["dataset"]]["features"]["image"]["shape"][1])
            slot["capacity_match"] = {
                "J_GLOBAL_WIDE": {"width": widths["wide_hidden_dim"], "relative_error": widths["wide_relative_error"], "matched": widths["wide_relative_error"] <= .01},
                "J_CONCAT": {"width": widths["concat_bottleneck"], "relative_error": widths["concat_relative_error"], "matched": widths["concat_relative_error"] <= .01},
            }
    registry = {
        "schema_version": 1,
        "study": "mag_problem_validation_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": git_value("branch", "--show-current"),
        "source_commit": commit,
        "initial_git_status": initial_status,
        "task_entry_git_snapshot": {"branch": "main",
            "head": "579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e",
            "status": [" D CLAUDE.md"],
            "note": "Captured before branch creation; the existing user deletion is preserved and excluded from every commit."},
        "source_hashes": hashes,
        "datasets": audits,
        "calibration_slots": calibration,
        "primary_slots": primary,
        "selected_common_lr": None,
        "unique_training_jobs_planned": 162,
        "planned_attempt_cap": ATTEMPT_CAP,
        "attempt_count": 0,
        "failures": [],
        "commands": {
            "calibration_resume": "conda run --no-capture-output -n yhf_env python scripts/run_problem_validation.py --stage calibration --resume --device cuda:1",
            "main_resume": "conda run --no-capture-output -n yhf_env python scripts/run_problem_validation.py --stage main --resume --device cuda:1",
            "evaluate_resume": "conda run --no-capture-output -n yhf_env python scripts/evaluate_locked_probe.py --resume --device cuda:1",
            "summarize": "conda run --no-capture-output -n yhf_env python scripts/evaluate_locked_probe.py --summary-only",
        },
    }
    atomic_json(REGISTRY_PATH, registry)
    atomic_json(REPORTS / "data_audit.json", {"datasets": audits})
    atomic_json(ROOT / "docs/problem_validation_v1/matrix.json", {
        "calibration_slots": calibration, "primary_slots": primary,
        "unique_training_jobs_planned": 162,
    })
    return registry


def _training_config_hash(cfg) -> str:
    payload = OmegaConf.to_container(cfg, resolve=True)
    # Operational output paths and the selected model variant are set per job;
    # the variant is part of the config and remains in the hash.
    payload["paths"].pop("output_root", None)
    payload.pop("device", None)  # execution placement does not alter the training protocol
    # Hydra's run directory embeds the current wall-clock timestamp and is unused
    # by this launcher; exclude it so separate calls have the same identity.
    payload.pop("hydra", None)
    return sha256_json(payload)


def runtime_job(slot: dict[str, Any], device: str) -> dict[str, Any]:
    cfg = config_for(slot["dataset"], slot["seed"], slot["variant"], float(slot["lr"]), device)
    cfg_hash = _training_config_hash(cfg)
    return {**slot, "config_hash": cfg_hash,
            "fingerprint": sha256_json({"dataset": slot["dataset"],
                "variant": slot["variant"], "seed": slot["seed"], "split_sha256": slot["split_sha256"],
                "dataset_input_hashes": slot.get("dataset_input_hashes"),
                "lr": float(slot["lr"]), "config_hash": cfg_hash,
                "source_commit": slot["code_commit"], "source_hashes": slot["source_hashes"]})}


def eval_labels(data) -> list[int]:
    supervised = torch.cat([data.y[data.train_idx], data.y[data.val_idx], data.y[data.test_idx]]).cpu()
    return sorted(int(v) for v in torch.unique(supervised).tolist() if 0 <= int(v) < int(data.num_classes))


def checkpoint_file_hash(path: Path) -> str:
    return sha256_file(path)


def append_result(row: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


DATA_CACHE: dict[str, Any] = {}
VERIFIED_DATASETS: set[str] = set()


def verify_dataset_inputs(dataset: str, registry: dict[str, Any]) -> None:
    if dataset in VERIFIED_DATASETS:
        return
    expected = next(item for item in registry["datasets"] if item["dataset"] == dataset)
    for path, item in expected["input_file_hashes"].items():
        p = Path(path)
        if (not p.is_file() or p.stat().st_size != item["bytes"] or
                sha256_file(p) != item["sha256"]):
            raise RuntimeError(f"dataset input hash/size changed since registration: {p}")
    if source_hashes() != registry["source_hashes"]:
        raise RuntimeError("source file hashes changed since registration")
    VERIFIED_DATASETS.add(dataset)


def train_job(slot: dict[str, Any], registry: dict[str, Any], device_name: str,
              resume: bool = True) -> dict[str, Any]:
    verify_dataset_inputs(slot["dataset"], registry)
    if sha256_file(Path(slot["split_path"])) != slot["split_sha256"]:
        raise RuntimeError(f"fixed split identity changed for {slot['job_id']}")
    requested_device = device_name
    job = runtime_job(slot, device_name)
    ckpt = RAW / "checkpoints" / slot["dataset"] / slot["variant"] / f"seed{slot['seed']}" / f"{slot['job_id']}.pt"
    log_path = RAW / "logs" / f"{slot['job_id']}.log"
    for record in registry["calibration_slots"] + registry["primary_slots"]:
        if record["job_id"] == slot["job_id"]:
            record.update({k: job[k] for k in ("config_hash", "fingerprint")})
            record["checkpoint_path"] = str(ckpt.resolve())
            record["log_path"] = str(log_path.resolve())
            record["status"] = "running"
            break
    previous = None
    if RESULTS_PATH.exists():
        for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("job_id") == slot["job_id"] and item.get("record_type", "training") == "training":
                previous = item
    if resume and previous and previous.get("fingerprint") == job["fingerprint"] and ckpt.is_file() and sha256_file(ckpt) == previous.get("checkpoint_sha256"):
        for record in registry["calibration_slots"] + registry["primary_slots"]:
            if record["job_id"] == slot["job_id"]:
                record.update(status="succeeded", checkpoint_sha256=previous["checkpoint_sha256"],
                              resumed_skip=True)
        atomic_json(REGISTRY_PATH, registry)
        return previous

    device_name = select_idle_gpu(device_name)
    device = torch.device(device_name)
    if registry["attempt_count"] >= ATTEMPT_CAP:
        raise RuntimeError(f"Attempt cap {ATTEMPT_CAP} exhausted before {slot['job_id']}")
    registry["attempt_count"] += 1
    registry["last_attempt_job_id"] = slot["job_id"]
    atomic_json(REGISTRY_PATH, registry)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    cfg = config_for(slot["dataset"], slot["seed"], slot["variant"], float(slot["lr"]), device_name)
    cfg_hash = _training_config_hash(cfg)
    if cfg_hash != job["config_hash"]:
        raise RuntimeError("configuration identity changed between registry and job startup")
    set_seed(int(slot["seed"]))
    try:
        data = get_data(cfg, DATA_CACHE)
        labels = eval_labels(data)
        data_x = data.x.to(device)
        y = data.y.to(device).long()
        train_idx = data.train_idx.to(device).long()
        val_idx = data.val_idx.to(device).long()
        edge = data.edge_index.to(device).long()
        info = {"num_nodes": data.num_nodes, "num_classes": data.num_classes,
                "text_dim": data.x_t.size(1), "visual_dim": data.x_i.size(1)}
        model = build_model(cfg, info).to(device)
        model.set_run_seed(int(slot["seed"]))
        head = nn.Linear(128, int(data.num_classes)).to(device)
        params = list(model.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=float(slot["lr"]), weight_decay=1e-4)
        p_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        p_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
        best_acc, best_epoch, best_val = -1.0, None, None
        best_model_state = best_head_state = None
        patience_left = 30
        train_seconds = inference_seconds = 0.0
        cuda_index = device.index or 0
        torch.cuda.reset_peak_memory_stats(cuda_index)
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write(json.dumps({"event": "start", "job": job, "config": OmegaConf.to_container(cfg, resolve=True)}, default=str) + "\n")
            epochs_run = 0
            for epoch in range(1, 301):
                epochs_run = epoch
                model.train(); head.train()
                model.set_epoch(epoch)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device); t0 = time.perf_counter()
                z, _, _, aux_loss, _ = model(data_x, edge)
                logits = head(z[train_idx])
                loss = nn.functional.cross_entropy(logits, y[train_idx]) + 0.0 * aux_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite train loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(device); train_seconds += time.perf_counter() - t0
                del z, logits

                model.eval(); head.eval()
                torch.cuda.synchronize(device); t1 = time.perf_counter()
                with torch.no_grad():
                    z, _, _, _, _ = model(data_x, edge)
                    val_logits = head(z[val_idx])
                    pred = val_logits.argmax(dim=1).cpu()
                    target = data.y[val_idx.cpu()].long().cpu()
                    val_metrics = confusion_metrics(target, pred, int(data.num_classes), labels)
                torch.cuda.synchronize(device); inference_seconds += time.perf_counter() - t1
                log.write(json.dumps({"event": "epoch", "epoch": epoch,
                    "train_loss": float(loss.detach().cpu()), "val_acc": val_metrics["acc"],
                    "val_macro_f1": val_metrics["macro_f1"]}) + "\n")
                improved = val_metrics["acc"] > best_acc + 1e-4
                if improved:
                    best_acc = val_metrics["acc"]
                    best_epoch = epoch
                    best_val = val_metrics
                    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    patience_left = 30
                elif epoch >= 30:
                    patience_left -= 1
                if epoch >= 30 and patience_left <= 0:
                    break
                del z, val_logits
            if best_model_state is None or best_head_state is None:
                raise RuntimeError("training ended without a validation-selected checkpoint")
            model.load_state_dict(best_model_state); head.load_state_dict(best_head_state)
        checkpoint = {
            "schema_version": 1, "task": "nc", "model": "mag_probe",
            "variant": slot["variant"], "dataset": slot["dataset"], "job_id": slot["job_id"], "seed": int(slot["seed"]),
            "split_path": slot["split_path"], "split_sha256": slot["split_sha256"],
            "dataset_input_hashes": slot["dataset_input_hashes"],
            "lr": float(slot["lr"]), "config_hash": cfg_hash,
            "fingerprint": job["fingerprint"], "source_commit": slot["code_commit"],
            "source_hashes": slot["source_hashes"], "selection": "best_val_accuracy",
            "best_epoch": best_epoch, "model_state": best_model_state,
            "head_state": best_head_state, "data_info": info,
            "macro_f1_label_set": labels,
        }
        torch.save(checkpoint, ckpt)
        if device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory = None
        checkpoint_sha = checkpoint_file_hash(ckpt)
        row = {
            "record_type": "training", "job_id": slot["job_id"],
            "dataset": slot["dataset"], "variant": slot["variant"],
            "stage": slot["stage"], "phase": slot.get("phase"), "seed": int(slot["seed"]),
            "split_path": slot["split_path"], "split_sha256": slot["split_sha256"],
            "dataset_input_hashes": slot["dataset_input_hashes"],
            "config_hash": cfg_hash, "code_commit": slot["code_commit"],
            "source_hashes": slot["source_hashes"], "fingerprint": job["fingerprint"],
            "status": "succeeded", "lr": float(slot["lr"]), "val": best_val,
            "test": None, "test_null_reason": "locked evaluation is a separate post-training command",
            "macro_f1_label_set": labels, "best_epoch": int(best_epoch),
            "hyperparameters": {"optimizer": "AdamW", "learning_rate": float(slot["lr"]), "weight_decay": 1e-4,
                "epochs_max": 300, "patience": 30, "early_stop_min_epoch": 30,
                "early_stop_min_delta": 1e-4, "grad_clip": 1.0, "scheduler": None,
                "dropout": 0.2, "out_dim": 128, "base_hidden_dim": 128,
                "aux_loss_weight": 0.0, "full_graph": True, "evaluate_test": False,
                "optimizer_steps_per_epoch": 1,
                "feature_row_normalization": "L2 epsilon=1e-8; zero rows stay zero",
                "active_adapters": {"text": model.adapter_t is not None, "image": model.adapter_i is not None},
                "phi": "Linear -> LayerNorm -> GELU -> Dropout -> Linear",
                "receiver_operator": "sparse P[dst,src], no self-loop, deduplicated, weight 1/receiver-indegree, one hop",
                "fusion_temperature": 1.0 if model.router is not None else None,
                "router_architecture": "Linear(516,32)->GELU->Linear(32,4), final layer zero initialized" if model.router is not None else None,
                "shuffle_policy": "run_seed+epoch+mag_probe_context_v1; train-specific epoch permutations; validation_test fixed namespace"},
            "parameters": {"model": p_model, "head": p_head, "total": p_model + p_head,
                "capacity_match": getattr(model, "capacity_match", None),
                "effective_phi_hidden_dim": model.hidden_dim,
                "concat_bottleneck": model.concat_bottleneck if model.concat is not None else None,
                "router_parameters": (sum(p.numel() for p in model.router.parameters()) if model.router else 0)},
            "timing_seconds": {"training": train_seconds, "full_graph_validation_inference": inference_seconds,
                "total": train_seconds + inference_seconds,
                "mean_validation_inference": inference_seconds / max(epochs_run, 1),
                "epochs_run": epochs_run, "device": str(device), "requested_device": requested_device},
            "peak_gpu_memory_bytes": peak_memory,
            "checkpoint_path": str(ckpt.resolve()), "checkpoint_sha256": checkpoint_sha,
            "log_path": str(log_path.resolve()), "log_sha256": sha256_file(log_path),
            "confusion_schema": list(range(int(data.num_classes))),
        }
        append_result(row)
        for record in registry["calibration_slots"] + registry["primary_slots"]:
            if record["job_id"] == slot["job_id"]:
                record.update(status="succeeded", best_epoch=int(best_epoch), val_acc=best_val["acc"],
                              val_macro_f1=best_val["macro_f1"], checkpoint_path=str(ckpt.resolve()),
                              checkpoint_sha256=checkpoint_sha, completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(REGISTRY_PATH, registry)
        return row
    except Exception as exc:
        failure = {"job_id": slot["job_id"], "dataset": slot["dataset"],
            "variant": slot["variant"], "stage": slot["stage"], "seed": slot["seed"],
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "lr": float(slot["lr"]), "split_path": slot["split_path"],
            "split_sha256": slot["split_sha256"], "config_hash": job["config_hash"],
            "dataset_input_hashes": slot["dataset_input_hashes"],
            "fingerprint": job["fingerprint"], "code_commit": slot["code_commit"],
            "source_hashes": slot["source_hashes"],
            "checkpoint_path": str(ckpt.resolve()), "log_path": str(log_path.resolve()),
            "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
            "attempt": registry["attempt_count"]}
        registry["failures"].append(failure)
        for record in registry["calibration_slots"] + registry["primary_slots"]:
            if record["job_id"] == slot["job_id"]:
                record.update(status="failed", failure=failure)
        atomic_json(REGISTRY_PATH, registry)
        append_result({"record_type": "training", **failure, "fingerprint": job["fingerprint"],
            "config_hash": job["config_hash"], "code_commit": slot["code_commit"],
            "dataset_input_hashes": slot["dataset_input_hashes"],
            "split_path": slot["split_path"], "split_sha256": slot["split_sha256"],
            "test": None, "test_null_reason": "training failed; no checkpoint"})
        raise


def select_common_lr(registry: dict[str, Any]) -> float:
    training_rows = [json.loads(line) for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines()] if RESULTS_PATH.exists() else []
    rows = {row["job_id"]: row for row in training_rows if row.get("record_type") == "training" and row.get("status") == "succeeded"}
    expected = len(registry["calibration_slots"])
    if any(slot["job_id"] not in rows for slot in registry["calibration_slots"]):
        raise RuntimeError(f"LR calibration incomplete: expected {expected} successful slots")
    scores = {}
    for lr in CAL_LRS:
        values = []
        for ds in ("Movies", "Grocery"):
            for model in CAL_MODELS:
                slot = next(x for x in registry["calibration_slots"] if x["dataset"] == ds and x["variant"] == model and math.isclose(float(x["lr"]), lr))
                values.append(float(rows[slot["job_id"]]["val"]["acc"]))
        scores[str(lr)] = {"cell_best_val_acc": values, "equal_weight_mean": sum(values) / len(values)}
    order = (1e-3, 3e-4, 3e-3)
    best_score = max(scores[str(lr)]["equal_weight_mean"] for lr in CAL_LRS)
    tied = [lr for lr in order if best_score - scores[str(lr)]["equal_weight_mean"] <= 1e-6]
    selected = tied[0]
    registry["lr_selection"] = {"scores": scores, "rule": "equal-weight mean best-val Acc across six dataset/model cells; tie within 1e-6 prefers 1e-3, 3e-4, 3e-3", "selected_lr": selected}
    registry["selected_common_lr"] = selected
    cal_by_identity = {(s["dataset"], s["variant"], s["seed"], float(s["lr"])): s for s in registry["calibration_slots"]}
    for slot in registry["primary_slots"]:
        slot["lr"] = selected
        main_identity = runtime_job(slot, "cuda:0")
        slot.update({"config_hash": main_identity["config_hash"],
                    "fingerprint": main_identity["fingerprint"]})
        candidate = cal_by_identity.get((slot["dataset"], slot["variant"], slot["seed"], float(selected)))
        if slot["dataset"] in ("Movies", "Grocery") and slot["variant"] in CAL_MODELS and slot["seed"] == 42 and candidate:
            train_row = rows.get(candidate["job_id"])
            if train_row and candidate.get("fingerprint") == slot["fingerprint"] and train_row["fingerprint"] == candidate.get("fingerprint"):
                slot["reuses_training_job_id"] = candidate["job_id"]
                slot["reuse_identity_match"] = True
            else:
                slot["reuse_identity_match"] = False
                slot["reuse_rejected_reason"] = "selected calibration checkpoint absent or fingerprint mismatch"
    registry["planned_unique_training_jobs_after_reuse"] = len(registry["calibration_slots"]) + sum(not s.get("reuses_training_job_id") for s in registry["primary_slots"])
    atomic_json(REGISTRY_PATH, registry)
    atomic_json(REPORTS / "lr_selection.json", registry["lr_selection"])
    atomic_json(ROOT / "docs/problem_validation_v1/matrix.json", {
        "calibration_slots": registry["calibration_slots"],
        "primary_slots": registry["primary_slots"],
        "lr_selection": registry["lr_selection"],
        "unique_training_jobs_planned": registry["planned_unique_training_jobs_after_reuse"],
    })
    return float(selected)


def run_stage(registry: dict[str, Any], stage: str, device: str, resume: bool) -> None:
    slots = registry["calibration_slots"] if stage == "calibration" else registry["primary_slots"]
    if stage == "main" and registry.get("selected_common_lr") is None:
        raise RuntimeError("the common LR is not frozen; complete calibration before the primary matrix")
    for slot in slots:
        if stage == "main" and slot.get("reuses_training_job_id"):
            source_id = slot["reuses_training_job_id"]
            source = next((x for x in registry["calibration_slots"] if x["job_id"] == source_id), None)
            source_path = Path(source["checkpoint_path"]) if source and source.get("checkpoint_path") else None
            checkpoint_ok = bool(source_path and source_path.is_file() and
                sha256_file(source_path) == source.get("checkpoint_sha256"))
            if (source and source.get("status") == "succeeded" and checkpoint_ok and
                    source.get("fingerprint") == slot.get("fingerprint")):
                slot.update(status="reused", checkpoint_path=source.get("checkpoint_path"),
                            checkpoint_sha256=source.get("checkpoint_sha256"),
                            training_fingerprint=source.get("fingerprint"))
                atomic_json(REGISTRY_PATH, registry)
                continue
            slot["reuses_training_job_id"] = None
            slot["reuse_rejected_reason"] = "calibration checkpoint identity or status does not match"
        if stage == "main":
            job = runtime_job(slot, device)
            slot.update({"config_hash": job["config_hash"], "fingerprint": job["fingerprint"]})
            atomic_json(REGISTRY_PATH, registry)
        if slot.get("status") == "running":
            slot["status"] = "interrupted"
        if registry["attempt_count"] >= ATTEMPT_CAP:
            slot.update(status="not_run_attempt_cap", failure={"reason": f"hard cap {ATTEMPT_CAP} reached"})
            for remaining in slots[slots.index(slot) + 1:]:
                if stage == "main" and remaining.get("reuses_training_job_id"):
                    continue
                remaining.update(status="not_run_attempt_cap", failure={"reason": f"hard cap {ATTEMPT_CAP} reached"})
            atomic_json(REGISTRY_PATH, registry)
            break
        print(f"RUN {slot['job_id']} dataset={slot['dataset']} model={slot['variant']} seed={slot['seed']} lr={slot['lr']}", flush=True)
        try:
            train_job(slot, registry, device, resume=resume)
        except GPUUnavailable as exc:
            print(f"WAITING {slot['job_id']}: {exc}", flush=True)
            slot.update(status="waiting_for_idle_gpu", resource_wait_reason=str(exc))
            atomic_json(REGISTRY_PATH, registry)
            break
        except Exception as exc:
            print(f"FAILED {slot['job_id']}: {type(exc).__name__}: {exc}", flush=True)
            # Keep going so an isolated data or numerical failure does not erase independent results.
            continue
    if stage == "calibration" and all(x.get("status") == "succeeded" for x in registry["calibration_slots"]):
        selected = select_common_lr(registry)
        print(f"Selected common LR: {selected:g}", flush=True)
    elif stage == "calibration":
        print("Calibration incomplete; rerun the calibration stage after resolving failures or GPU availability.", flush=True)


def run_smoke(device_name: str, datasets: tuple[str, ...] = ("Movies", "ele-fashion")) -> dict[str, Any]:
    requested_device = device_name
    device_name = select_idle_gpu(device_name)
    device = torch.device(device_name)
    output = {"device": str(device), "requested_device": requested_device, "datasets": []}
    for dataset in datasets:
        cfg = config_for(dataset, 42, "J_NODE", 1e-3, device_name)
        set_seed(42)
        data = get_data(cfg)
        x, y = data.x.to(device), data.y.to(device).long()
        train_idx, val_idx = data.train_idx.to(device), data.val_idx.to(device)
        edge = data.edge_index.to(device)
        info = {"num_nodes": data.num_nodes, "num_classes": data.num_classes,
                "text_dim": data.x_t.size(1), "visual_dim": data.x_i.size(1)}
        model = build_model(cfg, info).to(device); model.set_run_seed(42)
        head = nn.Linear(128, data.num_classes).to(device)
        params = list(model.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats(device)
        step_seconds, losses, grads = [], [], {}
        for step in range(1, 3):
            model.train(); head.train(); model.set_epoch(step)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device); start = time.perf_counter()
            z, _, _, aux, _ = model(x, edge)
            loss = nn.functional.cross_entropy(head(z[train_idx]), y[train_idx]) + 0.0 * aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{dataset}: smoke loss is nonfinite")
            loss.backward()
            if step == 2:
                for name, parameter in list(model.named_parameters()) + [("head.weight", head.weight)]:
                    grads[name] = parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.norm()) > 0
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device); step_seconds.append(time.perf_counter() - start)
            losses.append(float(loss.detach().cpu()))
        model.eval(); head.eval()
        torch.cuda.synchronize(device); start = time.perf_counter()
        with torch.no_grad():
            z, _, _, _, _ = model(x, edge)
            pred = head(z[val_idx])
        torch.cuda.synchronize(device); infer_sec = time.perf_counter() - start
        if z.shape != (data.num_nodes, 128) or not torch.isfinite(z).all() or not torch.isfinite(pred).all():
            raise AssertionError(f"{dataset}: smoke inference dimensions or values invalid")
        required_grads = {"adapter_t.weight", "adapter_i.weight", "phi.layers.0.weight",
                          "phi.layers.4.weight", "router.0.weight", "router.2.weight", "head.weight"}
        missing = sorted(name for name in required_grads if not grads.get(name, False))
        if missing:
            raise AssertionError(f"{dataset}: finite nonzero gradients missing for {missing}")
        item = {"dataset": dataset, "num_nodes": data.num_nodes, "edges": data.num_edges,
                "split_path": str(fixed_split_path(dataset).resolve()),
                "losses": losses, "step_seconds": step_seconds,
                "validation_inference_seconds": infer_sec,
                "estimated_300_epoch_seconds": 300 * (step_seconds[-1] + infer_sec),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "gradient_checks": grads, "test_evaluation": False,
                "operator_rebuilds": model.operator_rebuilds,
                "parameters_with_head": sum(p.numel() for p in params),
                "device": str(device), "requested_device": requested_device}
        output["datasets"].append(item)
        del model, head, x, y, edge, z, pred
        torch.cuda.empty_cache()
    output["source_commit"] = git_value("rev-parse", "HEAD")
    output["source_hashes"] = source_hashes()
    atomic_json(RAW / "smoke.json", output)
    return output


def print_commands(registry: dict[str, Any], stage: str, device: str) -> None:
    prefix = "conda run --no-capture-output -n yhf_env python scripts/run_problem_validation.py"
    slots = registry["calibration_slots"] if stage == "calibration" else registry["primary_slots"]
    print(f"Registry: {REGISTRY_PATH}")
    print(f"Planned slots: {len(slots)}")
    for slot in slots:
        print(f"{prefix} --stage {stage} --job-id {slot['job_id']} --device {device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered MAG NC problem-validation jobs")
    parser.add_argument("--stage", choices=("calibration", "main", "all"), default="all")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dry-run", action="store_true", help="audit inputs, freeze registry, and print commands")
    parser.add_argument("--resume", action="store_true", help="skip successful jobs only on exact fingerprint and checkpoint hash")
    parser.add_argument("--job-id", help="run one exact registered slot")
    parser.add_argument("--smoke", action="store_true", help="two-step Movies and ele-fashion GPU smoke, no test")
    parser.add_argument("--force-registry", action="store_true", help="regenerate registry before any training has started")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(run_smoke(args.device), indent=2))
        return
    registry = initial_registry(force=args.force_registry)
    if args.dry_run:
        print_commands(registry, "calibration", args.device)
        print_commands(registry, "main", args.device)
        print("Locked evaluation: " + registry["commands"]["evaluate_resume"])
        return
    if args.job_id:
        slots = registry["calibration_slots"] + registry["primary_slots"]
        slot = next((item for item in slots if item["job_id"] == args.job_id), None)
        if slot is None:
            raise SystemExit(f"Unknown job ID {args.job_id!r}")
        if slot["stage"] == "main" and registry.get("selected_common_lr") is None:
            raise SystemExit("A main slot cannot run before calibration selects a common LR")
        train_job(slot, registry, args.device, resume=args.resume)
        return
    if args.stage in {"calibration", "all"}:
        run_stage(registry, "calibration", args.device, args.resume)
        registry = read_registry()
    if args.stage in {"main", "all"}:
        run_stage(registry, "main", args.device, args.resume)


if __name__ == "__main__":
    main()
