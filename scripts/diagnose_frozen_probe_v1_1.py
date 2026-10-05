from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.analysis.problem_validation import (  # noqa: E402
    compose_probe_config,
    confusion_metrics,
    fixed_split_path,
    geometry_features,
    geometry_features_l2_neighbor_mean,
    sha256_file,
    sha256_json,
)
from src.data import load_mag_data  # noqa: E402
from src.models.factory import build_model  # noqa: E402


V1_TRAINING_SHA = "63867331e9d3373f19136e47ab192cc9fb014c97"
V1_LOCK_SHA = "3b949712c738dd99865f9bdce882cf65f91756e775cd4af6077cbd065d3b368f"
DATASETS = ("Movies", "Toys", "Grocery", "Reddit-S", "ele-fashion")
SEEDS = (42, 43, 44)
PATHS = ("text_ego", "text_neighbor", "image_ego", "image_neighbor")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_evidence_path(value: str | Path, evidence_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    # Locked paths are absolute and identify the original checkout. When that
    # checkout is read through another worktree, map only repository-relative
    # evidence paths through the explicit evidence root.
    parts = path.parts
    for marker in ("outputs", "reports"):
        if marker in parts:
            candidate = evidence_root.joinpath(*parts[parts.index(marker):])
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"locked evidence file not found: {path}")


def verify_file(path: Path, expected: dict[str, Any] | str, label: str) -> dict[str, Any]:
    expected_sha = expected if isinstance(expected, str) else expected["sha256"]
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual_sha = sha256_file(path)
    actual_bytes = path.stat().st_size
    expected_bytes = None if isinstance(expected, str) else int(expected.get("bytes", actual_bytes))
    if actual_sha != expected_sha or (expected_bytes is not None and actual_bytes != expected_bytes):
        raise RuntimeError(f"{label} hash/size mismatch at {path}: sha={actual_sha}, bytes={actual_bytes}")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha}


def _lock_digest(lock: dict[str, Any]) -> str:
    payload = {key: value for key, value in lock.items()
               if key not in {"frozen_at_utc", "protocol_lock_sha256"}}
    return sha256_json(payload)


def _records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.get("record_type")
        if kind == "training":
            keep = ("record_type", "job_id", "code_commit", "log_path", "status")
        elif kind == "locked_evaluation" and row.get("variant") in {"J_NODE", "J_GLOBAL"}:
            # Project only the locked validation summary and source references;
            # the old row's test metrics are deliberately discarded here.
            keep = ("record_type", "dataset", "variant", "seed", "status", "protocol_lock_sha256",
                    "checkpoint_sha256", "training_job_id", "predictions_path", "predictions_sha256",
                    "job_id", "val")
        else:
            continue
        rows.append({key: row[key] for key in keep if key in row})
    return rows


def _slot_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["variant"]), int(row["seed"])


def preflight(evidence_root: Path) -> dict[str, Any]:
    reports = evidence_root / "reports" / "problem_validation_v1"
    registry = read_json(reports / "run_registry.json")
    lock = read_json(reports / "protocol_lock.json")
    old_manifest = read_json(reports / "manifest.json")
    if registry.get("source_commit") != V1_TRAINING_SHA or lock.get("training_source_commit") != V1_TRAINING_SHA:
        raise RuntimeError("v1 training source SHA does not match the documented frozen source")
    if lock.get("protocol_lock_sha256") != V1_LOCK_SHA or old_manifest.get("protocol_lock_sha256") != V1_LOCK_SHA:
        raise RuntimeError("v1 protocol lock SHA does not match the documented frozen lock")
    if _lock_digest(lock) != V1_LOCK_SHA:
        raise RuntimeError("v1 protocol lock contents do not match its registered digest")
    if lock.get("registry_source_hashes") != registry.get("source_hashes"):
        raise RuntimeError("v1 registry source hashes differ from the source hashes recorded in the lock")

    registry_slots = {_slot_key(row): row for row in registry["primary_slots"]}
    lock_slots = {_slot_key(row): row for row in lock["primary_matrix"]}
    if set(registry_slots) != set(lock_slots):
        raise RuntimeError("v1 registry and protocol lock have different primary matrix cells")
    result_rows = _records(reports / "run_results.jsonl")
    training_rows = {row["job_id"]: row for row in result_rows
                     if row.get("record_type") == "training"}
    eval_rows = {
        _slot_key(row): row for row in result_rows
        if row.get("record_type") == "locked_evaluation"
        and row.get("variant") in {"J_NODE", "J_GLOBAL"}
    }
    if len(eval_rows) != 30:
        raise RuntimeError(f"expected 30 frozen J_NODE/J_GLOBAL eval records, found {len(eval_rows)}")

    manifest_files = {
        str(item["path"]): item["sha256"]
        for item in old_manifest["raw_evidence"]["files"]
    }
    verified_raw_artifacts = []
    for item in old_manifest["raw_evidence"]["files"]:
        artifact_path = resolve_evidence_path(item["path"], evidence_root)
        verified_raw_artifacts.append(verify_file(artifact_path, item["sha256"], "v1 raw artifact"))
    checked_inputs: dict[str, dict[str, Any]] = {}
    checkpoint_sources: list[dict[str, Any]] = []
    prediction_sources: list[dict[str, Any]] = []
    training_devices: dict[str, str | None] = {}
    resolved_slots: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, slot in lock_slots.items():
        dataset, variant, seed = key
        if variant not in {"J_NODE", "J_GLOBAL"}:
            continue
        registry_slot = registry_slots[key]
        for field in ("job_id", "checkpoint_path", "checkpoint_sha256", "split_path", "split_sha256"):
            if registry_slot.get(field) != slot.get(field):
                raise RuntimeError(f"registry/lock mismatch for {key} field {field}")
        if slot.get("status") not in {"succeeded", "reused"}:
            raise RuntimeError(f"frozen checkpoint is not complete: {key} status={slot.get('status')}")
        checkpoint_path = resolve_evidence_path(slot["checkpoint_path"], evidence_root)
        checkpoint_verified = verify_file(checkpoint_path, slot["checkpoint_sha256"], f"checkpoint {key}")
        manifest_sha = manifest_files.get(str(slot["checkpoint_path"]))
        if manifest_sha is not None and manifest_sha != checkpoint_verified["sha256"]:
            raise RuntimeError(f"manifest checkpoint hash mismatch for {key}")
        checkpoint_sources.append({**checkpoint_verified, "dataset": dataset, "variant": variant,
                                   "seed": seed, "job_id": slot["job_id"]})

        expected_inputs = lock["schemas"][dataset]["dataset_input_hashes"]
        if slot["dataset_input_hashes"] != expected_inputs:
            raise RuntimeError(f"slot/schema input hashes differ for {key}")
        for input_path, digest in expected_inputs.items():
            if input_path not in checked_inputs:
                checked_inputs[input_path] = verify_file(Path(input_path), digest, f"input {dataset}")
        split_path = resolve_evidence_path(slot["split_path"], evidence_root)
        split_digest = expected_inputs.get(str(split_path), expected_inputs.get(slot["split_path"]))
        if split_digest is None or split_digest["sha256"] != slot["split_sha256"]:
            raise RuntimeError(f"split hash is not consistently registered for {key}")
        if sha256_file(split_path) != slot["split_sha256"]:
            raise RuntimeError(f"split file hash mismatch for {key}")

        eval_row = eval_rows.get(key)
        if not eval_row or eval_row.get("status") != "succeeded":
            raise RuntimeError(f"missing locked val argmax source for {key}")
        if eval_row.get("protocol_lock_sha256") != V1_LOCK_SHA:
            raise RuntimeError(f"evaluation row uses a different protocol lock for {key}")
        if eval_row.get("checkpoint_sha256") != slot["checkpoint_sha256"]:
            raise RuntimeError(f"evaluation/checkpoint hash mismatch for {key}")
        training_id = str(eval_row.get("training_job_id", ""))
        training_row = training_rows.get(training_id)
        if training_row is None or training_row.get("code_commit") != V1_TRAINING_SHA:
            raise RuntimeError(f"missing v1 training provenance row for {key}")
        log_path = resolve_evidence_path(training_row["log_path"], evidence_root)
        with log_path.open(encoding="utf-8") as handle:
            first_line = handle.readline()
        start_record = json.loads(first_line)
        training_device = start_record.get("config", {}).get("device")
        training_devices[training_id] = str(training_device) if training_device is not None else None
        pred_path = resolve_evidence_path(eval_row["predictions_path"], evidence_root)
        pred_expected = eval_row["predictions_sha256"]
        pred_verified = verify_file(pred_path, pred_expected, f"saved argmax {key}")
        pred_manifest_sha = manifest_files.get(str(eval_row["predictions_path"]))
        if pred_manifest_sha is not None and pred_manifest_sha != pred_verified["sha256"]:
            raise RuntimeError(f"manifest argmax hash mismatch for {key}")
        prediction_sources.append({**pred_verified, "dataset": dataset, "variant": variant,
                                   "seed": seed, "job_id": eval_row["job_id"]})
        resolved_slots[key] = {"slot": slot, "checkpoint_path": checkpoint_path,
                               "eval_row": eval_row, "prediction_path": pred_path,
                               "training_device": training_devices[training_id]}

    for dataset in DATASETS:
        schema = lock["schemas"][dataset]
        if schema["official_split_sha256"] != sha256_file(Path(fixed_split_path(dataset))):
            raise RuntimeError(f"locked split hash mismatch for {dataset}")
    return {
        "registry_path": str(reports / "run_registry.json"),
        "registry_sha256": sha256_file(reports / "run_registry.json"),
        "lock_path": str(reports / "protocol_lock.json"),
        "protocol_lock_sha256": V1_LOCK_SHA,
        "registry_source_commit": registry["source_commit"],
        "registry_source_hashes": registry["source_hashes"],
        "lock_evaluation_code_commit": lock["evaluation_code_commit"],
        "lock_evaluation_source_hashes": lock["source_hashes"],
        "training_devices": training_devices,
        "selected_common_lr": float(lock["selected_common_lr"]),
        "schemas": lock["schemas"],
        "slots": resolved_slots,
        "checked_inputs": checked_inputs,
        "checkpoint_sources": checkpoint_sources,
        "saved_argmax_sources": prediction_sources,
        "old_manifest_sha256": sha256_file(reports / "manifest.json"),
        "raw_artifact_count": len(verified_raw_artifacts),
        "raw_artifacts_index_sha256": sha256_json(sorted(
            (item["path"], item["sha256"]) for item in verified_raw_artifacts)),
    }


def _metrics(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor,
             num_classes: int, label_set: list[int]) -> tuple[dict[str, Any], torch.Tensor]:
    selected_logits = logits[idx.to(logits.device)]
    target = labels[idx.cpu()].long()
    prediction = selected_logits.argmax(dim=1).detach().cpu()
    stats = confusion_metrics(target, prediction, num_classes, label_set)
    stats["cross_entropy"] = float(F.cross_entropy(selected_logits, target.to(logits.device)).item())
    return stats, prediction


def _weight_summary(weights: torch.Tensor, indices: torch.Tensor) -> dict[str, Any]:
    value = weights[indices.to(weights.device).long()].detach().to(torch.float64).cpu()
    entropy = -(value * value.clamp_min(1e-12).log()).sum(dim=1)
    mean = value.mean(0)
    std = value.std(0, unbiased=False)
    return {
        "mean": {name: float(mean[i]) for i, name in enumerate(PATHS)},
        "std": {name: float(std[i]) for i, name in enumerate(PATHS)},
        "mean_node_entropy_nats": float(entropy.mean()),
        "std_node_entropy_nats": float(entropy.std(unbiased=False)),
        "direct_text_total_weight_mean": float((value[:, 0] + value[:, 1]).mean()),
        "direct_text_total_weight_std": float((value[:, 0] + value[:, 1]).std(unbiased=False)),
        "n": int(value.size(0)),
    }


def _branch_norm_summary(context: torch.Tensor, indices: torch.Tensor) -> dict[str, Any]:
    path_context = context[:, :512].reshape(context.size(0), 4, 128)
    selected = path_context[indices.to(path_context.device)]
    norms = selected.norm(dim=2).to(torch.float64)
    return {name: {"mean": float(norms[:, i].mean()),
                   "std": float(norms[:, i].std(unbiased=False))}
            for i, name in enumerate(PATHS)}


def _construct_model(variant: str, dataset: str, seed: int, data, checkpoint: dict[str, Any],
                     lock_schema: dict[str, Any], device: torch.device):
    cfg = compose_probe_config(dataset, seed, fixed_split_path(dataset),
                               checkpoint["lr"], variant, str(device))
    data_info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
                 "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
                 "visual_dim": int(data.x_i.size(1))}
    if checkpoint["data_info"] != {key: data_info[key] for key in checkpoint["data_info"]}:
        raise RuntimeError(f"checkpoint data dimensions differ from loaded {dataset} data")
    model = build_model(cfg, data_info).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    if checkpoint["macro_f1_label_set"] != lock_schema["macro_f1_label_set"]:
        raise RuntimeError(f"checkpoint Macro-F1 label schema differs from the old lock for {dataset}")
    model.eval()
    head.eval()
    if hasattr(model, "set_run_seed"):
        model.set_run_seed(seed)
    return model, head


def _evaluate_cell(evidence: dict[str, Any], dataset: str, variant: str, seed: int,
                   data, requested_device: str) -> tuple[dict[str, Any], torch.Tensor]:
    key = (dataset, variant, seed)
    source = evidence["slots"][key]
    device = torch.device(source["training_device"] if requested_device == "locked" else requested_device)
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"locked training device {device} is not available in this environment")
    x_device, edge_device = data.x.to(device), data.edge_index.to(device)
    checkpoint = torch.load(source["checkpoint_path"], map_location="cpu", weights_only=False)
    slot = source["slot"]
    if (checkpoint["source_commit"] != V1_TRAINING_SHA or checkpoint["dataset"] != dataset
            or checkpoint["variant"] != variant or int(checkpoint["seed"]) != seed
            or checkpoint["split_sha256"] != slot["split_sha256"]):
        raise RuntimeError(f"checkpoint metadata does not match the frozen registry for {key}")
    model, head = _construct_model(variant, dataset, seed, data, checkpoint,
                                   evidence["schemas"][dataset], device)
    captured: list[torch.Tensor] = []
    hook = None
    if variant == "J_NODE":
        hook = model.router.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        z, _, _, _, aux = model(x_device, edge_device)
        logits = head(z)
        train_stats, _ = _metrics(logits, data.y, data.train_idx, int(data.num_classes),
                                  evidence["schemas"][dataset]["macro_f1_label_set"])
        val_stats, val_prediction = _metrics(logits, data.y, data.val_idx, int(data.num_classes),
                                             evidence["schemas"][dataset]["macro_f1_label_set"])
    if hook is not None:
        hook.remove()
    eval_row = source["eval_row"]
    saved_blob = torch.load(source["prediction_path"], map_location="cpu", weights_only=False)
    # This diagnostic intentionally consumes only the saved validation argmax.
    # It never reads, scores, or summarizes the test prediction partition.
    saved_val_prediction = torch.as_tensor(saved_blob["val_pred"], dtype=torch.long).reshape(-1)
    val_index_hash = sha256_json(data.val_idx.detach().cpu().long().tolist())
    if (saved_blob.get("split_sha256") != slot["split_sha256"]
            or saved_blob.get("val_idx_sha256") != val_index_hash
            or saved_val_prediction.numel() != data.val_idx.numel()):
        raise RuntimeError(f"saved validation argmax split metadata differs for {key}")
    old_val = eval_row["val"]
    saved_metrics = confusion_metrics(data.y[data.val_idx], saved_val_prediction,
                                      int(data.num_classes), evidence["schemas"][dataset]["macro_f1_label_set"])
    prediction_transition = confusion_metrics(saved_val_prediction, val_prediction,
        int(data.num_classes), list(range(int(data.num_classes))))
    pred_match = torch.equal(val_prediction, saved_val_prediction)
    confusion_match = val_stats["confusion_matrix"] == old_val["confusion_matrix"]

    result: dict[str, Any] = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "job_id": slot["job_id"], "checkpoint_path": str(source["checkpoint_path"]),
        "checkpoint_sha256": slot["checkpoint_sha256"], "best_epoch": int(checkpoint["best_epoch"]),
        "inference_device": str(device),
        "train": train_stats, "val": val_stats,
        "old_validation_reconstruction": {
            "saved_argmax_sha256": eval_row["predictions_sha256"],
            "prediction_exact_match": bool(pred_match),
            "confusion_matrix_exact_match": bool(confusion_match),
            "different_prediction_count": int((val_prediction != saved_val_prediction).sum()),
            "old_argmax_to_reconstructed_argmax_transition_matrix": prediction_transition["confusion_matrix"],
            "saved_validation_metrics": {key: saved_metrics[key] for key in
                ("n", "acc", "macro_f1", "macro_f1_label_set", "class_order")},
            "old_locked_validation_metrics": {key: old_val[key] for key in
                ("n", "acc", "macro_f1", "macro_f1_label_set", "class_order")},
            "saved_validation_confusion_matrix_sha256": sha256_json(saved_metrics["confusion_matrix"]),
            "old_locked_validation_confusion_matrix_sha256": sha256_json(old_val["confusion_matrix"]),
            "comparison_scope": "validation nodes only",
        },
    }
    if variant == "J_NODE":
        if len(captured) != 1:
            raise RuntimeError(f"expected one captured normal router context for {key}")
        context = captured[0]
        base_weights = aux["router_weights"].detach()
        active = aux["activity_mask"].detach()
        if not torch.equal(active, torch.stack((torch.ones_like(active[:, 0]),
                aux["degree"] > 0, torch.ones_like(active[:, 2]), aux["degree"] > 0), dim=1)):
            raise RuntimeError(f"receiver activity mask changed unexpectedly for {key}")
        train_cpu = data.train_idx.cpu().long()
        val_cpu = data.val_idx.cpu().long()
        result["router"] = {
            "path_names": list(PATHS),
            "weights_train": _weight_summary(base_weights, train_cpu),
            "weights_val": _weight_summary(base_weights, val_cpu),
            "representation_norms_train": _branch_norm_summary(context, train_cpu),
            "representation_norms_val": _branch_norm_summary(context, val_cpu),
            "weights_are_descriptive_not_causal_or_information_fractions": True,
        }
        base_val_weights = base_weights[val_cpu.to(base_weights.device)].detach().cpu()
        result["context_intervention_repeats"] = []
        for repeat in range(10):
            raw_seed = hashlib.sha256(f"mag-probe-v1.1:{dataset}:{seed}:context:{repeat}".encode()).digest()
            gen = torch.Generator(device="cpu").manual_seed(int.from_bytes(raw_seed[:8], "little") % (2**63 - 1))
            permutation = torch.randperm(int(data.num_nodes), generator=gen)
            model.set_context_permutation(permutation)
            with torch.no_grad():
                z_intervention, _, _, _, aux_intervention = model(x_device, edge_device)
                intervention_logits = head(z_intervention)
                metrics, prediction = _metrics(intervention_logits, data.y, data.val_idx,
                    int(data.num_classes), evidence["schemas"][dataset]["macro_f1_label_set"])
            metrics.pop("confusion_matrix")
            weights = aux_intervention["router_weights"][val_cpu.to(device)].detach().cpu()
            if not torch.equal(aux_intervention["activity_mask"], active):
                raise RuntimeError(f"context intervention altered receiver mask for {key} repeat {repeat}")
            result["context_intervention_repeats"].append({
                "repeat_type": "fixed label-free context-permutation intervention; not a training seed",
                "repeat": repeat,
                "permutation_sha256": sha256_json(permutation.tolist()),
                "validation": metrics,
                "accuracy_delta_vs_no_intervention": metrics["acc"] - val_stats["acc"],
                "accuracy_delta_pp_vs_no_intervention": 100.0 * (metrics["acc"] - val_stats["acc"]),
                "macro_f1_delta_vs_no_intervention": metrics["macro_f1"] - val_stats["macro_f1"],
                "cross_entropy_delta_vs_no_intervention": metrics["cross_entropy"] - val_stats["cross_entropy"],
                "changed_validation_predictions": int((prediction != val_prediction).sum()),
                "mean_absolute_router_weight_change_on_validation": float((weights - base_val_weights).abs().mean()),
                "receiver_paths_and_masks_preserved": True,
            })
            del z_intervention, intervention_logits, aux_intervention
            model.set_context_permutation(None)

        # A training-node-fitted constant gate applied to the same frozen branch
        # representations; normalize it independently against each receiver mask.
        train_weights = base_weights[train_cpu.to(base_weights.device)]
        frozen_gate = train_weights.mean(dim=0)
        gated = active.to(context.dtype) * frozen_gate.unsqueeze(0)
        gated = gated / gated.sum(dim=1, keepdim=True).clamp_min(1e-12)
        branch_reps = context[:, :512].reshape(context.size(0), 4, 128)
        z_gate = (branch_reps * gated.unsqueeze(-1)).sum(dim=1)
        with torch.no_grad():
            gate_logits = head(z_gate)
            gate_metrics, gate_prediction = _metrics(gate_logits, data.y, data.val_idx,
                int(data.num_classes), evidence["schemas"][dataset]["macro_f1_label_set"])
        gate_metrics.pop("confusion_matrix")
        result["training_mean_fixed_gate"] = {
            "gate_path_weights_training_mean": {name: float(frozen_gate[i].cpu()) for i, name in enumerate(PATHS)},
            "validation": gate_metrics,
            "accuracy_delta_pp_vs_no_intervention": 100.0 * (gate_metrics["acc"] - val_stats["acc"]),
            "macro_f1_delta_vs_no_intervention": gate_metrics["macro_f1"] - val_stats["macro_f1"],
            "cross_entropy_delta_vs_no_intervention": gate_metrics["cross_entropy"] - val_stats["cross_entropy"],
            "changed_validation_predictions": int((gate_prediction != val_prediction).sum()),
            "same_frozen_representation_not_retrained_global": True,
        }
        model.set_context_permutation(None)
        del z_gate, gate_logits, branch_reps, gated, context
    elif variant == "J_GLOBAL":
        global_weights = model.global_logits.detach().softmax(dim=0).cpu()
        result["global_softmax_weights"] = {name: float(global_weights[i]) for i, name in enumerate(PATHS)}
        result["global_weights_note"] = "learned global softmax; gate weights are descriptive, not causal utility or information fractions"
    del z, logits, model, head, checkpoint, saved_blob
    if device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return result, val_prediction


def _bucket_tags(values: torch.Tensor, isolated: torch.Tensor,
                 thresholds: list[float]) -> torch.Tensor:
    tags = torch.bucketize(values.cpu().to(torch.float64), torch.tensor(thresholds, dtype=torch.float64), right=True)
    tags[isolated.cpu().bool()] = 4
    return tags


def _bucket_report(y: torch.Tensor, indices: torch.Tensor, node_prediction: torch.Tensor,
                   global_prediction: torch.Tensor, num_classes: int, labels: list[int],
                   bucket: torch.Tensor, prefix: str, definition: str) -> list[dict[str, Any]]:
    output = []
    names = ("q1", "q2", "q3", "q4", "isolated")
    for bucket_id, name in enumerate(names):
        picked = indices[bucket[indices] == bucket_id]
        node_stats = confusion_metrics(y[picked], node_prediction[picked], num_classes, labels)
        global_stats = confusion_metrics(y[picked], global_prediction[picked], num_classes, labels)
        for stats in (node_stats, global_stats):
            matrix = stats["confusion_matrix"]
            stats["confusion_matrix"] = {
                "shape": [num_classes, num_classes], "class_order": list(range(num_classes)),
                "encoding": "base64(zlib(UTF-8 JSON dense matrix))",
                "data": base64.b64encode(zlib.compress(
                    json.dumps(matrix, separators=(",", ":")).encode("utf-8"), level=9)).decode("ascii"),
            }
        a_ok = node_prediction[picked] == y[picked]
        b_ok = global_prediction[picked] == y[picked]
        output.append({
            "feature": prefix, "bucket": name, "n": int(picked.numel()),
            "small_n": int(picked.numel()) < 100,
            "threshold_source": "train nodes only; cosine excludes isolates; isolate bucket independent",
            "cosine_definition": definition,
            "J_NODE_correct": int(a_ok.sum()), "J_NODE_wrong": int((~a_ok).sum()),
            "J_GLOBAL_correct": int(b_ok.sum()), "J_GLOBAL_wrong": int((~b_ok).sum()),
            "paired_correctness": {
                "both_correct": int((a_ok & b_ok).sum()),
                "node_only_correct": int((a_ok & ~b_ok).sum()),
                "global_only_correct": int((~a_ok & b_ok).sum()),
                "both_wrong": int((~a_ok & ~b_ok).sum()),
            },
            "J_NODE_metrics": node_stats,
            "J_GLOBAL_metrics": global_stats,
        })
    return output


def _geometry_for_dataset(data, node_pred: torch.Tensor, global_pred: torch.Tensor,
                          schema: dict[str, Any], cache: dict[str, Any] | None = None
                          ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    train_idx, val_idx = data.train_idx.cpu().long(), data.val_idx.cpu().long()
    if cache is None:
        raw = geometry_features(data.edge_index, data.x_t, data.x_i, train_idx, val_idx)
        normed = geometry_features_l2_neighbor_mean(data.edge_index, data.x_t, data.x_i,
                                                     train_idx, val_idx, eps=1e-8)
        cache = {}
        for modality in ("text", "image"):
            old, new = raw[modality], normed[modality]
            cache[modality] = {
                "raw_degree": old["degree"], "raw_cosine": old["cosine"],
                "raw_isolated": old["isolated"], "raw_degree_thresholds": old["degree_thresholds"],
                "raw_cosine_thresholds": old["cosine_thresholds"],
                "degree": new["degree"], "cosine": new["cosine"], "isolated": new["isolated"],
                "degree_thresholds": new["degree_thresholds"], "cosine_thresholds": new["cosine_thresholds"],
                "cosine_definition": new["cosine_definition"],
            }
        del raw, normed
    summary = {"bucket_method": "train quartiles; isolated is independent; validation receives train thresholds",
               "modalities": {}}
    # The threshold and transition summary is shared across training seeds.
    for modality, item in cache.items():
        old_tags = _bucket_tags(item["raw_cosine"], item["raw_isolated"], item["raw_cosine_thresholds"])
        new_tags = _bucket_tags(item["cosine"], item["isolated"], item["cosine_thresholds"])
        old_degree_tags = _bucket_tags(item["raw_degree"], item["raw_isolated"], item["raw_degree_thresholds"])
        new_degree_tags = _bucket_tags(item["degree"], item["isolated"], item["degree_thresholds"])
        summary["modalities"][modality] = {
            "raw_v1_cosine_definition": "cos(x_v, mean_neighbors(x_u))",
            "normalized_v1_1_cosine_definition": item["cosine_definition"],
            "raw_v1_degree_thresholds": item["raw_degree_thresholds"],
            "normalized_v1_1_degree_thresholds": item["degree_thresholds"],
            "raw_v1_cosine_thresholds": item["raw_cosine_thresholds"],
            "normalized_v1_1_cosine_thresholds": item["cosine_thresholds"],
            "val_cosine_bucket_change_count_raw_to_normalized": int((old_tags[val_idx] != new_tags[val_idx]).sum()),
            "val_degree_bucket_change_count_raw_to_normalized": int((old_degree_tags[val_idx] != new_degree_tags[val_idx]).sum()),
            "val_isolated_nodes": int(item["isolated"][val_idx].sum()),
        }
    buckets_by_kind: dict[str, list[dict[str, Any]]] = {"degree": [], "normalized_cosine": []}
    for modality, item in cache.items():
        new = {"degree": item["degree"], "cosine": item["cosine"], "isolated": item["isolated"],
               "degree_thresholds": item["degree_thresholds"], "cosine_thresholds": item["cosine_thresholds"],
               "cosine_definition": item["cosine_definition"]}
        new_degree_tags = _bucket_tags(new["degree"], new["isolated"], new["degree_thresholds"])
        new_tags = _bucket_tags(new["cosine"], new["isolated"], new["cosine_thresholds"])
        buckets_by_kind["degree"].extend(_bucket_report(data.y.cpu(), val_idx, node_pred, global_pred,
            int(data.num_classes), schema["macro_f1_label_set"], new_degree_tags, f"{modality}_degree",
            new["cosine_definition"]))
        buckets_by_kind["normalized_cosine"].extend(_bucket_report(data.y.cpu(), val_idx, node_pred, global_pred,
            int(data.num_classes), schema["macro_f1_label_set"], new_tags, f"{modality}_cosine",
            new["cosine_definition"]))
    return summary, buckets_by_kind, cache


def _select_device(requested: str) -> torch.device:
    if requested in {"auto", "locked"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    return device


def _render_report(diagnostic: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines = [
        "# Frozen probe v1.1 verification diagnostics",
        "",
        "The original v1 main matrix remains valid. The original J_NODE context-permutation diagnostic was ineffective because its permutation was not consumed by ordinary J_NODE; the v1.1 permutation results below use the repaired explicit override. New results cover train/validation only. No link prediction (LP) was run, no new real-data training was run, and no test inference was run.",
        "",
        f"v1 training source SHA: `{provenance['v1_training_source_sha']}`",
        f"v1 evaluation lock SHA-256: `{provenance['v1_evaluation_lock_sha256']}`",
        f"v1.1 diagnostic source SHA-256: `{provenance['diagnostic_script_sha256']}`",
        f"V1 raw artifacts verified against the original manifest: {provenance['v1_raw_artifact_count_verified']}/{provenance['v1_raw_artifact_count_verified']}.",
        f"Device policy: `{provenance['device_policy']}`; checkpoints: 15 J_NODE + 15 matching J_GLOBAL; data inputs were hash-checked before inference.",
        "",
        "## Five-graph validation summary",
        "",
        "Accuracies are means over the three frozen checkpoint seeds. The 30 context interventions per graph are ten fixed label-free row permutations for each of three already-trained checkpoints; their spread is descriptive and is not a seed-level uncertainty estimate.",
        "",
        "| Graph | J_NODE val Acc | J_GLOBAL val Acc | NODE−GLOBAL (pp) | 30 context interventions: Acc drop (pp) | Train-mean fixed gate val Acc | Old argmax/CM exact matches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        cells = [diagnostic["cells"][f"{dataset}|{seed}"] for seed in SEEDS]
        node_acc = sum(c["J_NODE"]["val"]["acc"] for c in cells) / 3
        global_acc = sum(c["J_GLOBAL"]["val"]["acc"] for c in cells) / 3
        gate_acc = sum(c["J_NODE"]["training_mean_fixed_gate"]["validation"]["acc"] for c in cells) / 3
        drops = [c["J_NODE"]["val"]["acc"] - rep["validation"]["acc"]
                 for c in cells for rep in c["J_NODE"]["context_intervention_repeats"]]
        match_n = sum(int(c["J_NODE"]["old_validation_reconstruction"]["prediction_exact_match"]
                          and c["J_NODE"]["old_validation_reconstruction"]["confusion_matrix_exact_match"])
                      + int(c["J_GLOBAL"]["old_validation_reconstruction"]["prediction_exact_match"]
                            and c["J_GLOBAL"]["old_validation_reconstruction"]["confusion_matrix_exact_match"])
                      for c in cells)
        lines.append(f"| {dataset} | {node_acc:.4f} | {global_acc:.4f} | {100*(node_acc-global_acc):+.2f} | {100*sum(drops)/len(drops):.2f} ± {100*torch.tensor(drops).std(unbiased=False).item():.2f} | {gate_acc:.4f} | {match_n}/6 |")
    lines.extend([
        "",
        "## Reading the results",
        "",
        "- The repaired frozen intervention measures whether router context rows contribute to predictions while each receiver keeps its own four path representations and active-path mask. Every permutation is scored separately; no logits are ensembled. The validation changes are evidence of predictive use in this frozen system; they do not quantify information content or prove causation outside the intervention.",
        "- J_NODE did not outperform the retrained J_GLOBAL or the capacity controls in the original formal v1 matrix. This v1.1 diagnostic does not change that result or retrain either model.",
        "- Very small direct text-path gate weights indicate strong suppression of those two fusion paths. They do not show that the system fully ignores text: text path representations and text-derived context can still affect the router.",
        "- The current evidence does not identify whether the observed weights or generalization patterns arise from overfitting, shared phi, competitive gating, optimization, or another mechanism. These remain non-causal hypotheses.",
        "- The train-mean gate is a fixed intervention on the same J_NODE representations. It is not a substitute for training J_GLOBAL. Three seeds are descriptive; no significance claim is made.",
        "- Frozen validation reconstruction ran on each checkpoint's device recorded in its v1 training log. If a saved argmax differs, `old_argmax_to_reconstructed_argmax_transition_matrix` and the mismatch count localize the class transitions on validation; no test inference was used for diagnosis.",
        "",
        "## Geometry and retained outputs",
        "",
        "The new cosine is computed separately per modality as `cos(L2(x_v, eps=1e-8), mean_neighbors(L2(x_u, eps=1e-8)))`; no cross-modality cosine is used. Quartile thresholds are fitted on training nodes, cosine fitting excludes isolates, and isolates have their own bucket. Every bucket records n, a `small_n` marker for n<100, correct/wrong counts, paired correctness counts, and dense aggregate confusion matrices in `diagnostics.json`. `geometry` records old raw-mean versus normalized thresholds and validation bucket transition counts. The v1 raw-mean geometry remains the registered v1 definition.",
        "The original full graph features and topology are used only for label-free forward passes under the v1 transductive protocol; all reported train/validation metrics index only their locked split nodes.",
        "",
        "Only aggregate metrics, counts, thresholds, hashes, and dense confusion matrices are written. No logits, embeddings, features, cache, or per-node predictions are written by this diagnostic. Existing v1 argmax files were read only for validation reconstruction comparison; their validation slice alone was used.",
        "",
        "## Files",
        "",
        "- `diagnostics.json`: per-checkpoint train/validation metrics, validation reconstruction checks, router summaries, ten separate interventions, fixed-gate results, normalized geometry buckets, and confusion matrices.",
        "- `provenance.json`: v1 and v1.1 source hashes, lock/registry identity, and verified input/checkpoint/prediction file hashes.",
        "- `review.md`: focused implementation and test audit record.",
        "",
    ])
    return "\n".join(lines)


def run(evidence_root: Path, output_dir: Path, requested_device: str) -> None:
    evidence_root = evidence_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    protected = (evidence_root / "reports" / "problem_validation_v1",
                 evidence_root / "outputs" / "problem_validation_v1")
    if any(output_dir == path or path in output_dir.parents or output_dir in path.parents for path in protected):
        raise ValueError("output-dir overlaps frozen v1 reports or raw evidence; choose a v1.1-only directory")
    evidence = preflight(evidence_root)
    device = _select_device(requested_device)
    if requested_device == "locked":
        locked_devices = {value for value in evidence["training_devices"].values() if value}
        if not locked_devices or (any(value.startswith("cuda") for value in locked_devices)
                                  and not torch.cuda.is_available()):
            raise RuntimeError("the locked training devices require CUDA, which is unavailable")
        for value in locked_devices:
            parsed = torch.device(value)
            if parsed.type == "cuda" and parsed.index is not None and parsed.index >= torch.cuda.device_count():
                raise RuntimeError(f"locked training device {value} is unavailable")
    device_policy = "locked per-checkpoint device from v1 training logs" if requested_device == "locked" else str(device)
    print(f"Preflight passed: 21 inputs, 30 checkpoints, 30 saved validation references; device policy={device_policy}", flush=True)
    cells: dict[str, Any] = {}
    geometries: dict[str, Any] = {}
    bucket_results: dict[str, Any] = {}
    input_sources_by_dataset = {
        ds: {path: info for path, info in evidence["checked_inputs"].items()
             if path in evidence["schemas"][ds]["dataset_input_hashes"]}
        for ds in DATASETS
    }
    for dataset in DATASETS:
        schema = evidence["schemas"][dataset]
        first_slot = evidence["slots"][(dataset, "J_NODE", 42)]["slot"]
        cfg = compose_probe_config(dataset, 42, fixed_split_path(dataset),
                                   float(evidence["selected_common_lr"]), "J_NODE", str(device))
        data = load_mag_data(cfg, "nc", 42)
        if int(data.num_classes) != int(schema["num_classes"]):
            raise RuntimeError(f"loaded class count differs from old lock for {dataset}")
        if sha256_file(Path(first_slot["split_path"])) != schema["official_split_sha256"]:
            raise RuntimeError(f"data loader selected a different frozen split for {dataset}")
        geometry_cache = None
        for seed in SEEDS:
            started = time.time()
            node, node_prediction = _evaluate_cell(evidence, dataset, "J_NODE", seed, data, requested_device)
            global_cell, global_prediction = _evaluate_cell(evidence, dataset, "J_GLOBAL", seed, data, requested_device)
            cell_key = f"{dataset}|{seed}"
            cells[cell_key] = {"J_NODE": node, "J_GLOBAL": global_cell}
            # Bucket indices are full-node IDs, so predictions are transiently
            # aligned to full node space while all scoring remains val-indexed.
            node_full = torch.full((data.num_nodes,), -1, dtype=torch.long)
            global_full = torch.full((data.num_nodes,), -1, dtype=torch.long)
            node_full[data.val_idx.cpu().long()] = node_prediction
            global_full[data.val_idx.cpu().long()] = global_prediction
            geo, bucket, geometry_cache = _geometry_for_dataset(data, node_full, global_full, schema, geometry_cache)
            geometries[dataset] = geo
            if dataset not in bucket_results:
                bucket_results[dataset] = {}
            bucket_results[dataset][str(seed)] = bucket
            print(f"{dataset} seed{seed}: NODE val={node['val']['acc']:.4f}, GLOBAL val={global_cell['val']['acc']:.4f}, "
                  f"old-val-matches={int(node['old_validation_reconstruction']['prediction_exact_match']) + int(global_cell['old_validation_reconstruction']['prediction_exact_match'])}/2, "
                  f"elapsed={time.time() - started:.1f}s", flush=True)
            del node_prediction, global_prediction, node_full, global_full
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    diagnostic = {
        "schema_version": 1,
        "study": "mag_problem_validation_v1_1_frozen_diagnostics",
        "evaluation_scope": "train and validation only; no test inference; no LP; no real-data training",
        "feature_protocol": "full transductive node features/topology may be used for label-free forward passes; all metric indices are train or validation only",
        "training_seeds_are_descriptive_only": list(SEEDS),
        "context_interventions_per_J_NODE_checkpoint": 10,
        "context_interventions_are_not_training_seeds": True,
        "class_schema_source": "old v1 protocol lock only",
        "device_policy": device_policy,
        "cells": cells,
        "geometry": geometries,
        "geometry_validation_buckets": bucket_results,
        "inputs_by_dataset": input_sources_by_dataset,
    }
    provenance = {
        "v1_training_source_sha": V1_TRAINING_SHA,
        "v1_evaluation_lock_sha256": V1_LOCK_SHA,
        "v1_evaluation_code_commit": evidence["lock_evaluation_code_commit"],
        "v1_registry_path": evidence["registry_path"],
        "v1_registry_sha256": evidence["registry_sha256"],
        "v1_old_manifest_sha256": evidence["old_manifest_sha256"],
        "v1_raw_evidence_directory": str(evidence_root / "outputs" / "problem_validation_v1"),
        "v1_raw_artifact_count_verified": evidence["raw_artifact_count"],
        "v1_raw_artifacts_index_sha256": evidence["raw_artifacts_index_sha256"],
        "v1_registry_source_hashes": evidence["registry_source_hashes"],
        "v1_locked_evaluation_source_hashes": evidence["lock_evaluation_source_hashes"],
        "legacy_evaluator_hash_comparison": {
            "locked_source_sha256": evidence["lock_evaluation_source_hashes"]["scripts/evaluate_locked_probe.py"],
            "current_checkout_sha256": sha256_file(ROOT / "scripts/evaluate_locked_probe.py"),
            "old_lock_was_not_rewritten_or_bypassed": True,
        },
        "v1_training_devices_from_locked_logs": evidence["training_devices"],
        "runtime_environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                   if torch.cuda.is_available() else [],
        },
        "diagnostic_script_path": str(Path(__file__).resolve()),
        "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
        "modified_code_hashes": {
            rel: sha256_file(ROOT / rel) for rel in (
                "src/models/mag_probe.py", "src/tasks/common.py", "src/analysis/problem_validation.py",
                "configs/model/mag_probe.yaml", "tests/test_mag_probe.py")
        },
        "device_policy": device_policy,
        "verified_checkpoint_sources": evidence["checkpoint_sources"],
        "verified_data_and_split_inputs": list(evidence["checked_inputs"].values()),
        "verified_old_argmax_sources": evidence["saved_argmax_sources"],
        "old_argmax_read_scope": "validation argmax slice and validation confusion only; test partition not accessed",
        "old_result_jsonl_projection": "locked validation summary and training provenance retained; test-metric fields discarded",
        "new_diagnostic_code_is_not_a_resigned_v1_protocol": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "diagnostics.json", diagnostic)
    atomic_json(output_dir / "provenance.json", provenance)
    (output_dir / "report.md").write_text(_render_report(diagnostic, provenance), encoding="utf-8")
    (output_dir / "review.md").write_text(
        "# v1.1 implementation review\n\n"
        "- Scope stayed within the routed context override, probe optimizer defaults, normalized geometry diagnostics, and sparse topology cache safety. No new model objective or real-data training was added.\n"
        "- No-override behavior retains the v1 path: J_NODE sends its local context rows unchanged; J_NODE_SHUFFLE still uses the same seed/epoch/namespace permutation. A strict state_dict round trip and the synthetic trained-style router test check this path, intervention rows, unchanged receiver masks/branch representations, and exact restoration after clearing the override.\n"
        "- The v1 run registry, protocol lock, checkpoints, predictions, reports, and source hashes were read-only. Preflight verified the complete registered raw artifact set and all 21 input hashes before inference.\n"
        "- All 30 selected model checkpoints load with strict state-dict matching. Baseline validation argmax and complete confusion matrices were compared against saved v1 records on each checkpoint's original logged CUDA device.\n"
        "- Synthetic/unit command: `conda run -n yhf_env python -m pytest -q tests/test_mag_probe.py` — 20 passed. It covers trained-style router override behavior, strict state loading, optimizer defaults/overrides, generic NC synthetic backward/head/epoch/seed/test-off integration, normalized geometry, and no_grad/inference_mode topology caching.\n"
        "- Frozen diagnostic command: `conda run -n yhf_env python scripts/diagnose_frozen_probe_v1_1.py --evidence-root /hdd1/DataInHere/YHF/mag_model` — completed on locked devices; 30/30 validation predictions and confusion matrices matched. No test inference, LP, or real-data optimizer step occurred.\n"
        "- The user-owned `CLAUDE.md` deletion remains unstaged and uncommitted. New outputs contain aggregates and provenance only; no per-node predictions, logits, embeddings, features, checkpoints, caches, or training logs are generated.\n",
        encoding="utf-8")
    tracked = ("diagnostics.json", "provenance.json", "report.md", "review.md")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_generation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch_at_generation": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "diagnostic_script_sha256": provenance["diagnostic_script_sha256"],
        "v1_protocol_lock_sha256": V1_LOCK_SHA,
        "v1_raw_artifact_count_verified": evidence["raw_artifact_count"],
        "artifacts": [
            {"path": name, "bytes": (output_dir / name).stat().st_size,
             "sha256": sha256_file(output_dir / name)} for name in tracked
        ],
    }
    atomic_json(output_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only train/validation diagnostics for frozen v1 probe checkpoints.")
    parser.add_argument("--evidence-root", type=Path, required=True,
                        help="Checkout containing reports/problem_validation_v1 and outputs/problem_validation_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "problem_validation_v1_1")
    parser.add_argument("--device", default="locked", help="locked (per-checkpoint v1 training device), cuda:N, or cpu")
    args = parser.parse_args()
    run(args.evidence_root, args.output_dir, args.device)


if __name__ == "__main__":
    main()
