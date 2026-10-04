from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.run_problem_validation import (  # noqa: E402
    DATASETS, RAW, REGISTRY_PATH, REPORTS, RESULTS_PATH, _training_config_hash,
    append_result, atomic_json, config_for, read_registry, runtime_job, select_idle_gpu,
    source_hashes, verify_dataset_inputs,
)
from src.analysis.problem_validation import (  # noqa: E402
    bucket_id, bucket_metrics, compose_probe_config, confusion_metrics,
    correctness_pair, fixed_split_path, geometry_features, modality_complementarity,
    sha256_file, sha256_json, train_train_homophily, validate_loaded_data,
)
from src.data import load_mag_data  # noqa: E402
from src.models.factory import build_model  # noqa: E402


LOCK_PATH = REPORTS / "protocol_lock.json"
PRED_DIR = RAW / "predictions"


def jsonl_rows(path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def load_data(dataset: str, seed: int, lr: float, device: str, cache: dict[str, Any]):
    if dataset not in cache:
        cfg = compose_probe_config(dataset, seed, fixed_split_path(dataset), lr, "J_FIXED", device)
        cache[dataset] = load_mag_data(cfg, "nc", seed)
    return cache[dataset]


def labels_for(data) -> list[int]:
    idx = torch.cat((data.train_idx, data.val_idx, data.test_idx)).long()
    observed = torch.unique(data.y[idx]).tolist()
    return sorted(int(x) for x in observed if 0 <= int(x) < int(data.num_classes))


def lock_payload(registry: dict[str, Any], schema_by_dataset: dict[str, Any]) -> dict[str, Any]:
    matrix = []
    for slot in registry["primary_slots"]:
        matrix.append({key: slot.get(key) for key in (
            "job_id", "phase", "dataset", "variant", "seed", "lr", "split_path",
            "split_sha256", "dataset_input_hashes", "config_hash", "fingerprint", "status", "checkpoint_path",
            "checkpoint_sha256", "reuses_training_job_id")})
    return {
        "schema_version": 1,
        "study": registry["study"], "branch": registry["branch"],
        "training_source_commit": registry["source_commit"],
        "evaluation_code_commit": registry["source_commit"],
        "source_hashes": source_hashes(),
        "registry_source_hashes": registry["source_hashes"],
        "selected_common_lr": registry["selected_common_lr"],
        "training_attempt_count_at_lock": registry["attempt_count"],
        "primary_matrix": matrix,
        "schemas": schema_by_dataset,
        "evaluation_schema": {"confusion_matrix": "complete num_classes x num_classes; rows=true; columns=predicted; class order 0..C-1",
            "accuracy": "trace(confusion)/n", "macro_f1": "derive TP/FP/FN from complete confusion row/column sums; average only fixed macro_f1_label_set",
            "macro_f1_label_set_source": "union of labels observed on official supervised train/val/test node indices; fixed before any test prediction and used identically for all models"},
        "diagnostic_definitions": {
            "error_complementarity": ["both_correct", "text_only_correct", "image_only_correct", "both_wrong", "prediction_disagreement", "oracle=1-both_wrong/n (label-using upper bound)"],
            "paired_correctness": "2x2 paired correctness counts on identical node IDs; accuracy deltas in [0,1] internally and pp in report",
            "geometry": "degree and raw same-modality ego-to-neighbor cosine quartiles fit on training nodes only; cosine fit among nonisolated training nodes; apply thresholds to validation nodes; isolated bucket separate; small_n when n<100",
            "homophily": "only edges with both endpoints in train; excludes self-loops; uses train labels only",
            "router": "mean active path weights, argmax frequency, per-node entropy, validation degree-bin means, inactive-neighborhood mask violations",
            "shuffle_perturbation": "ten deterministic fixed router-context permutations for each trained J_NODE checkpoint; each val prediction reported separately; no logit ensemble"},
        "selection_and_claim_limits": "No test result selects an LR/model/design; no significance claim with three seeds; node weights and path counts are descriptive only.",
    }


def freeze_protocol(registry: dict[str, Any], data_cache: dict[str, Any], device: str) -> dict[str, Any]:
    if registry.get("selected_common_lr") is None:
        raise RuntimeError("common LR has not been selected")
    if source_hashes() != registry["source_hashes"]:
        raise RuntimeError("source code/config hashes differ from the training registry")
    schema = {}
    lr = float(registry["selected_common_lr"])
    for ds in DATASETS:
        verify_dataset_inputs(ds, registry)
        data = load_data(ds, 42, lr, device, data_cache)
        audit = validate_loaded_data(data)
        schema[ds] = {"num_classes": int(data.num_classes),
                      "class_order": list(range(int(data.num_classes))),
                      "macro_f1_label_set": labels_for(data),
                      "official_split_sha256": sha256_file(fixed_split_path(ds)),
                      "dataset_input_hashes": next(item["input_file_hashes"] for item in registry["datasets"] if item["dataset"] == ds),
                      "split_sizes": [int(data.train_idx.numel()), int(data.val_idx.numel()), int(data.test_idx.numel())],
                      "audit_feature_order": audit["feature_order"]}
    payload = lock_payload(registry, schema)
    protocol_hash = sha256_json(payload)
    candidate = {**payload,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_lock_sha256": protocol_hash}
    if LOCK_PATH.exists():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if existing.get("protocol_lock_sha256") != protocol_hash:
            raise RuntimeError("existing protocol lock differs; refusing to evaluate under a changed test protocol")
        return existing
    atomic_json(LOCK_PATH, candidate)
    return candidate


def result_by_job(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if row.get("record_type") == "training" and row.get("status") == "succeeded":
            out[row["job_id"]] = row
    return out


def eval_fingerprint(lock: dict[str, Any], slot: dict[str, Any], checkpoint_sha: str) -> str:
    return sha256_json({"protocol_lock_sha256": lock["protocol_lock_sha256"],
                        "job_id": slot["job_id"], "checkpoint_sha256": checkpoint_sha,
                        "split_sha256": slot["split_sha256"], "config_hash": slot["config_hash"]})


def find_existing_eval(rows: list[dict[str, Any]], job_id: str, fingerprint: str):
    return next((row for row in reversed(rows) if row.get("record_type") == "locked_evaluation" and
                 row.get("job_id") == job_id and row.get("evaluation_fingerprint") == fingerprint and
                 row.get("status") == "succeeded"), None)


def predictions_path(job_id: str, fingerprint: str) -> Path:
    return PRED_DIR / f"{job_id}-{fingerprint[:16]}.pt"


def summarize_router(aux: dict[str, Any], val_idx: torch.Tensor,
                     degree_thresholds: list[float]) -> dict[str, Any]:
    weights = aux["router_weights"].detach().cpu()[val_idx.cpu().long()]
    mask = aux["activity_mask"].detach().cpu()[val_idx.cpu().long()]
    degree = aux["degree"].detach().cpu()[val_idx.cpu().long()]
    eps = 1e-12
    entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=1)
    choice = weights.argmax(dim=1)
    bins = torch.bucketize(degree.to(torch.float64), torch.tensor(degree_thresholds, dtype=torch.float64), right=True)
    summaries = {}
    for key, selected in [(f"degree_q{i + 1}", bins == i) for i in range(4)]:
        if selected.any():
            summaries[key] = {"n": int(selected.sum()),
                "mean_weights": weights[selected].mean(0).tolist(),
                "choice_frequency": torch.bincount(choice[selected], minlength=4).float().div(selected.sum()).tolist(),
                "mean_entropy": float(entropy[selected].mean())}
    isolated = degree == 0
    if isolated.any():
        summaries["isolated"] = {"n": int(isolated.sum()),
            "mean_weights": weights[isolated].mean(0).tolist(),
            "choice_frequency": torch.bincount(choice[isolated], minlength=4).float().div(isolated.sum()).tolist(),
            "mean_entropy": float(entropy[isolated].mean())}
    violation = (~mask) & (weights > 1e-7)
    return {"n_val": int(weights.size(0)), "path_order": ["E_t", "N_t", "E_i", "N_i"],
            "mean_weights": weights.mean(0).tolist(),
            "choice_frequency": torch.bincount(choice, minlength=4).float().div(max(choice.numel(), 1)).tolist(),
            "mean_entropy": float(entropy.mean()) if entropy.numel() else None,
            "inactive_neighborhood_mask_violations": int(violation[:, [1, 3]].sum()),
            "buckets": summaries}


def fixed_permutation(seed: int, index: int, n: int) -> torch.Tensor:
    raw = hashlib.sha256(f"{seed}:node_validation_perturbation:{index}:v1".encode()).digest()
    value = int.from_bytes(raw[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(value)
    return torch.randperm(n, generator=generator)


def evaluate_slot(slot: dict[str, Any], lock: dict[str, Any], data, device_name: str,
                  train_rows: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    training_id = slot.get("reuses_training_job_id") or slot["job_id"]
    training_row = train_rows.get(training_id)
    if training_row is None:
        return ({"record_type": "locked_evaluation", "job_id": slot["job_id"],
            "training_job_id": training_id, "stage": "main", "status": "not_evaluated",
            "test": None, "test_null_reason": "registered training job has no successful checkpoint",
            "evaluation_fingerprint": None, "split_sha256": slot["split_sha256"]}, {})
    checkpoint_path = Path(slot.get("checkpoint_path") or training_row["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"registered checkpoint is missing: {checkpoint_path}")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != slot.get("checkpoint_sha256", training_row.get("checkpoint_sha256")):
        raise RuntimeError(f"checkpoint hash mismatch for {slot['job_id']}")
    if sha256_file(Path(slot["split_path"])) != slot["split_sha256"]:
        raise RuntimeError(f"split hash mismatch for {slot['job_id']}")
    cfg = config_for(slot["dataset"], slot["seed"], slot["variant"],
                     float(slot["lr"]), device_name)
    runtime = runtime_job(slot, device_name)
    if _training_config_hash(cfg) != slot["config_hash"] or runtime["fingerprint"] != slot["fingerprint"]:
        raise RuntimeError(f"training identity/config mismatch for {slot['job_id']}")
    fingerprint = eval_fingerprint(lock, slot, checkpoint_sha)
    pred_path = predictions_path(slot["job_id"], fingerprint)
    previous = find_existing_eval(rows, slot["job_id"], fingerprint)
    expected_schema = lock["schemas"][slot["dataset"]]
    y = data.y.long().cpu()
    val_idx, test_idx = data.val_idx.long().cpu(), data.test_idx.long().cpu()
    if previous and pred_path.is_file() and sha256_file(pred_path) == previous.get("predictions_sha256"):
        pred_blob = torch.load(pred_path, map_location="cpu", weights_only=False)
        if pred_blob.get("evaluation_fingerprint") != fingerprint:
            raise RuntimeError("saved predictions are not tied to the locked evaluation identity")
        if pred_blob.get("val_idx_sha256") != sha256_json(val_idx.tolist()) or pred_blob.get("test_idx_sha256") != sha256_json(test_idx.tolist()):
            raise RuntimeError("saved predictions do not match the locked split node order")
        val_pred, test_pred = pred_blob["val_pred"], pred_blob["test_pred"]
        router_diagnostics = previous.get("router_diagnostics")
        perturbations = previous.get("node_shuffle_perturbations")
        inference_seconds = previous.get("evaluation_inference_seconds", 0.0)
        print(f"SKIP locked test {slot['job_id']} (matching fingerprint and saved prediction hash)", flush=True)
    elif previous:
        # Respect the one-test-evaluation lock even if an ignored raw prediction artifact
        # was removed. Aggregates remain usable, but node-paired diagnostics cannot be rebuilt.
        return previous, None
    else:
        model = build_model(cfg, {"num_nodes": data.num_nodes, "num_classes": data.num_classes,
            "text_dim": int(data.x_t.size(1)), "visual_dim": int(data.x_i.size(1))})
        head = nn.Linear(128, int(data.num_classes))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("fingerprint") != slot["fingerprint"]:
            raise RuntimeError(f"checkpoint training fingerprint mismatch for {slot['job_id']}")
        if checkpoint.get("source_hashes") != lock["source_hashes"]:
            raise RuntimeError(f"checkpoint source-code hash mismatch for {slot['job_id']}")
        if checkpoint.get("dataset_input_hashes") != slot.get("dataset_input_hashes"):
            raise RuntimeError(f"checkpoint dataset hash mismatch for {slot['job_id']}")
        model.load_state_dict(checkpoint["model_state"])
        head.load_state_dict(checkpoint["head_state"])
        model.set_run_seed(int(slot["seed"]))
        device = torch.device(device_name)
        model.to(device); head.to(device)
        model.eval(); head.eval()
        x = data.x.to(device); edge = data.edge_index.to(device)
        torch.cuda.synchronize(device); start = time.perf_counter()
        with torch.no_grad():
            z, _, _, _, aux = model(x, edge)
            val_pred = head(z[val_idx.to(device)]).argmax(1).cpu()
            test_pred = head(z[test_idx.to(device)]).argmax(1).cpu()
        torch.cuda.synchronize(device); inference_seconds = time.perf_counter() - start
        router_diagnostics = None
        perturbations = None
        if slot["variant"] in {"J_NODE", "J_NODE_SHUFFLE"}:
            # Quartile thresholds are computed from training nodes only.
            geometry = geometry_features(data.edge_index, data.x_t, data.x_i,
                                         data.train_idx, data.val_idx)
            router_diagnostics = summarize_router(aux, val_idx,
                geometry["text"]["degree_thresholds"])
            if slot["variant"] == "J_NODE":
                vals = []
                for perturb_id in range(10):
                    model.set_context_permutation(fixed_permutation(int(slot["seed"]), perturb_id, data.num_nodes))
                    with torch.no_grad():
                        z_perturb, _, _, _, _ = model(x, edge)
                        p = head(z_perturb[val_idx.to(device)]).argmax(1).cpu()
                    metrics = confusion_metrics(y[val_idx], p, int(data.num_classes), expected_schema["macro_f1_label_set"])
                    vals.append({"permutation_id": perturb_id, "seed_namespace": "node_validation_perturbation",
                                 "val_acc": metrics["acc"], "val_macro_f1": metrics["macro_f1"],
                                 "accuracy_delta_vs_fixed_validation_context": metrics["acc"] - float((val_pred == y[val_idx]).float().mean())})
                perturbations = vals
                model.set_context_permutation(None)
        if z.shape != (data.num_nodes, 128):
            raise AssertionError(f"{slot['job_id']}: checkpoint output dimension mismatch")
        del model, head, x, edge, z
        # Save only argmax classes, indices, and their identity hashes. This ignored raw file
        # supports exact paired-count resumption; no logits, features, or embeddings are retained.
        pred_blob = {"evaluation_fingerprint": fingerprint,
            "split_sha256": slot["split_sha256"], "val_idx_sha256": sha256_json(val_idx.tolist()),
            "test_idx_sha256": sha256_json(test_idx.tolist()),
            "val_pred": val_pred, "test_pred": test_pred}
        atomic_torch_save(pred_path, pred_blob)

    if val_pred.numel() != val_idx.numel() or test_pred.numel() != test_idx.numel():
        raise RuntimeError(f"saved prediction length mismatch for {slot['job_id']}")
    if not torch.equal(val_pred, val_pred.long()) or not torch.equal(test_pred, test_pred.long()):
        raise RuntimeError("prediction labels are not integer classes")
    val_metrics = confusion_metrics(y[val_idx], val_pred, int(data.num_classes), expected_schema["macro_f1_label_set"])
    test_metrics = confusion_metrics(y[test_idx], test_pred, int(data.num_classes), expected_schema["macro_f1_label_set"])
    trained_val = training_row["val"]
    val_check = {"accuracy_abs_difference": abs(val_metrics["acc"] - trained_val["acc"]),
                 "macro_f1_abs_difference": abs(val_metrics["macro_f1"] - trained_val["macro_f1"]),
                 "confusion_matches": val_metrics["confusion_matrix"] == trained_val["confusion_matrix"]}
    if (val_check["accuracy_abs_difference"] > 1e-12 or
            val_check["macro_f1_abs_difference"] > 1e-12 or
            not val_check["confusion_matches"]):
        raise RuntimeError(f"val checkpoint reproducibility check failed: {slot['job_id']} {val_check}")
    pred_sha = sha256_file(pred_path)
    row = {"record_type": "locked_evaluation", "job_id": slot["job_id"],
        "training_job_id": training_id, "dataset": slot["dataset"], "variant": slot["variant"],
        "stage": "main", "phase": slot.get("phase"), "seed": int(slot["seed"]),
        "status": "succeeded", "lr": float(slot["lr"]), "split_path": slot["split_path"],
        "split_sha256": slot["split_sha256"], "config_hash": slot["config_hash"],
        "code_commit": registry_code_commit(lock), "source_hashes": lock["source_hashes"],
        "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
        "evaluation_fingerprint": fingerprint, "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "val": val_metrics, "test": test_metrics,
        "macro_f1_label_set": expected_schema["macro_f1_label_set"],
        "val_reproducibility_check": val_check,
        "evaluation_inference_seconds": inference_seconds,
        "router_diagnostics": router_diagnostics,
        "node_shuffle_perturbations": perturbations,
        "predictions_path": str(pred_path.resolve()), "predictions_sha256": pred_sha,
        "predictions_contents": "val/test argmax labels only; no logits, embedding, or feature tensors"}
    return row, {"val_pred": val_pred, "test_pred": test_pred,
                 "predictions_path": str(pred_path.resolve()), "predictions_sha256": pred_sha}


def registry_code_commit(lock: dict[str, Any]) -> str:
    return lock["training_source_commit"]


def mark_old_evaluations_superseded(rows: list[dict[str, Any]], job_id: str, new_fingerprint: str) -> None:
    old = [r for r in rows if r.get("record_type") == "locked_evaluation" and r.get("job_id") == job_id
           and r.get("status") == "succeeded" and r.get("evaluation_fingerprint") != new_fingerprint]
    for item in old:
        append_result({"record_type": "superseded", "job_id": job_id,
            "superseded_evaluation_fingerprint": item["evaluation_fingerprint"],
            "replacement_evaluation_fingerprint": new_fingerprint,
            "reason": "new locked-evaluation fingerprint; old evidence retained"})


def calculate_diagnostics(registry: dict[str, Any], lock: dict[str, Any],
                          eval_rows: list[dict[str, Any]], pred_cache: dict[tuple[str, int, str], dict[str, Any]],
                          data_cache: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_key = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in eval_rows if r.get("status") == "succeeded"}
    error = {"label_free_selector": "none; oracle columns are diagnostics only",
             "partitions": {"val": {}, "test": {}}}
    pairs: dict[str, Any] = {"paired_counts": {"val": {}, "test": {}}, "engineering_screen": {}}
    geometry: dict[str, Any] = {}
    router_rows = []
    for ds in DATASETS:
        data = data_cache[ds]
        for seed in SEEDS:
            labels = lock["schemas"][ds]["macro_f1_label_set"]
            targets = {"val": data.y[data.val_idx].long().cpu(), "test": data.y[data.test_idx].long().cpu()}
            pred_te = pred_cache.get((ds, seed, "T_E"))
            pred_ie = pred_cache.get((ds, seed, "I_E"))
            if pred_te and pred_ie:
                for partition, idx in (("val", data.val_idx), ("test", data.test_idx)):
                    p_t, p_i = pred_te[f"{partition}_pred"], pred_ie[f"{partition}_pred"]
                    key = f"{ds}/seed{seed}"
                    error["partitions"][partition][key] = modality_complementarity(
                        targets[partition], p_t, p_i)
            for control in ("J_GLOBAL", "J_GLOBAL_WIDE", "J_CONCAT"):
                a = pred_cache.get((ds, seed, "J_NODE"))
                b = pred_cache.get((ds, seed, control))
                if a and b:
                    for partition, idx in (("val", data.val_idx), ("test", data.test_idx)):
                        key = f"{ds}/seed{seed}/J_NODE_vs_{control}"
                        pairs["paired_counts"][partition][key] = correctness_pair(
                            targets[partition], a[f"{partition}_pred"], targets[partition], b[f"{partition}_pred"])
            if ds in {"Movies", "Grocery"}:
                for modality, center, mixed in (("text", "T_E", "T_M"), ("image", "I_E", "I_M")):
                    a = pred_cache.get((ds, seed, mixed)); b = pred_cache.get((ds, seed, center))
                    if a and b:
                        key = f"{ds}/seed{seed}/{mixed}_vs_{center}"
                        for partition in ("val", "test"):
                            pairs["paired_counts"][partition][key] = correctness_pair(
                                targets[partition], a[f"{partition}_pred"],
                                targets[partition], b[f"{partition}_pred"])
            if ds in {"Movies", "Grocery"}:
                fixed = pred_cache.get((ds, seed, "J_FIXED"))
                neighbor = pred_cache.get((ds, seed, "J_N"))
                if fixed and neighbor:
                    for partition in ("val", "test"):
                        key = f"{ds}/seed{seed}/J_FIXED_vs_J_N"
                        pairs["paired_counts"][partition][key] = correctness_pair(
                            targets[partition], fixed[f"{partition}_pred"],
                            targets[partition], neighbor[f"{partition}_pred"])
            node_row = by_key.get((ds, seed, "J_NODE"))
            if node_row and node_row.get("router_diagnostics"):
                router_rows.append({"dataset": ds, "seed": seed,
                                    "router_diagnostics": node_row["router_diagnostics"],
                                    "perturbations": node_row.get("node_shuffle_perturbations")})

            geom_features = geometry_features(data.edge_index, data.x_t, data.x_i,
                data.train_idx, data.val_idx)
            key = f"{ds}/seed{seed}"
            geometry[key] = {"split_sha256": sha256_file(fixed_split_path(ds)),
                "homophily": train_train_homophily(data.edge_index, data.y, data.train_idx),
                "modalities": {}}
            for modality, center, mixed in (("text", "T_E", "T_M"), ("image", "I_E", "I_M")):
                g = geom_features[modality]
                degree_bin = bucket_id(g["degree"], g["degree_thresholds"])
                cosine_bin = bucket_id(g["cosine"], g["cosine_thresholds"])
                combined = torch.zeros_like(degree_bin)
                noniso = ~g["isolated"]
                combined[noniso] = 1 + degree_bin[noniso] * 4 + cosine_bin[noniso]
                names = ["isolated"] + [f"degree_q{d+1}_cosine_q{c+1}" for d in range(4) for c in range(4)]
                val_bins = combined[data.val_idx.long()]
                bucket_out = {}
                for comparison_a, comparison_b in ((mixed, center),):
                    pa, pb = pred_cache.get((ds, seed, comparison_a)), pred_cache.get((ds, seed, comparison_b))
                    if pa and pb:
                        bucket_out[f"{comparison_a}_vs_{comparison_b}"] = bucket_metrics(
                            data.y, pa["val_pred"], pb["val_pred"], int(data.num_classes), labels,
                            data.val_idx, combined, names, g["threshold_source"])
                pn = pred_cache.get((ds, seed, "J_NODE"))
                for control in ("J_GLOBAL", "J_GLOBAL_WIDE", "J_CONCAT"):
                    pc = pred_cache.get((ds, seed, control))
                    if pn and pc:
                        bucket_out[f"J_NODE_vs_{control}"] = bucket_metrics(
                            data.y, pn["val_pred"], pc["val_pred"], int(data.num_classes), labels,
                            data.val_idx, combined, names, g["threshold_source"])
                geometry[key]["modalities"][modality] = {
                    "degree_thresholds_training_only": g["degree_thresholds"],
                    "raw_ego_neighborhood_cosine_thresholds_training_only": g["cosine_thresholds"],
                    "threshold_source": g["threshold_source"],
                    "isolated_validation_n": int(g["isolated"][data.val_idx].sum()),
                    "validation_buckets": bucket_out,
                }
    # Add test paired NODE-control counts and summarize per graph, preserving seed rows.
    for control in ("J_GLOBAL", "J_GLOBAL_WIDE", "J_CONCAT"):
        graph_summaries = {}
        pass_graphs = []
        for ds in DATASETS:
            per_seed = []
            for seed in SEEDS:
                a, b = by_key.get((ds, seed, "J_NODE")), by_key.get((ds, seed, control))
                if a and b:
                    per_seed.append({"seed": seed,
                        "acc_difference": a["test"]["acc"] - b["test"]["acc"],
                        "macro_f1_difference": a["test"]["macro_f1"] - b["test"]["macro_f1"]})
            if len(per_seed) == 3:
                acc = [x["acc_difference"] for x in per_seed]
                f1 = [x["macro_f1_difference"] for x in per_seed]
                mean_acc = sum(acc) / 3
                mean_f1 = sum(f1) / 3
                same_direction = sum(x > 0 for x in acc)
                passed = mean_acc >= 0.003 and same_direction >= 2 and mean_f1 >= -0.003
                graph_summaries[ds] = {"seed_differences": per_seed,
                    "mean_accuracy_difference": mean_acc,
                    "mean_accuracy_difference_pp": mean_acc * 100,
                    "mean_macro_f1_difference": mean_f1,
                    "same_direction_seeds": same_direction,
                    "ddof0_seed_accuracy_difference_sd": float(torch.tensor(acc, dtype=torch.float64).std(unbiased=False)),
                    "passes_engineering_screen": passed}
                if passed:
                    pass_graphs.append(ds)
            else:
                graph_summaries[ds] = {"status": "incomplete", "available_seed_count": len(per_seed)}
        pairs["engineering_screen"][control] = {"graphs": graph_summaries,
            "passing_graphs": pass_graphs,
            "pass_count": len(pass_graphs), "criterion": "mean test Acc gain >=0.3pp; at least 2/3 seeds positive; mean Macro-F1 delta >=-0.3pp"}
    wide = set(pairs["engineering_screen"]["J_GLOBAL_WIDE"]["passing_graphs"])
    concat = set(pairs["engineering_screen"]["J_CONCAT"]["passing_graphs"])
    common_pass = sorted(wide & concat)
    pairs["engineering_screen"]["same_graphs_pass_both_capacity_controls"] = common_pass
    pairs["engineering_screen"]["overall_exploratory_screen_pass"] = len(common_pass) >= 3
    router = {"interpretation": "descriptive fusion weights/frequencies; not causal utilities or mature MoE load",
              "runs": router_rows,
              "shuffle_perturbations_are_separate_not_ensembled": True}
    return error, pairs, {"geometry": geometry, "router": router}


def aggregate_metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    vals = [float(row[field]) for row in rows if row.get(field) is not None]
    if not vals:
        return {"mean": None, "std_ddof0": None, "n": 0}
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return {"mean": mean, "std_ddof0": sd, "n": len(vals), "units": "fraction [0,1]"}


def build_summary(registry: dict[str, Any], lock: dict[str, Any], all_rows: list[dict[str, Any]],
                  eval_rows: list[dict[str, Any]], diagnostics: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    train_rows = [r for r in all_rows if r.get("record_type") == "training"]
    evaluations = [r for r in eval_rows if r.get("status") == "succeeded"]
    metrics: dict[str, Any] = {"aggregation": "mean and population standard deviation (ddof=0)",
        "internal_units": "fractions in [0,1]; report displays percentages and percentage-point deltas",
        "test_evaluation": "one locked evaluation per successful primary slot; null for failures",
        "calibration": {}, "primary": {}}
    cal_by_lr = {}
    for slot in registry["calibration_slots"]:
        row = next((r for r in reversed(train_rows) if r.get("job_id") == slot["job_id"] and r.get("status") == "succeeded"), None)
        if row:
            cal_by_lr.setdefault(str(slot["lr"]), []).append({"dataset": slot["dataset"], "variant": slot["variant"],
                "seed": slot["seed"], "val_acc": row["val"]["acc"], "val_macro_f1": row["val"]["macro_f1"]})
    metrics["calibration"] = {"selected_common_lr": registry.get("selected_common_lr"),
        "equal_weight_scores": registry.get("lr_selection", {}).get("scores", {}),
        "slots_by_lr": cal_by_lr, "test_metrics": None}
    by_ds_variant = {}
    for ds in DATASETS:
        by_ds_variant[ds] = {}
        variants = sorted({r["variant"] for r in evaluations if r["dataset"] == ds})
        for variant in variants:
            items = [r for r in evaluations if r["dataset"] == ds and r["variant"] == variant]
            by_ds_variant[ds][variant] = {
                "successful_seed_count": len(items),
                "seeds": [{"seed": int(r["seed"]), "val_acc": r["val"]["acc"],
                    "val_macro_f1": r["val"]["macro_f1"], "test_acc": r["test"]["acc"],
                    "test_macro_f1": r["test"]["macro_f1"], "best_epoch": next((x.get("best_epoch") for x in train_rows if x.get("job_id") == r.get("training_job_id")), None)} for r in items],
                "val_acc": aggregate_metric([r["val"] for r in items], "acc"),
                "val_macro_f1": aggregate_metric([r["val"] for r in items], "macro_f1"),
                "test_acc": aggregate_metric([r["test"] for r in items], "acc"),
                "test_macro_f1": aggregate_metric([r["test"] for r in items], "macro_f1"),
            }
    metrics["primary"] = by_ds_variant
    atomic_json(REPORTS / "metrics_summary.json", metrics)
    error, pairs, extra = diagnostics
    atomic_json(REPORTS / "error_complementarity.json", error)
    atomic_json(REPORTS / "paired_comparisons.json", pairs)
    atomic_json(REPORTS / "geometry_summary.json", extra["geometry"])
    atomic_json(REPORTS / "router_summary.json", extra["router"])
    times = [r for r in train_rows if r.get("status") == "succeeded"]
    training_s = sum(r.get("timing_seconds", {}).get("training", 0) for r in times)
    inference_s = sum(r.get("timing_seconds", {}).get("full_graph_validation_inference", 0) for r in times)
    evaluation_s = sum(r.get("evaluation_inference_seconds", 0) for r in evaluations)
    resource = {"device": "single NVIDIA GPU per sequential job (recorded in each training record)",
        "successful_unique_training_job_count": len(times), "failed_training_attempt_count": registry["attempt_count"] - len(times),
        "successful_locked_evaluations": len(evaluations),
        "training_gpu_seconds": training_s, "training_gpu_hours": training_s / 3600,
        "validation_inference_seconds": inference_s, "locked_evaluation_inference_seconds": evaluation_s,
        "aggregate_recorded_gpu_hours": (training_s + inference_s + evaluation_s) / 3600,
        "peak_gpu_memory_max_bytes": max((r.get("peak_gpu_memory_bytes") or 0 for r in times), default=None),
        "soft_planning_target_gpu_hours": 12, "hard_attempt_cap": ATTEMPT_CAP,
        "actual_attempt_count": registry["attempt_count"],
        "profiling": json.loads((RAW / "smoke.json").read_text(encoding="utf-8")) if (RAW / "smoke.json").exists() else None,
        "raw_log_directory": str((RAW / "logs").resolve()),
        "raw_checkpoint_directory": str((RAW / "checkpoints").resolve()),
        "raw_prediction_directory": str(PRED_DIR.resolve())}
    atomic_json(REPORTS / "resource_summary.json", resource)
    missing = []
    for slot in registry["primary_slots"]:
        if not any(r.get("job_id") == slot["job_id"] and r.get("status") == "succeeded" for r in evaluations):
            missing.append({"job_id": slot["job_id"], "dataset": slot["dataset"],
                "variant": slot["variant"], "seed": slot["seed"], "status": slot.get("status"),
                "reason": slot.get("failure", {}).get("error", "no locked test evaluation row")})
    atomic_json(REPORTS / "failures.json", {"training_failures": registry.get("failures", []),
        "missing_or_unevaluated_primary_slots": missing,
        "superseded": [r for r in all_rows if r.get("record_type") == "superseded"],
        "attempt_count": registry["attempt_count"], "attempt_cap": ATTEMPT_CAP})
    write_report(registry, lock, metrics, pairs, resource, missing)
    manifest = {"study": registry["study"], "branch": registry["branch"],
        "source_commit": registry["source_commit"],
        "final_commit_ref": "the commit containing this report package; exact SHA is returned in the handoff",
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "selected_common_lr": registry.get("selected_common_lr"),
        "training_attempt_count": registry["attempt_count"],
        "successful_training_rows": len(times), "successful_locked_evaluations": len(evaluations),
        "missing_or_unevaluated_primary_slots": missing,
        "raw_evidence": {"directory": str(RAW.resolve()),
            "logs": str((RAW / "logs").resolve()), "checkpoints": str((RAW / "checkpoints").resolve()),
            "predictions": str(PRED_DIR.resolve()),
            "files": [{"path": r.get("checkpoint_path"), "sha256": r.get("checkpoint_sha256")}
                for r in times] + [{"path": r.get("predictions_path"), "sha256": r.get("predictions_sha256")}
                for r in evaluations] + [{"path": r.get("log_path"), "sha256": r.get("log_sha256")}
                for r in times] + ([{"path": str((RAW / "smoke.json").resolve()),
                    "sha256": sha256_file(RAW / "smoke.json")} ] if (RAW / "smoke.json").is_file() else []),
            "tracked_input_audit": {"path": str((REPORTS / "data_audit.json").resolve()),
                "sha256": sha256_file(REPORTS / "data_audit.json") if (REPORTS / "data_audit.json").is_file() else None}},
        "resume_commands": registry["commands"]}
    atomic_json(REPORTS / "manifest.json", manifest)


def fmt_metric(obj: dict[str, Any]) -> str:
    if not obj or obj.get("mean") is None:
        return "—"
    return f"{obj['mean']*100:.2f} ± {obj['std_ddof0']*100:.2f}"


def write_report(registry, lock, metrics, pairs, resource, missing) -> None:
    def metric_mean(ds: str, variant: str, metric: str):
        item = metrics["primary"].get(ds, {}).get(variant, {}).get(metric, {})
        return item.get("mean") if item else None

    joint_deltas = []
    for ds in DATASETS:
        t, i, joint = (metric_mean(ds, variant, "test_acc") for variant in ("T_E", "I_E", "J_E"))
        if t is not None and i is not None and joint is not None:
            joint_deltas.append(joint - max(t, i))
    joint_gain_pp = (sum(joint_deltas) / len(joint_deltas) * 100) if joint_deltas else None
    comp_rows = list(json.loads((REPORTS / "error_complementarity.json").read_text(encoding="utf-8")).get("partitions", {}).get("test", {}).values()) if (REPORTS / "error_complementarity.json").exists() else []
    comp_rates = {}
    for key in ("text_only_correct", "image_only_correct", "both_wrong", "prediction_disagreement"):
        denom_key = "n"
        comp_rates[key] = (sum(row[key] / row[denom_key] for row in comp_rows) / len(comp_rows)) if comp_rows else None

    neighborhood = []
    center = []
    for key, row in pairs["paired_counts"].get("test", {}).items():
        if "_M_vs_" in key:
            neighborhood.append(row["accuracy_difference_a_minus_b"])
        if "J_FIXED_vs_J_N" in key:
            center.append(row["accuracy_difference_a_minus_b"])
    neighbor_pp = (sum(neighborhood) / len(neighborhood) * 100) if neighborhood else None
    center_pp = (sum(center) / len(center) * 100) if center else None
    center_directions = sum(v > 0 for v in center)
    lines = ["# MAG NC problem validation v1", "",
        f"Branch `{registry['branch']}`; training source commit `{registry['source_commit']}`; final report commit is cited in `manifest.json` and the handoff.",
        f"Protocol lock SHA-256: `{lock['protocol_lock_sha256']}`. Common LR: `{registry.get('selected_common_lr')}`.",
        f"Training attempts: **{registry['attempt_count']}** / {ATTEMPT_CAP}; successful unique training records: **{resource['successful_unique_training_job_count']}**; locked primary test evaluations: **{resource['successful_locked_evaluations'] if 'successful_locked_evaluations' in resource else 0}**; missing primary evaluations: **{len(missing)}**.",
        "", "## Primary NC results", "",
        "Mean ± population SD across three training seeds (`ddof=0`), in percent. These are descriptive; three seeds do not support significance claims. A dash indicates a variant was outside that dataset's registered matrix or no successful locked evaluation is available.",
        "", "| Dataset | T_E Acc/F1 | I_E Acc/F1 | J_E Acc/F1 | J_FIXED Acc/F1 | J_GLOBAL Acc/F1 | J_NODE Acc/F1 | J_GLOBAL_WIDE Acc/F1 | J_CONCAT Acc/F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        entry = metrics["primary"].get(ds, {})
        cells = []
        for variant in ("T_E", "I_E", "J_E", "J_FIXED", "J_GLOBAL", "J_NODE", "J_GLOBAL_WIDE", "J_CONCAT"):
            x = entry.get(variant)
            cells.append(f"{fmt_metric(x['test_acc'])} / {fmt_metric(x['test_macro_f1'])}" if x else "—")
        lines.append("| " + ds + " | " + " | ".join(cells) + " |")
    lines += ["", "Each table cell is test Accuracy / Macro-F1, with mean ± ddof0 SD. The complete per-seed and validation table, full confusion matrices, and class schemas are in `metrics_summary.json` and `run_results.jsonl`.",
        "", "## Calibration", "",
        f"The single LR was selected by the registered equal-weight mean best-validation-accuracy rule across six Movies/Grocery × model cells. Selected LR: **{registry.get('selected_common_lr')}**. Scores and all 18 validation-only runs are in `lr_selection.json`, `run_registry.json`, and `run_results.jsonl`; calibration has no test metrics.",
        "", "## Q1: modality and neighborhood signal", "",
        (f"Across the five graphs with equal graph weighting, J_E test Accuracy was {joint_gain_pp:+.3f} pp versus the better of T_E and I_E. Across dataset/seed cells, the mean rates of text-only-correct, image-only-correct, both-wrong, and prediction disagreement were {comp_rates.get('text_only_correct')!r}, {comp_rates.get('image_only_correct')!r}, {comp_rates.get('both_wrong')!r}, and {comp_rates.get('prediction_disagreement')!r}. On Movies/Grocery the equal-cell paired test Accuracy change for T_M−T_E and I_M−I_E pooled together was {neighbor_pp:+.3f} pp. These are descriptive results; single-modality controls also remove an adapter and its parameters, and dual-modality gains cannot be assigned entirely to complementarity." if joint_gain_pp is not None and neighbor_pp is not None else "Q1 aggregate is incomplete because required model/test results are missing; inspect the per-run values in `metrics_summary.json` and `error_complementarity.json`."),
        "`error_complementarity.json` reports validation and locked-test both-correct/text-only/image-only/both-wrong cells, disagreement, and the label-using oracle upper bound for every graph/seed. The oracle is not deployable. `geometry_summary.json` reports validation-only bucket comparisons using train-only degree and within-modality raw ego/neighborhood cosine thresholds.",
        "", "## Q2: node-conditioned fusion", "",
        f"NODE-versus-capacity-control test screening result: **{'pass' if pairs['engineering_screen']['overall_exploratory_screen_pass'] else 'weak/mixed or incomplete'}**. Graphs passing both GLOBAL_WIDE and CONCAT checks: `{pairs['engineering_screen']['same_graphs_pass_both_capacity_controls']}`. The registered threshold is a paired mean Acc gain ≥0.3 pp, positive in at least 2/3 seeds, and mean Macro-F1 no worse than −0.3 pp, for at least three of five graphs against each control on the same graph set. This is an engineering screen, not statistical evidence. Per-graph/per-seed deltas and paired 2×2 counts are in `paired_comparisons.json`.",
        "", "## Q3: direct center paths", "",
        (f"On the six Movies/Grocery test seed cells, J_FIXED−J_N paired Accuracy averaged {center_pp:+.3f} pp and was positive in {center_directions}/6 cells. `paired_comparisons.json` contains their full paired correctness counts; Macro-F1 is in `metrics_summary.json`. This supports or weakens a center-path contribution only for the tested design and split protocol; it does not prove uniquely extracted information." if center_pp is not None else "J_FIXED/J_N test comparison is incomplete; inspect `failures.json`."),
        "", "## Capacity, router, and cost", "",
        "The registry records exact parameter counts including the shared external NC head and exhaustive widths. Any match outside 1% is marked unmatched. `router_summary.json` reports validation weights, argmax frequencies, entropy, degree/isolation bucket means, mask violations, and ten separate fixed-context-permutation results per J_NODE run. These statistics do not identify causal utility or mature MoE expert loads.",
        f"Recorded training plus validation/test inference cost: **{resource['aggregate_recorded_gpu_hours']:.3f} GPU-hours**; training-only **{resource['training_gpu_hours']:.3f} GPU-hours**; peak allocated memory **{resource['peak_gpu_memory_max_bytes']} bytes**. The 12-hour planning target was soft; attempt cap was {ATTEMPT_CAP}.",
        "", "## Completion and reproducibility", "",
        f"Primary registered slots: {len(registry['primary_slots'])}; missing or unevaluated: {len(missing)}. Training attempts include retries. Failed, missing, and superseded records remain listed in `failures.json`; test metrics remain null where no locked evaluation succeeded.",
        f"Raw evidence is local and ignored by Git under `{RAW.resolve()}`. Checkpoints and per-node argmax predictions are content-hashed in `manifest.json`; predictions contain class IDs only and are retained locally to resume aggregate paired analyses.",
        "", "Commands: see `manifest.json` for complete calibration, main-matrix, locked-evaluation, and summary commands.",
        ""]
    (REPORTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(device_name: str, resume: bool) -> None:
    device_name = select_idle_gpu(device_name)
    registry = read_registry()
    cache: dict[str, Any] = {}
    lock = freeze_protocol(registry, cache, device_name)
    rows = jsonl_rows()
    training = result_by_job(rows)
    all_eval_rows = [r for r in rows if r.get("record_type") == "locked_evaluation"]
    pred_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    current_eval_rows = []
    for slot in registry["primary_slots"]:
        data = load_data(slot["dataset"], int(slot["seed"]), float(slot["lr"]), device_name, cache)
        training_id = slot.get("reuses_training_job_id") or slot["job_id"]
        training_row = training.get(training_id)
        if training_row is None:
            row = {"record_type": "locked_evaluation", "job_id": slot["job_id"],
                "training_job_id": training_id, "dataset": slot["dataset"],
                "variant": slot["variant"], "stage": "main", "seed": slot["seed"],
                "status": "not_evaluated", "test": None,
                "test_null_reason": "no successful best-validation checkpoint",
                "evaluation_fingerprint": None}
            append_result(row); current_eval_rows.append(row)
            continue
        # Populate the formal slot's checkpoint reference from the aliased calibration job.
        slot["checkpoint_path"] = training_row["checkpoint_path"]
        slot["checkpoint_sha256"] = training_row["checkpoint_sha256"]
        slot["status"] = "reused" if slot.get("reuses_training_job_id") else slot.get("status")
        row, pred = evaluate_slot(slot, lock, data, device_name, training, rows)
        if resume and row.get("status") == "succeeded":
            found = find_existing_eval(rows, row["job_id"], row["evaluation_fingerprint"])
            if found and pred:
                old = torch.load(predictions_path(row["job_id"], row["evaluation_fingerprint"]), map_location="cpu", weights_only=False)
                pred = {"val_pred": old["val_pred"], "test_pred": old["test_pred"],
                        "predictions_path": found.get("predictions_path"),
                        "predictions_sha256": found.get("predictions_sha256")}
                row = found
        if row.get("status") == "succeeded" and not any(r.get("evaluation_fingerprint") == row.get("evaluation_fingerprint") and r.get("job_id") == row.get("job_id") for r in rows):
            mark_old_evaluations_superseded(rows, row["job_id"], row["evaluation_fingerprint"])
            append_result(row)
        elif row.get("status") == "not_evaluated":
            append_result(row)
        current_eval_rows.append(row)
        if pred:
            pred_cache[(slot["dataset"], int(slot["seed"]), slot["variant"])] = pred
        print(f"{'OK' if row.get('status') == 'succeeded' else 'MISSING'} {slot['job_id']}", flush=True)
    # Rehydrate caches for earlier runs when resuming a partially completed evaluation.
    rows = jsonl_rows()
    latest_eval = {}
    for row in rows:
        if row.get("record_type") == "locked_evaluation":
            latest_eval[(row.get("job_id"), row.get("evaluation_fingerprint"))] = row
    for slot in registry["primary_slots"]:
        evals = [r for r in rows if r.get("record_type") == "locked_evaluation" and r.get("job_id") == slot["job_id"] and r.get("status") == "succeeded"]
        if not evals:
            continue
        row = evals[-1]
        key = (slot["dataset"], int(slot["seed"]), slot["variant"])
        if key in pred_cache:
            continue
        fp = row["evaluation_fingerprint"]
        path = predictions_path(slot["job_id"], fp)
        if path.is_file() and sha256_file(path) == row.get("predictions_sha256"):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            pred_cache[key] = {"val_pred": blob["val_pred"], "test_pred": blob["test_pred"],
                "predictions_path": str(path.resolve()), "predictions_sha256": row["predictions_sha256"]}
    current_eval_rows = [r for r in jsonl_rows() if r.get("record_type") == "locked_evaluation" and
        r.get("protocol_lock_sha256") == lock["protocol_lock_sha256"]]
    diagnostic_data = {}
    for ds in DATASETS:
        diagnostic_data[ds] = load_data(ds, 42, float(registry["selected_common_lr"]), device_name, cache)
    diagnostics = calculate_diagnostics(registry, lock, current_eval_rows, pred_cache, diagnostic_data, device_name)
    build_summary(registry, lock, jsonl_rows(), current_eval_rows, diagnostics)
    # Persist registry updates for calibration aliases, while leaving the protocol lock immutable.
    atomic_json(REGISTRY_PATH, registry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked, checkpoint-only NC evaluation and aggregate summaries")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    registry = read_registry()
    if args.summary_only:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        rows = jsonl_rows()
        eval_rows = [r for r in rows if r.get("record_type") == "locked_evaluation" and
                     r.get("protocol_lock_sha256") == lock["protocol_lock_sha256"]]
        cache = {ds: load_data(ds, 42, float(registry["selected_common_lr"]), "cpu", {}) for ds in DATASETS}
        pred_cache = {}
        for slot in registry["primary_slots"]:
            matches = [r for r in eval_rows if r.get("job_id") == slot["job_id"] and r.get("status") == "succeeded"]
            if not matches:
                continue
            row = matches[-1]
            p = predictions_path(slot["job_id"], row["evaluation_fingerprint"])
            if p.is_file() and sha256_file(p) == row.get("predictions_sha256"):
                blob = torch.load(p, map_location="cpu", weights_only=False)
                pred_cache[(slot["dataset"], int(slot["seed"]), slot["variant"])] = blob
        diagnostics = calculate_diagnostics(registry, lock, eval_rows, pred_cache, cache, "cpu")
        build_summary(registry, lock, rows, eval_rows, diagnostics)
        return
    run_evaluation(args.device, args.resume)


if __name__ == "__main__":
    main()
