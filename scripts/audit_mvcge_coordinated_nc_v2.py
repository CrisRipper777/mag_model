from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_mvcge_coordinated_nc_v2 import (
    CALIBRATION_SEED,
    CONFIRMATION_SEEDS,
    DATASETS,
    MODELS,
    OUTPUT_ROOT,
    REPORT_ROOT,
    SELECTION_PATH,
    STUDY,
    attempt_id,
    json_safe,
    sha256_file,
    write_json,
)


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes(order="C")).hexdigest()


def _latest(registry: dict[str, Any], base: str) -> tuple[str, dict[str, Any] | None]:
    related = [(key, value) for key, value in registry["attempts"].items() if key == base or key.startswith(base + "_retry")]
    if not related:
        return base, None
    related.sort(key=lambda pair: (0 if pair[0] == base else int(pair[0].rsplit("retry", 1)[-1])))
    return related[-1]


def _compare_metric(observed: float, recorded: float, name: str, tolerance: float = 1e-7) -> None:
    if not np.isfinite(observed) or not np.isfinite(recorded) or abs(float(observed) - float(recorded)) > tolerance:
        raise AssertionError(f"{name} differs: saved predictions={observed}, run result={recorded}, tolerance={tolerance}")


def _audit_predictions(result: dict[str, Any], data_profile: dict[str, Any], source_lock: dict[str, Any], protocol_lock_sha256: str) -> dict[str, Any]:
    dataset, model = result["dataset"], result["model"]
    classes = int(data_profile["datasets"][dataset]["num_classes"])
    for split, metric_key in (("train", "train"), ("validation", "validation")):
        ids = np.asarray(result["_arrays"][f"{split}_ids"], dtype=np.int64)
        targets = np.asarray(result["_arrays"][f"{split}_targets"], dtype=np.int64)
        argmax = np.asarray(result["_arrays"][f"{split}_argmax"], dtype=np.int64)
        if ids.ndim != 1 or targets.shape != ids.shape or argmax.shape != ids.shape:
            raise AssertionError(f"malformed saved {split} predictions for {result['attempt_id']}")
        if np.unique(ids).size != ids.size or np.any(targets < 0) or np.any(targets >= classes) or np.any(argmax < 0) or np.any(argmax >= classes):
            raise AssertionError(f"invalid IDs/targets/argmax values in {result['attempt_id']} {split}")
        expected_split_hash = result["train_ids_sha256"] if split == "train" else result["validation_ids_sha256"]
        if _sha_ids(ids) != expected_split_hash:
            raise AssertionError(f"saved {split} IDs do not match frozen split hash for {result['attempt_id']}")
        matrix = confusion_matrix(targets, argmax, labels=np.arange(classes))
        correct = int(np.equal(targets, argmax).sum())
        accuracy = correct / max(int(targets.size), 1)
        class_f1 = f1_score(targets, argmax, labels=np.arange(classes), average=None, zero_division=0)
        macro = float(class_f1.mean())
        saved = result[metric_key]
        if correct != int(saved["correct"]) or int(targets.size) != int(saved["count"]):
            raise AssertionError(f"integer prediction metrics differ for {result['attempt_id']} {split}")
        _compare_metric(accuracy, saved["accuracy"], f"{result['attempt_id']} {split} accuracy")
        _compare_metric(macro, saved["macro_f1"], f"{result['attempt_id']} {split} macro-F1")
        if not np.array_equal(matrix, np.asarray(saved["confusion_matrix"], dtype=np.int64)):
            raise AssertionError(f"confusion matrix differs for {result['attempt_id']} {split}")
        if not np.allclose(class_f1, np.asarray(saved["class_f1"]), atol=1e-7, rtol=0):
            raise AssertionError(f"per-class F1 differs for {result['attempt_id']} {split}")
    if result.get("test_task_head_calls") != 0 or result.get("test_metrics_computed") is not False or result.get("real_lp_runs") != 0:
        raise AssertionError(f"forbidden task route counters in {result['attempt_id']}")
    if result.get("train_evaluator_calls", 0) < 1 or result.get("validation_evaluator_calls", 0) < 1:
        raise AssertionError(f"dynamic evaluator counters missing for {result['attempt_id']}")
    if result.get("source_hashes") != source_lock["source_hashes"]:
        raise AssertionError(f"source hash map differs from protocol lock for {result['attempt_id']}")
    if result.get("protocol_lock_sha256") != protocol_lock_sha256:
        raise AssertionError(f"protocol lock hash differs for {result['attempt_id']}")
    output_dir = Path(result["output_dir"])
    prediction = Path(result["prediction_file"])
    checkpoint = Path(result["best_checkpoint"])
    if not prediction.is_file() or not checkpoint.is_file() or not Path(result["last_checkpoint"]).is_file() or not Path(result["step_log"]).is_file() or not Path(result["diagnostics_file"]).is_file():
        raise AssertionError(f"required output evidence missing for {result['attempt_id']}")
    if sha256_file(prediction) != result["prediction_sha256"] or sha256_file(checkpoint) != result["best_checkpoint_sha256"]:
        raise AssertionError(f"prediction/checkpoint hash mismatch for {result['attempt_id']}")
    if not (output_dir / "resolved_config.json").is_file():
        raise AssertionError(f"resolved config missing for {result['attempt_id']}")
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    for key, expected in (("phase", result["phase"]), ("dataset", dataset), ("model", model), ("seed", result["training_seed"]), ("config_index", result["config_index"]), ("lr", result["lr"]), ("weight_decay", result["weight_decay"])):
        if resolved.get(key) != expected:
            raise AssertionError(f"resolved config {key} mismatch for {result['attempt_id']}")
    best_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    last_payload = torch.load(Path(result["last_checkpoint"]), map_location="cpu", weights_only=False)
    expected_input_hashes = data_profile["datasets"][dataset]["input_file_hashes"]
    for name, payload, required in (
        ("best", best_payload, ("model", "head", "best_step", "best_key", "source_commit", "source_hashes", "protocol_lock_sha256", "input_hashes")),
        ("last", last_payload, ("model", "head", "optimizer", "noise_rng", "pair_rng", "torch_rng_cpu", "torch_rng_cuda", "python_rng", "numpy_rng", "step", "best_key", "best_step", "source_commit", "source_hashes", "protocol_lock_sha256", "input_hashes", "train_ids_sha256", "validation_ids_sha256", "training_complete")),
    ):
        missing_fields = [key for key in required if key not in payload]
        if missing_fields:
            raise AssertionError(f"{name} checkpoint misses recovery/provenance fields {missing_fields} for {result['attempt_id']}")
        if payload["source_commit"] != result["source_commit"] or payload["source_hashes"] != source_lock["source_hashes"] or payload["protocol_lock_sha256"] != result["protocol_lock_sha256"] or payload["input_hashes"] != expected_input_hashes:
            raise AssertionError(f"{name} checkpoint provenance mismatch for {result['attempt_id']}")
    if last_payload["train_ids_sha256"] != result["train_ids_sha256"] or last_payload["validation_ids_sha256"] != result["validation_ids_sha256"] or int(last_payload["step"]) != int(result["updates"]):
        raise AssertionError(f"last checkpoint state does not match recorded run state for {result['attempt_id']}")
    if not last_payload["training_complete"]:
        raise AssertionError(f"last checkpoint is not marked at a completed training state for {result['attempt_id']}")
    return {"attempt_id": result["attempt_id"], "train_correct": int(result["train"]["correct"]), "validation_correct": int(result["validation"]["correct"]), "prediction_sha256": result["prediction_sha256"]}


def _result_with_arrays(run: dict[str, Any]) -> dict[str, Any]:
    path = Path(run["prediction_file"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    value = dict(run); value["_arrays"] = arrays
    return value


def _configured_expected(phase: str, selection: dict[str, Any] | None = None) -> list[tuple[str, str, int, int | None]]:
    if phase == "calibration":
        return [(dataset, model, CALIBRATION_SEED, index) for dataset in DATASETS for model in MODELS for index in range(9)]
    if selection is None:
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selections"]
    return [
        (dataset, model, seed, int(selection[f"{dataset}/{model}"]["config_index"]))
        for dataset in DATASETS for model in MODELS for seed in CONFIRMATION_SEEDS
    ]


def audit(phase: str) -> None:
    registry_path = REPORT_ROOT / "run_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    data_profile = json.loads((REPORT_ROOT / "data_profile.json").read_text(encoding="utf-8"))
    current_lock_path = ROOT / "docs" / STUDY / "protocol_lock.json"
    current_lock = json.loads(current_lock_path.read_text(encoding="utf-8"))
    current_lock_sha = sha256_file(current_lock_path)
    protocol_versions = {current_lock_sha: current_lock}
    protocol_names = {current_lock_sha: current_lock_path.name}
    for prior_lock_path in sorted((ROOT / "docs" / STUDY).glob("protocol_lock_v*.json")):
        prior_hash = sha256_file(prior_lock_path)
        protocol_versions[prior_hash] = json.loads(prior_lock_path.read_text(encoding="utf-8"))
        protocol_names[prior_hash] = prior_lock_path.name
    selection = None
    if phase == "confirmation":
        if not SELECTION_PATH.is_file():
            raise FileNotFoundError("confirmation audit requires the immutable calibration selection lock")
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    expected = _configured_expected(phase, None if selection is None else selection["selections"])
    results = []
    missing, unsuccessful = [], []
    for dataset, model, seed, config in expected:
        base = attempt_id(phase, dataset, model, seed, config)
        run_id, run = _latest(registry, base)
        if run is None:
            missing.append(base); continue
        if run.get("status") != "success":
            unsuccessful.append({"attempt_id": run_id, "status": run.get("status"), "last_completed_update": run.get("last_completed_update"), "error": run.get("error")})
            continue
        if phase == "calibration" and int(run["config_index"]) != config:
            raise AssertionError(f"calibration config index mismatch in {run_id}")
        results.append(_result_with_arrays(run))
    if missing or unsuccessful or len(results) != len(expected):
        partial = {"study": STUDY, "phase": phase, "status": "incomplete", "expected": len(expected), "successful": len(results), "missing": missing, "unsuccessful": unsuccessful, "checked_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
        write_json(REPORT_ROOT / ("calibration_audit.json" if phase == "calibration" else "audit_result.json"), partial)
        raise RuntimeError(f"cannot pass {phase} audit: {len(results)}/{len(expected)} successful, {len(missing)} missing, {len(unsuccessful)} unsuccessful")
    result_versions = {}
    for result in results:
        lock_sha = result.get("protocol_lock_sha256")
        source_lock = protocol_versions.get(lock_sha)
        if source_lock is None or result.get("source_hashes") != source_lock.get("source_hashes"):
            raise AssertionError(f"no matching frozen source/protocol version for {result['attempt_id']}")
        if result.get("source_commit") != source_lock.get("training_source_commit"):
            raise AssertionError(f"source commit differs from its frozen protocol lock for {result['attempt_id']}")
        if lock_sha != current_lock_sha:
            lock_name = protocol_names[lock_sha]
            permitted_historical = phase == "confirmation" and (
                (lock_name == "protocol_lock_v1.json" and result.get("model") in {"A0", "N0"})
                or (lock_name == "protocol_lock_v3.json" and result.get("model") in {"M0", "M1", "P0", "P1"})
            )
            if not permitted_historical:
                raise AssertionError(f"historical protocol version is not permitted for {result['attempt_id']}")
        result_versions[result["attempt_id"]] = lock_sha
    audited = [
        _audit_predictions(result, data_profile, protocol_versions[result_versions[result["attempt_id"]]], result_versions[result["attempt_id"]])
        for result in results
    ]
    if len({row["attempt_id"] for row in audited}) != len(expected):
        raise AssertionError("duplicate/omitted attempts in audit selection")
    common = {
        "study": STUDY, "phase": phase, "status": "passed", "expected_cells": len(expected),
        "success_cells": len(results), "independently_recomputed_train_validation_prediction_sets": 2 * len(results),
        "test_evaluations": 0, "real_lp_runs": 0, "results": audited,
        "checks": ["all expected cell statuses", "train/validation ID hashes", "saved train/validation argmax metrics", "fixed-C macro-F1 and confusion matrices", "source/protocol hashes", "checkpoint/prediction/output evidence", "runtime test/LP counters"],
        "self_audit_limitation": "The experiment implementer wrote this audit; it is not independent peer review.",
    }
    if phase == "calibration":
        selections = {}
        for dataset in DATASETS:
            for model in MODELS:
                group = [run for run in results if run["dataset"] == dataset and run["model"] == model]
                if len(group) != 9:
                    raise AssertionError(f"expected 9 candidates for {dataset}/{model}, found {len(group)}")
                winner = min(group, key=lambda run: (-int(run["validation"]["correct"]), -float(run["validation"]["macro_f1"]), int(run["config_index"])))
                selections[f"{dataset}/{model}"] = {
                    "dataset": dataset, "model": model, "config_index": int(winner["config_index"]), "lr": float(winner["lr"]),
                    "weight_decay": float(winner["weight_decay"]), "validation_correct": int(winner["validation"]["correct"]),
                    "validation_macro_f1": float(winner["validation"]["macro_f1"]), "best_step": int(winner["best_step"]),
                    "attempt_id": winner["attempt_id"],
                }
        selection_lock = {
            "study": STUDY, "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "phase": "locked_after_162_successful_calibration_cells_and_prediction_audit", "calibration_seed": CALIBRATION_SEED,
            "calibration_cells": 162, "selection_rule": "correct descending, macro-F1 descending, config_index ascending",
            "calibration_audit_sha256": None, "protocol_lock_sha256": sha256_file(ROOT / "docs" / STUDY / "protocol_lock.json"),
            "selections": selections,
        }
        write_json(REPORT_ROOT / "calibration_audit.json", common)
        selection_lock["calibration_audit_sha256"] = sha256_file(REPORT_ROOT / "calibration_audit.json")
        write_json(SELECTION_PATH, selection_lock)
    else:
        for run in results:
            key = f"{run['dataset']}/{run['model']}"
            picked = selection["selections"][key]
            if int(run["config_index"]) != int(picked["config_index"]) or float(run["lr"]) != float(picked["lr"]) or float(run["weight_decay"]) != float(picked["weight_decay"]):
                raise AssertionError(f"confirmation configuration differs from lock for {run['attempt_id']}")
        calibration_path = REPORT_ROOT / "calibration_audit.json"
        if not calibration_path.is_file() or json.loads(calibration_path.read_text())["status"] != "passed":
            raise AssertionError("calibration audit is absent or incomplete")
        common["calibration_audit_sha256"] = sha256_file(calibration_path)
        common["selection_lock_sha256"] = sha256_file(SELECTION_PATH)
        common["calibration_success_cells"] = 162
        common["confirmation_success_cells"] = 54
        common["total_success_cells"] = 216
        common["confirmation_configuration_lock_verified"] = True
        version_summary = {}
        for result in results:
            lock_sha = result_versions[result["attempt_id"]]
            version_summary.setdefault(lock_sha, {"protocol_lock_file": protocol_names[lock_sha], "protocol_lock_sha256": lock_sha, "source_commit": result["source_commit"], "cells": 0, "models": set()})
            version_summary[lock_sha]["cells"] += 1
            version_summary[lock_sha]["models"].add(result["model"])
        common["implementation_versions"] = [
            {**value, "models": sorted(value["models"])}
            for _, value in sorted(version_summary.items())
        ]
        if current_lock.get("amendment"):
            common["protocol_amendment"] = current_lock["amendment"]
        write_json(REPORT_ROOT / "audit_result.json", common)
    print(f"{phase} audit passed: {len(results)} cells, {len(audited) * 2} prediction sets recomputed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    args = parser.parse_args()
    audit(args.phase)


if __name__ == "__main__":
    main()
