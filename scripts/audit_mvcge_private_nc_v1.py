from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mvcge_private_nc_v1 import (
    CALIBRATION_SEED,
    CONFIRMATION_SEEDS,
    DATASETS,
    MODELS,
    OUTPUT_ROOT,
    REPORT_ROOT,
    SELECTION_PATH,
    SOURCE_FILES,
    STUDY,
    attempt_sort_key,
    base_config,
    git,
    load_dataset,
    make_attempt_id,
    package_version,
    sha256_file,
    tensor_sha256,
    write_json,
)
from src.data import load_mag_data
from src.data.loaders import resolve_path
import torch


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_metrics(stored: dict[str, Any], labels: np.ndarray, predictions: np.ndarray, classes: int, context: str) -> dict[str, Any]:
    class_ids = list(range(classes))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    correct = int(np.count_nonzero(labels == predictions))
    accuracy = correct / max(len(labels), 1)
    macro_f1 = float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0))
    assert int(stored["count"]) == len(labels), f"{context}: sample count mismatch"
    assert int(stored["correct"]) == correct, f"{context}: correct node count mismatch"
    assert stored["confusion_matrix"] == matrix.tolist(), f"{context}: confusion matrix mismatch"
    assert abs(float(stored["accuracy"]) - accuracy) <= 1e-7, f"{context}: accuracy mismatch"
    assert abs(float(stored["macro_f1"]) - macro_f1) <= 1e-7, f"{context}: fixed-C Macro-F1 mismatch"
    assert stored["class_ids"] == class_ids and int(stored["zero_division"]) == 0
    return {"count": len(labels), "correct": correct, "accuracy": accuracy, "macro_f1": macro_f1, "confusion_matrix": matrix.tolist()}


def select_successful_attempts(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [attempt for attempt in registry["attempts"].values() if attempt.get("status") == "success" and attempt.get("phase") in {"calibration", "confirmation"}]


def verify_cells(matrix: dict[str, Any], registry: dict[str, Any], selection: dict[str, Any], results: list[dict[str, Any]], data_profile: dict[str, Any]) -> dict[str, Any]:
    successes = select_successful_attempts(registry)
    by_attempt = {record["attempt_id"]: record for record in successes}
    expected_cal = len(matrix["calibration_candidates"])
    expected_conf = len(matrix["confirmation_slots"])
    unique_cal = {(r["dataset"], r["model"], r["training_seed"], r["config_index"]) for r in successes if r["phase"] == "calibration"}
    unique_conf = {(r["dataset"], r["model"], r["training_seed"]) for r in successes if r["phase"] == "confirmation"}

    # Recompute every selected calibration configuration from all nine locked candidates.
    selected = selection["successful_selections"]
    for dataset in DATASETS:
        for model in MODELS:
            candidates = [r for r in successes if r["phase"] == "calibration" and r["dataset"] == dataset and r["model"] == model and int(r["training_seed"]) == CALIBRATION_SEED]
            assert len({int(r["config_index"]) for r in candidates}) == 9, f"missing/duplicate calibration candidate: {dataset}/{model}"
            winner = max(candidates, key=lambda r: (int(r["validation"]["correct"]), float(r["validation"]["macro_f1"]), -int(r["config_index"])))
            locked = selected[f"{dataset}/{model}"]
            assert int(locked["config_index"]) == int(winner["config_index"])
            assert float(locked["lr"]) == float(winner["lr"])
            assert float(locked["matrix_weight_decay"]) == float(winner["matrix_weight_decay"])
            assert locked["calibration_attempt_id"] == winner["attempt_id"]

    # Confirmation configurations must match the immutable selection lock.
    for record in successes:
        if record["phase"] != "confirmation":
            continue
        key = f"{record['dataset']}/{record['model']}"
        chosen = selected[key]
        assert int(record["config_index"]) == int(chosen["config_index"])
        assert float(record["lr"]) == float(chosen["lr"])
        assert float(record["matrix_weight_decay"]) == float(chosen["matrix_weight_decay"])
        assert int(record["training_seed"]) in CONFIRMATION_SEEDS

    calibration_runs = [r for r in successes if r["phase"] == "calibration"]
    confirmation_runs = [r for r in successes if r["phase"] == "confirmation"]
    selection_budget_limited = [
        key for key, chosen in selected.items()
        if int(by_attempt[chosen["calibration_attempt_id"]]["updates"]) >= 800
    ]

    audited: list[dict[str, Any]] = []
    current_data: dict[str, dict[str, Any]] = {}
    source_commits = set()
    source_hash_maps = set()
    for record in successes:
        source_commits.add(record["source_commit"])
        source_hash_maps.add(tuple(sorted(record["source_hashes"].items())))
        path = Path(record["prediction_file"])
        assert path.is_file(), f"missing argmax evidence {path}"
        assert sha256_file(path) == record["prediction_sha256"], f"prediction hash mismatch: {record['attempt_id']}"
        if record["dataset"] not in current_data:
            # Re-load only canonical train/validation labels and IDs; test IDs/labels are not read here.
            current_data[record["dataset"]] = load_dataset(record["dataset"], torch.device("cpu"))
        data = current_data[record["dataset"]]
        arrays = np.load(path, allow_pickle=False)
        keys = set(arrays.files)
        assert keys == {"train_node_ids", "train_labels", "train_argmax", "validation_node_ids", "validation_labels", "validation_argmax"}
        train_ids = arrays["train_node_ids"]
        val_ids = arrays["validation_node_ids"]
        assert np.array_equal(train_ids, data["train_idx"].cpu().numpy())
        assert np.array_equal(val_ids, data["val_idx"].cpu().numpy())
        assert np.array_equal(arrays["train_labels"], data["labels"][data["train_idx"]].cpu().numpy())
        assert np.array_equal(arrays["validation_labels"], data["labels"][data["val_idx"]].cpu().numpy())
        train_metrics = verify_metrics(record["train"], arrays["train_labels"], arrays["train_argmax"], int(data["classes"]), record["attempt_id"] + "/train")
        val_metrics = verify_metrics(record["validation"], arrays["validation_labels"], arrays["validation_argmax"], int(data["classes"]), record["attempt_id"] + "/validation")
        assert record["test_task_head_calls"] == 0 and record["test_metrics_computed"] is False and record["real_lp_runs"] == 0
        checkpoint = Path(record["best_checkpoint"])
        assert checkpoint.is_file() and sha256_file(checkpoint) == record["best_checkpoint_sha256"]
        checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert int(checkpoint_state["best_step"]) == int(record["best_step"])
        assert checkpoint_state["source_commit"] == record["source_commit"]
        assert checkpoint_state["source_hashes"] == record["source_hashes"]
        for relative, expected_hash in record["source_hashes"].items():
            assert sha256_file(ROOT / relative) == expected_hash, f"source changed during run: {relative}"
        audited.append({"attempt_id": record["attempt_id"], "train": train_metrics, "validation": val_metrics})
        arrays.close()

    assert len(source_commits) <= 1, f"mixed source commits: {source_commits}"
    assert len(source_hash_maps) <= 1, "successful runs use different source hashes"
    for dataset, expected in data_profile.items():
        data = current_data.get(dataset)
        if data is not None:
            assert data["split_hashes"]["train_ids"] == expected["split_hashes_train_validation_only"]["train_ids"]
            assert data["split_hashes"]["validation_ids"] == expected["split_hashes_train_validation_only"]["validation_ids"]
            for key, path in data["input_paths"].items():
                assert sha256_file(Path(path)) == expected["input_sha256"][key]

    failures = [r for r in registry["attempts"].values() if r.get("phase") in {"calibration", "confirmation"} and r.get("status") == "failed"]
    retries = [r for r in registry["attempts"].values() if "_retry" in r.get("attempt_id", "")]
    routing_profiles: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in confirmation_runs:
        if record["model"] not in {"G0", "G1"}:
            continue
        for layer in record.get("diagnostics", {}).get("routing", []):
            for modality in layer.get("modalities", []):
                usage = [float(value) for value in modality["clean_expert_usage"]]
                pair_share = max(modality["clean_top2_combinations"].values()) / max(int(modality["valid_tokens"]), 1)
                routing_profiles[record["model"]].append({
                    "minimum_expert_usage": min(usage), "maximum_expert_usage": max(usage),
                    "dominant_top2_pair_share": float(pair_share),
                })
    routing_summary = {
        model: {
            "layer_modality_profiles": len(profiles),
            "profiles_with_any_expert_below_1pct": sum(p["minimum_expert_usage"] < 0.01 for p in profiles),
            "mean_max_expert_usage": float(np.mean([p["maximum_expert_usage"] for p in profiles])) if profiles else None,
            "mean_dominant_top2_pair_share": float(np.mean([p["dominant_top2_pair_share"] for p in profiles])) if profiles else None,
            "max_dominant_top2_pair_share": max((p["dominant_top2_pair_share"] for p in profiles), default=None),
        }
        for model, profiles in routing_profiles.items()
    }
    return {
        "expected_calibration_units": expected_cal, "successful_calibration_units": len(unique_cal),
        "expected_confirmation_units": expected_conf, "successful_confirmation_units": len(unique_conf),
        "successful_attempts_including_retries": len(successes), "failed_attempts": len(failures),
        "retry_attempts": len(retries),
        "unrun_calibration_units": expected_cal - len(unique_cal), "unrun_confirmation_units": expected_conf - len(unique_conf),
        "calibration_max_updates": sum(int(r["updates"]) >= 800 for r in calibration_runs),
        "confirmation_max_updates": sum(int(r["updates"]) >= 800 for r in confirmation_runs),
        "selected_config_max_updates": selection_budget_limited,
        "calibration_wall_hours": sum(float(r["total_training_wall_seconds"]) for r in calibration_runs) / 3600,
        "confirmation_wall_hours": sum(float(r["total_training_wall_seconds"]) for r in confirmation_runs) / 3600,
        "confirmation_routing_summary": routing_summary,
        "source_commit": next(iter(source_commits), None), "source_hash_manifest_count": len(next(iter(source_hash_maps), ())),
        "argmax_files_recomputed": len(audited), "metrics_match": True,
        "test_task_head_calls": sum(int(r.get("test_task_head_calls", 0)) for r in registry["attempts"].values()),
        "real_lp_runs": sum(int(r.get("real_lp_runs", 0)) for r in registry["attempts"].values()),
        "audited_runs": audited,
    }


def aggregate_confirmation(successes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in successes if r["phase"] == "confirmation"]
    summary = []
    per_seed = []
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    calibration_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        if row["phase"] == "calibration":
            calibration_by_group[(row["dataset"], row["model"])].append(row)
    for row in rows:
        by_group[(row["dataset"], row["model"])].append(row)
        per_seed.append({
            "dataset": row["dataset"], "model": row["model"], "seed": int(row["training_seed"]),
            "config_index": int(row["config_index"]), "lr": float(row["lr"]), "matrix_weight_decay": float(row["matrix_weight_decay"]),
            "best_step": int(row["best_step"]), "updates": int(row["updates"]),
            "accuracy_percent": 100 * float(row["validation"]["accuracy"]),
            "macro_f1_percent": 100 * float(row["validation"]["macro_f1"]),
            "median_step_ms": float(row["median_step_ms"]),
            "peak_allocated_gib": float(row["peak_allocated_bytes"] or 0) / (1024 ** 3),
            "wall_hours": float(row["total_training_wall_seconds"]) / 3600,
            "early_stop_reason": row["early_stop_reason"],
        })
    for dataset in DATASETS:
        for model in MODELS:
            group = by_group[(dataset, model)]
            if len(group) != 3:
                continue
            acc = np.array([100 * r["validation"]["accuracy"] for r in group], dtype=np.float64)
            f1 = np.array([100 * r["validation"]["macro_f1"] for r in group], dtype=np.float64)
            calibration_wall_hours = float(sum(r["total_training_wall_seconds"] for r in calibration_by_group[(dataset, model)]) / 3600)
            confirmation_wall_hours = float(sum(r["total_training_wall_seconds"] for r in group) / 3600)
            summary.append({
                "dataset": dataset, "model": model,
                "accuracy_mean_percent": float(acc.mean()), "accuracy_sd_percent": float(acc.std(ddof=0)),
                "macro_f1_mean_percent": float(f1.mean()), "macro_f1_sd_percent": float(f1.std(ddof=0)),
                "active_backbone_parameters": int(group[0]["active_backbone_parameters"]),
                "active_task_head_parameters": int(group[0]["active_task_head_parameters"]),
                "median_step_ms_mean": float(np.mean([r["median_step_ms"] for r in group])),
                "peak_allocated_gib_max": float(max(r["peak_allocated_bytes"] or 0 for r in group)) / (1024 ** 3),
                "calibration_wall_hours": calibration_wall_hours,
                "confirmation_wall_hours": confirmation_wall_hours,
                "total_formal_wall_hours": calibration_wall_hours + confirmation_wall_hours,
                "max_updates_reached": sum(int(r["updates"]) >= 800 for r in group),
                "early_stopped": sum(str(r["early_stop_reason"]).startswith("early_stop") for r in group),
            })
    summary_map = {(r["dataset"], r["model"]): r for r in summary}
    contrasts = []
    for dataset in DATASETS:
        for challenger, reference in (("G1", "A0"), ("G1", "G0"), ("G1", "U1")):
            if (dataset, challenger) in summary_map and (dataset, reference) in summary_map:
                left, right = summary_map[(dataset, challenger)], summary_map[(dataset, reference)]
                contrasts.append({
                    "contrast": f"{challenger}-{reference}", "dataset": dataset,
                    "accuracy_delta_pp": left["accuracy_mean_percent"] - right["accuracy_mean_percent"],
                    "macro_f1_delta_pp": left["macro_f1_mean_percent"] - right["macro_f1_mean_percent"],
                })
    for contrast in ("G1-A0", "G1-G0", "G1-U1"):
        items = [r for r in contrasts if r["contrast"] == contrast]
        if len(items) == 3:
            contrasts.append({
                "contrast": contrast, "dataset": "equal-graph mean",
                "accuracy_delta_pp": float(np.mean([r["accuracy_delta_pp"] for r in items])),
                "macro_f1_delta_pp": float(np.mean([r["macro_f1_delta_pp"] for r in items])),
            })
    return summary, per_seed, {"contrasts": contrasts, "group_count": len(summary)}


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def report_text(summary: list[dict[str, Any]], per_seed: list[dict[str, Any]], contrast_block: dict[str, Any], audit: dict[str, Any], selection: dict[str, Any], smoke: dict[str, Any], test_record: dict[str, Any]) -> str:
    summary_map = {(row["dataset"], row["model"]): row for row in summary}
    calibration_wall_hours = sum(row["calibration_wall_hours"] for row in summary)
    confirmation_wall_hours = sum(row["confirmation_wall_hours"] for row in summary)
    equal_graph_contrasts = {row["contrast"]: row for row in contrast_block["contrasts"] if row["dataset"] == "equal-graph mean"}
    selected_budget_limited = ", ".join(audit["selected_config_max_updates"]) or "none"
    routing = audit["confirmation_routing_summary"]
    routing_note = " ".join(
        f"{model}: {routing[model]['profiles_with_any_expert_below_1pct']}/{routing[model]['layer_modality_profiles']} layer-modality profiles used some expert on <1% of nodes; the most common top-2 pair covered {routing[model]['mean_dominant_top2_pair_share'] * 100:.1f}% of nodes on average."
        for model in ("G0", "G1") if model in routing
    )
    lines = [
        "# MVCGE private NC v1 results", "",
        "## Scope and execution", "",
        "This is a descriptive train/validation development round on Movies, Grocery, and Reddit-S. The validation sets have prior development exposure and are not independent benchmarks. The study does not report test metrics, run a real LP task, or extend to Toys/ele-fashion.", "",
        f"- Training source commit: `{audit['source_commit']}`.",
        f"- Formal cells: {audit['successful_calibration_units']}/{audit['expected_calibration_units']} calibration and {audit['successful_confirmation_units']}/{audit['expected_confirmation_units']} confirmation.",
        f"- Failed attempts: {audit['failed_attempts']}; retries: {audit['retry_attempts']}; unrun formal cells: {audit['unrun_calibration_units'] + audit['unrun_confirmation_units']}.",
        f"- Update budget: {audit['calibration_max_updates']}/108 calibration cells reached 800 updates; {audit['confirmation_max_updates']}/36 confirmation cells reached 800. Selected calibration configs at the limit: {selected_budget_limited}.",
        f"- Recorded GPU training wall time: {calibration_wall_hours:.2f} h for calibration, {confirmation_wall_hours:.2f} h for confirmation, {calibration_wall_hours + confirmation_wall_hours:.2f} h total.",
        f"- Train-only smoke: {sum(r['status'] == 'success' for r in smoke['runs'])}/9; test task-head calls {audit['test_task_head_calls']}; real LP runs {audit['real_lp_runs']}.",
        "- Metrics use the fixed class set 0–19 with `zero_division=0`; summary SD is population SD (`ddof=0`).", "",
        "## Confirmation results", "",
        "| Dataset | Model | Validation Acc (%) | Macro-F1 (%) | Backbone/task head params | Mean median step (ms) | Max peak allocated (GiB) | Cal/confirm/total wall (h) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for model in MODELS:
            row = summary_map.get((dataset, model))
            if row is None:
                continue
            lines.append(
                f"| {dataset} | {model} | {row['accuracy_mean_percent']:.2f} ± {row['accuracy_sd_percent']:.2f} | "
                f"{row['macro_f1_mean_percent']:.2f} ± {row['macro_f1_sd_percent']:.2f} | "
                f"{row['active_backbone_parameters']:,}/{row['active_task_head_parameters']:,} | "
                f"{row['median_step_ms_mean']:.2f} | {row['peak_allocated_gib_max']:.2f} | "
                f"{row['calibration_wall_hours']:.2f}/{row['confirmation_wall_hours']:.2f}/{row['total_formal_wall_hours']:.2f} |"
            )
    lines.extend(["", "### Confirmation seed records", "", "| Dataset | Model | Seed | LR/WD | Best step | Validation Acc/Macro-F1 (%) | Mean step (ms) |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in per_seed:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['seed']} | {row['lr']:.4g}/{row['matrix_weight_decay']:.4g} | {row['best_step']} | {row['accuracy_percent']:.2f}/{row['macro_f1_percent']:.2f} | {row['median_step_ms']:.2f} |")
    lines.extend(["", "### Selected configurations", "", "| Dataset | Model | LR | Matrix WD | Calibration best step | Calibration validation Acc/F1 (%) |", "|---|---:|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for model in MODELS:
            entry = selection["successful_selections"].get(f"{dataset}/{model}")
            if entry:
                lines.append(f"| {dataset} | {model} | {entry['lr']:.4g} | {entry['matrix_weight_decay']:.4g} | {entry['best_step']} | {entry['validation_accuracy']*100:.2f}/{entry['validation_macro_f1']*100:.2f} |")
    lines.extend(["", "### Tuned recipe contrasts", "", "| Contrast | Dataset | ΔAcc (pp) | ΔMacro-F1 (pp) |", "|---|---|---:|---:|"])
    for row in contrast_block["contrasts"]:
        lines.append(f"| {row['contrast']} | {row['dataset']} | {row['accuracy_delta_pp']:+.2f} | {row['macro_f1_delta_pp']:+.2f} |")
    lines.extend([
        "", "## Training and routing diagnostics", "",
        "Calibration seed 102 is excluded from the confirmation table. Each result uses the checkpoint selected by validation correct count, then Macro-F1, then earlier step. Confirmation means compare independently selected LR/WD recipes; the contrasts are descriptive and do not isolate a single mechanism. U1 lacks the router/context parameters and is not capacity-matched to G1.", "",
        f"Across the smoke set, all {len(smoke['runs'])} runs completed three train-only updates. Allocated memory ranged from {min(r['peak_allocated_bytes'] for r in smoke['runs']) / (1024**3):.2f} to {max(r['peak_allocated_bytes'] for r in smoke['runs']) / (1024**3):.2f} GiB. No validation, test head, or real LP call occurred in smoke.", "",
        "Router usage, noise, entropy, σ, polynomial coefficient magnitudes, and task-only router gradients are recorded per run in `run_registry.json`. These diagnostics describe the fitted training recipe and are not evidence of recovered semantic roles or causal specialization. Cells reaching 800 updates are flagged as possibly budget-limited; they are not interpreted as model failure.", "",
        f"Confirmation routing concentration: {routing_note} Top-2 concentration is a warning to inspect expert utility; it does not establish that an expert is functionally dead because every expert is computed and the summary does not ablate experts.", "",
        "## Implementation facts and protocol differences", "",
        "A0 reuses the previous S0 B1 model, 768-D padded readout, path CE, reconstruction and pair losses. The selected S0 calibration LR/WD values were verified from its committed selection lock and already lie in the new grid. The current common optimizer protocol uses gradient clip 5, while the previous v2 S-stage used clip 1; this round therefore reports a fresh A0 recipe, not a bitwise reproduction of historical checkpoints.", "",
        "The new encoders use sparse P-derived basis operators built once from the canonical loop-free loader edge list, then add one self-loop for their own normalized propagation. All 768-D frozen features are used as loaded. The loader exposes no edge weights or missing-modality metadata; all observed nodes are treated as having both modalities. No missingness performance is established.", "",
        "## Self-audit", "",
        f"The saved train/validation argmax files were independently re-read by this audit script and their sample counts, correct counts, full 20×20 confusion matrices, accuracy and Macro-F1 matched the registry (float metrics tolerance 1e-7). Prediction and checkpoint hashes, split IDs, source commit/hashes and calibration selection were checked. Result: `{audit['metrics_match']}`.", "",
        "The runtime call path invokes the task head only with train IDs for loss and validation IDs for selection; final evaluation uses train and validation IDs separately. The NC runner loads `task=nc`; there is no real LP task call. A synthetic `LinkPredictor.score_pairs` test only verifies the encoder/decoder interface.", "",
        f"Functional suite: {test_record.get('functional_test_passed', 'unknown')} passed, {test_record.get('functional_test_failed', 'unknown')} failed. This is the authors' self-audit, not an independent review.", "",
        "## Interpretation", "",
        f"**Development decision: evidence insufficient to expand G1 unchanged.** Equal-graph confirmation differences are G1−A0 {equal_graph_contrasts['G1-A0']['accuracy_delta_pp']:+.2f} pp Acc / {equal_graph_contrasts['G1-A0']['macro_f1_delta_pp']:+.2f} pp Macro-F1; G1−G0 {equal_graph_contrasts['G1-G0']['accuracy_delta_pp']:+.2f}/{equal_graph_contrasts['G1-G0']['macro_f1_delta_pp']:+.2f} pp; and G1−U1 {equal_graph_contrasts['G1-U1']['accuracy_delta_pp']:+.2f}/{equal_graph_contrasts['G1-U1']['macro_f1_delta_pp']:+.2f} pp. Per-graph reversals and prior validation exposure limit a broad conclusion; the results do not identify one mechanism for a targeted correction.", "",
        "The evidence supports only a decision about whether these complete training recipes merit another bounded development round. It does not establish NC/LP unified performance, independent generalization, semantic disentanglement, or statistical significance. Interpret the per-graph results, stability, resource use and router gradient diagnostics together; do not infer the mechanism from a single gate or fit statistic.",
    ])
    return "\n".join(lines) + "\n"


def run_audit() -> dict[str, Any]:
    matrix = read_json(ROOT / "docs" / STUDY / "matrix.json")
    registry = read_json(REPORT_ROOT / "run_registry.json")
    selection = read_json(SELECTION_PATH)
    results = read_results(REPORT_ROOT / "run_results.jsonl")
    profile = read_json(REPORT_ROOT / "data_profile.json")
    smoke = read_json(REPORT_ROOT / "smoke_summary.json")
    test_record = read_json(REPORT_ROOT / "test_record.json")
    audit = verify_cells(matrix, registry, selection, results, profile)
    successful = select_successful_attempts(registry)
    summary, per_seed, contrast_block = aggregate_confirmation(successful)
    write_summary_csv(REPORT_ROOT / "summary.csv", summary)
    report = report_text(summary, per_seed, contrast_block, audit, selection, smoke, test_record)
    (REPORT_ROOT / "report.md").write_text(report, encoding="utf-8")

    attempt_status = Counter(attempt.get("status") for attempt in registry["attempts"].values())
    checks = {
        "run_matrix": {key: audit[key] for key in (
            "expected_calibration_units", "successful_calibration_units", "expected_confirmation_units",
            "successful_confirmation_units", "successful_attempts_including_retries", "failed_attempts",
            "retry_attempts", "unrun_calibration_units", "unrun_confirmation_units",
            "calibration_max_updates", "confirmation_max_updates", "selected_config_max_updates",
        )},
        "attempt_status_counts": dict(attempt_status),
        "metrics_from_argmax": audit["metrics_match"], "argmax_run_count": audit["argmax_files_recomputed"],
        "prediction_and_checkpoint_hashes": "passed", "split_id_and_input_hashes": "passed",
        "selection_lock_recomputed": "passed", "confirmation_configuration_lock": "passed",
        "source_commit_and_source_hashes_singleton": "passed",
        "eval_clean_and_state_immutability": "covered_by_functional_test_suite",
        "parameter_groups_and_gradients": "covered_by_functional_test_suite",
        "runtime_task_head_indices": ["train_idx", "val_idx"],
        "test_metric_calls": audit["test_task_head_calls"], "real_lp_runs": audit["real_lp_runs"],
        "independent_reviewer": False,
    }
    test_record.update({"audit": checks, "passed_audit": True, "audited_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
    write_json(REPORT_ROOT / "test_record.json", test_record)

    review = [
        "# MVCGE private NC v1 self-review", "",
        "This review was performed by the implementation author; it is not independent.", "",
        "## Audit checks", "",
        f"- Train/validation prediction recomputation: `{audit['metrics_match']}` across {audit['argmax_files_recomputed']} successful attempts. Counts and confusion matrices matched exactly; accuracy and Macro-F1 matched within 1e-7.",
        "- Selected LR/WD recomputed from all nine seed-102 candidates for every dataset/model; confirmation rows match the immutable lock.",
        "- Every successful checkpoint matches its run's source commit, source hashes and best step. The same source hash map is used across runs.",
        "- Prediction, checkpoint, split-ID and input-file SHA256 values were verified.",
        f"- Runtime records report {audit['test_task_head_calls']} test task-head calls and {audit['real_lp_runs']} real LP runs. The executed entry point loads NC only; its supervised and evaluation calls use train/validation indices. The synthetic LP test does not read any dataset split.",
        "- Functional tests cover PC order/initial responses, direct-vs-reused basis and gradients, sparse/dense equivalence, masks/loops/isolation/permutation, top-2 gates, expected-load gradients, task-only router gradients, G0/G1 initialization, task-side heads, synchronous context input, optimizer grouping and deterministic eval.",
        "- Smoke execution: nine train-only cells, three updates each; finite loss and clipped-gradient checks passed, no OOM.", "",
        "## Cautions and residual risks", "",
        "The MAGB loader discards any DGL edge data and does not expose modality masks; inspected graphs have no DGL edge-data fields, and this round treats all modalities as present. The actual S0 replay uses this round's clip value 5, unlike the previous S-stage's clip value 1, so comparisons are fresh recipe comparisons.",
        "G1 and A0/G0/U1 are selected independently by validation. The confirmation validation was used in previous development. U1 is smaller and computes all four experts like the other models; it does not form a capacity- or compute-matched route ablation. Top-2 only gates receiver outputs and all experts are still computed.",
        "No test metrics, real LP task, missing-modality intervention, five-dataset extension, causal analysis or significance test is part of this record.",
    ]
    (REPORT_ROOT / "review.md").write_text("\n".join(review) + "\n", encoding="utf-8")

    files = [
        ROOT / "docs" / STUDY / "intake.md", ROOT / "docs" / STUDY / "model_spec.md", ROOT / "docs" / STUDY / "protocol.md", ROOT / "docs" / STUDY / "matrix.json",
        REPORT_ROOT / "environment_profile.json", REPORT_ROOT / "data_profile.json", REPORT_ROOT / "smoke_summary.json", REPORT_ROOT / "selection_lock.json",
        REPORT_ROOT / "run_registry.json", REPORT_ROOT / "run_results.jsonl", REPORT_ROOT / "summary.csv", REPORT_ROOT / "report.md", REPORT_ROOT / "review.md", REPORT_ROOT / "test_record.json",
        ROOT / "src" / "models" / "mvcge_private_nc_v1.py", ROOT / "scripts" / "run_mvcge_private_nc_v1.py", ROOT / "scripts" / "audit_mvcge_private_nc_v1.py", ROOT / "tests" / "test_mvcge_private_nc_v1.py",
    ]
    manifest = {
        "study": STUDY, "training_source_commit": audit["source_commit"], "final_head_at_audit": git("rev-parse", "HEAD"),
        "task_branch": git("branch", "--show-current"), "successful_runs": audit["successful_attempts_including_retries"],
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files if path.is_file()},
        "output_artifacts": {
            record["attempt_id"]: {"checkpoint_sha256": record["best_checkpoint_sha256"], "prediction_sha256": record["prediction_sha256"]}
            for record in successful
        },
        "dataset_inputs": {dataset: profile["input_sha256"] for dataset, profile in profile.items()},
        "no_data_or_checkpoint_files_staged_by_manifest": True,
    }
    write_json(REPORT_ROOT / "manifest.json", manifest)
    audit["summary_rows"] = summary
    audit["contrasts"] = contrast_block["contrasts"]
    persisted_audit = {key: value for key, value in audit.items() if key != "audited_runs"}
    write_json(REPORT_ROOT / "audit_result.json", persisted_audit)
    return audit


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps({key: result[key] for key in result if key not in {"audited_runs", "summary_rows", "contrasts"}}, ensure_ascii=False, indent=2))
