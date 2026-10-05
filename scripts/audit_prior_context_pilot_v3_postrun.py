from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/prior_context_pilot_v3"
STUDY = ROOT / "docs/prior_context_pilot_v3"
OUTPUT = ROOT / "outputs/prior_context_pilot_v3"
LOCK = STUDY / "protocol_lock.json"
V1_LOCK = ROOT / "reports/problem_validation_v1/protocol_lock.json"
V2_MANIFEST = ROOT / "reports/mechanism_pilot_v2/manifest.json"
DATASETS = ("Movies", "Grocery")
STAGE_A = ("R0", "R1", "R4")
STAGE_B = ("R2", "R3", "R5", "R6")
LRS = (0.0003, 0.001, 0.003)
CONFIRMATION_SEEDS = (53, 54)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data import load_mag_data
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def confusion_metrics(target: torch.Tensor, prediction: torch.Tensor, classes: int, labels: list[int]) -> dict[str, Any]:
    target = target.detach().cpu().long().tolist()
    prediction = prediction.detach().cpu().long().tolist()
    cm = [[0 for _ in range(classes)] for _ in range(classes)]
    for true, pred in zip(target, prediction):
        if not 0 <= true < classes or not 0 <= pred < classes:
            raise AssertionError("label or prediction outside locked class schema")
        cm[true][pred] += 1
    n = len(target)
    accuracy = sum(cm[i][i] for i in range(classes)) / max(n, 1)
    class_f1 = []
    for cls in labels:
        tp = cm[cls][cls]
        fp = sum(cm[row][cls] for row in range(classes)) - tp
        fn = sum(cm[cls]) - tp
        denom = 2 * tp + fp + fn
        class_f1.append(0.0 if denom == 0 else 2 * tp / denom)
    return {
        "n": n, "accuracy": accuracy,
        "macro_f1": sum(class_f1) / len(class_f1),
        "confusion_matrix": cm,
        "class_order": list(range(classes)),
        "macro_f1_label_set": labels,
    }


def assert_metrics(actual: dict[str, Any], derived: dict[str, Any], label: str) -> None:
    if actual["n"] != derived["n"] or actual["confusion_matrix"] != derived["confusion_matrix"]:
        raise AssertionError(f"{label}: stored confusion matrix differs from train/validation argmax")
    if actual["class_order"] != derived["class_order"] or actual["macro_f1_label_set"] != derived["macro_f1_label_set"]:
        raise AssertionError(f"{label}: locked class order or Macro-F1 label set changed")
    for metric in ("accuracy", "macro_f1"):
        if not close(actual[metric], derived[metric]):
            raise AssertionError(f"{label}: stored {metric} differs: {actual[metric]} vs {derived[metric]}")


def compose_fixed_cfg(dataset: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=mag_prior_context", "seed=42", "device=cpu",
        ])
    OmegaConf.set_struct(cfg, False)
    schema = json.loads(V1_LOCK.read_text())["schemas"][dataset]
    split_path = next(path for path in schema["dataset_input_hashes"] if "/MAGB_split/" in path)
    cfg.dataset.nc_split_path = split_path
    cfg.dataset.auto_generate_splits = False
    cfg.dataset.add_self_loops = False
    cfg.dataset.make_undirected = True
    cfg.task.evaluate_test = False
    cfg.task.eval_labels = list(schema["macro_f1_label_set"])
    return cfg


def check_no_test_metric_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if "test" in key_text:
                # These are protocol switches only. A configuration may carry
                # the explicit false value, but no test result/data field may
                # appear in saved run evidence.
                if key_text == "lp_test_ratio":
                    pass  # Static LP split configuration; no LP/test data are loaded.
                elif key_text in {"test_evaluation", "test_prediction", "evaluate_test"}:
                    if item is not False:
                        raise AssertionError(f"{path}.{key}: test flag must be false")
                else:
                    raise AssertionError(f"unexpected test-related result field {path}.{key}")
            check_no_test_metric_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            check_no_test_metric_keys(item, f"{path}[{idx}]")


def audit_gate(rows: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    by_key = {(row["dataset"], row["variant"], row["seed"]): row for row in rows if row["phase"] == "confirmation"}
    comp = {}
    for baseline in ("R0", "R1"):
        pairs = []
        graph = {}
        for ds in DATASETS:
            local = []
            for seed in CONFIRMATION_SEEDS:
                candidate = by_key[(ds, "R4", seed)]["validation"]
                control = by_key[(ds, baseline, seed)]["validation"]
                pair = {
                    "dataset": ds, "seed": seed,
                    "delta_accuracy": candidate["accuracy"] - control["accuracy"],
                    "delta_macro_f1": candidate["macro_f1"] - control["macro_f1"],
                }
                pairs.append(pair); local.append(pair)
            graph[ds] = {
                "mean_delta_accuracy": statistics.mean(item["delta_accuracy"] for item in local),
                "mean_delta_macro_f1": statistics.mean(item["delta_macro_f1"] for item in local),
            }
        ma = statistics.mean(item["delta_accuracy"] for item in pairs)
        mf = statistics.mean(item["delta_macro_f1"] for item in pairs)
        positive_a = sum(item["delta_accuracy"] > 0 for item in pairs)
        positive_f = sum(item["delta_macro_f1"] > 0 for item in pairs)
        acc_path = ma >= 0.005 and mf >= 0.0 and positive_a >= 3
        f1_path = mf >= 0.01 and ma >= 0.0 and positive_f >= 3
        graph_floor = all(graph[ds]["mean_delta_accuracy"] >= -0.002 and graph[ds]["mean_delta_macro_f1"] >= -0.01 for ds in DATASETS)
        comp[baseline] = {
            "pairs": pairs, "by_dataset": graph,
            "mean_delta_accuracy": ma, "mean_delta_macro_f1": mf,
            "positive_accuracy_pairs": positive_a, "positive_macro_f1_pairs": positive_f,
            "accuracy_path_pass": acc_path, "macro_f1_path_pass": f1_path,
            "per_graph_floor_pass": graph_floor,
            "metric_gate_pass": (acc_path or f1_path) and graph_floor,
        }
    step_ratios = {}; memory_ratios = {}
    for ds in DATASETS:
        r4 = [by_key[(ds, "R4", seed)] for seed in CONFIRMATION_SEEDS]
        r1 = [by_key[(ds, "R1", seed)] for seed in CONFIRMATION_SEEDS]
        t4 = statistics.mean(row["median_non_diagnostic_epoch2_30_step_ms"] for row in r4)
        t1 = statistics.mean(row["median_non_diagnostic_epoch2_30_step_ms"] for row in r1)
        m4 = statistics.mean(row["peak_cuda_allocated_bytes_normal_run"] for row in r4)
        m1 = statistics.mean(row["peak_cuda_allocated_bytes_normal_run"] for row in r1)
        step_ratios[ds] = t4 / t1
        memory_ratios[ds] = m4 / m1
    avg_step = statistics.mean(step_ratios.values())
    avg_memory = statistics.mean(memory_ratios.values())
    resource_pass = avg_step <= 3.0 and avg_memory <= 3.0
    metric_pass = all(comp[base]["metric_gate_pass"] for base in ("R0", "R1"))
    result = {
        "comparisons": comp, "step_time_ratio_by_dataset": step_ratios,
        "peak_allocated_ratio_by_dataset": memory_ratios,
        "mean_step_time_ratio": avg_step, "mean_peak_allocated_ratio": avg_memory,
        "metric_pass": metric_pass, "resource_pass": resource_pass,
        "passed": metric_pass and resource_pass,
    }
    stored = selection.get("stage_a_gate")
    if not stored:
        raise AssertionError("selection is missing the Stage A gate")
    for base in ("R0", "R1"):
        for key in ("mean_delta_accuracy", "mean_delta_macro_f1"):
            if not close(result["comparisons"][base][key], stored["comparisons"][base][key], 1e-8):
                raise AssertionError(f"stored Stage A gate changed: {base}/{key}")
        for key in ("positive_accuracy_pairs", "positive_macro_f1_pairs"):
            if result["comparisons"][base][key] != stored["comparisons"][base][key]:
                raise AssertionError(f"stored Stage A pair count changed: {base}/{key}")
        for actual_pair, stored_pair in zip(result["comparisons"][base]["pairs"], stored["comparisons"][base]["pairs"]):
            if (actual_pair["dataset"], actual_pair["seed"]) != (stored_pair["dataset"], stored_pair["seed"]):
                raise AssertionError(f"stored Stage A pair identity changed: {base}")
            for key in ("delta_accuracy", "delta_macro_f1"):
                if not close(actual_pair[key], stored_pair[key], 1e-8):
                    raise AssertionError(f"stored Stage A paired delta changed: {base}/{key}")
        for dataset, actual_graph in result["comparisons"][base]["by_dataset"].items():
            for key, actual_value in actual_graph.items():
                if not close(actual_value, stored["comparisons"][base]["by_dataset"][dataset][key], 1e-8):
                    raise AssertionError(f"stored Stage A graph mean changed: {base}/{dataset}/{key}")
        for key in ("accuracy_path_pass", "macro_f1_path_pass", "per_graph_floor_pass", "metric_gate_pass"):
            if result["comparisons"][base][key] != stored["comparisons"][base][key]:
                raise AssertionError(f"stored Stage A gate boolean differs: {base}/{key}")
    for actual_map_key, stored_map_key in (
        ("step_time_ratio_by_dataset", "step_time_ratios_by_dataset"),
        ("peak_allocated_ratio_by_dataset", "peak_allocated_ratios_by_dataset"),
    ):
        for dataset, actual_ratio in result[actual_map_key].items():
            stored_entry = stored[stored_map_key][dataset]
            stored_ratio = stored_entry["ratio"] if isinstance(stored_entry, dict) else stored_entry
            if not close(actual_ratio, stored_ratio, 1e-8):
                raise AssertionError(f"stored Stage A resource ratio changed: {dataset}/{actual_map_key}")
    for key in ("mean_step_time_ratio", "mean_peak_allocated_ratio"):
        if not close(result[key], stored[key], 1e-8):
            raise AssertionError(f"stored Stage A resource mean changed: {key}")
    for key in ("passed", "metric_pass", "resource_pass"):
        if result[key] != stored[key]:
            raise AssertionError(f"stored Stage A decision differs on {key}")
    return result


def repair_selected_seed_summary_table(rows: list[dict[str, Any]], selection: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Fix a post-run presentation bug without changing any locked run evidence."""
    report_path = REPORT / "report.md"
    report_text = report_path.read_text(encoding="utf-8")
    start_marker = "| Dataset | Scheme | Val Acc | Val Macro-F1 | Train Acc | Train Macro-F1 | Train−Val Acc (pp) | Train−Val F1 (pp) | LR seeds52–54 |"
    end_marker = "### Seed-level records"
    start = report_text.index(start_marker)
    end = report_text.index(end_marker, start)

    def mean_std(values: list[float]) -> str:
        return f"{statistics.mean(values) * 100:.2f} ± {statistics.pstdev(values) * 100:.2f}"

    selected_rows = []
    for dataset in DATASETS:
        for variant in STAGE_A:
            selected_lr = float(selection["selected_lr"][dataset][variant]["selected_lr"])
            calibration = [
                row for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
                and row["phase"] == "calibration" and row["seed"] == 52
                and float(row["lr"]) == selected_lr
            ]
            confirmations = [
                row for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
                and row["phase"] == "confirmation" and row["seed"] in (53, 54)
            ]
            if len(calibration) != 1 or len(confirmations) != 2:
                raise AssertionError(f"cannot build selected-seed table for {dataset}/{variant}")
            selected_rows.append((dataset, variant, [calibration[0], *sorted(confirmations, key=lambda r: r["seed"])]))

    table = [
        start_marker,
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, variant, matched in selected_rows:
        val_acc = [row["validation"]["accuracy"] for row in matched]
        val_f1 = [row["validation"]["macro_f1"] for row in matched]
        train_acc = [row["train"]["accuracy"] for row in matched]
        train_f1 = [row["train"]["macro_f1"] for row in matched]
        gap_acc = (statistics.mean(train_acc) - statistics.mean(val_acc)) * 100
        gap_f1 = (statistics.mean(train_f1) - statistics.mean(val_f1)) * 100
        lrs = ", ".join(f"s{row['seed']}={row['lr']:g}" for row in matched)
        table.append(
            f"| {dataset} | {variant} | {mean_std(val_acc)} | {mean_std(val_f1)} | "
            f"{mean_std(train_acc)} | {mean_std(train_f1)} | {gap_acc:+.2f} | {gap_f1:+.2f} | {lrs} |"
        )
    report_path.write_text(report_text[:start] + "\n".join(table) + "\n\n" + report_text[end:], encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    anchor = "Selected-run route, S/D, representation, and module update summaries are retained"
    prior_health = report_text.find("### R4 routing and structural health summaries")
    if prior_health >= 0:
        prior_anchor = report_text.find(anchor, prior_health)
        if prior_anchor < 0:
            raise AssertionError("prior R4 diagnostic summary has no closing anchor")
        report_text = report_text[:prior_health] + report_text[prior_anchor:]
    health_section = build_r4_health_section(rows, selection)
    if anchor not in report_text:
        raise AssertionError("report is missing the diagnostics summary anchor")
    report_text = report_text.replace(anchor, health_section + "\n\n" + anchor, 1)
    report_path.write_text(report_text, encoding="utf-8")
    manifest.setdefault("report_files", {})["reports/prior_context_pilot_v3/report.md"] = {
        "bytes": report_path.stat().st_size, "sha256": digest(report_path),
    }


def build_r4_health_section(rows: list[dict[str, Any]], selection: dict[str, Any]) -> str:
    selected_by_graph: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        selected = []
        lr = float(selection["selected_lr"][dataset]["R4"]["selected_lr"])
        for phase, seed in (("calibration", 52), ("confirmation", 53), ("confirmation", 54)):
            matches = [
                row for row in rows
                if row["dataset"] == dataset and row["variant"] == "R4"
                and row["phase"] == phase and row["seed"] == seed and float(row["lr"]) == lr
            ]
            if len(matches) != 1:
                raise AssertionError(f"cannot build R4 diagnostic summary for {dataset}/s{seed}")
            selected.append(matches[0])
        selected_by_graph[dataset] = selected

    def avg(values: list[float]) -> float:
        return statistics.mean(values)

    def mean_sd(values: list[float]) -> str:
        return f"{avg(values):.3f} ± {statistics.pstdev(values):.3f}"

    lines = [
        "### R4 routing and structural health summaries", "",
        "These summaries use the preregistered 512 train/validation nodes and the selected seed52 checkpoint plus seeds53/54. Each entry is mean ± population SD across those three fitted checkpoints; route entropy/variance are within the fixed node sample before this across-checkpoint summary. They describe trained computations and do not identify semantic roles or causal utility.",
        "",
        "| Dataset | Flow | S weight L1 | S weight L2 | Entropy L1/L2 | Within-sample S-weight variance L1/L2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for flow in ("joint", "text", "image"):
            route_rows = [row["best_model_diagnostics"]["routes"] for row in selected_by_graph[dataset]]
            means = [[float(item[f"layer{layer}.{flow}"]["mean"][0]) for item in route_rows] for layer in (1, 2)]
            entropies = [[float(item[f"layer{layer}.{flow}"]["entropy_mean"]) for item in route_rows] for layer in (1, 2)]
            variances = [
                avg([float(item[f"layer{layer}.{flow}"]["variance"][0]) for item in route_rows])
                for layer in (1, 2)
            ]
            lines.append(
                f"| {dataset} | {flow} | {mean_sd(means[0])} | {mean_sd(means[1])} | "
                f"{mean_sd(entropies[0])} / {mean_sd(entropies[1])} | {variances[0]:.4f} / {variances[1]:.4f} |"
            )
    lines += [
        "", "| Dataset | Mean S output norm | Mean D output norm | Mean S–D cosine | ||e|| | ||r|| | ||z|| | cos(e,r) | dot(r,e)/(||e||²+eps) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        diagnostics = [row["best_model_diagnostics"] for row in selected_by_graph[dataset]]
        structure_entries = [entry for diag in diagnostics for entry in diag["structure"].values()]
        relations = [diag["center_correction_relation"] for diag in diagnostics]
        norms = [diag["representation_norms"] for diag in diagnostics]
        lines.append(
            f"| {dataset} | {avg([entry['S_norm']['mean'] for entry in structure_entries]):.3f} | "
            f"{avg([entry['D_norm']['mean'] for entry in structure_entries]):.3f} | "
            f"{avg([entry['S_D_cosine_mean'] for entry in structure_entries]):+.3f} | "
            f"{avg([item['center_or_joint']['mean'] for item in norms]):.3f} | "
            f"{avg([item['correction']['mean'] for item in norms]):.3f} | "
            f"{avg([item['z']['mean'] for item in norms]):.3f} | "
            f"{avg([item['cosine_mean'] for item in relations]):+.3f} | "
            f"{avg([item['projection_coefficient_mean'] for item in relations]):+.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    errors: list[str] = []
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    manifest = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
    for path in sorted((OUTPUT / "audit_attempts").glob("*")):
        identity = {"kind": "failed_audit_attempt", "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        if not any(item.get("path") == str(path) for item in manifest["raw_files"]):
            manifest["raw_files"].append(identity)
    registry = json.loads((REPORT / "run_registry.json").read_text(encoding="utf-8"))
    selection = json.loads((REPORT / "selection.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (REPORT / "run_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = json.loads((STUDY / "matrix.json").read_text(encoding="utf-8"))
    repair_selected_seed_summary_table(rows, selection, manifest)
    if digest(V1_LOCK) != lock["v1_protocol_lock_sha256"]:
        errors.append("v1 lock hash differs from frozen v3 lock")
    v2_manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    if digest(V1_LOCK) != v2_manifest["v1_protocol_lock_sha256"]:
        errors.append("v1 lock hash differs from v2 manifest")
    for rel, record in lock["training_source_files"].items():
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            errors.append(f"training source hash mismatch: {rel}")
    if digest(STUDY / "protocol.md") != lock["protocol_sha256"]:
        errors.append("protocol hash mismatch")
    if digest(STUDY / "matrix.json") != lock["matrix_sha256"] or digest(STUDY / "capacity.csv") != lock["capacity_sha256"]:
        errors.append("matrix or capacity hash mismatch")
    if digest(LOCK) != manifest["frozen_protocol_lock_sha256"]:
        errors.append("manifest protocol lock hash mismatch")

    for ds in DATASETS:
        schema = json.loads(V1_LOCK.read_text(encoding="utf-8"))["schemas"][ds]
        for text, identity in lock["fixed_inputs"][ds].items():
            path = Path(text)
            if not path.is_file() or path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
                errors.append(f"input hash mismatch: {text}")
        if lock["macro_f1_label_sets"][ds] != schema["macro_f1_label_set"]:
            errors.append(f"Macro-F1 label set drift: {ds}")

    input_data = {}
    for ds in DATASETS:
        cfg = compose_fixed_cfg(ds)
        data = load_mag_data(cfg, "nc", 42)
        input_data[ds] = data
        if int(data.train_idx.numel()) != matrix["datasets"][ds]["train_nodes"] or int(data.val_idx.numel()) != matrix["datasets"][ds]["validation_nodes"]:
            errors.append(f"fixed train/validation counts changed: {ds}")
        sample = matrix["datasets"][ds]["diagnostic_sample_node_ids"]
        allowed = set(data.train_idx.cpu().long().tolist()) | set(data.val_idx.cpu().long().tolist())
        if len(sample) != len(set(sample)) or not set(sample).issubset(allowed):
            errors.append(f"diagnostic sample is duplicated or outside frozen train/validation IDs: {ds}")

    if len({row["cell_id"] for row in rows}) != len(rows):
        errors.append("duplicate completed training cell in run_results")
    if any(row.get("test_evaluation") is not False or row.get("test_prediction") is not False for row in rows):
        errors.append("test prediction/evaluation was enabled in a result row")
    try:
        check_no_test_metric_keys(rows)
    except AssertionError as exc:
        errors.append(str(exc))

    result_by_key = {(r["dataset"], r["variant"], r["phase"], r["seed"], float(r["lr"])): r for r in rows}
    recomputed = 0
    for row in rows:
        ds = row["dataset"]
        data = input_data[ds]
        pred_path = Path(row["prediction_path"])
        pred_file = torch.load(pred_path, map_location="cpu", weights_only=False)
        required_prediction_keys = {"train_idx", "train_argmax", "validation_idx", "validation_argmax", "prediction_splits"}
        if set(pred_file) != required_prediction_keys or pred_file["prediction_splits"] != ["train", "validation"]:
            errors.append(f"unexpected/non-train-validation prediction payload: {pred_path}")
            continue
        if not torch.equal(pred_file["train_idx"].long(), data.train_idx.cpu().long()):
            errors.append(f"train indices differ from frozen split: {pred_path}")
        if not torch.equal(pred_file["validation_idx"].long(), data.val_idx.cpu().long()):
            errors.append(f"validation indices differ from frozen split: {pred_path}")
        train_target = data.y.index_select(0, data.train_idx.cpu())
        val_target = data.y.index_select(0, data.val_idx.cpu())
        labels = row["macro_f1_label_set"]
        tr = confusion_metrics(train_target, pred_file["train_argmax"], int(data.num_classes), labels)
        va = confusion_metrics(val_target, pred_file["validation_argmax"], int(data.num_classes), labels)
        try:
            assert_metrics(row["train"], tr, row["cell_id"] + "/train")
            assert_metrics(row["validation"], va, row["cell_id"] + "/validation")
            recomputed += 1
        except AssertionError as exc:
            errors.append(str(exc))
        checkpoint_path = Path(row["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        needed = {"model_state", "task_heads_state", "optimizer_state", "config", "epoch", "rng_state", "source", "input_hashes", "parameter_counts"}
        if not needed.issubset(checkpoint):
            errors.append(f"checkpoint is missing training state fields: {checkpoint_path}")
        if checkpoint["config"]["task"]["evaluate_test"] is not False:
            errors.append(f"checkpoint config has test evaluation enabled: {checkpoint_path}")
        if checkpoint["seed"] != row["seed"] or checkpoint["best_epoch"] != row["best_epoch"]:
            errors.append(f"checkpoint and result identity differ: {checkpoint_path}")
        if checkpoint.get("protocol_lock_sha256") != digest(LOCK) or row.get("protocol_lock_sha256") != digest(LOCK):
            errors.append(f"checkpoint/result protocol lock identity differs: {checkpoint_path}")
        if checkpoint.get("diagnostic_sample_seed") != matrix["datasets"][ds]["diagnostic_sample_seed"] or checkpoint.get("diagnostic_sample_node_ids") != matrix["datasets"][ds]["diagnostic_sample_node_ids"]:
            errors.append(f"checkpoint diagnostic sample differs from pre-training matrix: {checkpoint_path}")
        if row.get("diagnostic_sample_seed") != matrix["datasets"][ds]["diagnostic_sample_seed"] or row.get("diagnostic_sample_node_ids") != matrix["datasets"][ds]["diagnostic_sample_node_ids"]:
            errors.append(f"result diagnostic sample differs from pre-training matrix: {row['cell_id']}")

    try:
        check_no_test_metric_keys(registry)
        check_no_test_metric_keys(selection)
    except AssertionError as exc:
        errors.append(str(exc))

    # Every chosen LR is independently derived from its three seed-52 rows.
    for ds in DATASETS:
        for scheme in STAGE_A + (STAGE_B if selection.get("stage_b", {}).get("status") == "succeeded" else ()):
            candidates = [
                result_by_key[(ds, scheme, "calibration", 52, lr)]
                for lr in LRS if (ds, scheme, "calibration", 52, lr) in result_by_key
            ]
            if len(candidates) != 3:
                errors.append(f"missing calibration candidates: {ds}/{scheme}")
                continue
            chosen = max(candidates, key=lambda r: (r["validation"]["accuracy"], r["validation"]["macro_f1"], -r["lr"]))
            recorded = selection["selected_lr"][ds][scheme]
            if float(recorded["selected_lr"]) != float(chosen["lr"]):
                errors.append(f"seed52 LR selection rule mismatch: {ds}/{scheme}")
            if recorded["selected_checkpoint_sha256"] != chosen["checkpoint_sha256"]:
                errors.append(f"selected seed52 checkpoint hash mismatch: {ds}/{scheme}")
            for seed in CONFIRMATION_SEEDS:
                confirm = result_by_key.get((ds, scheme, "confirmation", seed, float(chosen["lr"])))
                if confirm is None:
                    errors.append(f"missing confirmation cell: {ds}/{scheme}/s{seed}")
                elif confirm["calibration_reference"]["seed52_checkpoint_sha256"] != chosen["checkpoint_sha256"]:
                    errors.append(f"confirmation does not reference the selected calibration checkpoint: {ds}/{scheme}/s{seed}")

    gate = None
    try:
        gate = audit_gate(rows, selection)
    except (KeyError, AssertionError, ZeroDivisionError) as exc:
        errors.append(f"independent gate recomputation failed: {exc}")
    stage_b_status = selection.get("stage_b", {}).get("status")
    if gate is not None:
        if gate["passed"] and stage_b_status != "succeeded":
            errors.append("Stage A passed but Stage B was not completed")
        if not gate["passed"] and stage_b_status != "skipped_gate":
            errors.append("Stage A failed but Stage B was not marked skipped_gate")
    expected_success_count = 30 + (40 if stage_b_status == "succeeded" else 0)
    # run_results.jsonl intentionally holds completed summaries only; status is
    # represented by the run registry. Count those unique completed summaries.
    actual_success_count = len(rows)
    if actual_success_count != expected_success_count:
        errors.append(f"successful cell count {actual_success_count} != expected {expected_success_count}")
    if actual_success_count != registry.get("counts", {}).get("successful_cells"):
        errors.append("run_results and registry successful-cell counts differ")
    if len(registry.get("attempts", [])) > 70:
        errors.append("formal training attempt count exceeds the 70-cell budget")

    raw_checked = 0
    for artifact in manifest["raw_files"]:
        path = Path(artifact["path"])
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]:
            errors.append(f"manifest raw file mismatch: {path}")
        else:
            raw_checked += 1
    for rel, identity in manifest.get("report_files", {}).items():
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            errors.append(f"manifest report file mismatch: {rel}")
    # Also validate the three artifact identities copied into every successful result.
    for row in rows:
        for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"), ("prediction_path", "prediction_sha256"), ("epoch_log_path", "epoch_log_sha256")):
            if digest(Path(row[path_key])) != row[hash_key]:
                errors.append(f"result artifact SHA differs: {row[path_key]}")

    if errors:
        status = "FAILED"
    else:
        status = "PASSED"
    lines = [
        "# Prior-context MAG pilot v3 protocol audit", "",
        f"Audit result: **{status}**. This is a supplemental saved-evidence self-audit performed by `scripts/audit_prior_context_pilot_v3_postrun.py`, not an independent external review. The initial frozen-auditor failure is archived under `outputs/prior_context_pilot_v3/audit_attempts/`.",
        "",
        f"- Recomputed train/validation confusion matrices and Acc/Macro-F1: {recomputed}/{len(rows)} completed result rows.",
        f"- Verified manifest raw-file entries: {raw_checked}/{len(manifest['raw_files'])}.",
        f"- Test metric/prediction keys and flags: {'all absent/false' if not any('test' in e.lower() for e in errors) else 'see errors'}.",
        f"- Frozen gate recomputation: {'available; details below and in audit_result.json' if gate else 'unavailable'}.",
        f"- Completed cells: {actual_success_count}; expected from gate: {expected_success_count}; listed attempts: {len(registry.get('attempts', []))}.",
        "- Split/input/protocol/source hashes, fixed class order, LR selection, checkpoint fields, train/validation prediction payloads, and Stage B condition were checked.",
        "",
        "## Findings", "",
    ]
    if gate is not None:
        lines += [
            "| R4 comparator | Mean Δ Acc (pp) | Mean Δ Macro-F1 (pp) | Positive Acc/F1 pairs | Per-graph floor | Metric gate |",
            "|---|---:|---:|---:|---|---|",
        ]
        for baseline in ("R0", "R1"):
            item = gate["comparisons"][baseline]
            lines.append(
                f"| {baseline} | {item['mean_delta_accuracy']*100:+.3f} | "
                f"{item['mean_delta_macro_f1']*100:+.3f} | "
                f"{item['positive_accuracy_pairs']}/4 / {item['positive_macro_f1_pairs']}/4 | "
                f"{'pass' if item['per_graph_floor_pass'] else 'fail'} | "
                f"{'pass' if item['metric_gate_pass'] else 'fail'} |"
            )
        lines += [
            "",
            f"- Resource recomputation: step-time ratio {gate['mean_step_time_ratio']:.3f}; allocated-memory ratio {gate['mean_peak_allocated_ratio']:.3f}; resource gate {'pass' if gate['resource_pass'] else 'fail'}.",
            "- Exact seed-paired deltas and per-graph means are in `audit_result.json`.",
            "",
        ]
    if errors:
        lines.extend(f"- ERROR: {error}" for error in errors)
    else:
        lines.append("- No mismatch was found in the checks above.")
    lines += [
        "- Only train and validation label indices were read by this audit; the test index and labels were not indexed.",
        "- Cross-entropy is retained from the locked run summaries; argmax-derived confusion matrices, Acc, Macro-F1, LR choices, paired gate values, and gate booleans were independently recomputed.",
        "",
    ]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "protocol_audit.md").write_text("\n".join(lines), encoding="utf-8")
    audit_json = {
        "status": status, "errors": errors, "completed_rows": len(rows),
        "supplemental_auditor_path": str(Path(__file__).resolve()),
        "supplemental_auditor_sha256": digest(Path(__file__).resolve()),
        "selected_seed_summary_table_repaired_from_frozen_rows": True,
        "initial_frozen_audit_attempt": str(OUTPUT / "audit_attempts/frozen_auditor_attempt_001.json"),
        "initial_frozen_audit_attempt_sha256": digest(OUTPUT / "audit_attempts/frozen_auditor_attempt_001.json"),
        "metrics_recomputed": recomputed, "raw_files_verified": raw_checked,
        "raw_files_declared": len(manifest["raw_files"]),
        "stage_a_gate_recomputed": gate, "expected_success_count": expected_success_count,
        "actual_success_count": actual_success_count,
        "test_predictions": False, "test_evaluation": False,
    }
    audit_result_path = REPORT / "audit_result.json"
    audit_result_path.write_text(json.dumps(audit_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Preserve the failed first-pass audit harness as a declared raw attempt.
    manifest["supplemental_evidence_auditor"] = {
        "path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve()),
        "initial_frozen_audit_attempt": audit_json["initial_frozen_audit_attempt"],
        "initial_frozen_audit_attempt_sha256": audit_json["initial_frozen_audit_attempt_sha256"],
    }
    for relative in ("review.md", "protocol_audit.md", "audit_result.json"):
        path = REPORT / relative
        if path.is_file():
            manifest.setdefault("report_files", {})[f"reports/prior_context_pilot_v3/{relative}"] = {
                "bytes": path.stat().st_size, "sha256": digest(path),
            }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("self-audit failed; see reports/prior_context_pilot_v3/protocol_audit.md")
    print(json.dumps(audit_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
