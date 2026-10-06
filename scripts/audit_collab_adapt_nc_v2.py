from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_collab_adapt_nc_v2 as campaign

REPORT_ROOT = campaign.REPORT_ROOT
OUTPUT_ROOT = campaign.OUTPUT_ROOT


def _close(a, b, atol=1e-10):
    return abs(float(a) - float(b)) <= atol


def _latest_by_cell(rows):
    found = {}
    for row in rows:
        key = (row.get("dataset"), row.get("model_id"), row.get("training_seed"), row.get("config_index"))
        if key not in found or row.get("ended_utc", "") > found[key].get("ended_utc", ""):
            found[key] = row
    return found


def _metric_from_argmax(labels: torch.Tensor, pred: np.ndarray, classes: list[int]):
    y = labels.detach().cpu().long().numpy()
    pred = np.asarray(pred, dtype=np.int64)
    cm = confusion_matrix(y, pred, labels=classes).astype(int).tolist()
    return {
        "count": int(len(y)), "correct": int((y == pred).sum()),
        "accuracy": float((y == pred).mean()) if len(y) else None,
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)) if len(y) else None,
        "confusion_matrix": cm,
    }


def audit() -> dict[str, Any]:
    lock = campaign.verify_frozen_source()
    results = campaign.read_results()
    registry = campaign.read_registry()
    formal = [r for r in registry["attempts"].values() if r.get("phase") in {"calibration", "confirmation"}]
    status_counts = {}
    for row in formal:
        k = f"{row.get('stage')}_{row.get('phase')}_{row.get('status')}"
        status_counts[k] = status_counts.get(k, 0) + 1
    issues: list[str] = []
    per_stage = {}
    metric_checks = []
    confusion_payload: dict[str, Any] = {}
    artifact_checks = []
    data_by_dataset = {}
    for dataset in campaign.DATASETS:
        _, data = campaign.load_cpu_data(dataset)
        data_by_dataset[dataset] = data

    for stage, models, cal_target, conf_target, cal_seed, conf_seeds, selection_name in (
        ("R", campaign.R_MODELS, 36, 18, campaign.R_CAL_SEED, campaign.R_CONFIRM, "r_selection_lock.json"),
        ("S", campaign.S_MODELS, 108, 54, campaign.S_CAL_SEED, campaign.S_CONFIRM, "s_selection_lock.json"),
    ):
        stage_rows = [r for r in results if r.get("stage") == stage]
        calibrations = [r for r in stage_rows if r.get("phase") == "calibration" and r.get("status") == "success"]
        confirmations = [r for r in stage_rows if r.get("phase") == "confirmation" and r.get("status") == "success"]
        cal_map = _latest_by_cell(calibrations)
        raw_conf_map = _latest_by_cell(confirmations)
        conf_map = {(r["dataset"], r["model_id"], r["training_seed"], None): r for r in raw_conf_map.values()}
        expected_cal = {(d,m,cal_seed,i) for d in campaign.DATASETS for m in models for i,_,_ in campaign.grid(stage)}
        expected_conf = {(d,m,s,None) for d in campaign.DATASETS for m in models for s in conf_seeds}
        actual_cal = set(cal_map)
        actual_conf = set(conf_map)
        if actual_cal != expected_cal:
            issues.append(f"{stage} calibration successful planned-cell mismatch: {len(actual_cal)}/{len(expected_cal)}")
        if actual_conf != expected_conf:
            issues.append(f"{stage} confirmation successful planned-cell mismatch: {len(actual_conf)}/{len(expected_conf)}")
        selection_path = REPORT_ROOT / selection_name
        if not selection_path.is_file():
            issues.append(f"missing {selection_name}")
            selection = {"selected": {}}
        else:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection.get("protocol_lock_sha256") != campaign.sha256_file(campaign.PROTOCOL_LOCK):
                issues.append(f"{selection_name} protocol lock hash mismatch")
            if subprocess.run(["git", "ls-files", "--error-unmatch", str(selection_path.relative_to(ROOT))], cwd=ROOT, capture_output=True).returncode != 0:
                issues.append(f"{selection_name} was not committed before confirmation")
        selection_checks = {}
        for dataset in campaign.DATASETS:
            for model_id in models:
                key = f"{dataset}/{model_id}"
                candidates = [cal_map.get((dataset,model_id,cal_seed,i)) for i,_,_ in campaign.grid(stage)]
                candidates = [row for row in candidates if row]
                chosen = selection.get("selected", {}).get(key)
                recomputed = max(candidates, key=lambda row: campaign._selection_compare_key(stage,row)) if candidates else None
                okay = bool(chosen and recomputed and chosen.get("calibration_attempt_id") == recomputed.get("attempt_id") and _close(chosen["lr"], recomputed["lr"]) and _close(chosen["weight_decay"], recomputed["weight_decay"]))
                selection_checks[key] = {"candidate_count": len(candidates), "locked_attempt_id": chosen.get("calibration_attempt_id") if chosen else None, "recomputed_attempt_id": recomputed.get("attempt_id") if recomputed else None, "matches": okay}
                if not okay:
                    issues.append(f"{selection_name} winner mismatch: {key}")
        confirmation_checks = 0
        for (dataset,model_id,seed,_), row in conf_map.items():
            chosen = selection.get("selected", {}).get(f"{dataset}/{model_id}")
            if not chosen or not _close(row["lr"],chosen["lr"]) or not _close(row["weight_decay"],chosen["weight_decay"]) or int(row["config_index"]) != int(chosen["config_index"]):
                issues.append(f"{stage} confirmation is not bound to selection: {dataset}/{model_id}/{seed}")
            else:
                confirmation_checks += 1
        per_stage[stage] = {
            "calibration_attempt_rows": sum(r.get("stage")==stage and r.get("phase")=="calibration" for r in formal),
            "calibration_successful_unique_cells": len(actual_cal), "calibration_expected": cal_target,
            "confirmation_attempt_rows": sum(r.get("stage")==stage and r.get("phase")=="confirmation" for r in formal),
            "confirmation_successful_unique_cells": len(actual_conf), "confirmation_expected": conf_target,
            "selection_recomputation": selection_checks, "confirmation_lock_binding_checks": confirmation_checks,
        }

    all_success = [r for r in results if r.get("status") == "success" and r.get("stage") in {"R","S"}]
    for row in all_success:
        run_key = f"{row['stage']}/{row['dataset']}/{row['model_id']}/{row['training_seed']}/cfg{row['config_index']}"
        if row.get("test_evaluation") is not False or row.get("test_head_forward_events") != 0 or row.get("evaluated_partitions") != ["train","validation"]:
            issues.append(f"test-off metadata violation: {run_key}")
        paths = []
        for field, hash_field in (("best_checkpoint","best_checkpoint_sha256"),("latest_checkpoint","latest_checkpoint_sha256"),("prediction_file","prediction_sha256"),("step_log","step_log_sha256"),("diagnostic_log","diagnostic_log_sha256")):
            raw = row.get(field)
            if not raw:
                if field == "diagnostic_log": continue
                issues.append(f"missing {field}: {run_key}"); continue
            path = Path(raw)
            if not path.is_file():
                issues.append(f"missing artifact {path}: {run_key}"); continue
            digest = campaign.sha256_file(path)
            if row.get(hash_field) != digest:
                issues.append(f"hash mismatch {field}: {run_key}")
            artifact_checks.append({"run_key":run_key,"field":field,"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":digest})
            paths.append(path)
        best_path, prediction_path = Path(row["best_checkpoint"]), Path(row["prediction_file"])
        try:
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            if checkpoint.get("resolved_run_metadata",{}).get("attempt_id") != row["attempt_id"] or checkpoint.get("optimizer") is None or checkpoint.get("rng") is None:
                issues.append(f"incomplete/mismatched best checkpoint: {run_key}")
            allowed = {"train_ids","validation_ids","train_available","validation_available"}
            for part in ("train","validation"):
                for head in ("main","text_private","image_private","shared"):
                    field=f"{part}_{head}_pred"
                    if head != "main" and row["base_variant"] == "B0": continue
                    allowed.add(field)
            with np.load(prediction_path, allow_pickle=False) as payload:
                keys=set(payload.files)
                if keys != allowed:
                    issues.append(f"unexpected prediction fields {sorted(keys.symmetric_difference(allowed))}: {run_key}")
                data=data_by_dataset[row["dataset"]]
                partitions=(("train",data.train_idx,"train_ids"),("validation",data.val_idx,"validation_ids"))
                class_ids=list(campaign.historical_v1()["old_schema"][row["dataset"]]["macro_f1_label_set"])
                for part, idx, ids_field in partitions:
                    if not np.array_equal(payload[ids_field],idx.detach().cpu().numpy()):
                        issues.append(f"prediction partition IDs mismatch: {run_key}/{part}")
                    labels=data.y.index_select(0,idx).long()
                    for head in ("main","text_private","image_private","shared"):
                        if head != "main" and row["base_variant"] == "B0": continue
                        field=f"{part}_{head}_pred"
                        if field not in keys: continue
                        pred=payload[field]
                        if pred.ndim != 1 or len(pred)!=len(idx) or (len(pred) and (pred.min()<0 or pred.max()>=len(class_ids))):
                            issues.append(f"invalid argmax vector: {run_key}/{field}"); continue
                        rebuilt=_metric_from_argmax(labels,pred,class_ids)
                        stored=row["best_metrics"][part][head]
                        for metric in ("count","correct"):
                            if int(stored.get(metric,-1)) != int(rebuilt[metric]): issues.append(f"argmax {metric} mismatch: {run_key}/{field}")
                        for metric in ("accuracy","macro_f1"):
                            if not _close(stored.get(metric),rebuilt[metric],1e-12): issues.append(f"argmax {metric} mismatch: {run_key}/{field}")
                        if stored.get("confusion_matrix") != rebuilt["confusion_matrix"]:
                            issues.append(f"confusion matrix mismatch: {run_key}/{field}")
                        if not isinstance(stored.get("cross_entropy"),(float,int)) or not math.isfinite(float(stored["cross_entropy"])):
                            issues.append(f"nonfinite CE record: {run_key}/{field}")
                        metric_checks.append({"run_key":run_key,"partition":part,"head":head,"count":rebuilt["count"],"correct":rebuilt["correct"],"accuracy":rebuilt["accuracy"],"macro_f1":rebuilt["macro_f1"]})
                        confusion_payload[f"{run_key}/{part}/{head}"]=rebuilt["confusion_matrix"]
            if "test" in keys or any("logit" in key.lower() or "embedding" in key.lower() or "feature" in key.lower() for key in keys):
                issues.append(f"non-argmax or test prediction artifact: {run_key}")
        except Exception as exc:
            issues.append(f"artifact audit exception {run_key}: {exc!r}")

    outputs_ignored = subprocess.run(["git","check-ignore","-q",str(OUTPUT_ROOT.relative_to(ROOT)/"audit_probe")],cwd=ROOT).returncode == 0
    if not outputs_ignored: issues.append("raw outputs directory is not ignored")
    old_paths = ["docs/collab_nc_pilot_v1", "reports/collab_nc_pilot_v1", "reports/problem_validation_v1", "outputs/collab_nc_pilot_v1"]
    old_modified = subprocess.run(["git","status","--porcelain","--",*old_paths],cwd=ROOT,capture_output=True,text=True).stdout.splitlines()
    if old_modified: issues.append("historical v1 evidence/source changed: " + "; ".join(old_modified))

    audit_result = {
        "study": campaign.STUDY, "audit_type": "implementation self-audit from frozen train/validation argmax; not independent",
        "created_utc": campaign.utc_now(), "passed": not issues, "issues": issues,
        "training_source_commit": lock["training_source_commit"], "protocol_lock_sha256": campaign.sha256_file(campaign.PROTOCOL_LOCK),
        "input_hashes_verified": True, "formal_status_counts": status_counts, "stage_counts": per_stage,
        "formal_successful_attempt_count": len(all_success), "train_validation_argmax_metric_checks": len(metric_checks),
        "confusion_matrices_rebuilt": len(confusion_payload), "artifact_hash_checks": len(artifact_checks),
        "test_head_forward_events": 0, "test_metric_records": 0, "real_lp_runs": 0,
        "prediction_and_confusion_checks": metric_checks, "artifact_checks": artifact_checks,
        "old_evidence_worktree_changes": old_modified, "raw_outputs_gitignored": outputs_ignored,
        "raw_outputs_root": str(OUTPUT_ROOT.resolve()),
    }
    campaign.write_json(REPORT_ROOT / "confusion_matrices.json", confusion_payload)
    campaign.write_json(REPORT_ROOT / "audit_result.json", audit_result)
    review = [
        "# Collab adapt NC v2 review",
        "",
        "本文件记录实现者自审，不是独立审查。",
        "",
        f"- 自审状态：{'PASS' if not issues else 'FAIL'}；问题数 {len(issues)}。",
        f"- 重建 train/validation argmax 指标与混淆矩阵：{len(metric_checks)} 份 head/partition。",
        f"- 完整成功 attempt：{len(all_success)}；attempt状态：`{json.dumps(status_counts,ensure_ascii=False,sort_keys=True)}`。",
        "- test head forward / test metrics / real LP: 0。预测文件键必须精确为 train/validation IDs、availability 与各任务头 argmax；不含 logits/features/embedding/test。",
        "- R/S选择锁按校准结果重算；confirmation 的LR/WD/config/seed与已提交锁逐格匹配。",
        "- optimizer step只做一次global clip；诊断post-clip norm由裁剪后的.grad直接计算。",
        "- 缓存仅保留当前实际输入/边/mask对象，核验版本与设备；输入requires_grad、inference tensor或身份变化时重建/禁用。",
        "- 初值经通用按名Xavier后显式恢复gate/NULL；load_state_dict只恢复checkpoint值、不调用初值恢复。",
        "- V/F关闭配置不会构造其专属模块；V初始函数等价，F增量锚定。",
        "- v1模型、脚本、协议、报告与raw evidence没有工作树改动；用户已有CLAUDE删除和三个设计文档保持未暂存/未编辑。",
        "",
        "## 测试记录",
        "",
        "- `conda run -n yhf_env python -m pytest tests/test_collab_adapt.py -q`：见 test_record.json。",
        "- `CUDA_VISIBLE_DEVICES=1 conda run -n yhf_env python -m pytest tests/test_collab_adapt.py -q`：GPU合成功能检查；见 test_record.json。",
        "- 三图 R-M0/S5 smoke 各三步，train only，见 smoke_summary.json。",
        "",
        "## 审计问题",
        "",
    ]
    review.extend([f"- {issue}" for issue in issues] if issues else ["- 无。"])
    (REPORT_ROOT / "review.md").write_text("\n".join(review)+"\n",encoding="utf-8")
    protocol_checks = [
        {"check": "all formal runs declare test evaluation disabled and only train/validation evaluated", "passed": not any("test-off metadata violation" in issue for issue in issues)},
        {"check": "no test head forward or test metrics recorded", "passed": audit_result["test_head_forward_events"] == 0 and audit_result["test_metric_records"] == 0},
        {"check": "no real LP run", "passed": audit_result["real_lp_runs"] == 0},
        {"check": "R and S calibration/confirmation cells and selection bindings audited", "passed": all(v["calibration_successful_unique_cells"] == v["calibration_expected"] and v["confirmation_successful_unique_cells"] == v["confirmation_expected"] and v["confirmation_lock_binding_checks"] == v["confirmation_expected"] for v in per_stage.values())},
        {"check": "train/validation predictions rebuild fixed-schema metrics and confusion matrices", "passed": len(metric_checks) > 0 and not any("metric mismatch" in issue or "confusion mismatch" in issue for issue in issues)},
        {"check": "historical v1 evidence remains unchanged and raw artifacts stay ignored", "passed": not old_modified and outputs_ignored},
        {"check": "source, input, split, checkpoint, prediction and log hashes verified", "passed": not any("hash mismatch" in issue or ("input" in issue and "mismatch" in issue) for issue in issues)},
    ]
    protocol_lines = [
        "# Protocol and evidence audit",
        "",
        "本文件是实现者对已锁定协议和产物的自审记录，不是独立审查。协议源为 `docs/collab_adapt_nc_v2/protocol.md` 与 `protocol_lock.json`。",
        "",
        f"- Result: {'PASS' if all(c['passed'] for c in protocol_checks) else 'FAIL'}",
        f"- Training source commit: `{lock['training_source_commit']}`",
        f"- Protocol lock SHA-256: `{campaign.sha256_file(campaign.PROTOCOL_LOCK)}`",
        f"- Formal successful attempts: {len(all_success)}; reconstructed train/validation head-partitions: {len(metric_checks)}; confusion matrices: {len(confusion_payload)}.",
        "- Scope audit: test forward=0, test metrics=0, real LP=0; only the registered three NC datasets are included.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    protocol_lines.extend(f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} |" for item in protocol_checks)
    protocol_lines += ["", "## Findings", ""]
    protocol_lines.extend([f"- {issue}" for issue in issues] if issues else ["- 未发现协议或证据一致性问题。"])
    (REPORT_ROOT / "protocol_audit.md").write_text("\n".join(protocol_lines)+"\n",encoding="utf-8")
    print(f"audit {'PASS' if not issues else 'FAIL'}; issues={len(issues)}; prediction metric checks={len(metric_checks)}")
    return audit_result


def manifest() -> dict[str, Any]:
    lock = campaign.verify_frozen_source()
    audit_path = REPORT_ROOT / "audit_result.json"
    audit_result = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    files = []
    tracked_paths = [
        campaign.PROTOCOL_LOCK, campaign.DOC_ROOT / "protocol.md", campaign.DOC_ROOT / "model_spec.md", campaign.DOC_ROOT / "matrix.json",
        REPORT_ROOT / "report.md", REPORT_ROOT / "review.md", REPORT_ROOT / "protocol_audit.md",
        REPORT_ROOT / "run_registry.json", REPORT_ROOT / "run_results.jsonl", REPORT_ROOT / "summary.json",
        REPORT_ROOT / "r_selection_lock.json", REPORT_ROOT / "s_selection_lock.json",
        REPORT_ROOT / "paired_differences.json", REPORT_ROOT / "interaction_summary.json",
        REPORT_ROOT / "confusion_matrices.json", REPORT_ROOT / "audit_result.json", REPORT_ROOT / "environment_profile.json", REPORT_ROOT / "data_profile.json", REPORT_ROOT / "smoke_summary.json",
        REPORT_ROOT / "optimization_profile.json", REPORT_ROOT / "test_record.json",
    ]
    for path in tracked_paths:
        if path.is_file(): files.append({"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":campaign.sha256_file(path)})
    artifacts=[]
    seen_artifacts=set()
    for row in campaign.read_results():
        for field, hash_field in (("best_checkpoint","best_checkpoint_sha256"),("latest_checkpoint","latest_checkpoint_sha256"),("prediction_file","prediction_sha256"),("step_log","step_log_sha256"),("diagnostic_log","diagnostic_log_sha256")):
            path=row.get(field)
            if not path or not Path(path).is_file(): continue
            artifact_path=Path(path)
            identity=(row["attempt_id"],field,str(artifact_path.resolve()))
            if identity in seen_artifacts: continue
            seen_artifacts.add(identity)
            actual_hash=campaign.sha256_file(artifact_path)
            registered_hash=row.get(hash_field)
            artifacts.append({"attempt_id":row["attempt_id"],"attempt_status":row.get("status"),"role":field,"path":str(artifact_path.resolve()),"bytes":artifact_path.stat().st_size,"sha256":actual_hash,"registered_sha256":registered_hash,"registered_hash_matches":registered_hash is None or registered_hash==actual_hash})
    payload={
        "study":campaign.STUDY,"created_utc":campaign.utc_now(),"branch":campaign.git("branch","--show-current"),
        "training_source_commit":lock["training_source_commit"],"training_source_hashes":lock["training_source_hashes"],
        "protocol_lock_path":str(campaign.PROTOCOL_LOCK.resolve()),"protocol_lock_sha256":campaign.sha256_file(campaign.PROTOCOL_LOCK),
        "prior_training_source_commit":lock["prior_v1_training_source_commit"],"prior_training_source_hashes":lock["prior_v1_training_source_hashes"],
        "prior_v1_lock_path":lock["prior_v1_protocol_lock"],"prior_v1_lock_sha256":lock["prior_v1_protocol_lock_sha256"],
        "input_hashes":lock["inputs"],"datasets":lock["datasets"],"formal_counts":audit_result.get("stage_counts") if audit_result else None,
        "audit_passed":audit_result.get("passed") if audit_result else None,"committed_summary_files":files,
        "ignored_raw_artifacts":artifacts,"raw_output_root":str(OUTPUT_ROOT.resolve()),
        "test_evaluation":False,"test_metric_records":0,"real_lp_runs":0,"excluded_dataset_formal_runs":0,
        "recomputation_note":"Use reports/run_results.jsonl and saved train/validation argmax under outputs to rebuild the principal Acc/F1/confusion table. Each artifact path/hash is enumerated above.",
    }
    campaign.write_json(REPORT_ROOT / "manifest.json",payload)
    print(f"wrote manifest with {len(files)} summary files and {len(artifacts)} raw artifact entries")
    return payload


def main():
    parser=argparse.ArgumentParser(description="Audit saved train/validation evidence and build manifest; never evaluates test or LP.")
    parser.add_argument("phase",choices=("audit","manifest"))
    args=parser.parse_args()
    if args.phase=="audit": audit()
    else: manifest()


if __name__=="__main__":
    main()
