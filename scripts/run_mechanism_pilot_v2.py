from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from importlib.metadata import PackageNotFoundError, version
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.tasks.common import build_optimizer
from src.utils.seeds import set_seed


DATASETS = ("Movies", "Grocery")
VARIANTS = tuple(f"P{i}" for i in range(7))
CALIBRATION_LRS = (0.0003, 0.001, 0.003)
SEEDS = (42, 43, 44)
LOCK_PATH = ROOT / "reports/problem_validation_v1/protocol_lock.json"
OUTPUT_ROOT = ROOT / "outputs/mechanism_pilot_v2"
REPORT_ROOT = ROOT / "reports/mechanism_pilot_v2"
BASE_COMMIT = "d1518bfe0087df35f399d8d6679b479b520977c2"
SOURCE_FILES = (
    "src/models/mag_collaborative.py",
    "src/models/__init__.py",
    "src/tasks/nc.py",
    "src/tasks/common.py",
    "src/data/loaders.py",
    "src/data/__init__.py",
    "src/data/graph_utils.py",
    "src/data/types.py",
    "src/utils/seeds.py",
    "configs/config.yaml",
    "configs/task/nc.yaml",
    "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml",
    "configs/model/mag_collaborative.yaml",
    "scripts/run_mechanism_pilot_v2.py",
    "docs/mechanism_pilot_v2/protocol.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_environment(device: str) -> dict[str, Any]:
    packages = {}
    for name in ("dgl", "torch-geometric", "numpy", "scikit-learn", "hydra-core", "omegaconf"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    device_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.device(device))
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda, "packages": packages,
        "device": device, "device_name": device_name,
    }


def locked_dataset_schema(dataset: str) -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return lock["schemas"][dataset]


def verify_inputs(dataset: str) -> dict[str, dict[str, Any]]:
    schema = locked_dataset_schema(dataset)
    actual: dict[str, dict[str, Any]] = {}
    for path_text, expected in schema["dataset_input_hashes"].items():
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"locked input is missing: {path}")
        item = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if item != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}; expected={expected}; actual={item}")
        actual[str(path)] = item
    split_path = next(Path(p) for p in actual if "/MAGB_split/" in p)
    expected_split = schema["official_split_sha256"]
    if actual[str(split_path)]["sha256"] != expected_split:
        raise RuntimeError(f"the locked seed42 split hash does not match for {dataset}")
    return actual


def source_manifest() -> dict[str, Any]:
    files = {name: {"bytes": (ROOT / name).stat().st_size, "sha256": sha256_file(ROOT / name)} for name in SOURCE_FILES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "baseline_commit": BASE_COMMIT, "source_commit": source_commit,
        "source_sha256": combined, "source_files": files,
    }


def compose_config(dataset: str, variant: str, seed: int, lr: float, device: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=mag_collaborative", f"seed={seed}", f"device={device}"
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.model.variant = variant
    cfg.model.dropout = 0.2
    cfg.task.lr = float(lr)
    cfg.task.weight_decay = 1e-4
    cfg.task.optimizer = "adamw"
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 0.0
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.scheduler = None
    cfg.task.evaluate_test = False
    cfg.task.eval_labels = list(locked_dataset_schema(dataset)["macro_f1_label_set"])
    cfg.task.source_aux_weight = 0.1 if variant in {"P5", "P6"} else 0.0
    cfg.task.loss.aux_weight = 0.0
    cfg.dataset.nc_split_path = str(Path("/hdd1/DataInHere/YHF/data/MAGB_split") / f"{dataset}_nc_seed42_train0.6_val0.2.pt")
    cfg.dataset.auto_generate_splits = False
    cfg.task.evaluate_test = False
    cfg.task.eval_labels = list(locked_dataset_schema(dataset)["macro_f1_label_set"])
    return cfg


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def confusion_and_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, macro_labels: list[int]) -> dict[str, Any]:
    prediction = logits.argmax(dim=-1)
    flat = targets * num_classes + prediction
    confusion = torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    diagonal = confusion.diag().to(torch.float64)
    denominator = 2 * diagonal + confusion.sum(dim=0) - diagonal + confusion.sum(dim=1) - diagonal
    f1_by_class = torch.where(denominator > 0, 2 * diagonal / denominator, torch.zeros_like(denominator))
    f1 = f1_by_class[torch.as_tensor(macro_labels, dtype=torch.long, device=logits.device)].mean()
    return {
        "n": int(targets.numel()),
        "accuracy": float((prediction == targets).float().mean()),
        "macro_f1": float(f1),
        "cross_entropy": float(nn.functional.cross_entropy(logits, targets)),
        "confusion_matrix": confusion.detach().cpu().tolist(),
        "class_order": list(range(num_classes)),
        "macro_f1_label_set": macro_labels,
        "predictions": prediction.detach().cpu(),
    }


def optimizer_groups(
    model: nn.Module, classifier: nn.Module, source_heads: nn.Module | None
) -> dict[str, list[nn.Parameter]]:
    named = dict(model.named_parameters())
    prefixes = {
        "input_projection": ("input_text.", "input_image."),
        "layer1_experts": ("layers.0.shared.", "layers.0.specialists.", "layers.0.wide."),
        "layer1_router": ("layers.0.routers.",),
        "layer2_experts": ("layers.1.shared.", "layers.1.specialists.", "layers.1.wide."),
        "layer2_router": ("layers.1.routers.",),
        "interaction_and_type": ("interaction.", "type_embedding"),
        "fusion": ("fusion.",),
        "update_normalization": ("update_norms.",),
    }
    result = {name: [parameter for key, parameter in named.items() if key.startswith(starts)]
              for name, starts in prefixes.items()}
    result["main_head"] = list(classifier.parameters())
    result["source_heads"] = list(source_heads.parameters()) if source_heads is not None else []
    return result


def group_step_stats(groups: dict[str, list[nn.Parameter]], before: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    result = {}
    for name, parameters in groups.items():
        if not parameters:
            result[name] = None
            continue
        param_sq = parameters[0].new_zeros((), dtype=torch.float32)
        grad_sq = parameters[0].new_zeros((), dtype=torch.float32)
        update_sq = parameters[0].new_zeros((), dtype=torch.float32)
        for parameter, previous in zip(parameters, before[name]):
            param_sq = param_sq + parameter.detach().float().square().sum()
            if parameter.grad is not None:
                grad_sq = grad_sq + parameter.grad.detach().float().square().sum()
            update_sq = update_sq + (parameter.detach().float() - previous.float()).square().sum()
        param_norm = torch.sqrt(param_sq)
        grad_norm = torch.sqrt(grad_sq)
        update_norm = torch.sqrt(update_sq)
        result[name] = {
            "parameter_norm": param_norm.detach(),
            "gradient_norm_after_clip": grad_norm.detach(),
            "update_norm": update_norm.detach(),
            "relative_update": (update_norm / param_norm.clamp_min(1e-12)).detach(),
        }
    return result


def evaluate_train_validation(
    model: nn.Module,
    classifier: nn.Module,
    source_heads: nn.Module | None,
    x: torch.Tensor,
    edges: torch.Tensor,
    available: torch.Tensor | None,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    macro_labels: list[int],
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    model.eval()
    classifier.eval()
    if source_heads is not None:
        source_heads.eval()
    with torch.no_grad():
        z, _, _, _, extras = model(x, edges, available)
        train_logits = classifier(z[train_idx])
        val_logits = classifier(z[val_idx])
        train_metrics = confusion_and_metrics(train_logits, train_y, num_classes, macro_labels)
        val_metrics = confusion_and_metrics(val_logits, val_y, num_classes, macro_labels)
        source_ce = 0.0
        source_acc: dict[str, float] = {}
        if source_heads is not None:
            losses = []
            for modality in ("text", "image"):
                logits = source_heads[modality](extras["source_states"][modality][train_idx])
                losses.append(nn.functional.cross_entropy(logits, train_y))
                source_acc[modality] = float((logits.argmax(dim=-1) == train_y).float().mean())
            source_ce = float(torch.stack(losses).mean())
        return (
            train_metrics, val_metrics, train_metrics.pop("predictions"),
            val_metrics.pop("predictions"), source_ce,
            {"source_train_accuracy": source_acc, "model": json_ready(extras.get("diagnostics", {}))},
        )


def run_one(
    dataset: str,
    variant: str,
    seed: int,
    lr: float,
    phase: str,
    device_text: str,
    data,
    input_hashes: dict[str, dict[str, Any]],
    calibration_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset not in DATASETS or variant not in VARIANTS or phase not in {"calibration", "confirmation", "micro"}:
        raise ValueError("run_one received a dataset, scheme, or phase outside the registered pilot")
    run_key = f"{phase}-{dataset}-{variant}-s{seed}-lr{lr:.7g}"
    prior = sorted((OUTPUT_ROOT / dataset / phase / variant).glob(f"{run_key}*"))
    expected_source = source_manifest()
    for directory in reversed(prior):
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "succeeded":
                integrity_ok = (
                    summary.get("source", {}).get("source_commit") == expected_source["source_commit"]
                    and summary.get("source", {}).get("source_sha256") == expected_source["source_sha256"]
                    and summary.get("input_hashes") == input_hashes
                )
                for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                                           ("prediction_path", "prediction_sha256"),
                                           ("epoch_log_path", "epoch_log_sha256")):
                    artifact_path = Path(summary.get(path_key, ""))
                    integrity_ok = integrity_ok and artifact_path.is_file() and sha256_file(artifact_path) == summary.get(hash_key)
                if integrity_ok:
                    return summary
                summary["status"] = "superseded"
                summary["superseded_reason"] = "source, locked inputs, or raw artifact hash changed before resume"
                write_json(summary_path, summary)
                attempt_path_previous = directory / "attempt.json"
                if attempt_path_previous.is_file():
                    previous_attempt = json.loads(attempt_path_previous.read_text(encoding="utf-8"))
                    previous_attempt["status"] = "superseded"
                    previous_attempt["superseded_reason"] = summary["superseded_reason"]
                    write_json(attempt_path_previous, previous_attempt)
    attempt_number = len(prior) + 1
    run_id = run_key if attempt_number == 1 else f"{run_key}-attempt{attempt_number:02d}"
    run_dir = OUTPUT_ROOT / dataset / phase / variant / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = run_dir / "attempt.json"
    event_path = run_dir / "epochs.jsonl"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    prediction_path = run_dir / "train_val_predictions.pt"
    summary_path = run_dir / "summary.json"
    cfg = compose_config(dataset, variant, seed, lr, device_text)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase, dataset, device_text]
    attempt = {
        "run_id": run_id, "cell_id": run_key, "phase": phase, "dataset": dataset,
        "variant": variant, "seed": seed, "lr": lr, "status": "running",
        "started_utc": utc_now(), "command": command, "device": device_text,
        "source": expected_source, "input_hashes": input_hashes,
        "environment": runtime_environment(device_text),
        "test_evaluation": False,
        "calibration_reference": calibration_reference,
    }
    write_json(attempt_path, attempt)
    start = time.perf_counter()
    peak_memory_bytes = 0
    try:
        set_seed(seed)
        if device_text.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device_text}")
        if device_text.startswith("cuda"):
            device = torch.device(device_text)
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
        data_info = {
            "input_dim": data.input_dim, "text_dim": int(data.x_t.shape[1]),
            "visual_dim": int(data.x_i.shape[1]), "num_classes": int(data.num_classes),
            "num_nodes": data.num_nodes,
        }
        model = build_model(cfg, data_info).to(device)
        classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
        aux_weight = float(cfg.task.source_aux_weight)
        source_heads = None
        if aux_weight > 0:
            source_heads = nn.ModuleDict({
                "text": nn.Linear(model.out_dim, int(data.num_classes)),
                "image": nn.Linear(model.out_dim, int(data.num_classes)),
            }).to(device)
        modules = [model, classifier] + ([source_heads] if source_heads is not None else [])
        parameters = [parameter for module in modules for parameter in module.parameters()]
        optimizer = build_optimizer(parameters, cfg, model=model)
        x = data.x.to(device)
        edges = data.edge_index.to(device)
        train_idx = data.train_idx.to(device)
        val_idx = data.val_idx.to(device)
        # Only train and validation labels are read by this runner; test labels are never indexed.
        train_y = data.y.index_select(0, data.train_idx).to(device)
        val_y = data.y.index_select(0, data.val_idx).to(device)
        raw_available = data.info.get("modality_available")
        available = None if raw_available is None else torch.as_tensor(raw_available, dtype=torch.bool, device=device)
        macro_labels = list(cfg.task.eval_labels)
        groups = optimizer_groups(model, classifier, source_heads)
        best_metrics: tuple[float, float] | None = None
        best_epoch = 0
        patience_left = int(cfg.task.patience)
        best_states: dict[str, Any] | None = None
        best_event: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        event_path.write_text("", encoding="utf-8")
        for epoch in range(1, int(cfg.task.epochs) + 1):
            model.train()
            classifier.train()
            if source_heads is not None:
                source_heads.train()
            if hasattr(model, "set_epoch"):
                model.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            before = {name: [parameter.detach().clone() for parameter in values] for name, values in groups.items()}
            z, _, _, model_aux_loss, extras = model(x, edges, available)
            train_logits = classifier(z[train_idx])
            main_train_ce_pre = nn.functional.cross_entropy(train_logits, train_y)
            source_train_ce_pre = z.new_zeros(())
            if source_heads is not None:
                source_losses = []
                source_states = extras["source_states"]
                source_available = extras.get("source_available")
                for modality_index, modality in enumerate(("text", "image")):
                    selected_idx = train_idx
                    selected_y = train_y
                    if source_available is not None:
                        keep = source_available[train_idx, modality_index]
                        selected_idx, selected_y = train_idx[keep], train_y[keep]
                    if selected_idx.numel():
                        source_logits = source_heads[modality](source_states[modality][selected_idx])
                        source_losses.append(nn.functional.cross_entropy(source_logits, selected_y))
                if source_losses:
                    source_train_ce_pre = torch.stack(source_losses).mean()
            loss = main_train_ce_pre + aux_weight * source_train_ce_pre + float(cfg.task.loss.aux_weight) * model_aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True)
            optimizer.step()
            step_stats = group_step_stats(groups, before)

            train_metrics, val_metrics, train_predictions, val_predictions, source_ce_eval, diagnostics = evaluate_train_validation(
                model, classifier, source_heads, x, edges, available, train_idx, val_idx,
                train_y, val_y, int(data.num_classes), macro_labels,
            )
            score = (float(val_metrics["accuracy"]), float(val_metrics["macro_f1"]))
            improved = best_metrics is None or score > best_metrics
            if improved:
                best_metrics = score
                best_epoch = epoch
                patience_left = int(cfg.task.patience)
                best_states = {
                    "model_state": cpu_state_dict(model),
                    "head_state": cpu_state_dict(classifier),
                    "source_head_state": cpu_state_dict(source_heads) if source_heads is not None else None,
                    "optimizer_state": clone_to_cpu(optimizer.state_dict()),
                    "_val_predictions": val_predictions.clone(),
                }
                best_event = {
                    "epoch": epoch, "gradient_parameter_update": json_ready(step_stats),
                    "representation_diagnostics": diagnostics["model"],
                }
            elif epoch >= int(cfg.task.early_stop_min_epoch):
                patience_left -= 1

            event = {
                "epoch": epoch,
                "train_main_ce_pre_step": float(main_train_ce_pre.detach()),
                "train_aux_ce_pre_step": float(source_train_ce_pre.detach()),
                "train_main_ce": train_metrics["cross_entropy"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_main_ce": val_metrics["cross_entropy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "source_train_ce": source_ce_eval,
                "source_train_accuracy": diagnostics["source_train_accuracy"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "patience_left": patience_left,
            }
            if epoch in {1, 10, 30} or improved:
                event["gradient_parameter_update"] = json_ready(step_stats)
                event["representation_diagnostics"] = diagnostics["model"]
            history.append(event)
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            if epoch >= int(cfg.task.early_stop_min_epoch) and patience_left <= 0:
                break

        if best_states is None or best_epoch < 1:
            raise RuntimeError("training ended without a validation-selected checkpoint")
        model.load_state_dict(best_states["model_state"])
        classifier.load_state_dict(best_states["head_state"])
        if source_heads is not None:
            source_heads.load_state_dict(best_states["source_head_state"])
        model.eval()
        classifier.eval()
        if source_heads is not None:
            source_heads.eval()
        final_train, final_val, train_predictions, val_predictions, source_ce, final_diag = evaluate_train_validation(
            model, classifier, source_heads, x, edges, available, train_idx, val_idx,
            train_y, val_y, int(data.num_classes), macro_labels,
        )
        # Exact prediction reproduction check after restoring the selected checkpoint state.
        if best_event is None or not torch.equal(val_predictions, best_states["_val_predictions"]):
            raise RuntimeError("selected validation predictions failed checkpoint restoration check")
        best_states.pop("_val_predictions", None)
        elapsed = time.perf_counter() - start
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        checkpoint = {
            "protocol": "mechanism_pilot_v2_train_validation_only",
            "source_commit": source_manifest()["source_commit"],
            "baseline_commit": BASE_COMMIT,
            "source": source_manifest(),
            "run_id": run_id,
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "lr": lr,
            "best_epoch": best_epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_state": best_states["model_state"],
            "head_state": best_states["head_state"],
            "source_head_state": best_states["source_head_state"],
            "optimizer_state": best_states["optimizer_state"],
        }
        torch.save(checkpoint, checkpoint_path)
        restored_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(restored_payload["model_state"])
        classifier.load_state_dict(restored_payload["head_state"])
        if source_heads is not None:
            source_heads.load_state_dict(restored_payload["source_head_state"])
        optimizer.load_state_dict(restored_payload["optimizer_state"])
        _, restored_val, _, restored_val_predictions, _, _ = evaluate_train_validation(
            model, classifier, source_heads, x, edges, available, train_idx, val_idx,
            train_y, val_y, int(data.num_classes), macro_labels,
        )
        if not torch.equal(restored_val_predictions, val_predictions) or restored_val["confusion_matrix"] != final_val["confusion_matrix"]:
            raise RuntimeError("serialized checkpoint failed validation prediction/confusion restore check")
        torch.save({
            "train_idx": data.train_idx.cpu(), "train_argmax": train_predictions,
            "val_idx": data.val_idx.cpu(), "val_argmax": val_predictions,
            "split_sha256": locked_dataset_schema(dataset)["official_split_sha256"],
        }, prediction_path)
        param_counts = {
            "total_backbone": parameter_count(model),
            "active_backbone": model.active_backbone_parameters,
            "main_head": parameter_count(classifier),
            "source_heads": parameter_count(source_heads) if source_heads is not None else 0,
        }
        summary = {
            **attempt,
            "status": "succeeded",
            "finished_utc": utc_now(),
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "selected_by": "max_validation_accuracy_then_macro_f1; exact ties keep earlier epoch",
            "train": {key: value for key, value in final_train.items() if key != "predictions"},
            "validation": {key: value for key, value in final_val.items() if key != "predictions"},
            "source_train_cross_entropy": source_ce,
            "source_train_accuracy": final_diag["source_train_accuracy"],
            "parameter_counts": param_counts,
            "optimizer": "AdamW",
            "optimizer_lr": lr,
            "optimizer_weight_decay": 1e-4,
            "actual_device": str(device),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory_bytes,
            "best_step_diagnostics": best_event,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "epoch_log_path": str(event_path),
            "epoch_log_sha256": sha256_file(event_path),
            "full_graph_transductive_features": True,
            "train_eval_indices_only": True,
            "test_inference": False,
        }
        write_json(summary_path, summary)
        attempt.update({"status": "succeeded", "finished_utc": utc_now(), "elapsed_seconds": elapsed})
        write_json(attempt_path, attempt)
        return summary
    except Exception as exc:
        attempt.update({
            "status": "failed", "finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - start,
            "error_type": type(exc).__name__, "error": str(exc),
        })
        write_json(attempt_path, attempt)
        write_json(summary_path, attempt)
        raise


def run_worker(phase: str, dataset: str, device: str, selection_path: Path | None = None) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"this pilot is restricted to {DATASETS}; got {dataset!r}")
    input_hashes = verify_inputs(dataset)
    torch.set_num_threads(4)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        torch.cuda.set_device(torch.device(device))
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "model=mag_collaborative", f"device={device}"])
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.nc_split_path = str(Path("/hdd1/DataInHere/YHF/data/MAGB_split") / f"{dataset}_nc_seed42_train0.6_val0.2.pt")
    cfg.dataset.auto_generate_splits = False
    data = load_mag_data(cfg, "nc", 42)
    expected_split_size = locked_dataset_schema(dataset)["split_sizes"]
    observed_split_size = [int(data.train_idx.numel()), int(data.val_idx.numel()), int(data.test_idx.numel())]
    if observed_split_size != expected_split_size:
        raise RuntimeError(f"fixed split sizes mismatch: {observed_split_size} != {expected_split_size}")
    if phase == "calibration":
        cells = [(variant, 42, lr, None) for variant in VARIANTS for lr in CALIBRATION_LRS]
    elif phase == "confirmation":
        if selection_path is None:
            selection_path = REPORT_ROOT / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        cells = [(variant, seed, float(selection["datasets"][dataset][variant]["selected_lr"]),
                  {"run_id": selection["datasets"][dataset][variant]["selected_calibration_run_id"],
                   "checkpoint_path": selection["datasets"][dataset][variant]["selected_checkpoint_path"],
                   "checkpoint_sha256": selection["datasets"][dataset][variant]["selected_checkpoint_sha256"]})
                 for variant in VARIANTS for seed in (43, 44)]
    else:
        raise ValueError(f"worker phase must be calibration|confirmation, got {phase!r}")
    for variant, seed, lr, calibration_reference in cells:
        run_one(dataset, variant, seed, lr, phase, device, data, input_hashes, calibration_reference)


def nvidia_inventory() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.splitlines():
            index, name, total, used, utilization = [part.strip() for part in line.split(",", 4)]
            free = int(total) - int(used)
            gpus.append({"index": int(index), "name": name, "memory_total_mib": int(total),
                         "memory_used_mib": int(used), "memory_free_mib": free,
                         "utilization_percent": int(utilization),
                         "selected": free >= 12000 and int(utilization) < 60})
        return {"source": "nvidia-smi", "gpus": gpus}
    except Exception as exc:
        return {"source": "nvidia-smi unavailable", "error": str(exc), "gpus": []}


def launch_worker(phase: str, dataset: str, device: str, selection_path: Path | None) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase, dataset, device]
    if selection_path is not None:
        command += ["--selection-file", str(selection_path)]
    started = utc_now()
    log_path = OUTPUT_ROOT / "orchestrator" / f"{phase}-{dataset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
    return {"phase": phase, "dataset": dataset, "device": device, "command": command,
            "started_utc": started, "finished_utc": utc_now(), "returncode": completed.returncode,
            "worker_log_path": str(log_path), "worker_log_sha256": sha256_file(log_path)}


def all_summaries(dataset: str | None = None) -> list[dict[str, Any]]:
    paths = OUTPUT_ROOT.glob("*/**/summary.json") if dataset is None else (OUTPUT_ROOT / dataset).glob("**/summary.json")
    rows = []
    for path in paths:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def select_calibration_lrs() -> dict[str, Any]:
    summaries = all_summaries()
    selections: dict[str, Any] = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for variant in VARIANTS:
            candidates = [row for row in summaries if row.get("status") == "succeeded"
                          and row.get("phase") == "calibration" and row.get("dataset") == dataset
                          and row.get("variant") == variant and row.get("seed") == 42]
            by_lr: dict[float, dict[str, Any]] = {}
            for row in candidates:
                by_lr[float(row["lr"])] = row
            if set(by_lr) != set(CALIBRATION_LRS):
                missing = sorted(set(CALIBRATION_LRS) - set(by_lr))
                raise RuntimeError(f"cannot select LR for {dataset}/{variant}; missing calibration LR(s): {missing}")
            ranked = sorted(by_lr.values(), key=lambda row: (
                -float(row["validation"]["accuracy"]),
                -float(row["validation"]["macro_f1"]),
                float(row["lr"]),
            ))
            chosen = ranked[0]
            selections[dataset][variant] = {
                "selected_lr": float(chosen["lr"]),
                "selected_calibration_run_id": chosen["run_id"],
                "selected_checkpoint_path": chosen["checkpoint_path"],
                "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
                "selection_metrics": chosen["validation"],
                "ranking": [{"lr": float(row["lr"]), "accuracy": row["validation"]["accuracy"],
                             "macro_f1": row["validation"]["macro_f1"], "run_id": row["run_id"]}
                            for row in ranked],
            }
    result = {
        "protocol": "seed42 max validation accuracy, then macro-F1, then lower LR",
        "datasets": selections,
        "selected_utc": utc_now(),
    }
    write_json(REPORT_ROOT / "selection.json", result)
    return result


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    return mean, (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _selected_rows(summaries: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            reference = selection["datasets"][dataset][variant]
            seed42 = next((row for row in summaries if row.get("status") == "succeeded"
                           and row.get("run_id") == reference["selected_calibration_run_id"]), None)
            if seed42 is not None:
                rows.append(seed42)
            for seed in (43, 44):
                candidates = [row for row in summaries if row.get("status") == "succeeded"
                              and row.get("phase") == "confirmation" and row.get("dataset") == dataset
                              and row.get("variant") == variant and int(row.get("seed", -1)) == seed]
                candidates.sort(key=lambda row: row.get("finished_utc", ""))
                if candidates:
                    rows.append(candidates[-1])
    return rows


def write_report(summaries: list[dict[str, Any]], selection: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    selected = _selected_rows(summaries, selection)
    succeeded_attempts = sum(row.get("status") == "succeeded" and row.get("phase") in {"calibration", "confirmation"} for row in attempts)
    failed_attempts = sum(row.get("status") == "failed" and row.get("phase") in {"calibration", "confirmation"} for row in attempts)
    lines = [
        "# MAG mechanism pilot v2 report",
        "",
        "## Scope and evidence status",
        "",
        "- The original v1 primary matrix remains valid and unchanged.",
        "- The v1.1 ordinary-J_NODE context permutation diagnostic was ineffective because the permutation was only consumed by J_NODE_SHUFFLE; the frozen v1.1 train/validation intervention is the valid context diagnostic.",
        "- This new report covers Movies and Grocery train/validation only. No LP and no test evaluation were run.",
        "- Movies and Grocery are development graphs. The three seeds are descriptive; no significance claim is made.",
        "",
        "## Selected validation results",
        "",
        "Entries are mean ± population standard deviation across the available selected seeds (42/43/44), in percent. Seed 42 reuses its selected calibration checkpoint.",
        "",
        "| Dataset | Scheme | Validation accuracy | Validation Macro-F1 | Train–validation Acc gap | Train–validation Macro-F1 gap |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            cell = [row for row in selected if row["dataset"] == dataset and row["variant"] == variant]
            val_acc = [100 * row["validation"]["accuracy"] for row in cell]
            val_f1 = [100 * row["validation"]["macro_f1"] for row in cell]
            acc_gap = [100 * (row["train"]["accuracy"] - row["validation"]["accuracy"]) for row in cell]
            f1_gap = [100 * (row["train"]["macro_f1"] - row["validation"]["macro_f1"]) for row in cell]
            acc_m, acc_s = _mean_std(val_acc)
            f1_m, f1_s = _mean_std(val_f1)
            gap_a, _ = _mean_std(acc_gap)
            gap_f, _ = _mean_std(f1_gap)
            lines.append(f"| {dataset} | {variant} | {acc_m:.2f} ± {acc_s:.2f} | {f1_m:.2f} ± {f1_s:.2f} | {gap_a:+.2f} pp | {gap_f:+.2f} pp |")

    lines += [
        "",
        "## Seed-level train/validation records",
        "",
        "| Dataset | Scheme | Seed | LR | Train Acc | Train Macro-F1 | Train CE | Val Acc | Val Macro-F1 | Val CE | Best epoch | Seconds | Peak GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selected, key=lambda value: (value["dataset"], value["variant"], value["seed"])):
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['seed']} | {row['lr']:.4g} | "
            f"{100*row['train']['accuracy']:.2f} | {100*row['train']['macro_f1']:.2f} | {row['train']['cross_entropy']:.4f} | "
            f"{100*row['validation']['accuracy']:.2f} | {100*row['validation']['macro_f1']:.2f} | {row['validation']['cross_entropy']:.4f} | "
            f"{row['best_epoch']} | {row['elapsed_seconds']:.1f} | {row['peak_cuda_memory_bytes']/1024**3:.2f} |"
        )

    lines += [
        "",
        "## Learning-rate selection",
        "",
        "| Dataset | Scheme | Selected LR | Seed-42 val Acc | Seed-42 val Macro-F1 | Candidates (LR: Acc / Macro-F1) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            item = selection["datasets"][dataset][variant]
            candidates = ", ".join(f"{value['lr']:.4g}: {100*value['accuracy']:.2f}/{100*value['macro_f1']:.2f}" for value in item["ranking"])
            metrics = item["selection_metrics"]
            lines.append(f"| {dataset} | {variant} | {item['selected_lr']:.4g} | {100*metrics['accuracy']:.2f} | {100*metrics['macro_f1']:.2f} | {candidates} |")

    comparisons = (
        ("P2−P0: dual-axis with wide experts", "P2", "P0"),
        ("P3−P1: dual-axis with conditional experts", "P3", "P1"),
        ("P3−P2: conditional experts vs width-matched wide", "P3", "P2"),
        ("P3−P4: learned vs retrained static specialists", "P3", "P4"),
        ("P5−P3: source supervision within conditional experts", "P5", "P3"),
        ("P6−P2: source supervision within wide experts", "P6", "P2"),
    )
    selected_map = {(row["dataset"], row["variant"], int(row["seed"])): row for row in selected}
    lines += [
        "",
        "## Matched validation differences",
        "",
        "Differences are right-hand scheme minus the comparator within matched dataset/seed cells. `n/6` counts positive accuracy cells; these are descriptive paired differences.",
        "",
        "| Comparison | Movies Δ Acc / Macro-F1 (pp) | Grocery Δ Acc / Macro-F1 (pp) | All cells Δ Acc / Macro-F1 (pp) | Acc positive cells |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, right, left in comparisons:
        dataset_text = {}
        acc_values, f1_values = [], []
        positives = 0
        for dataset in DATASETS:
            da, df = [], []
            for seed in SEEDS:
                r, l = selected_map.get((dataset, right, seed)), selected_map.get((dataset, left, seed))
                if r is None or l is None:
                    continue
                a = 100 * (r["validation"]["accuracy"] - l["validation"]["accuracy"])
                f = 100 * (r["validation"]["macro_f1"] - l["validation"]["macro_f1"])
                da.append(a); df.append(f); acc_values.append(a); f1_values.append(f); positives += a > 0
            ma, _ = _mean_std(da); mf, _ = _mean_std(df)
            dataset_text[dataset] = f"{ma:+.2f} / {mf:+.2f}"
        mean_acc, _ = _mean_std(acc_values); mean_f1, _ = _mean_std(f1_values)
        lines.append(f"| {label} | {dataset_text['Movies']} | {dataset_text['Grocery']} | {mean_acc:+.2f} / {mean_f1:+.2f} | {positives}/{len(acc_values)} |")

    lines += [
        "",
        "## Cost and training diagnostics",
        "",
        "Cost averages use selected seed-42/43/44 runs over both graphs. Expert width is matched by active expert MLP parameters (1025-wide MLP vs four 256-hidden expert MLPs); this does not equalize compute.",
        "",
        "| Scheme | Active backbone parameters | NC main head | Source heads | Mean seconds / run | Mean peak CUDA GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        cell = [row for row in selected if row["variant"] == variant]
        if not cell:
            continue
        counts = cell[0]["parameter_counts"]
        sec, _ = _mean_std([float(row["elapsed_seconds"]) for row in cell])
        memory, _ = _mean_std([float(row["peak_cuda_memory_bytes"]) / 1024**3 for row in cell])
        lines.append(f"| {variant} | {counts['active_backbone']} | {counts['main_head']} | {counts['source_heads']} | {sec:.1f} | {memory:.2f} |")
    lines += [
        "",
        f"Formal attempts: {succeeded_attempts} succeeded, {failed_attempts} failed (planned maximum 70).",
        "",
        "## Findings and limits",
        "",
        "- The dual-axis effect is judged against P0/P1 using paired validation changes above. It is not inferred from attention or router differences.",
        "- Conditional experts are compared with both the active-parameter-matched wide control and a separately retrained uniform-specialist model. A positive validation difference in this small matrix is descriptive evidence for this design and split only.",
        "- P5−P3 and P6−P2 distinguish source supervision in the conditional and wide architectures. A change in both would support a general auxiliary-supervision effect within these tested controls, not a causal account of the earlier probe result.",
        "- Low route weights on a path, if observed, do not establish zero use or a modality information fraction. Text may affect relation context and the final FusionMLP even when a direct branch is small.",
        "- The study cannot identify whether any shortfall comes from overfitting, shared expert functions, competitive routing, interaction optimization, or another factor. No causal mechanism claim is made.",
        "- These results do not determine the final five-NC or LP model. No new model modules or tuning grids are added here.",
        "",
        "Raw checkpoints, epoch logs, and train/validation argmax are under ignored `outputs/mechanism_pilot_v2/`. Aggregated train/validation confusion matrices are retained in `run_results.jsonl`.",
        "",
    ]
    (REPORT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def finalize() -> None:
    summaries = [row for row in all_summaries() if row.get("status") in {"succeeded", "failed"}]
    summaries.sort(key=lambda row: (row.get("dataset", ""), row.get("variant", ""), row.get("phase", ""),
                                   int(row.get("seed", 0)), float(row.get("lr", 0.0)), row.get("run_id", "")))
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = REPORT_ROOT / "run_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    attempts = []
    for path in OUTPUT_ROOT.glob("*/**/attempt.json"):
        try:
            attempts.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    attempts.sort(key=lambda row: row.get("started_utc", ""))
    selection_path = REPORT_ROOT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    write_json(REPORT_ROOT / "run_registry.json", {
        "study": "mag_mechanism_pilot_v2", "source_commit": source_manifest()["source_commit"],
        "baseline_commit": BASE_COMMIT,
        "finalizing_head": git_head(), "planned_formal_cells": 70,
        "attempt_count": len(attempts), "attempts": attempts,
        "selected_learning_rates": selection,
        "seed42_policy": "reuse each selected seed42 calibration checkpoint as the seed42 result; seeds43/44 use the selected LR",
    })
    data_manifest = {dataset: verify_inputs(dataset) for dataset in DATASETS}
    source = source_manifest()
    if selection is not None:
        write_report(summaries, selection, attempts)
    manifest = {
        "study": "mag_mechanism_pilot_v2",
        "source_commit": source["source_commit"],
        "baseline_commit": BASE_COMMIT,
        "finalizing_head": git_head(),
        "source": source,
        "protocol_sha256": sha256_file(ROOT / "docs/mechanism_pilot_v2/protocol.md"),
        "design_document_sha256": sha256_file(ROOT / "docs/mechanism_pilot_v2_design.md"),
        "v1_protocol_lock_path": str(LOCK_PATH),
        "v1_protocol_lock_sha256": sha256_file(LOCK_PATH),
        "v1_protocol_lock_internal_sha256": json.loads(LOCK_PATH.read_text(encoding="utf-8"))["protocol_lock_sha256"],
        "fixed_inputs": data_manifest,
        "fixed_input_policy": "hashes checked against reports/problem_validation_v1/protocol_lock.json before each worker; fixed seed42 split reused for training seeds 42/43/44",
        "formal_attempts": len(attempts),
        "succeeded": sum(row.get("status") == "succeeded" for row in attempts),
        "failed": sum(row.get("status") == "failed" for row in attempts),
        "test_inference": False,
        "results_path": str(results_path), "results_sha256": sha256_file(results_path),
        "registry_path": str(REPORT_ROOT / "run_registry.json"),
        "registry_sha256": sha256_file(REPORT_ROOT / "run_registry.json"),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path) if selection_path.is_file() else None,
        "report_path": str(REPORT_ROOT / "report.md") if selection is not None else None,
        "report_sha256": sha256_file(REPORT_ROOT / "report.md") if selection is not None else None,
        "raw_evidence_root": str(OUTPUT_ROOT),
        "raw_runs": [],
    }
    for row in summaries:
        if row.get("status") == "succeeded":
            manifest["raw_runs"].append({
                "run_id": row["run_id"], "dataset": row["dataset"], "variant": row["variant"],
                "phase": row["phase"], "seed": row["seed"], "lr": row["lr"],
                "checkpoint": {"path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"]},
                "train_val_predictions": {"path": row["prediction_path"], "sha256": row["prediction_sha256"]},
                "epoch_log": {"path": row["epoch_log_path"], "sha256": row["epoch_log_sha256"]},
            })
    write_json(REPORT_ROOT / "manifest.json", manifest)


def choose_devices(inventory: dict[str, Any]) -> list[str]:
    listed = inventory.get("gpus", [])
    selected = [f"cuda:{row['index']}" for row in listed if row.get("selected")]
    if selected:
        return selected[:2]
    if listed:
        raise RuntimeError("no GPU meets the free-memory/utilization threshold; not disturbing existing GPU jobs")
    if torch.cuda.is_available():
        return ["cuda:0"]
    raise RuntimeError("no usable CUDA device is visible; formal pilot training will not fall back to CPU")


def run_orchestration(phase: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    inventory = nvidia_inventory()
    devices = choose_devices(inventory)
    orchestration: dict[str, Any] = {
        "started_utc": utc_now(), "source_commit": git_head(), "gpu_inventory": inventory,
        "selected_devices": devices, "test_inference": False,
        "fixed_input_hashes": {dataset: verify_inputs(dataset) for dataset in DATASETS},
        "commands": [],
    }
    write_json(OUTPUT_ROOT / "orchestrator" / "preflight.json", orchestration)
    if phase in {"all", "calibration"}:
        jobs = [("calibration", dataset, devices[index % len(devices)], None) for index, dataset in enumerate(DATASETS)]
        with ThreadPoolExecutor(max_workers=min(len(jobs), len(devices))) as pool:
            futures = [pool.submit(launch_worker, *job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                orchestration["commands"].append(record)
                write_json(OUTPUT_ROOT / "orchestrator" / "progress.json", orchestration)
                if record["returncode"] != 0:
                    raise RuntimeError(f"calibration worker failed for {record['dataset']}; see {record['worker_log_path']}")
        if phase == "calibration":
            select_calibration_lrs()
            finalize()
            return
    if phase in {"all", "confirmation"}:
        if phase == "all":
            select_calibration_lrs()
        selection_path = REPORT_ROOT / "selection.json"
        jobs = [("confirmation", dataset, devices[index % len(devices)], selection_path) for index, dataset in enumerate(DATASETS)]
        with ThreadPoolExecutor(max_workers=min(len(jobs), len(devices))) as pool:
            futures = [pool.submit(launch_worker, *job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                orchestration["commands"].append(record)
                write_json(OUTPUT_ROOT / "orchestrator" / "progress.json", orchestration)
                if record["returncode"] != 0:
                    raise RuntimeError(f"confirmation worker failed for {record['dataset']}; see {record['worker_log_path']}")
    orchestration["finished_utc"] = utc_now()
    write_json(OUTPUT_ROOT / "orchestrator" / "progress.json", orchestration)
    finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fixed train/validation-only MAG mechanism pilot matrix.")
    parser.add_argument("--phase", choices=("all", "calibration", "confirmation", "finalize"), default="all")
    parser.add_argument("--worker", nargs=3, metavar=("PHASE", "DATASET", "DEVICE"), help=argparse.SUPPRESS)
    parser.add_argument("--selection-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        phase, dataset, device = args.worker
        run_worker(phase, dataset, device, args.selection_file)
    elif args.phase == "finalize":
        finalize()
    else:
        run_orchestration(args.phase)


if __name__ == "__main__":
    main()
