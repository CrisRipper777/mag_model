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
from src.data.loaders import resolve_path
from src.models.mag_collab_adapt import MAGCollabAdapt, parameter_groups_no_decay
from src.models.mvcge_private_nc_v1 import MVCGEPrivateNC, sparse_top2
from src.tasks.collab_nc_pilot import CollabTaskHeads, train_only_loss

STUDY = "mvcge_private_nc_v1"
DATASETS = ("Movies", "Grocery", "Reddit-S")
MODELS = ("A0", "G0", "G1", "U1")
NEW_MODELS = ("G0", "G1", "U1")
LRS = (0.0003, 0.001, 0.003)
WDS = (0.0001, 0.001, 0.01)
CALIBRATION_SEED = 102
CONFIRMATION_SEEDS = (103, 104, 105)
SPLIT_SEED = 42
MAX_UPDATES = 800
MIN_UPDATES = 200
EVAL_FIRST = 21
EVAL_INTERVAL = 5
PATIENCE_UPDATES = 150
GRADIENT_DIAGNOSTIC_STEPS = (1, 20, 21, 100)
DOC_ROOT = ROOT / "docs" / STUDY
REPORT_ROOT = ROOT / "reports" / STUDY
OUTPUT_ROOT = ROOT / "outputs" / STUDY
MATRIX_PATH = DOC_ROOT / "matrix.json"
SELECTION_PATH = REPORT_ROOT / "selection_lock.json"
REGISTRY_PATH = REPORT_ROOT / "run_registry.json"
RESULTS_PATH = REPORT_ROOT / "run_results.jsonl"
SOURCE_FILES = (
    "src/models/mvcge_private_nc_v1.py",
    "scripts/run_mvcge_private_nc_v1.py",
    "tests/test_mvcge_private_nc_v1.py",
    "src/models/mag_collab_adapt.py",
    "src/models/mag_collab_nc.py",
    "src/models/predictor.py",
    "src/tasks/collab_nc_pilot.py",
    "src/tasks/prior_context_pilot.py",
    "src/data/__init__.py",
    "src/data/loaders.py",
    "src/data/graph_utils.py",
    "src/data/splits.py",
    "src/data/types.py",
    "configs/config.yaml",
    "configs/task/nc.yaml",
    "configs/model/mag_collab_adapt.yaml",
    "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml",
    "configs/dataset/Reddit-S.yaml",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gpu_snapshot() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except Exception as exc:
        return [{"error": repr(exc)}]
    rows = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) == 6:
            rows.append({"physical_index": int(row[0]), "uuid": row[1], "name": row[2], "memory_used_mib": int(row[3]), "memory_total_mib": int(row[4]), "utilization_percent": int(row[5])})
    return rows


def base_config(dataset: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "model=mag_collab_adapt", f"seed={SPLIT_SEED}", "device=cpu"])
    OmegaConf.resolve(cfg)
    return cfg


def inspect_raw_dgl_graph(path: Path) -> dict[str, Any]:
    try:
        import dgl

        graphs, _ = dgl.load_graphs(str(path))
        graph = graphs[0]
        src, dst = graph.edges()
        keys = (dst.long() * int(graph.num_nodes()) + src.long()).cpu()
        unique_keys = torch.unique(keys)
        reverse_keys = src.long() * int(graph.num_nodes()) + dst.long()
        unique_reverse = torch.unique(reverse_keys)
        reverse_coverage = torch.isin(unique_keys, unique_reverse).float().mean().item() if unique_keys.numel() else 1.0
        return {
            "raw_nodes": int(graph.num_nodes()),
            "raw_directed_edge_rows": int(src.numel()),
            "raw_self_loop_rows": int((src == dst).sum().item()),
            "raw_duplicate_directed_rows": int(src.numel() - unique_keys.numel()),
            "reverse_edge_coverage": float(reverse_coverage),
            "raw_edata_keys": sorted(str(key) for key in graph.edata.keys()),
            "raw_weight_fields": [str(key) for key in graph.edata.keys() if any(token in str(key).lower() for token in ("weight", "wgt"))],
            "loader_uses_edge_weights": False,
        }
    except Exception as exc:
        return {"inspection_error": repr(exc), "loader_uses_edge_weights": False}


def load_dataset(dataset: str, device: torch.device, *, profile: bool = False) -> dict[str, Any]:
    cfg = base_config(dataset)
    data = load_mag_data(cfg, "nc", SPLIT_SEED)
    # This MAGB loader exposes no availability metadata; the canonical policy is all present.
    mask = torch.ones((data.num_nodes, 2), dtype=torch.bool)
    train_idx = data.train_idx.long().contiguous()
    val_idx = data.val_idx.long().contiguous()
    labels = torch.full((data.num_nodes,), -1, dtype=torch.long)
    labels[train_idx] = data.y.index_select(0, train_idx)
    labels[val_idx] = data.y.index_select(0, val_idx)
    edge_cpu = data.edge_index.long()
    src_cpu, dst_cpu = edge_cpu
    nonloop = src_cpu != dst_cpu
    unique_keys = torch.unique(dst_cpu[nonloop] * int(data.num_nodes) + src_cpu[nonloop])
    reverse_keys = torch.unique(src_cpu[nonloop] * int(data.num_nodes) + dst_cpu[nonloop])
    canonical_graph = {
        "edge_rows": int(edge_cpu.size(1)),
        "self_loop_rows": int((src_cpu == dst_cpu).sum()),
        "duplicate_nonself_directed_rows": int(nonloop.sum() - unique_keys.numel()),
        "unique_nonself_directed_pairs": int(unique_keys.numel()),
        "reverse_edge_coverage": float(torch.isin(unique_keys, reverse_keys).float().mean()) if unique_keys.numel() else 1.0,
        "weights_exposed_by_MAGData": False,
    }
    inputs = {
        "x": data.x.float().contiguous().to(device),
        "x_text": data.x_t.float().contiguous().to(device),
        "x_image": data.x_i.float().contiguous().to(device),
        "edges": data.edge_index.long().contiguous().to(device),
        "available": mask.to(device),
        "train_idx": train_idx.to(device),
        "val_idx": val_idx.to(device),
        "labels": labels.to(device),
        "n": int(data.num_nodes),
        "m": int(data.edge_index.size(1)),
        "classes": int(data.num_classes),
        "text_dim": int(data.x_t.size(1)),
        "image_dim": int(data.x_i.size(1)),
        "train_size": int(train_idx.numel()),
        "val_size": int(val_idx.numel()),
        # Read only the split count; IDs and labels from test are never copied or used.
        "test_size": int(data.test_idx.numel()),
        "canonical_graph": canonical_graph,
        "split_hashes": {"train_ids": tensor_sha256(train_idx), "validation_ids": tensor_sha256(val_idx)},
        "input_paths": {
            "graph": str(resolve_path(cfg.dataset.graph_path)),
            "text_features": str(resolve_path(cfg.dataset.text_feat_path)),
            "image_features": str(resolve_path(cfg.dataset.image_feat_path)),
            "nc_split": str(resolve_path(cfg.dataset.nc_split_path)),
        },
        "cfg": cfg,
    }
    if profile:
        inputs["raw_graph"] = inspect_raw_dgl_graph(Path(inputs["input_paths"]["graph"]))
        inputs["feature_finiteness"] = {
            "text_all_finite": bool(torch.isfinite(data.x_t).all()),
            "image_all_finite": bool(torch.isfinite(data.x_i).all()),
            "text_all_zero_rows_diagnostic_only": int((data.x_t == 0).all(-1).sum()),
            "image_all_zero_rows_diagnostic_only": int((data.x_i == 0).all(-1).sum()),
        }
        inputs["file_hashes"] = {key: sha256_file(Path(path)) for key, path in inputs["input_paths"].items()}
    return inputs


def make_matrix() -> dict[str, Any]:
    candidates = []
    for dataset in DATASETS:
        for model in MODELS:
            history = {"Movies": (0.003, 0.001), "Grocery": (0.001, 0.001), "Reddit-S": (0.001, 0.001)}[dataset] if model == "A0" else None
            for lr in LRS:
                for wd in WDS:
                    candidates.append({
                        "phase": "calibration", "dataset": dataset, "model": model,
                        "seed": CALIBRATION_SEED, "config_index": sum(1 for row in candidates if row["dataset"] == dataset and row["model"] == model),
                        "lr": lr, "matrix_weight_decay": wd,
                        "historical_s0_selected_config": ("in_grid" if history is not None and (lr, wd) == history else None),
                    })
    confirmation = [
        {"phase": "confirmation", "dataset": dataset, "model": model, "seed": seed, "selection_key": f"{dataset}/{model}"}
        for dataset in DATASETS for model in MODELS for seed in CONFIRMATION_SEEDS
    ]
    return {
        "study": STUDY,
        "created_utc": utc_now(),
        "datasets": list(DATASETS), "models": list(MODELS),
        "lr_grid": list(LRS), "matrix_weight_decay_grid": list(WDS),
        "calibration_seed": CALIBRATION_SEED, "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "split_seed": SPLIT_SEED, "max_updates": MAX_UPDATES, "min_updates": MIN_UPDATES,
        "validation_first_step": EVAL_FIRST, "validation_interval": EVAL_INTERVAL,
        "early_stop_patience_optimizer_updates": PATIENCE_UPDATES,
        "selection_rule": "validation correct node count descending, fixed-C Macro-F1 descending, then lower config_index",
        "candidate_config_order": "dataset, model, then LR major/WDecay minor, each ascending",
        "calibration_count": len(candidates), "confirmation_count": len(confirmation),
        "historical_s0_configs": {"Movies": {"lr": 0.003, "matrix_weight_decay": 0.001}, "Grocery": {"lr": 0.001, "matrix_weight_decay": 0.001}, "Reddit-S": {"lr": 0.001, "matrix_weight_decay": 0.001}},
        "historical_s0_configs_cited_from": "reports/collab_adapt_nc_v2/s_selection_lock.json",
        "historical_s0_selection_lock_sha256": sha256_file(ROOT / "reports/collab_adapt_nc_v2/s_selection_lock.json"),
        "historical_s0_configs_all_in_current_grid": True,
        "calibration_candidates": candidates,
        "confirmation_slots": confirmation,
    }


def prepare() -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matrix = make_matrix()
    write_json(MATRIX_PATH, matrix)
    data_profiles: dict[str, Any] = {}
    for dataset in DATASETS:
        data = load_dataset(dataset, torch.device("cpu"), profile=True)
        data_profiles[dataset] = {
            "num_nodes": data["n"], "loader_edge_rows": data["m"],
            "text_dim": data["text_dim"], "image_dim": data["image_dim"],
            "num_classes": data["classes"], "train_size": data["train_size"],
            "validation_size": data["val_size"], "test_size": data["test_size"],
            "split_hashes_train_validation_only": data["split_hashes"],
            "availability": {"source": "no missing-modality metadata in MAGData; canonical protocol all modalities present", "counts": [data["n"], data["n"]], "doubly_missing_nodes": 0},
            "feature_finiteness_and_zero_row_diagnostic": data["feature_finiteness"],
            "graph": data["raw_graph"], "canonical_loader_graph": {
                "config_make_undirected": bool(data["cfg"].dataset.make_undirected),
                "config_add_self_loops": bool(data["cfg"].dataset.add_self_loops),
                **data["canonical_graph"],
            },
            "feature_files": {"graph": data["input_paths"]["graph"], "text": data["input_paths"]["text_features"], "image": data["input_paths"]["image_features"], "split": data["input_paths"]["nc_split"]},
            "input_sha256": data["file_hashes"],
            "feature_preprocessing": {
                "text": Path(data["input_paths"]["text_features"]).name,
                "image": Path(data["input_paths"]["image_features"]).name,
                "loader_cast": "float32", "additional_normalization": "none",
            },
        }
    write_json(REPORT_ROOT / "data_profile.json", data_profiles)
    profile = {
        "created_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_geometric": package_version("torch-geometric"), "dgl": package_version("dgl"),
        "hydra_core": package_version("hydra-core"), "numpy": np.__version__,
        "scikit_learn": package_version("scikit-learn"),
        "cuda_available": torch.cuda.is_available(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "physical_gpu_snapshot": gpu_snapshot(),
        "planned_training_device": "cuda:0 with CUDA_VISIBLE_DEVICES=0",
        "logical_to_physical_gpu_mapping": [
            {"logical_index": 0, "physical_index": 0, "uuid": next((row.get("uuid") for row in gpu_snapshot() if row.get("physical_index") == 0), None)}
        ] if os.environ.get("CUDA_VISIBLE_DEVICES") == "0" else [],
        "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "determinism": {"use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)},
        "recorded_head": git("rev-parse", "HEAD"), "recorded_branch": git("branch", "--show-current"),
        "source_hashes_at_prepare": {path: sha256_file(ROOT / path) for path in SOURCE_FILES if (ROOT / path).is_file()},
        "data_protocol": "MAGB Movies/Grocery/Reddit-S NC; canonical split seed 42; feature and graph loaders unchanged",
    }
    write_json(REPORT_ROOT / "environment_profile.json", profile)
    intake = [
        "# Intake: MVCGE private NC v1", "",
        f"- Actual starting HEAD: `{git('rev-parse', 'HEAD')}`.",
        f"- Task branch: `{git('branch', '--show-current')}`.",
        "- No applicable `AGENTS.md` was found under the project or its parent search path. `README.md` and the active Hydra model/task/dataset configs were read.",
        "- Historical S0 is the `S0` row in `docs/collab_adapt_nc_v2/protocol.md`: `MAGCollabAdapt` base `B1`, 768-dimensional padded readout, plus fixed path/reconstruction/pair losses. Implementation: `src/models/mag_collab_adapt.py`; training/task adapter: `scripts/run_collab_adapt_nc_v2.py`, `src/tasks/collab_nc_pilot.py`.",
        "- S0 selected LR/WD was read from `reports/collab_adapt_nc_v2/s_selection_lock.json`: Movies 0.003/0.001, Grocery 0.001/0.001, Reddit-S 0.001/0.001. All three are included in the requested calibration grid; no extra history-only cell is needed.",
        "- Canonical ingestion: `src/data/loaders.py` reads DGL edge indices and frozen `.npy` features, casts features to float32, joins text then image, uses seed-42 NC split, and calls `preprocess_edge_index` (`remove_self_loops`, `to_undirected`, configured self loops false). This loader does not expose edge weights or modality-availability metadata.",
        "- New graph operators are built from exactly the loader edge list, after excluding loops and adding one normalized self-loop. No extra reverse-edge insertion or feature transformation is done by the runner.",
        "- The selected MAGB split sizes and input hashes are recorded in `reports/mvcge_private_nc_v1/data_profile.json`; test split size is read only as metadata and test IDs/labels are not copied into the task runner.",
        "- The standard NC runner is `src/tasks/nc.py`; its generic loop computes test metrics, so this study uses a separate NC-only runner that calls the existing loader and legacy S0 modules but evaluates only explicitly supplied train and validation IDs. The existing LP endpoint-product decoder is `src/models/predictor.py`; it is used only in a synthetic interface test.",
        "- User-owned working-tree state observed before implementation: deleted `CLAUDE.md`; untracked `docs/V3_design.md`, `docs/mechanism_pilot_v2_design.md`, `docs/plan.md`, and `reports/collab_adapt_nc_v2/s_calibration_selection.json`. These remain unstaged and are excluded from this task commit.",
        "",
        "| Dataset | Nodes | Loader edges | Raw graph | Text/image dim | Classes | Train/val/test split size |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for dataset, item in data_profiles.items():
        raw = item["graph"]
        direction = "symmetric" if raw.get("reverse_edge_coverage") == 1.0 else "directed/asymmetric"
        intake.append(
            f"| {dataset} | {item['num_nodes']:,} | {item['loader_edge_rows']:,} | "
            f"{raw.get('raw_directed_edge_rows', 'unknown'):,} rows, {direction}, "
            f"{raw.get('raw_self_loop_rows', 'unknown')} raw self-loops, no edge-data weights | "
            f"{item['text_dim']}/{item['image_dim']} | {item['num_classes']} | "
            f"{item['train_size']:,}/{item['validation_size']:,}/{item['test_size']:,} |"
        )
    intake.extend([
        "",
        "All three loaded feature pairs are finite and have zero exactly-all-zero rows. This is a data-quality check only; it is not used to infer availability. The loader has no missing-modality metadata, so the canonical mask is all-present. Features are float32 with no additional L2 normalization, standardization, PCA, or re-extraction. Macro-F1 uses all 20 classes (0–19) with `zero_division=0`.",
    ])
    (DOC_ROOT / "intake.md").write_text("\n".join(intake) + "\n", encoding="utf-8")
    print(f"prepared {len(matrix['calibration_candidates'])} calibration and {len(matrix['confirmation_slots'])} confirmation slots")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_task_head(seed: int, classes: int, device: torch.device, *, legacy: bool) -> nn.Module:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed) + 10_007)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 10_007)
    if legacy:
        head = CollabTaskHeads("B1", classes, output_dim=768)
    else:
        head = nn.Linear(256, classes)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return head.to(device)


def build_modules(dataset: dict[str, Any], model_id: str, seed: int, device: torch.device):
    seed_all(seed)
    info = {"input_dim": dataset["text_dim"] + dataset["image_dim"], "num_nodes": dataset["n"], "num_classes": dataset["classes"], "text_dim": dataset["text_dim"], "visual_dim": dataset["image_dim"]}
    if model_id == "A0":
        model = MAGCollabAdapt(info, "B1", dropout=0.2, init_seed=seed, typed_value=False, fusion=False, out_dim=768, collect_diagnostics=False, optimize_cache=True).to(device)
        head = create_task_head(seed, dataset["classes"], device, legacy=True)
        model.private_seed = int(seed)
        return model, head
    model = MVCGEPrivateNC(dataset["text_dim"], dataset["image_dim"], model_id, init_seed=seed).to(device)
    model.set_graph(dataset["edges"], dataset["available"])
    head = create_task_head(seed, dataset["classes"], device, legacy=False)
    return model, head


def make_optimizer(model: nn.Module, head: nn.Module, model_id: str, lr: float, wd: float):
    if model_id == "A0":
        groups = parameter_groups_no_decay(model, [head])
        for index, group in enumerate(groups):
            group["weight_decay"] = 0.0 if group.get("weight_decay") == 0.0 else float(wd)
            group["lr"] = float(lr)
            group["name"] = "A0_decay" if group["weight_decay"] else "A0_no_decay"
    else:
        groups = model.optimizer_groups(lr, wd)
        groups.extend([
            {"params": [head.weight], "lr": float(lr), "weight_decay": float(wd), "name": "task_head_matrix"},
            {"params": [head.bias], "lr": float(lr), "weight_decay": 0.0, "name": "task_head_bias"},
        ])
        flattened = [id(parameter) for group in groups for parameter in group["params"]]
        expected = {id(parameter) for module in (model, head) for parameter in module.parameters() if parameter.requires_grad}
        if len(flattened) != len(set(flattened)) or set(flattened) != expected:
            raise RuntimeError("optimizer groups contain duplicate or missing parameters")
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def make_attempt_id(phase: str, dataset: str, model: str, seed: int, config_index: int, retry: int = 0) -> str:
    base = f"{phase}_{dataset}_{model}_seed{seed}_cfg{config_index:02d}"
    return base if retry == 0 else f"{base}_retry{retry}"


def attempt_sort_key(attempt_id: str, base_id: str) -> int:
    if attempt_id == base_id:
        return 0
    return int(attempt_id.rsplit("_retry", 1)[1])


def read_registry() -> dict[str, Any]:
    if REGISTRY_PATH.is_file():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"study": STUDY, "attempts": {}}


def save_registry(value: dict[str, Any]) -> None:
    write_json(REGISTRY_PATH, value)


def eval_node_ids(model_id: str, model: nn.Module, head: nn.Module, data: dict[str, Any], node_ids: torch.Tensor, eval_step: int = MAX_UPDATES) -> dict[str, Any]:
    model.eval(); head.eval()
    with torch.no_grad():
        if model_id == "A0":
            model.set_epoch(eval_step)
            z, _, _, _, _ = model(data["x"], data["edges"], data["available"], compute_ssl=False, collect_diagnostics=False)
        else:
            z, _ = model(data["x_text"], data["x_image"], data["available"], stochastic_routing=False)
        selected_z = z.index_select(0, node_ids)
        logits = head.main(selected_z) if model_id == "A0" else head(selected_z)
        labels = data["labels"].index_select(0, node_ids)
        predictions = logits.argmax(dim=-1)
    target_np = labels.detach().cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    class_ids = list(range(int(data["classes"])))
    matrix = confusion_matrix(target_np, pred_np, labels=class_ids)
    return {
        "count": int(labels.numel()),
        "correct": int((predictions == labels).sum().item()),
        "accuracy": float((predictions == labels).float().mean().item()),
        "macro_f1": float(f1_score(target_np, pred_np, labels=class_ids, average="macro", zero_division=0)),
        "class_ids": class_ids,
        "zero_division": 0,
        "confusion_matrix": matrix.tolist(),
        "node_ids": node_ids.detach().cpu().numpy().astype(np.int64),
        "labels": labels.detach().cpu().numpy().astype(np.int64),
        "argmax": pred_np.astype(np.int64),
    }


def a0_train_loss(model, head, data, noise_generator):
    model.set_epoch(int(getattr(model, "_mvcge_step", 1)))
    z, _, _, _, extras = model(data["x"], data["edges"], data["available"], pair_generator=noise_generator, compute_ssl=True, collect_diagnostics=False)
    total, terms, _ = train_only_loss("B1", z, extras, head, data["labels"], data["train_idx"], path_weight=0.05, rec_weight=0.01, pair_weight=0.01)
    return total, terms, None


def update_lr(optimizer, target_lrs: list[float], step: int) -> None:
    multiplier = 1.0 if step >= 20 else 0.1 + 0.9 * (step - 1) / 19.0
    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr * multiplier


def write_predictions(path: Path, train: dict[str, Any], validation: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_node_ids=train["node_ids"], train_labels=train["labels"], train_argmax=train["argmax"],
        validation_node_ids=validation["node_ids"], validation_labels=validation["labels"], validation_argmax=validation["argmax"],
    )
    return sha256_file(path)


def route_summary(details: dict[str, Any], masks: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for layer_index, layer in enumerate(details["layers"]):
        if layer["clean_logits"] is None:
            rows.append({"layer": layer_index, "router": "absent_uniform", "uniform_gate_entropy": math.log(4.0)})
            continue
        clean = layer["clean_logits"].detach()
        noisy = layer["noisy_logits"].detach()
        sigma = layer["sigma"].detach()
        by_modality = []
        for modality, name in enumerate(("text", "image")):
            valid = masks[:, modality]
            clean_gates, clean_indices = sparse_top2(clean[valid, modality])
            noisy_gates, noisy_indices = sparse_top2(noisy[valid, modality])

            def usage(indices: torch.Tensor) -> list[float]:
                return torch.bincount(indices.reshape(-1), minlength=4).float().div(max(indices.numel(), 1)).cpu().tolist()

            def combos(indices: torch.Tensor) -> dict[str, int]:
                normalized = indices.sort(dim=-1).values.cpu()
                result: dict[str, int] = {}
                if normalized.numel():
                    for pair in normalized.tolist():
                        key = f"{pair[0]}+{pair[1]}"
                        result[key] = result.get(key, 0) + 1
                return result

            def entropy(gates: torch.Tensor) -> float:
                return float((-(gates * gates.clamp_min(1e-12).log()).sum(-1)).mean().cpu()) if gates.numel() else 0.0

            sig = sigma[valid, modality]
            by_modality.append({
                "modality": name, "valid_tokens": int(valid.sum()),
                "clean_expert_usage": usage(clean_indices), "noisy_expert_usage": usage(noisy_indices),
                "clean_top2_combinations": combos(clean_indices), "noisy_top2_combinations": combos(noisy_indices),
                "clean_gate_entropy": entropy(clean_gates), "noisy_gate_entropy": entropy(noisy_gates),
                "sigma_mean": float(sig.mean().cpu()), "sigma_std": float(sig.std(unbiased=False).cpu()),
                "sigma_min": float(sig.min().cpu()), "sigma_max": float(sig.max().cpu()),
            })
        rows.append({"layer": layer_index, "router": "contextual_top2", "modalities": by_modality})
    return rows


def model_diagnostics(model_id: str, model: nn.Module, data: dict[str, Any], seed: int, best_step: int) -> dict[str, Any]:
    if model_id == "A0":
        model.eval(); model.set_epoch(best_step)
        with torch.no_grad():
            _, _, _, _, extras = model(data["x"], data["edges"], data["available"], compute_ssl=False, collect_diagnostics=True)
        paths = extras.get("path_states", {})
        increments = extras.get("graph_increments", ())
        return {
            "router": "not_present_in_S0", "theta": None,
            "legacy_state_mean_norms": {name: float(value.norm(dim=-1).mean().cpu()) for name, value in paths.items()},
            "legacy_graph_increment_mean_norms": [float(value.norm(dim=-1).mean().cpu()) for value in increments],
            "legacy_backbone_diagnostics": extras.get("diagnostics", {}),
        }
    model.eval()
    generator = torch.Generator(device=data["x_text"].device).manual_seed(seed + 73_331)
    with torch.no_grad():
        _, noisy_details = model(data["x_text"], data["x_image"], data["available"], stochastic_routing=True, noise_generator=generator, return_details=True)
        _, clean_details = model(data["x_text"], data["x_image"], data["available"], stochastic_routing=False, return_details=True)
    route = route_summary(noisy_details, data["available"])
    layer_norms = []
    for layer_index, layer in enumerate(clean_details["layers"]):
        layer_norms.append({
            "layer": layer_index,
            "public_output_mean_norm_by_modality": layer["public_mean_norm"].detach().cpu().tolist(),
            "private_output_mean_norm_by_modality": layer["private_mean_norm"].detach().cpu().tolist(),
            "residual_state_mean_norm_by_modality": layer["state_mean_norm"].detach().cpu().tolist(),
        })
    return {"routing": route, "layer_output_norms": layer_norms, "theta_and_spectral": model.parameter_diagnostics()}


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_cell(phase: str, dataset_name: str, model_id: str, seed: int, config_index: int, lr: float, wd: float, data: dict[str, Any], device: torch.device, *, retry_failed: bool = False) -> dict[str, Any]:
    registry = read_registry()
    base_id = make_attempt_id(phase, dataset_name, model_id, seed, config_index)
    related_ids = [key for key in registry["attempts"] if key == base_id or key.startswith(base_id + "_retry")]
    related_ids.sort(key=lambda key: attempt_sort_key(key, base_id))
    latest_id = related_ids[-1] if related_ids else base_id
    existing = registry["attempts"].get(latest_id)
    run_id = latest_id
    output_dir = OUTPUT_ROOT / phase / dataset_name / model_id / run_id
    latest_path, best_path = output_dir / "latest.pt", output_dir / "best.pt"
    history_path = output_dir / "steps.jsonl"
    if existing and existing.get("status") == "success":
        return existing
    if existing and existing.get("status") == "failed" and retry_failed:
        retry_num = attempt_sort_key(latest_id, base_id) + 1
        run_id = f"{base_id}_retry{retry_num}"
        output_dir = OUTPUT_ROOT / phase / dataset_name / model_id / run_id
        latest_path, best_path, history_path = output_dir / "latest.pt", output_dir / "best.pt", output_dir / "steps.jsonl"
        existing = None
    elif existing and existing.get("status") in {"running", "interrupted"} and latest_path.is_file():
        pass
    elif existing and existing.get("status") == "failed":
        return existing
    elif existing and existing.get("status") in {"running", "interrupted"}:
        existing = None
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit = git("rev-parse", "HEAD")
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES if (ROOT / relative).is_file()}
    model, head = build_modules(data, model_id, seed, device)
    optimizer = make_optimizer(model, head, model_id, lr, wd)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 31_415)
    target_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    trainable = [parameter for module in (model, head) for parameter in module.parameters() if parameter.requires_grad]
    if len({id(p) for p in trainable}) != len(trainable):
        raise RuntimeError("model and task head share parameter objects unexpectedly")
    pair_generator = noise_generator
    start_step, best_key, best_step, best_metrics, patience_reason = 1, None, 0, None, None
    best_train_loss_row = None
    resume = bool(existing and existing.get("status") in {"running", "interrupted"} and latest_path.is_file())
    if resume:
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); head.load_state_dict(state["head"]); optimizer.load_state_dict(state["optimizer"])
        noise_generator.set_state(state["noise_rng"])
        torch.random.set_rng_state(state["torch_rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["torch_rng_cuda"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        start_step = int(state["step"]) + 1
        best_key, best_step, best_metrics = state["best_key"], int(state["best_step"]), state["best_metrics"]
        best_train_loss_row = state.get("best_train_loss_row")
        if history_path.is_file():
            kept = [row for row in history_path.read_text(encoding="utf-8").splitlines() if row.strip() and int(json.loads(row)["step"]) <= int(state["step"])]
            history_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    else:
        attempt = {
            "attempt_id": run_id, "phase": phase, "dataset": dataset_name, "model": model_id,
            "training_seed": seed, "split_seed": SPLIT_SEED, "config_index": config_index,
            "lr": lr, "matrix_weight_decay": wd, "status": "running", "started_utc": utc_now(),
            "output_dir": str(output_dir), "source_commit": source_commit, "source_hashes": source_hashes,
            "train_ids_sha256": data["split_hashes"]["train_ids"], "validation_ids_sha256": data["split_hashes"]["validation_ids"],
            "test_task_head_calls": 0, "real_lp_runs": 0,
        }
        registry["attempts"][run_id] = attempt
        save_registry(registry)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step_times_ms: list[float] = []
    reason = "max_updates"
    try:
        for step in range(start_step, MAX_UPDATES + 1):
            model.train(); head.train()
            if model_id == "A0":
                model._mvcge_step = step
                model.set_epoch(step)
            update_lr(optimizer, target_lrs, step)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
            else:
                wall_start = time.perf_counter()

            if model_id == "A0":
                total_loss, terms, _ = a0_train_loss(model, head, data, pair_generator)
                task_loss = terms["main_ce"]
                balance_loss = total_loss.new_zeros(())
            else:
                z, detail = model(data["x_text"], data["x_image"], data["available"], step=step, stochastic_routing=True, noise_generator=noise_generator)
                train_z = z.index_select(0, data["train_idx"])
                train_y = data["labels"].index_select(0, data["train_idx"])
                logits = head(train_z)
                task_loss = F.cross_entropy(logits, train_y, label_smoothing=0.0)
                balance_loss = detail["balance_loss"]
                total_loss = task_loss + (0.01 * balance_loss if model_id in {"G0", "G1"} else 0.0)
                terms = {"main_ce": task_loss, "balance": balance_loss}
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            task_router_gradient = None
            if model_id in {"G0", "G1"} and step in GRADIENT_DIAGNOSTIC_STEPS:
                grad_parts = torch.autograd.grad(task_loss, model.routers[0].logit_head.parameters(), retain_graph=True, allow_unused=True)
                task_router_gradient = float(torch.stack([g.detach().float().square().sum() for g in grad_parts if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in grad_parts) else 0.0
            total_loss.backward()
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0, error_if_nonfinite=True)
            group_gradients: dict[str, float] = {}
            if step in GRADIENT_DIAGNOSTIC_STEPS:
                if model_id == "A0":
                    group_specs = {"backbone": list(model.parameters()), "task_head": list(head.parameters())}
                else:
                    group_specs = {
                        "adapter": list(model.adapters.parameters()),
                        "experts": [p for layer in model.layers for p in layer.experts.parameters()],
                        "private_ffn": [p for layer in model.layers if layer.private is not None for p in layer.private.parameters()],
                        "router": list(model.routers.parameters()) if model.routers is not None else [],
                    }
                for name, params in group_specs.items():
                    norms = [p.grad.detach().float().square().sum() for p in params if p.grad is not None]
                    group_gradients[name] = float(torch.stack(norms).sum().sqrt().cpu()) if norms else 0.0
            optimizer.step()
            if device.type == "cuda":
                event_end.record(); torch.cuda.synchronize(device)
                step_ms = float(event_start.elapsed_time(event_end))
            else:
                step_ms = float((time.perf_counter() - wall_start) * 1000.0)
            step_times_ms.append(step_ms)
            ce_value = float(task_loss.detach().cpu())
            balance_value = float(balance_loss.detach().cpu())
            ratio = float(0.01 * balance_value / max(ce_value, 1e-12)) if model_id in {"G0", "G1"} else 0.0
            train_record = {
                "step": step, "train_ce": ce_value,
                "balance_loss": balance_value, "weighted_balance_to_ce_ratio": ratio,
                "total_loss": float(total_loss.detach().cpu()),
                "lr_by_group": {str(group.get("name", i)): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)},
                "optimizer_update_ms": step_ms,
                "pre_clip_global_grad_norm": float(pre_clip_norm.detach().cpu()) if step in GRADIENT_DIAGNOSTIC_STEPS else None,
                "task_only_router_logit_grad_norm": task_router_gradient,
                "module_gradient_norms": group_gradients if step in GRADIENT_DIAGNOSTIC_STEPS else None,
            }

            evaluated = False
            improved = False
            if step >= EVAL_FIRST and (step - EVAL_FIRST) % EVAL_INTERVAL == 0:
                evaluated = True
                validation = eval_node_ids(model_id, model, head, data, data["val_idx"], step)
                key = (int(validation["correct"]), float(validation["macro_f1"]), -int(step))
                improved = best_key is None or key > tuple(best_key)
                train_record["validation"] = {k: validation[k] for k in ("count", "correct", "accuracy", "macro_f1")}
                if improved:
                    best_key, best_step = key, step
                    best_metrics = {k: validation[k] for k in ("count", "correct", "accuracy", "macro_f1")}
                    best_train_loss_row = {k: train_record[k] for k in ("step", "train_ce", "balance_loss", "weighted_balance_to_ce_ratio", "total_loss")}
                    atomic_torch_save(best_path, {"model": model.state_dict(), "head": head.state_dict(), "best_step": step, "best_key": key, "validation": best_metrics, "source_commit": source_commit, "source_hashes": source_hashes})
            append_jsonl(history_path, train_record)

            latest_due = step % 10 == 0 or improved or step == MAX_UPDATES
            if latest_due:
                atomic_torch_save(latest_path, {
                    "model": model.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                    "noise_rng": noise_generator.get_state(), "torch_rng_cpu": torch.random.get_rng_state(),
                    "torch_rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "step": step, "best_key": best_key, "best_step": best_step,
                    "best_metrics": best_metrics, "best_train_loss_row": best_train_loss_row,
                })
            if step >= MIN_UPDATES and best_step > 0 and step - best_step >= PATIENCE_UPDATES:
                reason = "early_stop_patience_150_optimizer_updates"
                break
        else:
            step = MAX_UPDATES
        if best_key is None or not best_path.is_file():
            raise RuntimeError("run ended without an eligible validation checkpoint")
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model"]); head.load_state_dict(best_payload["head"])
        train_metrics = eval_node_ids(model_id, model, head, data, data["train_idx"], best_step)
        val_metrics = eval_node_ids(model_id, model, head, data, data["val_idx"], best_step)
        prediction_path = output_dir / "train_validation_argmax.npz"
        prediction_hash = write_predictions(prediction_path, train_metrics, val_metrics)
        diagnostics = model_diagnostics(model_id, model, data, seed, best_step)
        steps_trained = int(step)
        if steps_trained >= MAX_UPDATES and best_step >= MAX_UPDATES - EVAL_INTERVAL:
            reason = "max_updates_validation_still_improving_budget_may_be_insufficient"
        elif steps_trained >= MAX_UPDATES and reason != "early_stop_patience_150_optimizer_updates":
            reason = "max_updates_reached"
        elapsed = time.perf_counter() - started
        params_backbone = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
        result = {
            "attempt_id": run_id, "phase": phase, "dataset": dataset_name, "model": model_id,
            "training_seed": seed, "split_seed": SPLIT_SEED, "config_index": config_index,
            "lr": lr, "matrix_weight_decay": wd, "status": "success",
            "source_commit": source_commit, "source_hashes": source_hashes,
            "train_ids_sha256": data["split_hashes"]["train_ids"], "validation_ids_sha256": data["split_hashes"]["validation_ids"],
            "best_step": best_step, "updates": steps_trained, "early_stop_reason": reason,
            "validation_best_key": list(best_key), "best_step_training_loss": best_train_loss_row,
            "train": {k: v for k, v in train_metrics.items() if k not in {"node_ids", "labels", "argmax"}},
            "validation": {k: v for k, v in val_metrics.items() if k not in {"node_ids", "labels", "argmax"}},
            "active_backbone_parameters": params_backbone, "active_task_head_parameters": params_head,
            "median_step_ms": float(np.median(step_times_ms)),
            "median_non_diagnostic_step_ms": float(np.median([value for i, value in enumerate(step_times_ms, 1) if i not in GRADIENT_DIAGNOSTIC_STEPS])) if len(step_times_ms) > len(GRADIENT_DIAGNOSTIC_STEPS) else float(np.median(step_times_ms)),
            "total_training_wall_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "optimizer_groups": [{"name": str(group.get("name", index)), "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]), "parameter_count": sum(p.numel() for p in group["params"])} for index, group in enumerate(optimizer.param_groups)],
            "diagnostics": diagnostics, "prediction_file": str(prediction_path), "prediction_sha256": prediction_hash,
            "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256_file(best_path),
            "test_task_head_calls": 0, "test_metrics_computed": False, "real_lp_runs": 0,
        }
        attempt = dict(registry["attempts"].get(run_id, {})); attempt.update(result); attempt["ended_utc"] = utc_now()
        registry["attempts"][run_id] = attempt; save_registry(registry)
        append_jsonl(RESULTS_PATH, result)
        return attempt
    except Exception as exc:
        failed = {
            "attempt_id": run_id, "phase": phase, "dataset": dataset_name, "model": model_id,
            "training_seed": seed, "config_index": config_index, "lr": lr, "matrix_weight_decay": wd,
            "status": "failed", "failed_utc": utc_now(), "error": repr(exc), "traceback": traceback.format_exc(limit=20),
            "last_completed_update": (step if "step" in locals() else 0), "output_dir": str(output_dir),
            "source_commit": source_commit, "source_hashes": source_hashes,
            "test_task_head_calls": 0, "real_lp_runs": 0,
        }
        registry = read_registry(); registry["attempts"][run_id] = failed; save_registry(registry); append_jsonl(RESULTS_PATH, failed)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return failed
    finally:
        del model, head, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_smoke(device: torch.device) -> dict[str, Any]:
    rows = []
    for dataset_name in DATASETS:
        data = load_dataset(dataset_name, device)
        for model_id in NEW_MODELS:
            seed = 88_000 + DATASETS.index(dataset_name) * 10 + NEW_MODELS.index(model_id)
            model, head = build_modules(data, model_id, seed, device)
            optimizer = make_optimizer(model, head, model_id, 0.001, 0.001)
            gen = torch.Generator(device=device).manual_seed(seed + 31_415)
            times = []
            losses = []
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            for step in (1, 2, 3):
                model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
                start = time.perf_counter()
                z, details = model(data["x_text"], data["x_image"], data["available"], step=step, stochastic_routing=True, noise_generator=gen)
                logits = head(z.index_select(0, data["train_idx"]))
                target = data["labels"].index_select(0, data["train_idx"])
                ce = F.cross_entropy(logits, target)
                balance = details["balance_loss"]
                loss = ce + (0.01 * balance if model_id != "U1" else 0.0)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for m in (model, head) for p in m.parameters() if p.requires_grad], 5.0, error_if_nonfinite=True)
                optimizer.step()
                if device.type == "cuda": torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000.0)
                losses.append({"step": step, "ce": float(ce.detach().cpu()), "balance": float(balance.detach().cpu()), "total": float(loss.detach().cpu()), "gradient_norm": float(norm.detach().cpu())})
            rows.append({
                "dataset": dataset_name, "model": model_id, "seed": seed, "updates": 3,
                "status": "success", "train_only": True, "validation_called": False,
                "test_task_head_calls": 0, "real_lp_runs": 0, "losses": losses,
                "median_step_ms": float(np.median(times)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            })
            print(f"smoke {dataset_name}/{model_id}: success, 3 train-only updates; median {np.median(times):.2f} ms", flush=True)
            del model, head, optimizer
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        del data
    result = {"study": STUDY, "status": "success" if len(rows) == 9 and all(row["status"] == "success" for row in rows) else "incomplete", "runs": rows}
    write_json(REPORT_ROOT / "smoke_summary.json", result)
    return result


def save_calibration_selection() -> dict[str, Any]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    registry = read_registry()
    selected: dict[str, Any] = {}
    failures = []
    for dataset in DATASETS:
        for model in MODELS:
            rows = []
            for candidate in matrix["calibration_candidates"]:
                if candidate["dataset"] == dataset and candidate["model"] == model:
                    attempt_id = make_attempt_id("calibration", dataset, model, CALIBRATION_SEED, candidate["config_index"])
                    possible_ids = [key for key in registry["attempts"] if key == attempt_id or key.startswith(attempt_id + "_retry")]
                    possible_ids.sort(key=lambda key: attempt_sort_key(key, attempt_id))
                    successful = [registry["attempts"][key] for key in possible_ids if registry["attempts"][key].get("status") == "success"]
                    record = successful[-1] if successful else None
                    if not record or record.get("status") != "success":
                        failures.append(attempt_id)
                    else:
                        rows.append((candidate, record))
            if len(rows) != 9:
                continue
            candidate, record = max(rows, key=lambda pair: (pair[1]["validation"]["correct"], pair[1]["validation"]["macro_f1"], -pair[0]["config_index"]))
            selected[f"{dataset}/{model}"] = {
                "lr": candidate["lr"], "matrix_weight_decay": candidate["matrix_weight_decay"],
                "config_index": candidate["config_index"], "calibration_attempt_id": record["attempt_id"],
                "best_step": record["best_step"], "validation_correct": record["validation"]["correct"],
                "validation_accuracy": record["validation"]["accuracy"], "validation_macro_f1": record["validation"]["macro_f1"],
            }
    selection = {
        "study": STUDY, "stage": "calibration_selection", "created_utc": utc_now(),
        "rule": "validation correct count descending, fixed-C Macro-F1 descending, lower config_index",
        "calibration_seed": CALIBRATION_SEED, "matrix_sha256": sha256_file(MATRIX_PATH),
        "training_source_commit": git("rev-parse", "HEAD"), "selection_uses_confirmation": False,
        "complete": len(selected) == len(DATASETS) * len(MODELS) and not failures,
        "successful_selections": selected, "missing_or_failed_attempts": failures,
    }
    if not selection["complete"]:
        write_json(REPORT_ROOT / "calibration_selection_incomplete.json", selection)
        raise RuntimeError(f"selection cannot be locked; missing/failed cells={failures}")
    write_json(REPORT_ROOT / "calibration_selection.json", selection)
    write_json(SELECTION_PATH, selection)
    return selection


def verify_selection() -> dict[str, Any]:
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError("confirmation requires reports/mvcge_private_nc_v1/selection_lock.json")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if not selection.get("complete") or selection.get("selection_uses_confirmation"):
        raise RuntimeError("selection lock is incomplete or uses confirmation data")
    if selection["matrix_sha256"] != sha256_file(MATRIX_PATH):
        raise RuntimeError("matrix changed after calibration selection")
    return selection


def run_phase(phase: str, device: torch.device, *, retry_failed: bool = False) -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if phase == "confirmation":
        selection = verify_selection()["successful_selections"]
        slots = []
        for dataset in DATASETS:
            for model in MODELS:
                lock = selection[f"{dataset}/{model}"]
                for seed in CONFIRMATION_SEEDS:
                    slots.append({"dataset": dataset, "model": model, "seed": seed, "config_index": lock["config_index"], "lr": lock["lr"], "matrix_weight_decay": lock["matrix_weight_decay"]})
    else:
        slots = [row for row in matrix["calibration_candidates"]]
    loaded: dict[str, dict[str, Any]] = {}
    for slot in slots:
        dataset_name = slot["dataset"]
        if dataset_name not in loaded:
            loaded[dataset_name] = load_dataset(dataset_name, device)
        result = run_cell(phase, dataset_name, slot["model"], slot["seed"], slot["config_index"], slot["lr"], slot["matrix_weight_decay"], loaded[dataset_name], device, retry_failed=retry_failed)
        print(f"{phase} {dataset_name}/{slot['model']} seed={slot['seed']} cfg={slot['config_index']}: {result['status']} step={result.get('best_step')}", flush=True)
        del result
    if phase == "calibration":
        save_calibration_selection()


def main() -> None:
    parser = argparse.ArgumentParser(description="MVCGE private NC v1: explicit NC-only experiment runner")
    parser.add_argument("phase", choices=("prepare", "smoke", "calibrate", "confirm"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        return
    if args.phase == "smoke":
        run_smoke(torch.device(args.device))
        return
    if not MATRIX_PATH.is_file():
        raise FileNotFoundError("run prepare before training")
    if args.phase == "calibrate":
        run_phase("calibration", torch.device(args.device), retry_failed=args.retry_failed)
    else:
        run_phase("confirmation", torch.device(args.device), retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
