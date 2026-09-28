from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from src.data import MAGData
from src.data.loaders import PROJECT_ROOT, load_mag_data
from src.utils.seeds import set_seed


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
RUN_SEEDS = (42, 43, 44)
SPLIT_SEED = 42
PROBE_DIM = 128
BOOTSTRAP_REPLICATES = 1000
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "problem_validation"


def load_dataset(name: str) -> tuple[DictConfig, MAGData]:
    """Load NC data with the fixed data split seed, independently of run seed."""
    if name not in DATASETS:
        raise ValueError(f"Unsupported problem-validation dataset: {name}")
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={name}", "task=nc", "seed=42"])
    # Passing SPLIT_SEED here is intentional for MAGB; MM-Graph ignores it and
    # returns its official split. Do not substitute the model initialization seed.
    data = load_mag_data(cfg, "nc", seed=SPLIT_SEED)
    validate_modalities(name, data)
    return cfg, data


def validate_modalities(name: str, data: MAGData) -> None:
    if data.x_t is None or data.x_i is None or data.y is None:
        raise ValueError(f"{name}: loader must provide x_t, x_i, and y")
    if data.x_t.ndim != 2 or data.x_i.ndim != 2 or data.x_t.shape[0] != data.x_i.shape[0]:
        raise ValueError(f"{name}: invalid modality feature shapes")
    combined = torch.cat((data.x_t, data.x_i), dim=1)
    if data.x.shape != combined.shape or not torch.equal(data.x, combined):
        raise ValueError(f"{name}: loader x does not follow the configured [x_t, x_i] modality order")
    if not (torch.isfinite(data.x_t).all() and torch.isfinite(data.x_i).all()):
        raise ValueError(f"{name}: non-finite input features")
    if data.y.numel() != data.num_nodes or data.num_classes is None:
        raise ValueError(f"{name}: labels/classes do not match loader metadata")


def _tensor_hash(values: torch.Tensor) -> str:
    payload = values.detach().cpu().long().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def make_probe_split(data: MAGData, dataset: str, cache_root: Path = OUTPUT_ROOT / "splits") -> dict[str, Any]:
    """Create/cache the single stratified 80/20 split inside original train."""
    if data.train_idx is None or data.val_idx is None or data.test_idx is None or data.y is None:
        raise ValueError(f"{dataset}: incomplete NC split")
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{dataset}.pt"
    source = {
        "original_train_sha256": _tensor_hash(data.train_idx),
        "original_val_sha256": _tensor_hash(data.val_idx),
    }
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("source_split_hashes") != source:
            raise RuntimeError(f"{dataset}: cached probe split does not match current source split")
        result = cached
    else:
        train = data.train_idx.cpu().long()
        labels = data.y[train].cpu().numpy()
        if np.any(labels < 0) or np.any(labels >= int(data.num_classes)):
            raise ValueError(f"{dataset}: original train contains missing/out-of-range class labels")
        probe_train, probe_calib = train_test_split(
            train.numpy(), test_size=0.20, random_state=SPLIT_SEED, shuffle=True, stratify=labels
        )
        result = {
            "probe_train_idx": torch.as_tensor(probe_train, dtype=torch.long),
            "probe_calib_idx": torch.as_tensor(probe_calib, dtype=torch.long),
            "source_split_hashes": source,
            "dataset": dataset,
            "data_split_seed": SPLIT_SEED,
            "probe_train_fraction": 0.8,
            "probe_calib_fraction": 0.2,
        }
        torch.save(result, path)
    probe_train = set(result["probe_train_idx"].tolist())
    probe_calib = set(result["probe_calib_idx"].tolist())
    original_train = set(data.train_idx.cpu().tolist())
    original_val = set(data.val_idx.cpu().tolist())
    assert probe_train.isdisjoint(probe_calib)
    assert probe_train | probe_calib == original_train
    assert probe_train.isdisjoint(original_val)
    assert probe_calib.isdisjoint(original_val)
    return result


def audit_split(
    data: MAGData,
    probe_train_idx: torch.Tensor,
    probe_calib_idx: torch.Tensor,
    analysis_target_nodes: torch.Tensor,
) -> dict[str, Any]:
    """Audit intersections. test indices are used only for disjointness checks."""
    assert data.train_idx is not None and data.val_idx is not None and data.test_idx is not None
    ptrain, pcal, val, test = [
        set(x.detach().cpu().long().tolist())
        for x in (probe_train_idx, probe_calib_idx, data.val_idx, data.test_idx)
    ]
    target = set(analysis_target_nodes.detach().cpu().long().tolist())
    assert ptrain.isdisjoint(pcal)
    assert ptrain.isdisjoint(val)
    assert pcal.isdisjoint(val)
    assert target.issubset(val)
    assert target.isdisjoint(test)
    return {
        "original_train_size": int(data.train_idx.numel()),
        "original_val_size": int(data.val_idx.numel()),
        "original_test_size": int(data.test_idx.numel()),
        "probe_train_size": len(ptrain),
        "probe_calib_size": len(pcal),
        "analysis_target_count": len(target),
        "probe_train_calib_overlap": 0,
        "probe_train_original_val_overlap": 0,
        "probe_calib_original_val_overlap": 0,
        "analysis_target_test_overlap": 0,
    }


def make_physical_graph(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Undirect, drop self-loops, and coalesce the loader's physical edges."""
    edge_index = torch.as_tensor(edge_index, dtype=torch.long).cpu()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    if edge_index.numel() and bool((edge_index[0] == edge_index[1]).any()):
        raise AssertionError("self-loop remained in P0.1 physical graph")
    return edge_index.contiguous()


def sample_analysis_population(
    data: MAGData,
    edge_index: torch.Tensor,
    seed: int = 42,
    max_targets: int = 10_000,
    max_degree: int = 64,
) -> dict[str, torch.Tensor]:
    """Class-stratified target and fixed incident-edge sample, independent of run seed."""
    assert data.val_idx is not None and data.y is not None
    val = data.val_idx.cpu().long().numpy()
    val_labels = data.y[data.val_idx].cpu().numpy()
    target_nodes = val
    rng = np.random.default_rng(seed)
    if val.size > max_targets:
        labels = val_labels
        classes, counts = np.unique(labels, return_counts=True)
        if np.any(labels < 0):
            raise ValueError("original val split includes a missing class label")
        raw_quota = counts * (max_targets / val.size)
        quota = np.floor(raw_quota).astype(int)
        quota = np.maximum(quota, 1)
        while quota.sum() > max_targets:
            candidates = np.where(quota > 1)[0]
            if not len(candidates):
                raise ValueError("cannot retain every validation class within target budget")
            k = candidates[np.argmin(raw_quota[candidates] - quota[candidates])]
            quota[k] -= 1
        while quota.sum() < max_targets:
            candidates = np.where(quota < counts)[0]
            k = candidates[np.argmax(raw_quota[candidates] - quota[candidates])]
            quota[k] += 1
        picked: list[np.ndarray] = []
        for cls, n_pick in zip(classes, quota, strict=True):
            cls_nodes = val[labels == cls]
            picked.append(rng.choice(cls_nodes, size=int(n_pick), replace=False))
        target_nodes = np.sort(np.concatenate(picked))
    else:
        target_nodes = np.sort(target_nodes)

    # edge_index is src,dst; incoming relations for target i are src -> i.
    src, dst = edge_index.cpu().numpy()
    by_target: dict[int, list[int]] = {}
    for s, d in zip(src.tolist(), dst.tolist(), strict=True):
        by_target.setdefault(int(d), []).append(int(s))
    targets: list[int] = []
    neighbors: list[int] = []
    degrees: list[int] = []
    for node in target_nodes.tolist():
        nbs = np.asarray(sorted(by_target.get(int(node), [])), dtype=np.int64)
        if nbs.size == 0:
            continue
        full_degree = int(nbs.size)
        if full_degree > max_degree:
            nbs = np.sort(rng.choice(nbs, size=max_degree, replace=False))
        targets.extend([int(node)] * nbs.size)
        neighbors.extend(nbs.tolist())
        degrees.extend([full_degree] * nbs.size)
    return {
        "target_nodes": torch.as_tensor(target_nodes, dtype=torch.long),
        "target_node": torch.as_tensor(targets, dtype=torch.long),
        "neighbor_node": torch.as_tensor(neighbors, dtype=torch.long),
        "target_degree": torch.as_tensor(degrees, dtype=torch.long),
    }


def mean_neighbor_messages(embeddings: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple incoming mean; degree-zero rows are zero and returned in the mask."""
    n = embeddings.size(0)
    src, dst = edge_index.to(embeddings.device)
    sums = embeddings.new_zeros(embeddings.shape)
    degree = embeddings.new_zeros((n, 1))
    sums.index_add_(0, dst, embeddings[src])
    degree.index_add_(0, dst, embeddings.new_ones((dst.numel(), 1)))
    isolated = degree.squeeze(-1) == 0
    means = sums / degree.clamp_min(1)
    means[isolated] = 0
    return means, isolated


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    return {
        "acc": float(np.mean(pred == target)),
        "macro_f1": float(f1_score(target, pred, labels=list(range(num_classes)), average="macro", zero_division=0)),
    }


def set_run_seed(seed: int) -> None:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
