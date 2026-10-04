from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import torch
from omegaconf import OmegaConf


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def fixed_split_path(dataset: str) -> Path:
    if dataset == "ele-fashion":
        return Path("/hdd1/DataInHere/YHF/data/ele-fashion/split.pt")
    return Path(f"/hdd1/DataInHere/YHF/data/MAGB_split/{dataset}_nc_seed42_train0.6_val0.2.pt")


def compose_probe_config(dataset: str, training_seed: int, split_path: str | Path,
                         lr: float, variant: str, device: str = "cuda:0"):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        cfg = compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=mag_probe",
            f"seed={int(training_seed)}", "num_runs=1", f"device={device}",
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.nc_split_path = str(Path(split_path).resolve()) if dataset != "ele-fashion" else cfg.dataset.get("nc_split_path")
    cfg.dataset.auto_generate_splits = False
    cfg.task.lr = float(lr)
    cfg.task.weight_decay = 1e-4
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 1e-4
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.scheduler = None
    cfg.task.training_mode = "full_graph"
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 0.0
    cfg.model.variant = variant
    OmegaConf.resolve(cfg)
    return cfg


def dataset_input_paths(cfg) -> dict[str, Path]:
    values = OmegaConf.to_container(cfg.dataset, resolve=True)
    if str(values["source"]).lower() == "magb":
        return {key: Path(values[key]).resolve() for key in
                ("graph_path", "text_feat_path", "image_feat_path", "nc_split_path")}
    paths = {"joint_feat_path": Path(values["joint_feat_path"]).resolve(),
             "edge_path": Path(values["edge_path"]).resolve(),
             "label_path": Path(values["label_path"]).resolve(),
             "node_split_path": Path(values["node_split_path"]).resolve()}
    if "node_mapping_path" in values:
        paths["node_mapping_path"] = Path(values["node_mapping_path"]).resolve()
    return paths


def load_raw_edges(cfg) -> torch.Tensor:
    values = OmegaConf.to_container(cfg.dataset, resolve=True)
    if str(values["source"]).lower() == "magb":
        import dgl
        graphs, _ = dgl.load_graphs(str(Path(values["graph_path"]).resolve()))
        src, dst = graphs[0].edges()
        return torch.stack((src.long(), dst.long()), dim=0).contiguous()
    raw = torch.load(Path(values["edge_path"]).resolve(), map_location="cpu", weights_only=False)
    edge = torch.as_tensor(raw, dtype=torch.long)
    if edge.ndim != 2:
        raise ValueError("MM-Graph raw edges must be rank two")
    return edge.t().contiguous() if edge.size(1) == 2 and edge.size(0) != 2 else edge.contiguous()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def confusion_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int,
                      macro_f1_label_set: list[int]) -> dict[str, Any]:
    y_true = torch.as_tensor(y_true, dtype=torch.long).reshape(-1).cpu()
    y_pred = torch.as_tensor(y_pred, dtype=torch.long).reshape(-1).cpu()
    if y_true.numel() != y_pred.numel():
        raise ValueError("target and prediction lengths differ")
    if y_true.numel() and (int(y_true.min()) < 0 or int(y_true.max()) >= num_classes):
        raise ValueError("target contains a class outside the complete confusion schema")
    if y_pred.numel() and (int(y_pred.min()) < 0 or int(y_pred.max()) >= num_classes):
        raise ValueError("prediction contains a class outside the complete confusion schema")
    matrix = torch.bincount(y_true * num_classes + y_pred,
                            minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = matrix.diag().to(torch.float64)
    actual = matrix.sum(dim=1).to(torch.float64)
    predicted = matrix.sum(dim=0).to(torch.float64)
    den = actual + predicted
    f1 = torch.where(den > 0, 2 * tp / den, torch.zeros_like(den))
    labels = sorted(set(map(int, macro_f1_label_set)))
    if not labels or labels[0] < 0 or labels[-1] >= num_classes:
        raise ValueError("macro_f1_label_set must be a nonempty subset of 0..C-1")
    return {
        "n": int(y_true.numel()),
        "acc": float(tp.sum().item() / max(int(y_true.numel()), 1)),
        "macro_f1": float(f1[labels].mean().item()),
        "macro_f1_label_set": labels,
        "class_order": list(range(num_classes)),
        "confusion_matrix": matrix.tolist(),
    }


def correctness_pair(a_true: torch.Tensor, a_pred: torch.Tensor,
                     b_true: torch.Tensor, b_pred: torch.Tensor) -> dict[str, Any]:
    a_true, a_pred = a_true.cpu().long(), a_pred.cpu().long()
    b_true, b_pred = b_true.cpu().long(), b_pred.cpu().long()
    if not torch.equal(a_true, b_true):
        raise ValueError("paired predictions must use identical target nodes")
    a_ok, b_ok = a_true == a_pred, b_true == b_pred
    return {
        "n": int(a_true.numel()),
        "both_correct": int((a_ok & b_ok).sum()),
        "a_only_correct": int((a_ok & ~b_ok).sum()),
        "b_only_correct": int((~a_ok & b_ok).sum()),
        "both_wrong": int((~a_ok & ~b_ok).sum()),
        "accuracy_difference_a_minus_b": float(a_ok.float().mean() - b_ok.float().mean()),
    }


def modality_complementarity(y: torch.Tensor, pred_text: torch.Tensor,
                             pred_image: torch.Tensor) -> dict[str, Any]:
    y, pred_text, pred_image = y.cpu().long(), pred_text.cpu().long(), pred_image.cpu().long()
    tc, ic = pred_text == y, pred_image == y
    n = int(y.numel())
    return {
        "n": n,
        "both_correct": int((tc & ic).sum()),
        "text_only_correct": int((tc & ~ic).sum()),
        "image_only_correct": int((~tc & ic).sum()),
        "both_wrong": int((~tc & ~ic).sum()),
        "prediction_disagreement": int((pred_text != pred_image).sum()),
        "prediction_disagreement_rate": float((pred_text != pred_image).float().mean()) if n else None,
        "oracle_accuracy_label_using_upper_bound": float(1 - int((~tc & ~ic).sum()) / n) if n else None,
    }


def validate_loaded_data(data, raw_edge_index: torch.Tensor | None = None) -> dict[str, Any]:
    if data.x_t is None or data.x_i is None or data.y is None:
        raise ValueError(f"{data.name}: expected two modalities and node labels")
    n, c = int(data.num_nodes), int(data.num_classes)
    feature_report: dict[str, Any] = {}
    for name, feature in (("text", data.x_t), ("image", data.x_i)):
        if feature.ndim != 2 or feature.shape[0] != n:
            raise ValueError(f"{data.name}: {name} feature rows do not align with nodes")
        finite_rows = torch.isfinite(feature).all(dim=1)
        feature_report[name] = {
            "shape": list(feature.shape), "dtype": str(feature.dtype),
            "finite_values": bool(torch.isfinite(feature).all()),
            "nonfinite_rows": int((~finite_rows).sum()),
            "zero_rows": int((torch.linalg.vector_norm(torch.nan_to_num(feature), dim=1) == 0).sum()),
        }
        if not feature_report[name]["finite_values"]:
            raise ValueError(f"{data.name}: {name} features contain NaN or infinity")
    if data.x.shape != (n, int(data.x_t.size(1) + data.x_i.size(1))):
        raise ValueError(f"{data.name}: concatenated feature tensor has unexpected shape")
    labels = data.y.reshape(-1).long().cpu()
    if labels.numel() != n:
        raise ValueError(f"{data.name}: label rows do not align with nodes")
    label_min, label_max = int(labels.min()), int(labels.max())
    if label_min < -1 or label_max >= c:
        raise ValueError(f"{data.name}: labels must be -1 or in 0..{c-1}, got [{label_min},{label_max}]")
    splits: dict[str, Any] = {}
    indices = {}
    for name, idx in (("train", data.train_idx), ("val", data.val_idx), ("test", data.test_idx)):
        if idx is None:
            raise ValueError(f"{data.name}: missing {name} split")
        idx = idx.reshape(-1).long().cpu()
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n):
            raise ValueError(f"{data.name}: {name} split contains out-of-range node IDs")
        if idx.unique().numel() != idx.numel():
            raise ValueError(f"{data.name}: {name} split has duplicate node IDs")
        indices[name] = idx
        split_labels = labels[idx]
        splits[name] = {
            "n": int(idx.numel()),
            "label_counts": {str(int(k)): int(v) for k, v in zip(*torch.unique(split_labels, return_counts=True))},
            "unlabeled_nodes": int((split_labels < 0).sum()),
        }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if torch.isin(indices[a], indices[b]).any():
            raise ValueError(f"{data.name}: {a}/{b} split overlap")
    if (labels[indices["train"]] < 0).any() or (labels[indices["val"]] < 0).any() or (labels[indices["test"]] < 0).any():
        raise ValueError(f"{data.name}: supervised splits include missing labels")
    observed = torch.unique(labels[torch.cat(list(indices.values()))])
    macro_labels = sorted(int(v) for v in observed.tolist() if 0 <= int(v) < c)

    edge = data.edge_index.detach().long().cpu()
    if edge.ndim != 2 or edge.size(0) != 2:
        raise ValueError(f"{data.name}: processed edge_index is not [2,E]")
    def edge_counts(value):
        value = value.detach().long().cpu()
        if value.ndim != 2:
            raise ValueError("raw edge array must be rank two")
        if value.size(0) != 2 and value.size(1) == 2:
            value = value.t().contiguous()
        if value.size(0) != 2:
            raise ValueError("raw edge array must be [2,E] or [E,2]")
        nedge = value.size(1)
        out_of_range = int(((value < 0) | (value >= n)).any(dim=0).sum()) if nedge else 0
        self_loops = int((value[0] == value[1]).sum())
        valid = value[:, (value[0] >= 0) & (value[0] < n) & (value[1] >= 0) & (value[1] < n)]
        nonself = valid[:, valid[0] != valid[1]]
        unique = torch.unique(nonself.t().contiguous(), dim=0).size(0) if nonself.numel() else 0
        return {"edge_count": nedge, "out_of_range_edges": out_of_range,
                "self_loops": self_loops, "duplicate_directed_edges": int(nonself.size(1) - unique)}
    raw_report = edge_counts(raw_edge_index) if raw_edge_index is not None else None
    processed_report = edge_counts(edge)
    if processed_report["out_of_range_edges"] or processed_report["self_loops"] or processed_report["duplicate_directed_edges"]:
        raise ValueError(f"{data.name}: processed graph failed boundary/self-loop/duplicate validation: {processed_report}")
    return {
        "dataset": data.name, "source": data.source, "num_nodes": n,
        "num_classes_configured": c, "num_classes_observed": len(macro_labels),
        "class_schema": list(range(c)), "macro_f1_label_set": macro_labels,
        "label_min": label_min, "label_max": label_max,
        "missing_label_count": int((labels < 0).sum()),
        "feature_order": "[text,image]", "features": feature_report,
        "splits": splits, "split_hashes": data.info.get("split_hashes", {}),
        "raw_graph": raw_report, "processed_graph": processed_report,
        "processed_edges_sha256": tensor_sha256(edge),
    }


def receiver_mean(edge_index: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge = edge_index.detach().long().to(features.device)
    src, dst = edge[0], edge[1]
    keep = src != dst
    pairs = torch.stack((dst[keep], src[keep]), dim=1)
    if pairs.numel():
        pairs = torch.unique(pairs, dim=0, sorted=True)
        row, col = pairs[:, 0], pairs[:, 1]
        degree = torch.bincount(row, minlength=features.size(0))
        weight = degree[row].to(features.dtype).reciprocal()
        op = torch.sparse_coo_tensor(torch.stack((row, col)), weight,
                                     (features.size(0), features.size(0)), device=features.device).coalesce()
        return torch.sparse.mm(op, features), degree
    return features.new_zeros(features.shape), torch.zeros(features.size(0), dtype=torch.long, device=features.device)


def train_train_homophily(edge_index: torch.Tensor, y: torch.Tensor, train_idx: torch.Tensor) -> dict[str, Any]:
    y = y.cpu().long()
    train = torch.zeros(y.numel(), dtype=torch.bool)
    train[train_idx.cpu().long()] = True
    src, dst = edge_index.detach().long().cpu()
    keep = train[src] & train[dst] & (src != dst) & (y[src] >= 0) & (y[dst] >= 0)
    src, dst = src[keep], dst[keep]
    same = int((y[src] == y[dst]).sum())
    total = int(src.numel())
    return {"train_train_directed_edges": total, "same_label_edges": same,
            "homophily_rate": (same / total) if total else None,
            "label_source": "train labels and edges with both endpoints in train"}


def geometry_features(edge_index: torch.Tensor, x_t: torch.Tensor, x_i: torch.Tensor,
                      train_idx: torch.Tensor, val_idx: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, x in (("text", x_t), ("image", x_i)):
        neigh, degree = receiver_mean(edge_index, x)
        dot = (x * neigh).sum(1)
        den = x.norm(dim=1) * neigh.norm(dim=1)
        cosine = torch.where(den > 0, dot / den.clamp_min(1e-12), torch.zeros_like(den))
        eligible = degree > 0
        tr = train_idx.cpu().long()
        tr_degree = degree[tr].to(torch.float64)
        tr_degree = tr_degree[tr_degree > 0]
        tr_cos = cosine[tr][eligible[tr]].to(torch.float64)
        if tr_degree.numel():
            degree_q = torch.quantile(tr_degree, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        else:
            degree_q = torch.zeros(3, dtype=torch.float64)
        if tr_cos.numel():
            cosine_q = torch.quantile(tr_cos, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        else:
            cosine_q = torch.zeros(3, dtype=torch.float64)
        result[name] = {
            "degree": degree.cpu(), "cosine": cosine.cpu(), "isolated": (degree == 0).cpu(),
            "degree_thresholds": [float(v) for v in degree_q],
            "cosine_thresholds": [float(v) for v in cosine_q],
            "threshold_source": "training nodes only; nonisolated nodes for modality cosine",
            "val_idx": val_idx.cpu().long(),
        }
    return result


def bucket_id(values: torch.Tensor, thresholds: list[float]) -> torch.Tensor:
    cuts = torch.tensor(thresholds, dtype=torch.float64)
    return torch.bucketize(values.cpu().to(torch.float64), cuts, right=True)


def bucket_metrics(y: torch.Tensor, pred_a: torch.Tensor, pred_b: torch.Tensor,
                   num_classes: int, label_set: list[int], indices: torch.Tensor,
                   bucket: torch.Tensor, bucket_names: list[str], threshold_source: str) -> list[dict[str, Any]]:
    y, pred_a, pred_b = y.cpu().long(), pred_a.cpu().long(), pred_b.cpu().long()
    indices, bucket = indices.cpu().long(), bucket.cpu().long()
    out = []
    if pred_a.numel() == y.numel() and pred_b.numel() == y.numel():
        pred_at_nodes, pred_bt_nodes = pred_a, pred_b
    elif pred_a.numel() == indices.numel() and pred_b.numel() == indices.numel():
        # Predictions are stored in split order; translate picked node IDs to split offsets.
        node_to_offset = torch.full((y.numel(),), -1, dtype=torch.long)
        node_to_offset[indices] = torch.arange(indices.numel())
        pred_at_nodes = pred_a[node_to_offset.clamp_min(0)]
        pred_bt_nodes = pred_b[node_to_offset.clamp_min(0)]
    else:
        raise ValueError("bucket predictions must align with all nodes or the supplied evaluation split")

    def compressed_matrix(matrix: list[list[int]]) -> dict[str, Any]:
        size = len(matrix)
        payload = json.dumps(matrix, separators=(",", ":")).encode("utf-8")
        encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
        return {"shape": [size, size], "class_order": list(range(size)),
                "encoding": "base64(zlib(UTF-8 JSON dense matrix))", "data": encoded}

    for bin_id, name in enumerate(bucket_names):
        picked = indices[bucket[indices] == bin_id]
        if not picked.numel():
            continue
        a = confusion_metrics(y[picked], pred_at_nodes[picked], num_classes, label_set)
        b = confusion_metrics(y[picked], pred_bt_nodes[picked], num_classes, label_set)
        paired = correctness_pair(y[picked], pred_at_nodes[picked], y[picked], pred_bt_nodes[picked])
        out.append({"bucket": name, "n": int(picked.numel()),
                    "small_n": int(picked.numel()) < 100, "threshold_source": threshold_source,
                    "a_correct": int((y[picked] == pred_at_nodes[picked]).sum()),
                    "b_correct": int((y[picked] == pred_bt_nodes[picked]).sum()),
                    "a_confusion_matrix": compressed_matrix(a["confusion_matrix"]),
                    "b_confusion_matrix": compressed_matrix(b["confusion_matrix"]),
                    "paired_a_minus_b": {k: v for k, v in paired.items() if k != "accuracy_difference_a_minus_b"},
                    "paired_improvement": paired["a_only_correct"],
                    "paired_regression": paired["b_only_correct"]})
    return out
