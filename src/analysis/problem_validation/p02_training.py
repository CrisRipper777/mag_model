from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from src.analysis.problem_validation.common import _tensor_hash, make_physical_graph, set_run_seed
from src.analysis.problem_validation.p02_models import P02Model, Variant
from src.data import MAGData


class IncomingCSR:
    """Incoming adjacency for source -> target messages, using the full physical degree."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device):
        edge = torch.as_tensor(edge_index, dtype=torch.long).cpu()
        if edge.numel() and bool((edge[0] == edge[1]).any()):
            raise ValueError("physical P0.2 graph must not contain self-loops")
        if edge.numel():
            key = edge[1] * int(num_nodes) + edge[0]
            order = torch.argsort(key, stable=True)
            edge = edge[:, order]
        self.num_nodes = int(num_nodes)
        self.edge_index_cpu = edge.contiguous()
        self.src = edge[0].to(device)
        self.dst = edge[1].to(device)
        degree = torch.bincount(edge[1], minlength=num_nodes).long()
        self.degree = degree.to(device)
        self.row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), degree.cumsum(0))).to(device)
        self.device = device

    def select_edges(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source IDs, local target rows, and full-graph degrees for a target batch."""
        targets = targets.to(self.device, dtype=torch.long)
        counts = self.degree[targets]
        total = int(counts.sum().item())
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, counts
        row_offsets = counts.cumsum(0) - counts
        edge_base = torch.repeat_interleave(self.row_ptr[targets] - row_offsets, counts)
        positions = edge_base + torch.arange(total, device=self.device)
        local_target = torch.repeat_interleave(torch.arange(targets.numel(), device=self.device), counts)
        return self.src[positions], local_target, counts


def safe_labels(data: MAGData, indices: torch.Tensor, permitted_indices: torch.Tensor) -> torch.Tensor:
    """Index labels only after proving every requested node belongs to an allowed phase."""
    if data.y is None:
        raise ValueError("NC labels are absent")
    query = indices.detach().cpu().long().flatten()
    permitted = permitted_indices.detach().cpu().long().flatten()
    if query.numel() and not bool(torch.isin(query, permitted).all()):
        raise PermissionError("label access requested outside explicitly permitted train/calib/original-val nodes")
    return data.y[query.to(data.y.device)]


def load_fixed_split(data: MAGData, dataset: str, split_path: Path) -> dict[str, Any]:
    """Read, never create or rewrite, the exact P0 split and relation-population cache."""
    if not split_path.exists():
        raise FileNotFoundError(f"required frozen P0 split artifact is missing: {split_path}; stopping without regenerating it")
    if data.train_idx is None or data.val_idx is None or data.test_idx is None:
        raise ValueError(f"{dataset}: incomplete source NC split")
    cached = torch.load(split_path, map_location="cpu", weights_only=False)
    source = {
        "original_train_sha256": _tensor_hash(data.train_idx),
        "original_val_sha256": _tensor_hash(data.val_idx),
    }
    if cached.get("source_split_hashes") != source:
        raise RuntimeError(f"{dataset}: frozen P0 split cache does not match source train/val indices")
    ptrain = cached["probe_train_idx"].cpu().long()
    pcal = cached["probe_calib_idx"].cpu().long()
    train = data.train_idx.cpu().long()
    val = data.val_idx.cpu().long()
    test = data.test_idx.cpu().long()
    train_set, val_set, test_set = set(train.tolist()), set(val.tolist()), set(test.tolist())
    ptrain_set, pcal_set = set(ptrain.tolist()), set(pcal.tolist())
    if ptrain_set & pcal_set or ptrain_set | pcal_set != train_set:
        raise RuntimeError(f"{dataset}: probe_train/probe_calib do not exactly partition original train")
    if ptrain_set & val_set or pcal_set & val_set:
        raise RuntimeError(f"{dataset}: probe split overlaps original validation")
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError(f"{dataset}: original train/val/test index splits overlap")
    return {
        "probe_train_idx": ptrain,
        "probe_calib_idx": pcal,
        "original_train_idx": train,
        "original_val_idx": val,
        "original_test_idx": test,
        "analysis_target_nodes": cached["analysis_target_nodes"].cpu().long(),
        "sampled_edge_target_node": cached["sampled_edge_target_node"].cpu().long(),
        "sampled_edge_neighbor_node": cached["sampled_edge_neighbor_node"].cpu().long(),
        "source_split_hashes": source,
        "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "probe_train_sha256": _tensor_hash(ptrain),
        "probe_calib_sha256": _tensor_hash(pcal),
        "original_test_index_sha256": _tensor_hash(test),
        "original_val_sha256": _tensor_hash(val),
        "split_audit": {
            "probe_train_size": len(ptrain_set),
            "probe_calib_size": len(pcal_set),
            "original_train_size": len(train_set),
            "original_val_size": len(val_set),
            "original_test_index_count": len(test_set),
            "probe_train_calib_overlap": 0,
            "probe_train_original_val_overlap": 0,
            "probe_calib_original_val_overlap": 0,
            "train_val_test_index_overlap": 0,
        },
    }


def _edge_batch_features(
    model: P02Model,
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    targets: torch.Tensor,
    *,
    identity_function: bool = False,
) -> torch.Tensor:
    targets = targets.to(graph.device, dtype=torch.long)
    src, local_target, degree = graph.select_edges(targets)
    outputs: list[torch.Tensor] = []
    for modality_index, embeddings in enumerate((h_text, h_visual)):
        aggregate = embeddings.new_zeros((targets.numel(), embeddings.size(1)))
        if src.numel():
            target_global = targets[local_target]
            messages, _, _ = model.edge_messages(
                modality_index,
                embeddings[target_global],
                embeddings[src],
                identity_function=identity_function,
            )
            aggregate.index_add_(0, local_target, messages)
        aggregate = aggregate / degree.clamp_min(1).to(aggregate.dtype).unsqueeze(-1)
        outputs.append(aggregate)
    return torch.cat((h_text[targets], h_visual[targets], outputs[0], outputs[1]), dim=-1)


def predict_indices(
    model: P02Model,
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    identity_function: bool = False,
) -> torch.Tensor:
    model.eval()
    idx = indices.to(graph.device, dtype=torch.long)
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for targets in idx.split(batch_size):
            features = _edge_batch_features(
                model, graph, h_text, h_visual, targets, identity_function=identity_function
            )
            logits.append(model.classifier(features))
    return torch.cat(logits, dim=0) if logits else torch.empty((0, model.classifier.out_features), device=graph.device)


def _metric_values(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    logits = logits.detach()
    labels = labels.to(logits.device).long()
    preds = logits.argmax(-1)
    return {
        "acc": float((preds == labels).float().mean().item()),
        "macro_f1": float(
            f1_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "ce": float(F.cross_entropy(logits, labels).item()),
    }


def phase_metrics(
    data: MAGData,
    model: P02Model,
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    indices: torch.Tensor,
    permitted_indices: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    labels = safe_labels(data, indices, permitted_indices)
    logits = predict_indices(model, graph, h_text, h_visual, indices, batch_size=batch_size)
    return _metric_values(logits, labels, int(data.num_classes))


def train_variant(
    data: MAGData,
    variant: Variant,
    run_seed: int,
    split: dict[str, Any],
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    config: dict[str, Any],
    *,
    device: torch.device,
    checkpoint_path: Path,
    config_fingerprint: str,
) -> tuple[P02Model, dict[str, Any]]:
    """Full-physical-graph one-hop training with memory-bounded target mini-batches."""
    set_run_seed(run_seed)
    a = config["analysis"]
    model = P02Model(
        variant,
        num_classes=int(data.num_classes),
        semantic_dim=int(a["semantic_dim"]),
        relation_dim=int(a["relation_dim"]),
        relation_dropout=float(a["relation_dropout"]),
        scalar_gate_scale=float(a["scalar_gate_scale"]),
        feature_modulation_scale=float(a["feature_modulation_scale"]),
        similarity_init_alpha=float(a["similarity_init_alpha"]),
        similarity_init_beta=float(a["similarity_init_beta"]),
    ).to(device)
    optimizer_cfg = a["optimizer"]
    if str(optimizer_cfg["name"]).lower() != "adamw":
        raise ValueError("P0.2 training currently requires the configured AdamW optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    train_idx = split["probe_train_idx"].to(device)
    calib_idx = split["probe_calib_idx"]
    val_idx = split["original_val_idx"]
    permitted = torch.cat((split["probe_train_idx"], split["probe_calib_idx"], val_idx))
    generator = torch.Generator(device="cpu").manual_seed(int(run_seed))
    batch_size = int(a["target_batch_size"])
    best_acc = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = int(a["patience"])
    best_epoch_ce = float("nan")
    history: list[dict[str, float]] = []

    for epoch in range(1, int(a["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        permutation = torch.randperm(train_idx.numel(), generator=generator)
        order = train_idx[permutation.to(device)]
        for target_batch in order.split(batch_size):
            features = _edge_batch_features(model, graph, h_text, h_visual, target_batch)
            logits = model.classifier(features)
            labels = safe_labels(data, target_batch, split["probe_train_idx"]).to(device)
            loss = F.cross_entropy(logits, labels, reduction="sum") / train_idx.numel()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(a["grad_clip"]), error_if_nonfinite=True
        )
        optimizer.step()

        calib_metrics = phase_metrics(
            data,
            model,
            graph,
            h_text,
            h_visual,
            calib_idx,
            permitted,
            batch_size=batch_size,
        )
        history.append({"epoch": float(epoch), **{f"calib_{k}": v for k, v in calib_metrics.items()}})
        if calib_metrics["acc"] > best_acc + float(a["min_delta"]):
            best_acc = calib_metrics["acc"]
            best_epoch = epoch
            best_epoch_ce = calib_metrics["ce"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(a["patience"])
        elif epoch >= int(a["min_epoch"]):
            patience_left -= 1
            if patience_left <= 0:
                break
        if device.type == "cuda" and epoch % 25 == 0:
            torch.cuda.empty_cache()

    if best_state is None:
        raise RuntimeError(f"{variant} did not produce a probe-calib-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "dataset": data.name,
        "run_seed": int(run_seed),
        "variant": variant,
        "best_epoch": int(best_epoch),
        "best_probe_calib_ce": float(best_epoch_ce),
        "probe_train": phase_metrics(
            data, model, graph, h_text, h_visual, split["probe_train_idx"], split["probe_train_idx"], batch_size=batch_size
        ),
        "probe_calib": phase_metrics(
            data, model, graph, h_text, h_visual, calib_idx, permitted, batch_size=batch_size
        ),
        "heldout_original_val": phase_metrics(
            data, model, graph, h_text, h_visual, val_idx, permitted, batch_size=batch_size
        ),
        "checkpoint_selection": "probe_calib_accuracy_only",
        "parameter_counts": model.parameter_counts(),
        "training_history": history,
        "test_evaluation": False,
        "test_labels_accessed": False,
        "config_fingerprint": config_fingerprint,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "variant": variant,
            "dataset": data.name,
            "run_seed": int(run_seed),
            "best_epoch": int(best_epoch),
            "selection_metric": "probe_calib_accuracy",
            "semantic_embeddings_frozen": True,
            "config_fingerprint": config_fingerprint,
        },
        checkpoint_path,
    )
    return model, metrics


def make_model(data: MAGData, variant: Variant, config: dict[str, Any], device: torch.device) -> P02Model:
    a = config["analysis"]
    return P02Model(
        variant,
        num_classes=int(data.num_classes),
        semantic_dim=int(a["semantic_dim"]),
        relation_dim=int(a["relation_dim"]),
        relation_dropout=float(a["relation_dropout"]),
        scalar_gate_scale=float(a["scalar_gate_scale"]),
        feature_modulation_scale=float(a["feature_modulation_scale"]),
        similarity_init_alpha=float(a["similarity_init_alpha"]),
        similarity_init_beta=float(a["similarity_init_beta"]),
    ).to(device)


def load_checkpoint_model(
    checkpoint_path: Path,
    data: MAGData,
    variant: Variant,
    config: dict[str, Any],
    device: torch.device,
    expected_fingerprint: str,
) -> tuple[P02Model, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"checkpoint/config mismatch at {checkpoint_path}")
    model = make_model(data, variant, config, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def tensor_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()
