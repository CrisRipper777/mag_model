from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from src.analysis.problem_validation.common import set_run_seed
from src.analysis.problem_validation.p02_training import safe_labels
from src.data import MAGData


def build_context_readout_features(
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    contexts: dict[str, list[torch.Tensor]],
    readout: str,
) -> torch.Tensor:
    """L2-normalize each context block immediately before the diagnostic head."""
    if readout == "context_only":
        blocks = [contexts["text"][k] for k in (1, 2, 3)] + [
            contexts["visual"][k] for k in (1, 2, 3)
        ]
    elif readout == "full_bank":
        blocks = [h_text, h_visual]
        for k in (1, 2, 3):
            blocks.extend((contexts["text"][k], contexts["visual"][k]))
    else:
        raise ValueError(f"unknown context readout: {readout}")
    return torch.cat([F.normalize(block, p=2, dim=-1) for block in blocks], dim=-1)


def _phase_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    labels = labels.to(logits.device).long()
    prediction = logits.argmax(-1)
    return {
        "acc": float((prediction == labels).float().mean().item()),
        "macro_f1": float(
            f1_score(
                labels.detach().cpu().numpy(),
                prediction.detach().cpu().numpy(),
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "ce": float(F.cross_entropy(logits, labels).item()),
    }


def train_context_readout(
    data: MAGData,
    features: torch.Tensor,
    split: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Train only one linear readout; embeddings and relation states remain frozen."""
    a = config["analysis"]
    set_run_seed(seed)
    device = features.device
    model = nn.Linear(features.size(-1), int(data.num_classes)).to(device)
    optimizer_cfg = a["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    train_idx_cpu = split["probe_train_idx"]
    train_idx = train_idx_cpu.to(device)
    train_labels = safe_labels(data, train_idx_cpu, train_idx_cpu).to(device)
    calib_idx = split["probe_calib_idx"].to(device)
    calib_labels = safe_labels(data, split["probe_calib_idx"], split["probe_calib_idx"]).to(device)
    val_idx = split["original_val_idx"].to(device)
    permitted = torch.cat((split["probe_train_idx"], split["probe_calib_idx"], split["original_val_idx"]))
    val_labels = safe_labels(data, split["original_val_idx"], permitted).to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_acc = -1.0
    best_epoch = 0
    patience_left = int(a["patience"])
    best_state: dict[str, torch.Tensor] | None = None
    batch_size = int(a["target_batch_size"])
    history: list[dict[str, float]] = []

    for epoch in range(1, int(a["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        perm = torch.randperm(train_idx.numel(), generator=generator).to(device)
        order = train_idx[perm]
        for batch in order.split(batch_size):
            logits = model(features[batch])
            labels = safe_labels(data, batch, train_idx_cpu).to(device)
            (F.cross_entropy(logits, labels, reduction="sum") / train_idx.numel()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(a["grad_clip"]), error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            calib_logits = model(features[calib_idx])
            calib = _phase_metrics(calib_logits, calib_labels, int(data.num_classes))
        history.append({"epoch": float(epoch), **{f"calib_{k}": v for k, v in calib.items()}})
        if calib["acc"] > best_acc + float(a["min_delta"]):
            best_acc = calib["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(a["patience"])
        elif epoch >= int(a["min_epoch"]):
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("context diagnostic head did not produce a calibration-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_metrics = _phase_metrics(model(features[train_idx]), train_labels, int(data.num_classes))
        calib_metrics = _phase_metrics(model(features[calib_idx]), calib_labels, int(data.num_classes))
        val_metrics = _phase_metrics(model(features[val_idx]), val_labels, int(data.num_classes))
    return {
        "best_epoch": int(best_epoch),
        "probe_train": train_metrics,
        "probe_calib": calib_metrics,
        "heldout_original_val": val_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selection_metric": "probe_calib_accuracy",
        "frozen_relation_parameters": True,
        "test_evaluation": False,
        "test_labels_accessed": False,
        "training_history": history,
    }
