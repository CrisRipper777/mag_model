from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.analysis.problem_validation.common import (
    PROBE_DIM,
    classification_metrics,
    set_run_seed,
    write_json,
)
from src.data import MAGData


class SemanticProbe(nn.Module):
    def __init__(self, text_dim: int, visual_dim: int, num_classes: int):
        super().__init__()
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, PROBE_DIM),
            nn.LayerNorm(PROBE_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, PROBE_DIM),
            nn.LayerNorm(PROBE_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(2 * PROBE_DIM, num_classes)

    def encode(self, x_text: torch.Tensor, x_visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_projector(x_text), self.visual_projector(x_visual)

    def forward(self, x_text: torch.Tensor, x_visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_text, h_visual = self.encode(x_text, x_visual)
        logits = self.classifier(torch.cat((h_text, h_visual), dim=-1))
        return logits, h_text, h_visual


def train_semantic_probe(
    data: MAGData,
    run_seed: int,
    probe_train_idx: torch.Tensor,
    probe_calib_idx: torch.Tensor,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Fit P0.0 using only the inner train/calibration split."""
    assert data.x_t is not None and data.x_i is not None and data.y is not None
    run_dir.mkdir(parents=True, exist_ok=True)
    set_run_seed(run_seed)
    model = SemanticProbe(data.x_t.size(1), data.x_i.size(1), int(data.num_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_text = data.x_t.to(device)
    x_visual = data.x_i.to(device)
    train_idx = probe_train_idx.to(device)
    calib_idx = probe_calib_idx.to(device)
    train_labels = data.y[probe_train_idx].to(device)
    calib_labels = data.y[probe_calib_idx].to(device)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 30
    patience_left = patience
    calib_metrics: dict[str, float] = {}
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        h_text, h_visual = model.encode(x_text, x_visual)
        train_logits = model.classifier(torch.cat((h_text[train_idx], h_visual[train_idx]), dim=-1))
        loss = criterion(train_logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h_text, h_visual = model.encode(x_text, x_visual)
            calib_logits = model.classifier(torch.cat((h_text[calib_idx], h_visual[calib_idx]), dim=-1))
            calib_metrics = classification_metrics(calib_logits, calib_labels, int(data.num_classes))
        if calib_metrics["acc"] > best_acc + 1e-4:
            best_acc = calib_metrics["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        elif epoch >= 30:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("P0.0 training did not produce a calibration-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    # The checkpoint is frozen before original-val sanity metrics are computed.
    assert data.val_idx is not None
    with torch.no_grad():
        h_text, h_visual = model.encode(x_text, x_visual)
        train_logits = model.classifier(torch.cat((h_text[train_idx], h_visual[train_idx]), dim=-1))
        calib_logits = model.classifier(torch.cat((h_text[calib_idx], h_visual[calib_idx]), dim=-1))
        val_idx = data.val_idx.to(device)
        val_labels = data.y[data.val_idx].to(device)
        val_logits = model.classifier(torch.cat((h_text[val_idx], h_visual[val_idx]), dim=-1))
    train_metrics = classification_metrics(train_logits, train_labels, int(data.num_classes))
    calib_metrics = classification_metrics(calib_logits, calib_labels, int(data.num_classes))
    val_metrics = classification_metrics(val_logits, val_labels, int(data.num_classes))
    split_metadata = {
        "data_split_seed": 42,
        "probe_split_seed": 42,
        "probe_train_size": int(probe_train_idx.numel()),
        "probe_calib_size": int(probe_calib_idx.numel()),
        "original_val_size": int(data.val_idx.numel()),
        "selection_metric": "probe_calib_accuracy",
    }
    checkpoint = {
        "text_projector_state": {k: v.detach().cpu().clone() for k, v in model.text_projector.state_dict().items()},
        "visual_projector_state": {k: v.detach().cpu().clone() for k, v in model.visual_projector.state_dict().items()},
        "classifier_state": {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()},
        "run_seed": run_seed,
        "split_metadata": split_metadata,
        "best_epoch": best_epoch,
        "probe_train_acc": train_metrics["acc"],
        "probe_train_macro_f1": train_metrics["macro_f1"],
        "probe_calib_acc": calib_metrics["acc"],
        "probe_calib_macro_f1": calib_metrics["macro_f1"],
    }
    torch.save(checkpoint, run_dir / "semantic_probe.pt")
    torch.save(
        {
            "H_text": h_text.detach().cpu().float(),
            "H_visual": h_visual.detach().cpu().float(),
            "run_seed": run_seed,
        },
        run_dir / "semantic_embeddings.pt",
    )
    metrics = {
        "dataset": data.name,
        "run_seed": run_seed,
        "best_epoch": best_epoch,
        "probe_train": train_metrics,
        "probe_calib": calib_metrics,
        "heldout_analysis_val": val_metrics,
        "checkpoint_selection": "probe_calib_accuracy_only",
        "test_set_evaluated": False,
    }
    write_json(run_dir / "p00_metrics.json", metrics)
    return metrics


def load_frozen_projectors(checkpoint_path: Path, text_dim: int, visual_dim: int, num_classes: int, device: torch.device) -> SemanticProbe:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SemanticProbe(text_dim, visual_dim, num_classes)
    model.text_projector.load_state_dict(checkpoint["text_projector_state"])
    model.visual_projector.load_state_dict(checkpoint["visual_projector_state"])
    model.classifier.load_state_dict(checkpoint["classifier_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
