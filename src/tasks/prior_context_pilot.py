from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PilotTaskHeads(nn.Module):
    """External NC heads for R0-R6, including SUPRA-style R6 logit averaging."""

    def __init__(self, variant: str, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.variant = str(variant).upper()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        if self.variant == "R6":
            self.text = nn.Linear(128, self.num_classes)
            self.image = nn.Linear(128, self.num_classes)
            self.joint = nn.Linear(128, self.num_classes)
        else:
            self.main = nn.Linear(self.embedding_dim, self.num_classes)
            self.source_text = nn.Linear(128, self.num_classes)
            self.source_image = nn.Linear(128, self.num_classes)

    def forward(self, z: torch.Tensor, extras: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source = extras["source_states"]
        if self.variant == "R6":
            text_logits = self.text(source["text"])
            image_logits = self.image(source["image"])
            joint_logits = self.joint(z[:, 256:384])
            return (text_logits + image_logits + joint_logits) / 3.0, {
                "text": text_logits, "image": image_logits,
            }
        return self.main(z), {
            "text": self.source_text(source["text"]),
            "image": self.source_image(source["image"]),
        }


def train_only_loss(
    z: torch.Tensor,
    extras: dict[str, Any],
    heads: PilotTaskHeads,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    source_aux_weight: float = 0.1,
    *,
    labels_are_train_only: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute all supervised loss after indexing train nodes only."""
    selected = train_idx.to(device=z.device, dtype=torch.long)
    train_labels = labels if labels_are_train_only else labels.index_select(0, selected)
    if train_labels.numel() != selected.numel():
        raise ValueError("train labels and train indices must have matching lengths")
    source_states = extras["source_states"]
    train_extras = {
        "source_states": {
            name: value.index_select(0, selected) for name, value in source_states.items()
        }
    }
    train_z = z.index_select(0, selected)
    logits, source_logits = heads(train_z, train_extras)
    main_ce = F.cross_entropy(logits, train_labels)
    source_losses = [
        F.cross_entropy(source_logits[name], train_labels)
        for name in ("text", "image")
    ]
    source_ce = torch.stack(source_losses).mean()
    total = main_ce + float(source_aux_weight) * source_ce
    return total, {"main_ce": main_ce, "source_ce": source_ce}
