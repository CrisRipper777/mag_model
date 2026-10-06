from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mag_collab_nc import D
from src.tasks.prior_context_pilot import PilotTaskHeads


class CollabTaskHeads(nn.Module):
    """Task-side classifier heads; labels never enter the backbone."""

    def __init__(self, variant: str, num_classes: int, output_dim: int = 640):
        super().__init__()
        self.variant = str(variant).upper()
        self.num_classes = int(num_classes)
        if self.variant == "B2":
            self.legacy = PilotTaskHeads("R1", num_classes, embedding_dim=128)
            self.main = self.legacy.main
            self.source_text = self.legacy.source_text
            self.source_image = self.legacy.source_image
        else:
            self.main = nn.Linear(output_dim, num_classes)
            if self.variant == "B0":
                self.path_heads = nn.ModuleDict()
            else:
                self.path_heads = nn.ModuleDict({
                    "text_private": nn.Linear(D, num_classes),
                    "image_private": nn.Linear(D, num_classes),
                    "shared": nn.Linear(2 * D, num_classes),
                })
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward_new(self, z: torch.Tensor, extras: dict[str, Any]):
        main = self.main(z)
        path_logits = {name: head(extras["path_states"][name]) for name, head in self.path_heads.items()}
        return main, path_logits


def _zero(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def train_only_loss(
    variant: str,
    z: torch.Tensor,
    extras: dict[str, Any],
    heads: CollabTaskHeads,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    source_weight: float = 0.1,
    path_weight: float = 0.05,
    rec_weight: float = 0.01,
    pair_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Select the fixed train node IDs before reading labels or applying any task head."""
    variant = str(variant).upper()
    idx = train_idx.to(device=z.device, dtype=torch.long)
    y_train = labels.index_select(0, idx)
    z_train = z.index_select(0, idx)
    if variant == "B2":
        source_states = extras["source_states"]
        train_extras = {"source_states": {k: v.index_select(0, idx) for k, v in source_states.items()}}
        logits, sources = heads.legacy(z_train, train_extras)
        main_ce = F.cross_entropy(logits, y_train)
        valid_all = extras.get("source_available")
        source_losses = []
        for name in ("text", "image"):
            valid = torch.ones_like(idx, dtype=torch.bool) if valid_all is None else valid_all.index_select(0, idx)[:, 0 if name == "text" else 1]
            source_losses.append(F.cross_entropy(sources[name][valid], y_train[valid]) if bool(valid.any()) else _zero(sources[name]))
        source_ce = torch.stack(source_losses).mean()
        total = main_ce + float(source_weight) * source_ce
        return total, {"main_ce": main_ce, "source_ce": source_ce, "path_ce": _zero(main_ce), "rec": _zero(main_ce), "pair": _zero(main_ce)}, {"main": logits, **sources}

    extras_train = {"path_states": {name: tensor.index_select(0, idx) for name, tensor in extras.get("path_states", {}).items()}}
    main_logits, path_logits = heads.forward_new(z_train, extras_train)
    main_ce = F.cross_entropy(main_logits, y_train)
    path_ce = _zero(main_logits)
    source_losses = []
    if variant != "B0":
        available = extras["availability"].index_select(0, idx)
        masks = {
            "text_private": available[:, 0],
            "image_private": available[:, 1],
            "shared": available.any(-1),
        }
        for name in ("text_private", "image_private", "shared"):
            mask = masks[name]
            source_losses.append(F.cross_entropy(path_logits[name][mask], y_train[mask]) if bool(mask.any()) else _zero(path_logits[name]))
        path_ce = torch.stack(source_losses).mean()
    rec = extras["loss_terms"].get("rec", _zero(main_logits))
    pair = extras["loss_terms"].get("pair", _zero(main_logits))
    total = main_ce + float(path_weight) * path_ce
    if variant in {"B1", "M0"}:
        total = total + float(rec_weight) * rec + float(pair_weight) * pair
    return total, {"main_ce": main_ce, "source_ce": path_ce, "path_ce": path_ce, "rec": rec, "pair": pair}, {"main": main_logits, **path_logits}


def indexed_logits(
    variant: str,
    z: torch.Tensor,
    extras: dict[str, Any],
    heads: CollabTaskHeads,
    node_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply heads only to an explicitly supplied train or validation index."""
    idx = node_ids.to(device=z.device, dtype=torch.long)
    z_part = z.index_select(0, idx)
    if str(variant).upper() == "B2":
        part_extras = {
            "source_states": {k: v.index_select(0, idx) for k, v in extras["source_states"].items()},
        }
        logits, sources = heads.legacy(z_part, part_extras)
        available = extras.get("source_available")
        if available is None:
            masks = {"text": torch.ones_like(idx, dtype=torch.bool), "image": torch.ones_like(idx, dtype=torch.bool)}
        else:
            part = available.index_select(0, idx)
            masks = {"text": part[:, 0], "image": part[:, 1]}
        return {"main": logits, **sources}, masks
    path_states = {k: v.index_select(0, idx) for k, v in extras.get("path_states", {}).items()}
    main, paths = heads.forward_new(z_part, {"path_states": path_states})
    masks = {"text_private": extras["availability"].index_select(0, idx)[:, 0], "image_private": extras["availability"].index_select(0, idx)[:, 1], "shared": extras["availability"].index_select(0, idx).any(-1)}
    return {"main": main, **paths}, masks
