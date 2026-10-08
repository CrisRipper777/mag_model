from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinatedNCHeads(nn.Module):
    """Train/validation indexed heads for the five 640D coordinated encoders."""

    def __init__(self, variant: str, num_classes: int, *, input_dim: int = 640, init_seed: int | None = None):
        super().__init__()
        self.variant = str(variant).upper()
        if self.variant not in {"N0", "M0", "M1", "P0", "P1"}:
            raise ValueError(f"unsupported coordinated NC head variant: {variant}")
        self.main = nn.Sequential(nn.Dropout(0.2), nn.Linear(input_dim, int(num_classes)))
        self.semantic = nn.ModuleDict()
        if self.variant in {"M1", "P1"}:
            self.semantic = nn.ModuleDict({name: nn.Sequential(nn.Dropout(0.2), nn.Linear(128, int(num_classes))) for name in ("text", "image")})
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                generator = None
                if init_seed is not None:
                    digest = hashlib.blake2b(f"mvcge-coordinated-nc-v2:task:{int(init_seed)}:{name}".encode(), digest_size=8).digest()
                    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest, "little") & ((1 << 63) - 1))
                nn.init.xavier_uniform_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def main_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)

    def semantic_logits(self, p: tuple[torch.Tensor, torch.Tensor], ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.semantic:
            raise RuntimeError("semantic heads are only defined for M1 and P1")
        text, image = (value.index_select(0, ids) for value in p)
        return self.semantic["text"](text), self.semantic["image"](image)


def indexed_main_logits(heads: nn.Module, z: torch.Tensor, node_ids: torch.Tensor, *, split: str, allowed: dict[str, torch.Tensor], counters: dict[str, int]) -> torch.Tensor:
    """Apply the main head only to an explicitly frozen train/validation subset."""
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation evaluation are allowed")
    ids = torch.as_tensor(node_ids, dtype=torch.long, device=z.device).flatten()
    expected = allowed[split].to(device=z.device, dtype=torch.long)
    if ids.numel() != expected.numel() or not torch.equal(torch.sort(ids).values, torch.sort(expected).values):
        raise ValueError(f"node IDs do not match the frozen {split} split")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= z.size(0)):
        raise ValueError("node IDs are out of range")
    counters[split] = int(counters.get(split, 0)) + 1
    return heads.main_logits(z.index_select(0, ids))


def coordinated_loss(variant: str, logits: torch.Tensor, labels: torch.Tensor, *, balance_loss: torch.Tensor, semantic_logits: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    variant = str(variant).upper()
    main_ce = F.cross_entropy(logits, labels, label_smoothing=0.0)
    if variant == "N0":
        balance = main_ce.new_zeros(())
        semantic = main_ce.new_zeros(())
        loss = main_ce
    elif variant in {"M0", "P0"}:
        balance = balance_loss
        semantic = main_ce.new_zeros(())
        loss = main_ce + balance
    else:
        if semantic_logits is None:
            raise ValueError("M1/P1 require two train-index semantic logits")
        balance = balance_loss
        semantic = 0.5 * (F.cross_entropy(semantic_logits[0], labels, label_smoothing=0.0) + F.cross_entropy(semantic_logits[1], labels, label_smoothing=0.0))
        loss = main_ce + balance + 0.05 * semantic
    return loss, {"main_ce": main_ce, "balance_weighted": balance, "semantic_ce_mean": semantic}


def labels_train_val_only(full_labels: torch.Tensor, train_idx: torch.Tensor, val_idx: torch.Tensor) -> torch.Tensor:
    """Copy only the explicitly used labels into a sentinel-filled task tensor."""
    result = torch.full((full_labels.numel(),), -1, dtype=torch.long, device=full_labels.device)
    train = train_idx.to(device=full_labels.device, dtype=torch.long)
    val = val_idx.to(device=full_labels.device, dtype=torch.long)
    result[train] = full_labels.index_select(0, train)
    result[val] = full_labels.index_select(0, val)
    return result
