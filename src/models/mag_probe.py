from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = (
    "T_E", "I_E", "T_M", "I_M", "J_E", "J_EARLY", "J_N", "J_FIXED",
    "J_GLOBAL", "J_NODE", "J_NODE_SHUFFLE", "J_GLOBAL_WIDE", "J_CONCAT",
)
ROUTED = {"J_NODE", "J_NODE_SHUFFLE"}
PATH_VARIANTS = {"J_FIXED", "J_GLOBAL", "J_NODE", "J_NODE_SHUFFLE", "J_GLOBAL_WIDE", "J_CONCAT"}


def resolve_capacity_widths(num_classes: int, text_dim: int, image_dim: int,
                            hidden_dim: int = 128, out_dim: int = 128) -> dict[str, int | float]:
    """Choose width-only capacity controls using complete model + NC head counts."""
    def core_params(h: int) -> int:
        # Adapters (with bias) plus shared Linear-LayerNorm-GELU-Dropout-Linear.
        return (text_dim * out_dim + out_dim) + (image_dim * out_dim + out_dim) + (out_dim * h + h) + 2 * h + (h * out_dim + out_dim)

    node_total = core_params(hidden_dim) + 16676 + out_dim * num_classes + num_classes
    global_total_base = core_params(hidden_dim) + 4 + out_dim * num_classes + num_classes
    wide_candidates = range(128, 513)
    wide_h = min(wide_candidates, key=lambda h: (abs(core_params(h) + 4 + out_dim * num_classes + num_classes - node_total), h))
    concat_candidates = range(1, 129)
    concat_b = min(concat_candidates, key=lambda b: (abs(core_params(hidden_dim) + 512 * b + b + b * out_dim + out_dim + out_dim * num_classes + num_classes - node_total), b))
    wide_total = core_params(wide_h) + 4 + out_dim * num_classes + num_classes
    concat_total = core_params(hidden_dim) + 512 * concat_b + concat_b + concat_b * out_dim + out_dim + out_dim * num_classes + num_classes
    return {
        "node_total": node_total,
        "wide_hidden_dim": wide_h,
        "wide_total": wide_total,
        "wide_relative_error": abs(wide_total - node_total) / node_total,
        "concat_bottleneck": concat_b,
        "concat_total": concat_total,
        "concat_relative_error": abs(concat_total - node_total) / node_total,
        "global_base_total": global_total_base,
    }


class _SharedPhi(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Model(nn.Module):
    """Full-graph NC probe with fixed one-hop receiver mean aggregation."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        self.variant = str(cfg.model.get("variant", "J_FIXED")).upper()
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown mag_probe variant {self.variant!r}; expected one of {VARIANTS}")
        self.out_dim = int(cfg.model.get("out_dim", 128))
        self.hidden_dim = int(cfg.model.get("hidden_dim", 128))
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.text_dim = int(data_info["text_dim"])
        self.image_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.image_dim <= 0:
            raise ValueError("mag_probe requires aligned text and image features")

        use_text = self.variant.startswith("T_") or self.variant in PATH_VARIANTS or self.variant in {"J_E", "J_EARLY", "J_N"}
        use_image = self.variant.startswith("I_") or self.variant in PATH_VARIANTS or self.variant in {"J_E", "J_EARLY", "J_N"}
        self.use_text = use_text
        self.use_image = use_image
        self.adapter_t = nn.Linear(self.text_dim, self.out_dim) if use_text else None
        self.adapter_i = nn.Linear(self.image_dim, self.out_dim) if use_image else None

        num_classes = int(data_info["num_classes"])
        widths = resolve_capacity_widths(num_classes, self.text_dim, self.image_dim,
                                         self.hidden_dim, self.out_dim)
        h = int(widths["wide_hidden_dim"]) if self.variant == "J_GLOBAL_WIDE" else self.hidden_dim
        if self.variant == "J_GLOBAL_WIDE":
            self.hidden_dim = h
        self.concat_bottleneck = int(widths["concat_bottleneck"])
        self.capacity_match = widths
        self.phi = _SharedPhi(h, self.out_dim, self.dropout)

        if self.variant in {"J_GLOBAL", "J_GLOBAL_WIDE"}:
            self.global_logits = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("global_logits", None)
        if self.variant in ROUTED:
            self.router = nn.Sequential(
                nn.Linear(516, 32), nn.GELU(), nn.Linear(32, 4)
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        else:
            self.router = None
        if self.variant == "J_CONCAT":
            b = self.concat_bottleneck
            self.concat = nn.Sequential(nn.Linear(512, b), nn.GELU(), nn.Linear(b, self.out_dim))
        else:
            self.concat = None

        self._operator_cache: dict[str, Any] | None = None
        self._run_seed = int(cfg.seed)
        self._epoch = 0
        self._custom_context_permutation: torch.Tensor | None = None
        self._last_operator_rebuilds = 0

    def set_run_seed(self, seed: int) -> None:
        self._run_seed = int(seed)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._custom_context_permutation = None

    def set_context_permutation(self, permutation: torch.Tensor | None) -> None:
        self._custom_context_permutation = None if permutation is None else permutation.detach().cpu().long()

    @property
    def operator_rebuilds(self) -> int:
        return self._last_operator_rebuilds

    @staticmethod
    def _signature(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> tuple[Any, ...]:
        return (id(edge_index), edge_index.data_ptr(), int(getattr(edge_index, "_version", 0)),
                tuple(edge_index.shape), num_nodes, edge_index.device.type,
                edge_index.device.index, str(dtype))

    def _operator(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        signature = self._signature(edge_index, num_nodes, dtype)
        if self._operator_cache is not None and self._operator_cache["signature"] == signature:
            p = self._operator_cache["p"]
            degree = torch.bincount(p.indices()[0], minlength=num_nodes)
            return p, degree, degree == 0

        edge = edge_index.to(dtype=torch.long)
        if edge.ndim != 2 or edge.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= num_nodes):
            raise ValueError("edge_index contains node IDs outside the feature matrix")
        src, dst = edge[0], edge[1]
        keep = src != dst
        src, dst = src[keep], dst[keep]
        pairs = torch.stack((dst, src), dim=1)
        if pairs.numel():
            pairs = torch.unique(pairs, dim=0, sorted=True)
            row, col = pairs[:, 0], pairs[:, 1]
            degree = torch.bincount(row, minlength=num_nodes)
            values = degree[row].to(dtype=dtype).reciprocal()
            indices = torch.stack((row, col), dim=0)
        else:
            degree = torch.zeros(num_nodes, dtype=torch.long, device=edge.device)
            indices = torch.empty((2, 0), dtype=torch.long, device=edge.device)
            values = torch.empty((0,), dtype=dtype, device=edge.device)
        p = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes),
                                    dtype=dtype, device=edge.device).coalesce()
        isolated = degree == 0
        self._operator_cache = {"signature": signature, "p": p}
        self._last_operator_rebuilds += 1
        return p, degree, isolated

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1, eps=1e-8)

    @staticmethod
    def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        an, bn = a.norm(dim=1), b.norm(dim=1)
        den = an * bn
        dot = (a * b).sum(dim=1)
        return torch.where(den > 0, dot / den.clamp_min(1e-12), torch.zeros_like(den))

    @staticmethod
    def _weighted_active(paths: list[torch.Tensor], active: torch.Tensor,
                         weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # paths are [N,D] in fixed order, active is [N,K].
        if weights.ndim == 1:
            weighted = active.to(paths[0].dtype) * weights.unsqueeze(0)
        else:
            weighted = active.to(paths[0].dtype) * weights
        denominator = weighted.sum(dim=1, keepdim=True)
        normalized = torch.where(denominator > 0, weighted / denominator.clamp_min(1e-12),
                                torch.zeros_like(weighted))
        z = sum(path * normalized[:, i:i + 1] for i, path in enumerate(paths))
        return z, normalized

    def _shuffle_permutation(self, n: int) -> torch.Tensor:
        if self._custom_context_permutation is not None:
            p = self._custom_context_permutation
            if p.numel() != n or not torch.equal(torch.sort(p).values, torch.arange(n)):
                raise ValueError("custom context permutation must be a permutation of every node row")
            return p
        namespace = "train" if self.training else "validation_test"
        epoch = self._epoch if self.training else 0
        digest = hashlib.sha256(f"{self._run_seed}:{namespace}:{epoch}:mag_probe_context_v1".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randperm(n, generator=generator)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        if x.ndim != 2 or x.size(1) != self.text_dim + self.image_dim:
            raise ValueError(f"expected x shaped [N,{self.text_dim + self.image_dim}], got {tuple(x.shape)}")
        n = x.size(0)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)
        if self.variant in {"T_E", "I_E", "J_E"}:
            p, degree, isolated = None, None, None
        else:
            p, degree, isolated = self._operator(edge_index, n, x.dtype)
        xt = self._normalize(x[:, :self.text_dim])
        xi = self._normalize(x[:, self.text_dim:])

        at = self.adapter_t(xt) if self.adapter_t is not None else None
        ai = self.adapter_i(xi) if self.adapter_i is not None else None
        if self.variant == "J_EARLY":
            ea = self.phi((at + ai) * 0.5)
            na = torch.sparse.mm(p, ea)
            active = torch.stack((torch.ones(n, dtype=torch.bool, device=x.device), ~isolated), dim=1)
            z, _ = self._weighted_active([ea, na], active, torch.ones(2, device=x.device))
            return z, None, None, z.new_zeros(()), {}

        et = self.phi(at) if at is not None else None
        ei = self.phi(ai) if ai is not None else None
        if self.variant == "T_E":
            return et, None, None, et.new_zeros(()), {}
        if self.variant == "I_E":
            return ei, None, None, ei.new_zeros(()), {}
        if self.variant == "J_E":
            z = (et + ei) * 0.5
            return z, None, None, z.new_zeros(()), {}
        nt = torch.sparse.mm(p, et) if et is not None else None
        ni = torch.sparse.mm(p, ei) if ei is not None else None

        if self.variant in {"T_M", "I_M"}:
            ego, neigh = (et, nt) if self.variant == "T_M" else (ei, ni)
            active = torch.stack((torch.ones(n, dtype=torch.bool, device=x.device), ~isolated), dim=1)
            z, _ = self._weighted_active([ego, neigh], active, torch.ones(2, device=x.device))
            return z, None, None, z.new_zeros(()), {}

        paths = [et, nt, ei, ni]
        values = [v for v in paths]
        active = torch.stack((torch.ones(n, dtype=torch.bool, device=x.device), ~isolated,
                              torch.ones(n, dtype=torch.bool, device=x.device), ~isolated), dim=1)
        if self.variant == "J_N":
            active_n = torch.stack((~isolated, ~isolated), dim=1)
            z, _ = self._weighted_active([nt, ni], active_n, torch.ones(2, device=x.device))
            return z, None, None, z.new_zeros(()), {}
        if self.variant == "J_FIXED":
            z, _ = self._weighted_active(values, active, torch.ones(4, device=x.device))
            return z, None, None, z.new_zeros(()), {}
        if self.variant in {"J_GLOBAL", "J_GLOBAL_WIDE"}:
            weights = self.global_logits.softmax(dim=0)
            z, normalized = self._weighted_active(values, active, weights)
            aux = {"fusion_weights": normalized.detach()} if not self.training else {}
            return z, None, None, z.new_zeros(()), aux

        if self.variant in ROUTED:
            structural = torch.stack((degree.to(x.dtype).log1p(), isolated.to(x.dtype),
                                      self._safe_cosine(et, nt), self._safe_cosine(ei, ni)), dim=1)
            context = torch.cat((*values, structural), dim=1)
            if self.variant == "J_NODE_SHUFFLE":
                context = context[self._shuffle_permutation(n).to(context.device)]
            logits = self.router(context)
            weights = logits.softmax(dim=1)
            z, normalized = self._weighted_active(values, active, weights)
            aux = {"router_weights": normalized.detach(), "activity_mask": active.detach(),
                   "degree": degree.detach()} if not self.training else {}
            return z, None, None, z.new_zeros(()), aux

        if self.variant == "J_CONCAT":
            z = self.concat(torch.cat(values, dim=1))
            return z, None, None, z.new_zeros(()), {}
        raise AssertionError(self.variant)

    @torch.no_grad()
    def inference(self, x: torch.Tensor, edge_index: torch.Tensor | None,
                  device: torch.device, batch_size: int) -> torch.Tensor:
        # The operator and all node contexts are full graph; inference must match forward.
        self.eval()
        result, _, _, _, _ = self(x.to(device), edge_index.to(device) if edge_index is not None else None)
        return result.detach().cpu()
