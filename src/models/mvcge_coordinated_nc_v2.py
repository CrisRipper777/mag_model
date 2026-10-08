from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT = 128
EXPERTS = 4
TOP_K = 2


def _name_seed(seed: int, name: str) -> int:
    digest = hashlib.blake2b(f"mvcge-coordinated-nc-v2:{int(seed)}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _xavier(module: nn.Linear, seed: int, name: str, gain: float = 1.0) -> None:
    generator = torch.Generator(device="cpu").manual_seed(_name_seed(seed, name))
    nn.init.xavier_uniform_(module.weight, gain=gain, generator=generator)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def build_row_mean_operator(edge_index: torch.Tensor, num_nodes: int, *, device=None, dtype=torch.float32):
    """Build R[i,j]=1/deg(i) for each unique directed message j -> i.

    Self loops and duplicate directed coordinates are discarded. ``num_nodes`` is
    explicit so trailing isolated nodes are represented in the sparse shape.
    """
    edge = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge.ndim != 2 or edge.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= num_nodes):
        raise ValueError("edge_index contains a node outside [0,num_nodes)")
    src, dst = edge
    keep = src != dst
    src, dst = src[keep], dst[keep]
    key = dst * int(num_nodes) + src
    if key.numel():
        key = torch.unique(key, sorted=True)
        dst = torch.div(key, int(num_nodes), rounding_mode="floor")
        src = key.remainder(int(num_nodes))
    degree = torch.zeros((num_nodes,), dtype=dtype, device=edge.device)
    if dst.numel():
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=dtype, device=edge.device))
        weight = degree[dst].reciprocal()
    else:
        weight = torch.empty((0,), dtype=dtype, device=edge.device)
    indices = torch.stack((dst, src), dim=0) if dst.numel() else torch.empty((2, 0), dtype=torch.long, device=edge.device)
    op = torch.sparse_coo_tensor(indices, weight, (num_nodes, num_nodes), dtype=dtype, device=edge.device).coalesce()
    return op, degree


def apply_r(operator: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(operator, values)


def sparse_top2(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.size(-1) < 2:
        raise ValueError("router logits must have shape [tokens, experts>=2]")
    values, indices = logits.topk(2, dim=-1)
    gates = torch.zeros_like(logits).scatter(-1, indices, torch.softmax(values, dim=-1))
    return gates, indices


def _cv2(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return values.var(unbiased=False) / (values.mean().square() + 1e-8)


def expected_load(clean_logits: torch.Tensor, noisy_logits: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Differentiable top-2 expected load under independent Gaussian noise."""
    if clean_logits.shape != noisy_logits.shape or clean_logits.shape != sigma.shape:
        raise ValueError("clean/noisy logits and sigma must have matching shapes")
    if clean_logits.size(-1) < 3:
        raise ValueError("expected top-2 load requires at least three experts")
    vals, inds = noisy_logits.topk(3, dim=-1)
    selected = torch.zeros_like(noisy_logits, dtype=torch.bool).scatter_(1, inds[:, :2], True)
    threshold = torch.where(selected, vals[:, 2:3], vals[:, 1:2])
    z = (clean_logits.float() - threshold.float()) / sigma.float().clamp_min(1e-8)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return cdf.sum(dim=0)


def routing_eta(update: int) -> float:
    update = int(update)
    if update <= 50:
        return 0.0
    if update >= 100:
        return 1.0
    return (update - 50) / 50.0


class _ModalityProjector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, LATENT)
        self.norm1 = nn.LayerNorm(LATENT, eps=1e-5)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(LATENT, LATENT)
        self.norm2 = nn.LayerNorm(LATENT, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.linear2(self.dropout(F.gelu(self.norm1(self.linear1(x))))))


class _Expert(nn.Module):
    def __init__(self, pc: bool):
        super().__init__()
        self.w_self = nn.Linear(LATENT, LATENT, bias=False)
        self.w_nbr = nn.Linear(LATENT, LATENT, bias=False)
        self.hidden_bias = nn.Parameter(torch.zeros(LATENT))
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(LATENT, LATENT)
        if pc:
            self.theta0 = nn.Parameter(torch.ones(LATENT))
            self.theta1 = nn.Parameter(torch.zeros(LATENT))
        else:
            self.register_parameter("theta0", None)
            self.register_parameter("theta1", None)

    def forward(self, u: torch.Tensor, n: torch.Tensor, rn: torch.Tensor | None = None) -> torch.Tensor:
        neighbor = self.w_nbr(n)
        if self.theta0 is not None:
            if rn is None:
                raise ValueError("PC expert requires R N")
            phi1 = 0.5 * (n + rn)
            neighbor = neighbor * self.theta0 + self.w_nbr(phi1) * self.theta1
        hidden = self.w_self(u) + neighbor + self.hidden_bias
        return self.out(self.dropout(F.gelu(hidden)))


class _Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(516, 64)
        self.logits = nn.Linear(64, EXPERTS)
        self.noise = nn.Linear(64, EXPERTS)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.hidden(values))
        clean = self.logits(hidden)
        sigma = 0.01 + F.softplus(self.noise(hidden))
        return clean, sigma


class _ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(LATENT, eps=1e-5)
        self.down = nn.Linear(LATENT, 32)
        self.dropout = nn.Dropout(0.1)
        self.up = nn.Linear(32, LATENT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.dropout(F.gelu(self.down(self.norm(x)))))


class _Layer(nn.Module):
    def __init__(self, pc: bool, routed: bool):
        super().__init__()
        self.pre_norm = nn.ModuleList([nn.LayerNorm(LATENT, eps=1e-5) for _ in range(2)])
        self.experts = nn.ModuleList([_Expert(pc) for _ in range(EXPERTS)])
        self.adapters = nn.ModuleList([_ResidualAdapter(), _ResidualAdapter()])
        self.router = _Router() if routed else None
        self.residual_dropout = nn.Dropout(0.1)


class MVCGECoordinatedNCV2(nn.Module):
    """Coordinated graph experts for full graph MAG NC; no labels enter this encoder."""

    out_dim = 640

    def __init__(self, text_dim: int, image_dim: int, variant: str, *, init_seed: int = 0):
        super().__init__()
        self.variant = str(variant).upper()
        if self.variant not in {"N0", "M0", "M1", "P0", "P1"}:
            raise ValueError(f"unsupported coordinated NC variant: {variant}")
        self.text_dim, self.image_dim = int(text_dim), int(image_dim)
        self.pc = self.variant.startswith("P")
        self.auxiliary = self.variant in {"M1", "P1"}
        self.routed = self.variant != "N0"
        self.projectors = nn.ModuleList([_ModalityProjector(self.text_dim), _ModalityProjector(self.image_dim)])
        self.stems = nn.ModuleList([nn.Linear(LATENT, LATENT), nn.Linear(LATENT, LATENT)])
        self.context = nn.Sequential(
            nn.Linear(2 * LATENT, LATENT), nn.GELU(), nn.Linear(LATENT, LATENT), nn.LayerNorm(LATENT, eps=1e-5)
        )
        self.layers = nn.ModuleList([_Layer(self.pc, self.routed), _Layer(self.pc, self.routed)])
        self.delta_norms = nn.ModuleList([nn.LayerNorm(LATENT, eps=1e-5) for _ in range(2)])
        self.register_buffer("_operator", torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,)), (0, 0)).coalesce(), persistent=False)
        self.register_buffer("_degree", torch.empty((0,), dtype=torch.float32), persistent=False)
        self._graph_signature: tuple[Any, ...] | None = None
        self._graph_source: torch.Tensor | None = None
        self._initialize(int(init_seed))

    def _initialize(self, seed: int) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                gain = 0.5 if name.endswith(".out") else (0.01 if name.endswith(".router.logits") else 1.0)
                _xavier(module, seed, name, gain)
                if name.endswith(".router.noise"):
                    nn.init.zeros_(module.weight)
                    nn.init.constant_(module.bias, math.log(math.expm1(0.09)))
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for name, module in self.named_modules():
            if isinstance(module, _ResidualAdapter):
                _xavier(module.up, seed, f"{name}.up", 0.01)
        for stem in self.stems:
            with torch.no_grad():
                stem.weight.copy_(torch.eye(LATENT))
                stem.bias.zero_()

    def set_graph(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        operator, degree = build_row_mean_operator(edge_index, int(num_nodes), device=edge_index.device)
        self._operator = operator
        self._degree = degree
        self._graph_source = edge_index
        self._graph_signature = (id(edge_index), int(getattr(edge_index, "_version", 0)), tuple(edge_index.shape), int(num_nodes), edge_index.device.type, edge_index.device.index)

    def _ensure_graph(self, edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        signature = (id(edge_index), int(getattr(edge_index, "_version", 0)), tuple(edge_index.shape), int(num_nodes), edge_index.device.type, edge_index.device.index)
        if self._graph_signature != signature or self._operator.device != edge_index.device:
            self.set_graph(edge_index, num_nodes)
        return self._operator, self._degree

    @staticmethod
    def _noise(shape: torch.Size, reference: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        return torch.randn(shape, dtype=reference.dtype, device=reference.device, generator=generator)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        update: int = 100,
        noise_generator: torch.Generator | None = None,
        modality_available: torch.Tensor | None = None,
        return_details: bool = False,
        diagnostic_noise: bool = False,
    ):
        if x.ndim != 2 or x.size(1) != self.text_dim + self.image_dim:
            raise ValueError("x must concatenate the configured text then image features")
        n_nodes = int(x.size(0))
        if modality_available is not None:
            mask = torch.as_tensor(modality_available, dtype=torch.bool, device=x.device)
            if mask.shape != (n_nodes, 2) or not bool(mask.all()):
                raise ValueError("this study supports full modalities only; availability must be all true")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("full-modal input features must be finite")
        operator, degree = self._ensure_graph(edge_index, n_nodes)
        raw = (x[:, :self.text_dim], x[:, self.text_dim:])
        p = tuple(self.projectors[m](raw[m]) for m in range(2))
        h0 = tuple(self.stems[m](p[m]) for m in range(2))
        h = h0
        c0 = self.context(torch.cat(p, dim=-1))
        balance_layers: list[torch.Tensor] = []
        layer_details: list[dict[str, Any]] = []
        eta = routing_eta(update) if self.training and self.routed else 1.0
        for layer_idx, layer in enumerate(self.layers):
            u = tuple(layer.pre_norm[m](h[m]) for m in range(2))
            neigh = tuple(apply_r(operator, u[m]) for m in range(2))
            context = self.context(torch.cat(u, dim=-1))
            rneigh = tuple(apply_r(operator, neigh[m]) for m in range(2)) if self.pc else (None, None)
            gates: list[torch.Tensor] = []
            route_rows: list[dict[str, torch.Tensor]] = []
            if self.routed:
                global_means = tuple(u[m].mean(0, keepdim=True) for m in range(2))
                max_log_degree = torch.log1p(degree).max().clamp_min(1.0)
                scaled_degree = (torch.log1p(degree) / max_log_degree).unsqueeze(-1)
                for m in range(2):
                    local_scatter = (apply_r(operator, u[m].square().mean(-1, keepdim=True)) - neigh[m].square().mean(-1, keepdim=True)).clamp_min(0)
                    local_scatter = torch.where(degree[:, None] > 0, local_scatter, torch.zeros_like(local_scatter))
                    onehot = F.one_hot(torch.full((n_nodes,), m, device=x.device, dtype=torch.long), num_classes=2).to(x.dtype)
                    router_input = torch.cat((u[m], neigh[m], context, global_means[m].expand(n_nodes, -1), scaled_degree, torch.log1p(local_scatter), onehot), dim=-1)
                    clean, sigma = layer.router(router_input)
                    noisy = clean
                    if self.training or diagnostic_noise:
                        noisy = clean + sigma * self._noise(clean.shape, clean, noise_generator)
                    dense = torch.softmax(noisy, dim=-1)
                    sparse, top_indices = sparse_top2(noisy)
                    if self.training:
                        gate = (1.0 - eta) * dense + eta * sparse
                    else:
                        gate = sparse
                    gates.append(gate)
                    load = expected_load(clean, noisy, sigma)
                    route_rows.append({"clean_logits": clean, "noisy_logits": noisy, "sigma": sigma, "gates": gate, "dense_gates": dense, "top2_indices": top_indices, "expected_load": load})
                importance = torch.stack(gates, 0).sum(0).sum(0)
                balance = _cv2(importance)
                if eta > 0:
                    expected = route_rows[0]["expected_load"] + route_rows[1]["expected_load"]
                    balance = balance + eta * _cv2(expected)
                balance_layers.append(balance)
            else:
                gates = [u[m].new_full((n_nodes, EXPERTS), 1.0 / EXPERTS) for m in range(2)]
                route_rows = [{"gates": gate} for gate in gates]

            # Every modality/expert output uses the same layer input u; states update together.
            outputs_by_modality = []
            for m in range(2):
                expert_outputs = torch.stack([expert(u[m], neigh[m], rneigh[m]) for expert in layer.experts], dim=1)
                mixed = (expert_outputs * gates[m].unsqueeze(-1)).sum(1)
                outputs_by_modality.append(mixed)
            adapted = tuple(layer.adapters[m](outputs_by_modality[m]) for m in range(2))
            h_next = tuple(h[m] + layer.residual_dropout(adapted[m]) for m in range(2))
            h = h_next
            if return_details or diagnostic_noise:
                layer_details.append({"u": u, "neighbor": neigh, "context": context, "gates": tuple(gates), "routes": route_rows, "mixed": tuple(outputs_by_modality), "adapter_residual": tuple(adapted[m] - outputs_by_modality[m] for m in range(2)), "next": h})

        delta = tuple(h[m] - h0[m] for m in range(2))
        g = tuple(self.delta_norms[m](delta[m]) for m in range(2))
        z = torch.cat((p[0], p[1], c0, g[0], g[1]), dim=-1)
        if z.size(-1) != self.out_dim:
            raise RuntimeError(f"readout must be {self.out_dim}D, got {z.size(-1)}")
        weighted_balance = 0.01 * torch.stack(balance_layers).mean() if balance_layers else z.new_zeros(())
        details = {
            "p": p, "h0": h0, "delta": delta, "g": g, "c0": c0,
            "balance_loss": weighted_balance, "raw_balance": torch.stack(balance_layers).mean() if balance_layers else z.new_zeros(()),
            "eta": eta, "layers": layer_details,
        }
        return (z, details) if return_details else z

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, *, update: int = 100, noise_generator=None, modality_available=None):
        z, details = self.encode(x, edge_index, update=update, noise_generator=noise_generator, modality_available=modality_available, return_details=True)
        return z, None, None, details["balance_loss"], details

    def parameter_groups(self, lr: float, weight_decay: float):
        groups: list[dict[str, Any]] = []
        router_ids = {id(p) for layer in self.layers if layer.router is not None for p in layer.router.parameters()}
        theta_ids = {id(p) for layer in self.layers for expert in layer.experts for p in (expert.theta0, expert.theta1) if p is not None}
        norm_ids = {id(p) for module in self.modules() if isinstance(module, nn.LayerNorm) for p in module.parameters(recurse=False)}
        decay: dict[str, list[nn.Parameter]] = {"main": [], "router": [], "theta": []}
        no_decay: dict[str, list[nn.Parameter]] = {"main": [], "router": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in theta_ids:
                decay["theta"].append(parameter)
                continue
            bucket = "router" if id(parameter) in router_ids else "main"
            if name.endswith("bias") or id(parameter) in norm_ids:
                no_decay[bucket].append(parameter)
            else:
                decay[bucket].append(parameter)
        for key, params in decay.items():
            if params:
                groups.append({"params": params, "lr": float(lr) * (0.5 if key == "router" else 1.0), "weight_decay": 0.001 if key == "theta" else float(weight_decay), "name": key + "_decay"})
        for key, params in no_decay.items():
            if params:
                groups.append({"params": params, "lr": float(lr) * (0.5 if key == "router" else 1.0), "weight_decay": 0.0, "name": key + "_no_decay"})
        flat = [id(p) for group in groups for p in group["params"]]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(flat) != len(set(flat)) or set(flat) != expected:
            raise RuntimeError("encoder optimizer groups have duplicate or missing parameters")
        return groups


class Model(MVCGECoordinatedNCV2):
    """Repository factory-compatible constructor: Model(cfg, data_info)."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        variant = str(cfg.model.get("variant", "M0"))
        super().__init__(int(data_info["text_dim"]), int(data_info["visual_dim"]), variant, init_seed=int(getattr(cfg, "seed", 0)))
