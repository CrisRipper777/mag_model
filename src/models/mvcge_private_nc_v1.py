from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphOperator:
    """Fixed sparse normalized adjacency and observable-node mask for one modality."""

    p: torch.Tensor
    mask: torch.Tensor
    degree: torch.Tensor

    def apply_p(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.p, values)

    def apply_t(self, values: torch.Tensor) -> torch.Tensor:
        return self.mask.to(values.dtype).unsqueeze(-1) * values - self.apply_p(values)


def build_graph_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    mask: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> GraphOperator:
    """Construct P=D^-1/2(A+I)D^-1/2 from a canonical, unweighted MAGB edge list.

    The input edge list is already made undirected by the canonical loader. Existing
    loops are removed, duplicate input pairs are coalesced, and one unit loop is added
    per observed node. If optional weights are supplied, duplicate weights are summed.
    """
    device = edge_index.device
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(num_nodes)
    src, dst = edge_index.long()
    keep = (src != dst) & mask[src] & mask[dst]
    src, dst = src[keep], dst[keep]
    if edge_weight is None:
        weights = torch.ones(src.numel(), dtype=torch.float32, device=device)
    else:
        weights = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)[keep]
        if not torch.isfinite(weights).all() or bool((weights < 0).any()):
            raise ValueError("graph weights must be finite and nonnegative")

    # Coalesce repeated directed coordinates by summing, matching sparse adjacency
    # semantics; canonical MAGB input has already removed duplicates and has unit weights.
    key = dst * int(num_nodes) + src
    if key.numel():
        unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        if edge_weight is None:
            summed = weights.new_ones(unique_key.numel())
        else:
            summed = weights.new_zeros(unique_key.numel())
            summed.index_add_(0, inverse, weights)
        dst, src = torch.div(unique_key, num_nodes, rounding_mode="floor"), unique_key.remainder(num_nodes)
        weights = summed

    observed = torch.nonzero(mask, as_tuple=False).flatten()
    src = torch.cat((src, observed))
    dst = torch.cat((dst, observed))
    weights = torch.cat((weights, torch.ones(observed.numel(), dtype=torch.float32, device=device)))
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, dst, weights)
    inv_sqrt = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    norm_weight = inv_sqrt[dst] * weights * inv_sqrt[src]
    indices = torch.stack((dst, src), dim=0)
    p = torch.sparse_coo_tensor(indices, norm_weight, (num_nodes, num_nodes), device=device).coalesce()
    return GraphOperator(p=p, mask=mask, degree=degree)


def pc_coefficients(max_order: int = 3, bank_size: int = 3, t: float = 0.5) -> torch.Tensor:
    """Return C_n(k,t), including the requested maximum order."""
    coeff = torch.zeros((bank_size + 1, max_order + 1), dtype=torch.float64)
    coeff[:, 0] = 1.0
    for k in range(1, bank_size + 1):
        coeff[k, 1] = k - t
        for n in range(2, max_order + 1):
            coeff[k, n] = (k - n - t + 1) * coeff[k, n - 1] - (n - 1) * t * coeff[k, n - 2]
    return coeff


def sparse_pc_basis(operator: GraphOperator, values: torch.Tensor, max_order: int = 3) -> list[torch.Tensor]:
    basis = [operator.mask.to(values.dtype).unsqueeze(-1) * values]
    for n in range(1, max_order + 1):
        basis.append(-operator.apply_t(basis[-1]) / float(n))
    return basis


def _name_seed(seed: int, name: str) -> int:
    digest = hashlib.blake2b(f"mvcge-v1:{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


class _Expert(nn.Module):
    def __init__(self, dim: int, max_order: int, bank_size: int, coeff: torch.Tensor):
        super().__init__()
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.theta = nn.Parameter(torch.zeros(bank_size + 1, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(0.2)
        self.w_out = nn.Linear(dim, dim)
        self.register_buffer("coeff", coeff.float(), persistent=True)
        self.max_order = max_order

    def initialize_theta(self, t_coefficients: torch.Tensor) -> None:
        # theta @ C gives coefficients over B_n=(-T)^n/n!, so convert the target
        # T^n coefficients before solving C^T theta = target_gamma.
        n = torch.arange(t_coefficients.numel(), dtype=torch.float64)
        gamma_target = t_coefficients.double() * torch.pow(-torch.ones_like(n), n) * torch.exp(torch.lgamma(n + 1.0))
        theta = torch.linalg.solve(self.coeff.double().T, gamma_target)
        with torch.no_grad():
            self.theta.copy_(theta.float().unsqueeze(-1).expand_as(self.theta))

    def gamma(self) -> torch.Tensor:
        return self.theta[0].unsqueeze(0) * F.one_hot(
            torch.tensor(0, device=self.theta.device), self.max_order + 1
        ).to(self.theta.dtype).unsqueeze(-1) + torch.einsum("kd,kn->nd", self.theta[1:], self.coeff[1:].to(self.theta.dtype))

    def forward(self, basis: list[torch.Tensor]) -> torch.Tensor:
        gamma = self.gamma()
        filtered = None
        for order, values in enumerate(basis):
            term = self.w_in(values) * gamma[order]
            filtered = term if filtered is None else filtered + term
        hidden = F.gelu(filtered + self.bias)
        return self.w_out(self.dropout(hidden))


class _GraphLayer(nn.Module):
    def __init__(self, dim: int, experts: int, private: bool, coeff: torch.Tensor, max_order: int = 3, bank_size: int = 3):
        super().__init__()
        self.pre_norm = nn.ModuleList([nn.LayerNorm(dim, eps=1e-5) for _ in range(2)])
        self.experts = nn.ModuleList([_Expert(dim, max_order, bank_size, coeff) for _ in range(experts)])
        self.private = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, dim))
            for _ in range(2)
        ]) if private else None
        self.residual_dropout = nn.Dropout(0.1)
        self.expert_count = experts
        self.max_order = max_order


class _Router(nn.Module):
    def __init__(self, in_dim: int, hidden: int, experts: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-5)
        self.hidden = nn.Linear(in_dim, hidden)
        self.logit_head = nn.Linear(hidden, experts)
        self.noise_head = nn.Linear(hidden, experts)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.hidden(self.norm(values)))
        return self.logit_head(hidden), self.noise_head(hidden)


def _cv2(values: torch.Tensor) -> torch.Tensor:
    return values.var(unbiased=False) / (values.mean().square() + 1e-8)


def sparse_top2(logits: torch.Tensor, *, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    selected, indices = logits.topk(k, dim=-1)
    weights = torch.softmax(selected, dim=-1)
    dense = torch.zeros_like(logits).scatter(-1, indices, weights)
    return dense, indices


def expected_load(clean_logits: torch.Tensor, noisy_logits: torch.Tensor, sigma: torch.Tensor, valid: torch.Tensor, k: int = 2) -> torch.Tensor:
    if clean_logits.shape != noisy_logits.shape or clean_logits.shape != sigma.shape:
        raise ValueError("logits and sigma must have matching [tokens, experts] shapes")
    top_values, top_indices = noisy_logits.topk(k + 1, dim=-1)
    selected = torch.zeros_like(noisy_logits, dtype=torch.bool).scatter_(1, top_indices[:, :k], True)
    threshold = torch.where(selected, top_values[:, k:k + 1], top_values[:, k - 1:k])
    z = (clean_logits - threshold) / sigma.clamp_min(1e-8)
    probability = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return (probability * valid.to(probability.dtype).unsqueeze(-1)).sum(dim=0)


class MVCGEPrivateNC(nn.Module):
    """Private-modal adapters + shared PC experts + contextual sparse routing."""

    out_dim = 256
    latent_dim = 128
    expert_count = 4
    top_k = 2
    max_order = 3
    bank_size = 3

    def __init__(self, text_dim: int, image_dim: int, variant: str, init_seed: int = 0):
        super().__init__()
        self.variant = variant.upper()
        if self.variant not in {"G0", "G1", "U1"}:
            raise ValueError(f"unsupported MVCGE variant: {variant}")
        self.private_enabled = self.variant in {"G1", "U1"}
        self.router_enabled = self.variant in {"G0", "G1"}
        self.text_dim, self.image_dim = int(text_dim), int(image_dim)
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(self.text_dim, 128), nn.LayerNorm(128, eps=1e-5), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 128)),
            nn.Sequential(nn.Linear(self.image_dim, 128), nn.LayerNorm(128, eps=1e-5), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 128)),
        ])
        coeff64 = pc_coefficients(self.max_order, self.bank_size)
        self.register_buffer("pc_coefficients", coeff64.float(), persistent=True)
        self.register_buffer("pc_to_t_matrix", coeff64.T.float(), persistent=True)
        self.layers = nn.ModuleList([
            _GraphLayer(128, self.expert_count, self.private_enabled, coeff64, self.max_order, self.bank_size)
            for _ in range(2)
        ])
        if self.router_enabled:
            self.context_mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(258, 128), nn.GELU(), nn.Linear(128, 128), nn.LayerNorm(128, eps=1e-5))
                for _ in range(2)
            ])
            self.routers = nn.ModuleList([_Router(523, 64, self.expert_count) for _ in range(2)])
            self.type_embedding = nn.Parameter(torch.empty(2, 8))
        else:
            self.context_mlps = None
            self.routers = None
            self.register_parameter("type_embedding", None)
        self.prior_norms = nn.ModuleList([nn.LayerNorm(128, eps=1e-5) for _ in range(2)])
        self.state_norms = nn.ModuleList([nn.LayerNorm(128, eps=1e-5) for _ in range(2)])
        self.readout_norm = nn.LayerNorm(256, eps=1e-5)
        self.readout1 = nn.Linear(514, 256)
        self.readout_dropout = nn.Dropout(0.3)
        self.readout2 = nn.Linear(256, 256)
        self._graph_ops: list[GraphOperator] | None = None
        self._degree_scalar: torch.Tensor | None = None
        self._initialize(init_seed)

    def _initialize(self, seed: int) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                gain = 0.5 if name.endswith("w_out") or (".private." in name and name.endswith(".3")) else 1.0
                if name.endswith("logit_head"):
                    gain = 0.01
                generator = torch.Generator(device="cpu").manual_seed(_name_seed(seed, name))
                nn.init.xavier_uniform_(module.weight, gain=gain, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if self.router_enabled:
            with torch.no_grad():
                self.type_embedding.normal_(mean=0.0, std=0.02, generator=torch.Generator().manual_seed(_name_seed(seed, "type_embedding")))
                for router in self.routers:
                    router.noise_head.weight.zero_()
                    router.noise_head.bias.fill_(math.log(math.expm1(0.09)))
        t_responses = torch.tensor([
            [1.0, -0.1, 0.0, 0.0],
            [1.0, -0.5, 0.0, 0.0],
            [1.0, -1.2, 0.9, -0.2],
            [1.0, 0.1, 0.0, 0.0],
        ], dtype=torch.float64)
        for layer in self.layers:
            for index, expert in enumerate(layer.experts):
                expert.initialize_theta(t_responses[index])

    def set_graph(self, edge_index: torch.Tensor, modality_mask: torch.Tensor | None = None) -> None:
        n = int(modality_mask.size(0)) if modality_mask is not None else int(edge_index.max().item()) + 1
        if modality_mask is None:
            modality_mask = torch.ones((n, 2), dtype=torch.bool, device=edge_index.device)
        if tuple(modality_mask.shape) != (n, 2):
            raise ValueError("modality mask must have shape [N,2]")
        masks = [modality_mask[:, i].bool() for i in range(2)]
        if torch.equal(masks[0], masks[1]):
            shared = build_graph_operator(edge_index, n, masks[0])
            self._graph_ops = [shared, shared]
        else:
            self._graph_ops = [build_graph_operator(edge_index, n, masks[i]) for i in range(2)]
        src, dst = edge_index.long()
        nonloop = src != dst
        key = dst[nonloop] * n + src[nonloop]
        unique = torch.unique(key)
        dst_unique = torch.div(unique, n, rounding_mode="floor")
        degree = torch.zeros(n, dtype=torch.float32, device=edge_index.device)
        degree.index_add_(0, dst_unique, torch.ones_like(dst_unique, dtype=degree.dtype))
        raw = torch.log1p(degree)
        self._degree_scalar = raw / raw.max().clamp_min(1.0)

    def _router_inputs(self, states: tuple[torch.Tensor, torch.Tensor], bases: list[list[torch.Tensor]], masks: torch.Tensor, layer_index: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        ops = self._graph_ops
        assert ops is not None and self.context_mlps is not None and self.type_embedding is not None and self._degree_scalar is not None
        u_text, u_image = states
        entity_context = self.context_mlps[layer_index](torch.cat((u_text, u_image, masks.to(u_text.dtype)), dim=-1))
        inputs: list[torch.Tensor] = []
        for modality, u in enumerate(states):
            local = u + bases[modality][1]
            observed = masks[:, modality].to(u.dtype)
            graph_mean = (u * observed[:, None]).sum(dim=0) / observed.sum().clamp_min(1.0)
            graph_mean = torch.where(observed.sum() > 0, graph_mean, torch.zeros_like(graph_mean)).expand_as(u)
            type_embed = self.type_embedding[modality].expand(u.size(0), -1)
            q = torch.cat((u, local, entity_context, graph_mean, self._degree_scalar[:, None], masks.to(u.dtype), type_embed), dim=-1)
            inputs.append(q)
        return inputs, entity_context

    def forward(
        self,
        x_text: torch.Tensor,
        x_image: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
        *,
        step: int = 1,
        stochastic_routing: bool | None = None,
        noise_generator: torch.Generator | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._graph_ops is None:
            raise RuntimeError("set_graph(edge_index, modality_mask) must be called before forward")
        if x_text.ndim != 2 or x_image.ndim != 2 or x_text.size(0) != x_image.size(0):
            raise ValueError("text and image inputs must have shapes [N,D] and share node order")
        if x_text.size(1) != self.text_dim or x_image.size(1) != self.image_dim:
            raise ValueError("input modality widths do not match the model")
        masks = torch.ones((x_text.size(0), 2), dtype=torch.bool, device=x_text.device) if modality_mask is None else modality_mask.to(device=x_text.device, dtype=torch.bool)
        if tuple(masks.shape) != (x_text.size(0), 2):
            raise ValueError("modality_mask must have shape [N,2]")
        if not torch.isfinite(x_text).all() or not torch.isfinite(x_image).all():
            raise ValueError("canonical feature inputs must be finite")
        raw_inputs = (x_text.float(), x_image.float())
        states = tuple(self.adapters[m](raw_inputs[m]) * masks[:, m:m + 1].float() for m in range(2))
        own_semantics = states
        details: dict[str, Any] = {"layers": [], "balance_loss": x_text.new_zeros((), dtype=torch.float32)}
        if stochastic_routing is None:
            stochastic_routing = self.training
        for layer_index, layer in enumerate(self.layers):
            old_states = states
            normalized = tuple(layer.pre_norm[m](old_states[m]) * masks[:, m:m + 1].float() for m in range(2))
            bases = [sparse_pc_basis(self._graph_ops[m], normalized[m], self.max_order) for m in range(2)]
            if self.router_enabled:
                q_values, _ = self._router_inputs(normalized, bases, masks, layer_index)
                route_out = [self.routers[layer_index](q) for q in q_values]
                clean_logits = torch.stack([item[0] for item in route_out], dim=1)
                sigma = 0.01 + F.softplus(torch.stack([item[1] for item in route_out], dim=1))
                if stochastic_routing:
                    epsilon = torch.randn(clean_logits.shape, device=clean_logits.device, dtype=clean_logits.dtype, generator=noise_generator)
                    noisy_logits = clean_logits + sigma * epsilon
                else:
                    noisy_logits = clean_logits
                if self.training and int(step) <= 20:
                    gates = torch.softmax(noisy_logits, dim=-1)
                    selected = None
                else:
                    gates, selected = sparse_top2(noisy_logits if stochastic_routing else clean_logits)
                valid = masks.bool()
                pooled_gates = gates[valid]
                importance = pooled_gates.sum(0) if pooled_gates.numel() else gates.new_zeros(self.expert_count)
                importance_loss = _cv2(importance)
                if self.training and int(step) > 20:
                    clean_valid = clean_logits[valid]
                    noisy_valid = noisy_logits[valid]
                    sigma_valid = sigma[valid]
                    load = expected_load(clean_valid, noisy_valid, sigma_valid, torch.ones(clean_valid.size(0), dtype=torch.bool, device=x_text.device), self.top_k)
                    load_loss = _cv2(load)
                    balance = importance_loss + load_loss
                else:
                    load = None
                    load_loss = importance_loss.new_zeros(())
                    balance = importance_loss
                details_layer: dict[str, Any] = {
                    "clean_logits": clean_logits, "noisy_logits": noisy_logits, "sigma": sigma,
                    "gates": gates, "selected": selected, "importance": importance,
                    "expected_load": load, "importance_loss": importance_loss,
                    "load_loss": load_loss, "balance_loss": balance,
                }
                details["balance_loss"] = details["balance_loss"] + balance / len(self.layers)
            else:
                gates = torch.full((x_text.size(0), 2, self.expert_count), 1.0 / self.expert_count, device=x_text.device, dtype=torch.float32)
                details_layer = {"gates": gates, "clean_logits": None, "noisy_logits": None, "sigma": None, "selected": None, "importance": None, "expected_load": None, "importance_loss": None, "load_loss": None, "balance_loss": None}

            outputs: list[torch.Tensor] = []
            public_outputs: list[torch.Tensor] = []
            private_outputs: list[torch.Tensor] = []
            for modality in range(2):
                combined = normalized[modality].new_zeros(normalized[modality].shape)
                # Correct receiver-node gating requires every expert's full graph output.
                for expert_index, expert in enumerate(layer.experts):
                    output = expert(bases[modality])
                    combined = combined + gates[:, modality, expert_index:expert_index + 1] * output
                private = layer.private[modality](normalized[modality]) if layer.private is not None else torch.zeros_like(combined)
                update = combined + private
                next_state = old_states[modality] + layer.residual_dropout(update)
                outputs.append(next_state * masks[:, modality:modality + 1].float())
                public_outputs.append(combined)
                private_outputs.append(private)
            states = (outputs[0], outputs[1])
            # Keep scalar diagnostics only; full per-node routes are retained transiently for the caller.
            details_layer["public_mean_norm"] = torch.stack([v.norm(dim=-1).mean() for v in public_outputs])
            details_layer["private_mean_norm"] = torch.stack([v.norm(dim=-1).mean() for v in private_outputs])
            details_layer["state_mean_norm"] = torch.stack([v.norm(dim=-1).mean() for v in states])
            details["layers"].append(details_layer)

        readout_parts = [
            self.prior_norms[0](own_semantics[0]) * masks[:, 0:1].float(),
            self.prior_norms[1](own_semantics[1]) * masks[:, 1:2].float(),
            self.state_norms[0](states[0]) * masks[:, 0:1].float(),
            self.state_norms[1](states[1]) * masks[:, 1:2].float(),
            masks.float(),
        ]
        readout_input = torch.cat(readout_parts, dim=-1)
        z = self.readout_norm(self.readout2(self.readout_dropout(F.gelu(self.readout1(readout_input)))))
        details["own_semantics"] = own_semantics if return_details else None
        details["states"] = states if return_details else None
        details["readout_input_shape"] = tuple(readout_input.shape)
        details["readout_input"] = readout_input if return_details else None
        return z, details

    def optimizer_groups(self, base_lr: float, matrix_wd: float) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, float, float], list[nn.Parameter]] = {}
        norm_ids = {
            id(parameter)
            for module in self.modules()
            if isinstance(module, nn.LayerNorm)
            for parameter in module.parameters(recurse=False)
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            norm_or_bias = name.endswith("bias") or id(param) in norm_ids
            if name.endswith("theta") or ".theta" in name:
                category, lr, wd = "theta", base_lr, 0.001
            elif name.startswith(("routers.", "context_mlps.", "type_embedding")):
                category, lr = "router_context", 0.5 * base_lr
                wd = 0.0 if norm_or_bias or name.startswith("type_embedding") else matrix_wd
            else:
                category, lr = "matrix", base_lr
                wd = 0.0 if norm_or_bias else matrix_wd
            buckets.setdefault((category, lr, wd), []).append(param)
        groups = [{"params": params, "lr": lr, "weight_decay": wd, "name": category} for (category, lr, wd), params in buckets.items()]
        ids = [id(p) for group in groups for p in group["params"]]
        expected = [id(p) for p in self.parameters() if p.requires_grad]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise RuntimeError("optimizer grouping has duplicate or missing trainable parameters")
        return groups

    def parameter_diagnostics(self) -> dict[str, Any]:
        response_grid = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
        result: dict[str, Any] = {"active_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad), "experts": []}
        for layer_index, layer in enumerate(self.layers):
            for expert_index, expert in enumerate(layer.experts):
                gamma = expert.gamma().detach().double().cpu()
                t = 1.0 - response_grid
                response = sum(
                    gamma[n].unsqueeze(0) * ((-t) ** n).unsqueeze(-1) / math.factorial(n)
                    for n in range(self.max_order + 1)
                )
                result["experts"].append({
                    "layer": layer_index, "expert": expert_index,
                    "theta_norm": float(expert.theta.detach().norm().cpu()),
                    "gamma_norm_by_order": gamma.norm(dim=-1).tolist(),
                    "response_max_abs": float(response.abs().max()),
                })
        return result
