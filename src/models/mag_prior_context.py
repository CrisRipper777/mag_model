from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


FLOW_NAMES = ("joint", "text", "image")


def canonical_directed_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return the unique, loop-free, bidirectional src->dst message list."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edge = edge_index.to(dtype=torch.long)
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= int(num_nodes)):
        raise ValueError("edge_index contains node IDs outside the feature matrix")
    src, dst = edge
    keep = src != dst
    src, dst = src[keep], dst[keep]
    src = torch.cat((src, dst), dim=0)
    dst = torch.cat((dst, src[: dst.numel()]), dim=0)
    if src.numel() == 0:
        return edge.new_empty((2, 0))
    key = torch.unique(src * int(num_nodes) + dst, sorted=True)
    return torch.stack((key // int(num_nodes), key % int(num_nodes)), dim=0).contiguous()


def degree_and_mean(
    state: torch.Tensor, edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Incoming-neighbor mean and unique incoming degree for directed src->dst edges."""
    n = state.size(0)
    src, dst = edge_index
    degree = state.new_zeros((n,))
    aggregate = state.new_zeros(state.shape)
    if src.numel():
        degree.index_add_(0, dst, state.new_ones((dst.numel(),)))
        aggregate.index_add_(0, dst, state[src])
    aggregate = aggregate / degree.clamp_min(1).unsqueeze(-1)
    return aggregate, degree


class _StructureMLP(nn.Module):
    def __init__(self, hidden: int, *, difference: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, 128, bias=not difference),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _StructurePair(nn.Module):
    def __init__(self, hidden_s: int, hidden_d: int):
        super().__init__()
        self.phi_s = _StructureMLP(hidden_s, difference=False)
        self.phi_d = _StructureMLP(hidden_d, difference=True)

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
        edge_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n = state.size(0)
        s_message = state.new_zeros(state.shape)
        d_message = state.new_zeros(state.shape)
        if src.numel():
            # S transforms each sender node once, before neighbor averaging.
            transformed = self.phi_s(state)
            s_message.index_add_(0, dst, transformed[src])

            # D transforms each edge difference before averaging. Evaluate
            # phi_D(delta) and phi_D(0) in one batch so identical inputs produce
            # bit-identical outputs even when the backend uses shape-dependent
            # GEMM kernels. Both sides stay in autograd.
            step = max(int(edge_chunk_size), 1)
            for start in range(0, src.numel(), step):
                stop = min(start + step, src.numel())
                edge_src, edge_dst = src[start:stop], dst[start:stop]
                delta = state[edge_src] - state[edge_dst]
                paired = self.phi_d(torch.cat((delta, torch.zeros_like(delta)), dim=0))
                count = delta.size(0)
                transformed_delta = paired[:count] - paired[count:]
                d_message.index_add_(0, edge_dst, transformed_delta)
        divisor = degree.clamp_min(1).unsqueeze(-1)
        return s_message / divisor, d_message / divisor


class _Router(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(1409, 16), nn.GELU(), nn.Linear(16, 2))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class _ReadoutCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(640, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, 128, bias=False),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.001)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class Model(nn.Module):
    """MAG prior-context backbone; R0-R6 are frozen pilot comparators."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        self.variant = str(cfg.model.get("variant", "R4")).upper()
        if self.variant not in {f"R{i}" for i in range(7)}:
            raise ValueError(f"unknown prior-context variant {self.variant!r}")
        self.text_dim = int(data_info["text_dim"])
        self.image_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.image_dim <= 0:
            raise ValueError("mag_prior_context requires positive text and image dimensions")
        if self.text_dim + self.image_dim != int(data_info["input_dim"]):
            raise ValueError("input_dim must be the concatenation [text,image]")
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 16384))
        self.out_dim = 384 if self.variant == "R6" else 128
        self._collect_diagnostics = False
        self._epoch = 0
        self._dgl_graph = None
        self._dgl_graph_key = None
        self._edge_cache_reference: torch.Tensor | None = None
        self._edge_cache_signature: tuple[Any, ...] | None = None
        self._edge_cache_value: torch.Tensor | None = None

        if self.variant == "R6":
            self._init_r6()
        else:
            self.adapter_text = nn.Sequential(
                nn.Linear(self.text_dim, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.adapter_image = nn.Sequential(
                nn.Linear(self.image_dim, 128), nn.LayerNorm(128), nn.GELU()
            )
            if self.variant == "R0":
                width = int(cfg.model.get("width", 1596))
                self.node_mlp = nn.Sequential(
                    nn.Linear(256, width), nn.GELU(), nn.Dropout(self.dropout),
                    nn.Linear(width, 128),
                )
                return
            self.center_encoder = nn.Sequential(
                nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128)
            )
            self.joint_init = nn.Sequential(
                nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.state_norms = nn.ModuleList([
                nn.ModuleDict({flow: nn.LayerNorm(128) for flow in FLOW_NAMES})
                for _ in range(2)
            ])
            self.update_dropout = nn.Dropout(self.dropout)
            self.readout_correction = _ReadoutCorrection()

            if self.variant == "R1":
                width = int(cfg.model.get("width", 144))
                self.wide_updates = nn.ModuleList([
                    nn.ModuleDict({
                        flow: nn.Sequential(
                            nn.Linear(256, width), nn.GELU(), nn.Linear(width, 128)
                        ) for flow in FLOW_NAMES
                    }) for _ in range(2)
                ])
            elif self.variant in {"R2", "R3", "R4", "R5"}:
                self.routers = nn.ModuleList([
                    nn.ModuleDict({flow: _Router() for flow in FLOW_NAMES})
                    for _ in range(2)
                ]) if self.variant != "R2" else nn.ModuleList()
                if self.variant == "R2":
                    hidden_s = hidden_d = int(cfg.model.get("width", 108))
                elif self.variant == "R5":
                    hidden_s, hidden_d = 193, 192
                else:
                    hidden_s = hidden_d = 64
                self.structure = nn.ModuleList()
                for _ in range(2):
                    if self.variant == "R5":
                        layer = nn.ModuleDict({"shared": _StructurePair(hidden_s, hidden_d)})
                    else:
                        layer = nn.ModuleDict({
                            flow: _StructurePair(hidden_s, hidden_d) for flow in FLOW_NAMES
                        })
                    self.structure.append(layer)
            else:
                raise AssertionError(self.variant)

    def _init_r6(self) -> None:
        from dgl.nn import GraphConv

        self.prior_text = nn.Sequential(nn.Linear(self.text_dim, 128), nn.ReLU())
        self.prior_image = nn.Sequential(nn.Linear(self.image_dim, 128), nn.ReLU())
        self.gcn1 = GraphConv(256, 128, norm="both", allow_zero_in_degree=False)
        self.gcn2 = GraphConv(128, 128, norm="both", allow_zero_in_degree=False)
        self.gcn_dropout = nn.Dropout(self.dropout)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def diagnostic_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        if self.variant == "R6":
            modules.update({
                "prior_text": self.prior_text, "prior_image": self.prior_image,
                "gcn1": self.gcn1, "gcn2": self.gcn2,
            })
            return modules
        modules.update({"adapter_text": self.adapter_text, "adapter_image": self.adapter_image})
        if self.variant == "R0":
            modules["topology_free_mlp"] = self.node_mlp
            return modules
        modules.update({
            "center_encoder_E": self.center_encoder,
            "joint_init_J": self.joint_init,
            "readout_Phi": self.readout_correction,
        })
        for layer_idx in range(2):
            for flow in FLOW_NAMES:
                modules[f"layer{layer_idx + 1}.{flow}.LN"] = self.state_norms[layer_idx][flow]
                if self.variant == "R1":
                    modules[f"layer{layer_idx + 1}.{flow}.wide_MLP"] = self.wide_updates[layer_idx][flow]
                else:
                    pair = self._pair(layer_idx, flow)
                    if self.variant == "R5":
                        # Shared functions are counted and diagnosed once per layer.
                        continue
                    modules[f"layer{layer_idx + 1}.{flow}.phiS"] = pair.phi_s
                    modules[f"layer{layer_idx + 1}.{flow}.phiD"] = pair.phi_d
                    if self.variant != "R2":
                        modules[f"layer{layer_idx + 1}.{flow}.router"] = self.routers[layer_idx][flow]
            if self.variant == "R5":
                pair = self.structure[layer_idx]["shared"]
                modules[f"layer{layer_idx + 1}.shared.phiS"] = pair.phi_s
                modules[f"layer{layer_idx + 1}.shared.phiD"] = pair.phi_d
                for flow in FLOW_NAMES:
                    modules[f"layer{layer_idx + 1}.{flow}.router"] = self.routers[layer_idx][flow]
        return modules

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        if x.ndim != 2 or x.size(1) != self.text_dim + self.image_dim:
            raise ValueError(
                f"expected x [N,{self.text_dim + self.image_dim}], got {tuple(x.shape)}"
            )
        text = self.adapter_text(x[:, :self.text_dim]) if self.variant != "R6" else self.prior_text(x[:, :self.text_dim])
        image = self.adapter_image(x[:, self.text_dim:]) if self.variant != "R6" else self.prior_image(x[:, self.text_dim:])
        if self.variant == "R0":
            z = self.node_mlp(torch.cat((text, image), dim=-1))
            extras = {"source_states": {"text": text, "image": image}, "diagnostics": {}}
            if self._collect_diagnostics:
                extras["diagnostics"] = self._summary(text=text, image=image, center=z, correction=None, z=z)
            return z, None, None, z.new_zeros(()), extras

        if self.variant == "R6":
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edges = self._canonical_edges(edge_index, x.size(0))
            graph = self._get_dgl_graph(edges, x.size(0))
            joint = self.gcn1(graph, torch.cat((text, image), dim=-1))
            joint = self.gcn_dropout(F.relu(joint))
            joint = self.gcn2(graph, joint)
            z = torch.cat((text, image, joint), dim=-1)
            extras = {"source_states": {"text": text, "image": image}, "joint_state": joint, "diagnostics": {}}
            if self._collect_diagnostics:
                extras["diagnostics"] = self._summary(text=text, image=image, center=joint, correction=None, z=z)
            return z, None, None, z.new_zeros(()), extras

        center = self.center_encoder(torch.cat((text, image), dim=-1))
        h0 = {
            "text": text,
            "image": image,
            "joint": self.joint_init(torch.cat((text, image), dim=-1)),
        }
        states = dict(h0)
        edges = torch.empty((2, 0), dtype=torch.long, device=x.device) if edge_index is None else self._canonical_edges(edge_index, x.size(0))
        degree: torch.Tensor | None = None
        route_diagnostics: dict[str, Any] = {}
        structure_diagnostics: dict[str, Any] = {}
        for layer_idx in range(2):
            means: dict[str, torch.Tensor] = {}
            for flow in FLOW_NAMES:
                means[flow], degree = degree_and_mean(states[flow], edges)
            assert degree is not None
            context = torch.cat((
                text, image,
                states["joint"], states["text"], states["image"],
                means["joint"], means["text"], means["image"],
            ), dim=-1)
            if self.variant == "R3":
                context = context.mean(dim=0, keepdim=True).expand(x.size(0), -1)
            next_states: dict[str, torch.Tensor] = {}
            for flow in FLOW_NAMES:
                state = states[flow]
                if self.variant == "R1":
                    output = self.wide_updates[layer_idx][flow](torch.cat((state, means[flow]), dim=-1))
                else:
                    pair = self._pair(layer_idx, flow)
                    structure_s, structure_d = pair(state, edges, degree, self.edge_chunk_size)
                    if self.variant == "R2":
                        weights = state.new_full((state.size(0), 2), 0.5)
                    else:
                        local = torch.cat((
                            h0[flow], state, means[flow], torch.log1p(degree).unsqueeze(-1)
                        ), dim=-1)
                        logits = self.routers[layer_idx][flow](torch.cat((local, context), dim=-1))
                        weights = torch.softmax(logits, dim=-1)
                    output = weights[:, 0:1] * structure_s + weights[:, 1:2] * structure_d
                    if self._collect_diagnostics:
                        structure_diagnostics[f"layer{layer_idx + 1}.{flow}"] = (structure_s.detach(), structure_d.detach())
                        route_diagnostics[f"layer{layer_idx + 1}.{flow}"] = weights.detach()
                updated = self.state_norms[layer_idx][flow](state + self.update_dropout(output))
                next_states[flow] = updated
            states = next_states

        degree_mask = (degree > 0).to(x.dtype).unsqueeze(-1)
        structure_delta = torch.cat((
            degree_mask * (states["joint"] - h0["joint"]),
            degree_mask * (states["text"] - text),
            degree_mask * (states["image"] - image),
        ), dim=-1)
        phi_input = torch.cat((structure_delta, text, image), dim=-1)
        phi_zero_input = torch.cat((torch.zeros_like(structure_delta), text, image), dim=-1)
        correction = self.readout_correction(phi_input) - self.readout_correction(phi_zero_input)
        z = center + F.dropout(correction, p=self.dropout, training=self.training)
        extras = {
            "source_states": {"text": text, "image": image},
            "joint_initial": h0["joint"],
            "route_weights": route_diagnostics,
            "structure_outputs": structure_diagnostics,
            "correction": correction,
            "center": center,
            "degree": degree,
            "diagnostics": {},
        }
        if self._collect_diagnostics:
            extras["diagnostics"] = self._summary(
                text=text, image=image, center=center, correction=correction, z=z,
                routes=route_diagnostics, structures=structure_diagnostics, degree=degree,
            )
        return z, None, None, z.new_zeros(()), extras

    def _pair(self, layer_idx: int, flow: str) -> _StructurePair:
        layer = self.structure[layer_idx]
        return layer["shared"] if self.variant == "R5" else layer[flow]

    def _canonical_edges(self, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        try:
            signature = (
                id(edge_index), edge_index.data_ptr(), int(edge_index._version), n,
                edge_index.device.type, edge_index.device.index,
            )
        except RuntimeError:
            signature = None
        if signature is not None and signature == self._edge_cache_signature:
            if self._edge_cache_reference is edge_index and self._edge_cache_value is not None:
                return self._edge_cache_value
        cleaned = canonical_directed_edges(edge_index, n)
        if signature is not None and not torch.is_inference_mode_enabled():
            self._edge_cache_reference = edge_index
            self._edge_cache_signature = signature
            self._edge_cache_value = cleaned
        else:
            # Inference tensors do not expose a reliable mutation version.
            # Rebuild each call rather than risking stale topology reuse.
            self._edge_cache_reference = None
            self._edge_cache_signature = None
            self._edge_cache_value = None
        return cleaned

    def _get_dgl_graph(self, edge_index: torch.Tensor, num_nodes: int):
        import dgl

        try:
            key = (id(edge_index), edge_index.data_ptr(), int(edge_index._version), num_nodes, str(edge_index.device))
        except RuntimeError:
            key = None
        if key is not None and self._dgl_graph is not None and self._dgl_graph_key == key:
            return self._dgl_graph
        graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes, device=edge_index.device)
        graph = dgl.add_self_loop(graph)
        if key is not None and not torch.is_inference_mode_enabled():
            self._dgl_graph = graph
            self._dgl_graph_key = key
        else:
            self._dgl_graph = None
            self._dgl_graph_key = None
        return graph

    def _summary(
        self,
        *,
        text: torch.Tensor,
        image: torch.Tensor,
        center: torch.Tensor,
        correction: torch.Tensor | None,
        z: torch.Tensor,
        routes: dict[str, torch.Tensor] | None = None,
        structures: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        degree: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        def norm_summary(value: torch.Tensor) -> dict[str, torch.Tensor]:
            norms = value.detach().float().norm(dim=-1)
            return {"mean": norms.mean(), "std": norms.std(unbiased=False)}

        result: dict[str, Any] = {
            "representation_norms": {
                "p_text": norm_summary(text), "p_image": norm_summary(image),
                "center_or_joint": norm_summary(center), "z": norm_summary(z),
            },
            "context_mode": "global_mean" if self.variant == "R3" else "per_node",
        }
        if correction is not None:
            e = center.detach().float()
            r = correction.detach().float()
            eps = 1e-8
            result["representation_norms"]["correction"] = norm_summary(correction)
            result["center_correction_relation"] = {
                "cosine_mean": F.cosine_similarity(e, r, dim=-1, eps=eps).mean(),
                "projection_coefficient_mean": ((e * r).sum(-1) / (e.square().sum(-1) + eps)).mean(),
            }
        if routes:
            route_result: dict[str, Any] = {}
            assert degree is not None
            buckets = {
                "degree_0": degree == 0,
                "degree_1": degree == 1,
                "degree_2_5": (degree >= 2) & (degree <= 5),
                "degree_6_plus": degree >= 6,
            }
            for name, weights in routes.items():
                w = weights.detach().float()
                entropy = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)
                entry: dict[str, Any] = {
                    "mean": w.mean(dim=0),
                    "variance": w.var(dim=0, unbiased=False),
                    "entropy_mean": entropy.mean(),
                }
                by_degree = {}
                for bucket, mask in buckets.items():
                    if bool(mask.any()):
                        by_degree[bucket] = {"n": int(mask.sum()), "mean": w[mask].mean(dim=0)}
                    else:
                        by_degree[bucket] = {"n": 0, "mean": w.new_zeros((2,))}
                entry["by_degree"] = by_degree
                route_result[name] = entry
            result["routes"] = route_result
        if structures:
            structure_result: dict[str, Any] = {}
            for name, (s_value, d_value) in structures.items():
                s = s_value.detach().float()
                d = d_value.detach().float()
                structure_result[name] = {
                    "S_norm": norm_summary(s), "D_norm": norm_summary(d),
                    "S_D_cosine_mean": F.cosine_similarity(s, d, dim=-1, eps=1e-8).mean(),
                }
            result["structure"] = structure_result
        return result
