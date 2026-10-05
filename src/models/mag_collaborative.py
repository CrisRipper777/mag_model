from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = values.new_zeros((size, values.size(-1)))
    count = values.new_zeros((size, 1))
    if index.numel():
        out.index_add_(0, index, values)
        count.index_add_(0, index, values.new_ones((index.numel(), 1)))
    return out / count.clamp_min(1.0), count.squeeze(-1)


def _masked_group_softmax(
    logits: torch.Tensor, valid: torch.Tensor, group: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Softmax over edges that share a destination, returning zero for empty groups."""
    if logits.numel() == 0:
        return logits
    masked = logits.masked_fill(~valid, -torch.inf)
    maxima = logits.new_full((n_groups,), -torch.inf)
    maxima.scatter_reduce_(0, group, masked, reduce="amax", include_self=True)
    edge_max = maxima[group]
    safe_max = torch.where(torch.isfinite(edge_max), edge_max, torch.zeros_like(edge_max))
    exponent = torch.where(valid, torch.exp(logits - safe_max), torch.zeros_like(logits))
    denom = logits.new_zeros((n_groups,))
    denom.index_add_(0, group, exponent)
    return exponent / denom[group].clamp_min(torch.finfo(logits.dtype).tiny)


class _Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, state: torch.Tensor, aggregate: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, aggregate), dim=-1))


class _LayerExperts(nn.Module):
    """One expert library shared by the two modality states at one depth."""

    def __init__(self, kind: str, dim: int, dropout: float, router_hidden: int, type_dim: int):
        super().__init__()
        self.kind = kind
        if kind == "conditional":
            self.state_norm = nn.LayerNorm(dim)
            self.aggregate_norm = nn.LayerNorm(dim)
        if kind == "wide":
            self.wide = _Expert(dim, 1025, dropout)
            self.shared = None
            self.specialists = nn.ModuleList()
            self.router = None
        else:
            self.wide = None
            self.shared = _Expert(dim, 256, dropout)
            self.specialists = nn.ModuleList([_Expert(dim, 256, dropout) for _ in range(3)])
            if kind == "conditional":
                self.routers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(3 * dim + type_dim, router_hidden),
                        nn.GELU(),
                        nn.Linear(router_hidden, 3),
                    ) for _ in range(2)
                ])
                for router in self.routers:
                    nn.init.zeros_(router[-1].weight)
                    nn.init.zeros_(router[-1].bias)
            elif kind == "static":
                self.routers = nn.ModuleList()
            else:
                raise ValueError(f"unknown expert kind {kind!r}")

    def forward(
        self,
        state: torch.Tensor,
        aggregate: torch.Tensor,
        global_state: torch.Tensor,
        type_embedding: torch.Tensor,
        modality: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        if self.kind == "wide":
            assert self.wide is not None
            out = self.wide(state, aggregate)
            return out, [out], None

        assert self.shared is not None
        shared = self.shared(state, aggregate)
        specialist_outputs = [expert(state, aggregate) for expert in self.specialists]
        if self.kind == "static":
            mixed = torch.stack(specialist_outputs, dim=0).mean(dim=0)
            weights = state.new_full((state.size(0), 3), 1.0 / 3.0)
        else:
            router = self.routers[modality]
            features = torch.cat(
                (self.state_norm(state), self.aggregate_norm(aggregate),
                 global_state.expand(state.size(0), -1),
                 type_embedding.expand(state.size(0), -1)), dim=-1
            )
            weights = torch.softmax(router(features), dim=-1)
            mixed = sum(value * weights[:, i:i + 1] for i, value in enumerate(specialist_outputs))
        return shared + mixed, [shared, *specialist_outputs], weights


class _Interaction(nn.Module):
    """Bi-axis edge-conditioned message selector for the second state update."""

    def __init__(self, dim: int, type_dim: int):
        super().__init__()
        self.dim = dim
        self.type_dim = type_dim
        self.context = nn.Sequential(
            nn.Linear(6 * dim + 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.modality_score = nn.Sequential(
            nn.Linear(3 * dim + 2 * type_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.value_projection = nn.ModuleList(
            [nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)]) for _ in range(2)]
        )
        self.query = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])
        self.key = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        states: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        available: torch.Tensor,
        type_embedding: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        text, image = states
        n, dim = text.shape
        src, dst = edge_index
        if src.numel() == 0:
            zero = text.new_zeros((n, dim))
            return (zero, zero), {
                "modality_weights": text.new_empty((0, 2, 2)),
                "neighbor_weights": text.new_empty((0, 2)),
                "neighbor_valid": torch.empty((0, 2), dtype=torch.bool, device=text.device),
            }

        degree = text.new_zeros((n,))
        degree.index_add_(0, dst, text.new_ones((dst.numel(),)))
        degree_src = degree[src].clamp_min(0)
        degree_dst = degree[dst].clamp_min(0)
        link_context = self.context(torch.cat((
            text[dst], image[dst], text[src], image[src],
            (text[dst] - text[src]).abs(), (image[dst] - image[src]).abs(),
            torch.log1p(degree_dst).unsqueeze(-1), torch.log1p(degree_src).unsqueeze(-1),
        ), dim=-1))
        sender_states = (text[src], image[src])
        receiver_states = (text[dst], image[dst])
        pair_values: list[list[torch.Tensor]] = [[], []]
        pair_selectors: list[torch.Tensor] = []
        pair_link_logits: list[torch.Tensor] = []
        pair_valid: list[torch.Tensor] = []
        pair_sender_valid: list[torch.Tensor] = []

        for receiver_modality in range(2):
            scores = []
            projected = []
            sender_masks = []
            for sender_modality in range(2):
                sender_type = type_embedding[sender_modality].expand(src.numel(), -1)
                receiver_type = type_embedding[receiver_modality].expand(src.numel(), -1)
                score_input = torch.cat((
                    link_context, receiver_states[receiver_modality],
                    sender_states[sender_modality], receiver_type, sender_type,
                ), dim=-1)
                scores.append(self.modality_score(score_input).squeeze(-1))
                projected.append(
                    self.value_projection[receiver_modality][sender_modality](sender_states[sender_modality])
                )
                sender_masks.append(available[src, sender_modality])
            score_matrix = torch.stack(scores, dim=-1)
            sender_mask = torch.stack(sender_masks, dim=-1)
            has_sender = sender_mask.any(dim=-1)
            safe_scores = score_matrix.masked_fill(~sender_mask, -torch.inf)
            safe_scores = torch.where(has_sender.unsqueeze(-1), safe_scores, torch.zeros_like(safe_scores))
            selector = torch.softmax(safe_scores, dim=-1) * sender_mask.to(score_matrix.dtype)
            selector = selector / selector.sum(dim=-1, keepdim=True).clamp_min(1.0)
            message = sum(projected[b] * selector[:, b:b + 1] for b in range(2))

            query = self.query[receiver_modality](receiver_states[receiver_modality])
            key = self.key(link_context)
            link_logits = (query * key).sum(dim=-1) / (dim ** 0.5)
            edge_valid = has_sender & available[dst, receiver_modality]
            attention = _masked_group_softmax(link_logits, edge_valid, dst, n)
            aggregate = text.new_zeros((n, dim))
            aggregate.index_add_(0, dst, message * attention.unsqueeze(-1))
            pair_values[receiver_modality] = [aggregate]
            pair_selectors.append(selector)
            pair_link_logits.append(attention)
            pair_valid.append(edge_valid)
            pair_sender_valid.append(has_sender)

        stats = {
            "modality_weights": torch.stack(pair_selectors, dim=1).detach(),
            "neighbor_weights": torch.stack(pair_link_logits, dim=1).detach(),
            "neighbor_valid": torch.stack(pair_valid, dim=1).detach(),
            "sender_valid": torch.stack(pair_sender_valid, dim=1).detach(),
        }
        return (pair_values[0][0], pair_values[1][0]), stats


class Model(nn.Module):
    """Two-modality MAG backbone with shared experts and optional dual-axis interaction."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        self.variant = str(cfg.model.get("variant", "P3")).upper()
        if self.variant not in {f"P{i}" for i in range(7)}:
            raise ValueError(f"unknown pilot variant {self.variant!r}")
        self.text_dim = int(data_info["text_dim"])
        self.image_dim = int(data_info["visual_dim"])
        self.out_dim = int(cfg.model.get("out_dim", 128))
        if self.out_dim != 128:
            raise ValueError("mechanism pilot v2 fixes representation width at d=128")
        if self.text_dim < 1 or self.image_dim < 1:
            raise ValueError("mag_collaborative requires two positive-width feature modalities")
        if self.text_dim + self.image_dim != int(data_info["input_dim"]):
            raise ValueError("input_dim must be the concatenation [text,image]")
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.type_dim = 8
        uses_type = self.variant not in {"P0"}
        if uses_type:
            self.type_embedding = nn.Parameter(torch.empty(2, self.type_dim))
            nn.init.normal_(self.type_embedding, std=0.02)
        else:
            self.register_parameter("type_embedding", None)
        self.input_text = nn.Sequential(nn.Linear(self.text_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.input_image = nn.Sequential(nn.Linear(self.image_dim, 128), nn.LayerNorm(128), nn.GELU())

        self.expert_kind = "wide" if self.variant in {"P0", "P2", "P6"} else (
            "static" if self.variant == "P4" else "conditional"
        )
        self.interaction_enabled = self.variant in {"P2", "P3", "P4", "P5", "P6"}
        self.layers = nn.ModuleList([
            _LayerExperts(self.expert_kind, 128, self.dropout, 32, self.type_dim),
            _LayerExperts(self.expert_kind, 128, self.dropout, 32, self.type_dim),
        ])
        self.update_norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(128) for _ in range(2)]) for _ in range(2)])
        self.interaction = _Interaction(128, self.type_dim) if self.interaction_enabled else None
        self.fusion = nn.Sequential(
            nn.Linear(4 * 128, 256), nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(256, 128),
        )
        self._epoch = 0
        self._diagnostics: dict[str, Any] = {}
        self._edge_cache_reference: torch.Tensor | None = None
        self._edge_cache_signature: tuple[Any, ...] | None = None
        self._edge_cache_value: torch.Tensor | None = None
        self.active_backbone_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        modality_available: torch.Tensor | None = None,
    ):
        if x.ndim != 2 or x.size(1) != self.text_dim + self.image_dim:
            raise ValueError(f"expected x [N,{self.text_dim + self.image_dim}], got {tuple(x.shape)}")
        n = x.size(0)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)
        if modality_available is None:
            available = torch.ones((n, 2), dtype=torch.bool, device=x.device)
        else:
            available = torch.as_tensor(modality_available, device=x.device, dtype=torch.bool)
            if tuple(available.shape) != (n, 2):
                raise ValueError(f"modality_available must have shape [{n},2]")
        text_input = torch.where(available[:, 0:1], x[:, :self.text_dim], torch.zeros_like(x[:, :self.text_dim]))
        image_input = torch.where(available[:, 1:2], x[:, self.text_dim:], torch.zeros_like(x[:, self.text_dim:]))
        ht = self.input_text(text_input) * available[:, 0:1].to(x.dtype)
        hi = self.input_image(image_input) * available[:, 1:2].to(x.dtype)
        type_embedding = self.type_embedding
        if type_embedding is None:
            type_embedding = x.new_zeros((2, self.type_dim))
        h0 = (ht, hi)
        states = h0
        edge_index = self._validated_edges(edge_index, n)
        layer_diagnostics: list[dict[str, Any]] = []
        source_states: tuple[torch.Tensor, torch.Tensor] | None = None

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 1 and self.interaction is not None:
                aggregate, interaction_stats = self.interaction(
                    states, edge_index, available, type_embedding
                )
            else:
                aggregate = tuple(self._same_modality_mean(states[m], edge_index, available[:, m]) for m in range(2))
                interaction_stats = {}
            next_states = []
            layer_diag: dict[str, Any] = {"interaction": self._summarize_interaction(interaction_stats)}
            for modality in range(2):
                state = states[modality]
                availability_weight = available[:, modality:modality + 1].to(state.dtype)
                global_state = (state * availability_weight).sum(dim=0, keepdim=True)
                global_state = global_state / availability_weight.sum().clamp_min(1.0)
                expert_out, expert_values, router_weights = layer(
                    state, aggregate[modality], global_state,
                    type_embedding[modality:modality + 1],
                    modality,
                )
                update = self.update_norms[layer_idx][modality](
                    state + F.dropout(expert_out, p=self.dropout, training=self.training)
                )
                update = update * available[:, modality:modality + 1].to(update.dtype)
                next_states.append(update)
                layer_diag[f"modality_{modality}"] = self._summarize_experts(
                    expert_values, router_weights, available[:, modality]
                )
            states = (next_states[0], next_states[1])
            if layer_idx == 0:
                source_states = states
            layer_diagnostics.append(layer_diag)

        z = self.fusion(torch.cat((states[0], states[1], h0[0], h0[1]), dim=-1))
        self._diagnostics = {"layers": layer_diagnostics}
        assert source_states is not None
        extras = {
            "source_states": {"text": source_states[0], "image": source_states[1]},
            "source_available": available,
            "diagnostics": self._diagnostics,
        }
        # The source task heads supervise the first, pre-interaction states.
        # Keep them distinct from the second-layer collaborative representations.
        return z, None, None, z.new_zeros(()), extras

    def _validated_edges(self, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge = edge_index.long()
        try:
            signature: tuple[Any, ...] | None = (
                id(edge_index), edge_index.data_ptr(), int(edge_index._version), n,
                edge_index.device.type, edge_index.device.index,
            )
        except RuntimeError:
            signature = None
        if signature is not None and signature == self._edge_cache_signature:
            assert self._edge_cache_reference is edge_index and self._edge_cache_value is not None
            return self._edge_cache_value
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= n):
            raise ValueError("edge_index contains node IDs outside the feature matrix")
        src, dst = edge
        keep = src != dst
        pairs = torch.stack((src[keep], dst[keep]), dim=1)
        if pairs.numel():
            pairs = torch.unique(pairs, dim=0, sorted=True)
            result = pairs.t().contiguous()
        else:
            result = torch.empty((2, 0), dtype=torch.long, device=edge.device)
        if signature is not None and not torch.is_inference_mode_enabled():
            self._edge_cache_reference = edge_index
            self._edge_cache_signature = signature
            self._edge_cache_value = result
        else:
            self._edge_cache_reference = None
            self._edge_cache_signature = None
            self._edge_cache_value = None
        return result

    @staticmethod
    def _same_modality_mean(
        state: torch.Tensor, edge_index: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        valid = available[src]
        out = state.new_zeros(state.shape)
        count = state.new_zeros((state.size(0), 1))
        selected_dst = dst[valid]
        out.index_add_(0, selected_dst, state[src[valid]])
        count.index_add_(0, selected_dst, state.new_ones((selected_dst.numel(), 1)))
        return out / count.clamp_min(1.0)

    @staticmethod
    def _summarize_experts(
        expert_values: list[torch.Tensor], weights: torch.Tensor | None, available: torch.Tensor
    ) -> dict[str, Any]:
        mask = available.to(expert_values[0].dtype)
        denom = mask.sum().clamp_min(1.0)
        values = [((value.norm(dim=-1) * mask).sum() / denom).detach() for value in expert_values]
        differences = []
        for i in range(len(expert_values)):
            for j in range(i + 1, len(expert_values)):
                delta = expert_values[i] - expert_values[j]
                differences.append(((delta.norm(dim=-1) * mask).sum() / denom).detach())
        result: dict[str, Any] = {"expert_output_mean_norm": values, "expert_pair_mean_distance": differences}
        if weights is not None:
            route_mask = mask.unsqueeze(-1)
            route_denom = route_mask.sum().clamp_min(1.0)
            route_mean = (weights.detach() * route_mask).sum(dim=0) / route_denom
            result["router_mean"] = route_mean
            result["router_variance"] = (((weights.detach() - route_mean) ** 2) * route_mask).sum(dim=0) / route_denom
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)
            result["router_entropy_mean"] = ((entropy.detach() * mask).sum() / denom).detach()
        return result

    @staticmethod
    def _summarize_interaction(stats: dict[str, torch.Tensor]) -> dict[str, Any]:
        if not stats or stats.get("modality_weights") is None:
            return {"enabled": False}
        result: dict[str, Any] = {"enabled": True}
        modality = stats["modality_weights"]
        valid = stats["neighbor_valid"]
        mask = valid.to(modality.dtype).unsqueeze(-1)
        denom = mask.sum(dim=0).clamp_min(1.0)
        result["sender_modality_mean"] = (modality * mask).sum(dim=0) / denom
        result["sender_modality_variance"] = ((modality - result["sender_modality_mean"]) ** 2 * mask).sum(dim=0) / denom
        neighbor_weights = stats["neighbor_weights"]
        edge_mask = valid.to(neighbor_weights.dtype)
        result["neighbor_weight_mean"] = (neighbor_weights * edge_mask).sum(dim=0) / edge_mask.sum(dim=0).clamp_min(1.0)
        result["neighbor_weight_max_mean"] = neighbor_weights.masked_fill(~valid, 0.0).max(dim=0).values if valid.numel() else neighbor_weights.new_zeros((2,))
        return result
