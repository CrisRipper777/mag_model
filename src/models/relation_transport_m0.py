from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, remove_self_loops, scatter, to_undirected


# M0.1 freezes this eight-rotation schedule. Angle slots are consumed in order.
GIVENS_SCHEDULE = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 0),
)


def preprocess_physical_graph(
    edge_index: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove loops, make the graph undirected, and return target-side degrees."""
    clean, _ = remove_self_loops(edge_index.long())
    clean = to_undirected(clean, num_nodes=int(num_nodes))
    clean = coalesce(clean, num_nodes=int(num_nodes))
    source, target = clean
    degree = torch.bincount(target, minlength=int(num_nodes))
    log_degree = torch.log1p(degree.to(torch.float32))
    nonzero = degree > 0
    mean_log_degree = (
        log_degree[nonzero].mean()
        if bool(nonzero.any())
        else log_degree.new_tensor(1.0)
    )
    safe_log_degree = torch.where(nonzero, log_degree, torch.ones_like(log_degree))
    amplification = torch.where(
        nonzero,
        log_degree / mean_log_degree.clamp_min(torch.finfo(log_degree.dtype).eps),
        torch.ones_like(log_degree),
    )
    attenuation = torch.where(
        nonzero,
        mean_log_degree / safe_log_degree.clamp_min(torch.finfo(log_degree.dtype).eps),
        torch.ones_like(log_degree),
    )
    scalers = torch.stack(
        (torch.ones_like(log_degree), amplification, attenuation), dim=-1
    )
    return clean, degree, mean_log_degree, scalers


def _segment_statistics(
    message_chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
    degree: torch.Tensor,
    scalers: torch.Tensor,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute mean/max/min/std and the three PNA degree scalings."""
    num_nodes = int(degree.numel())
    sum_parts: list[torch.Tensor] = []
    square_parts: list[torch.Tensor] = []
    maximum: torch.Tensor | None = None
    minimum: torch.Tensor | None = None

    for messages, target in message_chunks:
        if messages.numel() == 0:
            continue
        target = target.long()
        sum_parts.append(scatter(messages, target, dim=0, dim_size=num_nodes, reduce="sum"))
        square_parts.append(
            scatter(messages.square(), target, dim=0, dim_size=num_nodes, reduce="sum")
        )
        chunk_degree = torch.bincount(target, minlength=num_nodes)
        has_message = chunk_degree > 0
        chunk_max = scatter(messages, target, dim=0, dim_size=num_nodes, reduce="max")
        chunk_min = scatter(messages, target, dim=0, dim_size=num_nodes, reduce="min")
        if maximum is None:
            maximum = torch.where(
                has_message[:, None], chunk_max, torch.full_like(chunk_max, -torch.inf)
            )
            minimum = torch.where(
                has_message[:, None], chunk_min, torch.full_like(chunk_min, torch.inf)
            )
        else:
            maximum = torch.where(has_message[:, None], torch.maximum(maximum, chunk_max), maximum)
            minimum = torch.where(has_message[:, None], torch.minimum(minimum, chunk_min), minimum)

    zeros = torch.zeros((num_nodes, hidden_dim), dtype=dtype, device=device)
    if sum_parts:
        total = sum(sum_parts[1:], sum_parts[0])
        square_total = sum(square_parts[1:], square_parts[0])
    else:
        total = zeros
        square_total = zeros
    positive_degree = degree > 0
    count = degree.to(dtype).clamp_min(1).unsqueeze(-1)
    mean = total / count
    variance = (square_total / count - mean.square()).clamp_min(torch.finfo(dtype).eps)
    std = torch.sqrt(variance)

    if maximum is None or minimum is None:
        maximum = zeros
        minimum = zeros
    else:
        maximum = torch.where(positive_degree[:, None], maximum, zeros)
        minimum = torch.where(positive_degree[:, None], minimum, zeros)
    mean = torch.where(positive_degree[:, None], mean, zeros)
    std = torch.where(positive_degree[:, None], std, zeros)

    stats = torch.stack((mean, maximum, minimum, std), dim=1)  # [N, 4, H]
    scaled = stats[:, :, None, :] * scalers.to(dtype=dtype)[:, None, :, None]
    return scaled.reshape(num_nodes, 12 * hidden_dim)


class PNAAggregator(nn.Module):
    """Distribution-sensitive PNA aggregation followed by a signed projection."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Linear(12 * self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        message_chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
        degree: torch.Tensor,
        scalers: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        features = _segment_statistics(
            message_chunks, degree, scalers, self.hidden_dim, dtype, device
        )
        return self.dropout(self.projection(features))


class ModalityRelations(nn.Module):
    """Translation, multiplicative, and bilinear relation primitives."""

    def __init__(self, hidden_dim: int, relation_dim: int):
        super().__init__()
        self.diff = nn.Linear(hidden_dim, relation_dim)
        self.prod = nn.Linear(hidden_dim, relation_dim)
        self.bilinear_target = nn.Linear(hidden_dim, relation_dim)
        self.bilinear_source = nn.Linear(hidden_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        translation = self.diff(source - target)
        multiplicative = self.prod(target * source)
        bilinear = self.bilinear_target(target) * self.bilinear_source(source)
        return F.gelu(self.norm(translation + multiplicative + bilinear))


class ModalityResponse(nn.Module):
    def __init__(self, hidden_dim: int, response_hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, response_hidden_dim)
        self.linear2 = nn.Linear(response_hidden_dim, hidden_dim)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, discrepancy: torch.Tensor) -> torch.Tensor:
        residual = self.linear2(F.gelu(self.linear1(self.norm(discrepancy))))
        return discrepancy + residual


class Model(nn.Module):
    """M0.1 relation-conditioned semantic transport prototype."""

    MAX_ORDER = 3

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", 0))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("relation_transport_m0 requires positive text_dim and visual_dim")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError(
                "relation_transport_m0 expects [text, visual] features and requires "
                f"input_dim == text_dim + visual_dim; got {self.input_dim}, "
                f"{self.text_dim} + {self.visual_dim}"
            )

        self.hidden_dim = int(cfg.model.get("hidden_dim", 128))
        self.relation_dim = int(cfg.model.get("relation_dim", 64))
        self.max_order = int(cfg.model.get("max_order", self.MAX_ORDER))
        if self.hidden_dim != 128:
            raise ValueError("M0.1 fixes hidden_dim=128")
        if self.relation_dim != 64:
            raise ValueError("M0.1 fixes relation_dim=64")
        if self.max_order != self.MAX_ORDER:
            raise ValueError("M0.1 fixes max_order=3")

        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.relation_dropout = float(cfg.model.get("relation_dropout", 0.1))
        self.transport_mode = str(cfg.model.get("transport_mode", "orthogonal")).lower()
        if self.transport_mode not in {"identity", "orthogonal"}:
            raise ValueError("transport_mode must be 'identity' or 'orthogonal'")
        if bool(cfg.model.get("relation_context", False)):
            raise NotImplementedError(
                "M0.1 does not implement contextual relation refinement; reserved for M0.2"
            )
        if str(cfg.model.get("aggregation", "pna")).lower() != "pna":
            raise ValueError("M0.1 fixes aggregation='pna'")
        if str(cfg.model.get("readout", "gpr")).lower() != "gpr":
            raise ValueError("M0.1 fixes readout='gpr'")
        self.group_dim = int(cfg.model.get("transport_group_dim", 8))
        self.transport_sweeps = int(cfg.model.get("transport_sweeps", 2))
        if self.hidden_dim % self.group_dim != 0:
            raise ValueError("hidden_dim must be divisible by transport_group_dim")
        if self.group_dim != 8 or self.transport_sweeps != 2:
            raise ValueError("M0.1 fixes transport_group_dim=8 and transport_sweeps=2")
        self.transport_max_angle = float(cfg.model.get("transport_max_angle", math.pi / 2))
        self.compatibility_scale = float(cfg.model.get("compatibility_scale", 2.0))
        self.anchor_beta = float(cfg.model.get("anchor_beta", 0.1))
        if not 0.0 <= self.anchor_beta <= 1.0:
            raise ValueError("anchor_beta must be between zero and one")
        self.response_hidden_dim = int(cfg.model.get("response_hidden_dim", 64))
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 65536))
        if self.edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.text_relations = ModalityRelations(self.hidden_dim, self.relation_dim)
        self.visual_relations = ModalityRelations(self.hidden_dim, self.relation_dim)
        self.shared_relation = nn.Sequential(
            nn.Linear(3 * self.relation_dim, self.relation_dim),
            nn.GELU(),
            nn.LayerNorm(self.relation_dim),
            nn.Dropout(self.relation_dropout),
        )
        self.text_shared_projection = nn.Linear(self.relation_dim, self.relation_dim)
        self.visual_shared_projection = nn.Linear(self.relation_dim, self.relation_dim)
        self.text_relation_norm = nn.LayerNorm(self.relation_dim)
        self.visual_relation_norm = nn.LayerNorm(self.relation_dim)
        self.text_relation_correction = nn.Linear(self.relation_dim, 1)
        self.visual_relation_correction = nn.Linear(self.relation_dim, 1)
        nn.init.zeros_(self.text_relation_correction.weight)
        nn.init.zeros_(self.text_relation_correction.bias)
        nn.init.zeros_(self.visual_relation_correction.weight)
        nn.init.zeros_(self.visual_relation_correction.bias)
        initial_tau = math.log(math.expm1(1.0))
        self.raw_tau_text = nn.Parameter(torch.tensor(initial_tau))
        self.raw_tau_visual = nn.Parameter(torch.tensor(initial_tau))

        if self.transport_mode == "orthogonal":
            self.text_angle_head = nn.Linear(self.relation_dim, len(GIVENS_SCHEDULE))
            self.visual_angle_head = nn.Linear(self.relation_dim, len(GIVENS_SCHEDULE))
            nn.init.zeros_(self.text_angle_head.weight)
            nn.init.zeros_(self.text_angle_head.bias)
            nn.init.zeros_(self.visual_angle_head.weight)
            nn.init.zeros_(self.visual_angle_head.bias)
        self.text_response = ModalityResponse(self.hidden_dim, self.response_hidden_dim)
        self.visual_response = ModalityResponse(self.hidden_dim, self.response_hidden_dim)
        self.text_aggregator = PNAAggregator(self.hidden_dim, self.dropout)
        self.visual_aggregator = PNAAggregator(self.hidden_dim, self.dropout)
        self.text_state_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_state_norm = nn.LayerNorm(self.hidden_dim)
        self.gamma_logits_text = nn.Parameter(torch.zeros(4))
        self.gamma_logits_visual = nn.Parameter(torch.zeros(4))
        self.fusion = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.out_dim = self.hidden_dim

        # Only graph structure is cached. No projected features or relation state survives a call.
        self._graph_cache_key = None
        self._graph_cache_ref: torch.Tensor | None = None
        self._graph_cache = None

    def _get_graph(self, edge_index: torch.Tensor, num_nodes: int):
        key = (
            id(edge_index),
            edge_index.data_ptr(),
            int(getattr(edge_index, "_version", 0)),
            tuple(edge_index.shape),
            edge_index.device,
            int(num_nodes),
        )
        if self._graph_cache_key == key and self._graph_cache is not None:
            return self._graph_cache
        graph = preprocess_physical_graph(edge_index, num_nodes)
        self._graph_cache_ref = edge_index
        self._graph_cache_key = key
        self._graph_cache = graph
        return graph

    @staticmethod
    def _givens_transport(source_state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Apply the frozen two-sweep orthogonal rotation without in-place writes."""
        edge_count, hidden_dim = source_state.shape
        grouped = source_state.reshape(edge_count, hidden_dim // 8, 8)
        for sweep_start in (0, 4):
            old = grouped.unbind(dim=-1)
            updated = list(old)
            for local_slot, (a, b) in enumerate(GIVENS_SCHEDULE[sweep_start : sweep_start + 4]):
                theta = angles[:, sweep_start + local_slot, None]
                cosine = torch.cos(theta)
                sine = torch.sin(theta)
                updated[a] = cosine * old[a] - sine * old[b]
                updated[b] = sine * old[a] + cosine * old[b]
            grouped = torch.stack(updated, dim=-1)
        return grouped.reshape(edge_count, hidden_dim)

    def _edge_angles(self, relation: torch.Tensor, modality: str) -> torch.Tensor | None:
        if self.transport_mode == "identity":
            return None
        head = self.text_angle_head if modality == "text" else self.visual_angle_head
        return self.transport_max_angle * torch.tanh(head(relation))

    def _relation_evidence(
        self,
        h0_text: torch.Tensor,
        h0_visual: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ):
        text_raw = self.text_relations(h0_text[target], h0_text[source])
        visual_raw = self.visual_relations(h0_visual[target], h0_visual[source])
        shared = self.shared_relation(
            torch.cat(
                (text_raw + visual_raw, (text_raw - visual_raw).abs(), text_raw * visual_raw),
                dim=-1,
            )
        )
        relation_text = self.text_relation_norm(
            text_raw + self.text_shared_projection(shared)
        )
        relation_visual = self.visual_relation_norm(
            visual_raw + self.visual_shared_projection(shared)
        )
        similarity_text = F.cosine_similarity(
            h0_text[target], h0_text[source], dim=-1, eps=1e-8
        )
        similarity_visual = F.cosine_similarity(
            h0_visual[target], h0_visual[source], dim=-1, eps=1e-8
        )
        conductance_text = self.compatibility_scale * torch.sigmoid(
            F.softplus(self.raw_tau_text) * similarity_text
            + self.text_relation_correction(relation_text).squeeze(-1)
        )
        conductance_visual = self.compatibility_scale * torch.sigmoid(
            F.softplus(self.raw_tau_visual) * similarity_visual
            + self.visual_relation_correction(relation_visual).squeeze(-1)
        )
        return (
            conductance_text,
            conductance_visual,
            relation_text,
            relation_visual,
        )

    def _aggregate_interactions(
        self,
        state: torch.Tensor,
        h0: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        conductance: torch.Tensor,
        angles: torch.Tensor | None,
        degree: torch.Tensor,
        scalers: torch.Tensor,
        response: ModalityResponse,
        aggregator: PNAAggregator,
    ) -> torch.Tensor:
        def messages():
            for start in range(0, int(source.numel()), self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, int(source.numel()))
                source_state = state[source[start:end]]
                if angles is not None:
                    source_state = self._givens_transport(source_state, angles[start:end])
                discrepancy = source_state - state[target[start:end]]
                message = conductance[start:end, None] * response(discrepancy)
                yield message, target[start:end]

        return aggregator(
            messages(), degree, scalers, state.dtype, state.device
        )

    def _compute(self, x: torch.Tensor, edge_index: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected x with shape [num_nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        clean_edge_index, degree, _mean_log_degree, scalers = self._get_graph(
            edge_index.to(device=x.device), int(x.size(0))
        )
        source, target = clean_edge_index
        h0_text = self.text_projector(x[:, : self.text_dim])
        h0_visual = self.visual_projector(
            x[:, self.text_dim : self.text_dim + self.visual_dim]
        )
        (
            conductance_text,
            conductance_visual,
            relation_text,
            relation_visual,
        ) = self._relation_evidence(h0_text, h0_visual, source, target)
        angles_text = self._edge_angles(relation_text, "text")
        angles_visual = self._edge_angles(relation_visual, "visual")

        states_text = [h0_text]
        states_visual = [h0_visual]
        for _order in range(1, self.max_order + 1):
            previous_text = states_text[-1]
            previous_visual = states_visual[-1]
            aggregate_text = self._aggregate_interactions(
                previous_text,
                h0_text,
                source,
                target,
                conductance_text,
                angles_text,
                degree,
                scalers,
                self.text_response,
                self.text_aggregator,
            )
            aggregate_visual = self._aggregate_interactions(
                previous_visual,
                h0_visual,
                source,
                target,
                conductance_visual,
                angles_visual,
                degree,
                scalers,
                self.visual_response,
                self.visual_aggregator,
            )
            next_text = self.text_state_norm(
                (1.0 - self.anchor_beta) * (previous_text + aggregate_text)
                + self.anchor_beta * h0_text
            )
            next_visual = self.visual_state_norm(
                (1.0 - self.anchor_beta) * (previous_visual + aggregate_visual)
                + self.anchor_beta * h0_visual
            )
            states_text.append(next_text)
            states_visual.append(next_visual)

        gamma_text = torch.softmax(self.gamma_logits_text, dim=0)
        gamma_visual = torch.softmax(self.gamma_logits_visual, dim=0)
        z_text = sum(gamma_text[k] * states_text[k] for k in range(4))
        z_visual = sum(gamma_visual[k] * states_visual[k] for k in range(4))
        fused = self.fusion(torch.cat((z_text, z_visual), dim=-1))
        return {
            "z": fused,
            "H0_text": h0_text,
            "H0_visual": h0_visual,
            "S_text": states_text,
            "S_visual": states_visual,
            "conductance_text": conductance_text,
            "conductance_visual": conductance_visual,
            "angles_text": angles_text,
            "angles_visual": angles_visual,
            "source": source,
            "target": target,
            "gamma_text": gamma_text,
            "gamma_visual": gamma_visual,
        }

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            raise ValueError("relation_transport_m0 requires the physical edge_index")
        result = self._compute(x, edge_index)
        z = result["z"]
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        del batch_size  # M0.1 inference is exact full-graph inference.
        if edge_index is None:
            raise ValueError("relation_transport_m0 requires the physical edge_index")
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()

    @staticmethod
    def _norm_summary(values: torch.Tensor) -> dict[str, float]:
        norm = values.norm(p=2, dim=-1).float()
        quantiles = torch.quantile(norm, norm.new_tensor([0.5, 0.95]))
        return {
            "mean": float(norm.mean().item()),
            "p50": float(quantiles[0].item()),
            "p95": float(quantiles[1].item()),
        }

    @staticmethod
    def _distribution(values: torch.Tensor) -> dict[str, float]:
        values = values.detach().float().reshape(-1)
        quantiles = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "p10": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p90": float(quantiles[2].item()),
        }

    @torch.no_grad()
    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """Compute requested scalar diagnostics without returning edge-level state."""
        self.eval()
        device = next(self.parameters()).device
        result = self._compute(x.to(device), edge_index.to(device))
        semantic = {}
        for name in ("text", "visual"):
            states = result[f"S_{name}"]
            semantic[name] = {
                f"S{order}" if order else "H0": self._norm_summary(states[order])
                for order in range(4)
            }
        conductance = {
            "text": self._distribution(result["conductance_text"]),
            "visual": self._distribution(result["conductance_visual"]),
        }
        angle_stats = {}
        displacement = {}
        source = result["source"]
        sample_count = min(int(source.numel()), 4096)
        if self.transport_mode == "identity":
            zero_distribution = {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
            for name in ("text", "visual"):
                angle_stats[name] = dict(zero_distribution)
                displacement[name] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "p90": 0.0,
                    "sampled_edges": sample_count,
                }
        elif sample_count:
            angle_stats = {
                "text": self._distribution(result["angles_text"].abs()),
                "visual": self._distribution(result["angles_visual"].abs()),
            }
            sample_idx = torch.linspace(
                0, source.numel() - 1, steps=sample_count, device=device
            ).round().long()
            sampled_source = source[sample_idx]
            for name in ("text", "visual"):
                h0 = result[f"H0_{name}"]
                transported = self._givens_transport(
                    h0[sampled_source], result[f"angles_{name}"][sample_idx]
                )
                relative = (transported - h0[sampled_source]).norm(dim=-1) / (
                    h0[sampled_source].norm(dim=-1) + 1e-12
                )
                q = torch.quantile(relative, relative.new_tensor([0.5, 0.9]))
                displacement[name] = {
                    "mean": float(relative.mean().item()),
                    "median": float(q[0].item()),
                    "p90": float(q[1].item()),
                    "sampled_edges": sample_count,
                }
        return {
            "semantic_norms": semantic,
            "conductance": conductance,
            "abs_angle": angle_stats,
            "transport_displacement": displacement,
            "gamma_text": result["gamma_text"].cpu().tolist(),
            "gamma_visual": result["gamma_visual"].cpu().tolist(),
            "num_physical_directed_edges": int(result["source"].numel()),
        }
