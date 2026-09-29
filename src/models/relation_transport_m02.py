from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from src.models.relation_transport_m0 import Model as M01Model


class RelationContextEncoder(nn.Module):
    """Encode incoming and outgoing distributions of pair relation states."""

    def __init__(self, relation_dim: int):
        super().__init__()
        self.linear = nn.Linear(4 * relation_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)

    def forward(self, statistics: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.linear(statistics)))


class EdgeRelationContext(nn.Module):
    """Refine one pair relation using target/source relation environments."""

    def __init__(self, relation_dim: int):
        super().__init__()
        self.target = nn.Linear(relation_dim, relation_dim)
        self.source = nn.Linear(relation_dim, relation_dim)
        self.difference = nn.Linear(relation_dim, relation_dim)
        self.norm = nn.LayerNorm(relation_dim)
        self.output = nn.Linear(relation_dim, relation_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        pair_relation: torch.Tensor,
        target_context: torch.Tensor,
        source_context: torch.Tensor,
        residual_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = self.norm(
            self.target(pair_relation * target_context)
            + self.source(pair_relation * source_context)
            + self.difference(target_context - source_context)
        )
        residual = residual_scale * torch.tanh(self.output(F.gelu(combined)))
        return pair_relation + residual, residual


class Model(M01Model):
    """M0.2 pair relation plus local relation-environment contextualization."""

    def __init__(self, cfg, data_info):
        if not bool(cfg.model.get("relation_context", False)):
            raise ValueError("relation_transport_m02 requires relation_context=true")
        residual_scale = float(cfg.model.get("context_residual_scale", 0.5))
        if abs(residual_scale - 0.5) > 1e-12:
            raise ValueError("M0.2 fixes context_residual_scale=0.5")
        if str(cfg.model.get("transport_mode", "orthogonal")).lower() != "orthogonal":
            raise ValueError("M0.2 pilot fixes transport_mode='orthogonal'")

        base_cfg = copy.deepcopy(cfg)
        base_cfg.model.relation_context = False
        super().__init__(base_cfg, data_info)

        # These modules are deliberately initialized after every M0.1 module.
        self.context_residual_scale = residual_scale
        self.text_context_encoder = RelationContextEncoder(self.relation_dim)
        self.visual_context_encoder = RelationContextEncoder(self.relation_dim)
        self.text_edge_context = EdgeRelationContext(self.relation_dim)
        self.visual_edge_context = EdgeRelationContext(self.relation_dim)
        self._relation_context_enabled = True

    @staticmethod
    def _relation_environment_statistics(
        relation: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return incoming/outgoing mean and population std, zero for isolates."""
        dim = int(relation.size(-1))
        if relation.numel() == 0:
            return relation.new_zeros((num_nodes, 4 * dim))

        groups = []
        for index in (target, source):
            degree = torch.bincount(index, minlength=num_nodes)
            count = degree.to(relation.dtype).clamp_min(1).unsqueeze(-1)
            total = scatter(relation, index, dim=0, dim_size=num_nodes, reduce="sum")
            square_total = scatter(
                relation.square(), index, dim=0, dim_size=num_nodes, reduce="sum"
            )
            mean = total / count
            variance = (square_total / count - mean.square()).clamp_min(
                torch.finfo(relation.dtype).eps
            )
            std = torch.sqrt(variance)
            non_isolated = degree > 0
            mean = torch.where(non_isolated[:, None], mean, torch.zeros_like(mean))
            std = torch.where(non_isolated[:, None], std, torch.zeros_like(std))
            groups.extend((mean, std))
        return torch.cat(groups, dim=-1)

    def _contextualize_one_modality(
        self,
        pair_relation: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        num_nodes: int,
        modality: str,
    ):
        statistics = self._relation_environment_statistics(
            pair_relation, source, target, num_nodes
        )
        encoder = (
            self.text_context_encoder
            if modality == "text"
            else self.visual_context_encoder
        )
        edge_context = (
            self.text_edge_context
            if modality == "text"
            else self.visual_edge_context
        )
        context = encoder(statistics)
        # A node with no incident physical relations has no relation environment.
        incident = (
            torch.bincount(source, minlength=num_nodes)
            + torch.bincount(target, minlength=num_nodes)
        ) > 0
        context = torch.where(incident[:, None], context, torch.zeros_like(context))
        contextual_relation, residual = edge_context(
            pair_relation,
            context[target],
            context[source],
            self.context_residual_scale,
        )
        return contextual_relation, residual, context

    def _relation_evidence(self, h0_text, h0_visual, source, target):
        pair = super()._relation_evidence(h0_text, h0_visual, source, target)
        if not self._relation_context_enabled:
            return pair

        num_nodes = int(h0_text.size(0))
        relation_text, _resid_text, _ctx_text = self._contextualize_one_modality(
            pair[2], source, target, num_nodes, "text"
        )
        relation_visual, _resid_visual, _ctx_visual = self._contextualize_one_modality(
            pair[3], source, target, num_nodes, "visual"
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
        return conductance_text, conductance_visual, relation_text, relation_visual

    def forward_without_relation_context(self, x: torch.Tensor, edge_index=None):
        """Analysis-only pair-relation path using the trained M0.2 parameters."""
        if edge_index is None:
            raise ValueError("relation_transport_m02 requires the physical edge_index")
        previous = self._relation_context_enabled
        self._relation_context_enabled = False
        try:
            return super().forward(x, edge_index)
        finally:
            self._relation_context_enabled = previous

    @torch.no_grad()
    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        self.eval()
        device = next(self.parameters()).device
        x_device = x.to(device)
        edge_device = edge_index.to(device)
        base = super().analyze(x_device, edge_device)
        # Recompute the transient pair/context edge states only for scalar diagnostics.
        clean, *_ = self._get_graph(edge_device, int(x_device.size(0)))
        source, target = clean
        h0_text = self.text_projector(x_device[:, : self.text_dim])
        h0_visual = self.visual_projector(
            x_device[:, self.text_dim : self.text_dim + self.visual_dim]
        )
        pair = super()._relation_evidence(h0_text, h0_visual, source, target)
        context_summaries = {}
        residual_summaries = {}
        conductance_changes = {}
        angle_changes = {}
        for modality, h0, pair_relation, pair_conductance in (
            ("text", h0_text, pair[2], pair[0]),
            ("visual", h0_visual, pair[3], pair[1]),
        ):
            contextual_relation, residual, context = self._contextualize_one_modality(
                pair_relation, source, target, int(x_device.size(0)), modality
            )
            contextual_conductance = self.compatibility_scale * torch.sigmoid(
                F.softplus(getattr(self, f"raw_tau_{modality}"))
                * F.cosine_similarity(h0[target], h0[source], dim=-1, eps=1e-8)
                + getattr(self, f"{modality}_relation_correction")(
                    contextual_relation
                ).squeeze(-1)
            )
            context_summaries[modality] = self._distribution(
                context.norm(dim=-1)
            )
            residual_summaries[modality] = self._distribution(
                residual.norm(dim=-1)
                / (pair_relation.norm(dim=-1) + 1e-12)
            )
            conductance_changes[modality] = self._distribution(
                (contextual_conductance - pair_conductance).abs()
            )
            pair_angles = self._edge_angles(pair_relation, modality)
            context_angles = self._edge_angles(contextual_relation, modality)
            angle_changes[modality] = self._distribution(
                (context_angles - pair_angles).abs().mean(dim=-1)
            )
        base["relation_context_norm"] = context_summaries
        base["contextual_relation_residual"] = residual_summaries
        base["context_conductance_abs_change"] = conductance_changes
        base["context_angle_mean_abs_change"] = angle_changes
        return base
