from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Variant = Literal["uniform", "similarity_scalar", "learned_scalar", "conditional_feature"]
MODALITIES = ("text", "visual")
VARIANTS: tuple[Variant, ...] = (
    "uniform",
    "similarity_scalar",
    "learned_scalar",
    "conditional_feature",
)


def relation_descriptor(h_target: torch.Tensor, h_source: torch.Tensor) -> torch.Tensor:
    """Descriptor for the directed message source -> target."""
    return torch.cat(
        (h_target, h_source, (h_target - h_source).abs(), h_target * h_source), dim=-1
    )


class RelationEncoder(nn.Module):
    """Shared V2/V3 relation-encoder definition."""

    def __init__(self, semantic_dim: int, relation_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4 * semantic_dim, relation_dim),
            nn.GELU(),
            nn.LayerNorm(relation_dim),
            nn.Dropout(dropout),
            nn.Linear(relation_dim, relation_dim),
            nn.GELU(),
            nn.LayerNorm(relation_dim),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.layers(descriptor)


class P02Model(nn.Module):
    """Local P0.2 relation probe. Not part of the formal model collection."""

    def __init__(
        self,
        variant: Variant,
        *,
        num_classes: int,
        semantic_dim: int,
        relation_dim: int,
        relation_dropout: float,
        scalar_gate_scale: float,
        feature_modulation_scale: float,
        similarity_init_alpha: float,
        similarity_init_beta: float,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown P0.2 variant: {variant}")
        self.variant = variant
        self.semantic_dim = int(semantic_dim)
        self.relation_dim = int(relation_dim)
        self.scalar_gate_scale = float(scalar_gate_scale)
        self.feature_modulation_scale = float(feature_modulation_scale)

        if variant == "similarity_scalar":
            self.similarity_alpha = nn.Parameter(torch.full((2,), float(similarity_init_alpha)))
            self.similarity_beta = nn.Parameter(torch.full((2,), float(similarity_init_beta)))
        else:
            self.register_parameter("similarity_alpha", None)
            self.register_parameter("similarity_beta", None)

        if variant in ("learned_scalar", "conditional_feature"):
            self.relation_encoders = nn.ModuleList(
                [RelationEncoder(semantic_dim, relation_dim, relation_dropout) for _ in MODALITIES]
            )
            self.scalar_heads = nn.ModuleList([nn.Linear(relation_dim, 1) for _ in MODALITIES])
        else:
            self.relation_encoders = nn.ModuleList()
            self.scalar_heads = nn.ModuleList()

        if variant == "conditional_feature":
            self.feature_heads = nn.ModuleList(
                [nn.Linear(relation_dim, semantic_dim) for _ in MODALITIES]
            )
        else:
            self.feature_heads = nn.ModuleList()

        self.classifier = nn.Linear(4 * semantic_dim, num_classes)

    def relation_function(
        self,
        modality_index: int,
        h_target: torch.Tensor,
        h_source: torch.Tensor,
        *,
        identity_function: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return edge scalar a and feature gate g for source -> target."""
        if self.variant == "uniform":
            a = h_source.new_ones((h_source.size(0), 1))
            g = h_source.new_ones(h_source.shape)
            return a, g
        if self.variant == "similarity_scalar":
            similarity = F.cosine_similarity(h_target, h_source, dim=-1).unsqueeze(-1)
            a = self.scalar_gate_scale * torch.sigmoid(
                self.similarity_alpha[modality_index] * similarity
                + self.similarity_beta[modality_index]
            )
            g = h_source.new_ones(h_source.shape)
            return a, g

        descriptor = relation_descriptor(h_target, h_source)
        relation = self.relation_encoders[modality_index](descriptor)
        a = self.scalar_gate_scale * torch.sigmoid(self.scalar_heads[modality_index](relation))
        if self.variant == "conditional_feature" and not identity_function:
            g = 1.0 + self.feature_modulation_scale * torch.tanh(
                self.feature_heads[modality_index](relation)
            )
        else:
            g = h_source.new_ones(h_source.shape)
        return a, g

    def edge_messages(
        self,
        modality_index: int,
        h_target: torch.Tensor,
        h_source: torch.Tensor,
        *,
        identity_function: bool = False,
        gate_a: torch.Tensor | None = None,
        gate_g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gate_a is None or gate_g is None:
            a, g = self.relation_function(
                modality_index, h_target, h_source, identity_function=identity_function
            )
        else:
            a, g = gate_a, gate_g
            if identity_function:
                g = torch.ones_like(g)
        message = a * (g * h_source)
        return message, a, g

    def parameter_counts(self) -> dict[str, int]:
        classifier = sum(p.numel() for p in self.classifier.parameters())
        relation = sum(
            p.numel()
            for name, p in self.named_parameters()
            if not name.startswith("classifier.")
        )
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"relation_parameter_count": relation, "classifier_parameter_count": classifier, "total_trainable_parameter_count": total}


def same_relation_encoder_definition(left: P02Model, right: P02Model) -> bool:
    """Check V2/V3 relation encoder module structures and tensor shapes."""
    if len(left.relation_encoders) != len(right.relation_encoders):
        return False
    for encoder_l, encoder_r in zip(left.relation_encoders, right.relation_encoders, strict=True):
        if type(encoder_l) is not type(encoder_r):
            return False
        if [(type(m), tuple(p.shape for p in m.parameters())) for m in encoder_l.modules()] != [
            (type(m), tuple(p.shape for p in m.parameters())) for m in encoder_r.modules()
        ]:
            return False
    return True
