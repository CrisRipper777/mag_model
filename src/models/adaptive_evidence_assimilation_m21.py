from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.provenance_evidence_m2 import Model as M2Model


class Model(M2Model):
    """M2 retrieval with intrinsic, modality-specific evidence assimilation."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        self.assimilation_mode = str(cfg.model.get("assimilation_mode", "adaptive_vector"))
        if self.assimilation_mode not in {"adaptive_vector", "global_scalar"}:
            raise ValueError("assimilation_mode must be adaptive_vector or global_scalar")
        self.assimilation_dropout = nn.Dropout(float(cfg.model.get("assimilation_dropout", 0.1)))
        for modality in self.MODALITIES:
            setattr(self, f"assimilation_evidence_norm_{modality}", nn.LayerNorm(self.evidence_dim))
            setattr(self, f"assimilation_in_{modality}", nn.Linear(4 * self.evidence_dim, self.evidence_dim))
            setattr(self, f"assimilation_out_{modality}", nn.Linear(self.evidence_dim, self.evidence_dim))
            # The adaptive vector field starts at exactly zero, making initial Z=H0.
            nn.init.zeros_(getattr(self, f"assimilation_out_{modality}").weight)
            nn.init.zeros_(getattr(self, f"assimilation_out_{modality}").bias)
        self.assimilation_alpha_text = nn.Parameter(torch.zeros(()))
        self.assimilation_alpha_visual = nn.Parameter(torch.zeros(()))

        # These M2 residual norms are inherited legacy assimilation parameters;
        # they are not part of the M2.1 conceptual architecture.
        for parameter in self.text_residual_norm.parameters():
            parameter.requires_grad_(False)
        for parameter in self.visual_residual_norm.parameters():
            parameter.requires_grad_(False)
        if self.assimilation_mode == "global_scalar":
            for modality in self.MODALITIES:
                for parameter in getattr(self, f"assimilation_in_{modality}").parameters():
                    parameter.requires_grad_(False)
                for parameter in getattr(self, f"assimilation_out_{modality}").parameters():
                    parameter.requires_grad_(False)

        self._active_assimilation_intervention = "full"
        self._active_assimilation_seed = 7311

    @property
    def legacy_frozen_parameter_count(self) -> int:
        return (
            super().legacy_frozen_parameter_count
            + sum(parameter.numel() for parameter in self.text_residual_norm.parameters())
            + sum(parameter.numel() for parameter in self.visual_residual_norm.parameters())
        )

    @staticmethod
    def _centered_diversity(vectors: torch.Tensor) -> dict[str, float]:
        values = vectors.detach().float()
        if values.size(0) == 0:
            return {"centered_effective_rank": 0.0, "top1_energy": 0.0, "top5_energy": 0.0,
                    "mean_channel_variance": 0.0, "active_fraction_abs_gt_0_1": 0.0,
                    "nodewise_norm_variance": 0.0}
        centered = values - values.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(1, centered.size(0))
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
        energy = eigenvalues.sum()
        if float(energy) > 1e-20:
            probs = eigenvalues / energy
            positive = probs[probs > 0]
            effective_rank = float(torch.exp(-(positive * positive.log()).sum()).item())
            top1 = float(probs[:1].sum().item())
            top5 = float(probs[:5].sum().item())
        else:
            effective_rank = top1 = top5 = 0.0
        return {
            "centered_effective_rank": effective_rank,
            "top1_energy": top1,
            "top5_energy": top5,
            "mean_channel_variance": float(values.var(dim=0, unbiased=False).mean().item()),
            "active_fraction_abs_gt_0_1": float((values.abs() > 0.1).float().mean().item()),
            "nodewise_norm_variance": float(values.norm(dim=-1).var(unbiased=False).item()),
        }

    def _assimilation_field(self, modality: str, h0: torch.Tensor, retrieved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = getattr(self, f"assimilation_evidence_norm_{modality}")(retrieved)
        if self.assimilation_mode == "global_scalar":
            alpha = getattr(self, f"assimilation_alpha_{modality}")
            field = torch.tanh(alpha).expand_as(h0)
        else:
            context = torch.cat((h0, u, h0 * u, (h0 - u).abs()), dim=-1)
            hidden = self.assimilation_dropout(F.gelu(getattr(self, f"assimilation_in_{modality}")(context)))
            field = torch.tanh(getattr(self, f"assimilation_out_{modality}")(hidden))
        return u, field

    def _represent(
        self,
        stage12: dict[str, Any],
        evidence: dict[str, Any],
        *,
        same_modality_only: bool = False,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        h0 = {m: stage12[f"H0_{m}"] for m in self.MODALITIES}
        queries = self._queries(h0)
        retrieval, attention = self._retrieve(
            queries, evidence["keys"], evidence["values"],
            same_modality_only=same_modality_only, return_attention=return_attention,
        )
        z_modalities: dict[str, torch.Tensor] = {}
        normalized_retrieval: dict[str, torch.Tensor] = {}
        fields: dict[str, torch.Tensor] = {}
        corrections: dict[str, torch.Tensor] = {}
        for index, modality in enumerate(self.MODALITIES):
            u, field = self._assimilation_field(modality, h0[modality], retrieval[:, index])
            intervention = self._active_assimilation_intervention
            if intervention == "assimilation_off":
                field = torch.zeros_like(field)
            elif intervention == "uniform_node":
                field = field.mean(dim=0, keepdim=True).expand_as(field)
            elif intervention == "dimension_shuffle":
                generator = torch.Generator(device="cpu").manual_seed(int(self._active_assimilation_seed) + index)
                permutation = torch.randperm(self.evidence_dim, generator=generator).to(field.device)
                field = field[:, permutation]
            elif intervention != "full":
                raise ValueError(f"Unsupported assimilation intervention: {intervention}")
            correction = field * u
            normalized_retrieval[modality] = u
            fields[modality] = field
            corrections[modality] = correction
            z_modalities[modality] = h0[modality] + correction

        z = self.final_fusion_norm(
            self.final_fusion(torch.cat((z_modalities["text"], z_modalities["visual"]), dim=-1))
        )
        z = self.final_dropout(F.gelu(z))
        return {
            "queries": queries,
            "retrieval": retrieval,
            "attention": attention,
            "Z_text": z_modalities["text"],
            "Z_visual": z_modalities["visual"],
            "normalized_retrieval": normalized_retrieval,
            "assimilation_field": fields,
            "assimilation_correction": corrections,
            "z": z,
        }

    def forward_with_intervention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        intervention: str = "full",
        shuffle_seed: int = 3407,
        return_details: bool = False,
        return_attention: bool = False,
        stage12_override: dict[str, Any] | None = None,
        assimilation_intervention: str = "full",
        assimilation_seed: int = 7311,
    ):
        if assimilation_intervention not in {"full", "assimilation_off", "uniform_node", "dimension_shuffle"}:
            raise ValueError(f"Unsupported M2.1 assimilation intervention: {assimilation_intervention}")
        previous_intervention = self._active_assimilation_intervention
        previous_seed = self._active_assimilation_seed
        self._active_assimilation_intervention = assimilation_intervention
        self._active_assimilation_seed = int(assimilation_seed)
        try:
            return super().forward_with_intervention(
                x, edge_index, intervention=intervention, shuffle_seed=shuffle_seed,
                return_details=return_details, return_attention=return_attention,
                stage12_override=stage12_override,
            )
        finally:
            self._active_assimilation_intervention = previous_intervention
            self._active_assimilation_seed = previous_seed

    @torch.no_grad()
    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, Any]:
        self.eval()
        report = super().analyze(x, edge_index)
        device = next(self.parameters()).device
        details = self.forward_with_intervention(x.to(device), edge_index.to(device), return_details=True)
        node_count = int(x.size(0))
        sample_count = min(node_count, 8192)
        indices = (torch.linspace(0, node_count - 1, steps=sample_count, device=device).round().long()
                   if sample_count else torch.empty(0, device=device, dtype=torch.long))
        fields_report, compatibility = {}, {}
        for modality in self.MODALITIES:
            field = details["assimilation_field"][modality]
            correction = details["assimilation_correction"][modality]
            h0 = details["stage12"][f"H0_{modality}"]
            u = details["normalized_retrieval"][modality]
            abs_field = field.abs()
            ratio = correction.norm(dim=-1) / (h0.norm(dim=-1) + 1e-12)
            report.setdefault("assimilation_correction_ratio", {})[modality] = self._distribution(ratio, (0.5, 0.9))
            fields_report[modality] = {
                "signed_mean": float(field.mean().item()),
                "absolute_value": self._distribution(abs_field, (0.5, 0.9)),
                "fraction_gt_0_1": float((field > 0.1).float().mean().item()),
                "fraction_lt_minus_0_1": float((field < -0.1).float().mean().item()),
                "fraction_abs_le_0_1": float((abs_field <= 0.1).float().mean().item()),
                "diversity": self._centered_diversity(field[indices]),
                "normalized_retrieval_norm": self._distribution(u.norm(dim=-1), (0.5, 0.9)),
            }
            cosine = F.cosine_similarity(h0, u, dim=-1, eps=1e-12)
            bins = torch.quantile(cosine, torch.tensor([0.2, 0.4, 0.6, 0.8], device=device))
            labels = torch.bucketize(cosine, bins, right=False)
            compatibility[modality] = {}
            for q in range(5):
                mask = labels == q
                compatibility[modality][f"Q{q + 1}"] = {
                    "nodes": int(mask.sum().item()),
                    "compatibility_cosine_mean": float(cosine[mask].mean().item()) if bool(mask.any()) else 0.0,
                    "mean_abs_field": float(abs_field[mask].mean().item()) if bool(mask.any()) else 0.0,
                    "correction_ratio_mean": float(ratio[mask].mean().item()) if bool(mask.any()) else 0.0,
                    "positive_fraction_gt_0_1": float((field[mask] > 0.1).float().mean().item()) if bool(mask.any()) else 0.0,
                    "negative_fraction_lt_minus_0_1": float((field[mask] < -0.1).float().mean().item()) if bool(mask.any()) else 0.0,
                    "near_zero_fraction_abs_le_0_1": float((abs_field[mask] <= 0.1).float().mean().item()) if bool(mask.any()) else 0.0,
                }
        report["assimilation_mode"] = self.assimilation_mode
        report["assimilation_field"] = fields_report
        report["compatibility_quintiles"] = compatibility
        report["legacy_readout_frozen_statement"] = (
            "Inherited legacy M2 assimilation parameters; not part of the M2.1 conceptual architecture."
        )
        report["parameter_counts"]["legacy_frozen"] = self.legacy_frozen_parameter_count
        report["parameter_counts"]["all"] = sum(p.numel() for p in self.parameters())
        return report
