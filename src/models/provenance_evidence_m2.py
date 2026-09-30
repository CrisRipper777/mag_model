from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.interaction_provenance_m1 import Model as M1Model


class Model(M1Model):
    """M1 Stage I/II with node-local retrieval over semantic evidence tokens."""

    EVIDENCE_DIM = 128
    NUM_HEADS = 4
    NUM_ORDERS = 3
    MODALITIES = ("text", "visual")

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        self.evidence_dim = int(cfg.model.get("evidence_dim", self.EVIDENCE_DIM))
        self.evidence_num_heads = int(
            cfg.model.get("evidence_num_heads", self.NUM_HEADS)
        )
        self.evidence_dropout = float(cfg.model.get("evidence_dropout", 0.1))
        self.provenance_conditioning = bool(
            cfg.model.get("provenance_conditioning", True)
        )
        if self.hidden_dim != self.EVIDENCE_DIM or self.evidence_dim != self.EVIDENCE_DIM:
            raise ValueError("M2 fixes hidden_dim=evidence_dim=128")
        if self.evidence_num_heads != self.NUM_HEADS:
            raise ValueError("M2 fixes evidence_num_heads=4")
        if self.evidence_dim % self.evidence_num_heads:
            raise ValueError("evidence_dim must be divisible by evidence_num_heads")
        if self.max_order != self.NUM_ORDERS or self.provenance_dim != 32:
            raise ValueError("M2 fixes max_order=3 and provenance_dim=32")
        if not 0.0 <= self.evidence_dropout < 1.0:
            raise ValueError("evidence_dropout must be in [0, 1)")

        d = self.evidence_dim
        g = self.provenance_dim
        for modality in self.MODALITIES:
            setattr(self, f"delta_projection_{modality}", nn.Linear(d, d))
            setattr(self, f"delta_norm_{modality}", nn.LayerNorm(d))
            setattr(self, f"provenance_projection_{modality}", nn.Linear(g, d))
            setattr(self, f"provenance_norm_{modality}", nn.LayerNorm(d))
            setattr(self, f"key_provenance_{modality}", nn.Linear(d, d, bias=False))
            setattr(self, f"key_interaction_{modality}", nn.Linear(d, d, bias=False))
            setattr(self, f"key_norm_{modality}", nn.LayerNorm(d))
            setattr(self, f"value_interaction_{modality}", nn.Linear(d, d, bias=False))
            setattr(self, f"value_norm_{modality}", nn.LayerNorm(d))

        # Kept in fixed order for identical M2-S / M2-P initialization.
        self.order_embedding = nn.Parameter(torch.empty(self.NUM_ORDERS, d))
        self.modality_embedding = nn.Parameter(torch.empty(2, d))
        nn.init.normal_(self.order_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.null_key = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.null_key, mean=0.0, std=0.02)

        for modality in self.MODALITIES:
            nn.init.zeros_(getattr(self, f"key_provenance_{modality}").weight)
            nn.init.zeros_(getattr(self, f"key_interaction_{modality}").weight)
            nn.init.zeros_(getattr(self, f"value_interaction_{modality}").weight)
            setattr(self, f"{modality}_query", nn.Linear(d, d, bias=False))
            setattr(self, f"{modality}_query_norm", nn.LayerNorm(d))

        self.query_head_projection = nn.Linear(d, d, bias=False)
        self.key_head_projection = nn.Linear(d, d, bias=False)
        self.value_head_projection = nn.Linear(d, d, bias=False)
        self.output_projection = nn.Linear(d, d, bias=False)
        self.attention_dropout = nn.Dropout(self.evidence_dropout)
        self.retrieval_dropout = nn.Dropout(self.evidence_dropout)
        self.text_residual_norm = nn.LayerNorm(d)
        self.visual_residual_norm = nn.LayerNorm(d)
        self.final_fusion = nn.Linear(2 * d, d)
        self.final_fusion_norm = nn.LayerNorm(d)
        self.final_dropout = nn.Dropout(self.dropout)
        self.out_dim = d

        # These inherited readout parameters are intentionally frozen and never
        # participate in the M2 representation. M1._compute still evaluates
        # them internally to preserve the frozen Stage I/II helper contract.
        self.gamma_logits_text.requires_grad_(False)
        self.gamma_logits_visual.requires_grad_(False)
        for parameter in self.fusion.parameters():
            parameter.requires_grad_(False)
        for parameter in self.provenance_adapter.parameters():
            parameter.requires_grad_(False)

    @property
    def legacy_frozen_parameter_count(self) -> int:
        return (
            self.gamma_logits_text.numel()
            + self.gamma_logits_visual.numel()
            + sum(parameter.numel() for parameter in self.fusion.parameters())
            + sum(parameter.numel() for parameter in self.provenance_adapter.parameters())
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _get_stage12(stage12: dict[str, Any]) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]]:
        semantic = {m: stage12[f"S_{m}"] for m in Model.MODALITIES}
        provenance = {m: stage12[f"G_{m}"] for m in Model.MODALITIES}
        return semantic, provenance

    def _make_evidence(
        self,
        semantic_states: dict[str, list[torch.Tensor]],
        provenance_states: dict[str, list[torch.Tensor]],
        *,
        use_provenance: bool | None = None,
    ) -> dict[str, Any]:
        """Create fixed text-k1..3, visual-k1..3 tokens and a zero-value null."""
        if use_provenance is None:
            use_provenance = self.provenance_conditioning
        keys_semantic: list[torch.Tensor] = []
        values_semantic: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        deltas: dict[str, list[torch.Tensor]] = {}
        projected_provenance: dict[str, list[torch.Tensor] | None] = {}
        for modality_index, modality in enumerate(self.MODALITIES):
            states = semantic_states[modality]
            modality_deltas = [states[k] - states[k - 1] for k in range(1, 4)]
            deltas[modality] = modality_deltas
            content = [
                getattr(self, f"delta_norm_{modality}")(
                    getattr(self, f"delta_projection_{modality}")(delta)
                )
                for delta in modality_deltas
            ]
            p_values = None
            if use_provenance:
                p_values = [
                    getattr(self, f"provenance_norm_{modality}")(
                        getattr(self, f"provenance_projection_{modality}")(
                            provenance_states[modality][order]
                        )
                    )
                    for order in range(1, 4)
                ]
            projected_provenance[modality] = p_values
            modality_embedding = self.modality_embedding[modality_index]
            for order in range(3):
                c = content[order]
                order_embedding = self.order_embedding[order]
                key_sem = getattr(self, f"key_norm_{modality}")(
                    c + order_embedding + modality_embedding
                )
                value_sem = getattr(self, f"value_norm_{modality}")(c)
                keys_semantic.append(key_sem)
                values_semantic.append(value_sem)
                if use_provenance:
                    p = p_values[order]
                    key = getattr(self, f"key_norm_{modality}")(
                        c
                        + getattr(self, f"key_provenance_{modality}")(p)
                        + getattr(self, f"key_interaction_{modality}")(c * p)
                        + order_embedding
                        + modality_embedding
                    )
                    value = getattr(self, f"value_norm_{modality}")(
                        c + getattr(self, f"value_interaction_{modality}")(c * p)
                    )
                else:
                    key, value = key_sem, value_sem
                keys.append(key)
                values.append(value)

        keys_sem = torch.stack(keys_semantic, dim=1)
        values_sem = torch.stack(values_semantic, dim=1)
        keys_tensor = torch.stack(keys, dim=1)
        values_tensor = torch.stack(values, dim=1)
        null_key = self.null_key.expand(keys_tensor.size(0), 1, -1)
        null_value = values_tensor.new_zeros((values_tensor.size(0), 1, self.evidence_dim))
        return {
            "deltas": deltas,
            "projected_provenance": projected_provenance,
            "keys_semantic": torch.cat((keys_sem, null_key), dim=1),
            "values_semantic": torch.cat((values_sem, null_value), dim=1),
            "keys": torch.cat((keys_tensor, null_key), dim=1),
            "values": torch.cat((values_tensor, null_value), dim=1),
        }

    def _queries(self, h0: dict[str, torch.Tensor]) -> torch.Tensor:
        query_text = self.text_query_norm(self.text_query(h0["text"]))
        query_visual = self.visual_query_norm(self.visual_query(h0["visual"]))
        return torch.stack((query_text, query_visual), dim=1)

    def _same_modality_mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((2, 7), dtype=torch.bool, device=device)
        mask[0, :3] = True
        mask[0, 6] = True
        mask[1, 3:6] = True
        mask[1, 6] = True
        return mask

    def _retrieve(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        same_modality_only: bool = False,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n, query_count, _ = queries.shape
        heads = self.evidence_num_heads
        head_dim = self.evidence_dim // heads
        q = self.query_head_projection(queries).reshape(n, query_count, heads, head_dim)
        k = self.key_head_projection(keys).reshape(n, 7, heads, head_dim)
        v = self.value_head_projection(values).reshape(n, 7, heads, head_dim)
        scores = torch.einsum("nqhd,nthd->nqht", q, k) / math.sqrt(head_dim)
        if same_modality_only:
            allowed = self._same_modality_mask(scores.device)
            scores = scores.masked_fill(~allowed[None, :, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        dropped_attention = self.attention_dropout(attention)
        retrieved = torch.einsum("nqht,nthd->nqhd", dropped_attention, v)
        retrieved = self.output_projection(retrieved.reshape(n, query_count, self.evidence_dim))
        return retrieved, attention if return_attention else None

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
            queries,
            evidence["keys"],
            evidence["values"],
            same_modality_only=same_modality_only,
            return_attention=return_attention,
        )
        z_text = self.text_residual_norm(h0["text"] + self.retrieval_dropout(retrieval[:, 0]))
        z_visual = self.visual_residual_norm(h0["visual"] + self.retrieval_dropout(retrieval[:, 1]))
        z = self.final_fusion_norm(self.final_fusion(torch.cat((z_text, z_visual), dim=-1)))
        z = self.final_dropout(F.gelu(z))
        return {
            "queries": queries,
            "retrieval": retrieval,
            "attention": attention,
            "Z_text": z_text,
            "Z_visual": z_visual,
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
    ):
        """Run full M2 or a frozen Stage-III counterfactual for audit scripts."""
        if edge_index is None:
            raise ValueError("provenance_evidence_m2 requires the physical edge_index")
        valid = {"full", "provenance_off", "provenance_shuffle", "order_mismatch", "same_modality_only"}
        if intervention not in valid:
            raise ValueError(f"Unsupported M2 intervention: {intervention}")
        stage12 = (
            stage12_override
            if stage12_override is not None
            else super()._compute(x, edge_index, provenance_mode="adapter_off")
        )
        semantic_states, provenance_states = self._get_stage12(stage12)
        if intervention == "provenance_shuffle":
            permutation = self._history_shuffle_permutation(stage12["degree"], shuffle_seed)
            provenance_states = {
                modality: [state[permutation] for state in provenance_states[modality]]
                for modality in self.MODALITIES
            }
        elif intervention == "order_mismatch":
            provenance_states = {
                modality: [
                    provenance_states[modality][0],
                    provenance_states[modality][2],
                    provenance_states[modality][3],
                    provenance_states[modality][1],
                ]
                for modality in self.MODALITIES
            }
        use_provenance = self.provenance_conditioning and intervention != "provenance_off"
        evidence = self._make_evidence(
            semantic_states, provenance_states, use_provenance=use_provenance
        )
        represented = self._represent(
            stage12,
            evidence,
            same_modality_only=intervention == "same_modality_only",
            return_attention=return_attention,
        )
        if return_details:
            return {"stage12": stage12, "evidence": evidence, **represented}
        return represented["z"]

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            raise ValueError("provenance_evidence_m2 requires the physical edge_index")
        z = self.forward_with_intervention(x, edge_index, intervention="full")
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        del batch_size
        if edge_index is None:
            raise ValueError("provenance_evidence_m2 requires the physical edge_index")
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        return self.forward(x.to(device), edge_index.to(device))[0].detach().cpu()

    @staticmethod
    def _distribution(values: torch.Tensor, quantiles=(0.5, 0.9)) -> dict[str, float]:
        values = values.detach().float().reshape(-1)
        if not values.numel():
            return {"mean": 0.0, **{f"p{int(q * 100)}": 0.0 for q in quantiles}}
        q_values = torch.quantile(values, values.new_tensor(quantiles))
        return {
            "mean": float(values.mean().item()),
            **{f"p{int(q * 100)}": float(qv.item()) for q, qv in zip(quantiles, q_values, strict=True)},
        }

    @staticmethod
    def _relative_change(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left - right).norm(dim=-1) / (right.norm(dim=-1) + 1e-12)

    @torch.no_grad()
    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, Any]:
        self.eval()
        device = next(self.parameters()).device
        x_device, edge_device = x.to(device), edge_index.to(device)
        details = self.forward_with_intervention(
            x_device, edge_device, intervention="full", return_details=True
        )
        stage12, evidence = details["stage12"], details["evidence"]
        node_count = int(x_device.size(0))
        sample_count = min(node_count, 8192)
        sample_indices = (
            torch.linspace(0, node_count - 1, steps=sample_count, device=device).round().long()
            if sample_count else torch.empty(0, device=device, dtype=torch.long)
        )
        sampled_queries = details["queries"][sample_indices]
        _, attention = self._retrieve(
            sampled_queries,
            evidence["keys"][sample_indices],
            evidence["values"][sample_indices],
            return_attention=True,
        )
        alpha = attention.mean(dim=2)  # [sampled nodes, query modality, token]
        token_names = [f"{m}-k{k}" for m in self.MODALITIES for k in range(1, 4)] + ["null"]
        attention_report: dict[str, Any] = {}
        for query_index, query_name in enumerate(self.MODALITIES):
            a = alpha[:, query_index, :]
            structural = a[:, :6]
            entropy = -(a * (a + 1e-12).log()).sum(dim=-1)
            token_mass = {name: float(a[:, i].mean().item()) for i, name in enumerate(token_names)}
            same_slice = slice(0, 3) if query_index == 0 else slice(3, 6)
            cross_slice = slice(3, 6) if query_index == 0 else slice(0, 3)
            node_variance = a.var(dim=0, unbiased=False)
            attention_report[query_name] = {
                "token_mass": token_mass,
                "same_modality_mass": float(a[:, same_slice].sum(dim=-1).mean().item()),
                "cross_modality_mass": float(a[:, cross_slice].sum(dim=-1).mean().item()),
                "null_mass": float(a[:, 6].mean().item()),
                "order_mass": {
                    f"order{k}": float(structural[:, [k - 1, k + 2]].sum(dim=-1).mean().item())
                    for k in range(1, 4)
                },
                "entropy": self._distribution(entropy, (0.5, 0.9)),
                "effective_tokens": self._distribution(entropy.exp(), (0.5, 0.9)),
                "nodewise_attention_weight_variance_mean": float(node_variance.mean().item()),
                "nodewise_attention_weight_variance_by_token": {
                    name: float(node_variance[i].item()) for i, name in enumerate(token_names)
                },
            }

        increment_report, conditioning_report, residual_report = {}, {}, {}
        structural_change = {}
        for modality in self.MODALITIES:
            increment_report[modality] = {}
            conditioning_report[modality] = {}
            residual_report[modality] = self._distribution(
                details["retrieval"][:, 0 if modality == "text" else 1].norm(dim=-1)
                / (stage12[f"H0_{modality}"].norm(dim=-1) + 1e-12),
                (0.5, 0.9),
            )
            z_modality = details[f"Z_{modality}"]
            structural_change[modality] = self._distribution(
                1.0 - F.cosine_similarity(
                    stage12[f"H0_{modality}"], z_modality, dim=-1, eps=1e-12
                ),
                (0.5, 0.9),
            )
            for order, delta in enumerate(evidence["deltas"][modality], start=1):
                increment_report[modality][f"k{order}"] = self._distribution(delta.norm(dim=-1), (0.5, 0.9))
                if self.provenance_conditioning:
                    key_sem = evidence["keys_semantic"][:, (0 if modality == "text" else 3) + order - 1]
                    value_sem = evidence["values_semantic"][:, (0 if modality == "text" else 3) + order - 1]
                    key_prov = evidence["keys"][:, (0 if modality == "text" else 3) + order - 1]
                    value_prov = evidence["values"][:, (0 if modality == "text" else 3) + order - 1]
                    conditioning_report[modality][f"k{order}"] = {
                        "key_relative_change": self._distribution(self._relative_change(key_prov, key_sem), (0.5, 0.9)),
                        "value_relative_change": self._distribution(self._relative_change(value_prov, value_sem), (0.5, 0.9)),
                    }
                else:
                    conditioning_report[modality][f"k{order}"] = None

        collapse = {}
        for modality, report in attention_report.items():
            masses = list(report["token_mass"].values())
            max_mass = max(masses) if masses else 0.0
            variance = report["nodewise_attention_weight_variance_mean"]
            collapse[modality] = {
                "null_collapse": report["null_mass"] >= 0.95,
                "single_token_collapse": max_mass >= 0.95,
                "node_invariant_retrieval": variance <= 1e-6,
                "max_mean_token_mass": max_mass,
            }
        return {
            "variant": "M2-P" if self.provenance_conditioning else "M2-S",
            "provenance_conditioning": self.provenance_conditioning,
            "parameter_counts": {
                "trainable": self.trainable_parameter_count(),
                "legacy_frozen": self.legacy_frozen_parameter_count,
                "all": sum(parameter.numel() for parameter in self.parameters()),
            },
            "legacy_readout_frozen_statement": (
                "Legacy parameters remain only because M2 inherits the frozen M1 implementation for correctness; "
                "they are not part of the M2 conceptual architecture and should be removed in the later architecture-freeze refactor."
            ),
            "delta_s_norms": increment_report,
            "provenance_conditioned_token_change": conditioning_report,
            "retrieval_residual_ratio": residual_report,
            "final_structural_change": structural_change,
            "attention": attention_report,
            "collapse_checks": collapse,
            "attention_sampled_nodes": sample_count,
            "num_nodes": node_count,
        }
