from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.relation_transport_m0 import (
    Model as M01Model,
    PNAAggregator,
)


class Model(M01Model):
    """M0.1 semantic transport with a parallel interaction-provenance stream."""

    PROVENANCE_DIM = 32
    SHUFFLE_BUCKETS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, None))

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        if self.transport_mode != "orthogonal":
            raise ValueError("M1 fixes transport_mode='orthogonal'")
        if bool(cfg.model.get("relation_context", False)):
            raise ValueError("M1 fixes relation_context=false")
        if str(cfg.model.get("provenance_composition", "distmult")).lower() != "distmult":
            raise ValueError("M1 fixes provenance_composition='distmult'")
        if str(cfg.model.get("provenance_adapter", "residual")).lower() != "residual":
            raise ValueError("M1 fixes provenance_adapter='residual'")

        self.provenance_dim = int(cfg.model.get("provenance_dim", self.PROVENANCE_DIM))
        if self.provenance_dim != self.PROVENANCE_DIM:
            raise ValueError("M1 pilot fixes provenance_dim=32")
        provenance_dropout = float(cfg.model.get("provenance_dropout", 0.1))

        # These modules are deliberately initialized only after M0.1's complete
        # constructor, preserving the initialization of every shared parameter.
        for modality in ("text", "visual"):
            setattr(
                self,
                f"{modality}_relation_to_provenance",
                nn.Linear(self.relation_dim, self.provenance_dim),
            )
            setattr(
                self,
                f"{modality}_effect_to_provenance",
                nn.Linear(self.hidden_dim, self.provenance_dim),
            )
            setattr(
                self,
                f"{modality}_interaction_cross",
                nn.Linear(self.provenance_dim, self.provenance_dim),
            )
            setattr(
                self,
                f"{modality}_relation_provenance_norm",
                nn.LayerNorm(self.provenance_dim),
            )
            setattr(
                self,
                f"{modality}_effect_provenance_norm",
                nn.LayerNorm(self.provenance_dim),
            )
            setattr(
                self,
                f"{modality}_atom_norm",
                nn.LayerNorm(self.provenance_dim),
            )
            setattr(
                self,
                f"{modality}_provenance_aggregator",
                PNAAggregator(self.provenance_dim, provenance_dropout),
            )
            setattr(
                self,
                f"{modality}_provenance_combine",
                nn.Linear(2 * self.provenance_dim, self.provenance_dim),
            )
            setattr(
                self,
                f"{modality}_provenance_state_norm",
                nn.LayerNorm(self.provenance_dim),
            )

        self.provenance_adapter = nn.Linear(
            2 * self.provenance_dim, self.hidden_dim, bias=False
        )
        nn.init.zeros_(self.provenance_adapter.weight)

    def _interaction_atom(
        self,
        relation: torch.Tensor,
        effect: torch.Tensor,
        modality: str,
        mode: str = "full",
    ) -> torch.Tensor:
        """Encode one latent relation together with its realized M0.1 message."""
        if modality not in {"text", "visual"}:
            raise ValueError(f"Unsupported modality: {modality}")
        if mode not in {"full", "relation_off", "effect_off"}:
            raise ValueError(f"Unsupported interaction atom mode: {mode}")
        r = getattr(self, f"{modality}_relation_provenance_norm")(
            getattr(self, f"{modality}_relation_to_provenance")(relation)
        )
        e = getattr(self, f"{modality}_effect_provenance_norm")(
            getattr(self, f"{modality}_effect_to_provenance")(effect)
        )
        if mode == "relation_off":
            r = torch.zeros_like(r)
        elif mode == "effect_off":
            e = torch.zeros_like(e)
        cross = getattr(self, f"{modality}_interaction_cross")(r * e)
        return torch.tanh(
            getattr(self, f"{modality}_atom_norm")(r + e + cross)
        )

    @staticmethod
    def _compose_path_history(
        previous_history: torch.Tensor,
        atom: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        return previous_history[source] * atom

    @classmethod
    def _degree_bucket(cls, degree: int) -> int:
        for bucket_index, (lower, upper) in enumerate(cls.SHUFFLE_BUCKETS):
            if degree >= lower and (upper is None or degree <= upper):
                return bucket_index
        return -1

    @classmethod
    def _history_shuffle_permutation(
        cls, degree: torch.Tensor, seed: int
    ) -> torch.Tensor:
        """Shuffle node history only within the protocol's physical-degree buckets."""
        permutation = torch.arange(int(degree.numel()), device=degree.device)
        degree_cpu = degree.detach().cpu().long()
        for bucket_index, (lower, upper) in enumerate(cls.SHUFFLE_BUCKETS):
            if upper is None:
                mask = degree_cpu >= lower
            else:
                mask = (degree_cpu >= lower) & (degree_cpu <= upper)
            indices = mask.nonzero(as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + bucket_index)
            shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
            permutation[indices.to(permutation.device)] = shuffled.to(permutation.device)
        return permutation

    def _provenance_step(
        self,
        previous: torch.Tensor,
        relation: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        state: torch.Tensor,
        conductance: torch.Tensor,
        angles: torch.Tensor | None,
        degree: torch.Tensor,
        scalers: torch.Tensor,
        response: nn.Module,
        modality: str,
        mode: str,
        history_mode: str,
        permutation: torch.Tensor | None,
        order: int,
        collect_diagnostics: bool,
        sample_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if history_mode == "history_off" and order > 1:
            history = torch.ones_like(previous)
        elif history_mode == "history_shuffle" and order > 1:
            if permutation is None:
                raise RuntimeError("history shuffle permutation was not prepared")
            history = previous[permutation]
        else:
            history = previous

        aggregator = getattr(self, f"{modality}_provenance_aggregator")
        relation_mode = mode if mode in {"relation_off", "effect_off"} else "full"
        atom_samples: list[torch.Tensor] = []

        def path_messages():
            for start in range(0, int(source.numel()), self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, int(source.numel()))
                edge_source = source[start:end]
                edge_target = target[start:end]
                edge_conductance = conductance[start:end]

                def realized_effect(
                    full_state, source_index, target_index, edge_weight, angle_values
                ):
                    source_state = full_state[source_index]
                    if angle_values is not None:
                        source_state = self._givens_transport(source_state, angle_values)
                    discrepancy = source_state - full_state[target_index]
                    # This is the same realized semantic edge message consumed
                    # by M0.1's PNA; no second effect proxy is introduced.
                    return edge_weight[:, None] * response(discrepancy)

                angle_values = angles[start:end] if angles is not None else None
                if self.training and torch.is_grad_enabled():
                    effect = checkpoint(
                        realized_effect,
                        state,
                        edge_source,
                        edge_target,
                        edge_conductance,
                        angle_values,
                        use_reentrant=False,
                    )
                else:
                    effect = realized_effect(
                        state, edge_source, edge_target, edge_conductance, angle_values
                    )
                atom = self._interaction_atom(
                    relation[start:end], effect, modality, relation_mode
                )
                path = self._compose_path_history(history, atom, source[start:end])
                if collect_diagnostics and sample_indices.numel():
                    selected = sample_indices[
                        (sample_indices >= start) & (sample_indices < end)
                    ] - start
                    if selected.numel():
                        atom_samples.append(atom[selected])
                yield path, target[start:end]

        aggregate = aggregator(
            path_messages(), degree, scalers, previous.dtype, previous.device
        )
        combined = torch.cat((history, aggregate), dim=-1)
        next_state = getattr(self, f"{modality}_provenance_state_norm")(
            getattr(self, f"{modality}_provenance_combine")(combined)
        )
        samples = torch.cat(atom_samples, dim=0) if atom_samples else None
        return next_state, samples

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
        response: nn.Module,
        aggregator: PNAAggregator,
    ) -> torch.Tensor:
        """M0.1 semantic aggregation with exact chunk recomputation in training."""
        del h0  # retained in the inherited helper signature for compatibility

        def messages():
            for start in range(0, int(source.numel()), self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, int(source.numel()))
                source_index = source[start:end]
                target_index = target[start:end]
                edge_weight = conductance[start:end]

                def semantic_message(full_state, src, dst, weight, angle_values):
                    source_state = full_state[src]
                    if angle_values is not None:
                        source_state = self._givens_transport(source_state, angle_values)
                    discrepancy = source_state - full_state[dst]
                    return weight[:, None] * response(discrepancy)

                angle_values = angles[start:end] if angles is not None else None
                if self.training and torch.is_grad_enabled():
                    message = checkpoint(
                        semantic_message,
                        state,
                        source_index,
                        target_index,
                        edge_weight,
                        angle_values,
                        use_reentrant=False,
                    )
                else:
                    message = semantic_message(
                        state, source_index, target_index, edge_weight, angle_values
                    )
                yield message, target_index

        return aggregator(messages(), degree, scalers, state.dtype, state.device)

    def _compute(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        provenance_mode: str = "full",
        history_shuffle_seed: int | None = None,
        collect_diagnostics: bool = False,
        semantic_cache: dict | None = None,
    ):
        """Compute a byte-for-formula M0.1 semantic stream plus M1 provenance."""
        if provenance_mode not in {
            "full",
            "adapter_off",
            "history_off",
            "history_shuffle",
            "relation_off",
            "effect_off",
        }:
            raise ValueError(f"Unsupported provenance mode: {provenance_mode}")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected x with shape [num_nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        if semantic_cache is None:
            clean_edge_index, degree, _mean_log_degree, scalers = self._get_graph(
                edge_index.to(device=x.device), int(x.size(0))
            )
            source, target = clean_edge_index

            # Keep the semantic computation in the exact M0.1 operation order.
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
            base_z = self.fusion(torch.cat((z_text, z_visual), dim=-1))
            semantic_cache = {
                "H0_text": h0_text,
                "H0_visual": h0_visual,
                "S_text": states_text,
                "S_visual": states_visual,
                "base_z": base_z,
                "conductance_text": conductance_text,
                "conductance_visual": conductance_visual,
                "angles_text": angles_text,
                "angles_visual": angles_visual,
                "R_text": relation_text,
                "R_visual": relation_visual,
                "source": source,
                "target": target,
                "degree": degree,
                "scalers": scalers,
                "gamma_text": gamma_text,
                "gamma_visual": gamma_visual,
            }
        else:
            h0_text = semantic_cache["H0_text"]
            h0_visual = semantic_cache["H0_visual"]
            states_text = semantic_cache["S_text"]
            states_visual = semantic_cache["S_visual"]
            base_z = semantic_cache["base_z"]
            conductance_text = semantic_cache["conductance_text"]
            conductance_visual = semantic_cache["conductance_visual"]
            angles_text = semantic_cache["angles_text"]
            angles_visual = semantic_cache["angles_visual"]
            relation_text = semantic_cache["R_text"]
            relation_visual = semantic_cache["R_visual"]
            source = semantic_cache["source"]
            target = semantic_cache["target"]
            degree = semantic_cache["degree"]
            scalers = semantic_cache["scalers"]
            gamma_text = semantic_cache["gamma_text"]
            gamma_visual = semantic_cache["gamma_visual"]

        ones_text = torch.ones(
            (int(x.size(0)), self.provenance_dim), device=x.device, dtype=h0_text.dtype
        )
        ones_visual = torch.ones_like(ones_text)
        provenance_text = [ones_text]
        provenance_visual = [ones_visual]
        permutation = None
        if provenance_mode == "history_shuffle":
            permutation = self._history_shuffle_permutation(
                degree, 3407 if history_shuffle_seed is None else history_shuffle_seed
            )
        sample_count = min(int(source.numel()), 8192) if collect_diagnostics else 0
        sample_indices = (
            torch.linspace(0, source.numel() - 1, steps=sample_count, device=x.device)
            .round()
            .long()
            if sample_count
            else torch.empty(0, device=x.device, dtype=torch.long)
        )
        atom_samples = {"text": [], "visual": []}

        for order in range(1, self.max_order + 1):
            next_text, atom_text = self._provenance_step(
                provenance_text[-1],
                relation_text,
                source,
                target,
                states_text[order - 1],
                conductance_text,
                angles_text,
                degree,
                scalers,
                self.text_response,
                "text",
                provenance_mode,
                provenance_mode,
                permutation,
                order,
                collect_diagnostics,
                sample_indices,
            )
            next_visual, atom_visual = self._provenance_step(
                provenance_visual[-1],
                relation_visual,
                source,
                target,
                states_visual[order - 1],
                conductance_visual,
                angles_visual,
                degree,
                scalers,
                self.visual_response,
                "visual",
                provenance_mode,
                provenance_mode,
                permutation,
                order,
                collect_diagnostics,
                sample_indices,
            )
            provenance_text.append(next_text)
            provenance_visual.append(next_visual)
            atom_samples["text"].append(atom_text)
            atom_samples["visual"].append(atom_visual)

        provenance_features = torch.cat(
            (provenance_text[-1], provenance_visual[-1]), dim=-1
        )
        provenance_residual = self.provenance_adapter(provenance_features)
        z = base_z if provenance_mode == "adapter_off" else base_z + provenance_residual
        return {
            "z": z,
            "base_z": base_z,
            "provenance_residual": provenance_residual,
            "H0_text": h0_text,
            "H0_visual": h0_visual,
            "S_text": states_text,
            "S_visual": states_visual,
            "G_text": provenance_text,
            "G_visual": provenance_visual,
            "conductance_text": conductance_text,
            "conductance_visual": conductance_visual,
            "angles_text": angles_text,
            "angles_visual": angles_visual,
            "R_text": relation_text,
            "R_visual": relation_visual,
            "source": source,
            "target": target,
            "degree": degree,
            "gamma_text": gamma_text,
            "gamma_visual": gamma_visual,
            "atom_samples": atom_samples,
            "history_permutation": permutation,
            "semantic_cache": semantic_cache,
        }

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            raise ValueError("interaction_provenance_m1 requires the physical edge_index")
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
        del batch_size
        if edge_index is None:
            raise ValueError("interaction_provenance_m1 requires the physical edge_index")
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()

    @staticmethod
    def _distribution(values: torch.Tensor, quantiles=(0.5, 0.9)) -> dict[str, float]:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return {"mean": 0.0, **{f"p{int(q * 100)}": 0.0 for q in quantiles}}
        q_values = torch.quantile(values, values.new_tensor(quantiles))
        return {
            "mean": float(values.mean().item()),
            **{
                f"p{int(q * 100)}": float(qv.item())
                for q, qv in zip(quantiles, q_values, strict=True)
            },
        }

    @staticmethod
    def _norm_summary(values: torch.Tensor) -> dict[str, float]:
        return Model._distribution(values.norm(dim=-1), (0.5, 0.95))

    @staticmethod
    def _diversity(values: torch.Tensor) -> dict[str, float]:
        values = values.detach().float()
        if values.dim() != 2 or values.size(0) < 2:
            return {
                "effective_rank": 0.0,
                "top1_energy": 0.0,
                "top5_energy": 0.0,
                "channel_variance_active_fraction": 0.0,
            }
        centered = values - values.mean(dim=0, keepdim=True)
        gram = centered.T @ centered
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
        total = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
        probabilities = eigenvalues / total
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
        variance = centered.var(dim=0, unbiased=False)
        return {
            "effective_rank": float(entropy.exp().item()),
            "top1_energy": float((eigenvalues[:1].sum() / total).item()),
            "top5_energy": float((eigenvalues[:5].sum() / total).item()),
            "channel_variance_active_fraction": float((variance > 1e-12).float().mean().item()),
        }

    @staticmethod
    def _centered_cka(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.detach().float() - left.detach().float().mean(dim=0, keepdim=True)
        right = right.detach().float() - right.detach().float().mean(dim=0, keepdim=True)
        cross = left.T @ right
        left_gram = left.T @ left
        right_gram = right.T @ right
        denominator = left_gram.norm() * right_gram.norm()
        if float(denominator.item()) <= torch.finfo(denominator.dtype).eps:
            return 0.0
        return float((cross.square().sum() / denominator).item())

    @torch.no_grad()
    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        self.eval()
        device = next(self.parameters()).device
        result = self._compute(
            x.to(device), edge_index.to(device), collect_diagnostics=True
        )
        node_count = int(x.size(0))
        node_sample_count = min(node_count, 8192)
        node_indices = (
            torch.linspace(0, node_count - 1, steps=node_sample_count, device=device)
            .round()
            .long()
            if node_sample_count
            else torch.empty(0, device=device, dtype=torch.long)
        )
        cka_count = min(node_count, 4096)
        cka_indices = (
            torch.linspace(0, node_count - 1, steps=cka_count, device=device)
            .round()
            .long()
            if cka_count
            else torch.empty(0, device=device, dtype=torch.long)
        )

        provenance = {}
        atom_diversity = {}
        cka = {}
        for modality in ("text", "visual"):
            states = result[f"G_{modality}"]
            semantic_states = result[f"S_{modality}"]
            provenance[modality] = {
                "state_norms": {
                    f"G{order}": self._norm_summary(state)
                    for order, state in enumerate(states)
                },
                "progressive_change": {},
                "state_diversity": {},
                "interaction_atoms": {},
            }
            for order in range(1, 4):
                previous = states[order - 1]
                current = states[order]
                relative = (current - previous).norm(dim=-1) / (
                    previous.norm(dim=-1) + 1e-12
                )
                cosine = F.cosine_similarity(current, previous, dim=-1, eps=1e-12)
                provenance[modality]["progressive_change"][f"k{order}"] = {
                    "relative_change": self._distribution(relative, (0.5, 0.9)),
                    "cosine_similarity": self._distribution(cosine, (0.1, 0.5)),
                }
                provenance[modality]["state_diversity"][f"G{order}"] = self._diversity(
                    current[node_indices]
                )
                atom = result["atom_samples"][modality][order - 1]
                if atom is not None:
                    atom_metrics = self._diversity(atom)
                    atom_metrics["norm"] = self._distribution(atom.norm(dim=-1), (0.5, 0.9))
                    atom_metrics["per_channel_variance"] = (
                        atom.detach().float().var(dim=0, unbiased=False).cpu().tolist()
                    )
                else:
                    atom_metrics = self._diversity(
                        torch.empty((0, self.provenance_dim), device=device)
                    )
                    atom_metrics["norm"] = self._norm_summary(
                        torch.empty((0, self.provenance_dim), device=device)
                    )
                    atom_metrics["per_channel_variance"] = [0.0] * self.provenance_dim
                provenance[modality]["interaction_atoms"][f"k{order}"] = atom_metrics
                delta_semantic = semantic_states[order] - semantic_states[order - 1]
                cka[f"{modality}_k{order}"] = {
                    "G_vs_delta_S": self._centered_cka(
                        current[cka_indices], delta_semantic[cka_indices]
                    ),
                    "G_vs_S": self._centered_cka(
                        current[cka_indices], semantic_states[order][cka_indices]
                    ),
                }

        joined = torch.cat((result["G_text"][-1], result["G_visual"][-1]), dim=-1)
        residual_ratio = result["provenance_residual"].norm(dim=-1) / (
            result["base_z"].norm(dim=-1) + 1e-12
        )
        adapter_weight_norm = float(self.provenance_adapter.weight.norm().item())

        conductance = {
            modality: self._distribution(result[f"conductance_{modality}"], (0.1, 0.5, 0.9))
            for modality in ("text", "visual")
        }
        angle_abs = {}
        for modality in ("text", "visual"):
            angles = result[f"angles_{modality}"]
            angle_abs[modality] = (
                self._distribution(angles.abs(), (0.1, 0.5, 0.9))
                if angles is not None
                else {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
            )

        return {
            "semantic_norms": {
                modality: {
                    "H0" if order == 0 else f"S{order}": self._norm_summary(state)
                    for order, state in enumerate(result[f"S_{modality}"])
                }
                for modality in ("text", "visual")
            },
            "provenance": provenance,
            "provenance_residual_ratio": self._distribution(
                residual_ratio, (0.5, 0.9)
            ),
            "provenance_adapter_weight_norm": adapter_weight_norm,
            "provenance_feature_norm": self._norm_summary(joined),
            "semantic_provenance_cka": cka,
            "conductance": conductance,
            "abs_angle": angle_abs,
            "gamma_text": result["gamma_text"].cpu().tolist(),
            "gamma_visual": result["gamma_visual"].cpu().tolist(),
            "num_physical_directed_edges": int(result["source"].numel()),
            "atom_sampled_edges": min(int(result["source"].numel()), 8192),
            "diversity_sampled_nodes": node_sample_count,
            "cka_sampled_nodes": cka_count,
        }
