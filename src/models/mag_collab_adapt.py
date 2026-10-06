from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


D = 64
HEADS = 2
HEAD_DIM = 32


def _index_add(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = values.new_zeros((size, *values.shape[1:]))
    if index.numel():
        out.index_add_(0, index, values)
    return out


def _grouped_softmax(logits: torch.Tensor, group: torch.Tensor, valid: torch.Tensor, size: int) -> torch.Tensor:
    """Stable edge softmax over valid edges grouped by receiver; empty groups are zero."""
    if logits.numel() == 0:
        return logits
    work = logits.float()
    mask = valid.reshape((-1,) + (1,) * (logits.ndim - 1))
    masked = work.masked_fill(~mask, -torch.inf)
    maxima = work.new_full((size, *work.shape[1:]), -torch.inf)
    maxima.scatter_reduce_(0, group.reshape((-1,) + (1,) * (work.ndim - 1)).expand_as(work), masked, reduce="amax", include_self=True)
    edge_max = maxima.index_select(0, group)
    safe_max = torch.where(torch.isfinite(edge_max), edge_max, torch.zeros_like(edge_max))
    # Mask before exp: torch.where evaluates both branches, and an invalid
    # extreme logit can otherwise overflow and poison the masked backward path.
    masked_work = work.masked_fill(~mask, -torch.inf)
    exp = torch.exp(masked_work - safe_max)
    denom = _index_add(exp, group, size).index_select(0, group)
    return (exp / denom.clamp_min(torch.finfo(exp.dtype).tiny)).to(logits.dtype)


def _grouped_softmax_with_null(
    edge_logits: torch.Tensor,
    group: torch.Tensor,
    valid: torch.Tensor,
    null_logits: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize edge logits and one node-local NULL logit in each receiver group."""
    if edge_logits.ndim != 2 or null_logits.ndim != 2:
        raise ValueError("edge and NULL logits must have shape [E,H] and [N,H]")
    work, null = edge_logits.float(), null_logits.float()
    if edge_logits.size(0):
        masked = work.masked_fill(~valid[:, None], -torch.inf)
        edge_max = work.new_full((size, work.size(1)), -torch.inf)
        edge_max.scatter_reduce_(0, group[:, None].expand_as(work), masked, reduce="amax", include_self=True)
    else:
        edge_max = work.new_full((size, work.size(1)), -torch.inf)
    maximum = torch.maximum(edge_max, null)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exp_null = torch.exp(null - safe_maximum)
    masked_work = work.masked_fill(~valid[:, None], -torch.inf)
    exp_edge = torch.exp(masked_work - safe_maximum.index_select(0, group)) if edge_logits.size(0) else work
    denom = exp_null + _index_add(exp_edge, group, size)
    null_weight = exp_null / denom.clamp_min(torch.finfo(denom.dtype).tiny)
    edge_weight = exp_edge / denom.index_select(0, group).clamp_min(torch.finfo(denom.dtype).tiny) if edge_logits.size(0) else exp_edge
    return edge_weight.to(edge_logits.dtype), null_weight.to(null_logits.dtype)


def _degree(dst: torch.Tensor, valid: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    result = torch.zeros((n,), dtype=dtype, device=dst.device)
    if dst.numel():
        selected = dst[valid]
        result.index_add_(0, selected, torch.ones_like(selected, dtype=dtype))
    return result


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:  # inference tensors do not expose a version counter
        return None


def _prepare_frozen_graph_inputs(
    x: torch.Tensor, edge_index: torch.Tensor, available: torch.Tensor, text_dim: int
) -> dict[str, torch.Tensor]:
    """Feature/edge-only cache; no training parameter or label enters this object."""
    n = x.size(0)
    src, dst = edge_index
    available_src = available.index_select(0, src) if src.numel() else available.new_empty((0, 2))
    edge_valid = available_src.any(-1) if src.numel() else available.new_empty((0,))
    degrees = torch.stack(
        [_degree(dst, available_src[:, m] if src.numel() else available.new_empty((0,)), n, x.dtype) for m in range(2)]
    )
    degree_common = _degree(dst, edge_valid, n, x.dtype)
    degree_all = _degree(dst, torch.ones_like(edge_valid), n, x.dtype) if src.numel() else x.new_zeros((n,))

    x_t, x_i = x[:, :text_dim], x[:, text_dim:]
    raw = []
    for values, col in ((x_t, 0), (x_i, 1)):
        finite = torch.isfinite(values).all(-1)
        row_valid = available[:, col] & finite
        clean = torch.where(row_valid[:, None] & torch.isfinite(values), values, torch.zeros_like(values))
        raw.append(F.normalize(clean, p=2, dim=-1, eps=1e-8))
    cosines = x.new_zeros((src.numel(), 2))
    # Bound temporary edge-by-input-width storage without changing the full
    # receiver-neighborhood softmax or its reduction domain.
    chunk = 16_384
    for start in range(0, src.numel(), chunk):
        stop = min(start + chunk, src.numel())
        s, d = src[start:stop], dst[start:stop]
        for m in range(2):
            seen = available[d, m] & available[s, m]
            cosine = (raw[m][d] * raw[m][s]).sum(-1)
            cosines[start:stop, m] = torch.where(seen, cosine, torch.zeros_like(cosine))
    observed_pair = torch.stack(
        [available[dst, m] & available[src, m] for m in range(2)], dim=-1
    ) if src.numel() else available.new_empty((0, 2))
    return {
        "available_src": available_src,
        "edge_valid": edge_valid,
        "degree_private": degrees,
        "degree_common": degree_common,
        "degree_all": degree_all,
        "raw_cosines": cosines,
        "observed_pair": observed_pair,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, *, bias: bool = True, last_bias: bool | None = None):
        super().__init__()
        if last_bias is None:
            last_bias = bias
        self.net = nn.Sequential(nn.Linear(in_dim, hidden, bias=bias), nn.GELU(), nn.Linear(hidden, out_dim, bias=last_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PrivatePath(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(D, D, bias=False)
        self.key = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.gamma = nn.Parameter(torch.ones(HEADS))
        self.degree_score = _MLP(2, 8, HEADS)
        self.null_score = _MLP(D, 16, 1)
        self.phi_local = _MLP(D, 128, D)
        self.phi_diff = _MLP(D, 128, D, bias=False, last_bias=False)
        self.gate = _MLP(3 * D + 1, D, 2 * D)
        self.update = _MLP(2 * D, 128, D)
        nn.init.zeros_(self.null_score.net[-1].weight)
        nn.init.constant_(self.null_score.net[-1].bias, float(torch.log(torch.tensor(1.0 / 9.0))))
        nn.init.zeros_(self.gate.net[-1].weight)
        nn.init.zeros_(self.gate.net[-1].bias)
        nn.init.xavier_uniform_(self.update.net[-1].weight, gain=0.1)

    def forward(
        self,
        state: torch.Tensor,
        raw: torch.Tensor,
        available: torch.Tensor,
        edge_index: torch.Tensor,
        tau: float,
        graph_inputs: dict[str, torch.Tensor] | None = None,
        modality: int = 0,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n = state.size(0)
        src, dst = edge_index
        q = self.query(state).view(n, HEADS, HEAD_DIM)
        k = self.key(state).view(n, HEADS, HEAD_DIM)
        v = self.value(state).view(n, HEADS, HEAD_DIM)
        valid = graph_inputs["available_src"][:, modality] if graph_inputs is not None else (available.index_select(0, src) if src.numel() else available.new_empty((0,)))
        degree = graph_inputs["degree_private"][modality] if graph_inputs is not None else _degree(dst, valid, n, state.dtype)
        if src.numel():
            logits = (q[dst] * k[src]).sum(-1) / (HEAD_DIM ** 0.5)
            if graph_inputs is not None:
                cosine = graph_inputs["raw_cosines"][:, modality]
            else:
                raw_norm = F.normalize(raw, p=2, dim=-1, eps=1e-8)
                cosine = (raw_norm[dst] * raw_norm[src]).sum(-1)
                pair_seen = available[dst] & available[src]
                cosine = torch.where(pair_seen, cosine, torch.zeros_like(cosine))
            degree_edge = torch.stack((torch.log1p(degree[dst]), torch.log1p(degree[src])), dim=-1)
            logits = logits + cosine[:, None] * self.gamma[None, :] + self.degree_score(degree_edge)
            retrieval = _grouped_softmax(logits / float(tau), dst, valid, n)
            weighted_v = retrieval.unsqueeze(-1) * v[src]
            mu = _index_add(weighted_v, dst, n).reshape(n, D)
            avg_logits = (logits / float(tau)).mean(-1)
        else:
            logits = state.new_empty((0, HEADS))
            retrieval = logits
            mu = state.new_zeros((n, D))
            avg_logits = state.new_empty((0,))
            cosine = state.new_empty((0,))
        null = self.null_score(state).squeeze(-1)
        null_logits = torch.where(degree > 0, degree.clamp_min(1).log() + null, torch.zeros_like(null))
        if src.numel():
            _, null_prob = _grouped_softmax_with_null(avg_logits[:, None], dst, valid, null_logits[:, None], n)
            rho = 1.0 - null_prob.squeeze(-1)
        else:
            rho = state.new_zeros((n,))
        delta = v.reshape(n, D) - mu
        local = self.phi_local(mu)
        difference = self.phi_diff(delta)
        gate_in = torch.cat((state, mu, delta.abs(), torch.log1p(degree)[:, None]), dim=-1)
        gate = 2.0 * torch.sigmoid(self.gate(gate_in))
        gate_local, gate_diff = gate.chunk(2, dim=-1)
        message = rho[:, None] * self.update(torch.cat((gate_local * local, gate_diff * difference), dim=-1))
        message = torch.where((degree > 0)[:, None], message, torch.zeros_like(message))
        stats = ({
            "rho": rho.detach(), "retrieval": retrieval.detach(), "degree": degree.detach(),
            "gate_local": gate_local.detach(), "gate_diff": gate_diff.detach(),
            "diff_norm": difference.norm(dim=-1).detach(), "raw_cosine": cosine.detach(), "dst": dst.detach(),
        } if collect_diagnostics else {})
        return message, stats


class _CommonPath(nn.Module):
    def __init__(self, typed_value: bool = False):
        super().__init__()
        self.query = nn.ModuleList([nn.Linear(192, D, bias=False) for _ in range(2)])
        self.key = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
        self.relation_same = nn.Parameter(torch.eye(HEAD_DIM).repeat(HEADS, 1, 1))
        self.relation_cross = nn.Parameter(torch.eye(HEAD_DIM).repeat(HEADS, 1, 1))
        self.channel_bias = nn.Parameter(torch.zeros(2, 2, HEADS))
        self.edge_query = nn.ModuleList([nn.Linear(256, D, bias=False) for _ in range(2)])
        self.edge_key = nn.Linear(256, D, bias=False)
        self.edge_score = _MLP(8, 16, HEADS)
        self.null_score = nn.ModuleList([_MLP(256, 16, HEADS) for _ in range(2)])
        self.typed_value = bool(typed_value)
        if self.typed_value:
            self.value_sender = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(2)])
            self.message_same = nn.Parameter(torch.eye(HEAD_DIM).repeat(HEADS, 1, 1))
            self.message_cross = nn.Parameter(torch.eye(HEAD_DIM).repeat(HEADS, 1, 1))
            self.value = None
        else:
            self.value = nn.Linear(D, D, bias=False)
            self.value_sender = None
            self.register_parameter("message_same", None)
            self.register_parameter("message_cross", None)
        self.output = nn.ParameterList([nn.Parameter(torch.eye(HEAD_DIM).repeat(HEADS, 1, 1)) for _ in range(2)])
        self.ffn = _MLP(D, 128, D, bias=False, last_bias=False)
        for item in self.null_score:
            nn.init.zeros_(item.net[-1].weight)
            nn.init.constant_(item.net[-1].bias, float(torch.log(torch.tensor(1.0 / 9.0))))
        nn.init.xavier_uniform_(self.ffn.net[-1].weight, gain=0.1)

    def forward(
        self,
        cbar: tuple[torch.Tensor, torch.Tensor],
        e: torch.Tensor,
        raw: tuple[torch.Tensor, torch.Tensor],
        available: torch.Tensor,
        edge_index: torch.Tensor,
        tau: float,
        graph_inputs: dict[str, torch.Tensor] | None = None,
        collect_diagnostics: bool = False,
        optimized: bool = True,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        n = e.size(0)
        src, dst = edge_index
        receiver_ctx = (cbar[0], cbar[1])
        r = torch.cat((e, cbar[0], cbar[1]), dim=-1)
        if src.numel():
            valid_src = available[src]
            edge_valid = valid_src.any(-1)
        else:
            valid_src = available.new_empty((0, 2))
            edge_valid = available.new_empty((0,))
        degree = graph_inputs["degree_common"] if graph_inputs is not None else _degree(dst, edge_valid, n, e.dtype)
        degree_all = torch.zeros((n,), dtype=e.dtype, device=e.device)
        if src.numel():
            if graph_inputs is not None:
                degree_all = graph_inputs["degree_all"]
            else:
                degree_all.index_add_(0, dst, torch.ones_like(dst, dtype=e.dtype))
        if src.numel():
            if graph_inputs is not None:
                cosines = [graph_inputs["raw_cosines"][:, m] for m in range(2)]
                observed_pair = [graph_inputs["observed_pair"][:, m].to(e.dtype) for m in range(2)]
            else:
                raw_norm = (F.normalize(raw[0], dim=-1, eps=1e-8), F.normalize(raw[1], dim=-1, eps=1e-8))
                cosines = []
                observed_pair = []
                for m in range(2):
                    seen = available[dst, m] & available[src, m]
                    cos = (raw_norm[m][dst] * raw_norm[m][src]).sum(-1)
                    cosines.append(torch.where(seen, cos, torch.zeros_like(cos)))
                    observed_pair.append(seen.to(e.dtype))
            zeta_base = torch.stack((cosines[0], cosines[1], observed_pair[0], observed_pair[1], torch.log1p(degree[dst]), torch.log1p(degree[src])), dim=-1)
            onehot = e.new_zeros((src.numel(), 2))
            edge_queries = [self.edge_query[t](r).view(n, HEADS, HEAD_DIM) for t in range(2)]
            edge_keys = self.edge_key(r).view(n, HEADS, HEAD_DIM)
        else:
            zeta_base = e.new_empty((0, 6))
            onehot = e.new_empty((0, 2))
            cosines = [e.new_empty((0,)), e.new_empty((0,))]
            edge_queries = [e.new_empty((n, HEADS, HEAD_DIM)) for _ in range(2)]
            edge_keys = e.new_empty((n, HEADS, HEAD_DIM))

        updates: list[torch.Tensor] = []
        modality_weights: list[torch.Tensor] = []
        neighbor_weights: list[torch.Tensor] = []
        null_weights: list[torch.Tensor] = []
        sender_keys = [self.key[sender](receiver_ctx[sender]).view(n, HEADS, HEAD_DIM) for sender in range(2)] if src.numel() and optimized else []
        if src.numel() and optimized:
            if self.typed_value:
                sender_values = [self.value_sender[sender](receiver_ctx[sender]).view(n, HEADS, HEAD_DIM) for sender in range(2)]
            else:
                shared_value = self.value
                assert shared_value is not None
                projected = shared_value(receiver_ctx[0]).view(n, HEADS, HEAD_DIM), shared_value(receiver_ctx[1]).view(n, HEADS, HEAD_DIM)
                sender_values = [projected[0], projected[1]]
        else:
            sender_values = []
        for target in range(2):
            if src.numel():
                q = self.query[target](torch.cat((receiver_ctx[target], e), dim=-1)).view(n, HEADS, HEAD_DIM)
                channel_scores = []
                channel_masks = []
                projected_values = []
                for sender in range(2):
                    rel = self.relation_same if target == sender else self.relation_cross
                    if optimized:
                        transformed = torch.einsum("nhd,hdf->nhf", q, rel)[dst]
                        key_edge = sender_keys[sender][src]
                        value_nodes = sender_values[sender]
                    else:
                        transformed = torch.einsum("ehd,hdf->ehf", q[dst], rel)
                        key_edge = self.key[sender](receiver_ctx[sender]).view(n, HEADS, HEAD_DIM)[src]
                        if self.typed_value:
                            value_nodes = self.value_sender[sender](receiver_ctx[sender]).view(n, HEADS, HEAD_DIM)
                        else:
                            assert self.value is not None
                            value_nodes = self.value(receiver_ctx[sender]).view(n, HEADS, HEAD_DIM)
                    score = (transformed * key_edge).sum(-1) / (HEAD_DIM ** 0.5)
                    score = score + self.channel_bias[sender, target][None, :]
                    channel_scores.append(score)
                    channel_masks.append(available[src, sender])
                    if self.typed_value:
                        message_relation = self.message_same if target == sender else self.message_cross
                        value_nodes = torch.einsum("nhd,hdf->nhf", value_nodes, message_relation)
                    projected_values.append(value_nodes[src])
                score_channels = torch.stack(channel_scores, dim=-1)
                mask_channels = torch.stack(channel_masks, dim=-1)
                safe = score_channels.masked_fill(~mask_channels[:, None, :], -torch.inf)
                has_sender = mask_channels.any(-1)
                safe = torch.where(has_sender[:, None, None], safe, torch.zeros_like(safe))
                pi = torch.softmax(safe.float() / float(tau), dim=-1).to(e.dtype)
                pi = pi * mask_channels[:, None, :].to(e.dtype)
                pi = pi / pi.sum(-1, keepdim=True).clamp_min(1.0)
                value_edge = pi[..., 0, None] * projected_values[0] + pi[..., 1, None] * projected_values[1]

                target_hot = onehot.clone()
                target_hot[:, target] = 1.0
                edge_aux = torch.cat((zeta_base, target_hot), dim=-1)
                qedge, kedge = edge_queries[target], edge_keys
                independent = (qedge[dst] * kedge[src]).sum(-1) / (HEAD_DIM ** 0.5)
                b = independent + self.edge_score(edge_aux)
                ell = self.null_score[target](r)
                null_logit = torch.where((degree > 0)[:, None], degree.clamp_min(1).log()[:, None] + ell, torch.zeros_like(ell))
                alpha, null_prob = _grouped_softmax_with_null(b / float(tau), dst, edge_valid, null_logit, n)
                weighted = alpha.unsqueeze(-1) * value_edge
                aggregate = _index_add(weighted, dst, n)
                aggregate = torch.einsum("nhd,hdf->nhf", aggregate, self.output[target]).reshape(n, D)
                update = self.ffn(aggregate)
            else:
                pi = e.new_empty((0, HEADS, 2))
                alpha = e.new_empty((0, HEADS))
                null_prob = e.new_ones((n, HEADS))
                update = self.ffn(e.new_zeros((n, D)))
            updates.append(update)
            if collect_diagnostics:
                modality_weights.append(pi.detach())
                neighbor_weights.append(alpha.detach())
                null_weights.append(null_prob.detach())
        stats = ({
            "modality_weights": torch.stack(modality_weights, dim=1),
            "neighbor_weights": torch.stack(neighbor_weights, dim=1),
            "null_probability": torch.stack(null_weights, dim=1),
            "degree": degree.detach(),
            "raw_cosines": torch.stack(cosines, dim=-1).detach() if src.numel() else e.new_empty((0, 2)),
            "edge_valid": edge_valid.detach(), "dst": dst.detach(),
        } if collect_diagnostics else {})
        return (updates[0], updates[1]), stats


class _CollaborativeLayer(nn.Module):
    def __init__(self, typed_value: bool = False):
        super().__init__()
        self.private_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.common_norm = nn.LayerNorm(D)
        self.private = nn.ModuleList([_PrivatePath() for _ in range(2)])
        self.common = _CommonPath(typed_value=typed_value)
        self.dropout = nn.Dropout(0.2)


class MAGCollabAdapt(nn.Module):
    """Corrected MAG collaboration encoder with optional typed values and fusion."""

    out_dim = 640

    def __init__(self, data_info: dict[str, Any], variant: str = "M0", dropout: float = 0.2, init_seed: int | None = None, *, typed_value: bool = False, fusion: bool = False, out_dim: int = 640, collect_diagnostics: bool = False, optimize_cache: bool = True):
        super().__init__()
        self.variant = str(variant).upper()
        if self.variant not in {"B0", "B1", "M0", "M1"}:
            raise ValueError(f"unsupported collaborative NC variant: {self.variant}")
        if float(dropout) != 0.2:
            raise ValueError("collab_nc_v1 fixes residual message dropout at 0.2")
        self.typed_value = bool(typed_value)
        if self.typed_value and self.variant not in {"M0", "M1"}:
            raise ValueError("typed same/cross values are defined only for the collaborative M0/M1 backbone")
        self.fusion_enabled = bool(fusion)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.optimize_cache = bool(optimize_cache)
        self.out_dim = int(out_dim)
        if self.out_dim not in {640, 768}:
            raise ValueError("collab-adapt readout dimension must be 640 or 768")
        if self.fusion_enabled and self.out_dim != 768:
            raise ValueError("the nonlinear fusion block is only valid for 768-dimensional S readouts")
        if self.out_dim == 640 and self.fusion_enabled:
            raise ValueError("R-stage models use the original 640-dimensional readout")
        self.text_dim = int(data_info["text_dim"])
        self.image_dim = int(data_info["visual_dim"])
        if min(self.text_dim, self.image_dim) <= 0:
            raise ValueError("both modality dimensions must be positive")
        if int(data_info["input_dim"]) != self.text_dim + self.image_dim:
            raise ValueError("input_dim must be [text,image] concatenation")
        self.p_text = nn.Linear(self.text_dim, 128)
        self.p_image = nn.Linear(self.image_dim, 128)
        self.p_norm = nn.ModuleList([nn.LayerNorm(128), nn.LayerNorm(128)])
        self.e_mlp = _MLP(256, 128, 128)
        self.e_norm = nn.LayerNorm(128)
        self.has_graph = self.variant != "B0"
        if self.has_graph:
            self.common_adapter = nn.ModuleList([nn.Linear(128, D) for _ in range(2)])
            self.common_stem = _MLP(D, D, D)
            self.common_stem_norm = nn.LayerNorm(D)
            self.private_stem = nn.ModuleList([_MLP(128, D, D) for _ in range(2)])
            self.private_stem_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
            if self.variant in {"M0", "M1"}:
                self.layers = nn.ModuleList([_CollaborativeLayer(typed_value=self.typed_value), _CollaborativeLayer(typed_value=self.typed_value)])
            else:
                self.b1_norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(128) for _ in range(2)]) for _ in range(2)])
                self.b1_updates = nn.ModuleList([nn.ModuleList([_MLP(256, 256, 128) for _ in range(2)]) for _ in range(2)])
                for layer in self.b1_updates:
                    for update in layer:
                        nn.init.xavier_uniform_(update.net[-1].weight, gain=0.1)
            self.readout_dropout = nn.Dropout(0.0)
        if self.variant in {"B1", "M0"}:
            self.decoder_text = nn.Linear(128, self.text_dim)
            self.decoder_image = nn.Linear(128, self.image_dim)
            self.pair_projection = nn.Linear(D, 32, bias=False)
        if self.fusion_enabled:
            self.fusion_e = nn.Linear(128, 128, bias=False)
            self.fusion_delta = nn.Linear(256, 128, bias=False)
            self.fusion_out = nn.Linear(128, 128, bias=False)
            self.fusion_bias = nn.Parameter(torch.zeros(128))
            self.fusion_dropout = nn.Dropout(0.2)
        else:
            self.fusion_e = self.fusion_delta = self.fusion_out = self.fusion_dropout = None
            self.register_parameter("fusion_bias", None)
        self.beta = 1.0
        self.tau = 1.0
        self._latest_diagnostics: dict[str, Any] = {}
        self.init_seed = int(torch.initial_seed() if init_seed is None else init_seed)
        self._frozen_cache: dict[str, Any] | None = None
        self._initialize(self.init_seed)

    @staticmethod
    def _seeded_xavier(module: nn.Linear, seed: int, name: str, gain: float = 1.0) -> None:
        token = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
        local_seed = int.from_bytes(token, "little") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(local_seed)
        nn.init.xavier_uniform_(module.weight, gain=gain, generator=generator)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def _initialize(self, seed: int) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                self._seeded_xavier(module, seed, name)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if self.variant in {"M0", "M1"}:
            for layer_idx, layer in enumerate(self.layers):
                for modality, path in enumerate(layer.private):
                    self._seeded_xavier(path.update.net[-1], seed, f"layers.{layer_idx}.private.{modality}.update.net.2", gain=0.1)
                    # Restore constructor intent after generic named Xavier init.
                    nn.init.zeros_(path.gate.net[-1].weight)
                    nn.init.zeros_(path.gate.net[-1].bias)
                    nn.init.zeros_(path.null_score.net[-1].weight)
                    nn.init.constant_(path.null_score.net[-1].bias, float(torch.log(torch.tensor(1.0 / 9.0))))
                self._seeded_xavier(layer.common.ffn.net[-1], seed, f"layers.{layer_idx}.common.ffn.net.2", gain=0.1)
                for null in layer.common.null_score:
                    nn.init.zeros_(null.net[-1].weight)
                    nn.init.constant_(null.net[-1].bias, float(torch.log(torch.tensor(1.0 / 9.0))))
                if layer.common.typed_value:
                    # V-on starts at the exact V-off function: both sender maps
                    # receive the V-off shared value map's name-derived seed.
                    reference = nn.Linear(D, D, bias=False)
                    self._seeded_xavier(reference, seed, f"layers.{layer_idx}.common.value")
                    with torch.no_grad():
                        for sender_map in layer.common.value_sender:
                            sender_map.weight.copy_(reference.weight)
                    for projection_idx, sender_map in enumerate(layer.common.value_sender):
                        self._seeded_xavier(sender_map, seed, f"layers.{layer_idx}.common.value_sender.{projection_idx}")
                        # Preserve the copied shared-value initial function.
                        with torch.no_grad():
                            sender_map.weight.copy_(reference.weight)
        if self.variant == "B1":
            for layer_idx, layer in enumerate(self.b1_updates):
                for modality, update in enumerate(layer):
                    self._seeded_xavier(update.net[-1], seed, f"b1_updates.{layer_idx}.{modality}.net.2", gain=0.1)
        if self.fusion_enabled:
            self._seeded_xavier(self.fusion_e, seed, "fusion.e", gain=1.0)
            self._seeded_xavier(self.fusion_delta, seed, "fusion.delta", gain=1.0)
            self._seeded_xavier(self.fusion_out, seed, "fusion.out", gain=0.1)
            nn.init.zeros_(self.fusion_bias)

    def set_epoch(self, step: int) -> None:
        s = max(int(step), 1)
        ramp = min((s - 1) / 19.0, 1.0)
        self.beta = 0.1 + 0.9 * ramp
        self.tau = 2.0 - ramp

    @staticmethod
    def infer_availability(x_t: torch.Tensor, x_i: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        finite_t = torch.isfinite(x_t).all(-1)
        finite_i = torch.isfinite(x_i).all(-1)
        zero_t = (torch.nan_to_num(x_t) == 0).all(-1)
        zero_i = (torch.nan_to_num(x_i) == 0).all(-1)
        mask = torch.stack((finite_t & ~zero_t, finite_i & ~zero_i), dim=-1)
        return mask, {
            "availability_source": "fallback: finite row and nonzero row heuristic",
            "text_nonfinite_rows": int((~finite_t).sum().item()),
            "image_nonfinite_rows": int((~finite_i).sum().item()),
            "text_all_zero_rows": int(zero_t.sum().item()),
            "image_all_zero_rows": int(zero_i.sum().item()),
        }

    def _initial_states(self, x: torch.Tensor, available: torch.Tensor):
        x_t = x[:, :self.text_dim]
        x_i = x[:, self.text_dim:]
        finite_t, finite_i = torch.isfinite(x_t), torch.isfinite(x_i)
        raw_t = torch.where(finite_t, x_t, torch.zeros_like(x_t))
        raw_i = torch.where(finite_i, x_i, torch.zeros_like(x_i))
        raw_t = torch.where(available[:, 0:1], raw_t, torch.zeros_like(raw_t))
        raw_i = torch.where(available[:, 1:2], raw_i, torch.zeros_like(raw_i))
        p_t = self.p_norm[0](F.gelu(self.p_text(raw_t))) * available[:, 0:1].to(x.dtype)
        p_i = self.p_norm[1](F.gelu(self.p_image(raw_i))) * available[:, 1:2].to(x.dtype)
        e = self.e_norm(self.e_mlp(torch.cat((p_t, p_i), dim=-1)))
        e = e * available.any(-1, keepdim=True).to(e.dtype)
        if not self.has_graph:
            return (p_t, p_i, e), (raw_t, raw_i), None
        c0, s0 = [], []
        for m, p in enumerate((p_t, p_i)):
            c = self.common_stem(self.common_adapter[m](p))
            c = self.common_stem_norm(c) * available[:, m:m + 1].to(c.dtype)
            s = self.private_stem[m](p)
            s = self.private_stem_norm[m](s) * available[:, m:m + 1].to(s.dtype)
            c0.append(c)
            s0.append(s)
        return (p_t, p_i, e), (raw_t, raw_i), (tuple(c0), tuple(s0))

    def _graph_inputs(self, x, edge_index, available, availability_identity, *, optimize_cache: bool):
        can_cache = (
            optimize_cache and not x.requires_grad and not edge_index.requires_grad
            and availability_identity is not None and not availability_identity.requires_grad
            and _tensor_version(x) is not None and _tensor_version(edge_index) is not None
            and _tensor_version(availability_identity) is not None
        )
        versions = (_tensor_version(x), _tensor_version(edge_index), _tensor_version(availability_identity)) if can_cache else None
        cached = self._frozen_cache
        if can_cache and cached is not None:
            refs = cached["refs"]
            if (
                refs[0] is x and refs[1] is edge_index and refs[2] is availability_identity
                and cached["versions"] == versions and cached["device"] == str(x.device)
            ):
                return cached["values"]
        values = _prepare_frozen_graph_inputs(x, edge_index, available, self.text_dim)
        if can_cache:
            self._frozen_cache = {
                "refs": (x, edge_index, availability_identity), "versions": versions,
                "device": str(x.device), "values": values,
            }
        else:
            self._frozen_cache = None
        return values

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        modality_available: torch.Tensor | None = None,
        *,
        pair_generator: torch.Generator | None = None,
        compute_ssl: bool = False,
        diagnostic_node_ids: torch.Tensor | None = None,
        collect_diagnostics: bool | None = None,
        optimize_cache: bool | None = None,
    ):
        if x.ndim != 2 or x.size(-1) != self.text_dim + self.image_dim:
            raise ValueError("feature tensor does not match configured [text,image] dimensions")
        availability_identity = modality_available
        if modality_available is None:
            available, _ = self.infer_availability(x[:, :self.text_dim], x[:, self.text_dim:])
        else:
            available = torch.as_tensor(modality_available, dtype=torch.bool, device=x.device)
            if tuple(available.shape) != (x.size(0), 2):
                raise ValueError("modality_available must have shape [N,2]")
            finite = torch.stack((torch.isfinite(x[:, :self.text_dim]).all(-1), torch.isfinite(x[:, self.text_dim:]).all(-1)), dim=-1)
            available = available & finite
        collect = self.collect_diagnostics if collect_diagnostics is None else bool(collect_diagnostics)
        use_cache = self.optimize_cache if optimize_cache is None else bool(optimize_cache)
        graph_inputs = self._graph_inputs(x, edge_index, available, availability_identity, optimize_cache=use_cache) if self.has_graph else None
        (p_t, p_i, e), raw, initial = self._initial_states(x, available)
        if not self.has_graph:
            z = torch.cat((p_t, p_i, e, e.new_zeros((e.size(0), 256))), dim=-1)
            if self.out_dim == 768:
                z = torch.cat((z, e.new_zeros((e.size(0), 128))), dim=-1)
            diagnostics = {}
            if collect:
                nodes = torch.arange(e.size(0), device=e.device) if diagnostic_node_ids is None else diagnostic_node_ids.to(device=e.device, dtype=torch.long)
                diagnostics = {"diagnostic_nodes": int(nodes.numel()), "p_text_mean_norm": float(p_t.index_select(0, nodes).norm(dim=-1).mean().detach().cpu()), "p_image_mean_norm": float(p_i.index_select(0, nodes).norm(dim=-1).mean().detach().cpu()), "e_mean_norm": float(e.index_select(0, nodes).norm(dim=-1).mean().detach().cpu())}
            extras = {"path_states": {}, "path_dims": {}, "loss_terms": {"rec": e.new_zeros(()), "pair": e.new_zeros(())}, "availability": available, "diagnostics": diagnostics, "graph_increments": tuple(e.new_zeros((e.size(0), D)) for _ in range(4)), "fusion_residual": e.new_zeros((e.size(0), 128))}
            self._latest_diagnostics = extras["diagnostics"]
            return z, None, None, z.new_zeros(()), extras
        c0, s0 = initial
        c, s = c0, s0
        layer_stats: list[dict[str, Any]] = []
        first_layer_private = None
        if self.variant in {"M0", "M1"}:
            for layer_idx, layer in enumerate(self.layers):
                cbar = (layer.common_norm(c[0]), layer.common_norm(c[1]))
                sbar = (layer.private_norm[0](s[0]), layer.private_norm[1](s[1]))
                s_updates, private_stats = [], []
                for m in range(2):
                    update, stats = layer.private[m](sbar[m], raw[m], available[:, m], edge_index, self.tau, graph_inputs, m, collect)
                    s_updates.append(update)
                    private_stats.append(stats)
                c_updates, common_stats = layer.common(cbar, e, raw, available, edge_index, self.tau, graph_inputs, collect, optimized=use_cache)
                # All four next states are computed from the same previous layer.
                s = tuple(s0[m] + 0.9 * (s[m] - s0[m]) + layer.dropout(s_updates[m]) for m in range(2))
                c = tuple(c0[m] + 0.9 * (c[m] - c0[m]) + layer.dropout(c_updates[m]) for m in range(2))
                if collect:
                    layer_stats.append({"private": private_stats, "common": common_stats})
                if layer_idx == 0:
                    first_layer_private = s
        else:
            h0 = tuple(torch.cat((c0[m], s0[m]), dim=-1) for m in range(2))
            h = h0
            for layer_idx in range(2):
                h_next = []
                for m in range(2):
                    bar = self.b1_norms[layer_idx][m](h[m])
                    valid = available[:, m][edge_index[0]] if edge_index.size(1) else available.new_empty((0,))
                    src, dst = edge_index
                    aggregate = bar.new_zeros(bar.shape)
                    degree = _degree(dst, valid, bar.size(0), bar.dtype)
                    if src.numel():
                        aggregate = _index_add(bar[src[valid]], dst[valid], bar.size(0)) / degree.clamp_min(1)[:, None]
                    update = self.b1_updates[layer_idx][m](torch.cat((bar, aggregate), dim=-1))
                    update = torch.where((degree > 0)[:, None], update, torch.zeros_like(update))
                    h_next.append(h0[m] + 0.9 * (h[m] - h0[m]) + F.dropout(update, p=0.2, training=self.training))
                h = (h_next[0], h_next[1])
                if collect:
                    layer_stats.append({"mean_degree": [graph_inputs["degree_private"][m].mean().detach() for m in range(2)]})
                if layer_idx == 0:
                    first_layer_private = (h[0][:, D:], h[1][:, D:])
            c = (h[0][:, :D], h[1][:, :D])
            s = (h[0][:, D:], h[1][:, D:])
        increments = (s[0] - s0[0], s[1] - s0[1], c[0] - c0[0], c[1] - c0[1])
        delta = torch.cat(increments, dim=-1)
        z640 = torch.cat((p_t, p_i, e, self.beta * delta), dim=-1)
        fusion_residual = e.new_zeros((e.size(0), 128))
        if self.out_dim == 768:
            if self.fusion_enabled:
                anchor = self.fusion_e(e) + self.fusion_bias
                fusion_residual = self.fusion_out(F.gelu(anchor + self.fusion_delta(delta)) - F.gelu(anchor))
                fusion_residual = self.fusion_dropout(fusion_residual)
            z = torch.cat((z640, self.beta * fusion_residual), dim=-1)
        else:
            z = z640
        loss_terms = {"rec": z.new_zeros(()), "pair": z.new_zeros(())}
        if compute_ssl and self.variant in {"B1", "M0"}:
            loss_terms = self.compute_ssl(x, available, c0, s0, pair_generator)
        path_states = {"text_private": s[0], "image_private": s[1], "shared": torch.cat(c, dim=-1)}
        diagnostics = self._summarize(p_t, p_i, e, c0, s0, c, s, increments, layer_stats, available, diagnostic_node_ids) if collect else {}
        extras = {"path_states": path_states, "path_dims": {"text_private": D, "image_private": D, "shared": 2 * D}, "source_states": {"text": first_layer_private[0], "image": first_layer_private[1]}, "loss_terms": loss_terms, "availability": available, "diagnostics": diagnostics, "graph_increments": increments, "fusion_residual": fusion_residual}
        if collect:
            extras["diagnostic_tensors"] = {"e": e, "delta": delta}
        self._latest_diagnostics = diagnostics
        return z, None, None, z.new_zeros(()), extras

    def compute_ssl(self, x: torch.Tensor, available: torch.Tensor, c0, s0, pair_generator: torch.Generator | None):
        rec_losses = []
        for m, (c, s, decoder) in enumerate(((c0[0], s0[0], self.decoder_text), (c0[1], s0[1], self.decoder_image))):
            target = x[:, :self.text_dim] if m == 0 else x[:, self.text_dim:]
            target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
            mask = available[:, m]
            chunks = []
            for start in range(0, x.size(0), 8192):
                stop = min(start + 8192, x.size(0))
                pred = decoder(torch.cat((c[start:stop], s[start:stop]), dim=-1))
                cos = F.cosine_similarity(pred, target[start:stop], dim=-1, eps=1e-8)
                chunks.append(((1.0 - cos) * mask[start:stop].to(cos.dtype)).sum())
            rec_losses.append(torch.stack(chunks).sum() / mask.sum().clamp_min(1))
        rec = torch.stack(rec_losses).mean()
        both = available.all(-1).nonzero(as_tuple=False).flatten()
        count = min(1024, both.numel())
        if count < 2:
            pair = rec.new_zeros(())
        else:
            if pair_generator is None:
                raise ValueError("M0 training requires a dedicated pair_generator")
            offsets = torch.randperm(both.numel(), generator=pair_generator, device=both.device)[:count]
            chosen = both.index_select(0, offsets)
            a = F.normalize(self.pair_projection(c0[0].index_select(0, chosen)), dim=-1, eps=1e-8)
            b = F.normalize(self.pair_projection(c0[1].index_select(0, chosen)), dim=-1, eps=1e-8)
            logits_ab = a @ b.t() / 0.2
            logits_ba = b @ a.t() / 0.2
            labels = torch.arange(count, dtype=torch.long, device=x.device)
            pair = 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels))
        return {"rec": rec, "pair": pair}

    @staticmethod
    def _summarize(p_t, p_i, e, c0, s0, c, s, increments, layer_stats, available, diagnostic_node_ids=None):
        nodes = torch.arange(e.size(0), device=e.device) if diagnostic_node_ids is None else diagnostic_node_ids.to(device=e.device, dtype=torch.long)
        sample_available = available.index_select(0, nodes)
        def mean_norm(x):
            return float(x.detach().index_select(0, nodes).norm(dim=-1).mean().cpu())
        common_null, common_true, common_degree, pi_mean = [], [], [], []
        rho_mean, gate_mean, gate_saturation, diff_mean = [], [], [], []
        for layer in layer_stats:
            if "common" not in layer:
                continue
            cs = layer["common"]
            selected_null = cs["null_probability"].index_select(0, nodes)
            common_null.append(float(selected_null.mean().cpu()))
            common_true.append(float((1.0 - selected_null).mean().cpu()))
            common_degree.append(float(cs["degree"].index_select(0, nodes).mean().cpu()))
            if cs["modality_weights"].numel():
                edge_mask = torch.isin(cs["dst"], nodes) & cs["edge_valid"]
                if bool(edge_mask.any()):
                    pi_mean.append(cs["modality_weights"][edge_mask].mean((0, 1, 2)).cpu().tolist())
            for ps in layer["private"]:
                rho_mean.append(float(ps["rho"].index_select(0, nodes).mean().cpu()))
                gl = ps["gate_local"].index_select(0, nodes)
                gd = ps["gate_diff"].index_select(0, nodes)
                gate_mean.append([float(gl.mean().cpu()), float(gd.mean().cpu())])
                gate_saturation.append([float(((gl < 0.1) | (gl > 1.9)).float().mean().cpu()), float(((gd < 0.1) | (gd > 1.9)).float().mean().cpu())])
                diff_mean.append(float(ps["diff_norm"].index_select(0, nodes).mean().cpu()))
        return {
            "p_text_mean_norm": mean_norm(p_t), "p_image_mean_norm": mean_norm(p_i), "e_mean_norm": mean_norm(e),
            "c0_mean_norm": [mean_norm(v) for v in c0], "s0_mean_norm": [mean_norm(v) for v in s0],
            "c_final_mean_norm": [mean_norm(v) for v in c], "s_final_mean_norm": [mean_norm(v) for v in s],
            "graph_increment_mean_norm": [mean_norm(v) for v in increments],
            "common_null_mean": common_null, "common_true_mass_mean": common_true,
            "common_pi_mean": pi_mean, "common_degree_mean": common_degree,
            "private_rho_mean": rho_mean, "private_gate_means": gate_mean,
            "private_gate_saturation_fraction": gate_saturation, "private_difference_output_norm_mean": diff_mean,
            "diagnostic_nodes": int(nodes.numel()),
            "diagnostic_available_counts": [int(sample_available[:, m].sum().item()) for m in range(2)],
        }


def parameter_groups_no_decay(module: nn.Module, extra_modules: list[nn.Module] | None = None):
    """Create AdamW groups with all biases and LayerNorm parameters exempt from weight decay."""
    modules = [module, *(extra_modules or [])]
    decay, no_decay = [], []
    seen: set[int] = set()
    for root in modules:
        norm_ids = {id(p) for sub in root.modules() if isinstance(sub, nn.LayerNorm) for p in sub.parameters(recurse=False)}
        for name, param in root.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if name.endswith("bias") or id(param) in norm_ids:
                no_decay.append(param)
            else:
                decay.append(param)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


class Model(MAGCollabAdapt):
    """Repository factory-compatible constructor: Model(cfg, data_info)."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__(
            data_info,
            variant=str(cfg.model.get("base_variant", cfg.model.get("variant", "M0"))),
            dropout=float(cfg.model.get("dropout", 0.2)),
            init_seed=int(getattr(cfg, "seed", torch.initial_seed())),
            typed_value=bool(cfg.model.get("typed_value", False)),
            fusion=bool(cfg.model.get("fusion", False)),
            out_dim=int(cfg.model.get("out_dim", 640)),
            collect_diagnostics=bool(cfg.model.get("collect_diagnostics", False)),
            optimize_cache=bool(cfg.model.get("optimize_cache", True)),
        )
