from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score
from torch_geometric.utils import scatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models.relation_transport_m02 import Model
from src.utils.device import get_device

RESULTS = ROOT / "results" / "model_design" / "m02_context_audit"
M02_RESULTS = ROOT / "results" / "model_design" / "m02"
CHECKPOINTS = ROOT / "outputs" / "model_design" / "m02" / "pilot" / "checkpoints"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
SHUFFLE_SEEDS = (3407, 3408, 3409)
DIVERSITY_SAMPLE_SEED = 20260929
DEGREE_BUCKET_LABELS = ("0", "1", "2-3", "4-7", "8-15", "16-31", ">=32")
DEGREE_EDGE_LABELS = ("1", "2-3", "4-10", ">10")
EPS = 1e-12


def checkpoint_path(dataset: str, seed: int) -> Path:
    return CHECKPOINTS / f"{dataset}_seed{seed}.pt"


def preflight_checkpoints() -> list[dict]:
    rows = []
    missing = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = checkpoint_path(dataset, seed)
            if not path.is_file():
                missing.append(path)
            else:
                rows.append({"dataset": dataset, "seed": seed, "path": path})
    if missing:
        raise FileNotFoundError(
            "STOP: required M0.2 checkpoints are missing; no training will be started: "
            + ", ".join(map(str, missing))
        )
    if len(rows) != 6:
        raise AssertionError(f"Expected six checkpoints, found {len(rows)}")
    return rows


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def repair_context_diagnostics(m02_runs_path: Path = M02_RESULTS / "m02_runs.json") -> int:
    """Rebuild both summaries from the canonical per-run diagnostics JSON."""
    runs = json.loads(m02_runs_path.read_text(encoding="utf-8"))
    rows = []
    keys = (
        ("context_norm", "relation_context_norm"),
        ("relative_context_residual", "contextual_relation_residual"),
        ("abs_conductance_change", "context_conductance_abs_change"),
        ("mean_abs_angle_change", "context_angle_mean_abs_change"),
    )
    for run in runs:
        for modality in ("text", "visual"):
            for diagnostic, source_key in keys:
                values = run["diagnostics"][source_key][modality]
                rows.append(
                    {
                        "dataset": run["dataset"],
                        "seed": run["seed"],
                        "modality": modality,
                        "diagnostic": diagnostic,
                        **values,
                    }
                )
    modalities = {row["modality"] for row in rows}
    if len(rows) != 48 or modalities != {"text", "visual"}:
        raise AssertionError(
            f"Corrected context diagnostics must have 48 rows and both modalities; "
            f"got {len(rows)} rows, modalities={modalities}"
        )
    columns = ["dataset", "seed", "modality", "diagnostic", "mean", "std", "p10", "p50", "p90"]
    _write_csv(M02_RESULTS / "context_diagnostics.csv", rows, columns)
    _write_csv(RESULTS / "corrected_context_diagnostics.csv", rows, columns)
    return len(rows)


def build_reverse_edge_mapping(
    source: torch.Tensor, target: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Map each coalesced directed edge to its reverse using sorted tensor keys."""
    source, target = source.long().reshape(-1), target.long().reshape(-1)
    if source.shape != target.shape:
        raise ValueError("source and target must have matching shapes")
    if torch.any(source == target):
        raise ValueError("physical graph must not contain self-loops")
    keys = source * int(num_nodes) + target
    if torch.unique(keys).numel() != keys.numel():
        raise ValueError("physical graph must be coalesced with unique directed edges")
    sorted_keys, sorted_indices = torch.sort(keys)
    reverse_keys = target * int(num_nodes) + source
    positions = torch.searchsorted(sorted_keys, reverse_keys)
    if torch.any(positions >= sorted_keys.numel()):
        raise ValueError("physical graph is missing a reverse edge")
    if not torch.equal(sorted_keys[positions], reverse_keys):
        raise ValueError("physical graph is missing a reverse edge")
    reverse = sorted_indices[positions]
    edge_ids = torch.arange(source.numel(), device=source.device)
    if not torch.equal(reverse[reverse], edge_ids):
        raise AssertionError("reverse mapping must be an involution")
    return reverse


def degree_bucket_ids(degree: torch.Tensor) -> torch.Tensor:
    degree = degree.long()
    result = torch.zeros_like(degree)
    result[degree == 1] = 1
    result[(degree >= 2) & (degree <= 3)] = 2
    result[(degree >= 4) & (degree <= 7)] = 3
    result[(degree >= 8) & (degree <= 15)] = 4
    result[(degree >= 16) & (degree <= 31)] = 5
    result[degree >= 32] = 6
    return result


def edge_degree_bin_ids(degree: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    edge_degree = torch.minimum(degree[source], degree[target])
    result = torch.zeros_like(edge_degree)
    result[edge_degree == 1] = 0
    result[(edge_degree >= 2) & (edge_degree <= 3)] = 1
    result[(edge_degree >= 4) & (edge_degree <= 10)] = 2
    result[edge_degree > 10] = 3
    return result


def degree_matched_permutation(degree: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a deterministic node permutation that stays inside exact degree buckets."""
    degree = degree.long()
    bucket = degree_bucket_ids(degree)
    permutation = torch.arange(degree.numel(), device=degree.device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for bucket_id in range(len(DEGREE_BUCKET_LABELS)):
        nodes = torch.nonzero(bucket.cpu() == bucket_id, as_tuple=False).reshape(-1)
        if nodes.numel() <= 1:
            continue
        order = torch.randperm(nodes.numel(), generator=generator)
        if torch.equal(order, torch.arange(nodes.numel())):
            order = order.roll(1)
        nodes_device = nodes.to(degree.device)
        permutation[nodes_device] = nodes[order].to(degree.device)
    if not torch.equal(torch.sort(permutation).values, torch.arange(degree.numel(), device=degree.device)):
        raise AssertionError("degree-matched mapping must be a permutation")
    if not torch.equal(bucket[permutation], bucket):
        raise AssertionError("degree-matched mapping changed a node's degree bucket")
    return permutation


def shuffle_context_pair(
    contexts: dict[str, torch.Tensor], permutation: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Apply one shared node mapping to both modalities."""
    return {modality: context[permutation] for modality, context in contexts.items()}


def _safe_distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    quantiles = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def _population_stats(
    sums: torch.Tensor, square_sums: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    count = counts.to(sums.dtype).clamp_min(1).unsqueeze(-1)
    mean = sums / count
    variance = (square_sums / count - mean.square()).clamp_min(0.0)
    std = torch.sqrt(variance)
    active = counts > 0
    mean = torch.where(active[:, None], mean, torch.zeros_like(mean))
    std = torch.where(active[:, None], std, torch.zeros_like(std))
    return mean, std


def _endpoint_lpo_statistics(
    relation: torch.Tensor,
    node_ids: torch.Tensor,
    removed_relation: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    square_sums: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loo_count = (counts[node_ids] - 1).clamp_min(0)
    loo_sum = sums[node_ids] - removed_relation
    loo_square_sum = square_sums[node_ids] - removed_relation.square()
    count = loo_count.to(relation.dtype).clamp_min(1).unsqueeze(-1)
    mean = loo_sum / count
    variance = (loo_square_sum / count - mean.square()).clamp_min(0.0)
    std = torch.sqrt(variance)
    active = loo_count > 0
    mean = torch.where(active[:, None], mean, torch.zeros_like(mean))
    std = torch.where(active[:, None], std, torch.zeros_like(std))
    return mean, std, loo_count


def leave_pair_out_statistics(
    relation: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    reverse_idx: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-edge source/target statistics after removing both pair directions."""
    dim = int(relation.size(-1))
    count_in = torch.bincount(target, minlength=num_nodes)
    count_out = torch.bincount(source, minlength=num_nodes)
    sum_in = scatter(relation, target, dim=0, dim_size=num_nodes, reduce="sum")
    sq_in = scatter(relation.square(), target, dim=0, dim_size=num_nodes, reduce="sum")
    sum_out = scatter(relation, source, dim=0, dim_size=num_nodes, reduce="sum")
    sq_out = scatter(relation.square(), source, dim=0, dim_size=num_nodes, reduce="sum")

    # Edge e is source -> target; reverse_idx[e] is target -> source.
    rev_relation = relation[reverse_idx]
    target_in_mean, target_in_std, target_in_count = _endpoint_lpo_statistics(
        relation, target, relation, count_in, sum_in, sq_in
    )
    target_out_mean, target_out_std, target_out_count = _endpoint_lpo_statistics(
        relation, target, rev_relation, count_out, sum_out, sq_out
    )
    source_in_mean, source_in_std, source_in_count = _endpoint_lpo_statistics(
        relation, source, rev_relation, count_in, sum_in, sq_in
    )
    source_out_mean, source_out_std, source_out_count = _endpoint_lpo_statistics(
        relation, source, relation, count_out, sum_out, sq_out
    )

    target_stats = torch.cat(
        (target_in_mean, target_in_std, target_out_mean, target_out_std), dim=-1
    ).reshape(-1, 4 * dim)
    source_stats = torch.cat(
        (source_in_mean, source_in_std, source_out_mean, source_out_std), dim=-1
    ).reshape(-1, 4 * dim)
    target_active = (target_in_count + target_out_count) > 0
    source_active = (source_in_count + source_out_count) > 0
    return target_stats, source_stats, target_active, source_active


def encode_leave_pair_out_contexts(
    model: Model,
    relation: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    reverse_idx: torch.Tensor,
    num_nodes: int,
    modality: str,
):
    target_stats, source_stats, target_active, source_active = leave_pair_out_statistics(
        relation, source, target, reverse_idx, num_nodes
    )
    encoder = model.text_context_encoder if modality == "text" else model.visual_context_encoder
    edge_module = model.text_edge_context if modality == "text" else model.visual_edge_context
    encoded = encoder(torch.cat((target_stats, source_stats), dim=0))
    edge_count = relation.size(0)
    target_context = encoded[:edge_count]
    source_context = encoded[edge_count:]
    target_context = torch.where(target_active[:, None], target_context, torch.zeros_like(target_context))
    source_context = torch.where(source_active[:, None], source_context, torch.zeros_like(source_context))
    contextual_relation, residual = edge_module(
        relation,
        target_context,
        source_context,
        model.context_residual_scale,
    )
    return contextual_relation, residual, target_context, source_context, target_stats, source_stats


def encode_full_node_contexts(
    model: Model,
    relation: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
    modality: str,
):
    statistics = model._relation_environment_statistics(relation, source, target, num_nodes)
    encoder = model.text_context_encoder if modality == "text" else model.visual_context_encoder
    context = encoder(statistics)
    incident = (torch.bincount(source, minlength=num_nodes) + torch.bincount(target, minlength=num_nodes)) > 0
    return torch.where(incident[:, None], context, torch.zeros_like(context))


def relation_conductance(model: Model, h0: torch.Tensor, relation: torch.Tensor, source: torch.Tensor, target: torch.Tensor, modality: str):
    similarity = F.cosine_similarity(h0[target], h0[source], dim=-1, eps=1e-8)
    tau = F.softplus(getattr(model, f"raw_tau_{modality}"))
    correction = getattr(model, f"{modality}_relation_correction")(relation).squeeze(-1)
    return model.compatibility_scale * torch.sigmoid(tau * similarity + correction)


def model_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _config(dataset: str, seed: int):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=relation_transport_m02",
                f"seed={seed}",
                "num_runs=1",
                "device=cuda:0",
                "task.inference_mode=full",
                "task.evaluate_test=false",
            ],
        )


def load_frozen_run(dataset: str, seed: int, checkpoint: Path):
    cfg = _config(dataset, seed)
    data = load_mag_data(cfg, str(cfg.task.name), seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Model(cfg, data_info)
    model.load_state_dict(saved["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    classifier.load_state_dict(saved["head_state"], strict=True)
    device = get_device("cuda:0")
    model.to(device).eval()
    classifier.to(device).eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    val_labels = data.y[data.val_idx].to(device)
    val_idx = data.val_idx.to(device)
    return model, classifier, data, x, edge_index, val_idx, val_labels, device


def _metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    prediction = logits.argmax(dim=-1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    return {
        "acc": float((prediction == target).mean()),
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=list(range(int(num_classes))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def validation_with_evidence(
    model: Model,
    classifier: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    val_idx: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    evidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    context_off: bool = False,
) -> dict[str, float]:
    if context_off:
        embeddings, *_ = model.forward_without_relation_context(x, edge_index)
    elif evidence is None:
        embeddings, *_ = model(x, edge_index)
    else:
        had_override = "_relation_evidence" in model.__dict__
        previous = model.__dict__.get("_relation_evidence")

        def override(h0_text, h0_visual, source, target):
            return evidence

        model._relation_evidence = override
        try:
            embeddings, *_ = model(x, edge_index)
        finally:
            if had_override:
                model._relation_evidence = previous
            else:
                model.__dict__.pop("_relation_evidence", None)
    return _metrics(classifier(embeddings[val_idx]), val_labels, num_classes)


def _stats_row(prefix: str, values: torch.Tensor) -> dict:
    summary = _safe_distribution(values)
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _edge_relations_and_evidence(model: Model, data, x, edge_index):
    clean, degree, _mean_degree, _scalers = model._get_graph(edge_index, int(x.size(0)))
    source, target = clean
    reverse_idx = build_reverse_edge_mapping(source, target, int(x.size(0)))
    degree_out = torch.bincount(source, minlength=x.size(0))
    degree_in = torch.bincount(target, minlength=x.size(0))
    if not torch.equal(degree_out, degree_in):
        raise AssertionError("coalesced physical graph is not bidirectional")
    if not torch.equal(degree.long(), degree_out.long()):
        raise AssertionError("physical degree differs from unique-neighbor degree")

    h0 = {
        "text": model.text_projector(x[:, : model.text_dim]),
        "visual": model.visual_projector(x[:, model.text_dim : model.text_dim + model.visual_dim]),
    }
    from src.models.relation_transport_m0 import Model as M01Model

    pair_evidence = M01Model._relation_evidence(model, h0["text"], h0["visual"], source, target)
    previous = model._relation_context_enabled
    model._relation_context_enabled = False
    try:
        m02_pair_evidence = model._relation_evidence(h0["text"], h0["visual"], source, target)
    finally:
        model._relation_context_enabled = previous
    for base_value, disabled_value in zip(pair_evidence, m02_pair_evidence, strict=True):
        if not torch.equal(base_value, disabled_value):
            raise AssertionError("M0.2 Context-Off relation evidence differs from frozen M0.1 pair path")

    full_evidence = model._relation_evidence(h0["text"], h0["visual"], source, target)
    pair_relations = {"text": pair_evidence[2], "visual": pair_evidence[3]}
    full_relations = {"text": full_evidence[2], "visual": full_evidence[3]}
    pair_conductance = {"text": pair_evidence[0], "visual": pair_evidence[1]}
    full_conductance = {"text": full_evidence[0], "visual": full_evidence[1]}
    full_contexts = {}
    for modality in ("text", "visual"):
        reconstructed, _residual, context = model._contextualize_one_modality(
            pair_relations[modality], source, target, int(x.size(0)), modality
        )
        if not torch.allclose(reconstructed, full_relations[modality], atol=2e-5, rtol=2e-5):
            max_error = float((reconstructed - full_relations[modality]).abs().max().item())
            raise AssertionError(f"Reconstructed full context differs for {modality}: max error={max_error}")
        full_contexts[modality] = context
    return {
        "clean": clean,
        "source": source,
        "target": target,
        "reverse_idx": reverse_idx,
        "degree": degree_out,
        "h0": h0,
        "pair_evidence": pair_evidence,
        "full_evidence": full_evidence,
        "pair_relations": pair_relations,
        "full_relations": full_relations,
        "pair_conductance": pair_conductance,
        "full_conductance": full_conductance,
        "full_contexts": full_contexts,
    }


def _leave_pair_out_audit(model: Model, state: dict, dataset: str, seed: int):
    rows = []
    bin_rows = []
    lpo_evidence = {}
    source, target, reverse_idx = state["source"], state["target"], state["reverse_idx"]
    degree = state["degree"]
    edge_bins = edge_degree_bin_ids(degree, source, target)
    for modality in ("text", "visual"):
        r0 = state["pair_relations"][modality]
        rfull = state["full_relations"][modality]
        r_lpo, residual_lpo, _ct, _cs, _st, _ss = encode_leave_pair_out_contexts(
            model, r0, source, target, reverse_idx, int(degree.numel()), modality
        )
        residual_full = rfull - r0
        ratio_full = residual_full.norm(dim=-1) / (r0.norm(dim=-1) + EPS)
        ratio_lpo = residual_lpo.norm(dim=-1) / (r0.norm(dim=-1) + EPS)
        retention = residual_lpo.norm(dim=-1) / (residual_full.norm(dim=-1) + EPS)
        agreement_mask = (residual_full.norm(dim=-1) > 1e-8) & (residual_lpo.norm(dim=-1) > 1e-8)
        direction = F.cosine_similarity(
            residual_full[agreement_mask], residual_lpo[agreement_mask], dim=-1, eps=1e-8
        )
        c_lpo = relation_conductance(
            model, state["h0"][modality], r_lpo, source, target, modality
        )
        angle_full = model._edge_angles(rfull, modality)
        angle_lpo = model._edge_angles(r_lpo, modality)
        angle_difference = (angle_full - angle_lpo).abs().mean(dim=-1)
        c_pair = state["pair_conductance"][modality]
        c_full = state["full_conductance"][modality]
        angle_pair = model._edge_angles(r0, modality)
        c_full_change = (c_full - c_pair).abs()
        c_lpo_change = (c_lpo - c_pair).abs()
        angle_full_change = (angle_full - angle_pair).abs().mean(dim=-1)
        angle_lpo_change = (angle_lpo - angle_pair).abs().mean(dim=-1)
        row = {"dataset": dataset, "seed": seed, "modality": modality, "num_edges": int(source.numel()), "direction_agreement_edges": int(agreement_mask.sum().item())}
        for name, values in (
            ("full_residual_ratio", ratio_full),
            ("lpo_residual_ratio", ratio_lpo),
            ("other_context_retention", retention),
            ("direction_cosine", direction),
            ("abs_conductance_full_minus_pair", c_full_change),
            ("abs_conductance_lpo_minus_pair", c_lpo_change),
            ("mean_abs_angle_full_minus_pair", angle_full_change),
            ("mean_abs_angle_lpo_minus_pair", angle_lpo_change),
            ("mean_abs_angle_full_minus_lpo", angle_difference),
        ):
            row.update(_stats_row(name, values))
        rows.append(row)
        for bin_id, label in enumerate(DEGREE_EDGE_LABELS):
            mask = edge_bins == bin_id
            if not bool(mask.any()):
                continue
            valid_direction = mask & agreement_mask
            bin_row = {
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "degree_bin": label,
                "num_edges": int(mask.sum().item()),
                "direction_cosine_edges": int(valid_direction.sum().item()),
                "full_residual_ratio_mean": float(ratio_full[mask].mean().item()),
                "lpo_residual_ratio_mean": float(ratio_lpo[mask].mean().item()),
                "retention_mean": float(retention[mask].mean().item()),
                "direction_cosine_mean": float(direction[valid_direction].mean().item()) if bool(valid_direction.any()) else None,
                "full_vs_lpo_angle_difference_mean": float(angle_difference[mask].mean().item()),
            }
            bin_rows.append(bin_row)
        lpo_evidence[modality] = {
            "relation": r_lpo,
            "conductance": c_lpo,
            "angles": angle_lpo,
            "residual": residual_lpo,
            "ratio_full": ratio_full,
            "ratio_lpo": ratio_lpo,
            "retention": retention,
            "direction_cosine": direction,
            "agreement_mask": agreement_mask,
            "angle_difference": angle_difference,
        }
    lpo_evidence["evidence"] = (
        lpo_evidence["text"]["conductance"],
        lpo_evidence["visual"]["conductance"],
        lpo_evidence["text"]["relation"],
        lpo_evidence["visual"]["relation"],
    )
    return rows, bin_rows, lpo_evidence


def _context_shuffle_audit(model: Model, state: dict, dataset: str, seed: int):
    source, target, degree = state["source"], state["target"], state["degree"]
    base_bucket = degree_bucket_ids(degree)
    rows = []
    counterfactuals = []
    outputs = {}
    for shuffle_seed in SHUFFLE_SEEDS:
        permutation = degree_matched_permutation(degree, shuffle_seed)
        if not torch.equal(base_bucket[permutation], base_bucket):
            raise AssertionError("shuffled context crossed a degree bucket")
        shuffled_contexts = shuffle_context_pair(state["full_contexts"], permutation)
        if not torch.equal(shuffled_contexts["text"], state["full_contexts"]["text"][permutation]):
            raise AssertionError("text context permutation mismatch")
        if not torch.equal(shuffled_contexts["visual"], state["full_contexts"]["visual"][permutation]):
            raise AssertionError("visual context permutation mismatch")
        for modality in ("text", "visual"):
            context = state["full_contexts"][modality]
            shuffled = shuffled_contexts[modality]
            norm_full = context.norm(dim=-1)
            norm_shuffled = shuffled.norm(dim=-1)
            norm_multiset_equal = torch.equal(torch.sort(norm_full).values, torch.sort(norm_shuffled).values)
            if not norm_multiset_equal:
                raise AssertionError("degree-matched shuffle changed context norm multiset")
            r0 = state["pair_relations"][modality]
            edge_module = model.text_edge_context if modality == "text" else model.visual_edge_context
            r_shuffle, residual_shuffle = edge_module(
                r0, shuffled[target], shuffled[source], model.context_residual_scale
            )
            r_full = state["full_relations"][modality]
            c_shuffle = relation_conductance(
                model, state["h0"][modality], r_shuffle, source, target, modality
            )
            theta_shuffle = model._edge_angles(r_shuffle, modality)
            theta_full = model._edge_angles(r_full, modality)
            full_residual = r_full - r0
            residual_ratio_shuffle = residual_shuffle.norm(dim=-1) / (r0.norm(dim=-1) + EPS)
            shuffle_distance_relative = (r_full - r_shuffle).norm(dim=-1) / (full_residual.norm(dim=-1) + EPS)
            c_delta = (state["full_conductance"][modality] - c_shuffle).abs()
            angle_delta = (theta_full - theta_shuffle).abs().mean(dim=-1)
            row = {
                "dataset": dataset,
                "seed": seed,
                "shuffle_seed": shuffle_seed,
                "modality": modality,
                "num_edges": int(source.numel()),
                "same_node_permutation_for_both_modalities": True,
                "degree_bucket_distribution_preserved": True,
                "context_norm_multiset_equal": norm_multiset_equal,
            }
            row.update(_stats_row("context_norm_full", norm_full))
            row.update(_stats_row("context_norm_shuffled", norm_shuffled))
            row.update(_stats_row("shuffle_residual_ratio", residual_ratio_shuffle))
            row.update(_stats_row("full_to_shuffle_distance_over_full_residual", shuffle_distance_relative))
            row.update(_stats_row("abs_conductance_full_minus_shuffle", c_delta))
            row.update(_stats_row("mean_abs_angle_full_minus_shuffle", angle_delta))
            rows.append(row)
            outputs[modality] = (c_shuffle, r_shuffle)
        evidence = (
            outputs["text"][0],
            outputs["visual"][0],
            outputs["text"][1],
            outputs["visual"][1],
        )
        counterfactuals.append((shuffle_seed, evidence))
    return rows, counterfactuals


def _residual_diversity(
    residual: torch.Tensor, dataset: str, seed: int, modality: str
) -> dict:
    edge_count = residual.size(0)
    generator = torch.Generator(device="cpu").manual_seed(DIVERSITY_SAMPLE_SEED)
    sample_count = min(8192, edge_count)
    sample_idx = torch.randperm(edge_count, generator=generator)[:sample_count].to(residual.device)
    sample = residual[sample_idx].float()
    mean = sample.mean(dim=0)
    centered = sample - mean
    rms = torch.sqrt(sample.square().sum(dim=-1).mean())
    global_bias_ratio = mean.norm() / (rms + EPS)
    channel_variance = sample.var(dim=0, unbiased=False)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total_energy = energy.sum()
    if float(total_energy.item()) > 0:
        proportions = energy / total_energy
        top1_energy = proportions[0]
        top5_energy = proportions[:5].sum()
        effective_rank = torch.exp(-(proportions * torch.log(proportions.clamp_min(EPS))).sum())
    else:
        top1_energy = energy.new_tensor(0.0)
        top5_energy = energy.new_tensor(0.0)
        effective_rank = energy.new_tensor(0.0)
    if float(mean.norm().item()) > 1e-8:
        direction_cosine = F.cosine_similarity(sample, mean.expand_as(sample), dim=-1, eps=1e-8)
        direction = _safe_distribution(direction_cosine)
        direction_status = "defined"
    else:
        direction = {"count": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
        direction_status = "undefined_mean_direction"
    return {
        "dataset": dataset,
        "seed": seed,
        "modality": modality,
        "sample_seed": DIVERSITY_SAMPLE_SEED,
        "sampled_edges": int(sample_count),
        "global_bias_ratio": float(global_bias_ratio.item()),
        "residual_rms": float(rms.item()),
        "centered_top1_energy": float(top1_energy.item()),
        "centered_top5_energy": float(top5_energy.item()),
        "centered_effective_rank": float(effective_rank.item()),
        "mean_channel_variance": float(channel_variance.mean().item()),
        "median_channel_variance": float(channel_variance.median().item()),
        "channels_variance_gt_1e_6": int((channel_variance > 1e-6).sum().item()),
        "channels_variance_fraction_gt_1e_6": float((channel_variance > 1e-6).float().mean().item()),
        "mean_direction_status": direction_status,
        "mean_direction_cosine_mean": direction["mean"],
        "mean_direction_cosine_p10": direction["p10"],
        "mean_direction_cosine_p50": direction["p50"],
        "mean_direction_cosine_p90": direction["p90"],
    }


def _signed_correction_rows(model: Model, state: dict, dataset: str, seed: int) -> list[dict]:
    rows = []
    for modality in ("text", "visual"):
        correction = getattr(model, f"{modality}_relation_correction")
        pair = correction(state["pair_relations"][modality]).squeeze(-1)
        full = correction(state["full_relations"][modality]).squeeze(-1)
        for name, values in (("pair_only", pair), ("full_context", full), ("full_minus_pair", full - pair)):
            stats = _safe_distribution(values)
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "relation_state": name,
                "signed_mean": stats["mean"],
                "signed_std": stats["std"],
                "p10": stats["p10"],
                "p50": stats["p50"],
                "p90": stats["p90"],
                "fraction_positive": float((values > 0).float().mean().item()),
                "fraction_negative": float((values < 0).float().mean().item()),
                "num_edges": int(values.numel()),
            })
    return rows


def audit_one(dataset: str, seed: int, checkpoint: Path):
    model, classifier, data, x, edge_index, val_idx, val_labels, device = load_frozen_run(dataset, seed, checkpoint)
    model_hash_before = model_state_hash(model)
    classifier_hash_before = model_state_hash(classifier)
    with torch.inference_mode():
        state = _edge_relations_and_evidence(model, data, x, edge_index)
        lpo_rows, lpo_bin_rows, lpo = _leave_pair_out_audit(model, state, dataset, seed)
        signed_rows = _signed_correction_rows(model, state, dataset, seed)
        diversity_rows = []
        for modality in ("text", "visual"):
            diversity_rows.append(
                _residual_diversity(
                    state["full_relations"][modality] - state["pair_relations"][modality],
                    dataset,
                    seed,
                    modality,
                )
            )

        pair_metrics = validation_with_evidence(
            model, classifier, x, edge_index, val_idx, val_labels, int(data.num_classes),
            evidence=state["pair_evidence"],
        )
        full_metrics = validation_with_evidence(
            model, classifier, x, edge_index, val_idx, val_labels, int(data.num_classes)
        )
        context_off_metrics = validation_with_evidence(
            model, classifier, x, edge_index, val_idx, val_labels, int(data.num_classes), context_off=True
        )
        if pair_metrics != context_off_metrics:
            raise AssertionError("explicit pair evidence differs from M0.2 Context-Off helper")
        lpo_metrics = validation_with_evidence(
            model, classifier, x, edge_index, val_idx, val_labels, int(data.num_classes),
            evidence=lpo["evidence"],
        )
        full_lpo_pair = {
            "dataset": dataset, "seed": seed,
            "pair_acc": pair_metrics["acc"], "pair_macro_f1": pair_metrics["macro_f1"],
            "full_context_acc": full_metrics["acc"], "full_context_macro_f1": full_metrics["macro_f1"],
            "leave_pair_out_acc": lpo_metrics["acc"], "leave_pair_out_macro_f1": lpo_metrics["macro_f1"],
            "context_off_acc": context_off_metrics["acc"], "context_off_macro_f1": context_off_metrics["macro_f1"],
            "full_minus_lpo_delta_acc": full_metrics["acc"] - lpo_metrics["acc"],
            "full_minus_lpo_delta_macro_f1": full_metrics["macro_f1"] - lpo_metrics["macro_f1"],
            "lpo_minus_context_off_delta_acc": lpo_metrics["acc"] - context_off_metrics["acc"],
            "lpo_minus_context_off_delta_macro_f1": lpo_metrics["macro_f1"] - context_off_metrics["macro_f1"],
        }

        shuffle_rows, shuffled_evidence = _context_shuffle_audit(model, state, dataset, seed)
        shuffle_counterfactual_rows = []
        for shuffle_seed, evidence in shuffled_evidence:
            metrics = validation_with_evidence(
                model, classifier, x, edge_index, val_idx, val_labels, int(data.num_classes), evidence=evidence
            )
            shuffle_counterfactual_rows.append({
                "row_type": "shuffle_seed",
                "dataset": dataset,
                "seed": seed,
                "shuffle_seed": shuffle_seed,
                "full_context_acc": full_metrics["acc"],
                "full_context_macro_f1": full_metrics["macro_f1"],
                "shuffled_context_acc": metrics["acc"],
                "shuffled_context_macro_f1": metrics["macro_f1"],
                "context_off_acc": context_off_metrics["acc"],
                "context_off_macro_f1": context_off_metrics["macro_f1"],
                "full_minus_shuffle_delta_acc": full_metrics["acc"] - metrics["acc"],
                "full_minus_shuffle_delta_macro_f1": full_metrics["macro_f1"] - metrics["macro_f1"],
            })
        if not model._relation_context_enabled:
            raise AssertionError("Context-Off analysis did not restore the full-context mode")
        if model_state_hash(model) != model_hash_before:
            raise AssertionError(f"M0.2 model state changed during audit for {dataset} seed {seed}")
        if model_state_hash(classifier) != classifier_hash_before:
            raise AssertionError(f"NC head state changed during audit for {dataset} seed {seed}")

    return {
        "lpo_rows": lpo_rows,
        "lpo_bin_rows": lpo_bin_rows,
        "lpo_counterfactual": full_lpo_pair,
        "shuffle_rows": shuffle_rows,
        "shuffle_counterfactual_rows": shuffle_counterfactual_rows,
        "diversity_rows": diversity_rows,
        "signed_rows": signed_rows,
        "state_hash": model_hash_before,
        "classifier_hash": classifier_hash_before,
    }


def _write_results(all_rows: dict[str, list[dict]]):
    _write_csv(RESULTS / "leave_pair_out.csv", all_rows["lpo_rows"], list(all_rows["lpo_rows"][0]))
    _write_csv(RESULTS / "leave_pair_out_degree_bins.csv", all_rows["lpo_bin_rows"], list(all_rows["lpo_bin_rows"][0]))
    _write_csv(RESULTS / "leave_pair_out_counterfactual.csv", all_rows["lpo_counterfactual"], list(all_rows["lpo_counterfactual"][0]))
    _write_csv(RESULTS / "context_shuffle.csv", all_rows["shuffle_rows"], list(all_rows["shuffle_rows"][0]))
    _write_csv(RESULTS / "context_shuffle_counterfactual.csv", all_rows["shuffle_counterfactual"], list(all_rows["shuffle_counterfactual"][0]))
    _write_csv(RESULTS / "residual_diversity.csv", all_rows["diversity_rows"], list(all_rows["diversity_rows"][0]))
    _write_csv(RESULTS / "relation_correction_signed.csv", all_rows["signed_rows"], list(all_rows["signed_rows"][0]))


def main():
    # Verify every checkpoint before loading data or running any model calculation.
    checkpoint_rows = preflight_checkpoints()
    RESULTS.mkdir(parents=True, exist_ok=True)
    repaired_rows = repair_context_diagnostics()
    aggregate = {
        "lpo_rows": [],
        "lpo_bin_rows": [],
        "lpo_counterfactual": [],
        "shuffle_rows": [],
        "shuffle_counterfactual": [],
        "diversity_rows": [],
        "signed_rows": [],
        "state_hashes": [],
    }
    for index, item in enumerate(checkpoint_rows, start=1):
        dataset, seed, checkpoint = item["dataset"], item["seed"], item["path"]
        print(f"[{index}/6] frozen audit {dataset} seed={seed}", flush=True)
        result = audit_one(dataset, seed, checkpoint)
        for key in (
            "lpo_rows", "lpo_bin_rows", "shuffle_rows", "diversity_rows", "signed_rows",
            "shuffle_counterfactual_rows",
        ):
            destination = "shuffle_counterfactual" if key == "shuffle_counterfactual_rows" else key
            aggregate[destination].extend(result[key])
        aggregate["lpo_counterfactual"].append(result["lpo_counterfactual"])
        aggregate["state_hashes"].append({
            "dataset": dataset, "seed": seed,
            "model_state_sha256": result["state_hash"],
            "head_state_sha256": result["classifier_hash"],
        })
        print(
            f"    LPO retention text={result['lpo_rows'][0]['other_context_retention_mean']:.4f} "
            f"visual={result['lpo_rows'][1]['other_context_retention_mean']:.4f}; "
            f"Context val Acc={result['lpo_counterfactual']['full_context_acc']:.4f}/"
            f"{result['lpo_counterfactual']['leave_pair_out_acc']:.4f}/"
            f"{result['lpo_counterfactual']['context_off_acc']:.4f}",
            flush=True,
        )
    _write_results(aggregate)
    metadata_path = RESULTS / "source_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["corrected_context_diagnostics"] = {
        "rows": repaired_rows,
        "modalities": ["text", "visual"],
        "original_file_repaired": str(M02_RESULTS / "context_diagnostics.csv"),
        "audit_copy": str(RESULTS / "corrected_context_diagnostics.csv"),
    }
    metadata["post_audit_model_and_task_hashes"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in metadata["frozen_source_sha256_before"]
    }
    metadata["per_run_state_hashes"] = aggregate["state_hashes"]
    metadata["all_model_and_classifier_state_dicts_unchanged"] = True
    metadata["validation_rows_evaluated"] = True
    metadata["test_split_metrics_or_labels_used"] = False
    for path, before_hash in metadata["frozen_source_sha256_before"].items():
        if metadata["post_audit_model_and_task_hashes"][path] != before_hash:
            raise AssertionError(f"Frozen file changed: {path}")
    metadata["checkpoint_sha256_after"] = [
        {"path": str(row["path"]), "sha256": hashlib.sha256(row["path"].read_bytes()).hexdigest()}
        for row in checkpoint_rows
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit outputs written under {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
