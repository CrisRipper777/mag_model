from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.utils.device import get_device

RESULTS = ROOT / "results/model_design/m01_audit"
CHECKPOINTS = ROOT / "outputs/model_design/m0/checkpoints"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)


def _cfg(dataset: str, seed: int, checkpoint: Path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=relation_transport_m0",
                "model.transport_mode=orthogonal",
                f"seed={seed}",
                "num_runs=1",
                "device=cuda:0",
                "task.training_mode=full_graph",
                "task.inference_mode=full",
                "task.evaluate_test=false",
                f"task.save_ckpt_path={checkpoint}",
            ],
        )


def _summary(values: torch.Tensor, quantiles=(0.5, 0.9)) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    q = torch.quantile(values, values.new_tensor(quantiles))
    out = {"mean": float(values.mean().item())}
    for prob, value in zip(quantiles, q, strict=True):
        out[f"p{int(prob * 100)}"] = float(value.item())
    return out


def _rho(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa = a.detach().float().cpu().numpy().reshape(-1)
    bb = b.detach().float().cpu().numpy().reshape(-1)
    result = spearmanr(aa, bb).statistic
    return None if not np.isfinite(result) else float(result)


def _metrics(model, head, data, z: torch.Tensor) -> dict[str, float]:
    idx = data.val_idx.long().cpu()
    labels = data.y[idx].long().cpu()
    head.eval()
    with torch.no_grad():
        logits = head(z[idx].to(next(head.parameters()).device))
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    target = labels.numpy()
    return {
        "acc": float((pred == target).mean()),
        "macro_f1": float(
            f1_score(
                target,
                pred,
                labels=list(range(int(data.num_classes))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _audit_one(dataset: str, seed: int, checkpoint: Path):
    cfg = _cfg(dataset, seed, checkpoint)
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = build_model(cfg, info)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("selection") != "best_val_accuracy":
        raise ValueError(f"Checkpoint is not a validation-selected checkpoint: {checkpoint}")
    model.load_state_dict(saved["model_state"])
    head = torch.nn.Linear(model.out_dim, int(data.num_classes))
    head.load_state_dict(saved["head_state"])
    device = get_device("cuda:0")
    model.to(device).eval()
    head.to(device).eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        result = model._compute(x, edge_index)
        source, target = result["source"], result["target"]
        dynamics_rows = []
        compatibility_rows = []
        association_rows = []
        h0_by_modality = {}
        relation_by_modality = {}
        conductance_by_modality = {}
        angles_by_modality = {}
        for modality in ("text", "visual"):
            h0 = result[f"H0_{modality}"]
            states = result[f"S_{modality}"]
            h0_by_modality[modality] = h0
            for order in (1, 2, 3):
                previous, current = states[order - 1], states[order]
                relative_delta = (current - previous).norm(dim=-1) / (
                    previous.norm(dim=-1) + 1e-12
                )
                adjacent_cos = torch.nn.functional.cosine_similarity(
                    previous, current, dim=-1, eps=1e-8
                )
                novelty = 1.0 - torch.nn.functional.cosine_similarity(
                    h0, current, dim=-1, eps=1e-8
                )
                dynamics_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": "orthogonal_transport",
                        "modality": modality,
                        "order": order,
                        **{f"relative_delta_{k}": v for k, v in _summary(relative_delta, (0.5, 0.9)).items()},
                        **{f"adjacent_cos_{k}": v for k, v in _summary(adjacent_cos, (0.1, 0.5)).items()},
                        **{f"cumulative_novelty_{k}": v for k, v in _summary(novelty, (0.5, 0.9)).items()},
                    }
                )

            rel_text, rel_visual = model._relation_evidence(
                result["H0_text"], result["H0_visual"], source, target
            )[2:]
            relation = rel_text if modality == "text" else rel_visual
            relation_by_modality[modality] = relation
            h = result[f"H0_{modality}"]
            similarity = torch.nn.functional.cosine_similarity(
                h[target], h[source], dim=-1, eps=1e-8
            )
            tau = torch.nn.functional.softplus(getattr(model, f"raw_tau_{modality}"))
            prior = tau * similarity
            correction = getattr(model, f"{modality}_relation_correction")(relation).squeeze(-1)
            logit = prior + correction
            conductance = model.compatibility_scale * torch.sigmoid(logit)
            conductance_by_modality[modality] = conductance
            angles = model._edge_angles(relation, modality)
            angles_by_modality[modality] = angles
            correction_share = correction.abs() / (prior.abs() + correction.abs() + 1e-12)
            valid = prior.abs() >= 1e-8
            flip_rate = float(
                (torch.sign(prior[valid]) != torch.sign(logit[valid])).float().mean().item()
            ) if bool(valid.any()) else float("nan")
            compatibility_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "prior_abs": _summary(prior.abs(), (0.5, 0.9)),
                    "correction_abs": _summary(correction.abs(), (0.5, 0.9)),
                    "correction_share": _summary(correction_share, (0.5, 0.9)),
                    "prior_flip_rate": flip_rate,
                    "softplus_tau": float(tau.item()),
                    "num_edges": int(source.numel()),
                }
            )
            sample_count = min(4096, int(source.numel()))
            sample_idx = torch.linspace(0, source.numel() - 1, sample_count, device=device).round().long()
            sampled_source = source[sample_idx]
            sampled_angles = angles[sample_idx]
            transported = model._givens_transport(h[sampled_source], sampled_angles)
            displacement = (transported - h[sampled_source]).norm(dim=-1) / (
                h[sampled_source].norm(dim=-1) + 1e-12
            )
            angle_magnitude = sampled_angles.abs().mean(dim=-1)
            sampled_cosine = similarity[sample_idx]
            sampled_conductance = conductance[sample_idx]
            sampled_correction = correction[sample_idx].abs()
            association_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "sampled_edges": sample_count,
                    "rho_conductance_displacement": _rho(sampled_conductance, displacement),
                    "rho_cosine_mean_abs_angle": _rho(sampled_cosine, angle_magnitude),
                    "rho_abs_correction_mean_abs_angle": _rho(sampled_correction, angle_magnitude),
                }
            )

        full_z = model.inference(data.x, data.edge_index, device=device)
        full_metrics = _metrics(model, head, data, full_z)
        counterfactual_rows = []
        original_mode = model.transport_mode
        model.transport_mode = "identity"
        try:
            transport_off_z = model.inference(data.x, data.edge_index, device=device)
        finally:
            model.transport_mode = original_mode
        transport_off_metrics = _metrics(model, head, data, transport_off_z)
        counterfactual_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "counterfactual": "transport_off",
                "full_acc": full_metrics["acc"],
                "full_macro_f1": full_metrics["macro_f1"],
                "counterfactual_acc": transport_off_metrics["acc"],
                "counterfactual_macro_f1": transport_off_metrics["macro_f1"],
                "delta_acc_full_minus_counterfactual": full_metrics["acc"] - transport_off_metrics["acc"],
                "delta_macro_f1_full_minus_counterfactual": full_metrics["macro_f1"] - transport_off_metrics["macro_f1"],
            }
        )

        before_hash = _state_hash(model)
        saved_corrections = {}
        for modality in ("text", "visual"):
            layer = getattr(model, f"{modality}_relation_correction")
            saved_corrections[modality] = (layer.weight.detach().clone(), layer.bias.detach().clone())
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()
        try:
            prior_only_z = model.inference(data.x, data.edge_index, device=device)
        finally:
            with torch.no_grad():
                for modality, (weight, bias) in saved_corrections.items():
                    layer = getattr(model, f"{modality}_relation_correction")
                    layer.weight.copy_(weight)
                    layer.bias.copy_(bias)
        after_hash = _state_hash(model)
        if before_hash != after_hash:
            raise AssertionError(f"M0.1 parameters were not restored exactly for {checkpoint}")
        prior_only_metrics = _metrics(model, head, data, prior_only_z)
        counterfactual_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "counterfactual": "relation_correction_off_prior_only",
                "full_acc": full_metrics["acc"],
                "full_macro_f1": full_metrics["macro_f1"],
                "counterfactual_acc": prior_only_metrics["acc"],
                "counterfactual_macro_f1": prior_only_metrics["macro_f1"],
                "delta_acc_full_minus_counterfactual": full_metrics["acc"] - prior_only_metrics["acc"],
                "delta_macro_f1_full_minus_counterfactual": full_metrics["macro_f1"] - prior_only_metrics["macro_f1"],
            }
        )

    # Flatten nested distribution summaries for a conventional CSV schema.
    flat_compatibility = []
    for row in compatibility_rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for key in ("prior_abs", "correction_abs", "correction_share"):
            for stat, value in row[key].items():
                flat[f"{key}_{stat}"] = value
        flat_compatibility.append(flat)
    return dynamics_rows, flat_compatibility, association_rows, counterfactual_rows, full_metrics


def main():
    required = [CHECKPOINTS / f"{ds}_orthogonal_transport_seed{seed}.pt" for ds in DATASETS for seed in SEEDS]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise SystemExit("Required frozen M0.1 checkpoint missing; audit stopped without training:\n" + "\n".join(map(str, missing)))
    RESULTS.mkdir(parents=True, exist_ok=True)
    collected = {"dynamics": [], "compatibility": [], "association": [], "counterfactuals": []}
    summaries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            checkpoint = CHECKPOINTS / f"{dataset}_orthogonal_transport_seed{seed}.pt"
            print(f"audit frozen checkpoint {dataset} seed={seed}: {checkpoint.name}", flush=True)
            rows = _audit_one(dataset, seed, checkpoint)
            for key, value in zip(collected, rows[:4], strict=True):
                collected[key].extend(value)
            summaries.append({"dataset": dataset, "seed": seed, **rows[4]})
    _write_csv(RESULTS / "progressive_dynamics.csv", collected["dynamics"])
    _write_csv(RESULTS / "compatibility_decomposition.csv", collected["compatibility"])
    _write_csv(RESULTS / "transport_association.csv", collected["association"])
    _write_csv(RESULTS / "frozen_counterfactuals.csv", collected["counterfactuals"])
    (RESULTS / "frozen_validation_metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"wrote frozen M0.1 audit to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
