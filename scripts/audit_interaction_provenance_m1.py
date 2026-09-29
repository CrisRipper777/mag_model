from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models.interaction_provenance_m1 import Model
from src.tasks.nc import _resolve_nc_eval_labels
from src.utils.device import get_device


def make_cfg(dataset: str, seed: int, checkpoint: Path):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=interaction_provenance_m1",
        f"seed={int(seed)}",
        "num_runs=1",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint.resolve()}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def evaluate_val(classifier, z, data, device):
    idx = data.val_idx.to(device)
    with torch.no_grad():
        pred = classifier(z[idx]).argmax(dim=-1).cpu()
    target = data.y[data.val_idx].cpu()
    labels = _resolve_nc_eval_labels(data)
    return {
        "val_acc": float((pred == target).float().mean().item()),
        "val_macro_f1": float(
            f1_score(
                target.numpy(),
                pred.numpy(),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def distribution(values: torch.Tensor):
    values = values.detach().float().reshape(-1)
    if not values.numel():
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    quantiles = torch.quantile(values, values.new_tensor([0.5, 0.9]))
    return {
        "mean": float(values.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
    }


def relative_state_difference(full, other):
    output = {}
    for modality in ("text", "visual"):
        for order, (left, right) in enumerate(
            zip(full[f"G_{modality}"], other[f"G_{modality}"], strict=True)
        ):
            relative = (left - right).norm(dim=-1) / (left.norm(dim=-1) + 1e-12)
            output[f"{modality}_G{order}"] = distribution(relative)
    return output


def atom_relative_difference(full, other):
    output = {}
    for modality in ("text", "visual"):
        for order, (left, right) in enumerate(
            zip(
                full["atom_samples"][modality],
                other["atom_samples"][modality],
                strict=True,
            ),
            start=1,
        ):
            if left is None or right is None:
                value = {"mean": 0.0, "p50": 0.0, "p90": 0.0}
            else:
                relative = (left - right).norm(dim=-1) / (left.norm(dim=-1) + 1e-12)
                value = distribution(relative)
            output[f"{modality}_k{order}"] = value
    return output


def semantic_max_error(full, other):
    maximum = 0.0
    for modality in ("text", "visual"):
        for left, right in zip(
            full[f"S_{modality}"], other[f"S_{modality}"], strict=True
        ):
            maximum = max(maximum, float((left - right).abs().max().item()))
    return maximum


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def ensure_finite(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            ensure_finite(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            ensure_finite(child, f"{path}[{index}]")
    elif isinstance(value, (float, int)):
        if not torch.isfinite(torch.tensor(float(value))):
            raise AssertionError(f"Non-finite value at {path}: {value}")


def run(args):
    checkpoint = Path(args.checkpoint)
    cfg = make_cfg(args.dataset, args.seed, checkpoint)
    if str(cfg.task.name) != "nc" or bool(cfg.task.evaluate_test):
        raise AssertionError("Audit must use NC validation only with test evaluation disabled")
    device = get_device(str(cfg.device))
    data = load_mag_data(cfg, "nc", int(args.seed))
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = Model(cfg, data_info).to(device)
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    classifier.load_state_dict(payload["head_state"])
    model.eval()
    classifier.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        full = model._compute(x, edge_index, collect_diagnostics=True)
        forward_z, *_ = model(x, edge_index)
        inferred = model.inference(x, edge_index, device=device, batch_size=4096).to(device)
    inference_max_error = float((forward_z - inferred).abs().max().item())
    if inference_max_error > 2e-5:
        raise AssertionError(f"M1 inference mismatch: {inference_max_error}")
    full_metrics = evaluate_val(classifier, full["z"], data, device)
    model_analysis = model.analyze(x, edge_index)
    ensure_finite(model_analysis)
    if not torch.isfinite(full["z"]).all():
        raise AssertionError("Full M1 output contains NaN/Inf")

    modes = ["adapter_off", "history_off", "relation_off", "effect_off"]
    variants: list[tuple[str, int | None]] = [(mode, None) for mode in modes]
    variants.extend(("history_shuffle", seed) for seed in (3407, 3408, 3409))
    state_hash_before = state_fingerprint(model)
    counterfactuals = []
    for mode, shuffle_seed in variants:
        with torch.no_grad():
            result = model._compute(
                x,
                edge_index,
                provenance_mode=mode,
                history_shuffle_seed=shuffle_seed,
                collect_diagnostics=True,
                semantic_cache=full["semantic_cache"],
            )
        sem_error = semantic_max_error(full, result)
        if sem_error > 2e-6:
            raise AssertionError(
                f"Semantic stream changed in {mode}/{shuffle_seed}: max error={sem_error}"
            )
        metrics = evaluate_val(classifier, result["z"], data, device)
        counterfactuals.append(
            {
                "mode": mode,
                "shuffle_seed": shuffle_seed,
                **metrics,
                "delta_val_acc_vs_full": metrics["val_acc"] - full_metrics["val_acc"],
                "delta_val_macro_f1_vs_full": metrics["val_macro_f1"]
                - full_metrics["val_macro_f1"],
                "semantic_max_abs_error": sem_error,
                "provenance_geometry": relative_state_difference(full, result),
                "atom_geometry": atom_relative_difference(full, result),
                "residual_relative_difference": distribution(
                    (full["provenance_residual"] - result["provenance_residual"])
                    .norm(dim=-1)
                    / (full["provenance_residual"].norm(dim=-1) + 1e-12)
                ),
                "G3_full_vs_variant_relative_difference": {
                    modality: distribution(
                        (full[f"G_{modality}"][3] - result[f"G_{modality}"][3]).norm(dim=-1)
                        / (full[f"G_{modality}"][3].norm(dim=-1) + 1e-12)
                    )
                    for modality in ("text", "visual")
                },
            }
        )
        ensure_finite(counterfactuals[-1])
    state_hash_after = state_fingerprint(model)
    if state_hash_before != state_hash_after:
        raise AssertionError("A frozen counterfactual modified checkpoint parameters")

    report = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "checkpoint": checkpoint.resolve().relative_to(ROOT).as_posix(),
        "protocol": {
            "task": "nc",
            "checkpoint_selection": "original validation accuracy",
            "test_evaluation": False,
            "lp": False,
        },
        "checkpoint_metrics": payload["metrics"],
        "full_validation": full_metrics,
        "inference_max_abs_error": inference_max_error,
        "semantic_invariance_tolerance": 2e-6,
        "parameter_state_unchanged": state_hash_before == state_hash_after,
        "parameter_sha256": state_hash_after,
        "analysis": model_analysis,
        "counterfactuals": counterfactuals,
    }
    ensure_finite(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "full_validation": full_metrics}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("Movies", "Grocery"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
