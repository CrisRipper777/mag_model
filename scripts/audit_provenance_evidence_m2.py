from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels
from src.utils.device import get_device


def make_cfg(dataset: str, seed: int, variant: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=provenance_evidence_m2",
                f"model.provenance_conditioning={'true' if variant == 'M2-P' else 'false'}",
                f"seed={seed}",
                "num_runs=1",
                "task.evaluate_test=false",
                "task.save_ckpt_path=null",
            ],
        )


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    if not values.numel():
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    q = torch.quantile(values, values.new_tensor([0.5, 0.9]))
    return {"mean": float(values.mean()), "p50": float(q[0]), "p90": float(q[1])}


def _relative(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-12)


def _stage12_max_error(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    errors = []
    for modality in ("text", "visual"):
        errors.append((reference[f"H0_{modality}"] - candidate[f"H0_{modality}"]).abs().max())
        for key in ("S", "G"):
            for left, right in zip(reference[f"{key}_{modality}"], candidate[f"{key}_{modality}"], strict=True):
                errors.append((left - right).abs().max())
    return max(float(error.item()) for error in errors)


def _sampled_attention(model, details, sample_indices, *, same_modality_only=False):
    with torch.no_grad():
        _, alpha = model._retrieve(
            details["queries"][sample_indices],
            details["evidence"]["keys"][sample_indices],
            details["evidence"]["values"][sample_indices],
            same_modality_only=same_modality_only,
            return_attention=True,
        )
    return alpha.mean(dim=2)  # [nodes, query modality, 7 tokens]


def _validation_metrics(classifier, z, data, device) -> dict[str, float]:
    return _evaluate_split(
        classifier,
        z.detach().cpu(),
        data.y,
        data.val_idx,
        device,
        4096,
        _resolve_nc_eval_labels(data),
    )


@torch.no_grad()
def _evaluate_variant(model, classifier, data, device, full_details, full_attention, intervention, shuffle_seed=None):
    kwargs = {"intervention": intervention, "return_details": True, "stage12_override": full_details["stage12"]}
    if shuffle_seed is not None:
        kwargs["shuffle_seed"] = shuffle_seed
    variant_details = model.forward_with_intervention(
        data.x.to(device), data.edge_index.to(device), **kwargs
    )
    invariant_error = _stage12_max_error(full_details["stage12"], variant_details["stage12"])
    if invariant_error > 2e-6:
        raise AssertionError(f"Stage I/II changed under {intervention}: max_abs={invariant_error}")

    full_evidence, changed_evidence = full_details["evidence"], variant_details["evidence"]
    full_z, changed_z = full_details["z"], variant_details["z"]
    full_retrieval, changed_retrieval = full_details["retrieval"], variant_details["retrieval"]
    indices = torch.linspace(0, data.num_nodes - 1, steps=min(data.num_nodes, 8192), device=device).round().long()
    changed_attention = _sampled_attention(
        model, variant_details, indices, same_modality_only=intervention == "same_modality_only"
    )
    attention_delta = (changed_attention - full_attention).abs().sum(dim=-1)
    record = {
        "mode": intervention,
        "shuffle_seed": shuffle_seed,
        "stage12_max_abs_error": invariant_error,
        "key_relative_change": _distribution(_relative(changed_evidence["keys"][:, :6], full_evidence["keys"][:, :6])),
        "value_relative_change": _distribution(_relative(changed_evidence["values"][:, :6], full_evidence["values"][:, :6])),
        "retrieval_relative_change": _distribution(_relative(changed_retrieval, full_retrieval)),
        "z_text_relative_change": _distribution(_relative(variant_details["Z_text"], full_details["Z_text"])),
        "z_visual_relative_change": _distribution(_relative(variant_details["Z_visual"], full_details["Z_visual"])),
        "final_z_relative_change": _distribution(_relative(changed_z, full_z)),
        "attention_l1_change": _distribution(attention_delta),
        "attention_sampled_nodes": int(indices.numel()),
        "validation": _validation_metrics(classifier, changed_z, data, device),
    }
    del variant_details, changed_evidence, changed_z, changed_retrieval, changed_attention
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _assert_finite_tree(value: Any, path: str = "root"):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite audit value at {path}: {value}")


def run(dataset: str, seed: int, variant: str, checkpoint_path: Path, output_path: Path):
    cfg = make_cfg(dataset, seed, variant)
    if str(cfg.task.name) != "nc" or bool(cfg.task.evaluate_test):
        raise AssertionError("M2 audit must be NC-only with test evaluation disabled")
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    device = get_device(str(cfg.device))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != seed:
        raise ValueError("Checkpoint seed does not match audit seed")
    model = build_model(cfg, data_info).to(device)
    model.load_state_dict(checkpoint["model_state"])
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"])
    model.eval()
    classifier.eval()

    with torch.no_grad():
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        full_details = model.forward_with_intervention(
            x, edge_index, intervention="full", return_details=True
        )
        z = full_details["z"]
        if not torch.isfinite(z).all():
            raise ValueError("Full M2 representation contains NaN/Inf")
        full_validation = _validation_metrics(classifier, z, data, device)
        analysis = model.analyze(data.x, data.edge_index)
        indices = torch.linspace(0, data.num_nodes - 1, steps=min(data.num_nodes, 8192), device=device).round().long()
        full_attention = _sampled_attention(model, full_details, indices)

    counterfactuals = []
    if variant == "M2-P":
        counterfactuals.append(
            _evaluate_variant(model, classifier, data, device, full_details, full_attention, "provenance_off")
        )
        for shuffle_seed in (3407, 3408, 3409):
            counterfactuals.append(
                _evaluate_variant(
                    model, classifier, data, device, full_details, full_attention,
                    "provenance_shuffle", shuffle_seed=shuffle_seed,
                )
            )
        counterfactuals.append(
            _evaluate_variant(model, classifier, data, device, full_details, full_attention, "order_mismatch")
        )
        counterfactuals.append(
            _evaluate_variant(model, classifier, data, device, full_details, full_attention, "same_modality_only")
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_memory = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    else:
        peak_memory = {"allocated_bytes": 0, "reserved_bytes": 0}
    report = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "checkpoint": str(checkpoint_path.resolve()),
        "protocol": {
            "task": "NC",
            "training_split": "original train",
            "selection_split": "original validation",
            "test_evaluation": False,
            "link_prediction": False,
        },
        "model_parameter_counts": analysis["parameter_counts"],
        "legacy_readout_frozen_statement": analysis["legacy_readout_frozen_statement"],
        "full_validation": full_validation,
        "analysis": analysis,
        "counterfactuals": counterfactuals,
        "counterfactual_invariant_tolerance": 2e-6,
        "peak_memory": peak_memory,
    }
    _assert_finite_tree(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": dataset, "seed": seed, "variant": variant, "full_validation": full_validation, "counterfactual_count": len(counterfactuals), "output": str(output_path)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Movies", "Grocery"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("M2-S", "M2-P"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.seed, args.variant, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
