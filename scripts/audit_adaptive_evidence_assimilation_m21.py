from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models.adaptive_evidence_assimilation_m21 import Model as M21Model
from src.models.provenance_evidence_m2 import Model as M2Model
from src.utils.device import get_device


def make_cfg(dataset: str, seed: int, variant: str, assimilation_mode: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=adaptive_evidence_assimilation_m21",
            f"model.provenance_conditioning={'true' if variant.endswith('-P') else 'false'}",
            f"model.assimilation_mode={assimilation_mode}", f"seed={seed}", "num_runs=1",
            "task.evaluate_test=false", "task.save_ckpt_path=null",
        ])


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    if not values.numel():
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    q = torch.quantile(values, values.new_tensor([0.5, 0.9]))
    return {"mean": float(values.mean()), "p50": float(q[0]), "p90": float(q[1])}


def relative(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).norm(dim=-1) / (right.norm(dim=-1) + 1e-12)


def classwise(classifier, z, data, classes):
    idx = data.val_idx.cpu()
    with torch.no_grad():
        pred = classifier(z[idx.to(z.device)]).argmax(dim=-1).cpu()
    target = data.y[idx].cpu()
    rows = []
    for c in classes:
        binary_target, binary_pred = target == c, pred == c
        tp = int((binary_target & binary_pred).sum())
        fp = int((~binary_target & binary_pred).sum())
        fn = int((binary_target & ~binary_pred).sum())
        rows.append({"class": int(c), "support": int(binary_target.sum()), "f1": float(f1_score(binary_target, binary_pred, zero_division=0)),
                     "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0})
    return rows


def validation(classifier, z, data, classes):
    idx = data.val_idx.to(z.device)
    with torch.no_grad():
        logits = classifier(z[idx])
        pred = logits.argmax(dim=-1).cpu()
    target = data.y[data.val_idx.cpu()].cpu()
    return {"acc": float((pred == target).float().mean().item()),
            "macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=classes, average="macro", zero_division=0))}


def stage12_error(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    errors = []
    for modality in ("text", "visual"):
        errors.append((reference[f"H0_{modality}"] - candidate[f"H0_{modality}"]).abs().max())
        for key in ("S", "G"):
            for left, right in zip(reference[f"{key}_{modality}"], candidate[f"{key}_{modality}"], strict=True):
                errors.append((left - right).abs().max())
    return max(float(value.item()) for value in errors)


def sampled_attention(model, details, indices):
    with torch.no_grad():
        _, attention = model._retrieve(details["queries"][indices], details["evidence"]["keys"][indices],
                                       details["evidence"]["values"][indices], return_attention=True)
    return attention.mean(dim=2)


@torch.no_grad()
def evaluate_counterfactual(model, classifier, data, device, reference, full_attention,
                            mode: str, *, seed: int | None = None, assimilation: bool = False):
    kwargs = {"return_details": True, "stage12_override": reference["stage12"]}
    if assimilation:
        kwargs["assimilation_intervention"] = mode
        if seed is not None:
            kwargs["assimilation_seed"] = seed
        intervention_name = mode
    else:
        kwargs["intervention"] = mode
        if seed is not None:
            kwargs["shuffle_seed"] = seed
        intervention_name = mode
    changed = model.forward_with_intervention(data.x.to(device), data.edge_index.to(device), **kwargs)
    stage_error = stage12_error(reference["stage12"], changed["stage12"])
    if stage_error > 2e-6:
        raise AssertionError(f"Stage I/II changed under {intervention_name}: max_abs={stage_error}")
    count = min(data.num_nodes, 8192)
    indices = torch.linspace(0, data.num_nodes - 1, steps=count, device=device).round().long()
    changed_attention = sampled_attention(model, changed, indices)
    key_change = distribution(relative(changed["evidence"]["keys"][:, :6], reference["evidence"]["keys"][:, :6]))
    value_change = distribution(relative(changed["evidence"]["values"][:, :6], reference["evidence"]["values"][:, :6]))
    ret_change = distribution(relative(changed["retrieval"], reference["retrieval"]))
    record = {"mode": intervention_name, "shuffle_seed": seed,
              "stage12_max_abs_error": stage_error,
              "key_relative_change": key_change, "value_relative_change": value_change,
              "retrieval_relative_change": ret_change,
              "final_z_relative_change": distribution(relative(changed["z"], reference["z"])),
              "attention_l1_change": distribution((changed_attention - full_attention).abs().sum(dim=-1)),
              "attention_sampled_nodes": count,
              "validation": validation(classifier, changed["z"], data, list(range(int(data.num_classes)))),
              "validation_classwise": classwise(classifier, changed["z"], data, list(range(int(data.num_classes))))}
    del changed, changed_attention
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return record


def gradient_probe(model, classifier, data, device):
    model.train(); classifier.train()
    model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)
    z = model(data.x.to(device), data.edge_index.to(device))[0]
    idx = data.train_idx.to(device)
    loss = torch.nn.functional.cross_entropy(classifier(z[idx]), data.y[idx.cpu()].to(device))
    loss.backward()
    names = ["assimilation_out_text.weight", "assimilation_out_visual.weight", "assimilation_in_text.weight",
             "assimilation_in_visual.weight", "text_query.weight", "visual_query.weight", "query_head_projection.weight",
             "key_head_projection.weight", "value_head_projection.weight", "output_projection.weight"]
    if model.provenance_conditioning:
        names += [f"{kind}_{modality}.weight" for modality in ("text", "visual") for kind in ("key_provenance", "key_interaction", "value_interaction")]
    report = {}
    for name in names:
        parameter = dict(model.named_parameters()).get(name)
        grad = None if parameter is None else parameter.grad
        report[name] = {"present": grad is not None,
                        "finite": bool(torch.isfinite(grad).all()) if grad is not None else False,
                        "nonzero": bool(torch.count_nonzero(grad)) if grad is not None else False,
                        "norm": float(grad.float().norm().item()) if grad is not None else 0.0}
    model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)
    model.eval(); classifier.eval()
    return report


def assert_finite(value: Any, path="root"):
    if isinstance(value, dict):
        for key, child in value.items(): assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value): assert_finite(child, f"{path}[{i}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")


def run(dataset: str, seed: int, variant: str, assimilation_mode: str, checkpoint_path: Path, output_path: Path):
    cfg = make_cfg(dataset, seed, variant, assimilation_mode)
    if str(cfg.task.name) != "nc" or bool(cfg.task.evaluate_test):
        raise AssertionError("M2.1 audit must be NC-only with test evaluation disabled")
    data = load_mag_data(cfg, "nc", seed)
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes,
            "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1])}
    device = get_device(str(cfg.device))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != seed: raise ValueError("Checkpoint seed mismatch")
    model = M21Model(cfg, info).to(device); model.load_state_dict(checkpoint["model_state"])
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"])
    model.eval(); classifier.eval()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    gradients = gradient_probe(model, classifier, data, device)
    with torch.no_grad():
        details = model.forward_with_intervention(data.x.to(device), data.edge_index.to(device), return_details=True, return_attention=True)
        inference = model.inference(data.x, data.edge_index, device=device)
        inference_error = float((inference.to(device) - details["z"]).abs().max().item())
        if inference_error > 1e-4: raise AssertionError(f"Inference mismatch: {inference_error}")
        full_validation = validation(classifier, details["z"], data, list(range(int(data.num_classes))))
        full_classwise = classwise(classifier, details["z"], data, list(range(int(data.num_classes))))
        analysis = model.analyze(data.x, data.edge_index)
        sample_count = min(data.num_nodes, 8192)
        indices = torch.linspace(0, data.num_nodes - 1, steps=sample_count, device=device).round().long()
        full_attention = sampled_attention(model, details, indices)

    # Rebuild frozen M2 from the same saved shared state and verify Stage III retrieval exactly.
    m2_cfg = make_cfg(dataset, seed, variant, assimilation_mode)
    m2 = M2Model(m2_cfg, info).to(device); m2.load_state_dict(checkpoint["model_state"], strict=False); m2.eval()
    with torch.no_grad():
        m2_details = m2.forward_with_intervention(data.x.to(device), data.edge_index.to(device), return_details=True, return_attention=True, stage12_override=details["stage12"])
        retrieval_error = float((m2_details["retrieval"] - details["retrieval"]).abs().max().item())
        attention_error = float((m2_details["attention"] - details["attention"]).abs().max().item())
    if retrieval_error > 2e-6 or attention_error > 2e-6:
        raise AssertionError(f"M2 retrieval changed: retrieval={retrieval_error}, attention={attention_error}")
    del m2, m2_details
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    counterfactuals = []
    if variant.endswith("-P") and assimilation_mode == "adaptive_vector":
        for mode in ("assimilation_off", "uniform_node", "dimension_shuffle"):
            counterfactuals.append(evaluate_counterfactual(model, classifier, data, device, details, full_attention,
                                                           mode, seed=7311 if mode == "dimension_shuffle" else None, assimilation=True))
        counterfactuals.append(evaluate_counterfactual(model, classifier, data, device, details, full_attention, "provenance_off"))
        for shuffle_seed in (3407, 3408, 3409):
            counterfactuals.append(evaluate_counterfactual(model, classifier, data, device, details, full_attention,
                                                           "provenance_shuffle", seed=shuffle_seed))
        counterfactuals.append(evaluate_counterfactual(model, classifier, data, device, details, full_attention, "order_mismatch"))
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak = {"allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    else: peak = {"allocated_bytes": 0, "reserved_bytes": 0}
    report = {"dataset": dataset, "seed": seed, "variant": variant, "assimilation_mode": assimilation_mode,
              "checkpoint": str(checkpoint_path.resolve()),
              "protocol": {"task": "NC", "training_split": "original train", "selection_split": "original validation",
                           "test_evaluation": False, "link_prediction": False},
              "full_validation": full_validation, "full_validation_classwise": full_classwise,
              "analysis": analysis, "counterfactuals": counterfactuals,
              "gradient_probe": gradients, "inference_max_abs_error": inference_error,
              "M2_retrieval_max_abs_error": retrieval_error, "M2_attention_max_abs_error": attention_error,
              "counterfactual_invariant_tolerance": 2e-6, "peak_memory": peak}
    assert_finite(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": dataset, "seed": seed, "variant": variant, "assimilation_mode": assimilation_mode,
                      "full_validation": full_validation, "counterfactual_count": len(counterfactuals),
                      "M2_retrieval_max_abs_error": retrieval_error, "output": str(output_path)}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Movies", "Grocery"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("M2.1-S", "M2.1-P"), required=True)
    parser.add_argument("--assimilation-mode", choices=("adaptive_vector", "global_scalar"), default="adaptive_vector")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.seed, args.variant, args.assimilation_mode, args.checkpoint, args.output)


if __name__ == "__main__": main()
