from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models.provenance_evidence_m2 import Model as M2Model
from src.utils.device import get_device


def make_cfg(dataset: str, seed: int, variant: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=provenance_evidence_m2",
            f"model.provenance_conditioning={'true' if variant == 'M2-P' else 'false'}",
            f"seed={seed}", "num_runs=1", "task.evaluate_test=false", "task.save_ckpt_path=null",
        ])


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {key: 0.0 for key in ("mean", "p10", "p50", "p90")}
    q = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
    return {"mean": float(values.mean()), "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}


def class_f1(y: torch.Tensor, pred: torch.Tensor, indices: torch.Tensor, classes: list[int]) -> dict[int, float]:
    target, predicted = y[indices].cpu().numpy(), pred[indices].cpu().numpy()
    return {int(c): float(f1_score(target == c, predicted == c, zero_division=0)) for c in classes}


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    from scipy.stats import spearmanr
    result = spearmanr(x.detach().float().cpu().numpy(), y.detach().float().cpu().numpy())
    return float(result.statistic) if math.isfinite(float(result.statistic)) else 0.0


def load_outputs(dataset: str, seed: int, variant: str, device: torch.device):
    cfg = make_cfg(dataset, seed, variant)
    data = load_mag_data(cfg, "nc", seed)
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes,
            "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1])}
    checkpoint_path = ROOT / f"outputs/model_design/m2/pilot/{variant}/{dataset}/seed{seed}/best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != seed:
        raise ValueError(f"Seed mismatch in {checkpoint_path}")
    model = M2Model(cfg, info).to(device)
    model.load_state_dict(checkpoint["model_state"])
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"])
    model.eval(); classifier.eval()
    with torch.no_grad():
        details = model.forward_with_intervention(data.x.to(device), data.edge_index.to(device), return_details=True)
        z = details["z"]
        logits = classifier(z)
        return data, details, logits, z


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/model_design/m21")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device("cuda:0" if torch.cuda.is_available() else "cpu")
    summary_rows, correctness_rows, class_rows, corr_rows = [], [], [], []
    for dataset in ("Movies", "Grocery"):
        for seed in (42, 43, 44):
            data, p_details, p_logits, _ = load_outputs(dataset, seed, "M2-P", device)
            idx = data.val_idx.cpu()
            labels = data.y.cpu()
            pred = p_logits.argmax(dim=-1).cpu()
            ce = torch.nn.functional.cross_entropy(p_logits[idx.to(device)], labels[idx].to(device), reduction="none").cpu()
            classes = list(range(int(data.num_classes)))
            p_class_f1 = class_f1(labels, pred, idx, classes)
            overall_correct = pred[idx] == labels[idx]
            for modality_index, modality in enumerate(("text", "visual")):
                h0 = p_details["stage12"][f"H0_{modality}"]
                u = p_details["retrieval"][:, modality_index]
                z = p_details[f"Z_{modality}"]
                ratio = u.norm(dim=-1) / (h0.norm(dim=-1) + 1e-12)
                compatibility = torch.nn.functional.cosine_similarity(h0, u, dim=-1, eps=1e-12)
                displacement = 1.0 - torch.nn.functional.cosine_similarity(h0, z, dim=-1, eps=1e-12)
                metric_map = {"retrieval_ratio": ratio, "compatibility_cosine": compatibility,
                              "structural_displacement": displacement}
                for metric, values in metric_map.items():
                    summary_rows.append({"dataset": dataset, "seed": seed, "variant": "M2-P", "modality": modality,
                                         "metric": metric, "split": "validation", **distribution(values[idx.to(device)])})
                    for group, mask in (("correct", overall_correct), ("incorrect", ~overall_correct)):
                        selected = idx[mask]
                        correctness_rows.append({"dataset": dataset, "seed": seed, "modality": modality,
                                                 "correctness": group, "metric": metric, "nodes": int(mask.sum()),
                                                 **distribution(values[selected.to(device)])})
                    corr_rows.append({"dataset": dataset, "seed": seed, "modality": modality,
                                      "metric": metric, "target": "per_node_cross_entropy", "spearman_rho": spearman(values[idx.to(device)], ce)})
            p_ratio = {m: (p_details["retrieval"][:, i].norm(dim=-1) /
                           (p_details["stage12"][f"H0_{m}"].norm(dim=-1) + 1e-12)) for i, m in enumerate(("text", "visual"))}
            p_displacement = {m: 1.0 - torch.nn.functional.cosine_similarity(
                p_details["stage12"][f"H0_{m}"], p_details[f"Z_{m}"], dim=-1, eps=1e-12) for m in ("text", "visual")}
            p_class_stats = {}
            for c in classes:
                class_idx = idx[labels[idx] == c]
                p_class_stats[c] = {}
                for modality in ("text", "visual"):
                    p_class_stats[c][modality] = {
                        "ratio": distribution(p_ratio[modality][class_idx.to(device)])["mean"],
                        "displacement": distribution(p_displacement[modality][class_idx.to(device)])["mean"],
                    }
            del p_details, p_logits
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # Historical class-wise F1 is computed from the existing frozen M2-S/P checkpoints.
            s_data, _, s_logits, _ = load_outputs(dataset, seed, "M2-S", device)
            if s_data.num_nodes != data.num_nodes:
                raise AssertionError("M2-S/M2-P NC data mismatch")
            s_f1 = class_f1(labels, s_logits.argmax(dim=-1).cpu(), idx, classes)
            for c in classes:
                class_rows.append({"dataset": dataset, "seed": seed, "class": c,
                                   "validation_support": int((labels[idx] == c).sum()),
                                   "M2_S_f1": s_f1[c], "M2_P_f1": p_class_f1[c],
                                   "delta_M2P_minus_M2S_f1": p_class_f1[c] - s_f1[c],
                                   "M2_P_text_retrieval_ratio": p_class_stats[c]["text"]["ratio"],
                                   "M2_P_visual_retrieval_ratio": p_class_stats[c]["visual"]["ratio"],
                                   "M2_P_text_displacement": p_class_stats[c]["text"]["displacement"],
                                   "M2_P_visual_displacement": p_class_stats[c]["visual"]["displacement"]})
            del data, s_data, s_logits
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"precheck completed: {dataset} seed {seed}", flush=True)
    def write(name, rows):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    write("precheck_summary.csv", summary_rows)
    write("precheck_correctness.csv", correctness_rows)
    write("precheck_classwise.csv", class_rows)
    write("precheck_correlations.csv", corr_rows)
    (output_dir / "precheck_metadata.json").write_text(json.dumps({
        "source": "six existing formal M2-P validation checkpoints; M2-S used only for historical per-class F1",
        "test_evaluation": False, "training": False, "descriptive_correlations_not_causal": True,
        "device": str(device), "output_files": ["precheck_summary.csv", "precheck_correctness.csv", "precheck_classwise.csv", "precheck_correlations.csv"]
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
