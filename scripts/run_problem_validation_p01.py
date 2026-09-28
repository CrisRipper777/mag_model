from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.problem_validation.common import (  # noqa: E402
    DATASETS,
    OUTPUT_ROOT,
    RUN_SEEDS,
    audit_split,
    json_safe,
    load_dataset,
    make_physical_graph,
    make_probe_split,
    sample_analysis_population,
    write_json,
)
from src.analysis.problem_validation.edge_utility import (  # noqa: E402
    analyze_and_save_edges,
    train_message_probe,
)
from src.analysis.problem_validation.semantic_probe import train_semantic_probe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P0.0/P0.1 problem-validation probes.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(RUN_SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run the requested small validation case; no protocol shortcuts are applied.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_seeds = sorted(set(args.seeds) - set(RUN_SEEDS))
    if invalid_seeds:
        raise SystemExit(f"training seeds must be selected from {RUN_SEEDS}; got {invalid_seeds}")
    if args.smoke and (args.datasets != ["Movies"] or args.seeds != [42]):
        raise SystemExit("--smoke is reserved for the prescribed Movies seed-42 validation run")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    print(f"Experiment invocation: datasets={args.datasets}; model run seeds={args.seeds}; data split seed=42")
    print(f"Device={device}; output={args.output_root}; bootstrap replicates={args.bootstrap_replicates}")

    for dataset in args.datasets:
        print(f"\n=== Dataset {dataset}: loading one fixed NC split and shared edge population ===", flush=True)
        cfg, data = load_dataset(dataset)
        if data.num_classes is None or int(cfg.dataset.num_classes) != int(data.num_classes):
            raise RuntimeError(f"{dataset}: configured class count disagrees with loader labels")
        split = make_probe_split(data, dataset, args.output_root / "splits")
        physical_edge_index = make_physical_graph(data.edge_index, data.num_nodes)
        population = sample_analysis_population(data, physical_edge_index, seed=42)
        split_audit = audit_split(data, split["probe_train_idx"], split["probe_calib_idx"], population["target_nodes"])
        if population["target_node"].numel() and bool((population["target_node"] == population["neighbor_node"]).any()):
            raise AssertionError(f"{dataset}: sampled analysis relation contains a self-loop")
        if population["target_node"].numel():
            directed = set(zip(physical_edge_index[0].tolist(), physical_edge_index[1].tolist(), strict=True))
            for target, neighbor in zip(population["target_node"].tolist(), population["neighbor_node"].tolist(), strict=True):
                if (neighbor, target) not in directed:
                    raise AssertionError(f"{dataset}: analysis relation is not present in physical graph")
        cache = dict(split)
        cache.update({
            "analysis_target_nodes": population["target_nodes"],
            "sampled_edge_target_node": population["target_node"],
            "sampled_edge_neighbor_node": population["neighbor_node"],
            "split_audit": split_audit,
            "original_train_size": int(data.train_idx.numel()),
            "original_val_size": int(data.val_idx.numel()),
            "original_test_size": int(data.test_idx.numel()),
            "physical_edge_count_directed": int(physical_edge_index.size(1)),
            "sampled_relation_count": int(population["target_node"].numel()),
            "test_labels_indexed": False,
        })
        torch.save(cache, args.output_root / "splits" / f"{dataset}.pt")
        print(
            f"Split audit: train={split_audit['original_train_size']} probe={split_audit['probe_train_size']}+"
            f"{split_audit['probe_calib_size']} val={split_audit['original_val_size']} "
            f"analysis_targets={population['target_nodes'].numel()} sampled_edges={population['target_node'].numel()}",
            flush=True,
        )

        # One dataset invocation contains the three independent model-init seeds.
        population_signature = torch.stack((population["target_node"], population["neighbor_node"]), dim=1)
        for run_seed in args.seeds:
            run_dir = args.output_root / dataset / f"seed{run_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"--- {dataset}: model run seed {run_seed} ---", flush=True)
            p00 = train_semantic_probe(
                data,
                run_seed,
                split["probe_train_idx"],
                split["probe_calib_idx"],
                run_dir,
                device,
            )
            embeddings = torch.load(run_dir / "semantic_embeddings.pt", map_location="cpu", weights_only=False)
            p01_train, message_model, features, full_logits, isolated_node_mask = train_message_probe(
                data,
                run_seed,
                embeddings["H_text"],
                embeddings["H_visual"],
                physical_edge_index,
                split["probe_train_idx"],
                split["probe_calib_idx"],
                run_dir,
                device,
            )
            edge_rows, p01_stats = analyze_and_save_edges(
                data,
                message_model,
                features,
                full_logits,
                embeddings["H_text"],
                embeddings["H_visual"],
                population,
                run_seed,
                run_dir,
                device,
                bootstrap_replicates=args.bootstrap_replicates,
                isolated_node_mask=isolated_node_mask,
            )
            if not torch.equal(population_signature, torch.stack((edge_rows["target_node"], edge_rows["neighbor_node"]), dim=1)):
                raise AssertionError("deterministic edge population changed within dataset run")
            chance = 1.0 / int(data.num_classes)
            calib_acc = float(p01_train["probe_calib"]["acc"])
            if calib_acc <= chance:
                raise RuntimeError(
                    f"STOP condition: {dataset} P0.1 probe_calib accuracy {calib_acc:.4f} "
                    f"did not exceed uniform chance {chance:.4f}"
                )
            merged = {
                "dataset": dataset,
                "run_seed": run_seed,
                "data_split_seed": 42,
                "split_audit": split_audit,
                "p00": p00,
                "p01_message_probe": p01_train,
                "p01_analysis": p01_stats,
                "decomposition_max_abs_error": edge_rows["decomposition_max_abs_error"],
                "sampled_relation_count": int(edge_rows["sampled_relation_count"]),
                "isolated_analysis_target_count": int(edge_rows["isolated_node_count"]),
                "test_set_evaluated": False,
                "test_labels_indexed": False,
            }
            write_json(run_dir / "metrics.json", json_safe(merged))
            print(
                f"Completed {dataset} seed{run_seed}: P00 best_epoch={p00['best_epoch']} "
                f"val_acc={p00['heldout_analysis_val']['acc']:.4f}; "
                f"P01 best_epoch={p01_train['best_epoch']} calib_acc={calib_acc:.4f}; "
                f"decomposition max error={edge_rows['decomposition_max_abs_error']:.3g}",
                flush=True,
            )
            del embeddings, p01_train, message_model, features, full_logits, isolated_node_mask, edge_rows, p01_stats, p00
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del data, cfg, physical_edge_index, population
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nRun summary generation starts.", flush=True)
    from scripts.summarize_problem_validation_p01 import main as summarize_main
    summarize_main(output_root=args.output_root)
    print("Problem-validation run complete.", flush=True)


if __name__ == "__main__":
    main()
