from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.problem_validation.common import (  # noqa: E402
    DATASETS,
    RUN_SEEDS,
    json_safe,
    load_dataset,
    make_physical_graph,
    write_json,
)
from src.analysis.problem_validation.p02_context_probe import (  # noqa: E402
    build_context_readout_features,
    train_context_readout,
)
from src.analysis.problem_validation.p02_diagnostics import (  # noqa: E402
    aggregate_with_edge_states,
    compatibility_masks,
    compute_edge_utilities,
    conflict_delta_rows,
    context_rollout,
    exact_decomposition_audit,
    make_all_node_features,
    mechanism_diagnostics,
    norm_profile,
    precompute_edge_states,
)
from src.analysis.problem_validation.p01plus_statistics import quintile_partition_indices  # noqa: E402
from src.analysis.problem_validation.p02_models import P02Model, VARIANTS, Variant
from src.analysis.problem_validation.p02_training import (  # noqa: E402
    IncomingCSR,
    load_fixed_split,
    load_checkpoint_model,
    phase_metrics,
    safe_labels,
    stable_json_fingerprint,
    tensor_sha256,
    train_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P0.2 scalar-vs-feature interaction analysis.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/analysis/problem_validation_p02.yaml")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true", help="Prescribed Movies seed42 smoke; full configured protocol, no shortcuts.")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def _path(template: str, *, dataset: str, seed: int) -> Path:
    return ROOT / template.format(dataset=dataset, seed=seed)


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def _torch_hash(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_p01_reference(path: Path, split: dict[str, Any], graph: IncomingCSR, seed: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required P0.1 edge artifact is missing: {path}; stopping")
    reference = torch.load(path, map_location="cpu", weights_only=False)
    if int(reference.get("run_seed", -1)) != seed:
        raise RuntimeError(f"P0.1 artifact seed mismatch: {path}")
    target = reference["target_node"].cpu().long()
    neighbor = reference["neighbor_node"].cpu().long()
    expected_target = split["sampled_edge_target_node"].cpu().long()
    expected_neighbor = split["sampled_edge_neighbor_node"].cpu().long()
    if not torch.equal(target, expected_target) or not torch.equal(neighbor, expected_neighbor):
        raise RuntimeError(f"P0.1 ordered relation population mismatch: {path}; stopping")
    if not torch.equal(reference["analysis_target_nodes"].cpu().long(), split["analysis_target_nodes"]):
        raise RuntimeError(f"P0.1 analysis target population mismatch: {path}; stopping")
    degree = graph.degree.cpu()[target]
    if not torch.equal(reference["target_degree"].cpu().long(), degree):
        raise RuntimeError(f"P0.1 full physical degree denominator mismatch: {path}; stopping")
    for key in (
        "probe_sim_text", "probe_sim_visual", "utility_ce_text", "utility_ce_visual", "utility_ce_joint"
    ):
        if key not in reference or reference[key].numel() != target.numel():
            raise RuntimeError(f"P0.1 reference field {key} is missing or misaligned: {path}")
    return reference


def preflight_all_sources(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Verify every required split/embedding/reference before the first fit."""
    a = config["analysis"]
    entries: dict[str, Any] = {}
    for dataset in a["datasets"]:
        print(f"Preflight fixed sources: {dataset}", flush=True)
        _, data = load_dataset(dataset)
        split_path = _path(a["split_cache"], dataset=dataset, seed=0)
        split = load_fixed_split(data, dataset, split_path)
        physical = make_physical_graph(data.edge_index, data.num_nodes)
        graph = IncomingCSR(physical, data.num_nodes, torch.device("cpu"))
        split_pairs = torch.stack(
            (split["sampled_edge_target_node"], split["sampled_edge_neighbor_node"]), dim=1
        )
        dataset_entries: dict[str, Any] = {
            "split_cache_path": str(split_path.relative_to(ROOT)),
            "split_cache_sha256": split["split_file_sha256"],
            "source_split_hashes": split["source_split_hashes"],
            "probe_train_sha256": split["probe_train_sha256"],
            "probe_calib_sha256": split["probe_calib_sha256"],
            "original_val_sha256": split["original_val_sha256"],
            "original_test_index_sha256": split["original_test_index_sha256"],
            "split_audit": split["split_audit"],
            "physical_graph_num_nodes": int(data.num_nodes),
            "physical_graph_directed_edge_count": int(physical.size(1)),
            "physical_graph_edge_sha256": _torch_hash(physical),
            "sampled_relation_count": int(split_pairs.size(0)),
            "seeds": {},
        }
        for seed in a["run_seeds"]:
            embedding_path = _path(a["frozen_embedding"], dataset=dataset, seed=int(seed))
            reference_path = _path(a["p01_reference"], dataset=dataset, seed=int(seed))
            metrics_path = _path(a["p01_metrics"], dataset=dataset, seed=int(seed))
            if not embedding_path.exists():
                raise FileNotFoundError(f"required frozen P0.0 embedding missing: {embedding_path}; STOP without retraining")
            if not metrics_path.exists():
                raise FileNotFoundError(f"required P0.1 source metrics missing: {metrics_path}; STOP")
            embedding = torch.load(embedding_path, map_location="cpu", weights_only=False)
            if int(embedding.get("run_seed", -1)) != int(seed):
                raise RuntimeError(f"frozen P0.0 embedding seed metadata mismatch: {embedding_path}")
            for key in ("H_text", "H_visual"):
                value = embedding[key]
                if value.shape != (data.num_nodes, int(a["semantic_dim"])):
                    raise RuntimeError(f"{embedding_path}: {key} has unexpected shape {tuple(value.shape)}")
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise RuntimeError(f"{embedding_path}: {key} must be finite float32")
            reference = load_p01_reference(reference_path, split, graph, int(seed))
            p01_metrics = json.loads(metrics_path.read_text())
            p00_metrics_path = embedding_path.with_name("p00_metrics.json")
            if not p00_metrics_path.exists():
                raise FileNotFoundError(f"P0.0 semantic-probe metadata missing: {p00_metrics_path}; STOP")
            p00_metrics = json.loads(p00_metrics_path.read_text())
            if int(p01_metrics.get("run_seed", -1)) != int(seed) or int(p00_metrics.get("run_seed", -1)) != int(seed):
                raise RuntimeError(f"P0.0/P0.1 metrics seed mismatch for {dataset} seed{seed}")
            for field, expected_size in (("original_train_size", split["split_audit"]["original_train_size"]),
                                         ("original_val_size", split["split_audit"]["original_val_size"]),
                                         ("original_test_size", split["split_audit"]["original_test_index_count"])):
                if int(p01_metrics.get(field, -1)) != int(expected_size):
                    raise RuntimeError(f"P0.1 {field} mismatch for {dataset} seed{seed}")
            p00_split = p00_metrics.get("split_metadata", {})
            if int(p00_split.get("probe_train_size", -1)) != split["split_audit"]["probe_train_size"] or int(p00_split.get("probe_calib_size", -1)) != split["split_audit"]["probe_calib_size"] or int(p00_split.get("original_val_size", -1)) != split["split_audit"]["original_val_size"]:
                raise RuntimeError(f"P0.0 probe/original-val split metadata mismatch for {dataset} seed{seed}")
            if p01_metrics.get("test_set_evaluated", False) or p01_metrics.get("test_labels_indexed", False) or p00_metrics.get("test_set_evaluated", False):
                raise RuntimeError(f"P0.0/P0.1 metrics flags indicate test access for {dataset} seed{seed}")
            dataset_entries["seeds"][str(seed)] = {
                "frozen_embedding_path": str(embedding_path.relative_to(ROOT)),
                "frozen_embedding_sha256": tensor_sha256(embedding_path),
                "h_text_sha256": _torch_hash(embedding["H_text"]),
                "h_visual_sha256": _torch_hash(embedding["H_visual"]),
                "run_seed_metadata": int(embedding["run_seed"]),
                "h_text_shape": list(embedding["H_text"].shape),
                "h_visual_shape": list(embedding["H_visual"].shape),
                "p01_reference_path": str(reference_path.relative_to(ROOT)),
                "p01_reference_sha256": tensor_sha256(reference_path),
                "ordered_population_sha256": _torch_hash(split_pairs),
                "p01_metrics_sha256": tensor_sha256(metrics_path),
                "p00_metrics_sha256": tensor_sha256(p00_metrics_path),
                "p01_population_count": int(reference["target_node"].numel()),
                "test_evaluation": False,
                "test_labels_accessed": False,
            }
        entries[dataset] = dataset_entries
        del data, physical, graph, embedding, reference
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "source_branch": "exp/problem_validation_p01plus",
        "source_sha": _git_value(["merge-base", "HEAD", "origin/exp/problem_validation_p01plus"]),
        "analysis_branch": _git_value(["branch", "--show-current"]),
        "initial_sha": _git_value(["merge-base", "HEAD", "origin/exp/problem_validation_p01plus"]),
        "all_frozen_sources_verified_before_training": True,
        "test_metrics_or_labels_accessed": False,
        "datasets": entries,
    }


def _metric_on_val(data: Any, logits: torch.Tensor, split: dict[str, Any]) -> dict[str, float]:
    from src.analysis.problem_validation.p02_training import _metric_values

    permitted = torch.cat((split["probe_train_idx"], split["probe_calib_idx"], split["original_val_idx"]))
    labels = safe_labels(data, split["original_val_idx"], permitted).to(logits.device)
    return _metric_values(logits, labels, int(data.num_classes))


def _shuffle_feature_gate(
    states: dict[str, dict[str, torch.Tensor]],
    *,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Shuffle g within full-graph probe-similarity quintiles, preserving each marginal exactly."""
    shuffled = {
        name: {key: value.clone() for key, value in values.items()}
        for name, values in states.items()
    }
    for modality_index, name in enumerate(("text", "visual")):
        state = shuffled[name]
        similarity = states[name]["probe_similarity"].numpy()
        groups = quintile_partition_indices(similarity)
        rng = np.random.default_rng(seed + modality_index * 10_000)
        original_g = states[name]["g"]
        new_g = original_g.clone()
        for indices in groups:
            perm = rng.permutation(indices)
            new_g[torch.as_tensor(indices, dtype=torch.long)] = original_g[torch.as_tensor(perm, dtype=torch.long)]
            lhs = original_g[torch.as_tensor(indices, dtype=torch.long)].sort(dim=0).values
            rhs = new_g[torch.as_tensor(indices, dtype=torch.long)].sort(dim=0).values
            if not torch.equal(lhs, rhs):
                raise AssertionError("function shuffle did not preserve the within-quintile gate multiset")
        state["g"] = new_g
    return shuffled


def _write_run_diagnostics(
    dataset: str,
    seed: int,
    data: Any,
    split: dict[str, Any],
    graph: IncomingCSR,
    h_text: torch.Tensor,
    h_visual: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    fingerprint: str,
    device: torch.device,
    seed_dir: Path,
) -> None:
    a = config["analysis"]
    edge_by_variant: dict[str, dict[str, Any]] = {}
    for variant_index, variant in enumerate(a["variants"]):
        variant = str(variant)
        variant_dir = seed_dir / variant
        checkpoint_path, metrics_path = variant_dir / "checkpoint.pt", variant_dir / "metrics.json"
        if not checkpoint_path.exists() or not metrics_path.exists():
            model, metrics = train_variant(
                data,
                variant,  # type: ignore[arg-type]
                seed,
                split,
                graph,
                h_text,
                h_visual,
                config,
                device=device,
                checkpoint_path=checkpoint_path,
                config_fingerprint=fingerprint,
            )
            write_json(metrics_path, json_safe(metrics))
        else:
            model, checkpoint = load_checkpoint_model(
                checkpoint_path,
                data,
                variant,  # type: ignore[arg-type]
                config,
                device,
                fingerprint,
            )
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("config_fingerprint") != fingerprint or metrics.get("variant") != variant:
                raise RuntimeError(f"stored P0.2 metrics/config mismatch in {metrics_path}")
        print(f"  {dataset} seed{seed} {variant}: one-hop inference and diagnostics", flush=True)
        states = precompute_edge_states(model, graph, h_text, h_visual, config)
        all_features = make_all_node_features(
            model,
            graph,
            h_text,
            h_visual,
            states,
            chunk_size=int(a["inference_edge_chunk_size"]),
        )
        if not torch.isfinite(all_features).all():
            raise FloatingPointError(f"{dataset} seed{seed} {variant}: non-finite one-hop features")
        val_logits = model.classifier(all_features[split["original_val_idx"].to(device)])
        inferred_val = _metric_on_val(data, val_logits, split)
        trained_val = metrics["heldout_original_val"]
        if abs(float(inferred_val["acc"]) - float(trained_val["acc"])) > 1e-12:
            raise AssertionError(f"cached/full inference does not match trained heldout-val metrics: {variant}")
        if abs(float(inferred_val["ce"]) - float(trained_val["ce"])) > 1e-4:
            raise AssertionError(f"cached/full inference CE mismatch: {variant}")
        utility = compute_edge_utilities(data, model, graph, all_features, states, reference, split)
        decomposition = exact_decomposition_audit(
            model,
            all_features,
            h_text,
            h_visual,
            utility["target_node"],
            utility["neighbor_node"],
            utility["target_degree"],
            {**states, "physical_edge_row": utility["physical_edge_row"]},
            seed=int(a["edge_contribution_audit_seed"]),
            sample_count=int(a["edge_contribution_audit_edges"]),
        )
        utility.update({
            "probe_sim_text": reference["probe_sim_text"].cpu().float(),
            "probe_sim_visual": reference["probe_sim_visual"].cpu().float(),
            "p01_reference_utility_ce_text": reference["utility_ce_text"].cpu().float(),
            "p01_reference_utility_ce_visual": reference["utility_ce_visual"].cpu().float(),
            "p01_reference_utility_ce_joint": reference["utility_ce_joint"].cpu().float(),
            "p01_reference_utility_margin_text": reference["utility_margin_text"].cpu().float(),
            "p01_reference_utility_margin_visual": reference["utility_margin_visual"].cpu().float(),
            "p01_reference_utility_margin_joint": reference["utility_margin_joint"].cpu().float(),
            "compatibility_quintile_text": torch.as_tensor(compatibility_masks(reference)["text"]["quintile"]),
            "compatibility_quintile_visual": torch.as_tensor(compatibility_masks(reference)["visual"]["quintile"]),
            "decomposition_audit": decomposition,
            "test_evaluation": False,
            "test_labels_accessed": False,
        })
        torch.save(utility, variant_dir / "edge_diagnostics.pt")
        metrics["decomposition_audit"] = decomposition
        metrics["p01_uniform_probe_heldout_val"] = json.loads(
            _path(a["p01_metrics"], dataset=dataset, seed=seed).read_text()
        )["p01_message_probe"]["heldout_analysis_val"]
        metrics["v0_structure_matches_p01_uniform"] = variant == "uniform"
        metrics["one_hop_reconstructed_val_metrics"] = inferred_val

        contexts, stability = context_rollout(
            graph,
            h_text,
            h_visual,
            states,
            max_order=int(a["context_max_order"]),
            chunk_size=int(a["inference_edge_chunk_size"]),
        )
        write_json(variant_dir / "context_stability.json", json_safe(stability))
        context_rows: dict[str, Any] = {}
        for readout_index, readout in enumerate(("context_only", "full_bank")):
            context_features = build_context_readout_features(h_text, h_visual, contexts, readout)
            context_metrics = train_context_readout(
                data,
                context_features,
                split,
                config,
                seed=int(seed + int(a["context_probe_seed_offset"]) + variant_index * 10 + readout_index),
            )
            context_rows[readout] = context_metrics
        write_json(variant_dir / "context_probe_metrics.json", json_safe(context_rows))
        metrics["context_probe_metrics"] = context_rows

        if variant == "conditional_feature":
            identity_features = make_all_node_features(
                model,
                graph,
                h_text,
                h_visual,
                states,
                identity_function=True,
                chunk_size=int(a["inference_edge_chunk_size"]),
            )
            identity_logits = model.classifier(identity_features[split["original_val_idx"].to(device)])
            identity_metrics = _metric_on_val(data, identity_logits, split)
            control_full = inferred_val
            identity = {
                "full_v3": control_full,
                "identity_function": identity_metrics,
                "delta_full_minus_identity": {
                    key: float(control_full[key] - identity_metrics[key]) for key in ("acc", "macro_f1", "ce")
                },
                "relation_encoder_and_scalar_gate_unchanged": True,
                "test_evaluation": False,
            }
            write_json(variant_dir / "function_ablation.json", json_safe(identity))

            shuffle_rows: list[dict[str, Any]] = []
            for rep in range(int(a["function_shuffle_repeats"])):
                shuffle_seed = int(a["shuffle_seed_base"]) + rep
                shuffled_states = _shuffle_feature_gate(states, seed=shuffle_seed)
                shuffled_features = make_all_node_features(
                    model,
                    graph,
                    h_text,
                    h_visual,
                    shuffled_states,
                    chunk_size=int(a["inference_edge_chunk_size"]),
                )
                shuffled_logits = model.classifier(shuffled_features[split["original_val_idx"].to(device)])
                shuffle_rows.append({
                    "repetition": rep,
                    "shuffle_seed": shuffle_seed,
                    "heldout_original_val": _metric_on_val(data, shuffled_logits, split),
                })
            means: dict[str, float] = {}
            stds: dict[str, float] = {}
            for key in ("acc", "macro_f1", "ce"):
                values = np.asarray([r["heldout_original_val"][key] for r in shuffle_rows], dtype=np.float64)
                means[key] = float(values.mean())
                stds[key] = float(values.std(ddof=0))
            shuffle = {
                "full_v3": control_full,
                "shuffle_repetitions": shuffle_rows,
                "shuffle_mean": means,
                "shuffle_population_std": stds,
                "delta_full_minus_shuffle_mean": {key: float(control_full[key] - means[key]) for key in means},
                "shuffle_within_full_graph_probe_similarity_quintile": True,
                "scalar_gate_edge_specific_and_unchanged": True,
                "test_evaluation": False,
            }
            write_json(variant_dir / "function_shuffle.json", json_safe(shuffle))

            mechanism_arrays, mechanism_summary = mechanism_diagnostics(
                model, states, reference, graph, split
            )
            torch.save(mechanism_arrays, variant_dir / "mechanism_diagnostics.pt")
            write_json(variant_dir / "mechanism_summary.json", json_safe(mechanism_summary))
            metrics["function_identity_control"] = identity
            metrics["function_shuffle_control"] = shuffle
            metrics["mechanism_summary"] = mechanism_summary

        write_json(metrics_path, json_safe(metrics))
        edge_by_variant[variant] = utility
        if device.type == "cuda":
            del all_features, states, contexts
            torch.cuda.empty_cache()

    p02_v2 = edge_by_variant["learned_scalar"]
    p02_v3 = edge_by_variant["conditional_feature"]
    conflict_rows = conflict_delta_rows(
        reference,
        p02_v2,
        p02_v3,
        split,
        bootstrap_replicates=int(a["node_bootstrap_replicates"]),
        bootstrap_seed=int(a["node_bootstrap_seed"]),
    )
    for row in conflict_rows:
        row.update({"dataset": dataset, "seed": seed})
    write_json(seed_dir / "conflict_subset_rows.json", json_safe({"rows": conflict_rows}))
    write_json(seed_dir / "run_complete.json", {
        "dataset": dataset,
        "seed": seed,
        "config_fingerprint": fingerprint,
        "all_variants_complete": True,
        "all_edge_decomposition_audits_pass": True,
        "context_rollout_finite": True,
        "test_evaluation": False,
        "test_labels_accessed": False,
    })


def _requested_values(args: argparse.Namespace, config: dict[str, Any]) -> tuple[list[str], list[int], list[str], str]:
    a = config["analysis"]
    datasets = args.datasets or list(a["datasets"])
    seeds = args.seeds or [int(x) for x in a["run_seeds"]]
    variants = args.variants or list(a["variants"])
    device = args.device or str(a["device"])
    if any(ds not in a["datasets"] for ds in datasets):
        raise SystemExit("requested dataset is not present in the P0.2 source-of-truth config")
    if any(seed not in a["run_seeds"] for seed in seeds):
        raise SystemExit("requested seed is not present in the P0.2 source-of-truth config")
    if set(variants) != set(a["variants"]):
        raise SystemExit("P0.2 requires all four variants in each run")
    if args.smoke and (datasets != ["Movies"] or seeds != [42]):
        raise SystemExit("--smoke is reserved for Movies seed 42 and still uses the full configured training protocol")
    return datasets, seeds, variants, device


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    a = config["analysis"]
    if bool(a.get("evaluate_test", True)):
        raise RuntimeError("P0.2 config must keep evaluate_test: false")
    datasets, seeds, variants, device_name = _requested_values(args, config)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)
    config_fingerprint = stable_json_fingerprint(config)
    output_root = ROOT / a["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    results_root = ROOT / a["results_root"]
    results_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Source={_git_value(['rev-parse','HEAD'])}; analysis branch={_git_value(['branch','--show-current'])}; "
        f"datasets={datasets}; seeds={seeds}; variants={variants}; device={device}; smoke={args.smoke} (no shortcuts)",
        flush=True,
    )
    preflight_path = output_root / "source_artifact_manifest.json"
    manifest = preflight_all_sources(config, device)
    manifest["config_fingerprint"] = config_fingerprint
    manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest["config_path"] = str(config_path.relative_to(ROOT))
    write_json(preflight_path, json_safe(manifest))
    write_json(results_root / "source_artifact_manifest.json", json_safe(manifest))

    for dataset in datasets:
        _, data = load_dataset(dataset)
        split_path = _path(a["split_cache"], dataset=dataset, seed=0)
        split = load_fixed_split(data, dataset, split_path)
        physical = make_physical_graph(data.edge_index, data.num_nodes)
        graph = IncomingCSR(physical, data.num_nodes, device)
        h_by_seed: dict[int, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
        for seed in seeds:
            emb_path = _path(a["frozen_embedding"], dataset=dataset, seed=seed)
            ref_path = _path(a["p01_reference"], dataset=dataset, seed=seed)
            emb = torch.load(emb_path, map_location="cpu", weights_only=False)
            expected = manifest["datasets"][dataset]["seeds"][str(seed)]["frozen_embedding_sha256"]
            if tensor_sha256(emb_path) != expected:
                raise RuntimeError(f"frozen P0.0 embedding changed after preflight: {emb_path}")
            ref = load_p01_reference(ref_path, split, graph, seed)
            h_text = emb["H_text"].detach().to(device=device, dtype=torch.float32)
            h_visual = emb["H_visual"].detach().to(device=device, dtype=torch.float32)
            h_by_seed[seed] = (h_text, h_visual, ref)
        for seed in seeds:
            print(f"\n=== P0.2 {dataset} seed{seed}: loading existing P0.0 embeddings only ===", flush=True)
            h_text, h_visual, reference = h_by_seed[seed]
            seed_dir = output_root / dataset / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            run_metadata = {
                "dataset": dataset,
                "run_seed": seed,
                "source_branch": manifest["source_branch"],
                "source_sha": manifest["source_sha"],
                "analysis_code_base_sha": _git_value(["rev-parse", "HEAD"]),
                "analysis_branch": _git_value(["branch", "--show-current"]),
                "config_fingerprint": config_fingerprint,
                "frozen_embedding_sha256": manifest["datasets"][dataset]["seeds"][str(seed)]["frozen_embedding_sha256"],
                "p01_reference_sha256": manifest["datasets"][dataset]["seeds"][str(seed)]["p01_reference_sha256"],
                "split_cache_sha256": split["split_file_sha256"],
                "physical_graph_edge_sha256": manifest["datasets"][dataset]["physical_graph_edge_sha256"],
                "split_audit": split["split_audit"],
                "sampled_relation_count": int(reference["target_node"].numel()),
                "message_convention": "source_node_j -> target_node_i; incoming mean uses original physical degree",
                "physical_graph_rule": "make_physical_graph: undirected, self-loops removed, duplicates coalesced",
                "test_evaluation": False,
                "test_labels_accessed": False,
            }
            write_json(seed_dir / "run_metadata.json", json_safe(run_metadata))
            _write_run_diagnostics(
                dataset,
                seed,
                data,
                split,
                graph,
                h_text,
                h_visual,
                reference,
                config,
                config_fingerprint,
                device,
                seed_dir,
            )
            del h_text, h_visual, reference
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del data, physical, graph, h_by_seed
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_summary:
        from scripts.summarize_problem_validation_p02 import main as summarize_main

        summarize_main(config_path=config_path, allow_partial=True)
    print("P0.2 requested run subset complete.", flush=True)


if __name__ == "__main__":
    main()
