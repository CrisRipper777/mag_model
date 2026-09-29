from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts import summarize_problem_validation_p01plus as summarize
from src.analysis.problem_validation.p01plus_statistics import (
    node_bootstrap_metric_samples,
    quintile_partition_indices,
    summarize_utility_by_similarity,
)


def test_base_rate_matches_manual() -> None:
    similarity = np.arange(5, dtype=float)
    utility = np.array([1.0, -2.0, 0.0, 3.0, -1.0])
    result = summarize_utility_by_similarity(similarity, utility)
    assert result["overall_beneficial_rate"] == 2 / 5
    assert result["overall_harmful_rate"] == 2 / 5
    assert result["overall_zero_rate"] == 1 / 5
    assert sum(result[key] for key in ("overall_beneficial_rate", "overall_harmful_rate", "overall_zero_rate")) == 1.0


def test_quintile_partition_covers_all_edges() -> None:
    # Equal similarities cross boundaries; stable input order makes assignment deterministic.
    similarity = np.array([2, 1, 1, 3, 2, 0, 4, 4, 1, 3, 5, 2, 0], dtype=float)
    groups = quintile_partition_indices(similarity)
    assert [len(group) for group in groups] == [3, 3, 3, 2, 2]
    flattened = np.concatenate(groups)
    assert np.array_equal(np.sort(flattened), np.arange(similarity.size))
    assert np.array_equal(flattened, np.argsort(similarity, kind="mergesort"))


def test_lift_is_conditional_minus_base() -> None:
    similarity = np.arange(5, dtype=float)
    utility = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    result = summarize_utility_by_similarity(similarity, utility)
    assert result["overall_beneficial_rate"] == 3 / 5
    assert result["q1_beneficial_rate"] == 1.0
    assert result["q1_beneficial_lift"] == result["q1_beneficial_rate"] - result["overall_beneficial_rate"]
    assert result["q5_harmful_lift"] == result["q5_harmful_rate"] - result["overall_harmful_rate"]


def test_node_bootstrap_lift_recomputed_per_replicate() -> None:
    target_node = np.array([10, 10, 20, 20, 30, 30, 30, 40, 40, 40], dtype=np.int64)
    similarity = np.array([0.1, 0.2, 0.1, 0.8, 0.2, 0.3, 0.9, 0.4, 0.7, 1.0])
    utility = np.array([1, -1, 1, -1, -1, 1, 1, -1, 1, -1], dtype=float)
    samples = node_bootstrap_metric_samples(
        similarity,
        {"ce": utility},
        target_node,
        np.array([10, 20, 30, 40]),
        replicates=80,
        seed=42,
    )["ce"]
    np.testing.assert_allclose(
        samples["q1_beneficial_lift"],
        samples["q1_beneficial_rate"] - samples["overall_beneficial_rate"],
        equal_nan=True,
    )


def test_no_test_artifact_access(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "Movies" / "seed42"
    run_dir.mkdir(parents=True)
    payload = {
        "dataset": "Movies",
        "run_seed": 42,
        "sampled_relation_count": 1,
        "statistics": {"text_visual": {"robust_disagreement_rate": 0.25}},
        "target_node": torch.tensor([1]),
        "analysis_target_nodes": torch.tensor([1]),
        "raw_sim_text": torch.tensor([0.1]),
        "raw_sim_visual": torch.tensor([0.1]),
        "probe_sim_text": torch.tensor([0.1]),
        "probe_sim_visual": torch.tensor([0.1]),
        "utility_ce_text": torch.tensor([1.0]),
        "utility_ce_visual": torch.tensor([-1.0]),
        "utility_margin_text": torch.tensor([1.0]),
        "utility_margin_visual": torch.tensor([-1.0]),
    }
    torch.save(payload, run_dir / "edge_analysis.pt")
    # A deliberately invalid sentinel proves the loader never opens unrelated files.
    (run_dir / "test_metrics.json").write_text("must not be read", encoding="utf-8")

    real_load = torch.load
    loaded_names: list[str] = []

    def checked_load(path, *args, **kwargs):
        loaded_names.append(Path(path).name)
        assert Path(path).name == "edge_analysis.pt"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(summarize.torch, "load", checked_load)
    result = summarize.load_existing_edge_artifacts(tmp_path, datasets=("Movies",), seeds=(42,))
    assert list(result) == [("Movies", 42)]
    assert loaded_names == ["edge_analysis.pt"]


def test_plot_uses_three_seed_text_visual_profiles(tmp_path: Path) -> None:
    rows = []
    for modality in ("Text", "Visual"):
        for seed in (42, 43, 44):
            row = {"dataset": "Movies", "modality": modality, "similarity_space": "probe", "seed": seed}
            row["overall_beneficial_rate"] = 0.6
            row["overall_harmful_rate"] = 0.4
            for q in range(1, 6):
                row[f"q{q}_beneficial_rate"] = 0.5 + q * 0.02
                row[f"q{q}_harmful_rate"] = 0.5 - q * 0.02
            rows.append(row)
    path = summarize.make_dataset_plot("Movies", rows, tmp_path)
    assert path.is_file() and path.stat().st_size > 0
