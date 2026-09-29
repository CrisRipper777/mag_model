# P0.1+ Supplementary Statistical Audit

This bundle re-analyzes only the 15 existing P0.1 `edge_analysis.pt` artifacts (five datasets × seeds 42, 43, and 44). It does not retrain either probe, regenerate relation populations, inspect test data, or evaluate a model.

## Files

- `p01plus_per_seed.csv`: 60 rows covering dataset × seed × modality × similarity space. CE is primary; margin profiles are retained as robustness results. Each row includes exact-sign base rates, quintile profiles, lifts, ratios, stratification ranges, five-bin Spearman values, robust Text/Visual disagreement copied from the original edge artifact, and run-specific node-bootstrap 95% intervals.
- `p01plus_cross_dataset.csv`: mean and population SD over the three seeds for each dataset × modality × similarity-space group. These seed-level summaries are separate from the node-bootstrap intervals.
- `plots/`: one two-panel plot per dataset. Panel a is Text and panel b is Visual. Lines show mean CE beneficial/harmful proportions across the three seeds; translucent bands show ± population SD; dashed lines show the corresponding overall sign base rates. Each figure is available as PNG, editable SVG, and vector PDF, with panel-alignment, PDF-text, and collision QA JSON alongside it.
- [`p01plus_report.md`](../../../docs/problem_validation/p01plus_report.md): scope, interpretation, full base-rate and quintile tables, raw/probe comparison, dataset-specific conclusions, and limits.
- [`figure_qa.md`](figure_qa.md): figure contract, panel-by-panel review, and QA results.

## Reproduction

From the repository root, after the P0.1 artifacts are present:

```bash
python scripts/summarize_problem_validation_p01plus.py \
  --source-commit c56f6589450ab0c7abf3960f0e7fb2996db383d6 \
  --analysis-code-commit bf1c93b02cdf5c57d41721bce0905ef8413d93ee \
  --bootstrap-replicates 1000 --bootstrap-seed 42
```

To regenerate only aggregate summaries, figures, and the report from the existing per-seed CSV without reading artifacts or rerunning bootstraps, add `--reuse-existing-csv`.

Quintiles are formed over the full sampled validation relation set within each dataset × seed × modality × similarity space, using stable mergesort and balanced `array_split` partitions. Utility sign uses strict comparisons (`U > 0`, `U < 0`, and exact `U == 0`); no epsilon is used. Bootstrap resamples target nodes (1,000 replicates, seed 42), carrying all sampled relations for each selected target. Lift is recomputed as conditional rate minus overall rate inside every replicate.

All rates describe the fixed P0.1 sampled validation relation population, with at most 64 sampled relations per target. They do not represent every edge in each graph. The original `edge_analysis.pt` files were hash-checked before and after the audit and were unchanged.

PNG files are 300-dpi report previews; editable PDF/SVG are the vector masters. No 600-dpi TIFF is included because no journal submission target was specified and the vector masters cover print-quality use.
