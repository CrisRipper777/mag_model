# P0.0 + P0.1 Result Bundle

This directory contains the compact, versioned result snapshot for the P0.0/P0.1 validation across Movies, Toys, Grocery, ele-fashion, and Reddit-S.

## Contents

- `summaries/`: aggregate CSVs for the P0.0 probe, P0.1 per-seed measurements, and cross-dataset summaries.
- `per_seed/<dataset>/seed<seed>/metrics.json`: full scalar metrics, bootstrap intervals, split audit, and exact-decomposition error for each of the 15 dataset/seed runs.
- `split_audit.csv`: one row per dataset/seed with original split sizes, probe split sizes, validation analysis population, relation sample count, overlap checks, and test-use flags.
- `plots/`: all 15 diagnostic PNGs (three per dataset).
- `p0_p1_report.md`: the complete experiment report, with links to the included figures.

## Seed protocol

The run used one runner invocation with all three seeds supplied together (`--seeds 42 43 44`) for the five datasets. These are three independent seeded model fits within that one run, with separate per-seed metrics; they were not launched as three one-seed runner runs. The data/probe split seed was fixed at 42. The reproduction command is in [`docs/problem_validation/README.md`](../../docs/problem_validation/README.md).

## Data scope

The original validation split was used only for held-out analysis. The test set was not evaluated, and test labels were not indexed. Test indices appear only in the disjointness audit. Split overlap counts are zero, and all saved decomposition errors are below `1e-5`.

Large tensor caches and embeddings remain in the locally generated, Git-ignored `outputs/problem_validation/` directory; this bundle preserves the aggregates and scalar records needed to inspect the findings without adding those large intermediates to Git. The complete interpretation and limitations are in [`p0_p1_report.md`](p0_p1_report.md).
