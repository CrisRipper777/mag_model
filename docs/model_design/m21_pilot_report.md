# M2.1 Pilot Report

## A. Source and protocol

Source branch `exp/provenance_evidence_m2` at `5ee0a5d5f095a5e43425188a6e8ea695d0b7c658`; experiment branch `exp/adaptive_evidence_assimilation_m21`.
NC only on Movies and Grocery. Original train was used for fitting, original validation for checkpoint selection and evaluation; test evaluation and LP were disabled.

## B. Frozen architecture contract

M2 Stage I/II and Stage III evidence creation, queries, attention retrieval, masks, token order and projections were reused. M2.1 changes only how retrieved evidence enters each modality. Final fusion remains 256→128, LayerNorm, GELU and dropout 0.2.
The inherited `text_residual_norm` and `visual_residual_norm` are frozen and recorded as ‘inherited legacy M2 assimilation parameters; not part of the M2.1 conceptual architecture.’ No normalization follows the M2.1 residual injection.

## C. M2 validation precheck

Six existing formal M2-P checkpoints supplied validation-only retrieval, compatibility and displacement measurements. Existing M2-S checkpoints supplied the historical per-class F1 comparison. Correlations below are descriptive associations with validation-node CE, not causal estimates.

| Dataset | Modality | Retrieval / H0 | Cosine(H0,U) | 1−Cosine(H0,Z) | Correct / incorrect retrieval ratio | Mean Spearman rho: ratio / cosine / displacement |
|---|---|---:|---:|---:|---:|---:|
| Movies | text | 0.658 | 0.200 | 0.246 | 0.664 / 0.651 | -0.06 / -0.12 / +0.01 |
| Movies | visual | 0.464 | 0.092 | 0.189 | 0.475 / 0.452 | -0.14 / -0.22 / -0.09 |
| Grocery | text | 0.942 | 0.145 | 0.316 | 0.976 / 0.794 | -0.45 / -0.41 / -0.20 |
| Grocery | visual | 1.158 | 0.159 | 0.349 | 1.189 / 1.021 | -0.35 / -0.51 / -0.04 |

`precheck_summary.csv`, `precheck_correctness.csv`, `precheck_classwise.csv`, and `precheck_correlations.csv` retain seed-level means/p10/p50/p90, support, historical F1, correctness strata and all six checkpoint measurements. The Spearman analyses use validation CE and make no causal claim.

## D. Initialization and gradient checks

Smoke used Movie seed42 for M2.1-S vector, M2.1-P vector and M2.1-P scalar, each for 2 epochs; all three smoke gradient and audit checks passed: True.
Adaptive vector S/P use paired seeds, initialize both assimilation output layers to zero, and start with `Z_text=H0_text` and `Z_visual=H0_visual`. Gradient traces confirm finite nonzero assimilation output-layer gradients at step one and nonzero upstream assimilation/retrieval gradients after step two.
Across all 14 trained checkpoints, maximum M2 retrieval and attention absolute differences under a shared Stage I/II state were 0.0e+00 and 0.0e+00; Stage I/II counterfactual maximum difference was 0.0e+00.

## E. Main validation results

| Dataset | Variant | Accuracy mean ± population SD | Macro-F1 mean ± population SD | Source |
|---|---|---:|---:|---|
| Movies | M2-S | 52.52% ± 0.53 pp | 31.84% ± 4.08 pp | reused frozen M2 pilot |
| Movies | M2-P | 52.96% ± 0.35 pp | 33.94% ± 0.53 pp | reused frozen M2 pilot |
| Movies | M2.1-S | 52.71% ± 0.46 pp | 33.26% ± 3.50 pp | retrained M2.1 |
| Movies | M2.1-P | 52.80% ± 0.31 pp | 33.50% ± 1.63 pp | retrained M2.1 |
| Grocery | M2-S | 82.12% ± 0.27 pp | 71.32% ± 1.25 pp | reused frozen M2 pilot |
| Grocery | M2-P | 81.76% ± 0.31 pp | 71.25% ± 1.05 pp | reused frozen M2 pilot |
| Grocery | M2.1-S | 81.08% ± 0.27 pp | 70.16% ± 1.32 pp | retrained M2.1 |
| Grocery | M2.1-P | 80.80% ± 0.24 pp | 69.61% ± 2.43 pp | retrained M2.1 |

### Seed-paired comparisons

| Dataset | Seed | Comparison | Δ accuracy | Δ Macro-F1 |
|---|---:|---|---:|---:|
| Movies | 42 | M2.1-P minus M2.1-S | +0.18 pp | -1.15 pp |
| Movies | 42 | M2.1-S minus M2-S | +0.48 pp | +6.66 pp |
| Movies | 42 | M2.1-P minus M2-P | -0.09 pp | -0.80 pp |
| Movies | 43 | M2.1-P minus M2.1-S | +0.21 pp | +3.00 pp |
| Movies | 43 | M2.1-S minus M2-S | -0.09 pp | -1.40 pp |
| Movies | 43 | M2.1-P minus M2-P | -0.18 pp | -1.81 pp |
| Movies | 44 | M2.1-P minus M2.1-S | -0.12 pp | -1.13 pp |
| Movies | 44 | M2.1-S minus M2-S | +0.18 pp | -1.00 pp |
| Movies | 44 | M2.1-P minus M2-P | -0.21 pp | +1.30 pp |
| Grocery | 42 | M2.1-P minus M2.1-S | +0.00 pp | +1.86 pp |
| Grocery | 42 | M2.1-S minus M2-S | -1.46 pp | -1.82 pp |
| Grocery | 42 | M2.1-P minus M2-P | -0.91 pp | +0.58 pp |
| Grocery | 43 | M2.1-P minus M2.1-S | -0.47 pp | -2.95 pp |
| Grocery | 43 | M2.1-S minus M2-S | -0.73 pp | +1.01 pp |
| Grocery | 43 | M2.1-P minus M2-P | -1.08 pp | -3.30 pp |
| Grocery | 44 | M2.1-P minus M2.1-S | -0.38 pp | -0.56 pp |
| Grocery | 44 | M2.1-S minus M2-S | -0.91 pp | -2.67 pp |
| Grocery | 44 | M2.1-P minus M2-P | -0.88 pp | -2.18 pp |

### Seed42 scalar sanity comparison

| Dataset | Metric | Adaptive vector | Global scalar | Scalar − vector |
|---|---|---:|---:|---:|
| Movies | Accuracy | 52.55% | 52.64% | +0.09 pp |
| Movies | Macro-F1 | 33.69% | 32.07% | -1.62 pp |
| Grocery | Accuracy | 80.94% | 81.93% | +1.00 pp |
| Grocery | Macro-F1 | 73.05% | 73.07% | +0.03 pp |

## F. Per-class validation results

`classwise_comparison.csv` contains validation support and historical M2 versus M2.1 class F1 for every seed and class. Across the 60 class-seed observations per cell, M2.1-S improved/lowered class F1 on Movies in 27/24 cases and Grocery in 17/42; M2.1-P improved/lowered class F1 on Movies in 21/29 and Grocery in 14/44. Full values and per-class retrieval/displacement are in the classwise precheck and comparison CSV files.

## G. Assimilation and evidence diagnostics

| Dataset | Variant | Mean correction ratio q: text / visual | Mean |a|: text / visual | Centered effective rank: text / visual | Active fraction |
|---|---|---:|---:|---:|---:|
| Movies | M2.1-S | 0.638 / 0.566 | 0.177 / 0.282 | 3.03 / 3.76 | 0.618 / 0.780 |
| Movies | M2.1-P | 0.731 / 0.632 | 0.209 / 0.316 | 3.19 / 3.83 | 0.686 / 0.814 |
| Grocery | M2.1-S | 0.980 / 1.065 | 0.391 / 0.449 | 5.40 / 6.87 | 0.854 / 0.882 |
| Grocery | M2.1-P | 1.050 / 1.149 | 0.408 / 0.475 | 5.32 / 6.60 | 0.866 / 0.893 |

Full M2 attention reports, field sign/near-zero fractions, nodewise norm variance, compatibility quintiles, parameter counts, gradients and resource data are in `mechanism_diagnostics.json`. Adaptive diversity uses at most 8192 deterministic evenly spaced nodes, centered covariance eigenvalues, top-1/top-5 energy, channel variance and active fraction.

## H. Frozen counterfactuals

The eight per-run M2.1-P vector interventions preserve Stage I/II. Assimilation-Off, Uniform-Node and Dimension Shuffle modify only the assimilation field. Provenance-Off, degree-matched Node Shuffle (seeds 3407/3408/3409) and Order Mismatch modify provenance copies only when constructing retrieval evidence.

| Dataset | Intervention | Runs | Mean Δ accuracy | Mean Δ Macro-F1 |
|---|---|---:|---:|---:|
| Movies | assimilation_off | 3 | -2.71 pp | -4.33 pp |
| Movies | uniform_node | 3 | -0.66 pp | -1.87 pp |
| Movies | dimension_shuffle | 3 | -3.79 pp | -4.97 pp |
| Movies | provenance_off | 3 | -1.21 pp | -1.46 pp |
| Movies | provenance_shuffle | 9 | -0.60 pp | -0.53 pp |
| Movies | order_mismatch | 3 | -6.60 pp | -4.31 pp |
| Grocery | assimilation_off | 3 | -7.20 pp | -5.76 pp |
| Grocery | uniform_node | 3 | -1.01 pp | -0.28 pp |
| Grocery | dimension_shuffle | 3 | -10.94 pp | -9.46 pp |
| Grocery | provenance_off | 3 | -2.49 pp | -1.61 pp |
| Grocery | provenance_shuffle | 9 | -3.23 pp | -2.60 pp |
| Grocery | order_mismatch | 3 | -8.81 pp | -7.97 pp |

The per-seed, per-shuffle outcomes are in `counterfactual_results.csv`; M2.1-P classwise counterfactual metrics are in each run audit JSON. These intervention outcomes describe this validation setup.

## I. Resources and parameter status

- Movies M2.1-S adaptive_vector seed 42: 83 epochs; 0.9 s/epoch; peak allocated/reserved 7.37/7.93 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Movies M2.1-P adaptive_vector seed 42: 87 epochs; 1.1 s/epoch; peak allocated/reserved 7.66/8.46 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Movies M2.1-S adaptive_vector seed 43: 68 epochs; 0.9 s/epoch; peak allocated/reserved 7.37/7.93 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Movies M2.1-P adaptive_vector seed 43: 72 epochs; 1.1 s/epoch; peak allocated/reserved 7.66/8.46 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Movies M2.1-S adaptive_vector seed 44: 76 epochs; 0.9 s/epoch; peak allocated/reserved 7.37/7.93 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Movies M2.1-P adaptive_vector seed 44: 74 epochs; 1.1 s/epoch; peak allocated/reserved 7.66/8.46 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Grocery M2.1-S adaptive_vector seed 42: 118 epochs; 0.8 s/epoch; peak allocated/reserved 7.00/7.49 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Grocery M2.1-P adaptive_vector seed 42: 127 epochs; 1.0 s/epoch; peak allocated/reserved 7.30/8.02 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Grocery M2.1-S adaptive_vector seed 43: 121 epochs; 0.8 s/epoch; peak allocated/reserved 7.00/7.49 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Grocery M2.1-P adaptive_vector seed 43: 95 epochs; 1.0 s/epoch; peak allocated/reserved 7.30/8.02 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Grocery M2.1-S adaptive_vector seed 44: 111 epochs; 0.8 s/epoch; peak allocated/reserved 7.00/7.49 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Grocery M2.1-P adaptive_vector seed 44: 104 epochs; 1.0 s/epoch; peak allocated/reserved 7.30/8.02 GiB; trainable parameters 1,196,822; inherited frozen 41,608.
- Movies M2.1-P global_scalar seed 42: 75 epochs; 1.1 s/epoch; peak allocated/reserved 7.62/8.46 GiB; trainable parameters 1,032,470; inherited frozen 41,608.
- Grocery M2.1-P global_scalar seed 42: 133 epochs; 1.0 s/epoch; peak allocated/reserved 7.26/8.02 GiB; trainable parameters 1,032,470; inherited frozen 41,608.

Scalar sanity results use seed42 only and are not a multi-seed estimate.

## J. Decision

**Decision: Assimilation alone does not solve limitation.** Adaptive vector M2.1-P did not improve both datasets over historical M2-P: Movies was slightly lower in mean accuracy and Macro-F1, while Grocery was lower by about 0.96 and 1.63 percentage points. M2.1-S also lost about 1.03 accuracy points on Grocery. Provenance-conditioned gains over M2.1-S varied by seed and dataset.
The learned vector field was active and the Assimilation-Off and Dimension-Shuffle checks reduced validation scores, especially on Grocery. This shows the model uses the field in its predictions, while the multi-seed M2 comparisons show that this use did not provide a reliable aggregate task gain. Global scalar was promising for Grocery seed42, but its Movies Macro-F1 was lower than adaptive vector and it was only one seed, so the pilot does not justify preferring scalar.

This decision is limited to the two-dataset, three-seed NC pilot and the seed42 scalar sanity runs. Test evaluation, LP, extra datasets, baselines, class weighting and formal ablations were excluded.
