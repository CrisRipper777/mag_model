# M0.1 Movies/Grocery Pilot Report

## Protocol and artifacts

- Branch: `exp/relation_transport_m0`; source base: `exp/problem_validation_p02` at `45b2e51772998c588c8ed1a1937b7a7084c0fdd1`.
- Unified full-graph NC (`unified_full_graph_nc_v1`), raw loader-provided frozen Text/Visual features, train split for fitting, original validation accuracy for checkpoint selection. All 18 formal runs set `task.evaluate_test=false`, `task.inference_mode=full`, `task.training_mode=full_graph`; no LP was run.
- No exact matching `multi_order_bank + gpr + hidden_dim=128` NC val-only result was present in project result artifacts, so B0 was run on both datasets and all three seeds.
- Each run uses the project defaults: up to 300 epochs, eval every epoch, early-stop patience 30 with minimum epoch 30. Checkpoints are selected by validation accuracy.
- Per-run logs, Hydra snapshots, and checkpoints are under `outputs/model_design/m0/`; per-seed metrics/diagnostics, aggregated tables, and the run index are under `results/model_design/m0/`. Diagnostic values below are the unweighted mean of per-seed summaries. Population SD is used for performance.
- Parameter count reports the trainable encoder/model; the NC linear head adds 2,580 parameters. The combined count is included.

## Tests and smoke

- New tests: **18 passed** (one upstream PyG deprecation warning).
- Movies seed 42, two-epoch GPU smoke completed for both transport modes with finite training/validation outputs and saved checkpoints. `task.evaluate_test=false` and full-graph inference were used.
- Peak CUDA memory allocated / reserved: identity **6.47 / 7.00 GiB**; orthogonal **7.46 / 8.02 GiB** on RTX 3090 GPU0.
- Checkpoint tensors and inference embeddings were finite. Exact `inference()` vs eval-forward maximum absolute error: identity **9.42e-6**, orthogonal **6.47e-6** (CUDA scatter accumulation).
- The 18 formal run logs contain no test metric lines and no NaN/Inf tokens.

## Validation performance

Values are percentages and reported as mean ± population SD over seeds 42/43/44. Best epoch shows the individual seed epochs.

| Dataset | Variant | Val Acc (%) | Val Macro-F1 (%) | Encoder params | Encoder + NC head | Best epoch mean ± SD [42/43/44] |
|---|---|---:|---:|---:|---:|---:|
| Movies | identity_transport | 50.71 ± 0.86 | 29.81 ± 2.17 | 745,420 | 748,000 | 39.3 ± 4.9 [45/33/40] |
| Movies | orthogonal_transport | 52.96 ± 0.26 | 38.03 ± 2.44 | 746,460 | 749,040 | 64.3 ± 12.1 [77/48/68] |
| Movies | multi_order_bank_gpr | 57.64 ± 0.19 | 49.73 ± 0.72 | 230,280 | 232,860 | 125.7 ± 20.7 [153/121/103] |
| Grocery | identity_transport | 81.00 ± 0.13 | 72.47 ± 0.64 | 745,420 | 748,000 | 132.7 ± 19.9 [160/125/113] |
| Grocery | orthogonal_transport | 81.52 ± 0.21 | 72.19 ± 0.30 | 746,460 | 749,040 | 95.0 ± 22.1 [69/123/93] |
| Grocery | multi_order_bank_gpr | 83.57 ± 0.19 | 75.52 ± 0.72 | 230,280 | 232,860 | 130.0 ± 28.2 [165/96/129] |

## Paired orthogonal minus identity differences

Accuracy and Macro-F1 deltas are percentage points (pp), aligned by seed; mean ± population SD is across the three paired differences.

| Dataset | Val Acc pp, seed42/43/44 | Val Acc pp mean ± SD | Val Macro-F1 pp, seed42/43/44 | Val Macro-F1 pp mean ± SD |
|---|---:|---:|---:|---:|
| Movies | +3.24/+1.98/+1.53 | +2.25 ± 0.72 | +8.02/+7.99/+8.64 | +8.22 ± 0.30 |
| Grocery | +0.67/+0.06/+0.85 | +0.53 ± 0.34 | -0.79/-0.69/+0.63 | -0.28 ± 0.65 |

## Comparison with B0

Orthogonal minus `multi_order_bank + gpr + hidden_dim=128`, using three-seed means. M0.1 is below B0 on both datasets and metrics.

| Dataset | Val Acc delta (pp) | Val Macro-F1 delta (pp) |
|---|---:|---:|
| Movies | -4.68 | -11.70 |
| Grocery | -2.05 | -3.33 |

## Relation and transport diagnostics

Conductance shows the mean across directed-edge distributions (parentheses: edge-wise SD averaged over seeds) and the across-seed average p10/p50/p90. Angles summarize absolute angles. Relative displacement uses 4,096 deterministic sampled directed edges per run. Identity has exact zero angles and displacement by definition.

| Dataset | Variant | Modality | Conductance mean (edge SD) | Conductance p10/p50/p90 | Abs angle mean/p50/p90 | Relative displacement mean/median/p90 |
|---|---|---|---:|---:|---:|---:|
| Movies | identity_transport | text | 0.335 (0.060) | 0.249/0.344/0.408 | 0.000/0.000/0.000 | 0.000/0.000/0.000 |
| Movies | identity_transport | visual | 0.524 (0.065) | 0.438/0.525/0.612 | 0.000/0.000/0.000 | 0.000/0.000/0.000 |
| Movies | orthogonal_transport | text | 0.342 (0.070) | 0.258/0.334/0.441 | 0.514/0.307/1.227 | 0.820/0.826/0.941 |
| Movies | orthogonal_transport | visual | 0.565 (0.074) | 0.472/0.563/0.656 | 0.603/0.526/1.180 | 0.868/0.875/0.976 |
| Grocery | identity_transport | text | 0.539 (0.139) | 0.361/0.532/0.721 | 0.000/0.000/0.000 | 0.000/0.000/0.000 |
| Grocery | identity_transport | visual | 0.679 (0.214) | 0.409/0.659/0.975 | 0.000/0.000/0.000 | 0.000/0.000/0.000 |
| Grocery | orthogonal_transport | text | 0.632 (0.138) | 0.474/0.611/0.819 | 0.659/0.655/1.150 | 0.933/0.930/1.078 |
| Grocery | orthogonal_transport | visual | 0.714 (0.145) | 0.523/0.711/0.909 | 0.653/0.688/1.146 | 0.925/0.933/1.042 |

## Semantic state norms

Each state reports node L2 norm `mean / p50 / p95`, averaged over three seed summaries.

| Dataset | Variant | Modality | H0 μ/p50/p95 | S1 μ/p50/p95 | S2 μ/p50/p95 | S3 μ/p50/p95 |
|---|---|---|---:|---:|---:|---:|
| Movies | identity_transport | text | 7.50/7.49/8.07 | 11.47/11.47/11.51 | 11.47/11.47/11.51 | 11.48/11.48/11.51 |
| Movies | identity_transport | visual | 7.68/7.68/8.24 | 11.56/11.56/11.60 | 11.56/11.56/11.60 | 11.55/11.56/11.60 |
| Movies | orthogonal_transport | text | 7.42/7.42/8.05 | 11.49/11.49/11.53 | 11.50/11.50/11.54 | 11.50/11.50/11.54 |
| Movies | orthogonal_transport | visual | 7.52/7.52/8.07 | 11.66/11.66/11.69 | 11.67/11.67/11.70 | 11.67/11.67/11.70 |
| Grocery | identity_transport | text | 7.59/7.58/8.25 | 11.98/11.98/12.04 | 11.98/11.99/12.04 | 11.98/11.98/12.04 |
| Grocery | identity_transport | visual | 7.52/7.56/8.34 | 11.93/11.93/11.98 | 11.93/11.93/11.99 | 11.93/11.93/11.99 |
| Grocery | orthogonal_transport | text | 7.46/7.46/8.07 | 11.76/11.76/11.81 | 11.76/11.76/11.81 | 11.77/11.77/11.82 |
| Grocery | orthogonal_transport | visual | 7.52/7.58/8.32 | 11.75/11.75/11.79 | 11.75/11.75/11.79 | 11.75/11.75/11.79 |

## GPR weights

Each M0 vector is the mean softmax weight for orders 0/1/2/3 over three seeds; entries sum to one.

| Dataset | Variant | Modality | γ0/γ1/γ2/γ3 |
|---|---|---|---:|
| Movies | identity_transport | text | 0.253/0.257/0.245/0.244 |
| Movies | identity_transport | visual | 0.252/0.257/0.247/0.245 |
| Movies | orthogonal_transport | text | 0.252/0.258/0.247/0.244 |
| Movies | orthogonal_transport | visual | 0.246/0.258/0.250/0.246 |
| Grocery | identity_transport | text | 0.231/0.259/0.260/0.250 |
| Grocery | identity_transport | visual | 0.233/0.255/0.261/0.252 |
| Grocery | orthogonal_transport | text | 0.238/0.258/0.256/0.249 |
| Grocery | orthogonal_transport | visual | 0.240/0.255/0.256/0.249 |

## Interpretation

- Orthogonal improves validation accuracy over identity for all three seeds on both Movies and Grocery. Movies Macro-F1 also improves on every seed; Grocery Macro-F1 is mixed and the mean paired delta is slightly negative.
- Learned orthogonal transport is nontrivial: mean absolute angles are about 0.51–0.66 radians and mean relative source displacement is about 0.82–0.93. Identity remains zero by definition.
- State norms are bounded and nearly flat from S1 through S3 after LayerNorm. Conductance remains finite within `(0, 2)` in every run. No numerical instability was observed.
- Both M0 variants trail B0 on both datasets, despite more than three times the encoder parameter count. This is a material negative finding for the prototype.
- Under the specified M0.1 interpretation rule, orthogonal has positive accuracy trends on both datasets, active transport, stable states, and no systematic degradation relative to identity. **Proceeding to M0.2 relation-context refinement is recommended**, while retaining the B0 gap as a central design problem. No M0.2 change is implemented here.

## Files and reproducibility

- Runner: `scripts/run_relation_transport_m0_pilot.py`
- Run index: `results/model_design/m0/pilot_records.json`
- Checkpoints: `outputs/model_design/m0/checkpoints/`
- Per-run logs and Hydra snapshots: `outputs/model_design/m0/runs/`
- Per-run JSON diagnostics: `results/model_design/m0/runs/`
- Machine-readable summaries: `performance_summary.csv`, `paired_differences.csv`, `orthogonal_vs_b0.csv`, `transport_diagnostics.csv`, `semantic_norm_diagnostics.csv`, `gpr_weights.csv`.
- Smoke logs/checkpoints/Hydra output: `outputs/model_design/m0/smoke/`
- Smoke diagnostics and inference checks: `results/model_design/m0/smoke/`
