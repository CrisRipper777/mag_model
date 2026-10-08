# MVCGE private NC v1 results

## Scope and execution

This is a descriptive train/validation development round on Movies, Grocery, and Reddit-S. The validation sets have prior development exposure and are not independent benchmarks. The study does not report test metrics, run a real LP task, or extend to Toys/ele-fashion.

- Training source commit: `d8d4e45b2d511363defea256e1796b48fd5ae3d2`.
- Formal cells: 108/108 calibration and 36/36 confirmation.
- Failed attempts: 0; retries: 0; unrun formal cells: 0.
- Update budget: 20/108 calibration cells reached 800 updates; 0/36 confirmation cells reached 800. Selected calibration configs at the limit: Grocery/G0, Grocery/G1, Reddit-S/U1.
- Recorded GPU training wall time: 1.71 h for calibration, 0.47 h for confirmation, 2.17 h total.
- Train-only smoke: 9/9; test task-head calls 0; real LP runs 0.
- Metrics use the fixed class set 0–19 with `zero_division=0`; summary SD is population SD (`ddof=0`).

## Confirmation results

| Dataset | Model | Validation Acc (%) | Macro-F1 (%) | Backbone/task head params | Mean median step (ms) | Max peak allocated (GiB) | Cal/confirm/total wall (h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | A0 | 53.44 ± 0.37 | 42.98 ± 2.32 | 893,056/20,560 | 37.03 | 1.50 | 0.04/0.01/0.05 |
| Movies | G0 | 52.84 ± 0.20 | 42.34 ± 1.39 | 868,940/5,140 | 94.58 | 2.10 | 0.07/0.04/0.10 |
| Movies | G1 | 53.18 ± 0.15 | 44.15 ± 0.90 | 902,348/5,140 | 96.80 | 2.19 | 0.08/0.05/0.13 |
| Movies | U1 | 52.42 ± 0.33 | 41.72 ± 1.74 | 732,288/5,140 | 74.32 | 1.65 | 0.06/0.03/0.10 |
| Grocery | A0 | 80.59 ± 0.17 | 72.91 ± 0.77 | 893,056/20,560 | 36.91 | 1.69 | 0.05/0.02/0.07 |
| Grocery | G0 | 81.31 ± 0.28 | 74.48 ± 0.33 | 868,940/5,140 | 96.16 | 2.33 | 0.23/0.06/0.29 |
| Grocery | G1 | 80.88 ± 0.21 | 73.56 ± 0.66 | 902,348/5,140 | 86.71 | 2.43 | 0.30/0.04/0.34 |
| Grocery | U1 | 80.81 ± 0.29 | 74.03 ± 0.64 | 732,288/5,140 | 74.33 | 1.88 | 0.20/0.04/0.24 |
| Reddit-S | A0 | 95.86 ± 0.04 | 92.29 ± 0.01 | 893,056/20,560 | 36.62 | 2.09 | 0.10/0.03/0.13 |
| Reddit-S | G0 | 94.81 ± 0.09 | 89.87 ± 0.49 | 868,940/5,140 | 100.23 | 2.41 | 0.16/0.05/0.22 |
| Reddit-S | G1 | 94.55 ± 0.06 | 89.50 ± 0.22 | 902,348/5,140 | 105.17 | 2.50 | 0.21/0.05/0.25 |
| Reddit-S | U1 | 94.24 ± 0.18 | 89.05 ± 0.30 | 732,288/5,140 | 79.87 | 1.98 | 0.21/0.04/0.25 |

### Confirmation seed records

| Dataset | Model | Seed | LR/WD | Best step | Validation Acc/Macro-F1 (%) | Mean step (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Grocery | A0 | 103 | 0.003/0.001 | 116 | 80.53/72.10 | 36.94 |
| Grocery | A0 | 104 | 0.003/0.001 | 201 | 80.82/73.95 | 36.98 |
| Grocery | A0 | 105 | 0.003/0.001 | 131 | 80.41/72.68 | 36.80 |
| Grocery | G0 | 103 | 0.003/0.01 | 486 | 81.61/74.85 | 95.08 |
| Grocery | G0 | 104 | 0.003/0.01 | 311 | 80.94/74.06 | 95.70 |
| Grocery | G0 | 105 | 0.003/0.01 | 366 | 81.38/74.53 | 97.69 |
| Grocery | G1 | 103 | 0.003/0.0001 | 201 | 80.97/74.10 | 97.20 |
| Grocery | G1 | 104 | 0.003/0.0001 | 206 | 80.59/72.62 | 66.90 |
| Grocery | G1 | 105 | 0.003/0.0001 | 326 | 81.08/73.95 | 96.02 |
| Grocery | U1 | 103 | 0.003/0.01 | 316 | 80.73/74.14 | 75.20 |
| Grocery | U1 | 104 | 0.003/0.01 | 166 | 80.50/73.19 | 74.83 |
| Grocery | U1 | 105 | 0.003/0.01 | 446 | 81.20/74.76 | 72.95 |
| Movies | A0 | 103 | 0.003/0.0001 | 51 | 53.96/44.31 | 37.13 |
| Movies | A0 | 104 | 0.003/0.0001 | 41 | 53.21/39.72 | 37.05 |
| Movies | A0 | 105 | 0.003/0.0001 | 51 | 53.15/44.92 | 36.91 |
| Movies | G0 | 103 | 0.001/0.01 | 81 | 52.70/40.52 | 94.85 |
| Movies | G0 | 104 | 0.001/0.01 | 101 | 52.70/42.63 | 95.06 |
| Movies | G0 | 105 | 0.001/0.01 | 301 | 53.12/43.89 | 93.84 |
| Movies | G1 | 103 | 0.003/0.0001 | 341 | 53.27/44.63 | 95.62 |
| Movies | G1 | 104 | 0.003/0.0001 | 236 | 52.97/42.89 | 97.83 |
| Movies | G1 | 105 | 0.003/0.0001 | 276 | 53.30/44.92 | 96.97 |
| Movies | U1 | 103 | 0.003/0.01 | 411 | 52.82/44.17 | 74.96 |
| Movies | U1 | 104 | 0.003/0.01 | 76 | 52.43/40.26 | 74.49 |
| Movies | U1 | 105 | 0.003/0.01 | 76 | 52.01/40.74 | 73.51 |
| Reddit-S | A0 | 103 | 0.0003/0.0001 | 281 | 95.91/92.27 | 36.52 |
| Reddit-S | A0 | 104 | 0.0003/0.0001 | 286 | 95.82/92.30 | 36.66 |
| Reddit-S | A0 | 105 | 0.0003/0.0001 | 196 | 95.85/92.30 | 36.67 |
| Reddit-S | G0 | 103 | 0.003/0.01 | 306 | 94.87/90.39 | 98.23 |
| Reddit-S | G0 | 104 | 0.003/0.01 | 271 | 94.87/89.99 | 99.09 |
| Reddit-S | G0 | 105 | 0.003/0.01 | 291 | 94.68/89.21 | 103.38 |
| Reddit-S | G1 | 103 | 0.003/0.001 | 171 | 94.62/89.81 | 108.25 |
| Reddit-S | G1 | 104 | 0.003/0.001 | 181 | 94.46/89.31 | 104.36 |
| Reddit-S | G1 | 105 | 0.003/0.001 | 331 | 94.56/89.39 | 102.90 |
| Reddit-S | U1 | 103 | 0.003/0.001 | 301 | 94.50/89.48 | 79.76 |
| Reddit-S | U1 | 104 | 0.003/0.001 | 146 | 94.12/88.86 | 80.44 |
| Reddit-S | U1 | 105 | 0.003/0.001 | 271 | 94.12/88.82 | 79.39 |

### Selected configurations

| Dataset | Model | LR | Matrix WD | Calibration best step | Calibration validation Acc/F1 (%) |
|---|---:|---:|---:|---:|---:|
| Movies | A0 | 0.003 | 0.0001 | 36 | 53.72/43.74 |
| Movies | G0 | 0.001 | 0.01 | 71 | 52.67/41.13 |
| Movies | G1 | 0.003 | 0.0001 | 66 | 52.67/38.28 |
| Movies | U1 | 0.003 | 0.01 | 61 | 52.37/37.88 |
| Grocery | A0 | 0.003 | 0.001 | 116 | 80.79/73.37 |
| Grocery | G0 | 0.003 | 0.01 | 726 | 81.26/74.00 |
| Grocery | G1 | 0.003 | 0.0001 | 666 | 81.79/74.63 |
| Grocery | U1 | 0.003 | 0.01 | 211 | 81.32/74.33 |
| Reddit-S | A0 | 0.0003 | 0.0001 | 596 | 96.04/92.67 |
| Reddit-S | G0 | 0.003 | 0.01 | 351 | 94.59/89.24 |
| Reddit-S | G1 | 0.003 | 0.001 | 506 | 94.65/89.76 |
| Reddit-S | U1 | 0.003 | 0.001 | 756 | 94.62/89.34 |

### Tuned recipe contrasts

| Contrast | Dataset | ΔAcc (pp) | ΔMacro-F1 (pp) |
|---|---|---:|---:|
| G1-A0 | Movies | -0.26 | +1.17 |
| G1-G0 | Movies | +0.34 | +1.80 |
| G1-U1 | Movies | +0.76 | +2.42 |
| G1-A0 | Grocery | +0.29 | +0.65 |
| G1-G0 | Grocery | -0.43 | -0.92 |
| G1-U1 | Grocery | +0.07 | -0.47 |
| G1-A0 | Reddit-S | -1.31 | -2.79 |
| G1-G0 | Reddit-S | -0.26 | -0.36 |
| G1-U1 | Reddit-S | +0.30 | +0.45 |
| G1-A0 | equal-graph mean | -0.43 | -0.32 |
| G1-G0 | equal-graph mean | -0.12 | +0.17 |
| G1-U1 | equal-graph mean | +0.38 | +0.80 |

## Training and routing diagnostics

Calibration seed 102 is excluded from the confirmation table. Each result uses the checkpoint selected by validation correct count, then Macro-F1, then earlier step. Confirmation means compare independently selected LR/WD recipes; the contrasts are descriptive and do not isolate a single mechanism. U1 lacks the router/context parameters and is not capacity-matched to G1.

Across the smoke set, all 9 runs completed three train-only updates. Allocated memory ranged from 1.58 to 2.20 GiB. No validation, test head, or real LP call occurred in smoke.

Router usage, noise, entropy, σ, polynomial coefficient magnitudes, and task-only router gradients are recorded per run in `run_registry.json`. These diagnostics describe the fitted training recipe and are not evidence of recovered semantic roles or causal specialization. Cells reaching 800 updates are flagged as possibly budget-limited; they are not interpreted as model failure.

Confirmation routing concentration: G0: 12/36 layer-modality profiles used some expert on <1% of nodes; the most common top-2 pair covered 78.6% of nodes on average. G1: 12/36 layer-modality profiles used some expert on <1% of nodes; the most common top-2 pair covered 75.0% of nodes on average. Top-2 concentration is a warning to inspect expert utility; it does not establish that an expert is functionally dead because every expert is computed and the summary does not ablate experts.

## Implementation facts and protocol differences

A0 reuses the previous S0 B1 model, 768-D padded readout, path CE, reconstruction and pair losses. The selected S0 calibration LR/WD values were verified from its committed selection lock and already lie in the new grid. The current common optimizer protocol uses gradient clip 5, while the previous v2 S-stage used clip 1; this round therefore reports a fresh A0 recipe, not a bitwise reproduction of historical checkpoints.

The new encoders use sparse P-derived basis operators built once from the canonical loop-free loader edge list, then add one self-loop for their own normalized propagation. All 768-D frozen features are used as loaded. The loader exposes no edge weights or missing-modality metadata; all observed nodes are treated as having both modalities. No missingness performance is established.

## Self-audit

The saved train/validation argmax files were independently re-read by this audit script and their sample counts, correct counts, full 20×20 confusion matrices, accuracy and Macro-F1 matched the registry (float metrics tolerance 1e-7). Prediction and checkpoint hashes, split IDs, source commit/hashes and calibration selection were checked. Result: `True`.

The runtime call path invokes the task head only with train IDs for loss and validation IDs for selection; final evaluation uses train and validation IDs separately. The NC runner loads `task=nc`; there is no real LP task call. A synthetic `LinkPredictor.score_pairs` test only verifies the encoder/decoder interface.

Functional suite: 16 passed, 0 failed. This is the authors' self-audit, not an independent review.

## Interpretation

**Development decision: evidence insufficient to expand G1 unchanged.** Equal-graph confirmation differences are G1−A0 -0.43 pp Acc / -0.32 pp Macro-F1; G1−G0 -0.12/+0.17 pp; and G1−U1 +0.38/+0.80 pp. Per-graph reversals and prior validation exposure limit a broad conclusion; the results do not identify one mechanism for a targeted correction.

The evidence supports only a decision about whether these complete training recipes merit another bounded development round. It does not establish NC/LP unified performance, independent generalization, semantic disentanglement, or statistical significance. Interpret the per-graph results, stability, resource use and router gradient diagnostics together; do not infer the mechanism from a single gate or fit statistic.
