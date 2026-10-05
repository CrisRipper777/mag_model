# Prior-context MAG pilot v3 report

## Decision and scope

Stage A did not pass the pre-registered validation and/or resource gate against both R0 and R1. Stage B was skipped exactly as specified; the matrix, loss, and selection rules were not expanded after seeing validation results.
The v2 frozen evidence and protocol remain unchanged. V2 already supplied unpropagated h0 to its final FusionMLP; v3 explicitly adds a coefficient-one center representation and source supervision, without claiming lossless preservation. This run covers Movies/Grocery train/validation only. No test predictions or metrics, LP, or five-NC extension were run.
Formal successful cells: 30; failed attempts: 0; incomplete/running attempts: 0; Stage B status: `skipped_gate`.
Synthetic GPU smoke attempts: 3 passed and 1 GPU-visibility environment failures recorded; these did not use real graph data.
Movies and Grocery are previously studied development graphs; seed52 is LR calibration, while only seeds53/54 enter gate comparisons. The seed summaries below are descriptive and do not imply statistical significance.

### Stage A gate

| R4 comparator | Mean Δ Acc (pp) | Mean Δ Macro-F1 (pp) | Positive pairs Acc/F1 | Metric path | Per-graph floor |
|---|---:|---:|---:|---|---|
| R4−R0 | +1.058 | -0.093 | 3/4 / 2/4 | fail | fail |
| R4−R1 | -1.349 | -5.340 | 0/4 / 0/4 | fail | fail |

Confirmation-seed mean R4/R1 step-time ratio: 4.522; allocated-memory ratio: 3.441; resource gate: fail.
Both R4 baselines must pass a metric path and the per-graph floor; resource ratios are compared against R1 and equally average the two graph ratios.

## Validation and train metrics

Mean ± population standard deviation (ddof=0), percentage units. Each selected row includes seed52 calibration checkpoint reused plus seeds53/54 confirmation.

| Dataset | Scheme | Val Acc | Val Macro-F1 | Train Acc | Train Macro-F1 | Train−Val Acc (pp) | Train−Val F1 (pp) | LR seeds52–54 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Movies | R0 | 51.33 ± 0.41 | 34.81 ± 0.87 | 72.49 ± 3.84 | 59.14 ± 5.68 | +21.16 | +24.33 | s52=0.0003, s53=0.0003, s54=0.0003 |
| Movies | R1 | 52.70 ± 0.11 | 39.97 ± 0.90 | 72.82 ± 1.19 | 65.20 ± 1.93 | +20.13 | +25.23 | s52=0.001, s53=0.001, s54=0.001 |
| Movies | R4 | 50.91 ± 0.52 | 34.60 ± 0.76 | 65.19 ± 0.78 | 54.12 ± 0.76 | +14.28 | +19.52 | s52=0.003, s53=0.003, s54=0.003 |
| Grocery | R0 | 77.19 ± 0.12 | 68.78 ± 1.01 | 95.01 ± 3.07 | 90.59 ± 5.38 | +17.82 | +21.81 | s52=0.0003, s53=0.0003, s54=0.0003 |
| Grocery | R1 | 80.41 ± 0.19 | 72.63 ± 0.62 | 97.35 ± 2.50 | 95.66 ± 4.48 | +16.94 | +23.03 | s52=0.003, s53=0.003, s54=0.003 |
| Grocery | R4 | 79.56 ± 0.17 | 67.99 ± 0.49 | 92.69 ± 0.69 | 84.32 ± 0.87 | +13.13 | +16.33 | s52=0.0003, s53=0.0003, s54=0.0003 |

### Seed-level records

| Dataset | Scheme | Seed | LR | Train Acc/F1 (%) | Train CE step | Val Acc/F1 (%) | Val CE | Best epoch | Wall (min) | Step median (ms) | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | R0 | 52 | 0.0003 | 98.53/96.89 | 0.0994/1.6039 | 77.19/69.81 | 0.9886 | 126 | 0.11 | 8.112 | 0.57 | 5.35 |
| Grocery | R0 | 52 | 0.001 | 99.96/99.97 | 0.0022/0.5775 | 76.93/69.00 | 2.0526 | 212 | 0.17 | 10.720 | 0.57 | 5.35 |
| Grocery | R0 | 52 | 0.003 | 95.18/90.92 | 0.1961/1.0315 | 75.90/67.45 | 1.1048 | 66 | 0.11 | 8.102 | 0.57 | 5.35 |
| Grocery | R0 | 53 | 0.0003 | 91.05/83.74 | 0.3546/1.8344 | 77.04/67.40 | 0.8495 | 89 | 0.09 | 10.210 | 0.57 | 5.35 |
| Grocery | R0 | 54 | 0.0003 | 95.45/91.14 | 0.2080/1.7440 | 77.34/69.12 | 0.8986 | 106 | 0.10 | 10.522 | 0.57 | 5.35 |
| Grocery | R1 | 52 | 0.0003 | 97.32/93.09 | 0.1744/1.6975 | 78.89/67.90 | 0.9571 | 272 | 0.37 | 23.828 | 1.26 | 5.35 |
| Grocery | R1 | 52 | 0.001 | 99.28/98.66 | 0.0534/1.2068 | 80.35/71.82 | 1.1278 | 133 | 0.23 | 21.824 | 1.26 | 5.35 |
| Grocery | R1 | 52 | 0.003 | 99.81/99.83 | 0.0151/0.8998 | 80.56/72.81 | 1.2452 | 91 | 0.14 | 23.379 | 1.26 | 5.35 |
| Grocery | R1 | 53 | 0.003 | 93.92/89.44 | 0.2374/1.0821 | 80.15/71.80 | 0.8185 | 62 | 0.11 | 22.458 | 1.26 | 5.35 |
| Grocery | R1 | 54 | 0.003 | 98.31/97.71 | 0.0755/0.9706 | 80.53/73.29 | 0.9676 | 79 | 0.13 | 23.710 | 1.26 | 5.35 |
| Grocery | R4 | 52 | 0.0003 | 93.15/85.16 | 0.2903/1.7317 | 79.44/67.77 | 0.7891 | 155 | 0.66 | 102.583 | 4.25 | 5.44 |
| Grocery | R4 | 52 | 0.001 | 87.01/77.49 | 0.4656/1.6182 | 78.65/67.18 | 0.7805 | 58 | 0.32 | 102.717 | 4.25 | 5.44 |
| Grocery | R4 | 52 | 0.003 | 97.68/95.63 | 0.1187/1.1169 | 78.86/69.24 | 1.0397 | 52 | 0.26 | 102.872 | 4.25 | 5.44 |
| Grocery | R4 | 53 | 0.0003 | 93.21/84.68 | 0.2804/1.7595 | 79.80/67.53 | 0.7914 | 161 | 0.65 | 102.765 | 4.25 | 5.44 |
| Grocery | R4 | 54 | 0.0003 | 91.71/83.13 | 0.3513/1.7855 | 79.44/68.68 | 0.7741 | 147 | 0.61 | 102.773 | 4.25 | 5.44 |
| Movies | R0 | 52 | 0.0003 | 72.48/60.03 | 0.8766/1.9596 | 51.89/35.95 | 1.5374 | 90 | 0.12 | 8.512 | 0.56 | 0.63 |
| Movies | R0 | 52 | 0.001 | 66.76/57.41 | 1.0249/1.8315 | 51.77/35.91 | 1.5097 | 47 | 0.06 | 9.186 | 0.56 | 0.63 |
| Movies | R0 | 52 | 0.003 | 66.37/60.19 | 1.0299/1.6512 | 50.84/39.87 | 1.5358 | 43 | 0.05 | 8.882 | 0.56 | 0.63 |
| Movies | R0 | 53 | 0.0003 | 77.20/65.61 | 0.7418/1.9457 | 50.93/34.65 | 1.5985 | 96 | 0.09 | 10.668 | 0.56 | 0.63 |
| Movies | R0 | 54 | 0.0003 | 67.79/51.79 | 1.0178/2.0215 | 51.17/33.84 | 1.5229 | 80 | 0.08 | 10.399 | 0.56 | 0.63 |
| Movies | R1 | 52 | 0.0003 | 66.30/48.87 | 1.1074/1.9540 | 51.20/31.33 | 1.5905 | 137 | 0.20 | 26.149 | 1.30 | 1.51 |
| Movies | R1 | 52 | 0.001 | 71.89/63.80 | 0.8914/1.7211 | 52.67/39.00 | 1.6009 | 66 | 0.12 | 23.426 | 1.30 | 1.51 |
| Movies | R1 | 52 | 0.003 | 85.43/84.18 | 0.5343/1.3825 | 52.64/41.12 | 1.8701 | 79 | 0.13 | 22.999 | 1.30 | 1.51 |
| Movies | R1 | 53 | 0.001 | 72.08/63.87 | 0.8787/1.7099 | 52.85/39.74 | 1.4824 | 72 | 0.16 | 22.237 | 1.30 | 1.51 |
| Movies | R1 | 54 | 0.001 | 74.51/67.93 | 0.8416/1.7281 | 52.58/41.17 | 1.6004 | 78 | 0.13 | 23.387 | 1.30 | 1.51 |
| Movies | R4 | 52 | 0.0003 | 68.34/47.76 | 0.9669/1.9476 | 50.18/30.06 | 1.6985 | 109 | 0.48 | 104.511 | 4.59 | 5.05 |
| Movies | R4 | 52 | 0.001 | 67.66/53.92 | 1.0018/1.7747 | 50.99/34.37 | 1.5634 | 62 | 0.29 | 104.716 | 4.59 | 5.05 |
| Movies | R4 | 52 | 0.003 | 65.65/54.17 | 1.0585/1.6445 | 51.26/35.38 | 1.5451 | 42 | 0.24 | 104.626 | 4.59 | 5.05 |
| Movies | R4 | 53 | 0.003 | 65.84/55.02 | 1.0398/1.6563 | 50.18/33.58 | 1.5918 | 41 | 0.24 | 104.718 | 4.59 | 5.05 |
| Movies | R4 | 54 | 0.003 | 64.09/53.17 | 1.1098/1.7025 | 51.29/34.85 | 1.5070 | 37 | 0.22 | 104.763 | 4.59 | 5.05 |

## Matched confirmation differences

Matched seed53/54 pairs; values are candidate minus comparator. These descriptive differences use 4 cells (2 graphs × 2 seeds).

| Comparison | Mean Δ Acc (pp) | Mean Δ Macro-F1 (pp) | Positive Acc pairs | Positive F1 pairs |
|---|---:|---:|---:|---:|
| R4-R0 | +1.058 | -0.093 | 3/4 | 2/4 |
| R4-R1 | -1.349 | -5.340 | 0/4 | 0/4 |

## Capacity, optimization cost, and diagnostics

Backbone parameter widths were chosen by active trainable backbone parameter count before results. Main/source heads are listed separately. Parameter matching does not imply matched compute, optimization, or function family.

| Dataset | Scheme | Width | Active backbone params | NC task heads | Total trainable | Δ vs R4 |
|---|---|---:|---:|---:|---:|---:|
| Movies | R0 | 1596 | 811,964 | 7,740 | 819,704 | -0.010% |
| Movies | R1 | 144 | 811,744 | 7,740 | 819,484 | -0.037% |
| Movies | R2 | 108 | 812,176 | 7,740 | 819,916 | +0.016% |
| Movies | R3 | fixed | 812,044 | 7,740 | 819,784 | +0.000% |
| Movies | R4 | fixed | 812,044 | 7,740 | 819,784 | +0.000% |
| Movies | R5 | fixed | 812,046 | 7,740 | 819,786 | +0.000% |
| Movies | R6 | fixed | 246,272 | 7,740 | 254,012 | -69.673% |
| Grocery | R0 | 1596 | 811,964 | 7,740 | 819,704 | -0.010% |
| Grocery | R1 | 144 | 811,744 | 7,740 | 819,484 | -0.037% |
| Grocery | R2 | 108 | 812,176 | 7,740 | 819,916 | +0.016% |
| Grocery | R3 | fixed | 812,044 | 7,740 | 819,784 | +0.000% |
| Grocery | R4 | fixed | 812,044 | 7,740 | 819,784 | +0.000% |
| Grocery | R5 | fixed | 812,046 | 7,740 | 819,786 | +0.000% |
| Grocery | R6 | fixed | 246,272 | 7,740 | 254,012 | -69.673% |

### R4 routing and structural health summaries

These summaries use the preregistered 512 train/validation nodes and the selected seed52 checkpoint plus seeds53/54. Each entry is mean ± population SD across those three fitted checkpoints; route entropy/variance are within the fixed node sample before this across-checkpoint summary. They describe trained computations and do not identify semantic roles or causal utility.

| Dataset | Flow | S weight L1 | S weight L2 | Entropy L1/L2 | Within-sample S-weight variance L1/L2 |
|---|---|---:|---:|---:|---:|
| Movies | joint | 0.005 ± 0.004 | 0.001 ± 0.001 | 0.029 ± 0.020 / 0.004 ± 0.003 | 0.0001 / 0.0000 |
| Movies | text | 0.033 ± 0.025 | 0.194 ± 0.214 | 0.122 ± 0.090 / 0.331 ± 0.284 | 0.0010 / 0.0002 |
| Movies | image | 0.386 ± 0.144 | 0.121 ± 0.170 | 0.603 ± 0.127 / 0.221 ± 0.305 | 0.0055 / 0.0003 |
| Grocery | joint | 0.669 ± 0.134 | 0.708 ± 0.016 | 0.559 ± 0.107 / 0.506 ± 0.027 | 0.0143 / 0.0403 |
| Grocery | text | 0.336 ± 0.073 | 0.527 ± 0.086 | 0.545 ± 0.050 / 0.538 ± 0.033 | 0.0335 / 0.0610 |
| Grocery | image | 0.465 ± 0.060 | 0.590 ± 0.022 | 0.590 ± 0.005 / 0.565 ± 0.016 | 0.0428 / 0.0501 |

| Dataset | Mean S output norm | Mean D output norm | Mean S–D cosine | ||e|| | ||r|| | ||z|| | cos(e,r) | dot(r,e)/(||e||²+eps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 3.098 | 3.414 | +0.039 | 13.783 | 4.435 | 14.951 | +0.126 | +0.031 |
| Grocery | 3.389 | 1.937 | +0.022 | 8.270 | 15.120 | 20.378 | +0.457 | +0.842 |

Selected-run route, S/D, representation, and module update summaries are retained in `diagnostics_summary.json` and raw epoch logs. Router values are health diagnostics only, not true roles, information fractions, causal utility, or evidence of gain. The best-step optimizer records include pre/post-clip gradients, parameter norms, actual relative updates, global pre-clip norm, and clip scale.

## Interpretation and limitations

R4 did not clear the frozen minimum validation/resource bar against both the topology-free MLP and wide ordinary graph control. No extra mechanism family was trained in Stage B.
S/D encode transformed sender values and transformed edge differences before aggregation; they do not establish homophily/heterophily roles. J/T/V are processing paths, not identifiable shared/unique information. The center path is an explicit coefficient-one route through learned 128-D adapters and can still be counteracted by the correction.
Potential overfitting, optimizer interactions, parameter-count versus compute mismatch, fixed split reuse, earlier exposure of the development graphs, and small descriptive seed count limit interpretation. No causal mechanism is claimed. R6 is a paper-style reference, not a complete author implementation reproduction.
No test labels were indexed, no test predictions/logits were produced, no LP or other NC datasets were run, and no post-result hyperparameter/module search was made.

## Evidence locations

- Frozen protocol and lock: `docs/prior_context_pilot_v3/`.
- Aggregated results, selection, registry, and hashes: `reports/prior_context_pilot_v3/`.
- Ignored raw checkpoints, per-epoch train/validation logs, and train/validation argmax only: `outputs/prior_context_pilot_v3/`.
