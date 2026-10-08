# Coordinated graph experts: results

The frozen matrix completed 162 calibration and 54 confirmation cells (216 total). Results below use only confirmation seeds 113-115; calibration performance is reported separately in `summary.csv` and the selection lock.

## Protocol amendment and implementation provenance

Calibration and its 18-entry selection lock were completed under `protocol_lock_v1.json`. The first confirmation pass completed all training updates, but 36 routed-model cells could not finalize because the gradient summary omitted unused parameter slots before comparing vectors. The 18 A0/N0 cells were unaffected and remain under the original lock. The diagnostic now fills unused slots with zeros in parameter order; the 36 routed cells were rerun as retained retry attempts under the amended lock. The amendment changes diagnostic aggregation only; model computation, losses, optimizer updates, seeds, data, splits, and selected recipes are unchanged.

The first retry dispatch hit a retry-ID parsing bug and exited before constructing a model; no run started under the superseded `protocol_lock_v2.json`, which is preserved. All 36 routed retries then completed under `protocol_lock_v3.json`. The first audit invocation resolved unconfigured placeholders and marked the audit incomplete; the audit-only correction in the final lock keys expected cells from the selection lock. The audit records each successful cell's source commit and lock hash in `audit_result.json`; the final 54-cell set therefore has 18 valid v1 records and 36 valid v3 records. No training was run under the audit-only final lock.

Validation is the repeatedly used development split, not an independent benchmark. Differences are descriptive across three seeds; no significance, generalization, or causal claim is made.

## Confirmation results

Acc and macro-F1 are mean ± population SD over the three training seeds. Train means describe fit only.

| Graph | Model | Val Acc | Val macro-F1 | Train Acc | Train macro-F1 | Selected LR / WD | Best updates | Max-budget share | Parameters (encoder + head) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Movies | A0 | 0.5351 ± 0.0038 | 0.4334 ± 0.0140 | 0.6979 ± 0.0326 | 0.6470 ± 0.0444 | 0.003 / 0.0001 | 36,46,41 | 0.00 | 893,056 + 20,560 = 913,616 |
| Movies | N0 | 0.5339 ± 0.0025 | 0.4385 ± 0.0213 | 0.7538 ± 0.0869 | 0.7088 ± 0.1079 | 0.003 / 0.0001 | 56,41,41 | 0.00 | 744,832 + 12,820 = 757,652 |
| Movies | M0 | 0.5346 ± 0.0014 | 0.4162 ± 0.0136 | 0.7808 ± 0.0586 | 0.7295 ± 0.0745 | 0.0003 / 0.001 | 166,146,126 | 0.00 | 812,048 + 12,820 = 824,868 |
| Movies | M1 | 0.5342 ± 0.0009 | 0.4286 ± 0.0102 | 0.8412 ± 0.0565 | 0.8070 ± 0.0686 | 0.0003 / 0.001 | 161,151,181 | 0.00 | 812,048 + 17,980 = 830,028 |
| Movies | P0 | 0.5348 ± 0.0018 | 0.4165 ± 0.0135 | 0.7809 ± 0.0585 | 0.7292 ± 0.0749 | 0.0003 / 0.001 | 166,146,126 | 0.00 | 814,096 + 12,820 = 826,916 |
| Movies | P1 | 0.5342 ± 0.0022 | 0.4323 ± 0.0115 | 0.8532 ± 0.0574 | 0.8207 ± 0.0695 | 0.0003 / 0.0001 | 171,151,181 | 0.00 | 814,096 + 17,980 = 832,076 |
| Grocery | A0 | 0.8070 ± 0.0034 | 0.7356 ± 0.0089 | 0.9995 ± 0.0004 | 0.9996 ± 0.0004 | 0.001 / 0.001 | 486,141,391 | 0.00 | 893,056 + 20,560 = 913,616 |
| Grocery | N0 | 0.8242 ± 0.0007 | 0.7515 ± 0.0046 | 0.9998 ± 0.0000 | 0.9999 ± 0.0000 | 0.003 / 0.01 | 216,216,301 | 0.00 | 744,832 + 12,820 = 757,652 |
| Grocery | M0 | 0.8257 ± 0.0012 | 0.7526 ± 0.0025 | 0.9998 ± 0.0000 | 0.9998 ± 0.0000 | 0.003 / 0.0001 | 311,346,161 | 0.00 | 812,048 + 12,820 = 824,868 |
| Grocery | M1 | 0.8267 ± 0.0031 | 0.7579 ± 0.0087 | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 | 0.001 / 0.01 | 161,281,481 | 0.00 | 812,048 + 17,980 = 830,028 |
| Grocery | P0 | 0.8266 ± 0.0015 | 0.7545 ± 0.0038 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.003 / 0.01 | 226,506,161 | 0.00 | 814,096 + 12,820 = 826,916 |
| Grocery | P1 | 0.8264 ± 0.0030 | 0.7597 ± 0.0042 | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 | 0.001 / 0.001 | 161,256,581 | 0.00 | 814,096 + 17,980 = 832,076 |
| Reddit-S | A0 | 0.9604 ± 0.0014 | 0.9234 ± 0.0026 | 0.9997 ± 0.0000 | 0.9992 ± 0.0000 | 0.003 / 0.0001 | 166,116,246 | 0.00 | 893,056 + 20,560 = 913,616 |
| Reddit-S | N0 | 0.9574 ± 0.0004 | 0.9121 ± 0.0024 | 0.9997 ± 0.0000 | 0.9992 ± 0.0000 | 0.001 / 0.0001 | 486,286,206 | 0.00 | 744,832 + 12,820 = 757,652 |
| Reddit-S | M0 | 0.9554 ± 0.0012 | 0.9098 ± 0.0027 | 0.9997 ± 0.0000 | 0.9991 ± 0.0001 | 0.0003 / 0.0001 | 806,266,416 | 0.00 | 812,048 + 12,820 = 824,868 |
| Reddit-S | M1 | 0.9595 ± 0.0010 | 0.9133 ± 0.0017 | 0.9997 ± 0.0000 | 0.9992 ± 0.0000 | 0.003 / 0.01 | 421,371,201 | 0.00 | 812,048 + 17,980 = 830,028 |
| Reddit-S | P0 | 0.9551 ± 0.0010 | 0.9082 ± 0.0007 | 0.9996 ± 0.0000 | 0.9990 ± 0.0001 | 0.0003 / 0.01 | 231,291,451 | 0.00 | 814,096 + 12,820 = 826,916 |
| Reddit-S | P1 | 0.9569 ± 0.0020 | 0.9120 ± 0.0026 | 0.9971 ± 0.0036 | 0.9940 ± 0.0073 | 0.0003 / 0.01 | 631,161,551 | 0.00 | 814,096 + 17,980 = 832,076 |

## Calibration selections

Each recipe was selected only from its nine seed-112 cells by validation correct count, macro-F1, then lower config index.

| Graph | Model | Config | LR / WD | Validation correct | Validation Acc | Validation macro-F1 | Best update |
|---|---|---:|---:|---:|---:|---:|---:|
| Grocery | A0 | 4 | 0.001 / 0.001 | 2787 / 3415 | 0.8161 | 0.7443 | 246 |
| Grocery | M0 | 6 | 0.003 / 0.0001 | 2786 / 3415 | 0.8158 | 0.7372 | 291 |
| Grocery | M1 | 5 | 0.001 / 0.01 | 2807 / 3415 | 0.8220 | 0.7486 | 701 |
| Grocery | N0 | 8 | 0.003 / 0.01 | 2784 / 3415 | 0.8152 | 0.7383 | 136 |
| Grocery | P0 | 8 | 0.003 / 0.01 | 2786 / 3415 | 0.8158 | 0.7347 | 216 |
| Grocery | P1 | 4 | 0.001 / 0.001 | 2811 / 3415 | 0.8231 | 0.7507 | 701 |
| Movies | A0 | 6 | 0.003 / 0.0001 | 1800 / 3334 | 0.5399 | 0.4423 | 41 |
| Movies | M0 | 1 | 0.0003 / 0.001 | 1798 / 3334 | 0.5393 | 0.4378 | 136 |
| Movies | M1 | 1 | 0.0003 / 0.001 | 1795 / 3334 | 0.5384 | 0.4283 | 126 |
| Movies | N0 | 6 | 0.003 / 0.0001 | 1804 / 3334 | 0.5411 | 0.4352 | 46 |
| Movies | P0 | 1 | 0.0003 / 0.001 | 1795 / 3334 | 0.5384 | 0.4372 | 136 |
| Movies | P1 | 0 | 0.0003 / 0.0001 | 1796 / 3334 | 0.5387 | 0.4284 | 126 |
| Reddit-S | A0 | 6 | 0.003 / 0.0001 | 3065 / 3179 | 0.9641 | 0.9285 | 311 |
| Reddit-S | M0 | 0 | 0.0003 / 0.0001 | 3044 / 3179 | 0.9575 | 0.9111 | 446 |
| Reddit-S | M1 | 8 | 0.003 / 0.01 | 3044 / 3179 | 0.9575 | 0.9085 | 346 |
| Reddit-S | N0 | 3 | 0.001 / 0.0001 | 3049 / 3179 | 0.9591 | 0.9132 | 771 |
| Reddit-S | P0 | 2 | 0.0003 / 0.01 | 3044 / 3179 | 0.9575 | 0.9120 | 581 |
| Reddit-S | P1 | 2 | 0.0003 / 0.01 | 3046 / 3179 | 0.9582 | 0.9124 | 916 |

## Paired validation differences

Each cell is the mean same-seed difference over seeds 113-115; rows preserve graph-specific changes. Positive values favor the left model. The equal-graph summary weights each graph equally.

| Contrast | Graph | Δ Acc | Δ macro-F1 |
|---|---|---:|---:|
| M0_vs_A0 | Movies | -0.0005 | -0.0172 |
| M0_vs_A0 | Grocery | +0.0186 | +0.0170 |
| M0_vs_A0 | Reddit-S | -0.0049 | -0.0137 |
| M0_vs_A0 | Equal-graph mean | +0.0044 | -0.0046 |
| M1_vs_A0 | Movies | -0.0009 | -0.0048 |
| M1_vs_A0 | Grocery | +0.0197 | +0.0223 |
| M1_vs_A0 | Reddit-S | -0.0008 | -0.0102 |
| M1_vs_A0 | Equal-graph mean | +0.0060 | +0.0024 |
| P0_vs_A0 | Movies | -0.0003 | -0.0169 |
| P0_vs_A0 | Grocery | +0.0196 | +0.0188 |
| P0_vs_A0 | Reddit-S | -0.0052 | -0.0152 |
| P0_vs_A0 | Equal-graph mean | +0.0047 | -0.0044 |
| P1_vs_A0 | Movies | -0.0009 | -0.0011 |
| P1_vs_A0 | Grocery | +0.0193 | +0.0240 |
| P1_vs_A0 | Reddit-S | -0.0035 | -0.0115 |
| P1_vs_A0 | Equal-graph mean | +0.0050 | +0.0038 |
| M0_vs_N0 | Movies | +0.0007 | -0.0223 |
| M0_vs_N0 | Grocery | +0.0015 | +0.0011 |
| M0_vs_N0 | Reddit-S | -0.0020 | -0.0023 |
| M0_vs_N0 | Equal-graph mean | +0.0001 | -0.0078 |
| P0_vs_M0 | Movies | +0.0002 | +0.0003 |
| P0_vs_M0 | Grocery | +0.0010 | +0.0018 |
| P0_vs_M0 | Reddit-S | -0.0003 | -0.0016 |
| P0_vs_M0 | Equal-graph mean | +0.0003 | +0.0002 |
| M1_vs_M0 | Movies | -0.0004 | +0.0124 |
| M1_vs_M0 | Grocery | +0.0011 | +0.0053 |
| M1_vs_M0 | Reddit-S | +0.0041 | +0.0035 |
| M1_vs_M0 | Equal-graph mean | +0.0016 | +0.0070 |
| P1_vs_P0 | Movies | -0.0006 | +0.0158 |
| P1_vs_P0 | Grocery | -0.0003 | +0.0052 |
| P1_vs_P0 | Reddit-S | +0.0018 | +0.0037 |
| P1_vs_P0 | Equal-graph mean | +0.0003 | +0.0082 |

`(P1-P0)-(M1-M0)` is a descriptive interaction under independently selected recipes, not a fixed-recipe causal interaction. Its per-graph and per-seed values are in `paired_differences.json`.

## Optimization and cost

| Model | Confirmation cells at 1000 updates | Median stable update (ms) | Median optimizer-update time (s) | Validation time total (s) | Diagnostic time total (s) | Max allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0/9 | 51.165 | 15.3 | 18.6 | 0.2 | 3.79 / 4.14 |
| N0 | 0/9 | 34.300 | 12.7 | 11.5 | 2.2 | 1.55 / 1.72 |
| M0 | 0/9 | 67.395 | 28.4 | 21.4 | 8.2 | 1.81 / 2.36 |
| M1 | 0/9 | 70.790 | 25.4 | 19.7 | 9.2 | 1.89 / 2.08 |
| P0 | 0/9 | 76.505 | 32.5 | 20.3 | 9.1 | 2.22 / 2.37 |
| P1 | 0/9 | 83.065 | 27.1 | 25.0 | 12.9 | 2.30 / 2.47 |

Selections at an LR or WD grid boundary: 16/18. Grocery/M0 (LR=0.003, WD=0.0001), Grocery/M1 (LR=0.001, WD=0.01), Grocery/N0 (LR=0.003, WD=0.01), Grocery/P0 (LR=0.003, WD=0.01), Movies/A0 (LR=0.003, WD=0.0001), Movies/M0 (LR=0.0003, WD=0.001), Movies/M1 (LR=0.0003, WD=0.001), Movies/N0 (LR=0.003, WD=0.0001), Movies/P0 (LR=0.0003, WD=0.001), Movies/P1 (LR=0.0003, WD=0.0001), Reddit-S/A0 (LR=0.003, WD=0.0001), Reddit-S/M0 (LR=0.0003, WD=0.0001), Reddit-S/M1 (LR=0.003, WD=0.01), Reddit-S/N0 (LR=0.001, WD=0.0001), Reddit-S/P0 (LR=0.0003, WD=0.01), Reddit-S/P1 (LR=0.0003, WD=0.01)
A selection at a grid boundary leaves optimization unresolved; a run that reaches 1000 updates is also unresolved. See the per-run columns in `summary.csv`.

## Interpretation limits

A0 is the full historical B1 reference, including its padded 768D output and historical task losses; it is not parameter-matched. N0 has no router parameters. M0-N0 also changes parameter count, P0-M0 changes neighbor propagation and coefficients, and M1/P1 were tuned independently. These are comparisons between complete recipes, not isolated causal interventions.

This stage evaluates only full-modal transductive NC on Movies, Grocery, and Reddit-S. The LP decoder was checked only with synthetic endpoint embeddings; no real LP score, test result, or additional-graph result was produced.

The audit is an implementer self-audit. The report's metrics were independently recomputed from saved train/validation IDs, targets, and argmax predictions by the audit script, but the review is not independent peer review.

Raw checkpoints, prediction arrays, and per-update logs remain under ignored `outputs/mvcge_coordinated_nc_v2/`; this repository commit contains concise metrics and hashes only.
