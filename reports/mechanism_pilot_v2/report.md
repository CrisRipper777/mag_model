# MAG mechanism pilot v2 report

## Scope and evidence status

- The original v1 primary matrix remains valid and unchanged.
- The v1.1 ordinary-J_NODE context permutation diagnostic was ineffective because the permutation was only consumed by J_NODE_SHUFFLE; the frozen v1.1 train/validation intervention is the valid context diagnostic.
- This new report covers Movies and Grocery train/validation only. No LP and no test evaluation were run.
- Movies and Grocery are development graphs. The three seeds are descriptive; no significance claim is made.

## Selected validation results

Entries are mean ± population standard deviation across the available selected seeds (42/43/44), in percent. Seed 42 reuses its selected calibration checkpoint.

| Dataset | Scheme | Validation accuracy | Validation Macro-F1 | Train–validation Acc gap | Train–validation Macro-F1 gap |
|---|---|---:|---:|---:|---:|
| Movies | P0 | 54.37 ± 0.27 | 40.54 ± 1.26 | +17.13 pp | +22.37 pp |
| Movies | P1 | 53.71 ± 0.06 | 39.99 ± 0.67 | +17.37 pp | +22.10 pp |
| Movies | P2 | 53.35 ± 0.25 | 40.29 ± 2.80 | +17.37 pp | +22.38 pp |
| Movies | P3 | 53.58 ± 0.33 | 38.10 ± 0.80 | +23.45 pp | +26.45 pp |
| Movies | P4 | 53.46 ± 0.36 | 40.36 ± 2.56 | +19.60 pp | +23.67 pp |
| Movies | P5 | 53.79 ± 0.55 | 40.32 ± 1.18 | +22.66 pp | +24.37 pp |
| Movies | P6 | 53.17 ± 0.12 | 39.55 ± 0.98 | +24.52 pp | +27.69 pp |
| Grocery | P0 | 81.10 ± 0.21 | 71.75 ± 0.87 | +15.36 pp | +20.44 pp |
| Grocery | P1 | 81.03 ± 0.22 | 72.18 ± 0.99 | +14.33 pp | +17.61 pp |
| Grocery | P2 | 80.90 ± 0.21 | 71.57 ± 1.27 | +15.21 pp | +19.46 pp |
| Grocery | P3 | 80.61 ± 0.02 | 72.21 ± 0.87 | +16.81 pp | +21.65 pp |
| Grocery | P4 | 80.41 ± 0.23 | 71.83 ± 1.22 | +16.14 pp | +21.45 pp |
| Grocery | P5 | 81.06 ± 0.05 | 72.69 ± 0.36 | +16.83 pp | +22.38 pp |
| Grocery | P6 | 81.21 ± 0.50 | 72.73 ± 1.05 | +15.36 pp | +20.86 pp |

## Seed-level train/validation records

| Dataset | Scheme | Seed | LR | Train Acc | Train Macro-F1 | Train CE | Val Acc | Val Macro-F1 | Val CE | Best epoch | Seconds | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Grocery | P0 | 42 | 0.0003 | 94.15 | 87.02 | 0.2397 | 80.94 | 70.91 | 0.7260 | 121 | 6.8 | 1.43 |
| Grocery | P0 | 43 | 0.0003 | 96.29 | 91.79 | 0.1615 | 81.41 | 71.40 | 0.7679 | 150 | 7.9 | 1.43 |
| Grocery | P0 | 44 | 0.0003 | 98.94 | 97.76 | 0.0715 | 80.97 | 72.95 | 0.8494 | 199 | 9.3 | 1.43 |
| Grocery | P1 | 42 | 0.0003 | 96.81 | 92.66 | 0.1472 | 81.32 | 73.54 | 0.7549 | 145 | 10.9 | 1.87 |
| Grocery | P1 | 43 | 0.0003 | 95.70 | 90.23 | 0.1766 | 81.00 | 71.78 | 0.7645 | 138 | 10.6 | 1.87 |
| Grocery | P1 | 44 | 0.0003 | 93.58 | 86.46 | 0.2512 | 80.79 | 71.21 | 0.7350 | 109 | 8.9 | 1.87 |
| Grocery | P2 | 42 | 0.0003 | 95.63 | 90.39 | 0.1850 | 81.20 | 71.29 | 0.7736 | 122 | 15.7 | 4.15 |
| Grocery | P2 | 43 | 0.0003 | 96.79 | 92.92 | 0.1506 | 80.73 | 73.25 | 0.8083 | 144 | 17.9 | 4.15 |
| Grocery | P2 | 44 | 0.0003 | 95.91 | 89.80 | 0.1703 | 80.76 | 70.18 | 0.8002 | 140 | 17.4 | 4.15 |
| Grocery | P3 | 42 | 0.0003 | 96.78 | 92.99 | 0.1473 | 80.59 | 72.22 | 0.8226 | 134 | 20.4 | 4.59 |
| Grocery | P3 | 43 | 0.0003 | 97.27 | 93.20 | 0.1237 | 80.64 | 71.14 | 0.8311 | 137 | 20.9 | 4.59 |
| Grocery | P3 | 44 | 0.0003 | 98.24 | 95.41 | 0.0923 | 80.61 | 73.28 | 0.8633 | 153 | 22.7 | 4.59 |
| Grocery | P4 | 42 | 0.001 | 99.95 | 99.96 | 0.0028 | 80.44 | 73.11 | 1.4125 | 162 | 22.0 | 4.38 |
| Grocery | P4 | 43 | 0.001 | 97.42 | 94.69 | 0.1120 | 80.67 | 72.19 | 0.8901 | 87 | 13.6 | 4.38 |
| Grocery | P4 | 44 | 0.001 | 92.28 | 85.20 | 0.2748 | 80.12 | 70.19 | 0.7727 | 58 | 10.4 | 4.38 |
| Grocery | P5 | 42 | 0.0003 | 99.64 | 99.48 | 0.0309 | 81.08 | 72.74 | 0.9656 | 200 | 28.9 | 4.59 |
| Grocery | P5 | 43 | 0.0003 | 96.52 | 91.71 | 0.1487 | 81.11 | 72.22 | 0.7788 | 126 | 20.0 | 4.59 |
| Grocery | P5 | 44 | 0.0003 | 97.52 | 94.03 | 0.1084 | 81.00 | 73.11 | 0.8253 | 147 | 22.6 | 4.59 |
| Grocery | P6 | 42 | 0.001 | 96.05 | 92.49 | 0.1471 | 81.58 | 71.43 | 0.8200 | 76 | 11.1 | 4.16 |
| Grocery | P6 | 43 | 0.001 | 97.64 | 96.10 | 0.0991 | 80.50 | 74.01 | 0.8684 | 95 | 13.1 | 4.16 |
| Grocery | P6 | 44 | 0.001 | 96.04 | 92.18 | 0.1459 | 81.55 | 72.75 | 0.8120 | 80 | 11.6 | 4.16 |
| Movies | P0 | 42 | 0.001 | 74.71 | 68.82 | 0.7453 | 54.62 | 41.11 | 1.4954 | 66 | 3.6 | 1.45 |
| Movies | P0 | 43 | 0.001 | 68.70 | 58.42 | 0.9027 | 54.50 | 38.79 | 1.4436 | 55 | 4.0 | 1.45 |
| Movies | P0 | 44 | 0.001 | 71.10 | 61.49 | 0.8828 | 53.99 | 41.72 | 1.4449 | 57 | 3.3 | 1.45 |
| Movies | P1 | 42 | 0.001 | 65.71 | 56.07 | 1.0037 | 53.78 | 40.93 | 1.4353 | 50 | 4.7 | 1.87 |
| Movies | P1 | 43 | 0.001 | 75.77 | 68.94 | 0.7271 | 53.72 | 39.54 | 1.5410 | 64 | 5.5 | 1.87 |
| Movies | P1 | 44 | 0.001 | 71.75 | 61.25 | 0.8471 | 53.63 | 39.48 | 1.4704 | 58 | 5.2 | 1.87 |
| Movies | P2 | 42 | 0.001 | 65.32 | 53.18 | 1.0009 | 53.48 | 37.25 | 1.4590 | 49 | 8.3 | 4.51 |
| Movies | P2 | 43 | 0.001 | 75.62 | 69.08 | 0.7397 | 53.00 | 39.61 | 1.5305 | 62 | 9.6 | 4.51 |
| Movies | P2 | 44 | 0.001 | 71.23 | 65.74 | 0.8507 | 53.57 | 44.01 | 1.4758 | 55 | 8.9 | 4.51 |
| Movies | P3 | 42 | 0.0003 | 70.36 | 54.78 | 0.8981 | 54.02 | 37.40 | 1.4584 | 81 | 14.1 | 4.93 |
| Movies | P3 | 43 | 0.0003 | 80.36 | 69.64 | 0.6125 | 53.48 | 37.68 | 1.6129 | 124 | 19.4 | 4.93 |
| Movies | P3 | 44 | 0.0003 | 80.38 | 69.23 | 0.6260 | 53.24 | 39.22 | 1.6103 | 123 | 19.3 | 4.93 |
| Movies | P4 | 42 | 0.001 | 69.15 | 57.97 | 0.9072 | 53.30 | 38.97 | 1.4707 | 52 | 9.8 | 4.73 |
| Movies | P4 | 43 | 0.001 | 73.45 | 64.51 | 0.7936 | 53.12 | 38.16 | 1.4916 | 63 | 11.1 | 4.73 |
| Movies | P4 | 44 | 0.001 | 76.58 | 69.61 | 0.7051 | 53.96 | 43.95 | 1.5433 | 65 | 11.3 | 4.73 |
| Movies | P5 | 42 | 0.0003 | 74.61 | 60.56 | 0.7785 | 54.47 | 38.66 | 1.4883 | 99 | 16.6 | 4.93 |
| Movies | P5 | 43 | 0.0003 | 75.78 | 64.91 | 0.7610 | 53.78 | 40.98 | 1.5007 | 106 | 17.6 | 4.93 |
| Movies | P5 | 44 | 0.0003 | 78.97 | 68.57 | 0.6810 | 53.12 | 41.31 | 1.5736 | 106 | 17.6 | 4.93 |
| Movies | P6 | 42 | 0.0003 | 80.35 | 69.90 | 0.6335 | 53.15 | 40.61 | 1.6446 | 115 | 15.5 | 4.51 |
| Movies | P6 | 43 | 0.0003 | 79.24 | 69.41 | 0.6525 | 53.33 | 38.25 | 1.6013 | 117 | 15.8 | 4.51 |
| Movies | P6 | 44 | 0.0003 | 73.49 | 62.42 | 0.8240 | 53.03 | 39.80 | 1.5242 | 87 | 12.7 | 4.51 |

## Learning-rate selection

| Dataset | Scheme | Selected LR | Seed-42 val Acc | Seed-42 val Macro-F1 | Candidates (LR: Acc / Macro-F1) |
|---|---|---:|---:|---:|---|
| Movies | P0 | 0.001 | 54.62 | 41.11 | 0.001: 54.62/41.11, 0.003: 54.29/44.12, 0.0003: 53.96/35.66 |
| Movies | P1 | 0.001 | 53.78 | 40.93 | 0.001: 53.78/40.93, 0.003: 53.63/42.63, 0.0003: 53.39/40.03 |
| Movies | P2 | 0.001 | 53.48 | 37.25 | 0.001: 53.48/37.25, 0.0003: 53.39/40.36, 0.003: 53.18/39.21 |
| Movies | P3 | 0.0003 | 54.02 | 37.40 | 0.0003: 54.02/37.40, 0.001: 53.78/39.64, 0.003: 52.61/39.30 |
| Movies | P4 | 0.001 | 53.30 | 38.97 | 0.001: 53.30/38.97, 0.0003: 53.12/38.04, 0.003: 52.79/36.99 |
| Movies | P5 | 0.0003 | 54.47 | 38.66 | 0.0003: 54.47/38.66, 0.001: 54.35/40.10, 0.003: 52.76/41.44 |
| Movies | P6 | 0.0003 | 53.15 | 40.61 | 0.0003: 53.15/40.61, 0.001: 52.67/38.10, 0.003: 52.40/40.96 |
| Grocery | P0 | 0.0003 | 80.94 | 70.91 | 0.0003: 80.94/70.91, 0.001: 80.70/70.88, 0.003: 80.18/71.11 |
| Grocery | P1 | 0.0003 | 81.32 | 73.54 | 0.0003: 81.32/73.54, 0.001: 81.00/72.53, 0.003: 80.29/72.27 |
| Grocery | P2 | 0.0003 | 81.20 | 71.29 | 0.0003: 81.20/71.29, 0.001: 80.97/72.83, 0.003: 79.82/73.08 |
| Grocery | P3 | 0.0003 | 80.59 | 72.22 | 0.0003: 80.59/72.22, 0.001: 80.29/70.08, 0.003: 79.85/72.46 |
| Grocery | P4 | 0.001 | 80.44 | 73.11 | 0.001: 80.44/73.11, 0.0003: 80.32/71.97, 0.003: 79.62/71.49 |
| Grocery | P5 | 0.0003 | 81.08 | 72.74 | 0.0003: 81.08/72.74, 0.001: 80.97/73.00, 0.003: 80.85/73.18 |
| Grocery | P6 | 0.001 | 81.58 | 71.43 | 0.001: 81.58/71.43, 0.0003: 81.08/71.63, 0.003: 80.47/71.26 |

## Matched validation differences

Differences are right-hand scheme minus the comparator within matched dataset/seed cells. `n/6` counts positive accuracy cells; these are descriptive paired differences.

| Comparison | Movies Δ Acc / Macro-F1 (pp) | Grocery Δ Acc / Macro-F1 (pp) | All cells Δ Acc / Macro-F1 (pp) | Acc positive cells |
|---|---:|---:|---:|---:|
| P2−P0: dual-axis with wide experts | -1.02 / -0.25 | -0.20 / -0.18 | -0.61 / -0.21 | 1/6 |
| P3−P1: dual-axis with conditional experts | -0.13 / -1.88 | -0.42 / +0.04 | -0.27 / -0.92 | 1/6 |
| P3−P2: conditional experts vs width-matched wide | +0.23 / -2.19 | -0.28 / +0.64 | -0.03 / -0.77 | 2/6 |
| P3−P4: learned vs retrained static specialists | +0.12 / -2.26 | +0.20 / +0.38 | +0.16 / -0.94 | 4/6 |
| P5−P3: source supervision within conditional experts | +0.21 / +2.22 | +0.45 / +0.48 | +0.33 / +1.35 | 5/6 |
| P6−P2: source supervision within wide experts | -0.18 / -0.74 | +0.31 / +1.16 | +0.07 / +0.21 | 3/6 |

## Cost and training diagnostics

Cost averages use selected seed-42/43/44 runs over both graphs. Expert width is matched by active expert MLP parameters (1025-wide MLP vs four 256-hidden expert MLPs); this does not equalize compute.

| Scheme | Active backbone parameters | NC main head | Source heads | Mean seconds / run | Mean peak CUDA GiB |
|---|---:|---:|---:|---:|---:|
| P0 | 1152130 | 2580 | 0 | 5.8 | 1.44 |
| P1 | 1203868 | 2580 | 0 | 7.6 | 1.87 |
| P2 | 1394899 | 2580 | 0 | 13.0 | 4.33 |
| P3 | 1446621 | 2580 | 0 | 19.5 | 4.76 |
| P4 | 1394897 | 2580 | 0 | 13.0 | 4.55 |
| P5 | 1446621 | 2580 | 5160 | 20.5 | 4.76 |
| P6 | 1394899 | 2580 | 5160 | 13.3 | 4.33 |

Formal attempts: 70 succeeded, 0 failed (planned maximum 70).

## Findings and limits

- The dual-axis change did not improve validation accuracy against the same-modality controls in this matrix: P2−P0 averaged −0.61 pp (positive in 1/6 paired cells), and P3−P1 averaged −0.27 pp (positive in 1/6). The observed means do not support paying the interaction cost for an accuracy gain here.
- Conditional experts had effectively tied accuracy but lower Macro-F1 than the wide control: P3−P2 was −0.03 pp Acc and −0.77 pp Macro-F1, with positive Acc in 2/6 cells. Against the separately retrained static mixture, P3−P4 was +0.16 pp Acc but −0.94 pp Macro-F1; this small mixed result does not establish a general routing benefit.
- Source supervision was more encouraging within conditional experts (P5−P3: +0.33 pp Acc, +1.35 pp Macro-F1; positive Acc in 5/6 cells) than within wide experts (P6−P2: +0.07 pp Acc, +0.21 pp Macro-F1; positive Acc in 3/6). With three descriptive seeds and two development graphs, this is a candidate effect for later confirmation, not a broad auxiliary-loss conclusion.
- The chosen models show sizeable train–validation gaps: selected-cell Acc gaps range from 14.33 to 24.52 pp and Macro-F1 gaps from 17.61 to 27.69 pp. This is consistent with overfitting pressure, but these results do not establish that overfitting caused the weak validation differences.
- At the selected checkpoints, mean specialist-router entropy (nats; averaged across the six dataset/seed cells for each scheme) was P1: 1.078/1.050 on layer 1 and 1.021/0.926 on layer 2 for text/image; P3: 1.081/1.068 and 0.842/0.940; P5: 1.048/1.062 and 0.802/0.929. P4 uses fixed 1/3 specialist coefficients; P0/P2/P6 have one wide expert and no specialist router. These are routing summaries, not semantic roles or evidence of utility.
- For dual-axis schemes, the selected-checkpoint sender-modality coefficient means for text-receiver/image-receiver states respectively (columns are text sender / image sender) were P2: `0.073/0.927`, `0.078/0.922`; P3: `0.064/0.936`, `0.066/0.934`; P4: `0.089/0.911`, `0.090/0.910`; P5: `0.087/0.913`, `0.089/0.911`; P6: `0.166/0.834`, `0.169/0.831`. Text still enters link context, text receiver states, residual pathways, and FusionMLP; these coefficients are not information fractions or causal effects.
- Every formal cell recorded epoch 1/10/30 and best-step diagnostics. At best epoch, all active parameter groups had finite nonzero gradients and nonzero updates; no non-finite-gradient stop or failed run occurred. The logged incoming-edge attention mean was 0.11185 across all dual-axis schemes because each destination group sums to one on the same graph; the recorded maximum was 1.0 because degree-one groups exist. Those two aggregates do not distinguish learned attention quality. Full edge weights were not persisted.
- The dual-axis effect is judged against P0/P1 using paired validation changes above. It is not inferred from attention or router differences.
- Conditional experts are compared with both the active-parameter-matched wide control and a separately retrained uniform-specialist model. A positive validation difference in this small matrix is descriptive evidence for this design and split only.
- P5−P3 and P6−P2 distinguish source supervision in the conditional and wide architectures. A change in both would support a general auxiliary-supervision effect within these tested controls, not a causal account of the earlier probe result.
- Low route weights on a path, if observed, do not establish zero use or a modality information fraction. Text may affect relation context and the final FusionMLP even when a direct branch is small.
- The study cannot identify whether any shortfall comes from overfitting, shared expert functions, competitive routing, interaction optimization, or another factor. No causal mechanism claim is made.
- These results do not determine the final five-NC or LP model. No new model modules or tuning grids are added here.

Raw checkpoints, epoch logs, and train/validation argmax are under ignored `outputs/mechanism_pilot_v2/`. Aggregated train/validation confusion matrices are retained in `run_results.jsonl`.
