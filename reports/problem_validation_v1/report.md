# MAG NC problem validation v1

Branch `codex/mag-problem-validation-v1`; training source commit `63867331e9d3373f19136e47ab192cc9fb014c97`; final report commit is cited in `manifest.json` and the handoff.
Protocol lock SHA-256: `3b949712c738dd99865f9bdce882cf65f91756e775cd4af6077cbd065d3b368f`. Common LR: `0.003`. Locked test inference used the pinned evaluation source; aggregate-only code commit and hashes are recorded separately in `manifest.json`.
Training attempts: **162** / 180; successful unique training records: **162**; locked primary test evaluations: **150**; missing primary evaluations: **0**.

## Primary NC results

Mean ± population SD across three training seeds (`ddof=0`), in percent. These are descriptive; three seeds do not support significance claims. A dash indicates a variant was outside that dataset's registered matrix or no successful locked evaluation is available.

| Dataset | T_E Acc/F1 | I_E Acc/F1 | J_E Acc/F1 | J_FIXED Acc/F1 | J_GLOBAL Acc/F1 | J_NODE Acc/F1 | J_GLOBAL_WIDE Acc/F1 | J_CONCAT Acc/F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 45.52 ± 0.38 / 28.10 ± 1.00 | 47.97 ± 0.39 / 34.17 ± 1.23 | 50.55 ± 0.40 / 38.17 ± 1.72 | 54.12 ± 0.39 / 45.10 ± 1.07 | 54.42 ± 0.75 / 45.18 ± 1.28 | 52.97 ± 0.20 / 42.56 ± 0.24 | 53.79 ± 0.24 / 44.47 ± 1.32 | 53.63 ± 0.32 / 42.74 ± 0.44 |
| Toys | 69.24 ± 0.11 / 66.69 ± 0.57 | 65.82 ± 0.18 / 62.09 ± 0.50 | 72.96 ± 0.21 / 69.82 ± 0.34 | 78.57 ± 0.33 / 75.46 ± 0.86 | 78.28 ± 0.52 / 75.48 ± 0.31 | 75.78 ± 0.61 / 72.70 ± 0.56 | 78.23 ± 0.27 / 74.95 ± 0.06 | 76.69 ± 0.13 / 73.03 ± 0.32 |
| Grocery | 75.18 ± 0.20 / 66.80 ± 0.86 | 64.49 ± 0.24 / 55.21 ± 0.95 | 77.53 ± 0.56 / 67.15 ± 1.09 | 82.04 ± 0.20 / 73.10 ± 0.49 | 81.76 ± 0.19 / 72.45 ± 0.21 | 77.70 ± 0.60 / 68.51 ± 1.13 | 81.41 ± 0.19 / 72.31 ± 0.42 | 80.07 ± 0.14 / 70.10 ± 0.70 |
| Reddit-S | 67.70 ± 0.39 / 61.08 ± 0.60 | 89.71 ± 0.21 / 84.20 ± 0.15 | 92.37 ± 0.01 / 86.29 ± 0.20 | 95.21 ± 0.08 / 90.54 ± 0.68 | 95.32 ± 0.08 / 91.08 ± 0.33 | 94.90 ± 0.99 / 90.78 ± 0.95 | 95.16 ± 0.11 / 91.01 ± 0.13 | 94.89 ± 0.37 / 90.42 ± 0.79 |
| ele-fashion | 87.30 ± 0.06 / 75.23 ± 0.05 | 82.73 ± 0.24 / 66.24 ± 1.15 | 86.89 ± 0.16 / 74.25 ± 0.17 | 86.13 ± 0.19 / 73.49 ± 1.21 | 87.11 ± 0.11 / 74.81 ± 0.27 | 87.23 ± 0.07 / 74.84 ± 0.25 | 87.06 ± 0.08 / 74.74 ± 0.46 | 87.25 ± 0.04 / 74.38 ± 0.44 |

Each table cell is test Accuracy / Macro-F1, with mean ± ddof0 SD. The complete per-seed and validation table, full confusion matrices, and class schemas are in `metrics_summary.json` and `run_results.jsonl`.

## Calibration

The single LR was selected by the registered equal-weight mean best-validation-accuracy rule across six Movies/Grocery × model cells. Selected LR: **0.003**. Scores and all 18 validation-only runs are in `lr_selection.json`, `run_registry.json`, and `run_results.jsonl`; calibration has no test metrics.

## Q1: modality and neighborhood signal

Across the five graphs with equal graph weighting, J_E test Accuracy was +2.181 pp versus the better of T_E and I_E. Across dataset/seed cells, the mean rates of text-only-correct, image-only-correct, both-wrong, and prediction disagreement were 0.125729069534797, 0.1372993528417067, 0.17282277946552868, and 0.3465608435234327. On Movies/Grocery the equal-cell paired test Accuracy change for T_M−T_E and I_M−I_E pooled together was +6.370 pp. These are descriptive results; single-modality controls also remove an adapter and its parameters, and dual-modality gains cannot be assigned entirely to complementarity.
`error_complementarity.json` reports validation and locked-test both-correct/text-only/image-only/both-wrong cells, disagreement, and the label-using oracle upper bound for every graph/seed. The oracle is not deployable. `geometry_summary.json` reports validation-only bucket comparisons using train-only degree and within-modality raw ego/neighborhood cosine thresholds. Bucket confusion matrices are losslessly compressed with explicit shape and class order.

## Q2: node-conditioned fusion

NODE-versus-capacity-control test screening result: **weak/mixed or incomplete**. Graphs passing both GLOBAL_WIDE and CONCAT checks: `[]`. The registered threshold is a paired mean Acc gain ≥0.3 pp, positive in at least 2/3 seeds, and mean Macro-F1 no worse than −0.3 pp, for at least three of five graphs against each control on the same graph set. This is an engineering screen, not statistical evidence. Per-graph/per-seed deltas and paired 2×2 counts are in `paired_comparisons.json`.

## Q3: direct center paths

On the six Movies/Grocery test seed cells, J_FIXED−J_N paired Accuracy averaged +6.645 pp and was positive in 6/6 cells. `paired_comparisons.json` contains their full paired correctness counts; Macro-F1 is in `metrics_summary.json`. This supports or weakens a center-path contribution only for the tested design and split protocol; it does not prove uniquely extracted information.

## Capacity, router, and cost

The registry records exact parameter counts including the shared external NC head and exhaustive widths. Any match outside 1% is marked unmatched. `router_summary.json` reports validation weights, argmax frequencies, entropy, degree/isolation bucket means, mask violations, and ten separate fixed-context-permutation results per J_NODE run. These statistics do not identify causal utility or mature MoE expert loads.
Recorded training plus validation/test inference cost: **0.095 GPU-hours**; training-only **0.064 GPU-hours**; peak allocated memory **2143803904 bytes**. The 12-hour planning target was soft; attempt cap was 180.

## Completion and reproducibility

Primary registered slots: 150; missing or unevaluated: 0. Training attempts include retries. Failed, missing, and superseded records remain listed in `failures.json`; test metrics remain null where no locked evaluation succeeded.
Raw evidence is local and ignored by Git under `/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation_v1`. Checkpoints and per-node argmax predictions are content-hashed in `manifest.json`; predictions contain class IDs only and are retained locally to resume aggregate paired analyses.

Commands: see `manifest.json` for calibration, main-matrix, locked-evaluation, and summary commands. The recorded locked-evaluation resume command is tied to the pinned source hashes; on the final checkout, `--summary-only` rebuilds aggregates without rerunning test inference.
