# M0.2 Contextual Relation: Six-Run NC Pilot

## Scope and setup

M0.1 is the **pair-level relation-conditioned transport prototype**. M0.2 is **pair relation plus local relation-environment contextualization**. M0.2 subclasses the frozen M0.1 implementation; downstream Stage II transport and all state-update components are inherited unchanged.

The experiment trained only `relation_transport_m02` on Movies and Grocery with seeds 42, 43, and 44. Full-graph NC used the original train split, original validation for early stopping/model selection, and the existing task optimizer/protocol. Test evaluation and LP were off. M0.1 references below are the existing `orthogonal_transport` rows in `results/model_design/m0/pilot_records.json`; no M0.1 or baseline retraining/comparison was run. The detailed run data are in [`m02_runs.json`](../../results/model_design/m02/m02_runs.json), [`m01_vs_m02.csv`](../../results/model_design/m02/m01_vs_m02.csv), [`context_diagnostics.csv`](../../results/model_design/m02/context_diagnostics.csv), [`context_counterfactual.csv`](../../results/model_design/m02/context_counterfactual.csv), and [`progressive_dynamics.csv`](../../results/model_design/m02/progressive_dynamics.csv).

## 1. Did M0.1 progressive states continue to change?

Yes. Across the six frozen M0.1 runs and two modalities, average relative state change was 1.056, 0.367, and 0.253 at orders 1, 2, and 3. Mean adjacent cosine increased from 0.743 to 0.930 and 0.966; cumulative novelty increased from 0.257 to 0.427 and 0.544. Each order still changed semantic states, with diminishing per-step movement. The complete quantiles are in the M0.1 audit outputs and [`m01_audit_report.md`](m01_audit_report.md).

## 2. What roles did the similarity prior and relation correction play?

In the frozen M0.1 audit, mean absolute prior logit was 0.527, mean absolute relation correction was 1.512, mean correction share was 0.739, and learned `softplus(raw_tau)` averaged 0.981. The relation correction flipped prior sign on 0.981 of eligible directed edges on average. Turning off only that correction while preserving the relation-conditioned transport reduced validation accuracy/Macro-F1 by 2.70/3.61 percentage points on Movies and 1.15/2.19 points on Grocery. The similarity prior remained present, while the correction made a substantial additional contribution.

## 3. Did frozen transport-off hurt M0.1?

Yes. Holding every trained M0.1 parameter and the NC head fixed, replacing orthogonal transport with identity reduced validation accuracy/Macro-F1 by 10.62/10.83 points on Movies and 16.30/16.71 points on Grocery, averaged over three seeds. This is a frozen functional counterfactual, not a retrained identity comparison. The transport-to-compatibility association statistics are heterogeneous across runs and do not show transport as a simple copy of similarity or conductance. See the per-run table in [`frozen_counterfactuals.csv`](../../results/model_design/m01_audit/frozen_counterfactuals.csv) and the mechanism audit report.

## 4. Was the M0.2 context residual learned and active?

Yes. Mean relative contextual residual (edgewise `||R-R0||/(||R0||+1e-12)`, averaged over run/modality summaries) was 0.282 on Movies and 0.157 on Grocery. The corresponding average p50 values were 0.299 and 0.163; average p90 values were 0.316 and 0.205. Thus the contextual relation was clearly nonzero in all six trained checkpoints, with a larger residual on Movies.

## 5. Did context change conductance, transport, or both?

Both. Mean absolute conductance change was 0.170 on Movies and 0.096 on Grocery. Mean per-edge absolute angle change, averaged over the eight Givens angles, was 0.115 and 0.076, respectively. The angle changes confirm that context altered learned transport as well as compatibility. Conductance and angle have different units, so their raw magnitudes should not be treated as a normalized attribution of which pathway changed more.

## 6. Did frozen Context-Off hurt M0.2?

Yes, on all six seeds. With the trained M0.2 model and NC head fixed, Context-Off reduced validation accuracy/Macro-F1 by 0.93/2.80 points on Movies and 0.61/1.07 points on Grocery on average. The per-seed accuracy decrease ranged from 0.24 to 1.35 points on Movies and 0.53 to 0.70 points on Grocery. This supports functional use of the relation environment, though the Grocery task effect is modest.

| Dataset | Full Context Acc / Macro-F1 | Context-Off Acc / Macro-F1 | Full minus Off Acc / Macro-F1 |
|---|---:|---:|---:|
| Movies | 54.02 ± 0.36% / 40.88 ± 0.43% | 53.09 ± 0.20% / 38.08 ± 0.64% | +0.93 ± 0.49 / +2.80 ± 1.05 pp |
| Grocery | 81.47 ± 0.37% / 71.57 ± 1.67% | 80.87 ± 0.41% / 70.50 ± 1.38% | +0.61 ± 0.07 / +1.07 ± 0.40 pp |

Values are mean ± population SD across the three seeds. Context-Off uses the same trained model parameters and classifier and evaluates the original validation labels with the fixed `num_classes` label set.

## 7. What were the paired M0.2 versus M0.1 task deltas?

The per-seed values and mean ± population SD are below. Deltas are M0.2 minus M0.1 in percentage points.

| Dataset | Seed | M0.1 Acc / Macro-F1 | M0.2 Acc / Macro-F1 | Paired Δ Acc / Macro-F1 |
|---|---:|---:|---:|---:|
| Movies | 42 | 52.91 / 37.82% | 54.44 / 41.18% | +1.53 / +3.36 pp |
| Movies | 43 | 52.67 / 35.15% | 54.05 / 41.20% | +1.38 / +6.05 pp |
| Movies | 44 | 53.30 / 41.12% | 53.57 / 40.28% | +0.27 / -0.84 pp |
| **Movies mean ± SD** | — | **52.96 ± 0.26 / 38.03 ± 2.44%** | **54.02 ± 0.36 / 40.88 ± 0.43%** | **+1.06 ± 0.56 / +2.86 ± 2.83 pp** |
| Grocery | 42 | 81.64 / 72.47% | 81.41 / 73.32% | -0.23 / +0.85 pp |
| Grocery | 43 | 81.23 / 71.77% | 81.96 / 72.07% | +0.73 / +0.30 pp |
| Grocery | 44 | 81.70 / 72.34% | 81.05 / 69.33% | -0.64 / -3.01 pp |
| **Grocery mean ± SD** | — | **81.52 ± 0.21 / 72.19 ± 0.30%** | **81.47 ± 0.37 / 71.57 ± 1.67%** | **-0.05 ± 0.58 / -0.62 ± 1.70 pp** |

Movies improves in mean accuracy and Macro-F1, with consistent positive accuracy deltas across seeds. Grocery mean accuracy is nearly unchanged and mean Macro-F1 is slightly lower, with seed 44 accounting for most of that decrease. The three-seed pilot is descriptive; it does not establish a statistically reliable dataset-wide gain.

## 8. Did context change progressive dynamics, and should it be retained?

M0.2 retained the same broad progressive pattern. Averaged across runs and modalities, M0.2 relative changes at orders 1/2/3 were 1.036/0.353/0.243 versus M0.1's 1.056/0.367/0.253. Adjacent cosine was 0.755/0.935/0.969 versus 0.743/0.930/0.966; cumulative novelty was 0.245/0.406/0.522 versus 0.257/0.427/0.544. Context slightly reduced per-order movement, while later orders continued to add state change. Order itself is not treated as the research contribution.

**Recommendation: retain contextual relation for the next design stage, provisionally.** The residual is active, changes both conductance and angles, and frozen Context-Off reduces validation metrics on every seed. Task effects favor M0.2 on Movies and are close to neutral on Grocery, so this pilot does not justify claiming a universal performance improvement. It also does not show systematic degradation across both datasets or evidence that the current formulation is redundant. Keep the current M0.2 formulation intact for a later integration decision; do not add complexity or retune from these six runs.

## Validation and integrity notes

- Both requested suites passed: `tests/test_relation_transport_m0.py` and `tests/test_relation_transport_m02.py` (33 passed total).
- The two-epoch Movies seed 42 smoke completed training, checkpoint save/load, validation inference, `analyze()`, and Context-Off. All checked outputs were finite; GPU process peak memory was 9,430 MiB. Smoke outputs are under ignored `outputs/model_design/m02/smoke/`.
- Full-pilot GPU process peak memory reported by the runner was 9,430 MiB for Movies and 8,696 MiB for Grocery. Logs and checkpoints are under ignored `outputs/model_design/m02/pilot/`.
- No test metrics were evaluated. No M0.1 retraining or baseline run was performed.
- `src/models/relation_transport_m0.py` remains byte-identical to the frozen source SHA-256 `951572639c77205c04bc481c8aadfb6bc25031b138ed9f52575ed0922a229045`.
