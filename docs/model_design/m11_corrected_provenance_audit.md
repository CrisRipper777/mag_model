# M1.1 Corrected Interaction-Provenance Audit

## A. Integrity

- **Source:** `exp/interaction_provenance_m1` at `fdc05a1af05a93a0c6d4c9d68b903f6cf018d811`; its working tree was clean before the audit branch was created.
- **Audit branch:** `exp/m11_corrected_provenance_audit`, created from that exact source SHA.
- All six requested Movies/Grocery checkpoints (seeds 42, 43, 44) were present and loaded with both `model_state` and `head_state`.
- No official checkpoint was trained or fine-tuned. The audit used `model.eval()` and no-grad inference. Metrics use only the original NC validation indices; the class set comes from the dataset's configured class count. Test metrics, test labels, and LP were not evaluated.
- The explicitly requested M0, M1, and M1.1 test files passed together: **52 passed**. The legacy gradient unit test uses a small synthetic graph; it did not touch an official checkpoint.
- Checkpoint file SHA-256 hashes and model/head tensor checksums matched before and after every run. All six original validation Acc/Macro-F1 pairs exactly matched the metrics stored in their checkpoints.
- Semantic states `S_text[0:4]` and `S_visual[0:4]` were reused unchanged in every counterfactual; maximum absolute error was **0.0**. Full atom reconstruction and all 36 full provenance steps also had maximum absolute error **0.0** against the frozen model.
- The audit ran on `cuda:0` with deterministic PyTorch/CUBLAS settings so the required `2e-6` reconstruction checks were reproducible.

| Dataset | Full validation Acc (mean ± population SD across seeds) | Full validation Macro-F1 (mean ± population SD) |
|---|---:|---:|
| Movies | 0.5300 ± 0.0011 | 0.3756 ± 0.0360 |
| Grocery | 0.8129 ± 0.0047 | 0.7115 ± 0.0115 |

Integrity details and checkpoint hashes are in [`source_metadata.json`](../../results/model_design/m11_audit/source_metadata.json).

## B. Corrected provenance progression

The `G0 → G1` transition is omitted from progressive-evidence claims. Values below summarize the six seed/modality runs per dataset; cosine and relative change are node-wise geometry summaries.

| Dataset | G1 → G2 relative change | G1 → G2 cosine | G2 → G3 relative change | G2 → G3 cosine | Effective rank, G1 / G2 / G3 (mean) |
|---|---:|---:|---:|---:|---:|
| Movies | 1.343 | 0.088 | 1.294 | 0.131 | 5.18 / 5.00 / 4.34 |
| Grocery | 1.391 | 0.018 | 1.328 | 0.072 | 5.18 / 5.33 / 4.82 |

Both later transitions continue to move the state substantially, with low average cosine similarity. Sampled effective ranks remain around 4–5 and every channel had nonzero sampled variance. This is evidence of continued reconstruction across orders, not a claim based on LayerNorm-controlled state norms. Per-run top-1/top-5 energy and channel variance are in [`provenance_dynamics_corrected.csv`](../../results/model_design/m11_audit/provenance_dynamics_corrected.csv).

## C. Source path vs target memory — local attribution

For each row, the table averages node-wise relative state differences across the three model seeds and both modalities. Source-History-Off keeps the full target memory in the combine step. Target-Memory-Off reuses the full source-path aggregate and changes only the target combine input. Shuffle values are the mean across seeds 3407/3408/3409; the ± value is the average within-run population SD across those three shuffle seeds.

| Dataset | Order | Source-History-Off | Target-Memory-Off | Correct Source-History-Shuffle |
|---|---:|---:|---:|---:|
| Movies | G2 | 0.564 | 1.130 | 0.161 ± 0.0007 |
| Movies | G3 | 0.619 | 1.090 | 0.230 ± 0.0010 |
| Grocery | G2 | 0.816 | 1.003 | 0.271 ± 0.0017 |
| Grocery | G3 | 0.737 | 0.958 | 0.392 ± 0.0024 |

Source-side path history changes `G2/G3` while the target memory is held at its Full value. Degree-matched source shuffles also change the resulting state for all three shuffle seeds. Target recurrent memory has a separate, strong geometric effect. These are geometry sensitivities; their magnitudes are not interpreted as a causal-importance ranking.

Each shuffle seed is separately reported in [`local_history_attribution.csv`](../../results/model_design/m11_audit/local_history_attribution.csv), together with the per-run source-off and target-off summaries.

## D. Recursive history counterfactuals

Each entry averages across the three model seeds. `G3 Δ` is relative to Full and is shown as text/visual; residual Δ uses the full provenance residual as denominator. Validation deltas are counterfactual minus Full, in percentage points. Correct source-history shuffle averages the three shuffle seeds.

| Dataset | Counterfactual | G3 Δ text / visual | Residual Δ | Validation Acc Δ / Macro-F1 Δ |
|---|---|---:|---:|---:|
| Movies | Source-history-off | 0.733 / 0.972 | 0.594 | −1.36 / −2.37 pp |
| Movies | Target-memory-off | 1.306 / 1.249 | 0.901 | −1.67 / −0.29 pp |
| Movies | Correct source-history-shuffle | 0.213 / 0.312 | 0.153 | −0.36 / −0.82 pp |
| Movies | Both-history-off legacy | 1.388 / 1.362 | 1.011 | −2.46 / −1.26 pp |
| Grocery | Source-history-off | 1.059 / 1.149 | 0.806 | −0.96 / −0.85 pp |
| Grocery | Target-memory-off | 1.146 / 1.229 | 0.829 | −0.97 / −0.94 pp |
| Grocery | Correct source-history-shuffle | 0.442 / 0.464 | 0.333 | −0.63 / −0.66 pp |
| Grocery | Both-history-off legacy | 1.283 / 1.375 | 0.975 | −1.20 / −1.32 pp |

The clean recursive interventions agree with the local analysis: source-path history affects the final state and residual, while target memory also contributes. The old both-history-off result is retained only as a sanity reference because it combines these mechanisms. Validation deltas are comparatively modest and are auxiliary evidence.

Full per-seed values are in [`recursive_history_counterfactuals.csv`](../../results/model_design/m11_audit/recursive_history_counterfactuals.csv).

## E. Relation / effect / cross decomposition

The component-off atoms use the corrected formulas: Relation-Off is `tanh(LN_A(e))`, Effect-Off is `tanh(LN_A(r))`, and Cross-Off is `tanh(LN_A(r + e))`. None calls the learned cross layer at zero, so its bias is fully removed from all three counterfactuals. Values below average the two modalities and three model seeds; the G3 state difference is averaged across text and visual.

| Dataset | Component removed | Atom relative Δ | G3 relative Δ | Residual relative Δ | Validation Acc Δ / Macro-F1 Δ |
|---|---|---:|---:|---:|---:|
| Movies | Relation-Off | 0.847 | 0.698 | 0.581 | −1.49 / −3.07 pp |
| Movies | Effect-Off | 0.820 | 0.447 | 0.289 | −0.25 / −1.24 pp |
| Movies | Cross-Off | 0.389 | 0.196 | 0.120 | −0.23 / −0.74 pp |
| Grocery | Relation-Off | 0.829 | 0.876 | 0.800 | −1.83 / −1.48 pp |
| Grocery | Effect-Off | 0.810 | 0.568 | 0.434 | −0.28 / −0.17 pp |
| Grocery | Cross-Off | 0.376 | 0.252 | 0.192 | +0.13 / +0.31 pp |

Corrected Effect-Off causes clear nonzero changes in the sampled atom, recursive state, and provenance residual across both datasets; the representation therefore does not reduce to a static relation-path encoding. Relation-Off also changes all these quantities, and Cross-Off has a smaller but nonzero effect, showing that the learned multiplicative interaction is used. Task-score changes are small or mixed for Effect-Off and Cross-Off, which does not negate the mechanism evidence.

Sampled component geometry (mean of per-run means across orders and seeds):

| Dataset / modality | `||r||` | `||e||` | `||x||` | `cos(r,e)` | `cos(r,x)` | `cos(e,x)` | `||interaction_cross.bias||` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies / text | 5.702 | 5.633 | 3.423 | 0.026 | 0.134 | 0.103 | 0.575 |
| Movies / visual | 5.700 | 5.613 | 3.346 | 0.113 | 0.057 | 0.006 | 0.521 |
| Grocery / text | 5.727 | 5.610 | 3.298 | 0.093 | 0.100 | 0.042 | 0.578 |
| Grocery / visual | 5.710 | 5.619 | 3.382 | 0.067 | 0.022 | 0.040 | 0.513 |

The full mean/p50/p90 and mean/p10/p50/p90 summaries are in [`component_geometry.csv`](../../results/model_design/m11_audit/component_geometry.csv). The deterministic ≤8192-edge samples are retained in [`corrected_atom_components.csv`](../../results/model_design/m11_audit/corrected_atom_components.csv), and recursive component counterfactuals are in [`corrected_component_counterfactuals.csv`](../../results/model_design/m11_audit/corrected_component_counterfactuals.csv).

## F. Overall M1 judgment

The corrected evidence supports an interaction-provenance representation with both source-carried path history and target-side recurrent memory. The atom depends on the latent relation, the realized semantic effect, and their learned cross term. Semantic-stream invariance and non-collapsed provenance dynamics also pass the audit checks.

**Decision: M1 is ready to proceed to M2.** This is a mechanism-based decision; the small or mixed validation changes do not change it. No M2 implementation was started in this audit.
