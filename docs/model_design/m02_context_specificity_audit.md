# M0.2 Context-Specificity Audit

## A. Integrity

The audit starts from `exp/contextual_relation_m02` at `8e2ddc540d6662063539f4c63abbfa7a1b2d659d` and runs on `exp/m02_context_specificity_audit`. The working tree was clean before branch creation. All six requested M0.2 pilot checkpoints were present before any inference. Their SHA-256 values before and after the audit are recorded in [`source_metadata.json`](../../results/model_design/m02_context_audit/source_metadata.json); they are unchanged.

No training, test evaluation, test-label evaluation, or LP work was performed. Each counterfactual used the same frozen M0.2 model and NC head, and evaluated original validation rows only. The script asserts that the M0.2 model and classifier state dictionaries remain unchanged, that Context-Off restores the normal context mode, and that its pair path agrees with explicit M0.1 pair evidence.

The frozen source hashes before and after are identical:

| File | SHA-256 |
|---|---|
| `src/models/relation_transport_m0.py` | `951572639c77205c04bc481c8aadfb6bc25031b138ed9f52575ed0922a229045` |
| `src/models/relation_transport_m02.py` | `af6aeff4516ffdc556ad2c1d0c6ebbd167e2ed9f7acb48b73e90cab9e169ee65` |
| `src/tasks/nc.py` | `3b1bf13fbb627f4e96db5e29d4799f761147be3609714b554dc76ffe598897e3` |
| `src/tasks/lp.py` | `0b69db942ff287538302daca6d42650d641a71b5dcf52aab43b8f59d30a2db59` |

The previous `results/model_design/m02/context_diagnostics.csv` had 24 data rows, all text. It has been regenerated from the six `m02_runs.json` records and now has 48 rows covering 6 runs × 2 modalities × 4 diagnostics. The audit copy in `results/model_design/m02_context_audit/corrected_context_diagnostics.csv` is identical and asserts both modalities are present.

## B. Leave-pair-out

For every physical directed edge, the reverse-edge map was built with sorted integer keys and verified to be an involution. Incoming/outgoing counts, sums, and square sums were accumulated once. For each edge, its own relation and the reverse relation were subtracted at both endpoints before encoding the edge-specific contexts. The relation encoder and edge-context parameters remain the trained checkpoint parameters.

After removing the current pair, the average LPO residual ratio remained close to the full-context residual ratio in both datasets. Averaged over the three seeds and two modalities:

| Dataset | Full residual ratio | LPO residual ratio | Other-context retention | Direction agreement |
|---|---:|---:|---:|---:|
| Movies | 0.240 | 0.239 | 0.995 | 0.999 |
| Grocery | 0.155 | 0.154 | 1.009 | 0.977 |

Retention is intentionally not clipped; values above one occur in Grocery. The high-degree strata show that the signal does not depend on the current pair contribution:

| Dataset | min endpoint degree | Mean full ratio | Mean LPO ratio | Mean retention | Mean direction cosine | Full–LPO mean angle difference |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 1 | 0.236 | 0.197 | 0.885 | 0.988 | 0.0234 |
| Movies | 2–3 | 0.240 | 0.237 | 0.986 | 0.997 | 0.0049 |
| Movies | 4–10 | 0.240 | 0.240 | 1.000 | 1.000 | 0.0009 |
| Movies | >10 | 0.241 | 0.241 | 1.000 | 1.000 | 0.0003 |
| Grocery | 1 | 0.172 | 0.139 | 0.890 | 0.816 | 0.0323 |
| Grocery | 2–3 | 0.156 | 0.164 | 1.110 | 0.914 | 0.0140 |
| Grocery | 4–10 | 0.153 | 0.154 | 1.006 | 0.994 | 0.0024 |
| Grocery | >10 | 0.154 | 0.153 | 1.000 | 0.999 | 0.0007 |

The bin is defined by the smaller endpoint degree; counts for every run/modality/bin are in [`leave_pair_out_degree_bins.csv`](../../results/model_design/m02_context_audit/leave_pair_out_degree_bins.csv). Mean directed edges per run/modality were about 5,870 / 13,668 / 48,890 / 92,374 for Movies and 7,944 / 17,382 / 45,196 / 71,740 for Grocery, respectively. LPO residuals remain nonzero even in the smallest bin, but that bin includes leaf-to-hub edges where the higher-degree endpoint still has other relations.

The same frozen NC classifier gives:

| Dataset | Full Context Acc / Macro-F1 | LPO Acc / Macro-F1 | Context-Off Acc / Macro-F1 | Full minus LPO Acc / Macro-F1 |
|---|---:|---:|---:|---:|
| Movies | 54.02 ± 0.36% / 40.88 ± 0.43% | 53.89 ± 0.31% / 40.83 ± 0.40% | 53.09 ± 0.20% / 38.08 ± 0.64% | +0.13 / +0.05 pp |
| Grocery | 81.47 ± 0.37% / 71.57 ± 1.67% | 81.30 ± 0.45% / 71.32 ± 1.55% | 80.87 ± 0.41% / 70.50 ± 1.38% | +0.18 / +0.25 pp |

Numbers are mean ± population SD over three seeds. LPO retains most of the frozen full-context validation behavior, and remains above Context-Off. This is a functional counterfactual, not a retrained architecture comparison. Full per-checkpoint metrics are in [`leave_pair_out_counterfactual.csv`](../../results/model_design/m02_context_audit/leave_pair_out_counterfactual.csv).

LPO preserves most of the mechanisms as well. Average absolute conductance change from pair-only to LPO was 0.148 on Movies and 0.095 on Grocery, versus 0.149 and 0.095 for full context. Average mean-absolute angle change from pair-only was 0.105 and 0.072 for LPO, versus 0.105 and 0.072 for full context. The mean full-versus-LPO angle difference was 0.0017 and 0.0046.

## C. Correct vs shuffled context

One deterministic node permutation was generated per run and shuffle seed, within physical-degree buckets; that same mapping was applied to text and visual context. Pair relations, edges, features, and labels were unchanged. All 36 run × shuffle seed × modality checks confirmed that degree buckets were preserved and the sorted context-norm multiset was exactly equal before and after shuffling.

Shuffling changed relation geometry, more on Grocery than Movies. Mean `||R_full − R_shuffle|| / (||R_full − R0|| + eps)` was 0.088 on Movies and 0.487 on Grocery. Mean absolute conductance differences were 0.0087 / 0.0299 and mean absolute angle differences were 0.0079 / 0.0236 for Movies / Grocery.

Frozen validation effects were small. Averaging the three shuffles within each checkpoint, then summarizing across the three checkpoints:

| Dataset | Full Context Acc / Macro-F1 | Shuffled Acc / Macro-F1 | Full minus Shuffled Acc / Macro-F1 |
|---|---:|---:|---:|
| Movies | 54.02 ± 0.36% / 40.88 ± 0.43% | 53.94 ± 0.38% / 40.93 ± 0.67% | +0.08 ± 0.05 / −0.04 ± 0.30 pp |
| Grocery | 81.47 ± 0.37% / 71.57 ± 1.67% | 81.39 ± 0.38% / 71.28 ± 1.49% | +0.08 ± 0.05 / +0.29 ± 0.18 pp |

Across individual shuffles, some seeds improve and others decline. Correct node correspondence has a modest positive mean accuracy effect on both datasets; Macro-F1 is slightly lower for Movies and modestly higher for Grocery. Shuffled context remains above Context-Off. The audit therefore shows that correspondence changes the mechanism, especially on Grocery, but it has limited and dataset-dependent frozen task impact. See [`context_shuffle.csv`](../../results/model_design/m02_context_audit/context_shuffle.csv) and [`context_shuffle_counterfactual.csv`](../../results/model_design/m02_context_audit/context_shuffle_counterfactual.csv).

## D. Residual diversity

The deterministic sample contains up to 8,192 directed edges for every dataset/seed/modality. The residual is not zero and has measurable variance in all 64 channels, but its variation is highly concentrated:

| Dataset | Global-bias ratio | Centered top-1 energy | Centered top-5 energy | Centered effective rank | Channels variance > 1e-6 |
|---|---:|---:|---:|---:|---:|
| Movies | 0.983 ± 0.019 | 0.943 ± 0.105 | 1.000 ± 0.000 | 1.20 ± 0.30 | 64/64 in every run/modality |
| Grocery | 0.772 ± 0.204 | 0.915 ± 0.089 | 0.999 ± 0.001 | 1.36 ± 0.30 | 64/64 in every run/modality |

Values are means ± population SD over six run/modality samples. A global-bias ratio near one and centered effective rank near one indicate that contextual residuals mostly share one direction, even though every channel has nonzero variance. The per-edge cosine with the mean residual direction averaged 0.986 on Movies and 0.771 on Grocery. Thus the contextual residual is active, but its diversity is much lower than the 64-dimensional representation size might suggest. Full results are in [`residual_diversity.csv`](../../results/model_design/m02_context_audit/residual_diversity.csv).

## E. Relation-correction signed distribution

The signed conductance correction is strongly negative and fairly concentrated. Averaged over three seeds and two modalities, pair-only correction mean / SD was −1.569 / 0.186 on Movies and −1.174 / 0.190 on Grocery. Full-context correction mean / SD was −2.027 / 0.242 and −1.339 / 0.263. More than 99.9% of pair-only and full-context edge corrections were negative in both datasets.

The change from pair-only to full-context correction was also predominantly negative: mean −0.458 on Movies, with 99.6% negative edges; mean −0.164 on Grocery, with 82.2% negative edges. Corrections vary across edges, so they are not literally a single constant. However, together with the residual-rank results, the strong signed offset is evidence that a substantial part of the correction acts in a common direction. Per-run quantiles and sign fractions are in [`relation_correction_signed.csv`](../../results/model_design/m02_context_audit/relation_correction_signed.csv).

## F. Overall Stage-I interpretation

**Conclusion: mixed evidence; retain the current M0.2 formulation provisionally, without claiming strong context-specific encoding.**

The leave-pair-out result is positive evidence for information beyond the current physical pair: excluding both directed pair contributions leaves nearly the same residual magnitude and direction, including on edges whose endpoints have many other neighbors. LPO also preserves most of the conductance/angle changes and frozen validation effect.

The counterevidence is substantial. Correct degree-matched correspondence has only small, inconsistent validation effects, while residuals are strongly concentrated in a common direction and signed relation correction is mostly a negative offset. These results leave room for the module to behave partly like a shared reparameterization rather than a rich edge-specific interpretation of each local environment.

The evidence does not meet the strong context-specific standard, because residual diversity is concentrated and correct-versus-shuffled task effects are modest. It also does not meet the weak/redundant pattern: LPO does not remove the residual or its functional effect. Keep the frozen M0.2 Stage I provisionally for a later provenance-aware utilization decision; do not tune or redesign it from this audit. No M1/M2 implementation was started.

## Validation

The requested three suites passed: `tests/test_relation_transport_m0.py`, `tests/test_relation_transport_m02.py`, and `tests/test_m02_context_specificity_audit.py` (48 passed total). All generated CSVs use LF line endings. Runtime checkpoints and logs remain under ignored `outputs/`; this audit added only result tables, metadata, tests, script, and report.
