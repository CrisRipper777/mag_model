# P0.2 Scalar Compatibility vs Interaction Function

> **Run status:** Complete 5-dataset × 3-seed matrix.

## 1. Scope and Protocol

- Source branch / SHA: `exp/problem_validation_p01plus` / `f7d68272038e5d866075e91d81707ac42e6a2a17`.
- Analysis branch / code SHA at report generation: `exp/problem_validation_p02` / `0c75d5832e14b2032ab669b25febc087f6e2dbc2`.
- P0.2 model/runner code SHA used for completed fits: `6512736595fc918f7709ab59167450708f32a9bb` (captured per run).
- Data: Movies, Toys, Grocery, ele-fashion, Reddit-S; run seeds 42, 43, 44.
- Existing P0.0 semantic embeddings are frozen; no semantic projector training was run.
- Original train/val/test index sets and cached probe_train/probe_calib split are reused and hash-checked. Test indices are used only for disjointness checks; no test label or metric is accessed.
- All rates and edge diagnostics use the exact ordered P0.1 sampled relation population; no new edge sample or split was created.
- NC protocol only. No LP, baseline modification, final architecture, P0.3, MoE, basis routing, prototype, OT, or CoSI/MoPF code.
- Any per-target propagation denominator is the original physical degree. Relations are directed as source `j →` target `i`; the graph is undirected, self-loop-free, and coalesced.

## 2. Implementation Audit

- P0.0 frozen embeddings: 15/15 dataset × seed artifacts verified with byte SHA-256, run-seed metadata, finite float32 values, and configured dimensions.
- P0.1 edge populations: 15/15 ordered `(target_node, neighbor_node)` arrays match the fixed cached sample exactly.
- Relation descriptor: `[H_target, H_source, |H_target-H_source|, H_target⊙H_source]`; independent Text and Visual encoders.
- V2 and V3 use the same `Linear(4D,64) → GELU → LayerNorm → Dropout → Linear(64,64) → GELU → LayerNorm` relation-encoder definition.
- V0/V1/V2/V3 share `[H_text,H_visual,N_text,N_visual] → Linear(4D,num_classes)`. Aggregation is an incoming mean with the unchanged physical degree; no attention, learned re-normalization, self-loop, residual, or hop coefficient is used.
- Full-graph training is computed in target chunks only for memory. The mathematical graph and aggregation are unchanged. Context relation `a/g` is computed once from H0 and fixed across orders.
- Exact edge contribution was audited on 256 random sampled edges per model/run, CPU float32, for Text/Visual/joint removal. Every audit must have max absolute error `<1e-5`.
- V3 has an additional feature-modulation head. Relation parameter counts and functional controls are reported below; gains are not attributed solely to parameter capacity.

### Frozen split / source-artifact summary

| Dataset | probe train + calib | Original val | Test indices (count only) | Sampled P0.1 relations | Physical directed edges |
|---|---:|---:|---:|---:|---:|
| Movies | 8002 + 2001 | 3334 | 3335 | 30552 | 160802 |
| Toys | 9933 + 2484 | 4139 | 4139 | 22528 | 113402 |
| Grocery | 8195 + 2049 | 3415 | 3415 | 26053 | 142262 |
| ele-fashion | 46927 + 11732 | 9777 | 29330 | 37264 | 399172 |
| Reddit-S | 7628 + 1908 | 3179 | 3179 | 39613 | 283080 |

### Parameter counts

| Dataset | Variant | Relation params | Classifier params | Total trainable |
|---|---|---:|---:|---:|
| Movies | uniform | 0 | 10,260 | 10,260 |
| Movies | similarity_scalar | 4 | 10,260 | 10,264 |
| Movies | learned_scalar | 74,626 | 10,260 | 84,886 |
| Movies | conditional_feature | 91,266 | 10,260 | 101,526 |
| Toys | uniform | 0 | 9,234 | 9,234 |
| Toys | similarity_scalar | 4 | 9,234 | 9,238 |
| Toys | learned_scalar | 74,626 | 9,234 | 83,860 |
| Toys | conditional_feature | 91,266 | 9,234 | 100,500 |
| Grocery | uniform | 0 | 10,260 | 10,260 |
| Grocery | similarity_scalar | 4 | 10,260 | 10,264 |
| Grocery | learned_scalar | 74,626 | 10,260 | 84,886 |
| Grocery | conditional_feature | 91,266 | 10,260 | 101,526 |
| ele-fashion | uniform | 0 | 6,156 | 6,156 |
| ele-fashion | similarity_scalar | 4 | 6,156 | 6,160 |
| ele-fashion | learned_scalar | 74,626 | 6,156 | 80,782 |
| ele-fashion | conditional_feature | 91,266 | 6,156 | 97,422 |
| Reddit-S | uniform | 0 | 10,260 | 10,260 |
| Reddit-S | similarity_scalar | 4 | 10,260 | 10,264 |
| Reddit-S | learned_scalar | 74,626 | 10,260 | 84,886 |
| Reddit-S | conditional_feature | 91,266 | 10,260 | 101,526 |

## 3. Immediate One-Hop Results

Table A reports held-out original-val mean ± population SD across available seeds. Per-seed train/calib/val Acc, Macro-F1, CE, best epoch, and parameter counts are in `p02_immediate_results.csv`.

| Dataset | Variant | Val Acc | Val Macro-F1 | Val CE | Params (total) |
|---|---|---:|---:|---:|---:|
| Movies | uniform | 0.540 ± 0.001 | 0.420 ± 0.006 | 1.362 ± 0.004 | 10,260 |
| Movies | similarity_scalar | 0.546 ± 0.001 | 0.436 ± 0.007 | 1.359 ± 0.004 | 10,264 |
| Movies | learned_scalar | 0.535 ± 0.008 | 0.419 ± 0.014 | 1.425 ± 0.055 | 84,886 |
| Movies | conditional_feature | 0.539 ± 0.002 | 0.414 ± 0.013 | 1.401 ± 0.012 | 101,526 |
| Toys | uniform | 0.784 ± 0.002 | 0.747 ± 0.005 | 0.732 ± 0.021 | 9,234 |
| Toys | similarity_scalar | 0.788 ± 0.001 | 0.756 ± 0.002 | 0.719 ± 0.016 | 9,238 |
| Toys | learned_scalar | 0.786 ± 0.003 | 0.755 ± 0.002 | 0.725 ± 0.005 | 83,860 |
| Toys | conditional_feature | 0.786 ± 0.000 | 0.756 ± 0.004 | 0.724 ± 0.008 | 100,500 |
| Grocery | uniform | 0.815 ± 0.001 | 0.735 ± 0.008 | 0.657 ± 0.005 | 10,260 |
| Grocery | similarity_scalar | 0.817 ± 0.000 | 0.746 ± 0.004 | 0.654 ± 0.004 | 10,264 |
| Grocery | learned_scalar | 0.812 ± 0.001 | 0.730 ± 0.002 | 0.658 ± 0.007 | 84,886 |
| Grocery | conditional_feature | 0.815 ± 0.002 | 0.737 ± 0.003 | 0.661 ± 0.004 | 101,526 |
| ele-fashion | uniform | 0.870 ± 0.001 | 0.682 ± 0.005 | 0.396 ± 0.003 | 6,156 |
| ele-fashion | similarity_scalar | 0.870 ± 0.001 | 0.684 ± 0.005 | 0.394 ± 0.003 | 6,160 |
| ele-fashion | learned_scalar | 0.869 ± 0.002 | 0.676 ± 0.006 | 0.413 ± 0.005 | 80,782 |
| ele-fashion | conditional_feature | 0.865 ± 0.002 | 0.678 ± 0.013 | 0.429 ± 0.010 | 97,422 |
| Reddit-S | uniform | 0.956 ± 0.005 | 0.911 ± 0.011 | 0.191 ± 0.042 | 10,260 |
| Reddit-S | similarity_scalar | 0.958 ± 0.003 | 0.920 ± 0.003 | 0.167 ± 0.013 | 10,264 |
| Reddit-S | learned_scalar | 0.958 ± 0.001 | 0.918 ± 0.003 | 0.160 ± 0.001 | 84,886 |
| Reddit-S | conditional_feature | 0.957 ± 0.001 | 0.915 ± 0.002 | 0.162 ± 0.003 | 101,526 |

V0 is structurally the P0.1 uniform one-hop mean with a newly fitted classifier. The following paired differences provide the smoke/pilot alignment check on the original-val Acc and Macro-F1 fields saved by P0.1 (P0.1 did not save CE in that metrics block).

| Dataset | P0.2 V0 − P0.1 Acc | Macro-F1 |
|---|---:|---:|
| Movies | -0.000 ± 0.000 | 0.000 ± 0.000 |
| Toys | -0.000 ± 0.000 | 0.000 ± 0.000 |
| Grocery | 0.000 ± 0.000 | 0.000 ± 0.000 |
| ele-fashion | -0.000 ± 0.000 | 0.000 ± 0.000 |
| Reddit-S | -0.000 ± 0.000 | 0.000 ± 0.000 |

## 4. Learned Scalar vs Conditional Feature

Paired seed deltas are computed within each run before aggregation. Positive Acc/F1 favors V3; negative CE favors V3. Context readouts remain frozen diagnostics, not benchmark-model scores.

| Dataset | ΔAcc V3−V2 | ΔF1 V3−V2 | ΔCE V3−V2 | Context-only ΔAcc / ΔF1 | Full-bank ΔAcc / ΔF1 |
|---|---:|---:|---:|---:|---:|
| Movies | 0.004 ± 0.007 | -0.004 ± 0.007 | -0.024 ± 0.044 | 0.007 ± 0.027 / -0.018 ± 0.118 | -0.010 ± 0.018 / -0.045 ± 0.093 |
| Toys | -0.000 ± 0.004 | 0.001 ± 0.005 | -0.001 ± 0.004 | 0.009 ± 0.001 / 0.014 ± 0.004 | 0.008 ± 0.003 / 0.018 ± 0.007 |
| Grocery | 0.002 ± 0.002 | 0.007 ± 0.003 | 0.004 ± 0.004 | 0.026 ± 0.006 / 0.034 ± 0.014 | 0.004 ± 0.004 / 0.006 ± 0.006 |
| ele-fashion | -0.004 ± 0.002 | 0.002 ± 0.010 | 0.016 ± 0.014 | 0.013 ± 0.002 / 0.048 ± 0.019 | 0.013 ± 0.002 / 0.036 ± 0.007 |
| Reddit-S | -0.001 ± 0.001 | -0.003 ± 0.003 | 0.002 ± 0.003 | 0.003 ± 0.002 / 0.007 ± 0.006 | -0.001 ± 0.002 / -0.005 ± 0.002 |

## 5. Model-Specific Edge Utility

`U = CE(logits − exact edge contribution, y) − CE(logits, y)` is model-specific and is not causal ground truth. The table gives V3−V2 mean ΔU on the union of P0.1 Text/Visual compatibility-conflict relations. Per-seed means, medians, positive fractions, and target-node bootstrap CIs for both modality-specific and joint scopes are in `p02_conflict_subset.csv`.

| Dataset | Scope | Conflict edges (mean count) | Mean ΔU (seed mean ± SD) | Fraction ΔU>0 |
|---|---|---:|---:|---:|
| Movies | text | 8799.7 | 0.000 ± 0.004 | 0.509 ± 0.002 |
| Movies | visual | 8799.7 | 0.002 ± 0.007 | 0.571 ± 0.008 |
| Movies | joint | 8799.7 | 0.006 ± 0.004 | 0.557 ± 0.008 |
| Toys | text | 5998.0 | 0.002 ± 0.002 | 0.517 ± 0.033 |
| Toys | visual | 5998.0 | -0.001 ± 0.010 | 0.500 ± 0.130 |
| Toys | joint | 5998.0 | 0.002 ± 0.011 | 0.517 ± 0.107 |
| Grocery | text | 5874.3 | 0.004 ± 0.005 | 0.519 ± 0.052 |
| Grocery | visual | 5874.3 | -0.007 ± 0.009 | 0.465 ± 0.075 |
| Grocery | joint | 5874.3 | -0.003 ± 0.010 | 0.476 ± 0.068 |
| ele-fashion | text | 9485.0 | 0.001 ± 0.009 | 0.487 ± 0.018 |
| ele-fashion | visual | 9485.0 | 0.014 ± 0.015 | 0.568 ± 0.049 |
| ele-fashion | joint | 9485.0 | 0.000 ± 0.014 | 0.530 ± 0.132 |
| Reddit-S | text | 11977.3 | -0.000 ± 0.001 | 0.317 ± 0.100 |
| Reddit-S | visual | 11977.3 | 0.003 ± 0.001 | 0.334 ± 0.087 |
| Reddit-S | joint | 11977.3 | 0.005 ± 0.001 | 0.328 ± 0.117 |

## 6. Compatibility-Conflict Analysis

Compatibility-conflict is defined against the original P0.1 uniform-message utility: Q1 with reference `U>0` or Q5 with reference `U<0`. Consistent relations use Q1 `U<0` or Q5 `U>0`. These fixed subsets are diagnostic only and were not used for fitting or checkpoint selection.

The conflict CSV separately contains low-similarity beneficial, high-similarity harmful, all conflict, consistent, and union subsets for Text/Visual reference definitions, with Text/Visual/joint P0.2 utility scopes. Each run recomputes ΔU per edge and then performs the 1,000-replicate target-node bootstrap for the mean; seed SD and bootstrap CI are not combined.

## 7. Non-Scalar Mechanism Analysis

V3 diagnostic distributions are computed on the fixed sampled original-val relation population. `channel_std` is population SD across the 128 feature-gate channels for each edge. Values are descriptive; no artificial success threshold is imposed.

| Dataset | Modality | Subset | Residual ratio | Non-collinearity | Channel std |
|---|---|---|---:|---:|---:|
| Movies | text | all_edges | 0.296 ± 0.004 | 0.038 ± 0.002 | 0.280 ± 0.007 |
| Movies | text | compatibility_conflict | 0.298 ± 0.004 | 0.040 ± 0.002 | 0.282 ± 0.006 |
| Movies | text | compatibility_consistent | 0.302 ± 0.004 | 0.039 ± 0.002 | 0.281 ± 0.006 |
| Movies | visual | all_edges | 0.324 ± 0.010 | 0.042 ± 0.004 | 0.302 ± 0.012 |
| Movies | visual | compatibility_conflict | 0.320 ± 0.011 | 0.043 ± 0.004 | 0.302 ± 0.012 |
| Movies | visual | compatibility_consistent | 0.321 ± 0.009 | 0.041 ± 0.003 | 0.298 ± 0.010 |
| Toys | text | all_edges | 0.307 ± 0.008 | 0.032 ± 0.002 | 0.272 ± 0.007 |
| Toys | text | compatibility_conflict | 0.288 ± 0.011 | 0.034 ± 0.003 | 0.269 ± 0.009 |
| Toys | text | compatibility_consistent | 0.314 ± 0.007 | 0.032 ± 0.003 | 0.271 ± 0.006 |
| Toys | visual | all_edges | 0.302 ± 0.007 | 0.031 ± 0.003 | 0.269 ± 0.009 |
| Toys | visual | compatibility_conflict | 0.280 ± 0.014 | 0.033 ± 0.004 | 0.266 ± 0.013 |
| Toys | visual | compatibility_consistent | 0.310 ± 0.006 | 0.031 ± 0.003 | 0.268 ± 0.007 |
| Grocery | text | all_edges | 0.321 ± 0.003 | 0.031 ± 0.002 | 0.276 ± 0.006 |
| Grocery | text | compatibility_conflict | 0.305 ± 0.001 | 0.038 ± 0.001 | 0.279 ± 0.004 |
| Grocery | text | compatibility_consistent | 0.328 ± 0.005 | 0.031 ± 0.002 | 0.271 ± 0.007 |
| Grocery | visual | all_edges | 0.323 ± 0.004 | 0.036 ± 0.002 | 0.290 ± 0.005 |
| Grocery | visual | compatibility_conflict | 0.313 ± 0.007 | 0.042 ± 0.003 | 0.296 ± 0.007 |
| Grocery | visual | compatibility_consistent | 0.329 ± 0.002 | 0.034 ± 0.001 | 0.284 ± 0.004 |
| ele-fashion | text | all_edges | 0.329 ± 0.007 | 0.041 ± 0.006 | 0.292 ± 0.014 |
| ele-fashion | text | compatibility_conflict | 0.320 ± 0.018 | 0.044 ± 0.006 | 0.298 ± 0.018 |
| ele-fashion | text | compatibility_consistent | 0.327 ± 0.005 | 0.039 ± 0.005 | 0.286 ± 0.012 |
| ele-fashion | visual | all_edges | 0.323 ± 0.013 | 0.040 ± 0.004 | 0.297 ± 0.015 |
| ele-fashion | visual | compatibility_conflict | 0.316 ± 0.019 | 0.041 ± 0.005 | 0.298 ± 0.017 |
| ele-fashion | visual | compatibility_consistent | 0.326 ± 0.014 | 0.040 ± 0.004 | 0.296 ± 0.015 |
| Reddit-S | text | all_edges | 0.320 ± 0.005 | 0.025 ± 0.003 | 0.254 ± 0.005 |
| Reddit-S | text | compatibility_conflict | 0.294 ± 0.002 | 0.032 ± 0.004 | 0.268 ± 0.008 |
| Reddit-S | text | compatibility_consistent | 0.331 ± 0.006 | 0.024 ± 0.003 | 0.245 ± 0.004 |
| Reddit-S | visual | all_edges | 0.349 ± 0.010 | 0.022 ± 0.003 | 0.242 ± 0.007 |
| Reddit-S | visual | compatibility_conflict | 0.320 ± 0.011 | 0.025 ± 0.003 | 0.252 ± 0.006 |
| Reddit-S | visual | compatibility_consistent | 0.348 ± 0.013 | 0.023 ± 0.004 | 0.241 ± 0.009 |

## 8. Text/Visual Disagreement Analysis

The robust P0.1 definition is reused with `epsilon = 0.1 × median(|U_ref,joint|)` (or `1e-8` if that median is zero); both modality utilities must exceed epsilon in magnitude. Disagreement means opposite signs; agreement means equal signs. Mechanism distributions for these two subsets appear in `p02_mechanism_summary.csv` beside all-edge/conflict/consistent groups. No requirement is imposed that disagreement edges have larger V3 modulation.

- Movies text: robust disagreement/agreement counts are 7481/15463; epsilon=0.00156862.
- Movies visual: robust disagreement/agreement counts are 7481/15463; epsilon=0.00156862.
- Toys text: robust disagreement/agreement counts are 2894/13709; epsilon=0.00237764.
- Toys visual: robust disagreement/agreement counts are 2894/13709; epsilon=0.00237764.
- Grocery text: robust disagreement/agreement counts are 3289/15418; epsilon=0.000873804.
- Grocery visual: robust disagreement/agreement counts are 3289/15418; epsilon=0.000873804.
- ele-fashion text: robust disagreement/agreement counts are 6071/17765; epsilon=0.000448466.
- ele-fashion visual: robust disagreement/agreement counts are 6071/17765; epsilon=0.000448466.
- Reddit-S text: robust disagreement/agreement counts are 2200/25730; epsilon=2.80164e-05.
- Reddit-S visual: robust disagreement/agreement counts are 2200/25730; epsilon=2.80164e-05.

| Dataset | Modality | Robust subset | Residual ratio | Non-collinearity | Channel std |
|---|---|---|---:|---:|---:|
| Movies | text | robust_tv_disagreement | 0.294 ± 0.006 | 0.038 ± 0.002 | 0.280 ± 0.006 |
| Movies | text | robust_tv_agreement | 0.296 ± 0.005 | 0.038 ± 0.003 | 0.280 ± 0.007 |
| Movies | visual | robust_tv_disagreement | 0.322 ± 0.009 | 0.042 ± 0.004 | 0.302 ± 0.012 |
| Movies | visual | robust_tv_agreement | 0.323 ± 0.010 | 0.042 ± 0.005 | 0.302 ± 0.012 |
| Toys | text | robust_tv_disagreement | 0.298 ± 0.010 | 0.032 ± 0.003 | 0.271 ± 0.007 |
| Toys | text | robust_tv_agreement | 0.306 ± 0.009 | 0.032 ± 0.002 | 0.271 ± 0.006 |
| Toys | visual | robust_tv_disagreement | 0.290 ± 0.010 | 0.031 ± 0.004 | 0.268 ± 0.011 |
| Toys | visual | robust_tv_agreement | 0.301 ± 0.006 | 0.031 ± 0.003 | 0.269 ± 0.009 |
| Grocery | text | robust_tv_disagreement | 0.312 ± 0.002 | 0.033 ± 0.001 | 0.277 ± 0.005 |
| Grocery | text | robust_tv_agreement | 0.321 ± 0.003 | 0.031 ± 0.002 | 0.275 ± 0.006 |
| Grocery | visual | robust_tv_disagreement | 0.315 ± 0.004 | 0.038 ± 0.003 | 0.291 ± 0.007 |
| Grocery | visual | robust_tv_agreement | 0.321 ± 0.005 | 0.036 ± 0.002 | 0.289 ± 0.004 |
| ele-fashion | text | robust_tv_disagreement | 0.322 ± 0.012 | 0.042 ± 0.005 | 0.293 ± 0.014 |
| ele-fashion | text | robust_tv_agreement | 0.328 ± 0.009 | 0.041 ± 0.007 | 0.294 ± 0.015 |
| ele-fashion | visual | robust_tv_disagreement | 0.319 ± 0.014 | 0.041 ± 0.002 | 0.298 ± 0.014 |
| ele-fashion | visual | robust_tv_agreement | 0.321 ± 0.013 | 0.040 ± 0.004 | 0.297 ± 0.016 |
| Reddit-S | text | robust_tv_disagreement | 0.314 ± 0.006 | 0.027 ± 0.002 | 0.260 ± 0.005 |
| Reddit-S | text | robust_tv_agreement | 0.320 ± 0.005 | 0.025 ± 0.003 | 0.255 ± 0.006 |
| Reddit-S | visual | robust_tv_disagreement | 0.316 ± 0.005 | 0.029 ± 0.003 | 0.248 ± 0.006 |
| Reddit-S | visual | robust_tv_agreement | 0.349 ± 0.010 | 0.021 ± 0.003 | 0.243 ± 0.006 |

## 9. Function Identity Ablation

The frozen V3 relation encoder, scalar gate, and classifier are kept unchanged while `g := 1` at original-val inference. Positive Acc/F1 `full−identity` and negative CE `full−identity` favor the learned feature function.

| Dataset | Full Acc | Identity Acc | ΔAcc | Full F1 | Identity F1 | ΔF1 | Full CE | Identity CE | ΔCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.539 ± 0.002 | 0.534 ± 0.003 | 0.005 ± 0.005 | 0.414 ± 0.013 | 0.403 ± 0.020 | 0.011 ± 0.007 | 1.401 ± 0.012 | 1.376 ± 0.010 | 0.026 ± 0.022 |
| Toys | 0.786 ± 0.000 | 0.785 ± 0.002 | 0.001 ± 0.002 | 0.756 ± 0.004 | 0.757 ± 0.005 | -0.001 ± 0.002 | 0.724 ± 0.008 | 0.738 ± 0.016 | -0.014 ± 0.008 |
| Grocery | 0.815 ± 0.002 | 0.812 ± 0.004 | 0.003 ± 0.002 | 0.737 ± 0.003 | 0.732 ± 0.005 | 0.005 ± 0.004 | 0.661 ± 0.004 | 0.665 ± 0.006 | -0.003 ± 0.003 |
| ele-fashion | 0.865 ± 0.002 | 0.866 ± 0.001 | -0.001 ± 0.001 | 0.678 ± 0.013 | 0.673 ± 0.015 | 0.006 ± 0.002 | 0.429 ± 0.010 | 0.419 ± 0.012 | 0.010 ± 0.014 |
| Reddit-S | 0.957 ± 0.001 | 0.959 ± 0.001 | -0.002 ± 0.001 | 0.915 ± 0.002 | 0.918 ± 0.003 | -0.003 ± 0.003 | 0.162 ± 0.003 | 0.167 ± 0.004 | -0.004 ± 0.002 |

## 10. Relation-Function Shuffle

The five frozen-checkpoint repetitions shuffle whole `g_ij` vectors within the same dataset, modality, and full-physical-graph probe-similarity quintile; edge-specific `a_ij` is untouched. Full-minus-shuffle is reported as requested; for CE, lower is better.

| Dataset | Full Acc | Shuffle Acc | ΔAcc | Full F1 | Shuffle F1 | ΔF1 | Full CE | Shuffle CE | ΔCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.539 ± 0.002 | 0.530 ± 0.001 | 0.009 ± 0.002 | 0.414 ± 0.013 | 0.396 ± 0.021 | 0.018 ± 0.008 | 1.401 ± 0.012 | 1.395 ± 0.005 | 0.006 ± 0.017 |
| Toys | 0.786 ± 0.000 | 0.783 ± 0.003 | 0.003 ± 0.002 | 0.756 ± 0.004 | 0.753 ± 0.004 | 0.003 ± 0.002 | 0.724 ± 0.008 | 0.738 ± 0.012 | -0.015 ± 0.006 |
| Grocery | 0.815 ± 0.002 | 0.809 ± 0.003 | 0.006 ± 0.002 | 0.737 ± 0.003 | 0.730 ± 0.003 | 0.007 ± 0.003 | 0.661 ± 0.004 | 0.669 ± 0.006 | -0.008 ± 0.003 |
| ele-fashion | 0.865 ± 0.002 | 0.862 ± 0.001 | 0.003 ± 0.001 | 0.678 ± 0.013 | 0.669 ± 0.012 | 0.009 ± 0.001 | 0.429 ± 0.010 | 0.434 ± 0.007 | -0.005 ± 0.005 |
| Reddit-S | 0.957 ± 0.001 | 0.957 ± 0.001 | -0.000 ± 0.001 | 0.915 ± 0.002 | 0.917 ± 0.002 | -0.003 ± 0.002 | 0.162 ± 0.003 | 0.165 ± 0.003 | -0.003 ± 0.001 |

## 11. Minimal Context Rollout

C1/C2/C3 are repeated applications of the fixed H0-derived relation state. No normalization or learned hop weight is inserted into propagation. Every context block was checked for finite values and norm summaries (mean, median, p95/max) are saved by seed in `p02_context_stability.csv`.

The output contains no NaN/Inf for completed runs. V3 C3 feature norms grow substantially relative to frozen H0 (H0 p95 is about 9–10); the maximum V3 C3 p95 across modality and seed reaches approximately 113.5 for Movies, 128.3 for Toys, 138.5 for Grocery, 151.4 for ele-fashion, and 199.0 for Reddit-S. These are observed activation growth values and remain a limitation for downstream use. No normalization was added after observing them. This is a frozen diagnostic, not a final MAG model.

## 12. Context-Only and Full-Bank Probe

Only a linear readout is trained on probe_train, selected by probe_calib accuracy, then evaluated on original val. Context blocks are L2-normalized immediately before concatenation. These are diagnostic readouts and must not be compared with formal benchmark-model scores.

| Dataset | Variant | Context-only Acc / F1 | Full-bank Acc / F1 |
|---|---|---:|---:|
| Movies | uniform | 0.491 ± 0.010 / 0.270 ± 0.043 | 0.536 ± 0.003 / 0.369 ± 0.006 |
| Movies | similarity_scalar | 0.487 ± 0.005 / 0.232 ± 0.024 | 0.540 ± 0.003 / 0.381 ± 0.002 |
| Movies | learned_scalar | 0.498 ± 0.017 / 0.286 ± 0.070 | 0.542 ± 0.004 / 0.389 ± 0.006 |
| Movies | conditional_feature | 0.505 ± 0.011 / 0.268 ± 0.056 | 0.532 ± 0.022 / 0.345 ± 0.099 |
| Toys | uniform | 0.768 ± 0.002 / 0.721 ± 0.006 | 0.790 ± 0.004 / 0.752 ± 0.007 |
| Toys | similarity_scalar | 0.771 ± 0.001 / 0.727 ± 0.002 | 0.789 ± 0.003 / 0.749 ± 0.003 |
| Toys | learned_scalar | 0.768 ± 0.005 / 0.716 ± 0.013 | 0.783 ± 0.002 / 0.736 ± 0.003 |
| Toys | conditional_feature | 0.778 ± 0.006 / 0.730 ± 0.016 | 0.791 ± 0.002 / 0.754 ± 0.004 |
| Grocery | uniform | 0.749 ± 0.005 / 0.601 ± 0.010 | 0.799 ± 0.005 / 0.678 ± 0.013 |
| Grocery | similarity_scalar | 0.745 ± 0.005 / 0.591 ± 0.007 | 0.803 ± 0.002 / 0.684 ± 0.004 |
| Grocery | learned_scalar | 0.755 ± 0.008 / 0.610 ± 0.015 | 0.804 ± 0.002 / 0.692 ± 0.005 |
| Grocery | conditional_feature | 0.781 ± 0.004 / 0.644 ± 0.008 | 0.808 ± 0.001 / 0.698 ± 0.003 |
| ele-fashion | uniform | 0.785 ± 0.004 / 0.472 ± 0.014 | 0.832 ± 0.002 / 0.574 ± 0.020 |
| ele-fashion | similarity_scalar | 0.789 ± 0.001 / 0.479 ± 0.011 | 0.831 ± 0.002 / 0.577 ± 0.011 |
| ele-fashion | learned_scalar | 0.793 ± 0.001 / 0.492 ± 0.013 | 0.828 ± 0.005 / 0.579 ± 0.022 |
| ele-fashion | conditional_feature | 0.806 ± 0.002 / 0.541 ± 0.023 | 0.841 ± 0.003 / 0.614 ± 0.015 |
| Reddit-S | uniform | 0.933 ± 0.003 / 0.861 ± 0.006 | 0.946 ± 0.001 / 0.883 ± 0.003 |
| Reddit-S | similarity_scalar | 0.940 ± 0.002 / 0.879 ± 0.010 | 0.945 ± 0.001 / 0.879 ± 0.002 |
| Reddit-S | learned_scalar | 0.939 ± 0.003 / 0.873 ± 0.006 | 0.948 ± 0.002 / 0.884 ± 0.004 |
| Reddit-S | conditional_feature | 0.942 ± 0.001 / 0.880 ± 0.011 | 0.947 ± 0.000 / 0.880 ± 0.002 |

## 13. H2 Evidence Matrix

### Immediate Task Evidence

Use the paired V3−V2 rows in Section 4. Immediate one-hop metrics answer whether the feature function improves this probe task; they are not a required standalone win condition.

### Non-Scalar Mechanism Evidence

Residual ratio measures feature changes relative to the incoming semantic vector; non-collinearity measures direction change; channel dispersion measures non-uniform feature gates. See the all-edge/conflict/consistent distributions in Section 7.

### Function Identity / Shuffle Evidence

Identity tests whether learned `g` changes predictions with `a` fixed. The quintile-constrained shuffle tests whether the relation-to-function assignment matters while preserving `a` and the within-quintile gate-vector multiset. See Sections 9–10.

### Compatibility-Conflict Evidence

P0.1 conflict subsets are held fixed and never train P0.2. ΔU is model-specific; positive values do not establish a downstream task gain. Target-node bootstrap intervals are per run and remain distinct from three-seed SD.

### Downstream Context Evidence

The frozen rollout and linear heads test whether the learned fixed interaction is useful after repeated structural composition. It is not final architecture performance.

### Overall Assessment

The matrix below is descriptive, based on the full evidence chain and the observed three-seed patterns. It is not a statistical significance classification. 'Weak' means similarity/feature interaction is not a reliable one-hop task discriminator; 'intermediate' means some controlled interaction or context evidence is present but does not establish a broad immediate gain; 'strong' would require consistent useful relation-function evidence across these diagnostics. No dataset meets the strong criterion.

| Dataset | Evidence regime | Brief evidence summary |
|---|---|---|
| Movies | Weak, with function-sensitive effects | V3−V2 one-hop is small and seed-variable; context-only/full-bank results are variable. Identity and within-quintile shuffle change predictions, while edge-utility effects are mixed. |
| Toys | Intermediate | One-hop V3 and V2 are near-tied; both context readouts improve Acc/F1 over V2 across all three seeds, though context-only CE is worse. Functional controls show relation assignment matters. |
| Grocery | Intermediate | One-hop V3−V2 is mixed and slightly favors V2 on CE; context-only Acc/F1 improve in all seeds. Full-bank gains and edge-utility effects are smaller or mixed. |
| ele-fashion | Intermediate, context-led | Immediate V3 Acc/CE are worse than V2 on average, while context-only and full-bank Acc/F1 improve in each seed. Seed variation and opposing one-hop/context evidence preclude a broad gain claim. |
| Reddit-S | Weak to intermediate, localized | V3 has non-scalar mechanism and positive mean conflict-subset ΔU, but only a minority of union-conflict edges have positive ΔU; immediate and full-bank task gains are absent or mixed. |

## 14. Overall Assessment

Overall, the evidence is **moderate and dataset-dependent** for testing feature-wise interaction further. V3 demonstrably learns non-uniform feature gates and changes some controlled predictions, but its immediate one-hop results are close to V2 and are not consistently better. Context readouts help in selected datasets, with substantial C3 norm growth that limits direct downstream use. This supports a bounded follow-up comparison if desired; it does not justify a final architecture choice.

## 15. What the Evidence Supports

- V3 learns non-uniform, non-collinear feature transformations on the fixed sampled validation relation population.
- Identity and within-quintile shuffle controls show that the learned relation-to-function assignment affects predictions, with dataset-dependent direction and size.
- V3−V2 model-specific edge utility differs on fixed P0.1 conflict/consistent subsets, while its distribution is heterogeneous across targets and datasets.
- Fixed relation functions can produce useful C1/C2/C3 diagnostic representations in some datasets under separately trained linear readouts.

## 16. What the Evidence Does NOT Support

This experiment does not establish that scalar learned relations are universally insufficient, that vector relation state is necessary, that semantic transformation is necessary, or that MoE, basis routing, low-rank operators, or a cross-modal relation mixer is needed. It does not establish general multi-hop heterogeneous context utility or determine the final architecture. C3 norm growth also needs to be addressed before treating the unnormalized rollout as a usable downstream representation.

## 17. Implications for a Later Stage

Carry forward the feature-wise interaction only as a controlled candidate, with particular attention to the context-positive Toys/Grocery/ele-fashion results and the weak or mixed immediate results. A later stage should first control capacity and C3 activation growth, retain identity/shuffle controls, and evaluate a predeclared dataset-specific hypothesis. No later-stage model is implemented here.

## Output Files

- `results/problem_validation/p02/p02_immediate_results.csv` — per-seed immediate train/calib/val metrics and parameters.
- `results/problem_validation/p02/p02_cross_dataset.csv` — across-seed mean and population SD.
- `results/problem_validation/p02/p02_pairwise_deltas.csv` — paired V3−V2, V2−V1, V1−V0, and context deltas.
- `results/problem_validation/p02/p02_mechanism_summary.csv` — V3 mechanism groups and distributions.
- `results/problem_validation/p02/p02_conflict_subset.csv` and `_summary.csv` — per-run conflict subsets/target-node bootstrap CIs and aggregated summaries.
- `results/problem_validation/p02/p02_function_controls.csv` — identity and shuffle controls.
- `results/problem_validation/p02/p02_context_probe.csv` and `p02_context_stability.csv` — context readouts and activation norms.
- `results/problem_validation/p02/plots/` — Figures A–C (PNG, 600 dpi TIFF, SVG, PDF) and panel-alignment / rendered-collision QA JSON.
