# M1 Interaction Provenance pilot

## Scope and protocol

M1 was implemented on `exp/interaction_provenance_m1` from frozen M0.1 source SHA `749d1b4f20c13886a308cddb4770e1e25a865ffd`. The M0.2 implementation was not used. Six NC runs covered Movies and Grocery with seeds 42/43/44. The original train split was used for training and original validation split for checkpoint selection; test evaluation and LP were disabled. The M0.1 reference uses only existing `orthogonal_transport` rows in `results/model_design/m0/pilot_records.json`; no baseline was retrained.

The provenance adapter is a temporary task-exposure mechanism for M1 development and is NOT the final Stage III evidence-composition design. Immediate task performance is supportive evidence only; M1 is primarily evaluated by whether it learns a non-degenerate, history-sensitive provenance state.

## M0.1 paired validation comparison

| Dataset | M0.1 Acc | M1 Acc | Paired Δ Acc | M0.1 Macro-F1 | M1 Macro-F1 | Paired Δ Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.5296 ± 0.0032 | 0.5300 ± 0.0014 | 0.0004 ± 0.0019 | 0.3803 ± 0.0299 | 0.3756 ± 0.0441 | -0.0047 ± 0.0352 |
| Grocery | 0.8152 ± 0.0026 | 0.8129 ± 0.0057 | -0.0023 ± 0.0052 | 0.7219 ± 0.0037 | 0.7115 ± 0.0140 | -0.0104 ± 0.0106 |

Per-seed paired values:

| Dataset | Seed | M0.1 Acc | M1 Acc | Δ Acc | M0.1 Macro-F1 | M1 Macro-F1 | Δ Macro-F1 | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 0.5291 | 0.5303 | +0.0012 | 0.3782 | 0.4141 | +0.0360 | 70 |
| Movies | 43 | 0.5267 | 0.5285 | +0.0018 | 0.3515 | 0.3276 | -0.0239 | 49 |
| Movies | 44 | 0.5330 | 0.5312 | -0.0018 | 0.4112 | 0.3851 | -0.0261 | 48 |
| Grocery | 42 | 0.8164 | 0.8085 | -0.0079 | 0.7247 | 0.7255 | +0.0008 | 98 |
| Grocery | 43 | 0.8123 | 0.8108 | -0.0015 | 0.7177 | 0.6974 | -0.0203 | 56 |
| Grocery | 44 | 0.8170 | 0.8193 | +0.0023 | 0.7234 | 0.7117 | -0.0117 | 168 |

## Training time and memory

Epoch time is total NC train/evaluation time divided by epochs actually completed before early stopping. CUDA peak memory is measured inside the isolated training worker.

| Dataset | Seed | Epochs completed | Seconds/epoch | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| Movies | 42 | 100 | 1.52 | 6.63 | 6.72 |
| Movies | 43 | 79 | 1.06 | 6.63 | 6.72 |
| Movies | 44 | 78 | 1.11 | 6.63 | 6.72 |
| Grocery | 42 | 128 | 1.01 | 6.24 | 6.39 |
| Grocery | 43 | 86 | 1.01 | 6.24 | 6.39 |
| Grocery | 44 | 198 | 1.01 | 6.24 | 6.39 |

## Provenance dynamics

| Modality | Order | G norm mean | Relative change mean | Cosine mean |
|---|---:|---:|---:|---:|
| text | 1 | 5.736 | 1.423 | 0.002 |
| text | 2 | 5.753 | 1.352 | 0.075 |
| text | 3 | 5.782 | 1.236 | 0.198 |
| visual | 1 | 5.762 | 1.425 | 0.003 |
| visual | 2 | 5.770 | 1.382 | 0.031 |
| visual | 3 | 5.804 | 1.386 | 0.005 |

## State and interaction-atom diversity

Centered effective rank is entropy-based over sampled node/edge states. Top energies and active-channel fractions are reported without pass thresholds.

| State/order | Effective rank | Top1 energy | Top5 energy | Active channel fraction |
|---|---:|---:|---:|---:|
| G1 | 5.18 / 32 | 0.524 | 0.888 | 1.000 |
| G2 | 5.16 / 32 | 0.530 | 0.890 | 1.000 |
| G3 | 4.58 / 32 | 0.553 | 0.912 | 1.000 |
| Atom k1 | 19.15 / 32 | 0.203 | 0.519 | 1.000 |
| Atom k2 | 17.97 / 32 | 0.210 | 0.553 | 1.000 |
| Atom k3 | 17.00 / 32 | 0.219 | 0.578 | 1.000 |

## Adapter and semantic-provenance CKA

| Measure | Mean across six runs/modalities/orders |
|---|---:|
| Adapter weight norm | 1.329 (range 0.933–1.930) |
| Residual/base embedding norm ratio | 0.353 (range 0.295–0.405) |
| CKA Gk vs ΔSk (k=1) | 0.334 |
| CKA Gk vs Sk (k=1) | 0.340 |
| CKA Gk vs ΔSk (k=2) | 0.219 |
| CKA Gk vs Sk (k=2) | 0.324 |
| CKA Gk vs ΔSk (k=3) | 0.196 |
| CKA Gk vs Sk (k=3) | 0.273 |

## Frozen counterfactuals

Each provenance-only counterfactual reused the same semantic cache and asserted text/visual S0–S3 equality within 2e-6; observed maximum absolute error was 0. These are frozen functional diagnostics, not retrained ablations.

| Mode | Runs | Mean Δ Acc vs Full | Mean Δ Macro-F1 vs Full | Mean G3 relative change | Mean residual relative change | Mean atom relative change |
|---|---:|---:|---:|---:|---:|---:|
| adapter_off | 6 | -0.0146 | -0.0140 | 0.000 | 0.000 | 0.000 |
| history_off | 6 | -0.0183 | -0.0129 | 1.352 | 0.993 | 0.000 |
| history_shuffle | 18 | -0.0186 | -0.0351 | 0.833 | 0.730 | 0.000 |
| relation_off | 6 | -0.0170 | -0.0208 | 0.771 | 0.694 | 0.833 |
| effect_off | 6 | -0.0030 | -0.0065 | 0.492 | 0.361 | 0.809 |

History Shuffle used the prescribed degree buckets (1, 2–3, 4–7, 8–15, 16–31, ≥32), a shared text/visual node permutation, and shuffle seeds 3407/3408/3409 at k=2/3. The table aggregates 18 validations; per-seed/per-shuffle values are in `history_shuffle.csv`.

## Evidence assessment

- **Active:** G states have mean norms around 5.7–5.8, all 32 sampled channels are active, adapter norm averages 1.329, and residual/base norm ratio averages 0.353.
- **Progressive:** mean relative G changes stay substantial from G1→G2→G3 (text 1.35 then 1.24; visual 1.38 then 1.39).
- **History-sensitive:** History-Off changes G3 by 1.35 relative norm on average; degree-matched History Shuffle changes it by 0.83. Their mean validation accuracy deltas are -1.83 and -1.86 percentage points.
- **Interaction-grounded:** Relation-Off and Effect-Off change sampled atoms by 0.83 and 0.81 relative norm, respectively; both components affect the recursively composed state. Relation-Off has the larger mean G3 change (0.77 vs 0.49).
- **Non-redundant:** mean linear CKA Gk vs Sk ranges from 0.27 to 0.34; this indicates related but non-identical sampled representations. G state effective rank averages roughly 5.2–4.6 of 32, so coordinates are active but anisotropic.
- **Task performance:** overall paired means are Δ Acc -0.10 pp and Δ Macro-F1 -0.76 pp; dataset/seed effects are mixed, so these results do not establish a task-performance improvement.

## Decision

The pilot supports proceeding to M2: the provenance branch is active and progressive; history correspondence, latent relation, and realized message each influence its geometry; and CKA does not indicate a copied semantic state. Keep the moderate state-rank concentration and mixed Macro-F1 deltas visible as limitations. This recommendation is about testing how provenance should be combined with semantic increments in M2, not a claim that M1 improves the task metric.
