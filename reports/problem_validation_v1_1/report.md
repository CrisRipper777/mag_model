# Frozen probe v1.1 verification diagnostics

The original v1 main matrix remains valid. The original J_NODE context-permutation diagnostic was ineffective because its permutation was not consumed by ordinary J_NODE; the v1.1 permutation results below use the repaired explicit override. New results cover train/validation only. No link prediction (LP) was run, no new real-data training was run, and no test inference was run.

v1 training source SHA: `63867331e9d3373f19136e47ab192cc9fb014c97`
v1 evaluation lock SHA-256: `3b949712c738dd99865f9bdce882cf65f91756e775cd4af6077cbd065d3b368f`
v1.1 diagnostic source SHA-256: `39207e1b9502d95d55935e5129eb67d8ba3d29f6a8544e7d7f437771ddacf766`
V1 raw artifacts verified against the original manifest: 475/475.
Device policy: `locked per-checkpoint device from v1 training logs`; checkpoints: 15 J_NODE + 15 matching J_GLOBAL; data inputs were hash-checked before inference.

## Five-graph validation summary

Accuracies are means over the three frozen checkpoint seeds. The 30 context interventions per graph are ten fixed label-free row permutations for each of three already-trained checkpoints; their spread is descriptive and is not a seed-level uncertainty estimate.

| Graph | J_NODE val Acc | J_GLOBAL val Acc | NODE−GLOBAL (pp) | 30 context interventions: Acc drop (pp) | Train-mean fixed gate val Acc | Old argmax/CM exact matches |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.5243 | 0.5541 | -2.98 | 0.94 ± 0.44 | 0.5201 | 6/6 |
| Toys | 0.7718 | 0.7922 | -2.04 | 2.66 ± 1.10 | 0.7611 | 6/6 |
| Grocery | 0.7764 | 0.8160 | -3.96 | 3.42 ± 1.48 | 0.7561 | 6/6 |
| Reddit-S | 0.9488 | 0.9565 | -0.77 | 3.50 ± 1.97 | 0.9241 | 6/6 |
| ele-fashion | 0.8713 | 0.8697 | +0.16 | 1.96 ± 0.13 | 0.8639 | 6/6 |

## Reading the results

- The repaired frozen intervention measures whether router context rows contribute to predictions while each receiver keeps its own four path representations and active-path mask. Every permutation is scored separately; no logits are ensembled. The validation changes are evidence of predictive use in this frozen system; they do not quantify information content or prove causation outside the intervention.
- J_NODE did not outperform the retrained J_GLOBAL or the capacity controls in the original formal v1 matrix. This v1.1 diagnostic does not change that result or retrain either model.
- Very small direct text-path gate weights indicate strong suppression of those two fusion paths. They do not show that the system fully ignores text: text path representations and text-derived context can still affect the router.
- The current evidence does not identify whether the observed weights or generalization patterns arise from overfitting, shared phi, competitive gating, optimization, or another mechanism. These remain non-causal hypotheses.
- The train-mean gate is a fixed intervention on the same J_NODE representations. It is not a substitute for training J_GLOBAL. Three seeds are descriptive; no significance claim is made.
- Frozen validation reconstruction ran on each checkpoint's device recorded in its v1 training log. If a saved argmax differs, `old_argmax_to_reconstructed_argmax_transition_matrix` and the mismatch count localize the class transitions on validation; no test inference was used for diagnosis.

## Geometry and retained outputs

The new cosine is computed separately per modality as `cos(L2(x_v, eps=1e-8), mean_neighbors(L2(x_u, eps=1e-8)))`; no cross-modality cosine is used. Quartile thresholds are fitted on training nodes, cosine fitting excludes isolates, and isolates have their own bucket. Every bucket records n, a `small_n` marker for n<100, correct/wrong counts, paired correctness counts, and dense aggregate confusion matrices in `diagnostics.json`. `geometry` records old raw-mean versus normalized thresholds and validation bucket transition counts. The v1 raw-mean geometry remains the registered v1 definition.
The original full graph features and topology are used only for label-free forward passes under the v1 transductive protocol; all reported train/validation metrics index only their locked split nodes.

Only aggregate metrics, counts, thresholds, hashes, and dense confusion matrices are written. No logits, embeddings, features, cache, or per-node predictions are written by this diagnostic. Existing v1 argmax files were read only for validation reconstruction comparison; their validation slice alone was used.

## Files

- `diagnostics.json`: per-checkpoint train/validation metrics, validation reconstruction checks, router summaries, ten separate interventions, fixed-gate results, normalized geometry buckets, and confusion matrices.
- `provenance.json`: v1 and v1.1 source hashes, lock/registry identity, and verified input/checkpoint/prediction file hashes.
- `review.md`: focused implementation and test audit record.
