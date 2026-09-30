# M2.1 Intrinsic-Anchored Adaptive Evidence Assimilation

## Frozen source and scope

M2.1 subclasses `src.models.provenance_evidence_m2.Model`. Stage I and II and every Stage III evidence, query, attention, mask, token ordering, and retrieval operation remain inherited from M2. M2.1 changes only the mapping from a modality's retrieved evidence to its modality representation. Final text/visual fusion continues to use M2's 256-to-128 linear layer, LayerNorm, GELU, and dropout 0.2.

The M2 `text_residual_norm` and `visual_residual_norm` modules are inherited legacy parameters. They are frozen and unused by M2.1; their exact description is “inherited legacy M2 assimilation parameters; not part of the M2.1 conceptual architecture.” There is no LayerNorm after the M2.1 residual injection.

## Adaptive vector assimilation

For each modality `m`, let `H0_m` be the intrinsic M0 state and `U_m` the unchanged M2 retrieval. M2.1 normalizes retrieved evidence:

`U_tilde_m = LayerNorm_128(U_m)`

The node-local context is the concatenation `[H0_m, U_tilde_m, H0_m * U_tilde_m, abs(H0_m - U_tilde_m)]`, with width 512. A modality-specific `Linear(512,128)`, GELU, dropout 0.1, and `Linear(128,128)` produce a signed vector field:

`a_m = tanh(assimilation_out_m(Dropout(GELU(assimilation_in_m(context)))))`

The final linear layer's weights and bias start at zero, so `a_m` starts exactly at zero. The correction and output are `C_m = a_m * U_tilde_m` and `Z_m = H0_m + C_m`. The field is vector-valued and bounded coordinate-wise to [-1,1].

`M2.1-S` disables provenance conditioning in the inherited evidence builder; `M2.1-P` enables it. Paired variants share initialization seed and have identical initial representations to numerical tolerance because both fields start at zero.

## Scalar sanity mode

`assimilation_mode=global_scalar` adds one trainable scalar per modality initialized at zero. Its field is `tanh(alpha_m)` broadcast across channels and nodes, with `Z_m = H0_m + tanh(alpha_m) * U_tilde_m`. This mode is only a two-run seed-42 sanity check, one dataset at a time; it is not a multi-seed comparison.

## Initialization and gradient behavior

At initialization, adaptive `assimilation_out` receives a finite nonzero gradient while the zero field blocks upstream gradients for that first update. After the first optimizer update, the input MLP and retrieval path can receive gradients. The pilot runner records gradient norms at every backward pass for assimilation, query/key/value/output projections, and provenance heads. Smoke requires nonzero finite output-layer gradients on step one and nonzero finite upstream gradients after step two.

## Frozen counterfactuals

All counterfactuals reuse the full run's Stage I/II state. `Assimilation-Off` sets `a=0`; `Uniform-Node` replaces each node's field with the graph-wide mean field; `Dimension Shuffle` permutes field dimensions with seed 7311. Provenance counterfactuals use M2's established semantics: Provenance-Off, degree-matched provenance shuffle seeds 3407/3408/3409, and Order Mismatch. They alter provenance copies only while forming retrieval evidence. Stage I/II invariance must remain within max absolute error 2e-6.

## Validation-only analysis

Attention analysis preserves M2's reports. M2.1 adds correction injection ratio `||a * U_tilde|| / (||H0|| + eps)`, absolute field summaries and positive/negative/near-zero fractions, centered diversity on a deterministic sample of at most 8192 nodes, and compatibility quintiles based on cosine similarity between `H0` and normalized retrieval. These are descriptive diagnostics, not causal scores.

## Pilot boundary

The formal pilot is NC-only: Movies and Grocery, seeds 42/43/44, M2.1-S/P adaptive vector, original train for fitting and original validation for selection. Two additional scalar sanity runs use seed 42. Test evaluation and link prediction stay disabled. No extra datasets, baselines, class weighting, architecture search, or formal ablation is part of this pilot.
