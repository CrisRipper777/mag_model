# M0.1: Relation-Conditioned Semantic Transport Prototype

## Provenance and scope

- Source branch: `exp/problem_validation_p02`
- Source SHA: `45b2e51772998c588c8ed1a1937b7a7084c0fdd1`
- Implementation branch: `exp/relation_transport_m0`
- Scope: Stage I pair-level multimodal relation induction, Stage II relation-conditioned semantic transport, and a temporary modality-specific GPR-style readout over orders 0–3.

> M0.1 is a model-design prototype, not another Problem Validation stage.

The model consumes the formal loader's concatenated frozen raw feature tensor `[text | visual]`. It learns independent text and visual semantic projectors in the NC training run, uses the unified full-graph NC protocol, trains on original train nodes, selects checkpoints on original validation accuracy, and has test evaluation disabled for every pilot run. It does not run LP.

## Stage I: pair-level relation evidence

For each modality, the projector is `Linear(input_dim_m, 128) -> LayerNorm(128) -> GELU -> Dropout(0.2)`. For directed physical edges `j -> i`, three relation primitives are summed:

- Translation: `W_diff (H_j - H_i)`.
- Multiplicative: `W_prod (H_i ⊙ H_j)`.
- Bilinear: `(U H_i) ⊙ (V H_j)`.

The modality relation is `E_m = GELU(LayerNorm(P_diff + P_prod + P_bil))`, with dimension 64. No endpoint feature concatenation is used. The shared relation evidence is

`E_S = Dropout(LayerNorm(GELU(Linear([E_T + E_V, |E_T - E_V|, E_T ⊙ E_V]))))`.

Modality-conditioned relations are `R_m = LayerNorm(E_m + W_S^m E_S)`. The optional local relation-context interface is reserved and disabled. Setting `relation_context: true` raises `NotImplementedError` with an M0.2 reservation message.

Task-adapted compatibility is

`c_ji^m = 2 sigmoid(softplus(raw_tau_m) cos(H_i^m, H_j^m) + w_c^m R_ji^m + b_c^m)`.

This gives values in `(0, 2)` for finite logits. `raw_tau` is initialized so `softplus(raw_tau) = 1`; each relation-correction linear layer starts at exactly zero, so the initial compatibility is a cosine-similarity prior.

## Stage II: transport, response, aggregation, and update

The two comparison modes change only the physical transport operator:

- `identity`: `T_ji = I`, implemented as the source state itself.
- `orthogonal`: each 128-vector is viewed as 16 groups of 8 values. A modality-specific `Linear(64, 8)` predicts angles from the Stage I relation. The angle head is zero-initialized and angles are `theta_max tanh(raw_theta)`, so transport starts exactly at identity.

The fixed two-sweep Givens schedule is `(0,1), (2,3), (4,5), (6,7)` followed by `(1,2), (3,4), (5,6), (7,0)`. The eight angle slots map to these eight pairs in order. Each rotation uses `x'_a = cos(theta)x_a - sin(theta)x_b` and `x'_b = sin(theta)x_a + cos(theta)x_b`. The implementation builds new component lists and stacks them, avoiding in-place autograd writes and preserving each group's L2 norm up to floating-point rounding.

For orders 1–3, transport angles and relations are computed once from H0 in a forward call and reused. With `S_0 = H_0`, the edge discrepancy is `delta_ji,k = T_ji S_j,k-1 - S_i,k-1`. The response is `Phi(delta) = delta + W2 GELU(W1 LayerNorm(delta))`, with dimensions `128 -> 64 -> 128`; W2 weight and bias are zero-initialized, so its initial behavior is the identity. Messages are `M_ji,k = c_ji Phi(delta_ji,k)`.

Incoming messages use mean, max, min, and `sqrt(max(E[x²] - E[x]², eps))`. Isolated-node statistics are explicitly zero. Every statistic is multiplied by identity, amplification, and attenuation degree scalers; the log-degree mean is computed over non-isolated nodes. The 12 resulting 128-vectors are concatenated, projected by `Linear(1536, 128)`, and dropped out without a ReLU. The anchored update is

`S_i,k = LayerNorm((1 - 0.1)(S_i,k-1 + C_i,k) + 0.1 H_i,0)`.

Text and visual each have four trainable zero-initialized GPR logits; softmax gives nonnegative weights summing to one. Each modality readout is the weighted sum of S0–S3, followed by `Linear([Z_text || Z_visual], 128)`. There is no adaptive cross-modal fusion.

## Why P0 artifacts are not used

M0 is the first model-design prototype. It directly consumes raw frozen Text and Visual features supplied by the formal dataset loader, and its projectors train jointly with the model. P0 semantic embeddings, probe splits, calibration data, and P0 artifacts do not enter training or model selection. This keeps the model under evaluation independent of the completed validation-stage artifacts.

## Initialization and controlled comparison

Both variants share projectors, relation encoders, compatibility, response, PNA aggregation, state updates, GPR readout, and fusion. Identity mode applies no learned map to edge states. Orthogonal mode adds only the relation-conditioned angle heads; zero initialization makes its initial operator exactly equal to identity. Relation compatibility and response also start at their specified stable priors. GPR weights start uniformly at 0.25.

## Numerical and memory considerations

Physical edges are converted to `long`, stripped of self-loops, symmetrized, and coalesced. The convention is always `source=edge_index[0] -> target=edge_index[1]`. No self-loops are added. Only this graph structure, degree, and degree scalers are cached by input graph identity; projected semantics and all relation/transport tensors are recomputed for each forward call.

Edge messages are processed in chunks of `edge_chunk_size` and reduced immediately into distribution statistics. This preserves full-edge semantics and autograd. Exact inference is full-graph; `batch_size` is accepted by the interface but intentionally does not change computation. Diagnostics sample up to 4096 directed physical edges deterministically for transport displacement and summarize conductance and angle values over all directed edges.

## Tests

`tests/test_relation_transport_m0.py` covers graph preprocessing, directed translation, relation/conductance shapes and prior, orthogonality and identity initialization, response identity, manual PNA statistics including isolated nodes, S0–S3, GPR normalization, model interface, exact inference equivalence, and relation/angle gradients. The recorded test result and run-time GPU smoke and pilot results are added to `docs/model_design/m0_pilot_report.md`.

## Known limitations

- The prototype fixes hidden size 128, relation size 64, three transport orders, the 8-dimensional Givens groups, two sweeps, PNA aggregation, and GPR readout.
- It has no local relation-context refinement; no Stage III provenance, recurrent update, attention, expert, prototype, optimal transport, rewiring, or auxiliary relation supervision.
- The identity variant has no angle-head parameters; orthogonal has modality-specific angle heads. Parameter counts are therefore reported for each variant.
- Diagnostics summarize relation quantities and do not retain full edge-level relation states.
- Results from this pilot describe a design prototype and do not establish a final-model claim.

## Reserved M0.2 relation-context interface

The modality-conditioned edge states `R_text` and `R_visual` are formed after shared cross-modal evidence and before compatibility and transport. M0.2 may refine local relation context at this interface. M0.1 deliberately performs no context propagation over relation states and rejects `relation_context: true` rather than silently changing its definition.
