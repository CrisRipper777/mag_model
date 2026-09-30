# M2: Provenance-Conditioned Evidence Retrieval

## Research question

M2 asks how structural semantic evidence produced through different interaction histories should be selectively used with intrinsic multimodal semantics. Its evidence unit is the consecutive semantic increment paired with its provenance state:

\[
(\Delta S_{i,k}^{m},G_{i,k}^{m}),\qquad
\Delta S_{i,k}^{m}=S_{i,k}^{m}-S_{i,k-1}^{m},\quad k\in\{1,2,3\}.
\]

M2 inherits the frozen M1 Stage I relation induction, semantic conductance, orthogonal Givens transport, semantic states \(S_0\ldots S_3\), interaction atoms, path-history provenance, and recurrent provenance states \(G_0\ldots G_3\). M0.2 relation context is disabled. M2 does not feed provenance back into semantic propagation and does not add self-attention among evidence tokens.

## Evidence construction

For each modality, a learned projection and LayerNorm map each increment to semantic content:

\[
C_{i,k}^{m}=\operatorname{LN}_{\Delta}^{m}(W_{\Delta}^{m}\Delta S_{i,k}^{m}).
\]

M2-P also maps provenance to the evidence dimension:

\[
P_{i,k}^{m}=\operatorname{LN}_{G}^{m}(W_G^m G_{i,k}^{m}).
\]

The semantic key and value are

\[
K_{i,k}^{m,S}=\operatorname{LN}_{K}^{m}(C_{i,k}^{m}+o_k+u_m),\qquad
V_{i,k}^{m,S}=\operatorname{LN}_{V}^{m}(C_{i,k}^{m}).
\]

M2-P conditions the key and value on provenance:

\[
K_{i,k}^{m,P}=\operatorname{LN}_{K}^{m}(C+W_{KP}^{m}P+W_{KX}^{m}(C\odot P)+o_k+u_m),
\]
\[
V_{i,k}^{m,P}=\operatorname{LN}_{V}^{m}(C+W_{VX}^{m}(C\odot P)).
\]

The order and modality embeddings are included in keys only. Provenance-specific heads \(W_{KP},W_{KX},W_{VX}\) are zero-initialized, so M2-P starts with exactly the same evidence tokens as M2-S. All modules are instantiated in both modes with the same order.

The bank order is text-k1, text-k2, text-k3, visual-k1, visual-k2, visual-k3. A learned null key is appended with an exactly zero value; attending to it means no structural semantic update.

## Intrinsic-query retrieval

Text and visual queries are computed independently from \(H_{i,0}^{text}\) and \(H_{i,0}^{visual}\). They are not fused before retrieval. A node-local four-head cross-attention reads the seven-token bank, with softmax over evidence tokens. Attention and output projections have no bias. M2 has no evidence self-attention.

Each retrieval output is dropped out and added to its matching intrinsic state, followed by a modality LayerNorm. The two updated modality states are concatenated and passed through a 256-to-128 linear layer, LayerNorm, GELU, and dropout 0.2. The final node representation has dimension 128.

## Frozen M1 readout parameters

M2 freezes `gamma_logits_text`, `gamma_logits_visual`, M1 `fusion`, and M1 `provenance_adapter`. It does not use M1 `base_z`, `provenance_residual`, or `z` for its output. They remain present because the M2 implementation inherits M1's frozen Stage I/II helper for formula correctness.

> Legacy parameters remain only because M2 inherits the frozen M1 implementation for correctness; they are not part of the M2 conceptual architecture and should be removed in the later architecture-freeze refactor.

## Analysis and frozen interventions

Aggregate analysis records \(\Delta S\) norms; provenance-conditioned key/value changes; retrieval residual ratios; cosine change from intrinsic states; and sampled attention mass, entropy, effective token count, and node-wise attention variance. At most 8192 deterministically chosen nodes are used for attention diagnostics. Attention weights describe retrieval behavior and are not causal importance scores.

The frozen M2-P audit holds the checkpoint, classifier, graph, H0/S/G, and all learned parameters fixed while applying Provenance-Off, degree-matched node shuffle, provenance-order mismatch, and same-modality-only retrieval. The node shuffle uses one shared permutation for both modalities and all three orders. Counterfactual evaluation reuses the exact same Stage I/II tensors, and asserts their maximum absolute difference is at most 2e-6.

## Pilot protocol

The M2 comparison is NC-only on Movies and Grocery with seeds 42, 43, and 44. M2-S and M2-P are retrained independently with matched initialization seeds. Original training labels fit the model, the original validation split selects checkpoints, and test evaluation and link prediction are disabled. No five-dataset benchmark, formal ablation, or LP run is part of this stage.
