# M0.2: Pair Relation with Local Relation-Environment Context

## Research question

Does the local relation environment improve the interpretation of a physical relation?

M0.2 starts from the M0.1 pair-level multimodal relation state and builds a separate relation environment for each modality. For each physical directed edge `j -> i`, it computes target incoming and outgoing means and population standard deviations from the M0.1 pair relation states. Isolated nodes have zero statistics and zero context. The four statistics are encoded by a modality-specific `Linear(4*64,64) -> LayerNorm(64) -> GELU` block.

For each edge and modality, target-conditioned, source-conditioned, and environment-difference terms are projected and summed, normalized, and passed through GELU and a zero-initialized output projection. The fixed-scale residual is `0.5 * tanh(output)`, added to the pair relation. Consequently, initial contextual relation equals the M0.1 pair relation exactly. Context modules are created after the M0.1 modules, preserving initialization of shared parameters under a common random seed.

## Downstream path

The refined relation is used by the inherited M0.1 conductance correction and orthogonal Givens angle head. Transport, response, PNA aggregation, anchored state update, temporary readout, fusion, and the node classification interface are inherited unchanged. M0.2 fixes `transport_mode=orthogonal`, hidden size 128, relation size 64, three propagation orders, group size 8, two Givens sweeps, and context residual scale 0.5.

## Analysis outputs

`analyze()` preserves M0.1 scalar diagnostics and adds context norm, relative contextual residual, absolute conductance change relative to the pair-only path, and mean absolute angle change. `forward_without_relation_context()` is an analysis-only frozen counterfactual using the trained M0.2 base parameters. Neither path stores edge-level context tensors on the model.

## Experiment scope

The pilot uses Movies and Grocery, seeds 42/43/44, full-graph NC training, original validation for model selection, and test/LP disabled. M0.1 references are the existing orthogonal pilot runs only. This model is a pair relation plus local relation-environment contextualization; its downstream Stage II transport matches M0.1. M0.2 findings and the decision about retaining context are reported in `m02_pilot_report.md` after the pilot completes.
