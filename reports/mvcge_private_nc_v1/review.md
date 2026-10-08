# MVCGE private NC v1 self-review

This review was performed by the implementation author; it is not independent.

## Audit checks

- Train/validation prediction recomputation: `True` across 144 successful attempts. Counts and confusion matrices matched exactly; accuracy and Macro-F1 matched within 1e-7.
- Selected LR/WD recomputed from all nine seed-102 candidates for every dataset/model; confirmation rows match the immutable lock.
- Every successful checkpoint matches its run's source commit, source hashes and best step. The same source hash map is used across runs.
- Prediction, checkpoint, split-ID and input-file SHA256 values were verified.
- Runtime records report 0 test task-head calls and 0 real LP runs. The executed entry point loads NC only; its supervised and evaluation calls use train/validation indices. The synthetic LP test does not read any dataset split.
- Functional tests cover PC order/initial responses, direct-vs-reused basis and gradients, sparse/dense equivalence, masks/loops/isolation/permutation, top-2 gates, expected-load gradients, task-only router gradients, G0/G1 initialization, task-side heads, synchronous context input, optimizer grouping and deterministic eval.
- Smoke execution: nine train-only cells, three updates each; finite loss and clipped-gradient checks passed, no OOM.

## Cautions and residual risks

The MAGB loader discards any DGL edge data and does not expose modality masks; inspected graphs have no DGL edge-data fields, and this round treats all modalities as present. The actual S0 replay uses this round's clip value 5, unlike the previous S-stage's clip value 1, so comparisons are fresh recipe comparisons.
G1 and A0/G0/U1 are selected independently by validation. The confirmation validation was used in previous development. U1 is smaller and computes all four experts like the other models; it does not form a capacity- or compute-matched route ablation. Top-2 only gates receiver outputs and all experts are still computed.
No test metrics, real LP task, missing-modality intervention, five-dataset extension, causal analysis or significance test is part of this record.
