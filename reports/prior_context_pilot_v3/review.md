# Prior-context MAG pilot v3 self-review

This is the implementation and evidence self-review by the authoring agent. It is not an independent external review.

## Scope and protocol checks

- Baseline was `1bc436b3eab2298fee7994c468375ac06e8707c0`; the new branch is `codex/mag-prior-context-pilot-v3`.
- The training source commit and source-file digest are recorded in `docs/prior_context_pilot_v3/protocol_lock.json`. All formal attempts use that lock and the v1 frozen input/split hashes.
- Only Movies/Grocery NC train and validation indices were used. The loader's full transductive features/graph were passed to the backbone; test labels were never indexed, test predictions/metrics were never generated, LP was not run, and no other NC dataset was trained.
- All 30 Stage A cells succeeded with no formal training retry. The pre-registered gate failed on validation and resource criteria, so all 40 Stage B cells are recorded as `skipped_gate` and were not launched.
- Checkpoints, epoch logs, and train/validation argmax remain under ignored `outputs/prior_context_pilot_v3/`; the Git report contains only aggregated metrics, confusion matrices, and provenance.

## Checks run

- `conda run --no-capture-output -n yhf_env python -m py_compile scripts/run_prior_context_pilot_v3.py scripts/audit_prior_context_pilot_v3.py src/models/mag_prior_context.py src/tasks/prior_context_pilot.py tests/test_mag_prior_context.py` — passed before source freeze.
- `conda run --no-capture-output -n yhf_env python -m pytest -q tests/test_mag_prior_context.py` — 19 passed; one upstream PyG distributed deprecation warning.
- A three-step synthetic R4 CUDA smoke on `cuda:1` passed, including all active backbone parameters changing, optimizer reuse, `set_epoch`, fixed-sample router diagnostics, finite deterministic evaluation, and test-off.
- A synthetic R6 CUDA/DGL 2.4.0+cu121 forward/backward/inference-mode smoke passed.
- Parameter-count planning instantiated all seven variants on both graphs; R0/R1/R2 were within ±5% of the R4 active-backbone count.
- `scripts/audit_prior_context_pilot_v3_postrun.py` recomputed the 30 saved train/validation confusion matrices, Acc and fixed-label Macro-F1, LR selections, paired deltas, and the Stage A decision. It verified input/source/protocol identities, checkpoint and prediction hashes, all declared raw evidence, and report hashes.

## Issues found and disposition

1. The first default-shell GPU smoke could not see `/dev/nvidia*`; PyTorch reported no CUDA devices. No step started. A host GPU execution context showed both RTX 3090s idle, and the authorized retry on `cuda:1` passed. Both attempts are listed in `outputs/prior_context_pilot_v3/micro_attempts.json`.
2. An early synthetic DGL cache test supplied integer features to a float Linear layer. The fixture dtype was corrected before formal work; no model code or formal run was affected.
3. Before formal work, the fixed-sample diagnostics test caught that router/structure outputs were not being collected for summaries. The dedicated diagnostic forward was corrected and all R0–R6 sampling tests passed before source freeze.
4. After Stage A, the frozen first-pass audit harness falsely treated the static `dataset.lp_test_ratio` config field as a test result, omitted `status` from its success count, and failed to return graph-level gate means it then checked. Its failed output is preserved at `outputs/prior_context_pilot_v3/audit_attempts/frozen_auditor_attempt_001.*`. A separate supplemental evidence auditor fixes only these post-run checks, retains the original training/protocol source hashes, and passes from the frozen predictions. This correction did not alter training, selection, or any checkpoint.
5. The normal per-epoch training path does not calculate extra module-wise gradient reductions. At registered diagnostic epochs, per-module pre-clip norms are reconstructed from observed post-clip gradients and the global clip coefficient; the exact global pre-clip norm is returned by PyTorch. The synthetic diagnostic test checked this reconstruction against directly measured pre-clip norms.
6. The frozen report generator selected every seed-52 LR calibration row when building a three-seed mean, so its aggregate validation table was empty. The initial report is archived under `outputs/prior_context_pilot_v3/audit_attempts/report_generation_attempt_001.md`. The supplemental post-run auditor rebuilds that table using only the selected seed-52 checkpoint plus seeds 53/54, adds aggregate R4 route/S-D/readout-health summaries from those same checkpoints, and hashes the corrected report; no run, checkpoint, or gate value changed.

## Remaining limitations

- Stage B is intentionally unrun because Stage A did not pass. No claim is made about R2/R3/R5/R6 validation performance on these graphs.
- The comparison covers two previously studied development graphs and two confirmation seeds. Results are descriptive, not statistically significant; development-set exposure and parameter-count versus compute mismatch remain limitations.
- The audit is self-review, not independent review. It verifies train/validation classification evidence and protocol identity; it does not independently reproduce floating-point cross-entropy from unsaved logits.
- No test evaluation, LP experiment, five-NC extension, additional seed, or post-result tuning was performed.
