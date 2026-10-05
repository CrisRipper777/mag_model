# Mechanism pilot v2 self-review

This is an implementation self-review, not an independent review.

## Scope and provenance checks

- Started from expected baseline `d1518bfe0087df35f399d8d6679b479b520977c2`; code/protocol/tests are frozen in source commit `70f658b266ad941df176f1f2c720785c03efbe37`.
- Worked on `codex/mag-mechanism-pilot-v2`. The user's deleted `CLAUDE.md` and untracked `docs/mechanism_pilot_v2_design.md` were not staged or modified.
- Movies/Grocery graph, two feature files, and seed-42 split byte counts and SHA-256 values were checked against the old v1 protocol lock before each worker and again during finalization. No v1/v1.1 file was changed.
- The committed training implementation and resolved configuration set `task.evaluate_test=false`. All 70 attempts record that setting; no test Acc, Macro-F1, CE, confusion matrix, checkpoint selection, or test-label-based troubleshooting is present in the new results. The only metric indices are train and validation.
- The train-run code hashes, baseline commit, locked input hashes, checkpoint/prediction/log hashes, calibration choices, and seed-42 reuse references are recorded in `run_registry.json`, `run_results.jsonl`, and `manifest.json`.

## Architecture and training-path review

- Text/image input projections, states, and modality routers are separate. Each layer's expert functions are shared across modalities. Layer 1 is same-modality mean propagation; layer 2 is the only optional dual-axis interaction.
- Sender modality softmax is over available send modalities for a single edge and receiver modality. Link attention is grouped by destination and receiver modality. Reciprocal graph edges remain two messages; duplicate directed pairs and self-loops are excluded consistently in every scheme.
- No feature magnitude is used to infer availability. These MAGB records contain no availability metadata, so both stored modalities are treated as available; an explicit availability mask is supported by the model.
- The source loss heads are created in the NC task and use only `train_idx`; the backbone receives no labels and has no class logits. P0/P2/P6 instantiate/train wide MLPs. P4 instantiates and trains its fixed uniform specialist mixture with no router.
- The wide MLP hidden width is 1025: 394,753 MLP parameters per layer versus 394,752 parameters in the four active hidden-256 experts. This matches active expert MLP parameters; it does not claim equal total parameters or compute.
- Every formal checkpoint contains the backbone, NC head, optional source heads, optimizer state, resolved config, best epoch, and source identity. Training reloads each serialized checkpoint and verifies the train/validation predictions and full validation confusion matrix.

## Test and run review

- `conda run --no-capture-output -n yhf_env python -m pytest -q tests/test_mag_collaborative.py tests/test_mag_probe.py`: **33 passed**. One PyG deprecation warning was emitted; no test failure remained.
- `conda run --no-capture-output -n yhf_env python -m py_compile src/models/mag_collaborative.py src/tasks/nc.py scripts/run_mechanism_pilot_v2.py tests/test_mag_collaborative.py`: passed.
- `git diff --cached --check`: passed before the training-code commit.
- Synthetic CUDA P3 smoke on `cuda:0` (RTX 3090): two optimizer steps on a 32-node, 62-directed-edge graph; finite loss, finite gradient path, repeatable eval output, 40.32 MiB peak allocated.
- Formal matrix: 42 LR calibrations plus 28 confirmations; **70/70 succeeded**, 0 failed. Both 3090s were selected from a pre-run inventory and used concurrently; no unrelated process was stopped. Highest recorded per-run peak was 4.93 GiB.
- A post-run integrity audit verified all 70 checkpoint, train/validation prediction, and epoch-log hashes; each selected scheme/dataset has seeds 42/43/44; all 70 logs include epochs 1, 10, 30, and best diagnostics; selected best-step parameter groups have finite, nonzero gradients and updates.

## Findings and limitations

- Validation does not show a benefit from the dual-axis change or a robust benefit from conditional experts over the matched wide/static controls. Source supervision is more encouraging in P5 than P6, but the effect is descriptive and limited to these two development graphs.
- The selected runs have sizeable train–validation gaps, consistent with overfitting pressure. The experiment does not identify whether overfitting, shared expert functions, competitive routing, or interaction optimization caused the results.
- The incoming-edge attention summary records the mean coefficient (0.11185) and maximum (1.0). The mean is determined by per-destination normalization and the fixed graph; the maximum is reached on degree-one groups. These aggregates do not discriminate learned attention quality. Full edge attention was intentionally not persisted.
- Router and sender-modality weights are training diagnostics, not semantic roles, causal utility, or modality information fractions. Text can still enter edge context, receiver states, residual paths, and FusionMLP.
- Three seeds are descriptive, not a significance design. There is no LP, test evaluation, five-NC expansion, new loss, or post hoc tuning grid in this round.
