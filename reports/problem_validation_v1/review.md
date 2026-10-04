# Independent review

## Verdict

The registered NC run completed as specified: 18/18 validation-only calibration jobs, 144 new primary fits plus six identity-matched calibration reuses, and 150/150 locked primary evaluations. There were 162 training attempts, no failed training jobs, no missing evaluations, and no superseded evaluation rows. This supports conclusions only for these five NC datasets, fixed splits, and probe definitions; it does not validate LP or a unified MAG model.

## Protocol and implementation checks

- The training source commit is `63867331e9d3373f19136e47ab192cc9fb014c97`. Every source-file hash stored in the registry was independently matched against that commit. The protocol lock is `3b949712c738dd99865f9bdce882cf65f91756e775cd4af6077cbd065d3b368f`.
- The matrix has 18 calibration slots and 150 primary slots (P1/P2/P3 = 54/24/72). The six reused jobs match dataset, variant, seed 42, split, LR, config, and source fingerprints. All training records have `test: null`; each formal primary slot has one successful test record.
- The data audit records explicit seed-42 MAGB splits for training seeds 42/43/44 and the official ele-fashion split, with split generation disabled. It verifies feature alignment/finiteness, split disjointness and label range, and raw/processed edge conditions. ele-fashion retains its official mapping and 12-class schema; label ID 5 is absent from the 11 observed classes, with no remapping.
- I reviewed the probe path, model factory, data loader/split behavior, and NC task. The 13 registered variants use the specified 128-dimensional head interface and sparse receiver-mean operator. The wide and concat controls use h=192 and b=26; their full model-plus-head parameter mismatches stay below 0.1% on every dataset. The router has 16,676 parameters.
- The final unit suite passes 15 tests. Two-step full-graph GPU smoke runs on Movies and ele-fashion had finite losses, nonzero finite gradients through adapters, shared phi, router, and head, valid dimensions, and no test evaluation. Runtime was `yhf_env` with DGL 2.4.0+cu121 / PyTorch 2.4.0+cu121 on `cuda:1`.

## Independent recomputation

- I recomputed validation and test confusion matrices, accuracy, and Macro-F1 from all 150 saved argmax files. Counts came from the complete configured class matrix; Macro-F1 used the locked observed-label set. The recomputed values match the reported rows. All 150 checkpoint validation predictions also exactly reproduce their stored validation confusion matrices and metrics.
- I independently recomputed 126 paired-correctness cells and 30 modality-complementarity cells from the saved predictions and fixed split indices. They match `paired_comparisons.json` and `error_complementarity.json`.
- I decoded and checked all 1,452 geometry bucket rows (2,904 complete confusion matrices): matrix dimensions and class order are explicit, sums equal bucket n, diagonal sums equal reported correct counts, and bucket small-n flags agree with n. The matrices use lossless zlib/base64 JSON encoding to keep the report directory compact.
- All 475 manifest-listed raw artifact hashes (162 checkpoints, 162 logs, 150 prediction files, and the smoke record) match the local files. No checkpoint, log, feature, logit, or embedding is tracked; per-node prediction artifacts contain only argmax labels and remain under ignored `outputs/problem_validation_v1/`.

## Results and claim limits

- **Q1, modalities:** J_E exceeds the better single-modality ego model by an equal-graph mean of 2.181 percentage points on four datasets and is 0.417 points lower on ele-fashion. Across all dataset/seed cells, text-only-correct and image-only-correct cases average 12.57% and 13.73%, with 17.28% both-wrong and 34.66% prediction disagreement. This indicates complementary errors in these probes, not a parameter-controlled estimate of modality information: single-modality models omit one adapter.
- **Q1, neighborhoods:** On Movies/Grocery, T_M−T_E averages +4.629 points and I_M−I_E +8.111 points; both are positive in all six paired seed cells. This is evidence for these one-hop neighborhood paths on the two exploratory graphs, not across all five datasets.
- **Q2, node-conditioned fusion:** No graph passes the registered engineering screen against both GLOBAL_WIDE and CONCAT. J_NODE is generally below the capacity-matched and ordinary nonlinear controls; its small positive deltas on some cells do not meet the threshold. The result favors simpler fusion for this tested setup and does not rule out conditional fusion under other training or data designs.
- **Q3, center paths:** On Movies/Grocery, J_FIXED−J_N averages +6.645 points and is positive in all six paired seed cells. This supports a center-path contribution in the tested setup, not uniquely extracted or causally identified center information.
- All comparisons are descriptive over three seeds (`ddof=0`); there are no significance claims. The common LR 0.003 was selected only from the registered six-cell validation mean. No test result selected a model or hyperparameter.

## Execution and aggregation notes

- A long-lived training-stage process paused at ele-fashion when `nvidia-smi` reported memory above the configured idle threshold. Those waits did not increment attempts or enter failures. The remaining exact registered jobs completed in separate `--job-id --resume` processes; the final registry is complete and below the 180-attempt cap.
- The initial locked test inference used the frozen evaluation source at the training source commit and wrote all 150 evaluation records and argmax files before aggregate generation. Aggregate-only runs then exposed missing imports (`SEEDS`, then `ATTEMPT_CAP`) and were repaired in follow-up commits. The final summary was rebuilt from the saved predictions and records; test inference was not repeated. `manifest.json` distinguishes the pinned evaluation source from the final aggregate code commit and hashes.
- Recorded training plus validation/test inference is 0.094529 GPU-hours; training-only is 0.063798 GPU-hours. These figures exclude the separate smoke and analysis/report-generation overhead. Peak allocated GPU memory was 2,143,803,904 bytes.
