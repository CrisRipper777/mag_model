# Experiment protocol: `mvcge_coordinated_nc_v2`

## Intake, data, and scope

Run only Movies, Grocery, and Reddit-S NC with loader split seed 42. The loader reads the stored graph, features, full label vector, and split object. The experiment copies only train/validation labels into a sentinel-filled task tensor, retains train/validation IDs, then drops test IDs/labels. The model uses all transductive features and graph topology. Do not call the generic NC evaluator, which also evaluates test. Do not run test metrics, real LP, or Toys/ele-fashion.

Use the existing frozen float32 text-then-image features without added normalization. There is no exposed edge-weight or availability metadata. The runner uses all-true modality availability and rejects other masks. Construct the row-mean graph operator from the existing undirected/deduplicated list, remove loops and duplicate directed pairs, and use explicit `num_nodes`; isolates receive zero rows.

## Candidates and training budget

Models run in fixed order A0, N0, M0, M1, P0, P1. Calibration uses seed 112 and the Cartesian grid LR `{0.0003,0.001,0.003}` by weight decay `{0.0001,0.001,0.01}`, LR outer and decay inner, config indices 0-8. This is 3 graphs × 6 models × 9 configurations = 162 cells. After all 162 succeed and prediction audit passes, select each graph/model configuration by validation correct count descending, fixed-C macro-F1 descending, then lower config index. Freeze the 18 choices before confirmation. Confirmation runs seeds 113, 114, and 115 for all 18 graph/model groups (54 cells); 216 total planned cells.

Use full-batch training, max 1000 updates and min 200. Linear LR warmup spans updates 1-20 from 0.1× to 1× the target, then stays constant. Evaluate first at update 21, then every five updates, plus update 1000 if needed. Stop after 150 optimizer updates without a new best, not before 200. Choose checkpoints by `(validation correct count, fixed-C macro-F1, -update)`; exact ties keep the earliest. Validation and checkpoint selection are development evidence, not independent testing. Confirmation does not select a new configuration or rescue a poor seed. Do not extend the grid or add a performance gate.

Evaluation always uses clean top-2 gates for routed models, including a best checkpoint saved while training gates were dense or transitional. A0 uses its complete historical B1 encoder and main/path/reconstruction/pair terms. N0 uses uniform gates without a router or balance term. M0/P0 add the encoder's already weighted balance loss. M1/P1 add 0.05 times the mean text/image semantic CE. No label smoothing, test-head calls, or extra self-supervised objectives.

## Failure and recovery

Keep every attempt and classify failures as infrastructure, numeric/implementation, or poor validation performance. Do not discard completed attempts or silently reduce dimensions, sample neighbors, or alter the grid. A code change that affects training invalidates affected runs; freeze a new implementation version and rerun those cells. Resume only when model, optimizer, training RNG, NumPy/Python RNG, router-noise generator, A0 pair generator, update count, and best/patience state are restored. Otherwise create a new attempt and explain the interruption.

## Diagnostics

Write lightweight per-update losses, group learning rates, gradient norm, update time, and validation metrics when evaluated. Save only best train/validation IDs, targets, argmax predictions, confusion matrices and metrics, plus checkpoints/logs/diagnostics under ignored `outputs/mvcge_coordinated_nc_v2/`.

At every best checkpoint, record train/validation class metrics; branch norms for projected `p`, initial states `H0`, graph increments `delta`, normalized increments `g`, `c0`, graph updates and adapter residuals; per-channel delta variance and LayerNorm inverse-standard-deviation summaries. For M/P, separately run complete clean and isolated-noise diagnostic forwards and report per-layer, per-modality expert coverage (denominator N), top-2 slot share (denominator 2N), unordered expert-pair counts, mixture weights, entropy and sigma. P additionally records theta, alpha1/alpha2, and neighbor polynomial response over [-1,1]; this is not a whole-model operator norm claim.

For confirmation new models, take gradient diagnostics on the same training forward at updates 1, 50, 100, 200, and the best clean-eval checkpoint. Use `autograd.grad` without writing `.grad`, advancing optimizers, or changing the training RNG. Split each layer's shared-expert main-CE gradient by text/image mixed-message output and report the VJP reconstruction error against the direct gradient. Compare main CE with weighted balance gradients on router parameters and main CE with weighted auxiliary gradients on semantic frontends. Zero-gradient cosine values are null. These are message-exit decompositions, not modality-causal effects. If diagnostic time exceeds 10%, keep best and update-100 core diagnostics and report the actual trimming; never trim a training cell.

## Audit and reporting

An independent audit script must recompute all saved train/validation accuracy, fixed-C macro-F1, supports and confusion matrices from saved IDs, targets and argmax; verify all 162 calibration cells before making the selection lock; then verify all 54 confirmation cells and their selected settings. Check each result's source/input/split hashes, call counters, test/LP fields, checkpoints, and output manifest. Use integer equality for counts and `1e-7` tolerance for float32-derived metrics. This is implementer self-audit, not independent peer review.

Report confirmation means and population SD (ddof 0) for Acc and fixed-C macro-F1, plus per-seed values and train metrics. Report calibration separately. Give same-seed paired differences for M0/M1/P0/P1 versus A0, M0-N0, P0-M0, M1-M0, P1-P0; average only across the three graphs for a cross-graph descriptive summary. `(P1-P0)-(M1-M0)` is a descriptive interaction under independently selected recipes, not a fixed-recipe causal interaction.

Include selected LR/WD, best updates, budget-hit counts, train-validation gaps, parameter counts, training/evaluation/diagnostic wall time, median stable update time (update >=110 and outside diagnostics), peak allocated/reserved memory, and observed GPU co-load. Explain N0's missing router parameters and A0's historical 768D padded readout; do not claim capacity matching. Mark boundary LR/WD selections and runs that reach the update limit as unresolved optimization. Make no significance, broad generalization, causal, or LP-performance claim.

## Git delivery

Freeze this protocol, model spec, matrix, source/tests/config hashes, input hashes, and split hashes before formal runs; commit that source package first. Commit only explicit v2 source, tests, configs, docs, and concise reports. Never stage output arrays, checkpoints, raw predictions, step logs, dataset files, credentials, or original user workspace edits. Push the authorized branch without force, verify the remote branch SHA, then stop for review.

## Protocol amendment 2: gradient diagnostic aggregation

The original pretraining lock is preserved as `protocol_lock_v1.json` (SHA-256 `9f8b01c4e746b0a4e73f77e9717e044d6e24144697c50c78ee8923a61bafba5b`). Calibration completed and passed its 162-cell prediction audit under that lock; the immutable 18-recipe selection lock and its choices are retained unchanged.

During the first confirmation pass, all 36 M0/M1/P0/P1 attempts completed their training loop and reached best-checkpoint reporting, then failed when the router gradient comparison concatenated only non-`None` gradients. The two objective gradients can use different parameter subsets, so omitting those slots produced vectors of unequal lengths. A0/N0 are unaffected. The amended implementation inserts zero tensors for unused slots while preserving parameter order, and a regression test covers this case. This is a diagnostic-only correction: it does not change model forwards, losses, optimizer steps, random-number use, data, split IDs, or the locked selection recipes.

The failed first attempts remain in the ignored run registry and output directory. Retry only those 36 routed confirmation slots under the amended source and protocol lock. Retain the 18 successful A0/N0 confirmation records from the original lock. The confirmation audit must accept these two explicit lineages and verify each result and checkpoint against its matching lock; no other historical source version is permitted. Record both commits and lock hashes in the concise final report and manifest.

## Protocol amendment 3: retry dispatch

The diagnostic amendment lock is preserved as `protocol_lock_v2.json` (SHA-256 `29b91a6e03cd1fd9458e4a049c9ba8478d51b539039ee7192c4e48e027dc85ae`). Its first `--retry-failed` dispatch exited while deriving the first retry ID: the base attempt ID was treated as a numeric retry suffix. The failure occurred before model construction, and no confirmation training attempt was started under v2; the existing 18 successes and 36 failures remain unchanged.

The retry ID parser now considers only numeric suffixes after `_retry`, with a regression test for a base attempt and nonconsecutive retry IDs. This amendment changes run-control bookkeeping only. The final lock supersedes v2 while retaining the original v1 lock as the only historical provenance allowed for the unaffected A0/N0 cells. Retry the 36 routed confirmation cells under the final lock and keep the earlier failed attempts intact.
