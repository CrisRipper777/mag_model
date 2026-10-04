# MAG NC problem validation v1

## Scope and claims

This is a finite node-classification probe on Movies, Toys, Grocery, Reddit-S, and ele-fashion. It addresses three questions: whether text/image and ego/neighborhood paths add predictive value and error complementarity; whether node-conditioned fusion beats global, capacity-matched, and ordinary nonlinear fusion; and whether direct center paths contribute. It does not establish a general MAG model, a mature MoE, causal path utility, role identifiability, or strict shared/private information separation. Link prediction on sports-copurchase and cloth-copurchase is outside this run.

## Frozen representation

Each used modality is row-L2-normalized with epsilon `1e-8` (zero rows remain zero), passed through a biased modality adapter to 128 dimensions, and through the shared `Linear(128,h) -> LayerNorm(h) -> GELU -> Dropout(0.2) -> Linear(h,128)` map. The base width is 128. The four ordered paths are `[E_t,N_t,E_i,N_i]`; `N_m=P E_m`.

`P` is a sparse receiver-mean operator with indices `[dst,src]`, no self-loops, duplicate arcs removed, and values `1/indegree(dst)`. It is one-hop, preserves the supplied directed edge semantics, and maps isolated receivers to zero. Only `P` is cached; degree and isolation are derived from its sparse row indices. Trainable `E` and `P E` are recomputed every forward. The cache is keyed by graph tensor identity, storage, in-place version, shape, node count, device, and dtype, and is absent from the model state dict.

Single-modality variants instantiate only the adapter they use. All variants produce 128-dimensional representations and use the same external linear NC head. All are run as full-graph models. `J_NODE` uses a 516-to-32-to-4 router with GELU, temperature 1, and zero output-layer initialization. `J_NODE_SHUFFLE` permutes complete router-context rows by a label-independent CPU generator keyed by run seed, epoch, and namespace; it leaves receiver paths and their activity mask in place. Training uses an epoch-specific permutation; validation and test use a fixed `validation_test` permutation. No auxiliary or router regularization loss is used.

`J_GLOBAL_WIDE` and `J_CONCAT` widths are computed by exhaustive integer search from the full model-plus-head parameter counts. The selected widths and relative mismatch are recorded per dataset. A match outside 1% is explicitly marked unmatched.

## Run matrix and selection

The machine-readable matrix is `matrix.json` beside this document. It contains 18 validation-only LR calibration slots and 150 primary slots. Calibration uses Movies/Grocery × J_FIXED/J_GLOBAL/J_NODE × seed 42 × `{3e-4,1e-3,3e-3}`. The common LR is chosen by equal-weight mean best-validation accuracy across the six dataset/model cells; ties within `1e-6` prefer `1e-3`, then `3e-4`, then `3e-3`. Only six identity-matching selected-LR checkpoints can be reused in the primary matrix. The planned number of unique training jobs is 162.

The primary matrix is fixed regardless of exploration outcomes: P1 has nine listed variants on Movies/Grocery (54 jobs); P2 has J_NODE/J_NODE_SHUFFLE/J_GLOBAL_WIDE/J_CONCAT on Movies/Grocery (24); P3 has T_E/I_E/J_E/J_FIXED/J_GLOBAL/J_NODE/J_GLOBAL_WIDE/J_CONCAT on Toys/Reddit-S/ele-fashion (72). Every primary cell has seeds 42/43/44.

MAGB split files are explicit seed-42 files for every training seed, with split generation disabled. ele-fashion uses its official split with generation disabled. Every training slot has one run and the training RNG seed is distinct from the split identity. Test evaluation is disabled throughout calibration and training. Checkpoints are selected only by validation accuracy (`min_delta=1e-4`); validation Macro-F1 is reported but never selects a checkpoint.

## Training and evaluation

Training uses AdamW, weight decay `1e-4`, 300 maximum epochs, one full-graph optimizer step per epoch, train-index cross-entropy only, gradient clipping at 1, validation every epoch, patience 30, minimum epoch 30, `min_delta=1e-4`, no scheduler, and zero auxiliary loss. A single GPU is used for one job at a time. The 12 aggregate GPU-hour budget is a soft planning target; 180 attempts is the hard limit including retries.

`evaluate_locked_probe.py` evaluates the frozen formal matrix after training without retraining. It verifies code, config, split, checkpoint, and evaluation fingerprints; evaluates each formal test split once; skips a matching completed fingerprint on resume; and retains older buggy results marked superseded. Confusion matrices always include all `C` labels in order `0..C-1`. Macro-F1 averages only the fixed label set observed across supervised splits, while TP/FP/FN come from full matrix row and column sums. That label set is metadata fixed before test predictions; it does not use performance to select a model.

## Diagnostics and interpretation

Per-node raw predictions, embeddings, and logits are never written. The analysis outputs only aggregate counts and metrics. Text/image error complementarity uses the four correctness cells, prediction disagreement, and the label-using oracle accuracy `1 - both_wrong/n`; the oracle is an undeployable upper bound. Paired NODE-versus-control counts use identical split nodes. Node buckets use training-node degree and modality-specific raw ego/neighborhood cosine quartiles, then apply those thresholds to validation nodes; isolated nodes have their own bucket. No cross-modal cosine is computed. Train-train labeled-edge homophily excludes validation/test labels.

Router weights and path selection rates are descriptive and are not causal utilities or mature expert loads. The ten fixed validation permutations for a trained NODE checkpoint are separate perturbation results, never logit-ensembled or treated as new models. With three seeds, the report gives paired seed differences and population standard deviation (`ddof=0`) without significance claims. The pre-registered engineering screen compares NODE independently with GLOBAL_WIDE and CONCAT: at least three of five graphs must have paired mean accuracy gain of 0.3 percentage points, at least two of three seeds must agree in direction, and mean Macro-F1 must not fall by more than 0.3 points; the same three graphs must pass for each control.

## Artifacts and resume

Small tracked evidence lives in `reports/problem_validation_v1/`. Training logs, checkpoints, profiling output, and any intermediate predictions remain under ignored `outputs/problem_validation_v1/`. The launcher emits deterministic job IDs, executable commands, and a persistent registry; resume skips a successful job only if its fingerprint matches. See the report and manifest for exact commands, source commit, hashes, attempts, and raw evidence paths.
