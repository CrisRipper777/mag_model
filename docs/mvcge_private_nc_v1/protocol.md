# MVCGE private NC v1 experiment protocol

## Frozen inputs and scope

Only Movies, Grocery, and Reddit-S node classification are included. The canonical MAGB seed-42 NC train/validation split is used. Test split size is recorded for the intake inventory, but test IDs/labels are not copied to the runner's task-label tensor, test task heads are never called, and test metrics are never calculated or used. Real LP and the Toys/ele-fashion datasets are outside scope.

The existing `load_mag_data` path and feature arrays are used as-is: text then image, float32, no extra normalization. The loader does not return modality-availability metadata, so the canonical mask is all-present. The observed feature arrays are finite and have no all-zero rows; this diagnostic did not define the mask. No node is removed or modified. Input file and train/validation split hashes are recorded in `reports/mvcge_private_nc_v1/data_profile.json`.

The DGL files have no edge data or weight fields. The canonical loader reads edge indices only, removes raw self-loops, applies `make_undirected=true`, and leaves self-loop addition to the model. Raw Movies and Grocery edge lists are directed; Reddit-S is already symmetric. The post-loader lists are symmetric, loop-free, and contain no duplicate directed rows. A0 receives this unchanged loop-free edge list as its historical model did. The new encoder removes any existing loops and adds exactly one loop per observed node before constructing normalized sparse P. Thus the same canonical physical edge list is shared, with self-loop behavior defined by each model's frozen specification.

No label enters an encoder. Train CE indexes only train IDs. Validation applies the task head only to validation IDs. The report's primary Macro-F1 uses all C=20 labels `[0,…,19]`, `zero_division=0`; this matches the previous fixed-20 S protocol. Accuracy and the checkpoint's correct-node count use the same fixed validation IDs.

## Matrix and selection

`matrix.json` fixes order and IDs for 108 calibration cells: dataset order Movies, Grocery, Reddit-S; model order A0, G0, G1, U1; LR outer order 0.0003, 0.001, 0.003; matrix WD inner order 0.0001, 0.001, 0.01; seed 102. Each dataset/model has nine cells. Historical S0 selected configurations are Movies 0.003/0.001, Grocery 0.001/0.001, Reddit-S 0.001/0.001, all already in the grid.

Each dataset/model independently selects a calibration config by validation correct count descending, fixed-20 Macro-F1 descending, then lower fixed config index. No test metric is available to this selection. The selection lock is written only when all 108 cells succeed. Confirmation uses the locked config for seeds 103, 104, and 105, for 36 cells. Confirmation does not have a performance gate and does not reselect. Calibration seed 102 is excluded from confirmation means.

## Training and checkpoint rules

Every run is full-batch, float32, one optimizer update per step, max 800 updates, min 200. Updates 1–20 linearly warm each optimizer group from 0.1× to its target LR; the target rate remains constant after update 20. Validation starts at update 21, then every five updates. Early stopping cannot occur before update 200 and patience is 150 optimizer updates since the most recent checkpoint-key improvement. There is no scheduler. The checkpoint key is `(validation correct count, Macro-F1, -step)`; the earliest step wins a full tie. The best checkpoint is reloaded and train/validation are recomputed in eval mode.

G0/G1 optimize `CE + 0.01 × mean_layer(balance_loss)`. Dense warmup uses importance CV² only. Sparse top-2 training uses importance CV² plus differentiable Gaussian expected-load CV². U1 optimizes CE only. A0 preserves S0's main/path/reconstruction/pair weights (`1.0 / 0.05 / 0.01 / 0.01`). All models use this round's AdamW family, 20-step LR warmup, and global clip 5; A0 retains its original parameter decay partition but not the old round's gradient clip 1.

The runner records per-update train CE, balance and weighted-balance/CE ratio, optimizer group LRs, update time, and fixed-step gradient diagnostics. At updates 1, 20, 21, and 100, G0/G1 record task-only router-logit gradient norm separately from combined-loss gradients. The best-checkpoint diagnostics record per-modality clean/noisy expert use, top-2 combinations, gate entropy, σ; θ and polynomial response norms; public/private/state norms; and optimizer groups. Outputs include best-checkpoint train/validation IDs, labels, argmax, exact confusion matrices and hashes. Checkpoints, prediction arrays, and step histories stay under ignored `outputs/mvcge_private_nc_v1/`.

## Execution and scope boundaries

The train-only smoke set is nine 3-update full-graph runs: G0/G1/U1 × the three datasets. Smoke never calls validation and is not a matrix cell. The functional tests also check a synthetic graph and the existing endpoint-product LP decoder only; they do not invoke a real LP split or benchmark.

The model can construct per-modality induced sparse graph operators for masks, but the real study contains no missing-modality metadata and does not establish performance for one- or two-modality-missing cases. No artificial missingness is introduced. The new PC basis is the specified T=I−P parameterization; the study does not claim a verbatim PCNet reproduction or a theoretical guarantee for the full routed model.

Three confirmation seeds provide descriptive estimates, not significance or causal evidence. These validation sets were used in earlier development and are not independent benchmarks. Different selected LR/WD values make G1−A0, G1−G0, and G1−U1 comparisons differences between independently tuned recipes. U1 is not a capacity-matched routing ablation. No single expert-use pattern, gate entropy or balance score is treated as proof of semantic specialization.
