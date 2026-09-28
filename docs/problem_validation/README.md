# P0.0 + P0.1 Problem Validation

This analysis tests the limited hypothesis that semantic similarity is not an adequate surrogate for the task utility of a physical relation. It also measures whether Text and Visual message utility can differ on the same physical relation. It does not implement or evaluate P0.2/P0.3 or a final model.

## Protocol

The fixed NC datasets are Movies, Toys, Grocery, ele-fashion, and Reddit-S. MAGB data are always loaded with split seed 42; ele-fashion uses its official split. A stratified 80/20 `probe_train` / `probe_calib` split is made once inside original `train_idx` with seed 42 and cached under `outputs/problem_validation/splits/`. Model initialization, dropout, and optimizer randomness use run seeds 42, 43, and 44. Run the experiment once with all three seeds in the same runner invocation (`--seeds 42 43 44`); this is one run containing three independent seeded model fits, each with its own output directory, rather than three one-seed runner invocations.

P0.0 uses independent Text and Visual `Linear → LayerNorm → GELU → Dropout(0.2)` projectors to 128 dimensions, concatenates the two outputs, and applies one linear classifier. It uses AdamW (lr 1e-3, weight decay 1e-4), cross-entropy, gradient clipping at 1.0, at most 300 epochs, and calibration-accuracy early stopping (patience 30, minimum epoch 30, minimum improvement 1e-4). Original validation metrics are computed only after the checkpoint is frozen and are marked `heldout_analysis_val`.

P0.1 freezes the P0.0 projectors. The physical graph is made undirected, self-loops are removed, and duplicate relations are coalesced. For each node, each modality message is the unweighted incoming-neighbor mean, with zero for isolated nodes. A single linear head receives `[H_text, H_visual, N_text, N_visual]`. It is trained on `probe_train` and selected on `probe_calib`. Edge logit contributions are computed analytically from the message blocks of that linear head.

For a relation `j → i`, CE utility is `CE(logits_without_message, y_i) - CE(logits_full, y_i)`. **`U > 0` means the removed message was beneficial**: removing it increases loss. `U < 0` means it was harmful. Margin utility is a secondary robustness measure.

Only original validation nodes are P0.1 targets. At most 10,000 targets are selected by class-stratified sampling with seed 42. Up to 64 incoming physical relations per target are selected with seed 42. The target and relation sample is shared by every model seed. Raw and frozen-probe cosine similarities are compared with CE and margin utility. Similarity quantiles are computed over the complete sampled edge population for each dataset × seed × modality. Confidence intervals use target-node cluster bootstrap with 1,000 replicates and seed 42. Three-seed standard deviations use population standard deviation and are reported separately from bootstrap intervals.

The runner checks split disjointness, original-validation target membership, target/test-index disjointness, absence of self-loops, deterministic edge populations, finite utilities, and analytic-versus-explicit message-removal logits (`max_abs_error < 1e-5`). The isolated-node mask is stored with the message-probe and edge-analysis artifacts. Test labels and test performance are not used or reported.

## Reproduction

```bash
conda activate yhf_env
cd /hdd1/DataInHere/YHF/mag_model
python scripts/run_problem_validation_p01.py \
  --datasets Movies Toys Grocery ele-fashion Reddit-S \
  --seeds 42 43 44 --device cuda:0
```

Smoke run:

```bash
python scripts/run_problem_validation_p01.py \
  --datasets Movies --seeds 42 --device cuda:0 --smoke
```

Regenerate aggregate CSVs, figures, and report from existing run outputs:

```bash
python scripts/summarize_problem_validation_p01.py
```

## Outputs

The tracked result bundle is under `results/problem_validation/`: it contains aggregate CSVs, all per-seed metrics JSON files, the split audit, diagnostic figures, and a self-contained report. The report and figure links are also maintained in `docs/problem_validation/p0_p1_report.md`. Raw tensor run artifacts are under `outputs/problem_validation/<dataset>/seed<seed>/`; probe split caches are under `outputs/problem_validation/splits/`. These large intermediates stay ignored by Git. After regenerating outputs, refresh the tracked result bundle before publishing updated results.
