# Problem validation v1.1 diagnostic design

## Scope

This revision repairs four narrow issues in the `mag_probe` path: an explicit context-row intervention for both routed variants, task-level optimizer fallback for probe configs, a normalized-neighbor geometry diagnostic, and safe topology caching around inference tensors. It does not introduce a new model, objective, optimizer schedule, or training protocol.

The frozen diagnostic reads the v1 `J_NODE` and matching `J_GLOBAL` checkpoints for Movies, Toys, Grocery, Reddit-S, and ele-fashion at seeds 42/43/44. It runs forward passes only and scores train/validation nodes. It does not run LP, perform a new real-data optimizer step, or infer on test nodes. Full graph features/topology remain available for label-free forwards under the existing transductive setup.

## Context intervention

`set_context_permutation()` is an explicit diagnostic override. It permutes complete 516-column context rows before the router for `J_NODE` and `J_NODE_SHUFFLE`. The four receiver-owned path representations and active mask are built before that permutation and remain attached to their original node rows. With no override, `J_NODE` remains unpermuted and `J_NODE_SHUFFLE` retains the original deterministic seed/epoch/namespace policy.

The frozen diagnostic uses ten deterministic, label-free permutations per checkpoint. Each is reported separately as an intervention repeat, never as another training seed or a logit ensemble. A train-node mean router gate is another fixed intervention on the same representation; it is not a retrained `J_GLOBAL` model.

## Optimizer and cache behavior

The probe model preset omits `lr` and `weight_decay`, so the standard NC task optimizer uses the configured task values. An explicit model-level value still takes precedence under the repository's generic optimizer rule.

Versioned ordinary edge tensors remain cacheable, and in-place changes increment their version and invalidate the sparse operator. Inference tensors have no usable version counter, so the operator is rebuilt and not cached. Calls made while `torch.inference_mode()` is active also avoid caching, including calls with ordinary input tensors, because the generated sparse operator may itself be an inference tensor.

## Geometry

The v1 registered definition remains raw feature neighbor averaging. V1.1 additionally computes, independently within each aligned modality,

```text
cos(L2(x_v, eps=1e-8), mean_neighbors(L2(x_u, eps=1e-8)))
```

No cross-modality cosine is calculated. Degree and cosine quartiles are fit on training nodes only; cosine quartiles exclude isolates. Isolates receive their own bucket. Validation nodes use the training thresholds, and every bucket is flagged when `n < 100`. The report includes old/new thresholds and validation bucket transition counts without rewriting the old v1 geometry report.

## Outputs and privacy boundary

The executable writes aggregate train/validation metrics, aggregate confusion matrices, bucket counts, permutation hashes, path-weight summaries, provenance, and audit notes under `reports/problem_validation_v1_1/`. It does not persist logits, embeddings, features, topology caches, checkpoints, or per-node predictions.

If a future registered workflow needs per-node argmax for paired validation, those class IDs may be written only under ignored `outputs/`. Logits and embeddings are never persisted or submitted. Aggregate metrics and evidence hashes may be retained in tracked reports. This states the observed v1 behavior accurately: v1's ignored output directory already contains class-ID argmax files, despite the original design prose saying no per-node predictions were written.

The CLI requires `--evidence-root`, so running from a worktree cannot silently select an empty local v1 evidence directory. Its output-path guard prevents overlap with the frozen v1 report/raw directories.
