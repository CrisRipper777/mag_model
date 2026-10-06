# MAG collaboration adaptation NC v2 protocol

Protocol source: user execution prompt received 2026-10-06. This is a limited train/validation mechanism prototype and a correction replay. It does not edit or re-sign the v1/v1.1/v2/v3 evidence.

## Scope and immutable data

The formal data are Movies, Grocery, and Reddit-S NC only. Each uses the already frozen MAGB split seed 42, original loader edge list, original frozen feature files, and the locked 20-class Macro-F1 label set from `docs/collab_nc_pilot_v1/protocol_lock.json`. No new split is generated. The loader edge list is used without adding reverse edges, self-loops, de-duplication, sampling, or truncation. All node features and topology may participate in full-graph, transductive forward computation. Labels are copied and indexed only for `train_idx` supervision or `val_idx` selection/reporting. The encoder forward signature has no labels, class count, split indices, or LP target. No test task-head forward/metric/selection, real LP, Toys, or ele-fashion run is exposed by the CLI.

The source and all four input files per graph are SHA256 locked in `protocol_lock.json`. The lock also records the v1 training commit and source identities. The v1 protocol lock and original problem-validation lock are re-hashed; a changed new-source hash never changes the old experiment identity. Modality availability follows the previously frozen finite/nonzero-row fallback and is checked against the v1 profile. This heuristic is not a feature-quality claim.

## Common backbone and corrected initialization

Both modality stems, the center projection/readout, p/e/c/s definitions, synchronous two-layer schedule, residual factor 0.9, residual dropout 0.2, private/common algorithms, availability behavior, raw same-modality cosine, NULL paths, path heads, SSL pair target, and β/τ schedule inherit `docs/collab_nc_pilot_v1/protocol.md`. The v2 model is a separate class/module and does not alter v1 sources.

After deterministic name-derived Xavier initialization of every Linear and standard LayerNorm initialization, each M0/M1 layer explicitly restores both private gate final layers to all-zero weight/bias (`2*sigmoid=1`), private and common NULL final weights to zero and biases to `log(1/9)`, update/FFN final gain 0.1, relation/output matrices to identity, and private γ to one. The initialization hook only runs at construction. A checkpoint load never reapplies initial values.

The grouped softmaxes mask invalid logits to `-inf` before exponentiation. Private κ is over valid incoming neighbors per receiver and head. Private ρ combines those neighbor scores and one receiver-local NULL. Common π is over available sender modalities per edge, receiver modality, and head. Common α is over the complete available incoming-edge neighborhood per receiver modality/head and one NULL. Softmax is never split into chunks. Empty/all-invalid groups produce zero edge mass and finite NULL mass. V-off values use the original shared projection. The same/cross message relationship is distinct from the original attention relation matrices.

## V: typed same/cross public value content

When V is active in the M0 collaboration path, each layer replaces shared `W_V` with independent bias-free sender maps `W_V_text,W_V_image:64→64`. Values are split into two 32-dimensional heads. Two trainable per-head maps `M_same,M_cross:[2,32,32]` are initialized to identity. For physical edge `j→i`, sender modality `m`, receiver modality `n`, and head `h`:

```text
v_m_j[h] = reshape_heads(W_V_m(bar_c_m_j))[h]
v_m_to_n_j[h] = v_m_j[h] @ M_role(m,n)[h]
role(m,n) = same iff m == n, otherwise cross
```

The existing per-edge sender-axis π combines only observed sender modalities. The existing receiver/neighborhood-axis α then weighs full incoming physical edges; aggregation sums `α_i_j[n,h] * sum_m π_i_j[n←m,h] * v_m_to_n_j[h]` over edges. The existing receiver map `O_n[h]` and shared bias-free common FFN follow. No relation ID, node label, or invented semantic role enters. V-on sender maps both copy the exact original shared `W_V` name-derived initial matrix; `M_same/M_cross` are identity, so the V-on initial function matches V-off. V adds 16,384 active backbone parameters across two layers. V-off constructs no V-only parameters.

## F: own-semantic-preserving nonlinear context fusion

For p/e, retain `p_text[128]`, `p_image[128]`, `e[128]`, and the four unscaled final increments `Δs_text,Δs_image,Δc_text,Δc_image[64]`. Let their concatenation be `δ[256]`:

```text
a = W_e(e) + b
r = W_o(GELU(a + W_delta(δ)) - GELU(a))
z = concat(p_text,p_image,e, β*δ, β*Dropout(r))  # 768 dimensions
```

`W_e:128→128`, `W_delta:256→128`, and `W_o:128→128` are bias-free; the sole bias `b[128]` starts at zero. W_e/W_delta use Xavier gain 1, W_o gain 0.1, and name-derived seeds independent of shared modules. The inner difference has no normalization, dropout, or stochastic statistic; dropout 0.2 applies only after r. No post-fusion LayerNorm is used. For δ=0, r is exactly zero for arbitrary e and b. F adds 65,664 parameters. With F off, output remains 768-dimensional by appending 128 constant-zero padding coordinates; the S-stage main head is always `Linear(768,20)`, and those last head columns are counted as inactive. Padding is not capacity matching. R models remain 640-dimensional and have neither F parameters nor padding.

This is a context-conditioned residual fusion adaptation. It is not claimed to reproduce RoleMAG verbatim or to identify semantic roles.

## Optimizer and shared execution semantics

All runs use AdamW, β=(0.9,0.999), eps=1e-8, float32, no AMP, no scheduler, all active parameters at one LR, bias/LayerNorm parameters exempt from weight decay, global norm clip 1.0, and full-graph execution. Each step has one optimizer update. The corrected post-clip diagnostic norms read the actual clipped gradients; pre-clip values are measured before clipping only at fixed diagnostic steps. Dropout and pair sampling use checkpointed RNG state. Pair RNG seed is training seed + 31415. Backbone seed is the training seed with deterministic per-name Xavier generators; external task heads are created after setting seed + 10007, restoring the backbone RNG state afterwards.

The runner caches only feature/edge/availability-derived values for the current exact input objects: same-space normalized raw cosine by edge, observed-source masks, valid degrees, and edge availability. It retains input references, checks mutation versions and device, refreshes on object/content identity changes, and bypasses the cache for inputs requiring gradients or inference tensors lacking safe version counters. No parameter-derived tensors are cached. Raw cosine edge gathers are prepared in chunks of 16,384, while receiver softmax remains over the full neighborhood. Key/value projections are computed once per sender/node and relationship transforms once per node/role before edge gathering. `collect_diagnostics=false` is default; diagnostic forward, statistics, and serialization are outside optimizer-step timing. The model's unoptimized path recomputes reference raw cosines/projections and uses edge-space relationship order for equivalence and timing comparisons.

## Stage R: corrected replay, 54 formal runs

R-M0 and R-M1 are fresh training runs of the corrected initialization and grouped softmax implementation, with V/F off and original `out_dim=640`. Training objective remains M0 `main + .05 path + .01 rec + .01 pair`, and M1 `main + .05 path`; no duplicate source/auxiliary term is added. Both use the task adapter's fixed 20-class main/path heads.

Calibration is three datasets × two models × LR `{0.0003,0.001,0.003}` × WD `{0.0001,0.001}` × seed 62 = 36. Per dataset/model choose by validation correct count descending, unrounded fixed-20 Macro-F1 descending, earlier best step, then lower config enumeration. Confirmation uses only that selection for seeds 63/64/65: 18 runs. Checkpoint key is `(val_correct, Macro-F1, -step)`, eligible from step 20, max 300, min stopping step 60, patience 40; patience resets only on strict correct-count increase. The R lock is committed before confirmation.

R and historical v1 same-seed results are descriptive pairs. Since initialization, softmax correctness, and forward optimization are corrected together, their difference is not uniquely an initialization effect.

## Stage S: V/F factorial and strong controls, 162 formal runs

All six models use 768-D output and are initialized/trained from scratch. P/e/c/s stems, two-layer sync schedule, same-modality baseline B1, and path/SSL terms are shared according to the table. S0/S1 base B1 are the original 128-D concatenated c/s mean aggregator in each modality; they do not use the M0 private/common attention. S2-S5 use the corrected M0 backbone.

| ID | Base | V | F |
|---|---|---:|---:|
| S0 | B1 | off | off |
| S1 | B1 | off | on |
| S2 | corrected M0 | off | off |
| S3 | corrected M0 | on | off |
| S4 | corrected M0 | off | on |
| S5 | corrected M0 | on | on |

All use main + 0.05 path + 0.01 reconstruction + 0.01 pair; pair temperature 0.2; β/τ use the inherited 20-step schedule. Train every step from the selected seed initialization, max 400 steps, validation each step. A checkpoint is eligible from step 20. The fixed development utility is `U=.5*(Acc+Macro-F1)` using unrounded 0–1 values and fixed 20 classes. Best key is `(U,Acc,Macro-F1,-step)`. Patience starts at the first eligible step, 60; it resets only when U reaches at least the existing patience reference +0.0005, while a smaller U improvement may still change the best checkpoint without resetting patience. Stop no earlier than step 20. No LR extension is allowed.

Seed72 calibrates three datasets × six models × LR `{0.0003,0.001,0.003}` × WD `{0.001,0.05}` = 108. Choose within each dataset/model by `(U,Acc,Macro-F1,-best_step,-config_index)`. Commit immutable S selection lock, then confirmation seeds 73/74/75 = 54. Confirmation cannot reselect. Calibration interaction `S5-S3-S4+S2` is computed from same seed72 config indices as a reference and is not added to confirmation means.

The 6 train-only smoke runs (R-M0 and S5, each graph) each perform exactly three updates using a dedicated smoke seed; they are registered outside the 216 formal cells and never enter model choice. Before formal training, an additional six-record optimization profile compares uncached reference and optimized execution on the same per-graph model/head seed: both paths use two excluded warmup updates and three measured optimizer steps with identical loss, diagnostic flag off, and CUDA-event timing boundary. These short train-only updates are not matrix cells. Fixed diagnostic step IDs are 1, 10, 30, and 100, plus every best checkpoint. Non-sampled parameter update norm is null. Best checkpoint probability metrics are NLL, fixed-20 multiclass Brier (sum over classes, then mean over nodes), and wrong-prediction mean max confidence. S1/S4/S5 fusion screening and S5 private/common/all-graph block masks use the same frozen main head; they are not retrained ablations or causal effect estimates.

## Artifacts and interpretation

Machine-readable source, input, environment and selection provenance are recorded in the lock, registry, JSONL results, and manifest. Checkpoints, training/diagnostic logs and train/validation-only argmax files stay under ignored `outputs/collab_adapt_nc_v2/`. No logits, embeddings, all-node task predictions, feature dump, or checkpoint is committed. Confusion matrices are aggregate train/validation results and are reconstructible from the saved argmax plus locked labels/schema.

The S within-stage paired contrasts and interaction are descriptive; three seeds are not significance evidence. Selection can give models different LR/WD, so confirmation interaction is about selected training recipes rather than a unique causal module synergy. Validation has prior development exposure. Param-count matching does not imply compute matching. Neither V, F, gating, path metrics, nor hidden states prove recovered real-world semantic roles or causal usefulness. Poor performance alone does not amend this protocol.
