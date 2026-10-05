# MAG mechanism pilot v2 protocol

Status: frozen implementation protocol for a train/validation-only prototype. It does not amend the v1 or v1.1 registries, locks, checkpoints, predictions, or reports.

## Scope and frozen inputs

This round trains Movies and Grocery only. It uses the existing MAGB graph, frozen RoBERTa and CLIP feature files, and the seed-42 NC split for every model seed. The configured feature order is `[text, image]`. The authoritative input byte counts, SHA-256 values, class order, Macro-F1 label set, and split sizes come from `reports/problem_validation_v1/protocol_lock.json`; the training entry point checks every input against that lock before loading it. Feature encoders and feature files are never trained or rewritten.

The fixed official split files are:

| Dataset | Split | SHA-256 | Train / validation / test node counts |
|---|---|---|---:|
| Movies | `/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt` | `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211` | 10003 / 3334 / 3335 |
| Grocery | `/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt` | `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e` | 10244 / 3415 / 3415 |

Both class schemas are 20 classes, ordered `0..19`; Macro-F1 averages those same 20 fixed labels in train and validation. These definitions are read from the v1 lock. Test indices and labels are never used to calculate metrics, choose checkpoints, or diagnose failures. Full transductive node features and graph topology can participate in label-free forward passes under the existing MAGB protocol. Every reported metric indexes only train or validation nodes.

The shared loader removes self-loops, makes each dataset graph undirected, and coalesces duplicate edges. The model validates node IDs, removes self-loops again defensively, and sorts/deduplicates directed `(src,dst)` pairs. Thus reciprocal edges remain two directed messages after undirecting; parallel duplicates count once; self-messages are excluded. This exact graph is used by every scheme. Neither relation IDs nor edge IDs are introduced.

No modality-availability metadata is supplied for these two datasets, so each stored text and image feature is treated as available. Availability is never inferred from feature magnitude or nonzero values. The model accepts an explicit `[N,2]` availability mask for callers that have metadata. It masks unavailable senders before sender softmax, masks unavailable receivers before link aggregation, and returns zero messages for empty neighborhoods or no available sender.

## Backbone

The width is fixed at `d=128`. Text and image features have separate `Linear(input_dim,128) → LayerNorm → GELU` projections. Their resulting states remain separate. A learned eight-dimensional type embedding identifies the text and image types; it carries no label or class information. No cosine is computed between the original RoBERTa and CLIP spaces.

There are two state updates. Layer 1 propagates same-modality messages only, preserving source-defined states for the optional source task. Layer 2 either propagates same-modality means or performs the dual-axis interaction below. The second-layer states are collaborative states and are not described as pure text/image embeddings. Each update uses a center residual and `LayerNorm(h + Dropout(expert_output))`.

### Dual-axis relation interaction

For every directed message edge `u→v`, let `h_t,h_i` be the current layer inputs and `d_v,d_u` their incoming degrees in the processed graph. Link context is

```text
c_uv = MLP([h_v,t; h_v,i; h_u,t; h_u,i;
            |h_v,t-h_u,t|; |h_v,i-h_u,i|;
            log(1+d_v); log(1+d_u)])
```

The context MLP is `Linear(770,128) → GELU → Linear(128,128)`. It uses current attributes and topology only, with no target labels, relation IDs, or edge identity.

For receiver modality `m` and sender modality `b`, the shared modality scorer receives `[c_uv; h_v,m; h_u,b; type_m; type_b]`. Its logits are normalized over the **two sender modalities `b` on this one edge for this one receiver modality**:

```text
s_uv,m,b = softmax_b(score_mod([c_uv; h_v,m; h_u,b; type_m; type_b]))
p_uv,m   = sum_b s_uv,m,b U_m,b h_u,b
```

`score_mod` is `Linear(400,32) → GELU → Linear(32,1)`. There are four independent bias-free `128×128` value projections `U_m,b`. Unavailable senders are excluded from the two-way softmax; when both are unavailable, both coefficients and the message are zero.

Link attention uses one learned query projection per receiver modality and one shared key projection:

```text
l_uv,m = (Q_m h_v,m)·(K c_uv) / sqrt(128)
a_uv,m = softmax over incoming edges u with the same receiver v
B_v,m  = sum_(u→v) a_uv,m p_uv,m
```

The second softmax is grouped by destination node `v` and receiver modality `m`. It is not a graph-wide softmax. If no incoming edge has an available sender for `m`, or the receiver modality is unavailable, the aggregate is zero. Empty graph neighborhoods are handled without NaNs. When interaction is disabled, both updates instead take a uniform mean over unique incoming same-modality neighbors, with zero for no available neighbors.

### Shared expert layers

At each layer the same expert functions are shared across text and image states, while each modality has its own state and, when enabled, its own router. Every expert is a real MLP:

```text
Linear(256,256) → GELU → Dropout(0.2) → Linear(256,128)
```

Conditional schemes have one always-added shared expert `F0`, three independently parameterized specialists `F1..F3`, and a modality-specific dense softmax router. The router input is `[LayerNorm(h_v,m); LayerNorm(B_v,m); g_m; type_m]`, where `g_m` is the current graph-wide mean state for modality `m`. Router output layers start at zero, so specialist weights start uniform. The update is `F0 + sum_k pi_v,m,k Fk` before dropout, residual, and LayerNorm. There is no noisy top-k or load-balancing loss.

Wide schemes use one MLP of hidden width 1025 for that layer. Its trainable MLP parameter count differs from the four active conditional experts by less than 0.01% per layer; inactive expert/router modules are not instantiated in wide schemes. Static scheme P4 has `F0 + (F1+F2+F3)/3` and no routers; this static architecture is trained from scratch.

The common FusionMLP receives `[h_t^2;h_i^2;h_t^0;h_i^0]` and is `Linear(512,256) → GELU → Dropout(0.2) → Linear(256,128)`. The NC classifier is task-owned and is the same `Linear(128,num_classes)` head for every scheme. The backbone has no class prediction layer.

For P5/P6 only, the NC task creates two external linear source heads over the first-layer text and image states. The source loss is the mean of their cross-entropies, computed only on `train_idx` and intersected with the relevant explicit availability mask if one is provided. The total loss is `main_CE + 0.1 × source_CE`; the main/source heads, backbone, and optimizer state are saved together. P0–P4 do not instantiate these auxiliary heads.

## Seven fixed schemes

| Scheme | Expert implementation | Layer-2 interaction | Source loss |
|---|---|---|---:|
| P0 | Wide MLP | off | 0 |
| P1 | Shared plus conditional specialists | off | 0 |
| P2 | Wide MLP | on | 0 |
| P3 | Shared plus conditional specialists | on | 0 |
| P4 | Shared plus static uniform specialists, retrained | on | 0 |
| P5 | Shared plus conditional specialists | on | 0.1 |
| P6 | Wide MLP | on | 0.1 |

The same FusionMLP and main NC head apply throughout. Comparisons P2−P0 and P3−P1 describe the overall dual-axis change; P3−P2 compares conditional experts with the width-matched nonlinear control; P3−P4 compares learned routing with a separately trained static mixture; P5−P3 and P6−P2 compare source supervision within conditional and wide models. The dual axes are not separately ablated in this round.

## Training and selection

The formal matrix has 42 seed-42 calibration cells (`2 datasets × 7 schemes × 3 learning rates`) and 28 confirmation cells (`2 datasets × 7 schemes × seeds 43,44`). Seed 42 is the training/validation calibration run and is reused as that seed's selected result; it is not trained a second time. Each training seed uses the exact same seed-42 split. There are at most 70 formal training cells.

For each dataset/scheme, seed 42 trains at backbone learning rates `{0.0003, 0.001, 0.003}`. The selected rate maximizes validation accuracy; exact ties use higher fixed-label Macro-F1, then lower LR. Seeds 43/44 then train once at that selected rate. All cells use AdamW, weight decay `1e-4`, dropout `0.2`, maximum 300 epochs, patience 30, minimum epoch 30, no scheduler, and global gradient clipping at 1.0. One optimizer instance persists across epochs; the same LR applies to projections, experts, routers, and task heads. Best checkpoint selection is validation accuracy, then Macro-F1; exact ties retain the earlier epoch. No test score is calculated.

At each epoch the raw ignored logs contain train main CE and train source CE, train/validation Accuracy and fixed-label Macro-F1, and validation CE. For epoch 1, 10, 30, and each selected best epoch they also contain aggregate module parameter/gradient/update norms, expert output norms and pair distances, per-modality route summaries, and dual-axis attention summaries. Source-head accuracy is train-only. These values are diagnostics, not semantic role identification or evidence of causal utility.

Training uses `yhf_env` and an available GPU selected from the pre-run `nvidia-smi` inventory. No unrelated process is stopped. Each checkpoint stores model, main head, source heads if used, optimizer state, resolved config, best epoch, source code identity, and input hashes. Raw checkpoints, per-epoch logs, and train/validation argmax remain under ignored `outputs/mechanism_pilot_v2/`. Git receives only code, protocol, tests, and small aggregate results.

## Interpretation limits

Movies and Grocery are known development graphs, and this is a finite prototype, not independent external validation. Three seeds are descriptive; there are no significance claims. Router differences are not prediction gains, causal effects, semantic roles, or modality information fractions. A validation change cannot establish whether a benefit arose from relation interaction, shared experts, source supervision, or their optimization interactions without the corresponding controlled comparisons. No LP, test evaluation, or five-NC expansion is part of this round.
