# MAG collaboration NC pilot v1 protocol

Protocol date: 2026-10-06. This is a new train/validation-only study on the existing unified MAG loader. It does not modify or re-sign the v1, v1.1, v2, or v3 protocols or evidence.

## Scope and fixed evidence

Formal graphs are Movies, Grocery, and Reddit-S NC. Their seed42 train/validation splits, 20-class label schema, and fixed Macro-F1 label set come from `reports/problem_validation_v1/protocol_lock.json`; the actual input files are re-hashed before preparation and every run phase. Model seeds are distinct from split seed42. The full graph's frozen features and allowed transductive topology are used for representation computation; supervised losses and metrics index only `train_idx` or `val_idx`. No test head forward, test metric, test-based choice, real LP, Toys, or ele-fashion is in this protocol.

The loader returns the original undirected MAGB `edge_index` with no self-loops. B0/B1/M0/M1 use this exact list, retaining its order and any duplicate entries if present; this avoids adding reverse edges, de-duplication, loops, or sampling in the new operator. B2 uses the legacy R1 canonical directed edge function (unique, loop-free, bidirectional). Its count, hash, and symmetric edge-set difference from the loader list are recorded per graph; B2 differences therefore include this preprocessing distinction.

MAGB exposes no explicit per-row availability metadata in the current loader. For this run a modality row is marked unavailable if any entry is non-finite or if the whole row is zero. This is a fallback heuristic, not a claim that zero means missing. Counts are locked and reported. The model receives a finite copy with unavailable rows zeroed, and preserves the original availability mask when choosing senders, selecting SSL pairs, and scoring source heads. No norm threshold is treated as data quality.

## Models

- **B0 Ego:** shared p/e definitions, no c/s stems or graph operator. The output has `[p_text,p_image,e]` in its first 384 coordinates and a fixed zero pad to 640. One external main head; no path or SSL head/module.
- **B1 same-modality mean:** p/e and c/s stems as the full candidate; two independent per-modality layers update `concat(c_m,s_m)` from its same-modality valid-neighbor mean. It has external path heads and both weak SSL terms. This is not a factor disentanglement claim.
- **B2 legacy R1:** existing `mag_prior_context.Model` R1, width 144/out_dim 128, with the existing task-side main and two source heads. It is freshly trained under this pilot's search and stopping schedule, not a reproduction of an earlier training protocol. Objective is main CE plus 0.1 times the mean of the two legacy source CEs.
- **M0 full:** 640-wide readout, two synchronous state layers, independent private paths, shared common path, and weak graph-pretraining SSL losses.
- **M1 task-only:** same full backbone and path heads, no decoder/projector modules and no SSL objective.

All learned heads are created in the task adapter. The encoder does not receive labels or class count. The task code selects train indices before reading labels or evaluating any head.

## Full M0/M1 tensor definitions

Feature order is `[text,image]`, using actual widths. `p_t,p_v` are 128-D modality-specific `LN(GELU(Linear(x_m)))` projections. `e` is `LN(MLP([p_t,p_v]))` with `256→128→128`. For each modality, `c_m0 = LN_C0(F_C(A_m(p_m)))`, where `A_m:128→64`, `F_C:64→64→64` is shared, and `LN_C0` is shared. `s_m0=LN_Pm0(F_Pm(p_m))`, with a modality-specific `128→64→64` MLP and LayerNorm. Each mapping is followed by its original-availability mask; `e` is zero when neither input modality is available.

There are two synchronous layers. Each reads `bar_c_m=LN_C(c_m)` with one shared common LayerNorm and `bar_s_m=LN_Pm(s_m)` with independent private LayerNorms. No state reads another state after it has been updated in the same layer. A writeback is exactly

```text
h_next = h0 + 0.9 * (h - h0) + Dropout(message_update)
```

with dropout 0.2 once at each writeback. There is no post-writeback normalization. If a receiver has no valid neighbors, its message update is exactly zero. Starting at an empty graph therefore leaves the graph increments zero.

For a physical edge `j→i`, each private path is modality-specific. Two bias-free 64→64 Q/K/V projections form two 32-D heads. Each edge/head score is QK scaled dot product, plus a learned raw same-modality feature cosine coefficient (initialized to one), plus an MLP of `log1p` valid in-degree for receiver and sender. The cosine is computed only within one original feature space and is zero unless both endpoint rows were originally observed. For each receiver/head, `κ` is softmax over valid real neighbors only and sums to one; it is used to form the private value mean `μ`. Separately, the mean head score divided by the graph temperature is normalized together with a scalar NULL logit `log(d_i)+ell_i`, where `ell_i` starts at `log(1/9)`. This gives NULL=0.1 and real mass `ρ=0.9` for equal zero logits, independent of nonzero neighbor count; an empty neighborhood has `ρ=0` without taking `log(0)`. `ρ` controls the complete private path. The local transform acts on `μ`; the difference transform acts on `v_i−μ` through two bias-free layers, so it is exactly zero at zero difference. Two independent sigmoid gates start at one (each is `2*sigmoid(0)`); the gated local and difference values pass through the private update MLP.

The common path has two separate edge decisions. For each edge, receiver modality `n`, head, and observed sender modality `m`, `π` scores Q from `[bar_c_i^n,e_i]` against K from `bar_c_j^m` through an identity-initialized `R_same` or `R_cross` matrix, plus a sender/receiver/head bias. `π` softmax is only over observed sender modalities on this edge and sums to one when any sender is observed. It mixes a shared projected sender value. A separate neighborhood score `b` uses an independent query/key pair on `r_i=[e_i,bar_c_i^t,bar_c_i^v]`, plus a small MLP over the two within-space raw cosines, two observed-pair indicators, receiver/sender valid degree, and receiver-modality one-hot. It does not reuse `u` or `π`. For each receiver-modality/head, `α` softmax jointly covers every valid physical neighbor plus a NULL logit `log(d_i*)+ell_i[n,h]`, with the NULL term not divided by temperature. Empty neighborhoods have NULL probability one. The `α`-weighted sender-modality mixture is aggregated over neighbors, passed through a receiver-specific identity-initialized head output map, then a shared bias-free `64→128→64` common MLP. All four states are updated synchronously.

The final vector is exactly

```text
z = [p_t, p_v, e, beta*(s_t-s_t0), beta*(s_v-s_v0),
     beta*(c_t-c_t0), beta*(c_v-c_v0)]   # 640-D
```

There is no whole-vector LayerNorm, neighbor softmax, or narrow final projection. The first 384 dimensions are learned own-feature projections, not a lossless copy of raw inputs. Graph increments are not asserted to be pure structural factors.

Task-side heads are `Linear(640,C)` for the main path, `Linear(64,C)` per final private state, and `Linear(128,C)` for concatenated final common states. Path loss is the average of text-private CE over observed-text train nodes, image-private CE over observed-image train nodes, and common CE over train nodes with at least one available modality. An empty eligible subset contributes zero. B0 creates no unused path heads.

## Objectives and optimization

M0/B1 minimize `L_main + .05 L_path + .01 L_rec + .01 L_pair`; M1 minimizes `L_main + .05 L_path`; B0 minimizes `L_main`; B2 minimizes legacy main plus `.1*mean(source-text CE, source-image CE)`. Each source CE indexes only the eligible train subset. There is no class loss in the encoder.

`L_rec` is pre-graph: one `Linear(128,input_dim_m)` decoder from `[c_m0,s_m0]` per modality predicts its frozen finite feature row; `1−cosine` is averaged over available graph nodes per modality, then averaged across modalities. It does not use `p_m` as a detached target. `L_pair` applies a shared bias-free `64→32` projector to pre-graph common states, L2-normalizes, and computes symmetric same-node InfoNCE at fixed temperature 0.2. It samples without replacement up to 1024 nodes visible in both modalities using a separate saved RNG. Labels do not affect either SSL term. Evaluation computes neither SSL term and consumes no pair RNG.

All active backbone and task parameters train jointly from step 1. AdamW applies the selected WD except to all biases and LayerNorm parameters. All active modules share LR. Global gradient norm clipping is 1.0, float32, no AMP or scheduler. The model schedule uses `beta(s)=.1+.9*min((s−1)/19,1)` and `tau(s)=2−min((s−1)/19,1)`. Maximum 300 full-graph optimizer steps, with validation after every step. Eligible best steps begin at 20 for B1/M0/M1 and at 1 for B0/B2. Best is validation correct count, then unrounded Macro-F1 on the fixed 20 labels, then earlier step. Patience initializes at 40 at the first eligible step, resets only on a strict increase in best validation correct count, otherwise decrements, and can stop only from step 60 onward. A Macro-F1 tie-break can update the checkpoint without resetting patience.

## Search and selection

Seed62 calibrates each graph/model on LR-major, WD-minor grid `{0.0003,0.001,0.003} × {0.0001,0.001}` (90 standard jobs). Per graph/model choose by validation correct count, then Macro-F1, then earlier best step, then earlier grid order. Freeze `selection_lock.json` before confirmation. Seeds63/64/65 run once at the selected LR/WD (45 jobs); seed62 is not included in confirmation means, which use population SD (`ddof=0`). Confirmation cannot alter selection.

One optional search extension is allowed only before selection lock and only when boundary LR evidence demonstrates the range does not cover a suitable optimization region. The candidate is 0.0001 or 0.01 and applies to all five models and both WDs for that graph (10 extra calibration jobs); at most one side per graph and 30 extra total. A boundary winner alone, a low score, or a desire to improve the model is insufficient. Any extension and its evidence must be recorded in `search_amendment.json` before its cells run. No extension is used unless such evidence is recorded.

## Diagnostics, checkpoints, and artifacts

Before training, each graph locks independent label-free samples of up to 512 train IDs and 512 validation IDs from seed20261006. Step logs record losses, train/validation fixed-schema Acc/F1/CE, best/patience state, schedule, and optimizer-step time. Steps 1/10/30 and each best step record module gradient/parameter/update norms plus sampled representation norms and routing/NULL/gate summaries. CUDA-event training step timing is distinct from wall time including validation/checkpointing; peak allocated/reserved memory is recorded. Complete checkpoints include model, task heads, optimizer, step/selection/patience, Python/NumPy/Torch CPU/CUDA and pair RNG states. Raw best/latest checkpoints, JSONL logs, and only train/validation argmax remain under ignored `outputs/collab_nc_pilot_v1/`; no all-node prediction, logits, feature, or embedding files are committed.

Preparation hashes source files, original protocol lock, and each frozen graph/feature/split file. Run phases recheck those identities. Results and audits are new v1 files; no historical registry, lock, report, checkpoint, prediction, or log is rewritten.

## Execution phases

Run in `yhf_env`. The idle physical RTX 3090 is selected as physical GPU 1 by `CUDA_VISIBLE_DEVICES=1`; inside the process it is `cuda:0`. The phase CLI intentionally has no test-evaluation or LP command.

```bash
conda activate yhf_env
cd /hdd1/DataInHere/YHF/mag_model
CUDA_VISIBLE_DEVICES=1 python scripts/run_collab_nc_pilot_v1.py --help
CUDA_VISIBLE_DEVICES=1 python scripts/run_collab_nc_pilot_v1.py prepare
CUDA_VISIBLE_DEVICES=1 python scripts/run_collab_nc_pilot_v1.py smoke
CUDA_VISIBLE_DEVICES=1 python scripts/run_collab_nc_pilot_v1.py calibrate
CUDA_VISIBLE_DEVICES=1 python scripts/run_collab_nc_pilot_v1.py lock-selection
# Commit the selection lock and calibration evidence; pass its SHA explicitly:
CUDA_VISIBLE_DEVICES=1 python scripts/run_collab_nc_pilot_v1.py confirm --selection-sha256 <sha256>
CUDA_VISIBLE_DEVICES=1 python scripts/run_collab_nc_pilot_v1.py report
CUDA_VISIBLE_DEVICES=1 python scripts/audit_collab_nc_pilot_v1.py audit
CUDA_VISIBLE_DEVICES=1 python scripts/audit_collab_nc_pilot_v1.py manifest
```

`prepare` requires committed training source and writes the immutable input/source lock. `smoke` performs exactly three train-only M0 updates per graph and is excluded from model choice. `calibrate` runs each missing standard cell and resumes saved `latest.pt`; it does not silently retry failed cells. An explicit `--retry-failed` creates a distinct registered attempt. If calibration evidence warrants the one allowed range extension, first commit `reports/collab_nc_pilot_v1/search_amendment.json`, then rerun `calibrate` before `lock-selection`. `confirm` requires the committed selection lock and exact SHA. `report`, `audit`, and `manifest` consume train/validation artifacts only.
