# Prior-context MAG pilot v3: frozen protocol

## Scope, inputs, and task

Only Movies and Grocery NC train/validation are used. Inputs are the v1 lock's original graph, frozen text/image feature files, and fixed seed-42 split; no encoder training, feature normalization, split regeneration, LP, test prediction, test metric, or five-NC extension is allowed. Training seeds are 52/53/54 and never select another split. Macro-F1 uses the fixed 20-class label set from `reports/problem_validation_v1/protocol_lock.json`. Full-graph unlabeled features and the fixed transductive graph may participate in forward passes. Losses and supervised metrics index train only; validation is for selection/reporting.

For each input, the v1 lock stores byte length and SHA-256. The runner must verify those values and split counts before an attempt. Graph preprocessing is the shared loader's undirected, loop-free graph followed by model-side removal of self-loops, directed-pair de-duplication, and reciprocal completion. Thus each unique undirected non-loop pair contributes one message in each direction. No relation IDs, inferred modality availability, feature-norm quality labels, edge sampling, truncation, or new randomness is used.

## R4 representation

Let frozen node attributes be `x_t` and `x_v`, with `d=128`.

```text
p_t = A_t(x_t); p_v = A_v(x_v)                         # Linear -> LN -> GELU
e   = E([p_t,p_v])                                     # Linear(256,128) -> GELU -> Linear(128,128)
h_T^0=p_t; h_V^0=p_v; h_J^0=J([p_t,p_v])              # Linear -> LN -> GELU
```

For each layer `l=1,2` and flow `s in {J,T,V}`, with unique incoming neighbors `N(i)` and degree `q_i`:

```text
B_i,s = mean_{j in N(i)} h_j,s
S_i,s = mean_j phiS_s,l(h_j,s)
D_i,s = mean_j [phiD_s,l(h_j,s - h_i,s) - phiD_s,l(0)]
```

`phiS` is `Linear(128,64,bias=True) -> GELU -> Linear(64,128,bias=True)`. `phiD` has the same hidden width and activations, with its final bias disabled. It has no internal dropout. `phiD(0)` is evaluated by the same differentiable function for each edge chunk; paired batched evaluation makes equal edge endpoints exactly zero despite shape-dependent GEMM arithmetic. Empty-neighbor means are exactly zero. Edge chunking preserves every edge and the full mean; it does not sample or clip messages.

Context and router axes are fixed:

```text
o_i,s = [p_i,s, h_i,s, B_i,s, log1p(q_i)]            # 3*128+1 = 385
c_i   = [p_t,p_v,h_J,h_T,h_V,B_J,B_T,B_V]            # 8*128 = 1024
router input = [o_i,s,c_i]                           # 1409
pi_i,s = softmax(Linear(1409,16)->GELU->Linear(16,2)) # last affine zero-init
u_i,s = pi_i,s,S*S_i,s + pi_i,s,D*D_i,s
h_i,s^(l+1) = LN_s,l(h_i,s^l + Dropout(u_i,s))
```

The router's two entries select S/D only; center, text, image, and flow identity are not competing candidates. R3 broadcasts the per-layer mean of `c_i`; each `o_i,s` remains node-specific. J/T/V states and their layer norms are independent. R5 shares the S/D functions across flows within a layer while retaining separate states and routers. R2 uses fixed 1/2 S/D coefficients and no router, trained from scratch.

Final readout:

```text
chi_i = 1[q_i > 0]
g_i = [chi_i*(h_J^2-h_J^0), chi_i*(h_T^2-p_t), chi_i*(h_V^2-p_v)]
Phi(a) = Linear(640,256,bias=True) -> GELU -> Linear(256,128,bias=False)
r_i = Phi([g_i,p_t,p_v]) - Phi([0,p_t,p_v])
z_i = e_i + Dropout(r_i)                              # out_dim=128
```

Both Phi calls share parameters. Phi has no internal dropout; its last weight is Normal(0,0.001), not zero. The final sum has no LayerNorm or softmax. On isolated nodes or an empty graph, `g=0`, `r=0`, and `z=e` exactly in train and eval. This is a path/degeneration guarantee, not an information-preservation guarantee.

R0 keeps the two adapters and replaces E/readout with an active topology-free `Linear(256,w)->GELU->Dropout(.2)->Linear(w,128)` encoder. R1 keeps A/E/J/readout and all three states but uses an independent wide `Linear(256,w)->GELU->Linear(w,128)` on `[h,B]` for each flow/layer. R6 uses two `Linear(input,128)->ReLU` priors, DGL `GraphConv(norm="both")` 256→128→128 with explicit internal self-loops, ReLU/dropout only between layers, three external NC heads, and arithmetic mean of the three logits. Its GCN self-loops do not alter any other scheme. R6 returns concat `[p_t,p_v,g_joint]` (384-D) only for its NC reference; R0–R5 return the common 128-D embedding.

## NC heads, objective, and model boundary

The backbone consumes only `x` and graph structure and returns embeddings plus source-state extras. It owns no class heads or loss. R0–R5 task code creates one `Linear(128,C)` main head and two `Linear(128,C)` source heads. R6 task code creates text, image, and joint `Linear(128,C)` heads; main logits are their arithmetic mean, and its text/image heads also define source loss. Every scheme minimizes:

```text
CE(main_logits[train_idx], y_train)
  + 0.1 * (CE(text_logits[train_idx], y_train)
           + CE(image_logits[train_idx], y_train)) / 2
```

The task code indexes train embeddings and labels before applying heads/loss. It never produces test logits. No label is passed to a backbone. All task heads and backbone parameters share AdamW LR and are checkpointed together. Candidate `out_dim=128` is compatible with the framework NC head and LP `LinkPredictor`; the LP check is synthetic only.

## Frozen controls and capacity

| Scheme | Definition |
|---|---|
| R0 | Active topology-free two-modality MLP; preserve A and source heads, replace E/readout. |
| R1 | A/E/J and final center correction as R4; ordinary same-flow mean and independent wide `[h,B]` MLP for every layer/flow. |
| R2 | R4 S/D experts and three states; static 1/2+1/2 S/D mixing; retrained, no router. |
| R3 | R4, but every node receives the unlabeled graph mean context at each router layer. |
| R4 | Full per-node-context, per-flow experts/router candidate. |
| R5 | R4 routes/states with one shared S/D pair per layer across J/T/V; S width193, D width192. |
| R6 | Independent SUPRA-style NC reference described above; no parameter-padding requirement. |

Before any training or validation result is read, instantiate each graph's model and select the nearest integer width by absolute active-backbone parameter difference (tie: smaller width): R0 width applies to its single MLP; R1 width applies to six independent ordinary MLPs; R2 width applies to all twelve independent S/D functions. Each must fall within ±5% of R4. No unused modules count. R5 width choices are fixed by the protocol and their small count delta is disclosed. NC heads are counted separately. Similar parameter counts do not imply matched compute, function families, or optimization.

## Optimization, checkpoint, and metrics

All schemes use AdamW, weight decay `1e-4`, dropout `.2`, one LR for all trainable modules, 300 epochs maximum, early-stop patience 30 after minimum epoch 30, no scheduler, global norm clip 1.0, and validation each epoch. LR grid is `{0.0003,0.001,0.003}`. Seed 52 calibration selects maximum validation accuracy, then fixed-label Macro-F1, then lower LR. Exact epoch-score ties keep the earlier checkpoint. Seeds 53/54 use the selected LR. Seed52 selected checkpoint is reused and is not an independent confirmation seed.

Each checkpoint stores backbone, task heads, optimizer, resolved config, epoch, RNG states, split/input/protocol/source identity. It is evaluated on train and validation only. No output includes logits, features, or embeddings. The per-node argmax file contains only train/validation indices and predictions and stays in the ignored output root.

Every epoch records train main CE, source CE, train/validation Acc and fixed-label Macro-F1, validation CE, best-checkpoint state, synchronized optimizer-step time, and memory. Epochs 1/10/30 and the best epoch record module-level pre/post-clip gradient norm, parameter norm, update norm, relative update, total pre-clip norm, and clip scale. The record also includes route means/variances/entropy, S/D output norms/cosine, p/e/r/z norms, center-correction cosine/projection, and degree-bucket summaries. These are optimization/health summaries, not roles, utility, information shares, or causal evidence.

The exact global pre-clip norm is the value returned by PyTorch's clipping operation. At registered diagnostic epochs, module post-clip norms are measured from the retained gradients and module pre-clip norms are reconstructed using the operation's single global clip coefficient; this avoids extra per-module gradient reductions on ordinary epochs. Parameter/update norms use snapshots immediately before and after that optimizer step.

Diagnostic summaries use a fixed, label-free sample of up to 512 node IDs drawn from the union of the frozen train and validation indices using seed `20261005`. The exact sorted IDs and seed are recorded in the pre-training matrix and copied into attempts/checkpoints. This sampling changes only the aggregate diagnostic summaries; model training, validation metrics, and predictions use the full required train/validation sets. No test node is sampled for diagnostics.

## Staged matrix and gate

Stage A: R0/R1/R4 × Movies/Grocery. Seed 52 × three LRs is 18 calibration jobs; selected LR × seeds53/54 is 12 confirmation jobs. Maximum A=30. Stage B is four more schemes × two graphs × three LRs × calibration seed52 plus two confirmation seeds = 40; maximum A+B=70.

The gate uses only the four matched confirmation cells (two graphs × two seeds) for each R4-vs-R0 and R4-vs-R1 comparison. For each baseline, R4 must pass one path:

1. Accuracy: equal-graph mean gain ≥0.005, Macro-F1 mean gain ≥0, and at least 3/4 paired accuracy deltas >0; or
2. Macro-F1: equal-graph mean gain ≥0.01, accuracy mean gain ≥0, and at least 3/4 paired F1 deltas >0.

For each baseline and each graph, the two-seed mean delta must be ≥−0.002 Acc and ≥−0.01 F1. Against R1, mean confirmation optimizer-step median-time ratio and mean peak allocated-memory ratio are each ≤3. Per-run step time is CUDA-synchronized median over epochs2–30 excluding diagnostic epochs, validation, saves, and diagnostics; per-graph confirmation-seed values are averaged before two graph ratios are equally averaged. Reserved memory and wall time are also reported. If either baseline gate or either resource bound fails, B is `skipped_gate`; the rule, LR grid, losses, and seeds stay frozen. These budget thresholds are not significance tests.

## Scope and interpretation

No test labels are indexed, test predictions/metrics are not generated, and test cannot select anything. No LP, other NC graph, post-hoc LR/width/loss/module search, extra seeds, or next model is run. Report all failures/voids/retries and preserve their attempts. Three seeds and two previously studied development graphs support descriptive comparisons only. A positive R4 result does not establish that every mechanism component is necessary; the controls isolate specific design changes, not universal causal mechanisms.
