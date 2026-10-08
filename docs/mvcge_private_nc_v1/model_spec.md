# MVCGE private NC v1 model specification

## Scope and names

This implementation uses one shared encoder design for each independently trained graph. It produces `z ∈ R[N,256]`. A task-side linear head supports NC; the existing endpoint-product MLP can consume the same node representation for LP. Only a synthetic LP shape/gradient test is run in this study.

`src/models/mvcge_private_nc_v1.py` contains the new G0/G1/U1 encoder and sparse graph/PC helpers. `scripts/run_mvcge_private_nc_v1.py` is the isolated train/validation-only entry point. A0 reuses the historical implementation without copying or approximating it.

## A0: historical S0

A0 constructs `MAGCollabAdapt(variant="B1", out_dim=768, typed_value=False, fusion=False)` from `src/models/mag_collab_adapt.py` and uses the existing `CollabTaskHeads("B1", output_dim=768)`. The readout is the original p-text, p-image, center, and four β-scaled graph increments followed by 128 zero-padding coordinates. Its loss is the S0 loss from the previous protocol: main CE + 0.05 path CE + 0.01 reconstruction + 0.01 pair loss. `set_epoch(step)` preserves the B1 β/τ schedule. The current task keeps S0's model and no-decay parameter partition, while applying this round's common 20-update warmup and gradient clip 5.

The previous S0 selection lock gives Movies LR/WD 0.003/0.001, Grocery 0.001/0.001, Reddit-S 0.001/0.001. All fall inside this round's 9-point grid, so there are no extra A0 cells. The full model source and original S0 design are referenced in `intake.md` and the previous experiment docs.

## G0, G1, and U1 dimensions

The implementation follows the request's fixed dimensions: two modalities, latent width 128, two graph layers, four shared experts per layer, top-2 routing, polynomial order Q=3, PC bank size S=3, t=0.5, private bottleneck 32, router hidden width 64, modality embedding width 8, and final width 256. All three datasets have 768-dimensional text and image inputs. Input features are cast to float32; the frozen features are not normalized, standardized, projected, or re-extracted by this implementation.

Each modality has its own `Linear(Dm,128) → LayerNorm → GELU → Dropout(0.1) → Linear(128,128)` adapter and its own pre-LayerNorm in each graph layer. The masked adapter output `a_m` is the initial graph state and also enters the final readout through a dedicated LayerNorm. It is a learned projection and is not described as an unprocessed input embedding.

The graph is supplied by the existing loader. It has no exposed edge weights; the graph operators are formed once from that canonical edge list. Existing loops are removed, and a unit loop is added once before forming `P=D^-1/2(A+I)D^-1/2`. Sparse `P` multiplication implements `T=diag(mask)-P`; no dense N×N matrix is constructed. For each modality and layer, Q=3 live basis SpMMs build B1–B3 from the current masked pre-normalized state. B0–B3 are reused within that forward for all four experts. They are never detached or cached across optimizer updates. For the two fully observed modalities and two layers this is 12 named basis SpMMs per forward; router local context reuses `U+B1`.

Each expert has a bias-free 128×128 input matrix, trainable `[4,128]` θ, one 128-vector bias, GELU/dropout(0.2), and a 128×128 output Linear with bias. The implementation solves the float64 PC-bank transform for the specified four T-polynomials, then stores float32 θ. Tests verify full rank, response equality, and direct-filter versus shared-basis output and gradient equality. θ remains unconstrained and trainable.

G0 and G1 share the same context router, four-expert blocks, adapters, pre-LNs, and readout. The context MLP uses `[U_text,U_image,M]`; each modality router receives U, `P U`, entity context, that modality's observed-node graph mean, normalized log-degree, M, and its learned 8-D type embedding (523 values). No label or split information enters the encoder. Router noise is `0.01+softplus(noise_head)` with the requested initial σ=0.1; the logit head uses Xavier gain 0.01. Training updates 1–20 use four-way noisy softmax and importance balance only. From update 21, training uses noisy top-2 gates and a differentiable Gaussian expected-load balance term. Evaluation uses clean top-2. All four graph experts are computed for each modality before receiver-node gate weighting; top-2 does not reduce full-graph expert computation to two experts.

G0 has no private correction FFN. G1 and U1 have one independent 128→32→128 FFN per modality and layer, with GELU and dropout(0.2); the second Linear has Xavier gain 0.5. These four FFNs add exactly 33,408 trainable backbone parameters. Their residual update uses the same residual dropout 0.1. No private scalar gate or orthogonality loss is present.

U1 retains G1's adapters, experts, private FFNs and readout but has no router, context MLP or modality-type table. It always weights each expert by 1/4 and has no balance loss. Its active capacity, number of computed experts and objective differ from G1; it is not a capacity-matched causal routing control.

The final readout concatenates four masked 128-D streams (two projected own-semantic streams and two final graph states) plus the 2-D modality mask, maps 514→256→256 with GELU/dropout(0.3), then LayerNorm. NC's `Linear(256,C)` head is outside the encoder. LP's existing endpoint-product MLP is also outside it.

## Initialization and parameter groups

The new model uses name-derived Xavier initialization so G0/G1 public parameters begin identically even though G1 adds FFNs. Linear biases and LayerNorm biases are zero; LayerNorm weights are one; type embeddings use Normal(0,0.02); expert θ is set by the requested spectral responses; expert output and private second matrices use gain 0.5. Task heads use a separate seed `training_seed+10007`.

The new optimizer uses AdamW with betas `(0.9,0.999)`, eps `1e-8`, float32, no AMP, and global norm clipping at 5. Matrix weights use the selected matrix WD; router/context matrix weights use half the base LR and the same matrix WD; θ uses base LR and fixed WD 0.001; all biases, LayerNorm parameters, and type embeddings use WD 0. Router/context/type embedding LR is half the base rate. The runner checks the parameter object sets for duplicates and omissions. A0 keeps the legacy decay/no-decay split and has no θ-specific group.

In the tested implementation, active backbone / task-head counts are G0 `868,940 / 5,140`, G1 `902,348 / 5,140`, and U1 `732,288 / 5,140`. These counts exclude fixed sparse graph operators. A0's active counts are emitted per run from the reused model.
