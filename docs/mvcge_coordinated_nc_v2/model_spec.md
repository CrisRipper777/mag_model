# Coordinated graph experts for multimodal MAG node classification

## Scope and relation to prior work

This is a protocol-defined implementation study for Movies, Grocery, and Reddit-S node classification. It borrows the broad ideas of graph-aware routed experts, self/neighborhood aggregation, and auxiliary semantic supervision, while fixing a different sparse row-mean operator and a specific low-order neighbor polynomial. It is not a reproduction of MvCGE, GraphSAGE, ACM, PCNet, or SUPRA. The MvCGE paper motivates shared graph experts with graph-aware routing; GraphSAGE motivates aggregating neighbor information alongside a node's own representation; ACM motivates allowing node-dependent aggregation behavior. This implementation uses its own exact equations below. [MvCGE paper](https://openreview.net/pdf?id=dsp8dUlZFq), [GraphSAGE paper](https://arxiv.org/abs/1706.02216), [ACM paper](https://arxiv.org/abs/2210.07606).

The P models implement only the explicitly specified first two low-order terms. The project uses the direct formulas `phi0=N`, `phi1=0.5(N+RN)` and does not port PCNet's higher-order recursion. The auxiliary-head contrast is motivated by SUPRA's reported dataset-dependent auxiliary-supervision changes; it is a reason to test supervision here, not a prediction for this experiment. [PC-Conv source](https://raw.githubusercontent.com/uestclbh/PC-Conv/main/PCpro.py), [SUPRA preprint](https://arxiv.org/html/2605.24684v1).

## Inputs and graph operator

The loader provides frozen float32 features in `[text, image]` order and an already undirected, deduplicated graph. No feature normalization, missing-modality inference, edge weights, or extra reverse-edge insertion is applied. All nodes have both modalities. A non-all-true availability mask is rejected.

For each directed message `j -> i`, the receiver-row operator is

`R[i,j] = 1 / degree(i)`.

Self loops and duplicate directed coordinates are removed. The sparse matrix has explicit shape `[num_nodes, num_nodes]`; a row for an isolate is all zero. Sparse multiplication uses the receiver row as the output node. No self loops are added.

## Shared encoder (N0/M0/M1/P0/P1)

Each modality projection is `p_m = LN2(Linear2(Dropout0.1(GELU(LN1(Linear1(x_m))))))`, with widths `F_m -> 128 -> 128`. A modality-specific 128-by-128 stem begins as identity. A shared context network maps concatenated text and image states `256 -> 128 -> 128`, with GELU between linears and final LayerNorm. It supplies both the readout context `c0=C(p_text,p_image)` and each graph-layer context `C_l=C(U_text,U_image)`.

There are two synchronous graph layers. For each modality, `U_m=LN_pre(H_m)` and `N_m=R U_m`. Each layer has four experts whose parameters are shared across modalities but independent across expert and layer:

`V_m,e = U_m W_self,e + N_m W_nbr,e + b_e`

`O_m,e = Linear_out(Dropout0.2(GELU(V_m,e)))`.

The self and neighbor transforms have separate bias-free 128-by-128 matrices; the hidden bias has width 128. Every expert output is evaluated; only the mixture weights are sparse. The mixed message is `D_m=sum_e g_m,e O_m,e`. A modality-specific residual adapter computes `A(D)=D+Linear_up(Dropout0.1(GELU(Linear_down(LN_A(D)))))` with widths `128 -> 32 -> 128`; the up matrix has Xavier gain 0.01. The layer update is `H_next=H+Dropout0.1(A(D))`. Both modalities use the previous layer states before either next state is assigned.

The final graph increment is `delta_m=H_m^2-H_m^0`, followed by a modality-specific LayerNorm `g_m=LN_delta(delta_m)`. The 640D output is exactly `concat(p_text,p_image,c0,g_text,g_image)`. No additional global normalization, fusion, or compression follows.

## N0 and routed models

N0 mixes all four experts uniformly. It has no router, Gaussian noise, balance loss, or unused auxiliary heads.

M0/M1/P0/P1 use one router per layer, shared across the two modalities. For each node-modality token, the router input is `concat(U_m,N_m,C_l,mean_nodes(U_m),d_i,log1p(s_m,i),onehot(m))`, width 516. Here `d_i=log1p(degree(i))/max(1,max_j log1p(degree(j)))` and `s_m,i=max(0, [R mean_channel(U_m^2)]_i-mean_channel(N_m,i^2))`, forced to zero for isolates. The full 516-vector is not normalized.

The router is `GELU(Linear(516,64))`, then separate `Linear(64,4)` clean logits and `Linear(64,4)` noise head. `sigma=0.01+softplus(noise_head)` starts at 0.10. Training adds independent standard Gaussian noise from a dedicated generator. Evaluation uses clean logits and exactly top-2 softmax gates.

During training, updates 1-50 use dense softmax gates; updates 51-99 linearly mix dense and noisy top-2 gates with `eta=(update-50)/50`; updates 100 onward use noisy top-2 gates. The balance term averages layers of actual-gate importance CV² plus `eta` times differentiable expected-load CV². CV² uses population variance over mean squared plus `1e-8`. The encoder returns this already weighted by 0.01.

## P0/P1 low-order neighbor filter

P replaces only the neighbor term. With `N=RU`, `phi0=N` and `phi1=0.5*(N+RN)`. Every expert has trainable, signed, per-channel `theta0,theta1` initialized to `(1,0)`:

`Q_m,e=(N_m W_nbr,e) * theta0,e + (phi1_m W_nbr,e) * theta1,e`.

This starts at the M neighbor response. Its effective polynomial coefficients are `alpha1=theta0+0.5*theta1` and `alpha2=0.5*theta1`, hence `alpha1 R + alpha2 R^2`. A two-hop return path can include the center node, so the neighbor term is not strictly center-free. This low-order basis change does not claim a larger function class than a freely reparameterized same-order polynomial.

## Task heads and losses

The main task head is Dropout(0.2) followed by `Linear(640,C)` and receives only train or validation rows. M1/P1 add separate Dropout(0.2) `Linear(128,C)` semantic heads applied to train rows of `p_text` and `p_image`. Their semantic loss is the mean of the two cross-entropies. N0/M0/P0 have no auxiliary heads.

- A0: historical `MAGCollabAdapt` B1 encoder, `typed_value=False`, `fusion=False`, 768D readout, existing main/path/reconstruction/pair training terms with weights `1/.05/.01/.01`.
- N0: main cross-entropy.
- M0/P0: main cross-entropy plus the encoder's 0.01-weighted balance term.
- M1/P1: M0/P0 loss plus `0.05 * mean(CE(text semantic), CE(image semantic))`.

All new models use label smoothing zero. No labels, logits, or task heads enter the encoder. A0 remains the full historical reference rather than a capacity-matched reduction.

## Initialization and optimization

Linear layers use name-derived Xavier uniform initialization and zero biases; LayerNorm affine parameters start at one and zero. Expert output matrices use gain 0.5, router logit matrices gain 0.01, stems are identity, adapter up matrices use gain 0.01, and PC coefficients start at `(1,0)`. Initialization names omit variant identifiers. The training RNG is reset after construction. The router Gaussian and A0 pair sampling use separate generators.

Optimization uses float32 AdamW with betas `(0.9,0.999)`, epsilon `1e-8`, gradient clipping at 5, and no AMP. Biases and LayerNorm parameters have zero weight decay; remaining matrices use the selected grid value. Router parameters use half the selected learning rate; context parameters use the base rate. P theta has base learning rate and fixed weight decay 0.001. Optimizer groups are checked for exact, nonduplicated parameter coverage.

## Public interface

`Model(cfg,data_info)` is available through `src.models.factory.build_model`. Its `forward(x,edge_index)` returns `(z,None,None,weighted_balance,extras)`. The specialized encoder API additionally accepts the update, an independent noise generator, and diagnostic flags. Graph caching keys on edge tensor identity/version, shape, node count, and device, and the sparse buffer follows `.to(device)`. This implementation is full-graph only; it makes no neighbor-sampling or layerwise inference equivalence claim.

The LP integration check uses only the existing synthetic endpoint-product `LinkPredictor(in_dim=640)` interface. It is not an LP result.
