# Interaction Provenance M1

## Question

Can a compact state recursively record how inferred physical relations and their realized semantic effects compose during progressive graph interaction?

M1 is based on the frozen M0.1 source commit `749d1b4f20c13886a308cddb4770e1e25a865ffd` on `exp/interaction_provenance_m1`. The M0.2 implementation is not a dependency.

## Parallel streams

The semantic stream follows M0.1 without provenance feedback:

```text
latent relation R -> conductance and orthogonal transport
-> discrepancy -> response -> semantic edge message M -> PNA -> S_k
```

For modality `m`, order `k`, and directed edge `j -> i`, M1 recomputes the exact M0.1 message

```text
M[j,i,k,m] = c[j,i,m] * response_m(T[j,i,m] S[j,k-1,m] - S[i,k-1,m])
```

from the same relation/effect tensors and feeds it only to the provenance stream. The interaction atom is

```text
r = LN_r(W_r R)
e = LN_e(W_e M)
x = W_x(r * e)
A = tanh(LN_A(r + e + x))
```

Text and visual modules have separate parameters. Starting state `G0` is a fixed all-ones tensor, with no learned node-specific initialization. Provenance messages use DistMult composition `G[j,k-1] * A[j,i,k]`, the M0.1 PNA implementation at width 32, and a signed linear-plus-LayerNorm combine without a final activation.

During training, M1 recomputes semantic and provenance edge-message chunks during backward to control GPU memory; the forward equations and shared M0.1 parameters are unchanged. The temporary zero-initialized `Linear(64, 128, bias=False)` adapter adds `[G_text,3 || G_visual,3]` to the unchanged M0.1 readout. It exists only to expose provenance to NC task gradients during M1 development. It is **not** the final Stage III evidence-composition design. M1 adds no auxiliary loss.

## Frozen counterfactuals

The audit runs Adapter-Off, History-Off, degree-bucketed History Shuffle (seeds 3407–3409), Relation-Off, and Realized-Effect-Off against each selected checkpoint. They are frozen functional diagnostics, not retrained ablations. Each audit asserts that text and visual semantic states S0–S3 stay within `2e-6` of the full model.

## Pilot scope

The pilot is NC-only on Movies and Grocery, seeds 42/43/44, using the existing train and validation splits. Test evaluation and LP are disabled. The M0.1 comparison reads only the frozen `orthogonal_transport` records from `results/model_design/m0/pilot_records.json`. Run logs and checkpoints belong in ignored `outputs/`; shareable summaries and analysis belong in `results/model_design/m1/`.
