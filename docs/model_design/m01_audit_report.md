# M0.1 Frozen-Checkpoint Mechanism Audit

This audit used the six existing orthogonal M0.1 checkpoints (Movies/Grocery, seeds 42/43/44). It ran inference on the original validation split only; no training or test evaluation was performed. Per-run values are in [`progressive_dynamics.csv`](../../results/model_design/m01_audit/progressive_dynamics.csv), [`compatibility_decomposition.csv`](../../results/model_design/m01_audit/compatibility_decomposition.csv), [`transport_association.csv`](../../results/model_design/m01_audit/transport_association.csv), and [`frozen_counterfactuals.csv`](../../results/model_design/m01_audit/frozen_counterfactuals.csv).

## 1. Do progressive interactions continue to change semantic states?

Yes. Averaged over both modalities and all six runs, the relative state change was 1.056 at order 1, 0.367 at order 2, and 0.253 at order 3. Mean adjacent-state cosine rose from 0.743 to 0.930 and 0.966. Mean cumulative novelty from H0 rose from 0.257 to 0.427 and 0.544. Thus later states continue to add semantic change, while each successive update is smaller and more aligned with its predecessor. The evidence does not indicate an immediate fixed point after one interaction. State norms are not used as evidence of dynamics because LayerNorm affects them.

## 2. Does relation correction act in addition to the similarity prior?

Yes. Across modalities and runs, mean absolute prior logit was 0.527, mean absolute relation correction was 1.512, and mean correction share was 0.739 (median across run/modality means: 0.732). Learned `softplus(raw_tau)` averaged 0.981. Prior sign flips after adding correction averaged 0.981, with run/modality values from 0.862 to 1.000. The correction is active and often changes the sign of the similarity prior's logit contribution. This describes the trained decomposition; it does not establish that every correction is useful.

## 3. Does transport merely copy similarity or conductance?

The sampled Spearman associations vary by run and modality rather than showing a uniform one-to-one relationship. Across the 12 run/modality samples, mean rho was 0.127 for conductance versus transport displacement (range -0.185 to 0.620), 0.315 for cosine similarity versus mean absolute angle (0.166 to 0.543), and 0.324 for absolute relation correction versus mean absolute angle (-0.695 to 0.946). These are mechanism diagnostics, not success thresholds. They do not support describing transport as a simple copy of compatibility, though some runs show moderate association.

## 4. Does the trained model depend on transport?

Yes, in this frozen functional counterfactual. With all trained parameters and the NC head fixed, replacing learned orthogonal transport by identity reduced mean validation accuracy by 0.106 on Movies and 0.163 on Grocery; mean Macro-F1 fell by 0.108 and 0.167, respectively. This diagnoses reliance within the trained model, not a retrained identity-versus-orthogonal comparison.

Turning off only the conductance relation correction, while preserving relation-conditioned angles and transport, reduced accuracy by 0.027 / 0.012 and Macro-F1 by 0.036 / 0.022 on Movies / Grocery, respectively. The prior-only counterfactual therefore also degrades validation performance, with a smaller effect than transport-off.

## 5. Were implementation or numerical anomalies found?

No. All six required checkpoints loaded, the audit remained finite, and the prior-only counterfactual restored the original model state exactly after evaluation. No source change or retraining was needed. Detailed values and validation metrics are retained in the CSV/JSON outputs linked above.
