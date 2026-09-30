# M2 Pilot Report

Source: `1793ba2136909255637e4ebd2a6d1231921434b7` on `exp/m11_corrected_provenance_audit`.
Branch: `exp/provenance_evidence_m2`. NC only; original train was used for fitting, original validation for checkpoint selection, test evaluation and LP were disabled.

## Main retrained comparison

| Dataset | Variant | Validation accuracy (mean ± population SD) | Validation Macro-F1 (mean ± population SD) | Trainable model parameters | Best epoch mean |
|---|---|---:|---:|---:|---:|
| Movies | M2-S | 52.52% ± 0.53% | 31.84% ± 4.08% | 1,032,468 | 46.3 |
| Movies | M2-P | 52.96% ± 0.35% | 33.94% ± 0.53% | 1,032,468 | 46.0 |
| Grocery | M2-S | 82.12% ± 0.27% | 71.32% ± 1.25% | 1,032,468 | 110.7 |
| Grocery | M2-P | 81.76% ± 0.31% | 71.25% ± 1.05% | 1,032,468 | 96.7 |

### Seed-paired M2-P minus M2-S

| Dataset | Seed | Δ validation accuracy | Δ validation Macro-F1 |
|---|---:|---:|---:|
| Movies | 42 | 0.75% | 6.32% |
| Movies | 43 | 0.30% | 3.42% |
| Movies | 44 | 0.27% | -3.43% |
| Grocery | 42 | -0.56% | -0.54% |
| Grocery | 43 | -0.12% | 1.37% |
| Grocery | 44 | -0.41% | -1.06% |

## Resources and parameter status

- Movies M2-S seed 42: 70 epochs, 0.9 s/epoch, 61.1 s total, peak allocated/reserved 7.29/7.92 GiB; trainable 1,032,468, legacy frozen 41,096.
- Movies M2-P seed 42: 84 epochs, 1.1 s/epoch, 93.6 s total, peak allocated/reserved 7.62/8.46 GiB; trainable 1,032,468, legacy frozen 41,096.
- Movies M2-S seed 43: 72 epochs, 0.9 s/epoch, 63.0 s total, peak allocated/reserved 7.29/7.92 GiB; trainable 1,032,468, legacy frozen 41,096.
- Movies M2-P seed 43: 71 epochs, 1.1 s/epoch, 79.0 s total, peak allocated/reserved 7.62/8.46 GiB; trainable 1,032,468, legacy frozen 41,096.
- Movies M2-S seed 44: 87 epochs, 0.9 s/epoch, 76.0 s total, peak allocated/reserved 7.29/7.92 GiB; trainable 1,032,468, legacy frozen 41,096.
- Movies M2-P seed 44: 73 epochs, 1.1 s/epoch, 81.3 s total, peak allocated/reserved 7.62/8.46 GiB; trainable 1,032,468, legacy frozen 41,096.
- Grocery M2-S seed 42: 157 epochs, 0.8 s/epoch, 126.2 s total, peak allocated/reserved 6.92/7.49 GiB; trainable 1,032,468, legacy frozen 41,096.
- Grocery M2-P seed 42: 151 epochs, 1.0 s/epoch, 155.3 s total, peak allocated/reserved 7.26/8.02 GiB; trainable 1,032,468, legacy frozen 41,096.
- Grocery M2-S seed 43: 111 epochs, 0.8 s/epoch, 92.6 s total, peak allocated/reserved 6.92/7.49 GiB; trainable 1,032,468, legacy frozen 41,096.
- Grocery M2-P seed 43: 115 epochs, 1.2 s/epoch, 136.3 s total, peak allocated/reserved 7.26/8.02 GiB; trainable 1,032,468, legacy frozen 41,096.
- Grocery M2-S seed 44: 154 epochs, 1.2 s/epoch, 179.2 s total, peak allocated/reserved 6.92/7.49 GiB; trainable 1,032,468, legacy frozen 41,096.
- Grocery M2-P seed 44: 114 epochs, 1.7 s/epoch, 189.9 s total, peak allocated/reserved 7.26/8.02 GiB; trainable 1,032,468, legacy frozen 41,096.

Legacy parameters remain only because M2 inherits the frozen M1 implementation for correctness; they are not part of the M2 conceptual architecture and should be removed in the later architecture-freeze refactor.

## Evidence and attention diagnostics

## Aggregate Stage III diagnostics

All 12 models have 1,032,468 trainable encoder parameters and 41,096 frozen legacy M1 readout parameters. The final unit checks measured a maximum absolute error of 0.0 for M1-to-M2 H0/S/G equivalence and 0.0 for M2-S/M2-P outputs at initialization. All 24 model × dataset × seed × query collapse checks were false.

Across the six M2-P checkpoints, provenance changed both keys and values:

| Modality | Order | Mean key relative change | Mean value relative change |
|---|---:|---:|---:|
| Text | 1 | 0.576 | 0.269 |
| Text | 2 | 0.516 | 0.234 |
| Text | 3 | 0.512 | 0.226 |
| Visual | 1 | 0.632 | 0.278 |
| Visual | 2 | 0.618 | 0.273 |
| Visual | 3 | 0.501 | 0.254 |

Average M2-P retrieval diagnostics by dataset and intrinsic query:

| Dataset | Query | Null mass | Same-modality mass | Cross-modality mass | Entropy | Effective tokens | Node-wise weight variance |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.074 | 0.244 | 0.682 | 1.556 | 4.824 | 0.00846 |
| Movies | Visual | 0.114 | 0.516 | 0.369 | 1.775 | 5.943 | 0.00499 |
| Grocery | Text | 0.063 | 0.338 | 0.599 | 1.494 | 4.584 | 0.01176 |
| Grocery | Visual | 0.054 | 0.595 | 0.351 | 1.414 | 4.260 | 0.01400 |

No query collapsed to the null, a single token, or a node-invariant attention vector. Same- and cross-modality evidence both receive mass. The corrected Same-Modality-Only intervention changes the sampled attention distribution by mean L1 1.052 on Movies and 0.950 on Grocery.

| Dataset | Variant | Retrieval/H0 norm ratio (text, visual) | 1-cos(H0,Z) mean (text, visual) |
|---|---|---|---|
| Movies | M2-S | 0.580, 0.459 | 0.216, 0.188 |
| Movies | M2-P | 0.662, 0.470 | 0.247, 0.190 |
| Grocery | M2-S | 0.937, 1.123 | 0.313, 0.341 |
| Grocery | M2-P | 0.965, 1.192 | 0.321, 0.354 |

The Grocery visual retrieval norm exceeds the intrinsic norm on average in both variants; M2-P is 1.192. The intrinsic state remains the explicit residual anchor, while this ratio and the final cosine changes show that retrieved structure is substantial and should be monitored in a broader benchmark.

### Movies M2-S seed 42

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0661 | 0.4734 | 0.4605 | 1.6612 | 5.340 | 8.202e-03 | none |
| visual | 0.1136 | 0.5252 | 0.3612 | 1.7567 | 5.834 | 4.459e-03 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Movies M2-P seed 42

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0777 | 0.3521 | 0.5702 | 1.6118 | 5.091 | 1.073e-02 | none |
| visual | 0.1024 | 0.5220 | 0.3755 | 1.7551 | 5.832 | 5.347e-03 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Movies M2-S seed 43

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0870 | 0.2575 | 0.6556 | 1.5754 | 4.875 | 7.913e-03 | none |
| visual | 0.1259 | 0.5048 | 0.3692 | 1.8201 | 6.198 | 3.655e-03 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Movies M2-P seed 43

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0804 | 0.1880 | 0.7316 | 1.4496 | 4.345 | 1.085e-02 | none |
| visual | 0.1177 | 0.5323 | 0.3501 | 1.7664 | 5.895 | 4.332e-03 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Movies M2-S seed 44

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0787 | 0.2791 | 0.6423 | 1.6915 | 5.463 | 4.525e-03 | none |
| visual | 0.1118 | 0.4878 | 0.4005 | 1.7929 | 6.042 | 5.772e-03 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Movies M2-P seed 44

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0642 | 0.1906 | 0.7452 | 1.6076 | 5.035 | 3.804e-03 | none |
| visual | 0.1230 | 0.4944 | 0.3825 | 1.8042 | 6.103 | 5.284e-03 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Grocery M2-S seed 42

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0575 | 0.4464 | 0.4961 | 1.5564 | 4.818 | 1.151e-02 | none |
| visual | 0.0514 | 0.5577 | 0.3909 | 1.5646 | 4.857 | 1.090e-02 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Grocery M2-P seed 42

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0447 | 0.3718 | 0.5834 | 1.4474 | 4.353 | 1.281e-02 | none |
| visual | 0.0387 | 0.6142 | 0.3471 | 1.4346 | 4.307 | 1.287e-02 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Grocery M2-S seed 43

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.1031 | 0.3416 | 0.5554 | 1.6490 | 5.294 | 8.400e-03 | none |
| visual | 0.0993 | 0.5689 | 0.3317 | 1.5891 | 4.996 | 1.001e-02 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Grocery M2-P seed 43

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0673 | 0.3152 | 0.6176 | 1.5433 | 4.806 | 9.466e-03 | none |
| visual | 0.0567 | 0.5870 | 0.3563 | 1.4098 | 4.241 | 1.324e-02 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Grocery M2-S seed 44

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0742 | 0.4165 | 0.5093 | 1.6068 | 5.111 | 1.129e-02 | none |
| visual | 0.0723 | 0.5433 | 0.3845 | 1.5174 | 4.684 | 1.359e-02 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

### Grocery M2-P seed 44

| Query | Null mass | Same-modality mass | Cross-modality mass | Entropy mean | Effective tokens mean | Node-wise attention variance | Collapse flags |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0769 | 0.3281 | 0.5950 | 1.4918 | 4.593 | 1.302e-02 | none |
| visual | 0.0662 | 0.5834 | 0.3503 | 1.3984 | 4.232 | 1.589e-02 | none |

ΔS norms and provenance-induced key/value relative changes by modality and order are recorded in `results/model_design/m2/mechanism_diagnostics.json`.

## Frozen M2-P counterfactuals

All Stage I/II invariance checks passed at max absolute error ≤ 2e-6. Attention changes are sampled over at most 8192 deterministically selected nodes. Attention mass describes retrieval behavior and is not a causal importance score.

| Dataset | Intervention | Mean Δ Val Acc | Mean Δ Val Macro-F1 |
|---|---|---:|---:|
| Movies | Provenance-Off | -0.65% | -0.66% |
| Movies | Node Shuffle (9 runs) | -0.53% | -0.52% |
| Movies | Order Mismatch | -2.36% | -1.95% |
| Movies | Same-Modality-Only | -1.12% | -1.49% |
| Grocery | Provenance-Off | -1.88% | -2.00% |
| Grocery | Node Shuffle (9 runs) | -5.62% | -5.07% |
| Grocery | Order Mismatch | -9.04% | -8.50% |
| Grocery | Same-Modality-Only | -0.89% | -0.63% |

Per-checkpoint values follow:

| Dataset | Seed | Intervention | Δ Val Acc | Δ Val Macro-F1 | Key change | Value change | Retrieval change | Final z change | Attention L1 change |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | provenance_off | -0.57% | -0.91% | 0.4923 | 0.2206 | 0.5389 | 0.2121 | 0.3634 |
| Movies | 42 | provenance_shuffle 3407 | -1.56% | -1.99% | 0.2851 | 0.1574 | 0.3788 | 0.1560 | 0.2027 |
| Movies | 42 | provenance_shuffle 3408 | -0.69% | -1.26% | 0.2845 | 0.1571 | 0.3761 | 0.1547 | 0.2025 |
| Movies | 42 | provenance_shuffle 3409 | -0.96% | -1.41% | 0.2842 | 0.1568 | 0.3754 | 0.1545 | 0.2017 |
| Movies | 42 | order_mismatch | -1.05% | -1.15% | 0.5391 | 0.2300 | 0.6291 | 0.2527 | 0.4203 |
| Movies | 42 | same_modality_only | -1.38% | -0.85% | 0.0000 | 0.0000 | 0.7608 | 0.2341 | 0.9458 |
| Movies | 43 | provenance_off | -0.21% | 0.97% | 0.4744 | 0.1735 | 0.5361 | 0.2060 | 0.3739 |
| Movies | 43 | provenance_shuffle 3407 | -0.12% | 0.57% | 0.2289 | 0.1087 | 0.3025 | 0.1140 | 0.1955 |
| Movies | 43 | provenance_shuffle 3408 | -0.21% | 0.16% | 0.2288 | 0.1087 | 0.3025 | 0.1142 | 0.1966 |
| Movies | 43 | provenance_shuffle 3409 | -0.42% | -0.51% | 0.2283 | 0.1084 | 0.3007 | 0.1133 | 0.1950 |
| Movies | 43 | order_mismatch | -3.30% | -1.04% | 0.5725 | 0.2157 | 0.8278 | 0.3237 | 0.5498 |
| Movies | 43 | same_modality_only | -1.41% | -2.75% | 0.0000 | 0.0000 | 0.8389 | 0.2713 | 1.0817 |
| Movies | 44 | provenance_off | -1.17% | -2.02% | 0.5305 | 0.1987 | 0.6006 | 0.2385 | 0.3479 |
| Movies | 44 | provenance_shuffle 3407 | -0.15% | -0.00% | 0.2096 | 0.1106 | 0.2586 | 0.1011 | 0.1170 |
| Movies | 44 | provenance_shuffle 3408 | -0.12% | 0.36% | 0.2096 | 0.1105 | 0.2591 | 0.1015 | 0.1177 |
| Movies | 44 | provenance_shuffle 3409 | -0.57% | -0.55% | 0.2092 | 0.1106 | 0.2584 | 0.1014 | 0.1169 |
| Movies | 44 | order_mismatch | -2.73% | -3.66% | 0.5727 | 0.2110 | 0.9011 | 0.3634 | 0.4131 |
| Movies | 44 | same_modality_only | -0.57% | -0.86% | 0.0000 | 0.0000 | 0.8556 | 0.2854 | 1.1277 |
| Grocery | 42 | provenance_off | -1.05% | -1.34% | 0.5886 | 0.3369 | 0.5955 | 0.2782 | 0.4967 |
| Grocery | 42 | provenance_shuffle 3407 | -7.26% | -6.94% | 0.5059 | 0.3428 | 0.5982 | 0.3411 | 0.3699 |
| Grocery | 42 | provenance_shuffle 3408 | -7.99% | -7.48% | 0.5035 | 0.3411 | 0.5940 | 0.3395 | 0.3668 |
| Grocery | 42 | provenance_shuffle 3409 | -6.79% | -5.33% | 0.5032 | 0.3405 | 0.5927 | 0.3380 | 0.3687 |
| Grocery | 42 | order_mismatch | -5.65% | -7.16% | 0.6455 | 0.3475 | 0.7046 | 0.3753 | 0.6286 |
| Grocery | 42 | same_modality_only | -1.00% | -0.97% | 0.0000 | 0.0000 | 0.6503 | 0.2712 | 0.9305 |
| Grocery | 43 | provenance_off | -2.75% | -2.46% | 0.6329 | 0.2886 | 0.6713 | 0.3207 | 0.6140 |
| Grocery | 43 | provenance_shuffle 3407 | -4.36% | -4.97% | 0.4091 | 0.2719 | 0.5282 | 0.2738 | 0.3786 |
| Grocery | 43 | provenance_shuffle 3408 | -3.89% | -2.98% | 0.4065 | 0.2703 | 0.5272 | 0.2735 | 0.3770 |
| Grocery | 43 | provenance_shuffle 3409 | -3.72% | -2.64% | 0.4083 | 0.2712 | 0.5270 | 0.2731 | 0.3757 |
| Grocery | 43 | order_mismatch | -13.32% | -11.97% | 0.7435 | 0.3059 | 0.9419 | 0.5354 | 0.8184 |
| Grocery | 43 | same_modality_only | -0.64% | -0.23% | 0.0000 | 0.0000 | 0.6724 | 0.2422 | 0.9739 |
| Grocery | 44 | provenance_off | -1.84% | -2.20% | 0.6277 | 0.3152 | 0.6584 | 0.3227 | 0.5679 |
| Grocery | 44 | provenance_shuffle 3407 | -5.77% | -5.48% | 0.4256 | 0.3014 | 0.5871 | 0.3082 | 0.3889 |
| Grocery | 44 | provenance_shuffle 3408 | -5.12% | -4.74% | 0.4227 | 0.2990 | 0.5847 | 0.3074 | 0.3863 |
| Grocery | 44 | provenance_shuffle 3409 | -5.65% | -5.09% | 0.4210 | 0.2976 | 0.5808 | 0.3058 | 0.3844 |
| Grocery | 44 | order_mismatch | -8.14% | -6.37% | 0.6778 | 0.2903 | 0.7674 | 0.4158 | 0.6706 |
| Grocery | 44 | same_modality_only | -1.02% | -0.70% | 0.0000 | 0.0000 | 0.6547 | 0.2527 | 0.9453 |

## Interpretation and decision

M2-P improves the Movies mean over M2-S by 0.44 percentage points in validation accuracy and 2.10 points in Macro-F1. Its paired accuracy delta is positive at all three Movies seeds. On Grocery, M2-P is lower by 0.36 points in mean accuracy and 0.08 points in mean Macro-F1; paired accuracy deltas are negative at all three seeds, while Macro-F1 is mixed. This supports a dataset-dependent effect, not a general task-metric win.

The mechanism evidence is clear. Provenance-specific heads learn sizable changes to keys and values. Provenance-Off changes M2-P retrieval and final representations and reduces mean validation accuracy by 0.65 points on Movies and 1.88 points on Grocery. Degree-matched provenance shuffling reduces mean validation accuracy by 0.53 points on Movies and 5.62 points on Grocery. Provenance order mismatch reduces it by 2.36 points on Movies and 9.04 points on Grocery. These frozen interventions show that the model uses both provenance content and its node/order correspondence, especially on Grocery. Same-Modality-Only changes retrieval substantially; its mean validation accuracy effect is smaller, about -1.12 points on Movies and -0.89 points on Grocery, consistent with cross-modal access being active without dominating the task metric.

Attention did not collapse. Null mass is low but nonzero, multiple structural tokens receive attention, and node-wise attention varies. The main limitation is task consistency: M2-P's task improvement appears on Movies, while Grocery is approximately tied or slightly lower. Retrieval also contributes substantial representation movement, particularly for Grocery visual states. These effects should remain explicit when results are expanded.

### Overall recommendation

**Freeze Stage I + Stage II + Stage III as the complete architecture for the next evaluation stage.** M2-P passes the mechanism checks: its heads learn, frozen Provenance-Off and correspondence interventions change retrieval, and attention does not collapse. Its task results are mixed but do not show a large systematic deterioration in this two-dataset pilot. Keep M2-S as the matched semantic-retrieval control and describe M2-P as provenance-sensitive rather than generally superior. The 2-dataset, 3-seed pilot does not establish broader generalization.

This report stops at the pilot. It does not run the five-dataset benchmark, formal ablations, or LP.
