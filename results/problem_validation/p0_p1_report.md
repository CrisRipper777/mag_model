# P0.0 + P0.1 Problem Validation Report

## 1. Scope
- branch: `exp/problem_validation`
- commit: `5cf6be750dcb467aa59d8736be0007b9b197f3b3`
- datasets: Movies, Toys, Grocery, ele-fashion, Reddit-S
- model run seeds: 42, 43, 44; data/probe split seed: 42
- protocol: P0.0 feature-only probe and P0.1 decomposable one-hop message probe; checkpoint selection used probe_calib only; original val was used only for held-out analysis
- test set was not evaluated; test labels were not used for training, selection, analysis, or reporting.

## 2. Data Split Audit

| Dataset | Original train | Probe train | Probe calib | Original val | Original test (size only) | Analysis targets | Sampled edges | Leakage checks |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Movies | 10003 | 8002 | 2001 | 3334 | 3335 | 3334 | 30552 | probe train/calib/val disjoint; targets⊆val and ∩test=∅ |
| Toys | 12417 | 9933 | 2484 | 4139 | 4139 | 4139 | 22528 | probe train/calib/val disjoint; targets⊆val and ∩test=∅ |
| Grocery | 10244 | 8195 | 2049 | 3415 | 3415 | 3415 | 26053 | probe train/calib/val disjoint; targets⊆val and ∩test=∅ |
| ele-fashion | 58659 | 46927 | 11732 | 9777 | 29330 | 9777 | 37264 | probe train/calib/val disjoint; targets⊆val and ∩test=∅ |
| Reddit-S | 9536 | 7628 | 1908 | 3179 | 3179 | 3179 | 39613 | probe train/calib/val disjoint; targets⊆val and ∩test=∅ |

Runtime assertions verify probe_train ∩ probe_calib = ∅, both inner splits are disjoint from original val, analysis targets are a subset of original val, and targets do not intersect test indices. Test indices are used only for that final disjointness check; their labels are not accessed by the analysis.

## 3. P0.0 Semantic Probe

### Movies

| Seed | Best epoch | Probe train Acc / Macro-F1 | Probe calib Acc / Macro-F1 | heldout_analysis_val Acc / Macro-F1 |
|---:|---:|---:|---:|---:|
| 42 | 64 | 0.786 / 0.741 | 0.495 / 0.394 | 0.513 / 0.378 |
| 43 | 63 | 0.776 / 0.716 | 0.500 / 0.396 | 0.514 / 0.353 |
| 44 | 55 | 0.735 / 0.657 | 0.495 / 0.366 | 0.516 / 0.354 |

### Toys

| Seed | Best epoch | Probe train Acc / Macro-F1 | Probe calib Acc / Macro-F1 | heldout_analysis_val Acc / Macro-F1 |
|---:|---:|---:|---:|---:|
| 42 | 61 | 0.875 / 0.860 | 0.744 / 0.713 | 0.753 / 0.726 |
| 43 | 64 | 0.893 / 0.883 | 0.742 / 0.718 | 0.751 / 0.723 |
| 44 | 91 | 0.965 / 0.962 | 0.746 / 0.717 | 0.754 / 0.727 |

### Grocery

| Seed | Best epoch | Probe train Acc / Macro-F1 | Probe calib Acc / Macro-F1 | heldout_analysis_val Acc / Macro-F1 |
|---:|---:|---:|---:|---:|
| 42 | 73 | 0.939 / 0.906 | 0.800 / 0.708 | 0.770 / 0.674 |
| 43 | 81 | 0.954 / 0.933 | 0.804 / 0.724 | 0.779 / 0.697 |
| 44 | 103 | 0.988 / 0.985 | 0.797 / 0.706 | 0.776 / 0.695 |

### ele-fashion

| Seed | Best epoch | Probe train Acc / Macro-F1 | Probe calib Acc / Macro-F1 | heldout_analysis_val Acc / Macro-F1 |
|---:|---:|---:|---:|---:|
| 42 | 214 | 0.933 / 0.819 | 0.872 / 0.695 | 0.872 / 0.687 |
| 43 | 126 | 0.900 / 0.751 | 0.871 / 0.686 | 0.871 / 0.669 |
| 44 | 177 | 0.920 / 0.792 | 0.872 / 0.693 | 0.873 / 0.682 |

### Reddit-S

| Seed | Best epoch | Probe train Acc / Macro-F1 | Probe calib Acc / Macro-F1 | heldout_analysis_val Acc / Macro-F1 |
|---:|---:|---:|---:|---:|
| 42 | 72 | 0.995 / 0.989 | 0.935 / 0.877 | 0.928 / 0.877 |
| 43 | 93 | 0.999 / 0.997 | 0.939 / 0.886 | 0.928 / 0.876 |
| 44 | 111 | 1.000 / 0.999 | 0.937 / 0.878 | 0.927 / 0.873 |

### Cross-dataset summary

| Dataset | Train Acc | Train Macro-F1 | Calib Acc | Calib Macro-F1 | heldout_analysis_val Acc | heldout_analysis_val Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.766 ± 0.022 | 0.705 ± 0.035 | 0.497 ± 0.002 | 0.385 ± 0.014 | 0.515 ± 0.001 | 0.362 ± 0.012 |
| Toys | 0.911 ± 0.039 | 0.902 ± 0.043 | 0.744 ± 0.002 | 0.716 ± 0.002 | 0.752 ± 0.001 | 0.726 ± 0.002 |
| Grocery | 0.960 ± 0.021 | 0.941 ± 0.033 | 0.801 ± 0.003 | 0.713 ± 0.008 | 0.775 ± 0.004 | 0.689 ± 0.010 |
| ele-fashion | 0.917 ± 0.013 | 0.788 ± 0.028 | 0.871 ± 0.001 | 0.691 ± 0.004 | 0.872 ± 0.001 | 0.679 ± 0.008 |
| Reddit-S | 0.998 ± 0.002 | 0.995 ± 0.004 | 0.937 ± 0.002 | 0.880 ± 0.004 | 0.928 ± 0.001 | 0.875 ± 0.002 |

## 4. P0.1 Similarity vs Message Utility

### 4.1 Raw Semantic Similarity

| Dataset | Modality | Spearman CE | AUROC | High-sim harmful | Low-sim beneficial |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.002 ± 0.008 | 0.521 ± 0.006 | 0.317 ± 0.014 | 0.644 ± 0.007 |
| Movies | Visual | 0.019 ± 0.005 | 0.531 ± 0.001 | 0.299 ± 0.008 | 0.636 ± 0.004 |
| Toys | Text | 0.032 ± 0.010 | 0.522 ± 0.007 | 0.165 ± 0.015 | 0.797 ± 0.004 |
| Toys | Visual | 0.071 ± 0.016 | 0.555 ± 0.006 | 0.136 ± 0.007 | 0.768 ± 0.002 |
| Grocery | Text | 0.038 ± 0.003 | 0.547 ± 0.003 | 0.147 ± 0.003 | 0.771 ± 0.005 |
| Grocery | Visual | 0.052 ± 0.004 | 0.598 ± 0.004 | 0.119 ± 0.005 | 0.728 ± 0.004 |
| ele-fashion | Text | 0.110 ± 0.006 | 0.584 ± 0.003 | 0.184 ± 0.013 | 0.685 ± 0.019 |
| ele-fashion | Visual | 0.014 ± 0.060 | 0.478 ± 0.038 | 0.350 ± 0.082 | 0.697 ± 0.084 |
| Reddit-S | Text | -0.015 ± 0.010 | 0.617 ± 0.008 | 0.028 ± 0.003 | 0.907 ± 0.008 |
| Reddit-S | Visual | -0.010 ± 0.021 | 0.709 ± 0.006 | 0.043 ± 0.007 | 0.838 ± 0.016 |

### 4.2 Task-aware Semantic Similarity

| Dataset | Modality | Spearman CE | AUROC | High-sim harmful | Low-sim beneficial |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.008 ± 0.007 | 0.577 ± 0.005 | 0.238 ± 0.006 | 0.589 ± 0.009 |
| Movies | Visual | 0.020 ± 0.004 | 0.582 ± 0.005 | 0.245 ± 0.011 | 0.572 ± 0.006 |
| Toys | Text | 0.073 ± 0.005 | 0.645 ± 0.011 | 0.112 ± 0.014 | 0.652 ± 0.014 |
| Toys | Visual | 0.052 ± 0.021 | 0.662 ± 0.006 | 0.087 ± 0.005 | 0.664 ± 0.005 |
| Grocery | Text | 0.115 ± 0.013 | 0.726 ± 0.007 | 0.078 ± 0.004 | 0.559 ± 0.005 |
| Grocery | Visual | 0.084 ± 0.005 | 0.730 ± 0.004 | 0.068 ± 0.005 | 0.556 ± 0.004 |
| ele-fashion | Text | 0.099 ± 0.012 | 0.731 ± 0.018 | 0.084 ± 0.008 | 0.569 ± 0.028 |
| ele-fashion | Visual | 0.063 ± 0.040 | 0.593 ± 0.036 | 0.266 ± 0.082 | 0.546 ± 0.073 |
| Reddit-S | Text | -0.031 ± 0.023 | 0.787 ± 0.005 | 0.010 ± 0.001 | 0.823 ± 0.014 |
| Reddit-S | Visual | -0.074 ± 0.017 | 0.834 ± 0.015 | 0.038 ± 0.008 | 0.769 ± 0.013 |

### 4.3 Text vs Visual

| Dataset | Exact sign disagreement | Robust sign disagreement | Robust edge coverage |
|---|---:|---:|---:|
| Movies | 0.317 ± 0.006 | 0.317 ± 0.006 | 0.747 ± 0.003 |
| Toys | 0.173 ± 0.011 | 0.165 ± 0.017 | 0.738 ± 0.017 |
| Grocery | 0.179 ± 0.006 | 0.179 ± 0.008 | 0.705 ± 0.011 |
| ele-fashion | 0.300 ± 0.063 | 0.335 ± 0.071 | 0.641 ± 0.023 |
| Reddit-S | 0.069 ± 0.005 | 0.067 ± 0.009 | 0.727 ± 0.019 |

### 4.4 CE Utility vs Margin Utility

| Dataset | Modality | Raw margin Spearman | Probe margin Spearman | CE/margin sign agreement | CE-vs-margin Spearman |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.029 ± 0.008 | 0.114 ± 0.007 | 0.855 ± 0.016 | 0.808 ± 0.010 |
| Movies | Visual | 0.045 ± 0.008 | 0.112 ± 0.014 | 0.873 ± 0.010 | 0.818 ± 0.009 |
| Toys | Text | 0.053 ± 0.011 | 0.239 ± 0.013 | 0.909 ± 0.010 | 0.719 ± 0.013 |
| Toys | Visual | 0.130 ± 0.010 | 0.304 ± 0.004 | 0.916 ± 0.008 | 0.708 ± 0.015 |
| Grocery | Text | 0.088 ± 0.002 | 0.339 ± 0.008 | 0.923 ± 0.003 | 0.698 ± 0.006 |
| Grocery | Visual | 0.153 ± 0.006 | 0.335 ± 0.013 | 0.919 ± 0.004 | 0.699 ± 0.005 |
| ele-fashion | Text | 0.086 ± 0.006 | 0.312 ± 0.032 | 0.962 ± 0.002 | 0.737 ± 0.004 |
| ele-fashion | Visual | 0.018 ± 0.064 | 0.161 ± 0.047 | 0.885 ± 0.064 | 0.783 ± 0.022 |
| Reddit-S | Text | 0.111 ± 0.010 | 0.199 ± 0.024 | 0.894 ± 0.021 | 0.667 ± 0.016 |
| Reddit-S | Visual | 0.108 ± 0.020 | 0.118 ± 0.027 | 0.983 ± 0.004 | 0.780 ± 0.041 |

### Node-bootstrap 95% CIs (1,000 target-node replicates, seed 42; reported per seed and separate from three-seed std)

| Dataset | Modality | Raw ρ CE | Probe ρ CE | Raw high-sim harmful | Probe high-sim harmful | Raw low-sim beneficial | Probe low-sim beneficial | Exact T/V disagreement | Robust T/V disagreement |
|---|---|---|---|---|---|---|---|---|---|
| Movies | Text | 42: [-0.019, 0.022]; 43: [-0.008, 0.032]; 44: [-0.027, 0.012] | 42: [-0.014, 0.035]; 43: [-0.009, 0.040]; 44: [-0.027, 0.027] | 42: [0.303, 0.364]; 43: [0.272, 0.322]; 44: [0.294, 0.348] | 42: [0.211, 0.282]; 43: [0.202, 0.266]; 44: [0.204, 0.271] | 42: [0.610, 0.659]; 43: [0.617, 0.663]; 44: [0.629, 0.677] | 42: [0.559, 0.609]; 43: [0.554, 0.606]; 44: [0.572, 0.629] | 42: [0.313, 0.338]; 43: [0.302, 0.323]; 44: [0.301, 0.323] | 42: [0.314, 0.337]; 43: [0.305, 0.325]; 44: [0.300, 0.321] |
| Movies | Visual | 42: [-0.004, 0.040]; 43: [-0.009, 0.035]; 44: [0.004, 0.047] | 42: [-0.005, 0.042]; 43: [-0.008, 0.043]; 44: [0.001, 0.053] | 42: [0.274, 0.326]; 43: [0.266, 0.317]; 44: [0.283, 0.340] | 42: [0.214, 0.282]; 43: [0.200, 0.262]; 44: [0.220, 0.296] | 42: [0.608, 0.654]; 43: [0.617, 0.664]; 44: [0.609, 0.654] | 42: [0.553, 0.597]; 43: [0.553, 0.601]; 44: [0.543, 0.587] | 42: [0.313, 0.338]; 43: [0.302, 0.323]; 44: [0.301, 0.323] | 42: [0.314, 0.337]; 43: [0.305, 0.325]; 44: [0.300, 0.321] |
| Toys | Text | 42: [0.005, 0.048]; 43: [0.024, 0.070]; 44: [0.003, 0.044] | 42: [0.053, 0.104]; 43: [0.047, 0.102]; 44: [0.040, 0.090] | 42: [0.163, 0.204]; 43: [0.127, 0.167]; 44: [0.144, 0.185] | 42: [0.113, 0.152]; 43: [0.085, 0.124]; 44: [0.083, 0.118] | 42: [0.774, 0.813]; 43: [0.784, 0.824]; 44: [0.776, 0.815] | 42: [0.625, 0.670]; 43: [0.645, 0.696]; 44: [0.614, 0.666] | 42: [0.172, 0.189]; 43: [0.150, 0.166]; 44: [0.171, 0.189] | 42: [0.165, 0.183]; 43: [0.133, 0.150]; 44: [0.170, 0.188] |
| Toys | Visual | 42: [0.048, 0.096]; 43: [0.064, 0.113]; 44: [0.027, 0.073] | 42: [0.026, 0.078]; 43: [0.051, 0.101]; 44: [-0.002, 0.051] | 42: [0.121, 0.156]; 43: [0.110, 0.144]; 44: [0.128, 0.162] | 42: [0.078, 0.111]; 43: [0.068, 0.101]; 44: [0.070, 0.100] | 42: [0.747, 0.790]; 43: [0.742, 0.785]; 44: [0.749, 0.789] | 42: [0.637, 0.687]; 43: [0.638, 0.684]; 44: [0.647, 0.693] | 42: [0.172, 0.189]; 43: [0.150, 0.166]; 44: [0.171, 0.189] | 42: [0.165, 0.183]; 43: [0.133, 0.150]; 44: [0.170, 0.188] |
| Grocery | Text | 42: [0.016, 0.056]; 43: [0.015, 0.055]; 44: [0.020, 0.061] | 42: [0.099, 0.158]; 43: [0.069, 0.127]; 44: [0.084, 0.145] | 42: [0.128, 0.170]; 43: [0.129, 0.172]; 44: [0.123, 0.165] | 42: [0.061, 0.101]; 43: [0.065, 0.105]; 44: [0.056, 0.093] | 42: [0.749, 0.799]; 43: [0.741, 0.787]; 44: [0.747, 0.798] | 42: [0.539, 0.590]; 43: [0.525, 0.583]; 44: [0.526, 0.585] | 42: [0.170, 0.190]; 43: [0.177, 0.197]; 44: [0.162, 0.179] | 42: [0.164, 0.188]; 43: [0.179, 0.202]; 44: [0.161, 0.181] |
| Grocery | Visual | 42: [0.030, 0.079]; 43: [0.021, 0.070]; 44: [0.032, 0.077] | 42: [0.054, 0.108]; 43: [0.051, 0.106]; 44: [0.062, 0.120] | 42: [0.096, 0.128]; 43: [0.105, 0.139]; 44: [0.104, 0.140] | 42: [0.047, 0.077]; 43: [0.054, 0.086]; 44: [0.059, 0.095] | 42: [0.706, 0.749]; 43: [0.702, 0.747]; 44: [0.712, 0.754] | 42: [0.535, 0.588]; 43: [0.527, 0.579]; 44: [0.528, 0.577] | 42: [0.170, 0.190]; 43: [0.177, 0.197]; 44: [0.162, 0.179] | 42: [0.164, 0.188]; 43: [0.179, 0.202]; 44: [0.161, 0.181] |
| ele-fashion | Text | 42: [0.074, 0.129]; 43: [0.091, 0.143]; 44: [0.086, 0.138] | 42: [0.078, 0.131]; 43: [0.056, 0.107]; 44: [0.085, 0.135] | 42: [0.174, 0.232]; 43: [0.143, 0.200]; 44: [0.155, 0.208] | 42: [0.075, 0.111]; 43: [0.073, 0.110]; 44: [0.059, 0.090] | 42: [0.641, 0.686]; 43: [0.687, 0.730]; 44: [0.657, 0.701] | 42: [0.516, 0.569]; 43: [0.580, 0.633]; 44: [0.529, 0.580] | 42: [0.218, 0.239]; 43: [0.367, 0.393]; 44: [0.278, 0.304] | 42: [0.244, 0.267]; 43: [0.415, 0.441]; 44: [0.311, 0.335] |
| ele-fashion | Visual | 42: [0.066, 0.112]; 43: [-0.008, 0.031]; 44: [-0.080, -0.038] | 42: [0.069, 0.111]; 43: [0.071, 0.113]; 44: [-0.015, 0.029] | 42: [0.211, 0.264]; 43: [0.408, 0.455]; 44: [0.354, 0.405] | 42: [0.141, 0.181]; 43: [0.335, 0.383]; 44: [0.253, 0.305] | 42: [0.733, 0.766]; 43: [0.557, 0.603]; 44: [0.741, 0.780] | 42: [0.577, 0.618]; 43: [0.418, 0.470]; 44: [0.572, 0.617] | 42: [0.218, 0.239]; 43: [0.367, 0.393]; 44: [0.278, 0.304] | 42: [0.244, 0.267]; 43: [0.415, 0.441]; 44: [0.311, 0.335] |
| Reddit-S | Text | 42: [-0.042, 0.026]; 43: [-0.044, 0.030]; 44: [-0.067, 0.010] | 42: [-0.051, 0.025]; 43: [-0.062, 0.026]; 44: [-0.104, -0.021] | 42: [0.025, 0.039]; 43: [0.017, 0.029]; 44: [0.022, 0.037] | 42: [0.006, 0.015]; 43: [0.005, 0.014]; 44: [0.006, 0.017] | 42: [0.882, 0.910]; 43: [0.901, 0.925]; 44: [0.899, 0.924] | 42: [0.776, 0.830]; 43: [0.811, 0.861]; 44: [0.799, 0.851] | 42: [0.069, 0.085]; 43: [0.059, 0.075]; 44: [0.057, 0.071] | 42: [0.069, 0.089]; 43: [0.050, 0.067]; 44: [0.056, 0.071] |
| Reddit-S | Visual | 42: [-0.071, 0.038]; 43: [-0.036, 0.074]; 44: [-0.081, 0.019] | 42: [-0.146, -0.036]; 43: [-0.104, 0.001]; 44: [-0.131, -0.030] | 42: [0.031, 0.070]; 43: [0.027, 0.068]; 44: [0.022, 0.046] | 42: [0.025, 0.065]; 43: [0.025, 0.065]; 44: [0.017, 0.040] | 42: [0.782, 0.868]; 43: [0.784, 0.868]; 44: [0.823, 0.894] | 42: [0.705, 0.808]; 43: [0.714, 0.816]; 44: [0.740, 0.830] | 42: [0.069, 0.085]; 43: [0.059, 0.075]; 44: [0.057, 0.071] | 42: [0.069, 0.089]; 43: [0.050, 0.067]; 44: [0.056, 0.071] |

## 5. Cross-dataset Summary

Values are mean ± population std over seeds. Bootstrap CIs remain listed separately above.

| Dataset | Modality | Spearman | AUROC | High-Sim Harmful | Low-Sim Beneficial | Robust T/V disagreement |
|---|---|---:|---:|---:|---:|---:|
| Movies | Text | 0.008 ± 0.007 | 0.577 ± 0.005 | 0.238 ± 0.006 | 0.589 ± 0.009 | 0.317 ± 0.006 |
| Movies | Visual | 0.020 ± 0.004 | 0.582 ± 0.005 | 0.245 ± 0.011 | 0.572 ± 0.006 | 0.317 ± 0.006 |
| Toys | Text | 0.073 ± 0.005 | 0.645 ± 0.011 | 0.112 ± 0.014 | 0.652 ± 0.014 | 0.165 ± 0.017 |
| Toys | Visual | 0.052 ± 0.021 | 0.662 ± 0.006 | 0.087 ± 0.005 | 0.664 ± 0.005 | 0.165 ± 0.017 |
| Grocery | Text | 0.115 ± 0.013 | 0.726 ± 0.007 | 0.078 ± 0.004 | 0.559 ± 0.005 | 0.179 ± 0.008 |
| Grocery | Visual | 0.084 ± 0.005 | 0.730 ± 0.004 | 0.068 ± 0.005 | 0.556 ± 0.004 | 0.179 ± 0.008 |
| ele-fashion | Text | 0.099 ± 0.012 | 0.731 ± 0.018 | 0.084 ± 0.008 | 0.569 ± 0.028 | 0.335 ± 0.071 |
| ele-fashion | Visual | 0.063 ± 0.040 | 0.593 ± 0.036 | 0.266 ± 0.082 | 0.546 ± 0.073 | 0.335 ± 0.071 |
| Reddit-S | Text | -0.031 ± 0.023 | 0.787 ± 0.005 | 0.010 ± 0.001 | 0.823 ± 0.014 | 0.067 ± 0.009 |
| Reddit-S | Visual | -0.074 ± 0.017 | 0.834 ± 0.015 | 0.038 ± 0.008 | 0.769 ± 0.013 | 0.067 ± 0.009 |

## 6. Hypothesis Decision

H1: **Supported**

5/5 datasets meet at least two H1 diagnostic criteria; robust Text/Visual disagreement has a positive node-bootstrap 95% CI in at least two seeds for 5/5 datasets. Movies: 4/4 H1 diagnostic criteria; mean robust T/V disagreement=0.317 ± 0.006; node-bootstrap lower CI > 0 in 3/3 seeds Toys: 4/4 H1 diagnostic criteria; mean robust T/V disagreement=0.165 ± 0.017; node-bootstrap lower CI > 0 in 3/3 seeds Grocery: 4/4 H1 diagnostic criteria; mean robust T/V disagreement=0.179 ± 0.008; node-bootstrap lower CI > 0 in 3/3 seeds ele-fashion: 4/4 H1 diagnostic criteria; mean robust T/V disagreement=0.335 ± 0.071; node-bootstrap lower CI > 0 in 3/3 seeds Reddit-S: 3/4 H1 diagnostic criteria; mean robust T/V disagreement=0.067 ± 0.009; node-bootstrap lower CI > 0 in 3/3 seeds

For the predeclared four diagnostic criteria, each dataset is counted when at least one modality/similarity-space mean meets that criterion: |Spearman| < 0.5, signed AUROC < 0.75, high-sim harmful rate ≥ 10%, or low-sim beneficial rate ≥ 10%. Robust Text/Visual disagreement is counted as non-zero when its target-node bootstrap 95% CI lower bound is above zero in at least two of three seeds. Strong support requires at least two criteria in at least four datasets plus non-zero robust disagreement in at least three datasets. The AUROC is not reflected around 0.5.

## 7. What the Evidence Supports

- Across 30 task-aware run × modality CE-correlation results (10 dataset × modality groups over three seeds), Spearman spans -0.092 to 0.130; this is weak rank association overall.
- Task-aware similarity AUROC is below 0.75 in 23/30 dataset × seed × modality observations; signed AUROC values are retained as measured.
- Probe-similarity top-20% edges are harmful at a rate ≥10% in 11/30 run × modality observations. Probe-similarity bottom-20% edges are beneficial in 30/30 observations; the observed rate range is 0.443–0.838.
- Robust Text/Visual CE-utility sign disagreement is non-zero in all five datasets; dataset means range 0.067–0.335. The robust coverage and per-seed node-bootstrap intervals are reported above.
- Raw and task-aware similarities differ by dataset and modality. These findings support the limited claim that similarity is an imperfect surrogate for the one-hop task utility measured here.

## 8. What the Evidence Does NOT Support

P0.1 does not establish that vector relation state is necessary, MoE is necessary, basis routing is necessary, semantic transformation is better than scalar weighting, deeper structural context is heterogeneous, or any final proposed architecture is superior.

## 9. Implications for P0.2

If proceeding, the next validation should directly compare a scalar similarity-reliability baseline with a relation-conditioned alternative under the same frozen semantic embeddings, fixed probe split, and held-out original-val utility analysis. The current evidence motivates that comparison but does not predetermine its winner; define the controls and selection protocol before implementing P0.2.

## Diagnostic Figures

- [Movies_figure_a_similarity_utility.png](plots/Movies_figure_a_similarity_utility.png)
- [Movies_figure_b_quantile_rates.png](plots/Movies_figure_b_quantile_rates.png)
- [Movies_figure_c_text_visual_quadrants.png](plots/Movies_figure_c_text_visual_quadrants.png)
- [Toys_figure_a_similarity_utility.png](plots/Toys_figure_a_similarity_utility.png)
- [Toys_figure_b_quantile_rates.png](plots/Toys_figure_b_quantile_rates.png)
- [Toys_figure_c_text_visual_quadrants.png](plots/Toys_figure_c_text_visual_quadrants.png)
- [Grocery_figure_a_similarity_utility.png](plots/Grocery_figure_a_similarity_utility.png)
- [Grocery_figure_b_quantile_rates.png](plots/Grocery_figure_b_quantile_rates.png)
- [Grocery_figure_c_text_visual_quadrants.png](plots/Grocery_figure_c_text_visual_quadrants.png)
- [ele-fashion_figure_a_similarity_utility.png](plots/ele-fashion_figure_a_similarity_utility.png)
- [ele-fashion_figure_b_quantile_rates.png](plots/ele-fashion_figure_b_quantile_rates.png)
- [ele-fashion_figure_c_text_visual_quadrants.png](plots/ele-fashion_figure_c_text_visual_quadrants.png)
- [Reddit-S_figure_a_similarity_utility.png](plots/Reddit-S_figure_a_similarity_utility.png)
- [Reddit-S_figure_b_quantile_rates.png](plots/Reddit-S_figure_b_quantile_rates.png)
- [Reddit-S_figure_c_text_visual_quadrants.png](plots/Reddit-S_figure_c_text_visual_quadrants.png)
