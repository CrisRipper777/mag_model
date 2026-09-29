# P0.1+ Utility Base-Rate and Enrichment Audit

## 1. Scope

- source branch / commit: `exp/problem_validation` / `c56f6589450ab0c7abf3960f0e7fb2996db383d6`
- analysis branch / code commit: `exp/problem_validation_p01plus` / `bf1c93b02cdf5c57d41721bce0905ef8413d93ee`
- analysis uses only the 15 existing `edge_analysis.pt` files; no training or model selection was run.
- no test files, test labels, or test metrics were accessed; no test evaluation was performed.
- each artifact's exact existing sampled validation relation population was reused. No relation population was regenerated.
- all rates describe the fixed sampled validation relation population (at most 64 sampled relations per target), not every edge in each graph.
- utility definitions are unchanged: `U > 0` beneficial, `U < 0` harmful, and exact floating-point `U == 0` zero; no epsilon was applied.
- quintiles use the complete dataset × seed × modality relation set. Stable mergesort followed by balanced `array_split` matches the original P0.1 quantile-figure rule; exact ties crossing boundaries are broken by artifact row order.
- uncertainty intervals are 1,000 target-node bootstrap replicates (seed 42); every selected target carries all of its sampled relations. Three-seed means ± population SD are reported separately.

## 2. Why This Supplement Is Needed

A conditional beneficial rate alone establishes only that beneficial relations exist in a similarity region. It does not establish enrichment: if the overall beneficial base rate is higher, the region is depleted relative to the population. This audit therefore reports `P(U > 0 | Qq) - P(U > 0)` (lift), and analogously for harmful utility, beside both conditional and overall rates.

## 3. Overall Utility Base Rates

Mean ± population SD over seeds 42, 43, and 44. CE is primary; margin is included as a robustness view. Similarity space does not affect these overall rates, so each dataset × modality appears once.

| Dataset | Modality | CE beneficial | CE harmful | CE zero | Margin beneficial | Margin harmful | Margin zero |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.655 ± 0.006 | 0.345 ± 0.006 | 0.000011 ± 0.000015 | 0.541 ± 0.013 | 0.459 ± 0.013 | 0.000011 ± 0.000015 |
| Movies | Visual | 0.662 ± 0.006 | 0.338 ± 0.006 | 0.000011 ± 0.000015 | 0.567 ± 0.009 | 0.433 ± 0.009 | 0.000000 ± 0.000000 |
| Toys | Text | 0.806 ± 0.010 | 0.193 ± 0.010 | 0.000015 ± 0.000021 | 0.731 ± 0.014 | 0.269 ± 0.014 | 0.000000 ± 0.000000 |
| Toys | Visual | 0.806 ± 0.003 | 0.194 ± 0.003 | 0.000000 ± 0.000000 | 0.740 ± 0.005 | 0.260 ± 0.005 | 0.000000 ± 0.000000 |
| Grocery | Text | 0.800 ± 0.004 | 0.200 ± 0.004 | 0.000026 ± 0.000018 | 0.753 ± 0.006 | 0.247 ± 0.006 | 0.000000 ± 0.000000 |
| Grocery | Visual | 0.786 ± 0.003 | 0.214 ± 0.002 | 0.000026 ± 0.000018 | 0.741 ± 0.002 | 0.259 ± 0.002 | 0.000000 ± 0.000000 |
| ele-fashion | Text | 0.825 ± 0.010 | 0.175 ± 0.010 | 0.000027 ± 0.000038 | 0.799 ± 0.011 | 0.201 ± 0.011 | 0.000000 ± 0.000000 |
| ele-fashion | Visual | 0.684 ± 0.074 | 0.316 ± 0.074 | 0.000098 ± 0.000025 | 0.591 ± 0.133 | 0.409 ± 0.133 | 0.000009 ± 0.000013 |
| Reddit-S | Text | 0.937 ± 0.007 | 0.063 ± 0.007 | 0.000227 ± 0.000144 | 0.844 ± 0.023 | 0.156 ± 0.023 | 0.000093 ± 0.000131 |
| Reddit-S | Visual | 0.943 ± 0.004 | 0.057 ± 0.004 | 0.000025 ± 0.000036 | 0.932 ± 0.001 | 0.068 ± 0.001 | 0.000000 ± 0.000000 |

## 4. Probe-Similarity Quintile Utility Profiles

Cells are mean ± population SD across seeds; rates use CE utility. Each table reports the conditional beneficial and harmful rates within that quintile.

### Movies

| Modality | Outcome | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| Text | beneficial | 0.589 ± 0.009 | 0.612 ± 0.003 | 0.633 ± 0.017 | 0.679 ± 0.007 | 0.762 ± 0.006 |
| Text | harmful | 0.411 ± 0.009 | 0.388 ± 0.003 | 0.367 ± 0.017 | 0.321 ± 0.007 | 0.238 ± 0.006 |
| Visual | beneficial | 0.572 ± 0.006 | 0.624 ± 0.004 | 0.666 ± 0.007 | 0.695 ± 0.006 | 0.755 ± 0.011 |
| Visual | harmful | 0.428 ± 0.006 | 0.376 ± 0.004 | 0.334 ± 0.006 | 0.305 ± 0.006 | 0.245 ± 0.011 |

### Toys

| Modality | Outcome | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| Text | beneficial | 0.652 ± 0.013 | 0.800 ± 0.016 | 0.833 ± 0.007 | 0.859 ± 0.008 | 0.888 ± 0.014 |
| Text | harmful | 0.348 ± 0.013 | 0.200 ± 0.016 | 0.167 ± 0.007 | 0.141 ± 0.008 | 0.112 ± 0.014 |
| Visual | beneficial | 0.664 ± 0.005 | 0.766 ± 0.011 | 0.816 ± 0.003 | 0.868 ± 0.007 | 0.913 ± 0.005 |
| Visual | harmful | 0.336 ± 0.005 | 0.234 ± 0.011 | 0.184 ± 0.003 | 0.132 ± 0.007 | 0.087 ± 0.005 |

### Grocery

| Modality | Outcome | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| Text | beneficial | 0.559 ± 0.005 | 0.769 ± 0.003 | 0.858 ± 0.006 | 0.895 ± 0.006 | 0.922 ± 0.004 |
| Text | harmful | 0.441 ± 0.005 | 0.231 ± 0.003 | 0.142 ± 0.006 | 0.105 ± 0.006 | 0.078 ± 0.004 |
| Visual | beneficial | 0.556 ± 0.004 | 0.725 ± 0.007 | 0.822 ± 0.004 | 0.896 ± 0.003 | 0.932 ± 0.005 |
| Visual | harmful | 0.444 ± 0.004 | 0.275 ± 0.007 | 0.178 ± 0.004 | 0.104 ± 0.003 | 0.068 ± 0.005 |

### ele-fashion

| Modality | Outcome | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| Text | beneficial | 0.569 ± 0.028 | 0.832 ± 0.019 | 0.892 ± 0.011 | 0.916 ± 0.007 | 0.916 ± 0.008 |
| Text | harmful | 0.431 ± 0.028 | 0.168 ± 0.019 | 0.108 ± 0.011 | 0.084 ± 0.007 | 0.084 ± 0.008 |
| Visual | beneficial | 0.546 ± 0.073 | 0.677 ± 0.084 | 0.726 ± 0.078 | 0.737 ± 0.075 | 0.733 ± 0.082 |
| Visual | harmful | 0.454 ± 0.073 | 0.323 ± 0.084 | 0.274 ± 0.078 | 0.263 ± 0.075 | 0.266 ± 0.082 |

### Reddit-S

| Modality | Outcome | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| Text | beneficial | 0.823 ± 0.014 | 0.923 ± 0.013 | 0.965 ± 0.006 | 0.984 ± 0.005 | 0.990 ± 0.001 |
| Text | harmful | 0.177 ± 0.013 | 0.076 ± 0.013 | 0.034 ± 0.006 | 0.016 ± 0.005 | 0.010 ± 0.001 |
| Visual | beneficial | 0.769 ± 0.013 | 0.991 ± 0.002 | 0.997 ± 0.001 | 0.999 ± 0.001 | 0.962 ± 0.008 |
| Visual | harmful | 0.231 ± 0.013 | 0.009 ± 0.002 | 0.003 ± 0.001 | 0.001 ± 0.001 | 0.038 ± 0.008 |

## 5. Enrichment Relative to Base Rate

Lift is a conditional rate minus its overall base rate. Positive lift is enrichment; negative lift is depletion. Values are mean ± population SD over the three seeds. Run-specific target-node bootstrap intervals remain separate in the per-seed CSV.

| Dataset | Modality | Low-sim beneficial lift (Q1) | High-sim harmful lift (Q5) | High-sim beneficial lift (Q5) | Low-sim harmful lift (Q1) |
|---|---|---:|---:|---:|---:|
| Movies | Text | -0.066 ± 0.008 | -0.107 ± 0.002 | 0.107 ± 0.002 | 0.066 ± 0.008 |
| Movies | Visual | -0.090 ± 0.004 | -0.092 ± 0.005 | 0.092 ± 0.005 | 0.090 ± 0.004 |
| Toys | Text | -0.154 ± 0.007 | -0.082 ± 0.011 | 0.082 ± 0.011 | 0.154 ± 0.007 |
| Toys | Visual | -0.141 ± 0.007 | -0.107 ± 0.004 | 0.107 ± 0.004 | 0.141 ± 0.007 |
| Grocery | Text | -0.242 ± 0.005 | -0.122 ± 0.003 | 0.122 ± 0.003 | 0.242 ± 0.005 |
| Grocery | Visual | -0.230 ± 0.003 | -0.145 ± 0.005 | 0.145 ± 0.005 | 0.230 ± 0.003 |
| ele-fashion | Text | -0.256 ± 0.018 | -0.091 ± 0.013 | 0.091 ± 0.013 | 0.256 ± 0.018 |
| ele-fashion | Visual | -0.138 ± 0.035 | -0.050 ± 0.014 | 0.050 ± 0.014 | 0.138 ± 0.035 |
| Reddit-S | Text | -0.114 ± 0.006 | -0.053 ± 0.007 | 0.053 ± 0.007 | 0.114 ± 0.006 |
| Reddit-S | Visual | -0.174 ± 0.009 | -0.018 ± 0.004 | 0.018 ± 0.004 | 0.174 ± 0.009 |

## 6. Similarity Stratification Strength

Range is `max_q P(sign | Qq) - min_q P(sign | Qq)`; quantile Spearman is descriptive over five ordered bins and is not tested for significance. The report uses probe similarity; raw values are available in the cross-dataset CSV.

| Dataset | Modality | Beneficial range | Harmful range | Beneficial quantile Spearman | Harmful quantile Spearman |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.173 ± 0.010 | 0.173 ± 0.010 | 1.000 ± 0.000 | -1.000 ± 0.000 |
| Movies | Visual | 0.182 ± 0.008 | 0.182 ± 0.008 | 1.000 ± 0.000 | -1.000 ± 0.000 |
| Toys | Text | 0.236 ± 0.018 | 0.236 ± 0.018 | 1.000 ± 0.000 | -1.000 ± 0.000 |
| Toys | Visual | 0.249 ± 0.006 | 0.249 ± 0.006 | 1.000 ± 0.000 | -1.000 ± 0.000 |
| Grocery | Text | 0.363 ± 0.007 | 0.363 ± 0.007 | 1.000 ± 0.000 | -1.000 ± 0.000 |
| Grocery | Visual | 0.376 ± 0.003 | 0.376 ± 0.003 | 1.000 ± 0.000 | -1.000 ± 0.000 |
| ele-fashion | Text | 0.348 ± 0.031 | 0.348 ± 0.031 | 0.967 ± 0.047 | -0.967 ± 0.047 |
| ele-fashion | Visual | 0.200 ± 0.054 | 0.200 ± 0.054 | 0.800 ± 0.141 | -0.800 ± 0.141 |
| Reddit-S | Text | 0.167 ± 0.013 | 0.167 ± 0.013 | 1.000 ± 0.000 | -1.000 ± 0.000 |
| Reddit-S | Visual | 0.230 ± 0.013 | 0.229 ± 0.013 | 0.400 ± 0.000 | -0.400 ± 0.000 |

## 7. Raw vs Task-Aware Similarity

Both similarity spaces are analyzed separately. Probe similarity has a larger mean beneficial-rate range than raw similarity in all 10 dataset × modality pairs. Probe Q1 beneficial lift is negative in all 10 pairs; raw Q1 lift is positive only for ele-fashion Visual and negative elsewhere. Probe Q5 harmful lift is negative in all 10 pairs; raw Q5 harmful lift is positive for both ele-fashion modalities and negative in the other eight pairs. The table below gives mean ± population SD over seeds; raw/probe differences do not force consistency.

| Dataset | Modality | Raw Q1 beneficial lift | Probe Q1 beneficial lift | Raw Q5 harmful lift | Probe Q5 harmful lift | Raw sign range | Probe sign range | Raw ρQ beneficial | Probe ρQ beneficial |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | Text | -0.011 ± 0.005 | -0.066 ± 0.008 | -0.028 ± 0.011 | -0.107 ± 0.002 | 0.058 ± 0.014 | 0.173 ± 0.010 | 0.833 ± 0.047 | 1.000 ± 0.000 |
| Movies | Visual | -0.027 ± 0.002 | -0.090 ± 0.004 | -0.039 ± 0.003 | -0.092 ± 0.005 | 0.066 ± 0.005 | 0.182 ± 0.008 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Toys | Text | -0.010 ± 0.006 | -0.154 ± 0.007 | -0.029 ± 0.006 | -0.082 ± 0.011 | 0.042 ± 0.009 | 0.236 ± 0.018 | 0.700 ± 0.216 | 1.000 ± 0.000 |
| Toys | Visual | -0.037 ± 0.005 | -0.141 ± 0.007 | -0.059 ± 0.004 | -0.107 ± 0.004 | 0.096 ± 0.009 | 0.249 ± 0.006 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Grocery | Text | -0.029 ± 0.002 | -0.242 ± 0.005 | -0.052 ± 0.003 | -0.122 ± 0.003 | 0.082 ± 0.004 | 0.363 ± 0.007 | 0.900 ± 0.000 | 1.000 ± 0.000 |
| Grocery | Visual | -0.058 ± 0.003 | -0.230 ± 0.003 | -0.095 ± 0.003 | -0.145 ± 0.005 | 0.154 ± 0.005 | 0.376 ± 0.003 | 0.967 ± 0.047 | 1.000 ± 0.000 |
| ele-fashion | Text | -0.140 ± 0.009 | -0.256 ± 0.018 | 0.008 ± 0.005 | -0.091 ± 0.013 | 0.199 ± 0.015 | 0.348 ± 0.031 | 0.300 ± 0.000 | 0.967 ± 0.047 |
| ele-fashion | Visual | 0.013 ± 0.043 | -0.138 ± 0.035 | 0.033 ± 0.024 | -0.050 ± 0.014 | 0.081 ± 0.049 | 0.200 ± 0.054 | -0.200 ± 0.535 | 0.800 ± 0.141 |
| Reddit-S | Text | -0.029 ± 0.001 | -0.114 ± 0.006 | -0.035 ± 0.005 | -0.053 ± 0.007 | 0.064 ± 0.006 | 0.167 ± 0.013 | 0.967 ± 0.047 | 1.000 ± 0.000 |
| Reddit-S | Visual | -0.106 ± 0.013 | -0.174 ± 0.009 | -0.014 ± 0.003 | -0.018 ± 0.004 | 0.152 ± 0.017 | 0.230 ± 0.013 | 0.600 ± 0.141 | 0.400 ± 0.000 |

## 8. Dataset-Specific Interpretation

### Movies
- Text: probe Q1 beneficial `0.589` vs overall beneficial base `0.655` (lift `-0.066`); Q5 harmful `0.238` vs overall harmful base `0.345` (lift `-0.107`). Beneficial/harmful ranges are `0.173` / `0.173`, with quantile Spearman `1.000` / `-1.000`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.
- Visual: probe Q1 beneficial `0.572` vs overall beneficial base `0.662` (lift `-0.090`); Q5 harmful `0.245` vs overall harmful base `0.338` (lift `-0.092`). Beneficial/harmful ranges are `0.182` / `0.182`, with quantile Spearman `1.000` / `-1.000`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.

### Toys
- Text: probe Q1 beneficial `0.652` vs overall beneficial base `0.806` (lift `-0.154`); Q5 harmful `0.112` vs overall harmful base `0.193` (lift `-0.082`). Beneficial/harmful ranges are `0.236` / `0.236`, with quantile Spearman `1.000` / `-1.000`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.
- Visual: probe Q1 beneficial `0.664` vs overall beneficial base `0.806` (lift `-0.141`); Q5 harmful `0.087` vs overall harmful base `0.194` (lift `-0.107`). Beneficial/harmful ranges are `0.249` / `0.249`, with quantile Spearman `1.000` / `-1.000`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.

### Grocery
- Text: probe Q1 beneficial `0.559` vs overall beneficial base `0.800` (lift `-0.242`); Q5 harmful `0.078` vs overall harmful base `0.200` (lift `-0.122`). Beneficial/harmful ranges are `0.363` / `0.363`, with quantile Spearman `1.000` / `-1.000`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.
- Visual: probe Q1 beneficial `0.556` vs overall beneficial base `0.786` (lift `-0.230`); Q5 harmful `0.068` vs overall harmful base `0.214` (lift `-0.145`). Beneficial/harmful ranges are `0.376` / `0.376`, with quantile Spearman `1.000` / `-1.000`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.

### ele-fashion
- Text: probe Q1 beneficial `0.569` vs overall beneficial base `0.825` (lift `-0.256`); Q5 harmful `0.084` vs overall harmful base `0.175` (lift `-0.091`). Beneficial/harmful ranges are `0.348` / `0.348`, with quantile Spearman `0.967` / `-0.967`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.
- Visual: probe Q1 beneficial `0.546` vs overall beneficial base `0.684` (lift `-0.138`); Q5 harmful `0.266` vs overall harmful base `0.316` (lift `-0.050`). Beneficial/harmful ranges are `0.200` / `0.200`, with quantile Spearman `0.800` / `-0.800`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.
- Seed-spread note: Text/Visual CE beneficial base rates are `0.825 ± 0.010` / `0.684 ± 0.074`, and Q1 conditional rates are `0.569 ± 0.028` / `0.546 ± 0.073`. Visual seed SD is notably wider; all three seed rows and their node-bootstrap intervals remain separate in the per-seed CSV.

### Reddit-S
- Text: probe Q1 beneficial `0.823` vs overall beneficial base `0.937` (lift `-0.114`); Q5 harmful `0.010` vs overall harmful base `0.063` (lift `-0.053`). Beneficial/harmful ranges are `0.167` / `0.167`, with quantile Spearman `1.000` / `-1.000`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds.
- Visual: probe Q1 beneficial `0.769` vs overall beneficial base `0.943` (lift `-0.174`); Q5 harmful `0.038` vs overall harmful base `0.057` (lift `-0.018`). Beneficial/harmful ranges are `0.230` / `0.229`, with quantile Spearman `0.400` / `-0.400`. Node-bootstrap Q1 beneficial lift: bootstrap 95% CI positive/negative/spans 0: 0/3/0 seeds; Q5 harmful lift: bootstrap 95% CI positive/negative/spans 0: 0/1/2 seeds.
- Base-rate check: Q1 beneficial remains high (`0.823 ± 0.014` Text; `0.769 ± 0.013` Visual), but the overall beneficial base is even higher (`0.937 ± 0.007`; `0.943 ± 0.004`). Thus the high Q1 conditional proportions reflect a high population-wide beneficial rate; Q1 is depleted relative to that base in both modalities.

## 9. Revised Interpretation of P0.1

1. **Does low-sim beneficial existence remain true?** Yes: `P(U > 0 | Q1) > 0` in 30/30 probe-similarity dataset × seed × modality rows. This is counterexample existence in the fixed sampled population.
2. **Is low-sim beneficial enriched or depleted?** Across 10 dataset × modality mean lifts, Q1 beneficial lift is positive in 0 and negative in 10; all-three-seed node-bootstrap CIs are negative in 10/10 comparisons (positive in 0/10). Interpret each run-specific interval in the CSV.
3. **Are high-sim harmful relations enriched?** Q5 harmful rate is nonzero in 30/30 probe dataset × seed × modality rows, so harmful counterexamples exist. Its lift is positive in 0 and negative in 10 of 10 dataset × modality mean comparisons; all-three-seed node-bootstrap CIs are negative in 9/10, while 1/10 have at least one interval spanning zero (positive in 0/10). A nonzero Q5 harmful rate by itself is not enrichment.
4. **Where is similarity informative?** Use the sign-rate ranges and five-bin monotonic profiles in Section 6. These are descriptive properties, not significance claims.
5. **Where is similarity insufficient?** Probe similarity stratifies CE utility sign in every dataset and modality, but strength differs: Grocery is strongest, Movies weakest, and Reddit-S Visual is less monotone. This sign result does not establish that similarity predicts the magnitude of utility.

**Updated P0.1 reading.** Counterexample existence is upheld: beneficial edges occur in every Q1 and harmful edges occur in every Q5. The enrichment interpretation changes: Q1-beneficial and Q5-harmful are depleted relative to overall base rates in every dataset × modality mean. Conversely, high-sim beneficial and low-sim harmful are enriched. The earlier weak continuous Spearman findings therefore do not imply weak utility-sign stratification; the five-bin sign profiles are strongly ordered in most dataset × modality pairs.

### Requested dataset regime labels (descriptive only)

These relative labels use observed probe-CE profiles across the five datasets. `Strong` is assigned only to a dataset with the largest non-weak mean sign-rate range composite among datasets whose beneficial-quantile Spearman direction is shared by both modalities and all three seeds. `Weak` is the dataset with the smallest mean composite range. Remaining datasets are `intermediate`. If no dataset meets the consistency condition for `Strong`, none is labeled strong. No threshold test or significance claim is implied.

- Movies: **weak similarity regime** (mean probe sign-rate range composite `0.178`; monotonic direction shared across modality × seed: `true`).
- Toys: **intermediate similarity regime** (mean probe sign-rate range composite `0.243`; monotonic direction shared across modality × seed: `true`).
- Grocery: **strong similarity-sign regime** (mean probe sign-rate range composite `0.370`; monotonic direction shared across modality × seed: `true`).
- ele-fashion: **intermediate similarity regime** (mean probe sign-rate range composite `0.274`; monotonic direction shared across modality × seed: `true`).
- Reddit-S: **intermediate similarity regime** (mean probe sign-rate range composite `0.198`; monotonic direction shared across modality × seed: `true`).

## 10. Implications for P0.2

A later P0.2 should treat a scalar similarity-reliability baseline as a serious comparator: probe-similarity utility-sign stratification is strongest in Grocery, weaker but monotone in Movies, and non-monotone for Reddit-S Visual; ele-fashion Visual also has wider seed spread. Compare the scalar baseline and a relation-conditioned alternative on a predeclared high-, low-, and boundary-separation set, and evaluate continuous CE utility as well as sign rates. Freeze controls and selection rules first. No P0.2 was implemented or run here.

## 11. What This Still Does Not Prove

This audit does not prove that a scalar learned relation is insufficient; vector relation state is necessary; semantic transformation is necessary; MoE, basis routing, or FiLM is necessary; or context heterogeneity is established. It reports only how the existing sampled validation relation utilities are distributed across observed similarity quintiles.

## Figures

- [p01plus_Movies_probe_quintile_utility.png](../../results/problem_validation/p01plus/plots/p01plus_Movies_probe_quintile_utility.png)
- [p01plus_Toys_probe_quintile_utility.png](../../results/problem_validation/p01plus/plots/p01plus_Toys_probe_quintile_utility.png)
- [p01plus_Grocery_probe_quintile_utility.png](../../results/problem_validation/p01plus/plots/p01plus_Grocery_probe_quintile_utility.png)
- [p01plus_ele-fashion_probe_quintile_utility.png](../../results/problem_validation/p01plus/plots/p01plus_ele-fashion_probe_quintile_utility.png)
- [p01plus_Reddit-S_probe_quintile_utility.png](../../results/problem_validation/p01plus/plots/p01plus_Reddit-S_probe_quintile_utility.png)

## Result Files

- `results/problem_validation/p01plus/p01plus_per_seed.csv`: all dataset × seed × modality × similarity-space rows; includes CE and margin profiles and run-specific bootstrap intervals.
- `results/problem_validation/p01plus/p01plus_cross_dataset.csv`: mean ± population SD across seeds.
- Node-bootstrap intervals are per seed and are not combined with three-seed SD.
