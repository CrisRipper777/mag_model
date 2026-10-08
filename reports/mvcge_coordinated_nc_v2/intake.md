# Intake: mvcge_coordinated_nc_v2

- Starting HEAD: `deb15363debd7260234964861446ec0d06938b11`.
- Working branch: `codex/mag-mvcge-coordinated-nc-v2`.
- Existing source branch commit is the v1 experiment parent; no prior v1 outcomes are used in this v2 matrix.
- Data source is the existing MAGB loader; text features precede image features. Features are float32, frozen, finite, and receive no extra normalization. Loader exposes no availability metadata or edge weights.
- The runner retains only train/validation IDs and labels. The generic loader materializes full stored label/split objects; those are dropped before model/task execution. Full graph features and topology remain transductive inputs.
- New R is constructed from the loader's undirected deduplicated edge list, drops loops/duplicates, uses receiver-row means, and keeps isolated rows zero.
- No test evaluator, real LP task, or additional dataset is invoked; LP is tested only against a synthetic graph.
- GPU intake snapshot and runtime versions are in `reports/mvcge_coordinated_nc_v2/environment.json`.
- User-owned original workspace changes preserved and excluded from stage commits: ` D CLAUDE.md`, `?? docs/V3_design.md`, `?? docs/mechanism_pilot_v2_design.md`, `?? docs/plan.md`, `?? reports/collab_adapt_nc_v2/s_calibration_selection.json`

| Dataset | Nodes | Loader edge rows | Text/image dim | Classes | Train/val/test sizes* | Raw graph |
|---|---:|---:|---:|---:|---:|---|
| Movies | 16,672 | 160,802 | 768/768 | 20 | 10,003/3,334/3,335 | 109,195 raw directed rows; reverse coverage 0.5274; 0 loops |
| Grocery | 17,074 | 142,262 | 768/768 | 20 | 10,244/3,415/3,415 | 85,670 raw directed rows; reverse coverage 0.3394; 1 loops |
| Reddit-S | 15,894 | 283,080 | 768/768 | 20 | 9,536/3,179/3,179 | 283,080 raw directed rows; reverse coverage 1.0000; 0 loops |

*Test size is metadata only; test IDs and labels are not copied into the task runner. Input and split file hashes are recorded in `data_profile.json`.
