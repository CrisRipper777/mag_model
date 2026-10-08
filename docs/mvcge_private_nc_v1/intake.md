# Intake: MVCGE private NC v1

- Actual starting HEAD: `0e79a5a81cbe414016d7782c3e6e046e1ec2b920`.
- Task branch: `codex/mag-mvcge-private-nc-v1`.
- No applicable `AGENTS.md` was found under the project or its parent search path. `README.md` and the active Hydra model/task/dataset configs were read.
- Historical S0 is the `S0` row in `docs/collab_adapt_nc_v2/protocol.md`: `MAGCollabAdapt` base `B1`, 768-dimensional padded readout, plus fixed path/reconstruction/pair losses. Implementation: `src/models/mag_collab_adapt.py`; training/task adapter: `scripts/run_collab_adapt_nc_v2.py`, `src/tasks/collab_nc_pilot.py`.
- S0 selected LR/WD was read from `reports/collab_adapt_nc_v2/s_selection_lock.json`: Movies 0.003/0.001, Grocery 0.001/0.001, Reddit-S 0.001/0.001. All three are included in the requested calibration grid; no extra history-only cell is needed.
- Canonical ingestion: `src/data/loaders.py` reads DGL edge indices and frozen `.npy` features, casts features to float32, joins text then image, uses seed-42 NC split, and calls `preprocess_edge_index` (`remove_self_loops`, `to_undirected`, configured self loops false). This loader does not expose edge weights or modality-availability metadata.
- New graph operators are built from exactly the loader edge list, after excluding loops and adding one normalized self-loop. No extra reverse-edge insertion or feature transformation is done by the runner.
- The selected MAGB split sizes and input hashes are recorded in `reports/mvcge_private_nc_v1/data_profile.json`; test split size is read only as metadata and test IDs/labels are not copied into the task runner.
- The standard NC runner is `src/tasks/nc.py`; its generic loop computes test metrics, so this study uses a separate NC-only runner that calls the existing loader and legacy S0 modules but evaluates only explicitly supplied train and validation IDs. The existing LP endpoint-product decoder is `src/models/predictor.py`; it is used only in a synthetic interface test.
- User-owned working-tree state observed before implementation: deleted `CLAUDE.md`; untracked `docs/V3_design.md`, `docs/mechanism_pilot_v2_design.md`, `docs/plan.md`, and `reports/collab_adapt_nc_v2/s_calibration_selection.json`. These remain unstaged and are excluded from this task commit.

| Dataset | Nodes | Loader edges | Raw graph | Text/image dim | Classes | Train/val/test split size |
|---|---:|---:|---|---:|---:|---:|
| Movies | 16,672 | 160,802 | 109,195 rows, directed/asymmetric, 0 raw self-loops, no edge-data weights | 768/768 | 20 | 10,003/3,334/3,335 |
| Grocery | 17,074 | 142,262 | 85,670 rows, directed/asymmetric, 1 raw self-loops, no edge-data weights | 768/768 | 20 | 10,244/3,415/3,415 |
| Reddit-S | 15,894 | 283,080 | 283,080 rows, symmetric, 0 raw self-loops, no edge-data weights | 768/768 | 20 | 9,536/3,179/3,179 |

All three loaded feature pairs are finite and have zero exactly-all-zero rows. This is a data-quality check only; it is not used to infer availability. The loader has no missing-modality metadata, so the canonical mask is all-present. Features are float32 with no additional L2 normalization, standardization, PCA, or re-extraction. Macro-F1 uses all 20 classes (0–19) with `zero_division=0`.
