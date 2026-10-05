# Problem validation v1.1 provenance

V1.1 is a validation-diagnostic repair against the completed v1 checkpoints. The v1 main matrix and its evaluation lock remain unchanged.

| Item | Frozen identity |
|---|---|
| V1 training source | `63867331e9d3373f19136e47ab192cc9fb014c97` |
| V1 final report/base commit | `7a7ba1aaefb1cfff5132bd2a53d10657344cd237` |
| V1 evaluation protocol lock SHA-256 | `3b949712c738dd99865f9bdce882cf65f91756e775cd4af6077cbd065d3b368f` |
| V1 raw evidence | `/hdd1/DataInHere/YHF/mag_model/outputs/problem_validation_v1/` |
| V1 reports and registry | `/hdd1/DataInHere/YHF/mag_model/reports/problem_validation_v1/` |
| New diagnostic | `scripts/diagnose_frozen_probe_v1_1.py` |
| New outputs | `reports/problem_validation_v1_1/` |

Before any inference, the diagnostic verifies the lock digest, registry-to-lock primary cells, the complete 475-file raw-artifact manifest, all 21 unique data/split inputs, all 30 selected checkpoints, and all 30 saved validation-argmax files. It records every selected checkpoint, input, split, and validation-reference path and SHA-256 in `provenance.json`. The old v1 evaluator source hash is recorded separately from the v1.1 diagnostic hash; the diagnostic does not modify or re-sign the old lock.

Training logs identify the original checkpoint devices as `cuda:0` or `cuda:1`. The default CLI follows the corresponding device for each checkpoint so validation reconstruction uses the original runtime device. The diagnostic uses the class order and Macro-F1 label set from the old lock; it does not derive a new schema from test labels or use test metrics.

Reproduction command:

```bash
conda run -n yhf_env python scripts/diagnose_frozen_probe_v1_1.py \
  --evidence-root /hdd1/DataInHere/YHF/mag_model
```

The generated `manifest.json` lists byte counts and hashes for the report, diagnostics JSON, provenance JSON, and review record. Its `git_head_at_generation` is the code base from which the diagnostic ran; the final v1.1 commit SHA is reported by Git after commit and cannot be self-referenced by that same commit's manifest.
