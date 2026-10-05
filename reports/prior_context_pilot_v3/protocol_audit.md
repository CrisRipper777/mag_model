# Prior-context MAG pilot v3 protocol audit

Audit result: **PASSED**. This is a supplemental saved-evidence self-audit performed by `scripts/audit_prior_context_pilot_v3_postrun.py`, not an independent external review. The initial frozen-auditor failure is archived under `outputs/prior_context_pilot_v3/audit_attempts/`.

- Recomputed train/validation confusion matrices and Acc/Macro-F1: 30/30 completed result rows.
- Verified manifest raw-file entries: 154/154.
- Test metric/prediction keys and flags: all absent/false.
- Frozen gate recomputation: available; details below and in audit_result.json.
- Completed cells: 30; expected from gate: 30; listed attempts: 30.
- Split/input/protocol/source hashes, fixed class order, LR selection, checkpoint fields, train/validation prediction payloads, and Stage B condition were checked.

## Findings

| R4 comparator | Mean Δ Acc (pp) | Mean Δ Macro-F1 (pp) | Positive Acc/F1 pairs | Per-graph floor | Metric gate |
|---|---:|---:|---:|---|---|
| R0 | +1.058 | -0.093 | 3/4 / 2/4 | fail | fail |
| R1 | -1.349 | -5.340 | 0/4 / 0/4 | fail | fail |

- Resource recomputation: step-time ratio 4.522; allocated-memory ratio 3.441; resource gate fail.
- Exact seed-paired deltas and per-graph means are in `audit_result.json`.

- No mismatch was found in the checks above.
- Only train and validation label indices were read by this audit; the test index and labels were not indexed.
- Cross-entropy is retained from the locked run summaries; argmax-derived confusion matrices, Acc, Macro-F1, LR choices, paired gate values, and gate booleans were independently recomputed.
