# Protocol and evidence audit

本文件是实现者对已锁定协议和产物的自审记录，不是独立审查。协议源为 `docs/collab_adapt_nc_v2/protocol.md` 与 `protocol_lock.json`。

- Result: PASS
- Training source commit: `524ccfd0586bc44f99d9ad3bfabbf81a7df0f83c`
- Protocol lock SHA-256: `805d81b8f362ccdc23f863d9cf8d17edf4cf2a96ee27635b30b3ddd9f4c2d2c9`
- Formal successful attempts: 216; reconstructed train/validation head-partitions: 1728; confusion matrices: 1728.
- Scope audit: test forward=0, test metrics=0, real LP=0; only the registered three NC datasets are included.

| Check | Result |
|---|---|
| all formal runs declare test evaluation disabled and only train/validation evaluated | PASS |
| no test head forward or test metrics recorded | PASS |
| no real LP run | PASS |
| R and S calibration/confirmation cells and selection bindings audited | PASS |
| train/validation predictions rebuild fixed-schema metrics and confusion matrices | PASS |
| historical v1 evidence remains unchanged and raw artifacts stay ignored | PASS |
| source, input, split, checkpoint, prediction and log hashes verified | PASS |

## Findings

- 未发现协议或证据一致性问题。

## Accuracy serialization precision

The original locked audit used 1e-10 for comparing float32-serialized accuracy against a float64 argmax reconstruction. The post-run audit wrapper records and allows 1e-7 rounding for accuracy only; correct counts, confusion matrices, and Macro-F1 are checked exactly/recomputed. This audit-only adjustment does not alter the locked training source or protocol. See `audit_precision_correction.json`.
