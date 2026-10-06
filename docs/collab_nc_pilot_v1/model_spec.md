# Model specification and implementation notes

The normative tensor and comparison specification is [`protocol.md`](protocol.md). The implementation lives in `src/models/mag_collab_nc.py` with the task-only heads and train-index loss in `src/tasks/collab_nc_pilot.py`; the full-graph campaign entry is `scripts/run_collab_nc_pilot_v1.py`.

## Architecture boundary

`MAGCollabNC` consumes `[x_text,x_image]`, the loader's `edge_index`, and a two-column modality availability mask. It does not accept labels, number of classes, split indices, or LP targets. It returns the framework's five-tuple and supplies task path states, path dimensions, availability, and unweighted SSL terms through `extras`. The adapter owns all NC class heads and selects node indices before reading labels. The output dimension for B0/B1/M0/M1 is 640; legacy B2 remains its R1 128-D output.

Implementation uses destination-indexed scatter reductions for edge softmax and aggregation, so the normalized dimensions are explicit: private κ over valid incoming neighbors per receiver/head; private NULL over those same neighbors plus one receiver NULL; common π over observed sender modalities per physical edge/receiver-modality/head; common α over the full receiver neighborhood plus one NULL per receiver-modality/head. No dense N×N matrix is built. Message states update synchronously.

## Deliberate interpretation limits

`private`, `common`, `same-modality`, and `cross-modality` name computational routes only. No node or neighbor receives a semantic role label. There is no raw text-image cosine. Learned p/e projections are not lossless raw-feature skips. The first-layer private state is supplied as a diagnostic source state because it reads only its own modality; final private/common states are what the three path heads supervise. No claim of causal utility, information share, or factor identification follows from path metrics, router values, or norms.

Only exact full-graph NC execution is implemented. No sampled/layerwise claim is made. The design has no top-k routing, OT, graph rewiring, extra contrastive objective, signed filter, or additional task loss.

