# Model specification and implementation review points

The implementation is [`src/models/mag_collab_adapt.py`](../../src/models/mag_collab_adapt.py); it exports `Model(cfg,data_info)` for the standard dynamic factory. The task adapter remains `CollabTaskHeads`/`train_only_loss` from the previous pilot; the new encoder receives no class count or label tensor. S0/S1 use `base_variant=B1`; S2-S5 use `base_variant=M0`; experiment IDs remain separate from the base variant passed into path/SSL task logic.

## Tensor dimensions and flow

- Input `[x_text,x_image]` uses actual loader widths; no 768-D input assumption.
- p-text and p-image each map their own input width to 128, with their own LayerNorm. Center `e` is the existing `[p_text,p_image]→128→128` normalized MLP.
- c and s states are independently initialized at 64 per modality; the common stem remains shared and private stems remain modality-specific.
- M0 has two synchronous 64-D private states and two synchronous 64-D common states per modality, each updated from the previous full layer. Layer one is same as inherited; the second layer uses its existing cross-modal choice mechanism.
- The final graph increments are ordered `(Δs_text, Δs_image, Δc_text, Δc_image)`, each 64-D. The p/e block stays in the first 384 coordinates. R appends `βδ` to make 640. S appends `βr` or a fixed-zero 128-D pad to make 768.
- Task outputs are external to the encoder. Path dimensions remain 64/64/128.

## Axes and masks

For input edge `src=j,dst=i`, private κ uses incoming-edge group `dst` per receiver/head. Common π's last axis is sender modality `m` for each edge, receiving modality `n`, and head; it sums to one when at least one sender is available and is zero for no available sender. Common α's group axis is all incoming physical edges grouped by `dst` for one receiver modality/head; it shares a denominator with exactly one node-local NULL. V content transforms the sender modality's projected value according to `(m,n)` before π then α combine it. Source masks are read from the original availability tensor and are not inferred from graph messages.

Empty edges return zero graph increments. Edges from nodes with no observed sender produce zero π/value contribution and retain receiver NULL. Isolated nodes never produce NaN. No cross-space raw cosine is computed: two raw cosine channels remain separately within text and within image.

## Initialization and output compatibility

All shared trainable layers use deterministic name-derived CPU Xavier seeds. After that generic initialization, gate/NULL values are explicitly restored and identity/gamma parameters remain identity/one. Model construction and checkpoint loading are separate; loading a non-initial checkpoint does not call initialization. The V-on output is compared against V-off using the same seeds and identity M; V adds exactly 16,384 parameters across two layers. F's zero-delta output is exactly zero, even when its learned anchor bias is changed. R's output/state shapes and task heads remain 640-D. S main heads all consume 768-D; F-off padding is constant zero and its final 2,560 main-head weights are ineffective.

## Computation and cache lifecycle

`optimize_cache=False` is the in-module reference path: raw cosines are recomputed from raw modalities and relationship matrices are applied in edge space. The optimized path computes same-space unit-normalized raw cosine in bounded edge chunks, plus availability and valid degrees once per exact input/edge/mask object. It retains object references, checks mutation version and device before reuse, and recreates on changes. Inference tensors that do not expose a version counter are conservatively not cached. Inputs requiring gradients bypass the cache to preserve gradients. No trainable parameter state is cached. Key/value projection is per node/sender; relation transform is per node/role before edge gather. The complete receiver group remains in each softmax.

`collect_diagnostics=false` is the default. Normal train/validation only returns computational states needed by the task/loss; it avoids edge-stat copies, `torch.isin`, and diagnostic CPU serialization. Scheduled diagnostic forwards run after the measured optimizer step and have separate memory accounting. For fixed steps, update norms use parameters snapshotted only at steps 1/10/30/100; a best step outside those points records update norm as null. Post-clip gradient norm is computed from clipped `.grad` values.

## Tests that target failure modes

`tests/test_collab_adapt.py` checks: a masked invalid logit of 1000 for finite/nonzero valid gradients; NULL and empty cases; restored initialized NULL/gate values and checkpoint load preservation; V-off/V-on first-forward equality and exact parameter increment; F empty-graph and arbitrary-anchor zero boundary plus gradients; grouped/common availability behavior; old-v1 forward/backward equivalence on regular synthetic inputs; cache invalidation on same-sized new features, changed edges, changed masks and in-place mutations; factory construction and endpoint-product LP head interface on synthetic embeddings. No real LP is run.
