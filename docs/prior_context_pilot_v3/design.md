# Prior-context MAG pilot v3: design record

Date: 2026-10-05. This is a frozen development study on Movies and Grocery train/validation splits. It defines an experiment; no benefit is assumed.

## Motivation and claims boundary

v2 already supplied unpropagated modality features to its final fusion MLP. v3 changes the interface by adding a coefficient-one center representation to the final node embedding and by supervising unpropagated modality adapters. This does not guarantee lossless preservation: the adapter compresses the frozen features to 128 dimensions and the learned structural correction can oppose the center vector.

SUPRA (arXiv:2605.24684v1) motivates separating modality priors from graph processing. Its reported classifier averages branch logits; this pilot keeps a shared 128-dimensional node embedding interface for later NC/LP use and includes a separate SUPRA-style NC reference. RoleMAG (arXiv:2604.12271v1) motivates keeping the center path outside a competition among structural experts. This pilot does not claim an author-code reproduction; the RoleMAG equations were checked, while author-code provenance was not.

The hypothesis is that joint, text-origin, and image-origin states may require different graph-structure responses, and that node-local cross-source and graph context may help choose between those responses. J/T/V mean joint/text-origin/image-origin processing paths. S/D mean transformed-neighbor and transformed-edge-difference operators. They are not identifiable shared/unique information or ground-truth homophily roles.

## Model family

The full equations, axis definitions, empty-neighborhood behavior, shared-parameter rules, initialization, output dimensions, and exact R0–R6 controls are frozen in [protocol.md](protocol.md). R4 is the candidate. R0 is topology-free; R1 is a wide ordinary-mean graph control; R2 is retrained static S/D mixing; R3 replaces per-node router context with its unlabeled graph mean; R5 shares S/D functions across J/T/V; R6 is a SUPRA-style NC reference.

All methods use the same frozen feature files, official seed-42 split, AdamW, 0.2 dropout, source-head loss weight 0.1, optimizer LR across backbone and task heads, and train/validation-only selection. R6's internal DGL normalized GCN self-loops apply only inside R6.

## Stage rule

Stage A trains R0/R1/R4 on Movies/Grocery: 18 seed-52 learning-rate calibration jobs and 12 seed-53/54 confirmation jobs. Stage B adds R2/R3/R5/R6 for 40 jobs only when R4 passes the frozen paired validation and resource gate against both R0 and R1. If the gate fails, B is recorded as skipped and this pilot ends without tuning the rule.

The pilot does not inspect test labels or compute test predictions/metrics. It runs no LP and does not expand to the other five NC datasets. Three seeds are descriptive. Movies and Grocery are previously used development graphs, so results carry selection/exposure limitations.
