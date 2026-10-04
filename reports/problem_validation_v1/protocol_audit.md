# Protocol audit

Source commit before training: `63867331e9d3373f19136e47ab192cc9fb014c97`.

## Code and task protocol

The inherited NC task path uses full-graph training for graph encoders, an external linear head, validation-accuracy checkpoint selection, and optional test evaluation. The probe is an explicit new model/config and does not change legacy defaults. Every probe uses full-graph training, one optimizer step per epoch, train-index cross-entropy, and sets test evaluation off.

The loader uses MAGB `torch.cat([text,image])` and MM-Graph slices `[text,image]`. MAGB graph preprocessing removes self-loops and makes graphs undirected; the new operator independently removes self-loops and duplicate receiver/sender arcs and computes receiver means. The model cache holds topology only.

## Dataset and split checks

### Movies

- Nodes/classes: 16672 / 20 configured; observed supervised Macro-F1 labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`.
- Feature order: `[text,image]`; text `[16672, 768]`, image `[16672, 768]`; nonfinite rows text/image `0/0`; zero rows `0/0`.
- Split sizes train/val/test: `10003/3334/3335`; mutually exclusive and label-range checks passed.
- Explicit split `/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt` SHA-256 `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211`; split generation disabled.
- Raw graph checks: `{'edge_count': 109195, 'out_of_range_edges': 0, 'self_loops': 0, 'duplicate_directed_edges': 0}`; processed graph checks: `{'edge_count': 160802, 'out_of_range_edges': 0, 'self_loops': 0, 'duplicate_directed_edges': 0}`.
- Capacity matching: wide h=192 (0.0385%), concat b=26 (0.0473%).

### Toys

- Nodes/classes: 20695 / 18 configured; observed supervised Macro-F1 labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]`.
- Feature order: `[text,image]`; text `[20695, 768]`, image `[20695, 768]`; nonfinite rows text/image `0/0`; zero rows `0/0`.
- Split sizes train/val/test: `12417/4139/4139`; mutually exclusive and label-range checks passed.
- Explicit split `/hdd1/DataInHere/YHF/data/MAGB_split/Toys_nc_seed42_train0.6_val0.2.pt` SHA-256 `cc0633526581d9281d0a92fa50d989309f8b81b982305409d3034f10a28e41d2`; split generation disabled.
- Raw graph checks: `{'edge_count': 63443, 'out_of_range_edges': 0, 'self_loops': 22, 'duplicate_directed_edges': 0}`; processed graph checks: `{'edge_count': 113402, 'out_of_range_edges': 0, 'self_loops': 0, 'duplicate_directed_edges': 0}`.
- Capacity matching: wide h=192 (0.0385%), concat b=26 (0.0474%).

### Grocery

- Nodes/classes: 17074 / 20 configured; observed supervised Macro-F1 labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`.
- Feature order: `[text,image]`; text `[17074, 768]`, image `[17074, 768]`; nonfinite rows text/image `0/0`; zero rows `0/0`.
- Split sizes train/val/test: `10244/3415/3415`; mutually exclusive and label-range checks passed.
- Explicit split `/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt` SHA-256 `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e`; split generation disabled.
- Raw graph checks: `{'edge_count': 85670, 'out_of_range_edges': 0, 'self_loops': 1, 'duplicate_directed_edges': 0}`; processed graph checks: `{'edge_count': 142262, 'out_of_range_edges': 0, 'self_loops': 0, 'duplicate_directed_edges': 0}`.
- Capacity matching: wide h=192 (0.0385%), concat b=26 (0.0473%).

### Reddit-S

- Nodes/classes: 15894 / 20 configured; observed supervised Macro-F1 labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`.
- Feature order: `[text,image]`; text `[15894, 768]`, image `[15894, 768]`; nonfinite rows text/image `0/0`; zero rows `0/0`.
- Split sizes train/val/test: `9536/3179/3179`; mutually exclusive and label-range checks passed.
- Explicit split `/hdd1/DataInHere/YHF/data/MAGB_split/Reddit-S_nc_seed42_train0.6_val0.2.pt` SHA-256 `2154e82ef061de9d3c018aa19674ea6e2220953f272bedebb1559396e43750a5`; split generation disabled.
- Raw graph checks: `{'edge_count': 283080, 'out_of_range_edges': 0, 'self_loops': 0, 'duplicate_directed_edges': 0}`; processed graph checks: `{'edge_count': 283080, 'out_of_range_edges': 0, 'self_loops': 0, 'duplicate_directed_edges': 0}`.
- Capacity matching: wide h=192 (0.0385%), concat b=26 (0.0473%).

### ele-fashion

- Nodes/classes: 97766 / 12 configured; observed supervised Macro-F1 labels `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]`.
- Feature order: `[text,image]`; text `[97766, 512]`, image `[97766, 512]`; nonfinite rows text/image `0/0`; zero rows `0/0`.
- Split sizes train/val/test: `58659/9777/29330`; mutually exclusive and label-range checks passed.
- Explicit split `/hdd1/DataInHere/YHF/data/ele-fashion/split.pt` SHA-256 `e48e74afd502df5bc5e4d638e611fc4fbfdecee461aa920f3f47a6446826c52f`; split generation disabled.
- Raw graph checks: `{'edge_count': 199602, 'out_of_range_edges': 0, 'self_loops': 16, 'duplicate_directed_edges': 0}`; processed graph checks: `{'edge_count': 399172, 'out_of_range_edges': 0, 'self_loops': 0, 'duplicate_directed_edges': 0}`.
- Capacity matching: wide h=192 (0.0525%), concat b=26 (0.0645%).
- Official split sizes match 58,659/9,777/29,330: `True`.
- Category schema: Configured label schema is 0..11. Current official labels contain the 11 IDs listed in macro_f1_label_set; class ID 5 is absent. Labels and official node mapping are unchanged; no category remapping is performed.
- Official node mapping checks: `{'entries': 97766, 'id_values_unique': True, 'id_values_are_0_to_N_minus_1': True, 'sha256': 'ee6d00b8285a7c8204ee76c808331c85d987e3a04d24d52262e81a71f8e378f3', 'mapping_preserved': True}`.

## Review boundaries

Raw file and processed-edge hashes are in `run_registry.json`. Val/test Macro-F1 uses the observed supervised label set fixed before prediction and always derives counts from the complete configured confusion matrix. No label-informed candidate construction or validation/test label enters graph structure.
