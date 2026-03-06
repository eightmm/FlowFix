# Cartesian v4 Baseline (rectified-flow-full-v4)

## Model
- **Architecture**: Joint graph, 6-layer GatingEquivariantLayer
- **Output**: Per-atom Cartesian velocity [N, 3]
- **Epochs**: 500
- **Checkpoint**: `save/rectified-flow-full-v4/checkpoints/latest.pt` (1.1GB)

## Results (Full Validation, 200 PDBs)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Mean RMSD | 3.20 A | 2.64 A | -0.56 A |
| Median RMSD | 3.00 A | 2.22 A | -0.78 A |

## Files

- `train_config.yaml` - Training configuration used
- `full_validation_results.json` - Full validation inference results (200 PDBs, all poses)
- `latest_train_valid_5.json` - Small train/valid 5 sample inference
- `train5_allposes_latest.json` - Train 5 all poses inference
- Visualization plots: `reports/assets/` (RMSD distribution, scatter, improvement)
