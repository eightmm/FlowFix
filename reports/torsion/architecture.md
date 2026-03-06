# SE(3) + Torsion Decomposition Architecture

> **Model**: `ProteinLigandFlowMatchingTorsion`
>
> **Output**: Translation [3] + Rotation [3] + Torsion [M] (instead of per-atom velocity [N, 3])

---

## 1. Overview

Molecular pose를 SE(3) + Torsion 공간으로 분해하여, Cartesian 3N차원 대신 (3 + 3 + M)차원에서 velocity field를 학습합니다.

- **Translation [3D]**: 분자 중심의 이동
- **Rotation [3D, SO(3)]**: 분자 전체 회전 (axis-angle)
- **Torsion [M]**: M개 rotatable bond의 회전각

### Dimension Reduction

일반적인 drug molecule (N=44 atoms, M=8 rotatable bonds):
- Cartesian: 44 x 3 = **132D**
- Torsion: 3 + 3 + 8 = **14D** (9.4x reduction)

---

## 2. Architecture

### 2.1 Backbone (Shared with Cartesian)

Encoder, interaction network, velocity blocks는 base `ProteinLigandFlowMatching`과 동일.

```
ProteinLigandFlowMatchingTorsion (inherits ProteinLigandFlowMatching)
├── [shared] protein_encoder, ligand_encoder
├── [shared] cross_attention, velocity_blocks
├── [new] translation_head: EquivariantMLP -> 1x1o [3D]
├── [new] rotation_head: EquivariantMLP -> 1x1o [3D]
└── [new] torsion_head: MLP(2*scalar_dim -> 1) per bond
```

### 2.2 Output Heads

**Translation & Rotation**: Global pooling (scatter_mean) -> EquivariantMLP -> 1x1o

```
h_scalar[mol_i] -> scatter_mean -> EquivariantMLP -> translation [3]
h_scalar[mol_i] -> scatter_mean -> EquivariantMLP -> rotation [3]
```

**Torsion**: Edge-level prediction (DiffDock approach)

```
For each rotatable bond (src, dst):
    cat(h_scalar[src], h_scalar[dst]) -> MLP -> scalar angle
```

### 2.3 Application Order

Torsion -> Translation -> Rotation (DiffDock convention)

```
1. Apply torsion angles (Rodrigues rotation per bond)
2. Translate center of mass
3. Rotate around center of mass (axis-angle via Rodrigues)
```

---

## 3. Loss Function

```
L = w_trans * L_trans + w_rot * L_rot + w_tor * L_tor + w_coord * L_coord

L_trans = MSE(pred_trans, target_trans)
L_rot   = MSE(pred_rot, target_rot)
L_tor   = circular_MSE(pred_tor, target_tor)    # atan2(sin, cos) wrapping
L_coord = MSE(reconstruct(x0, pred), x1)        # end-to-end reconstruction
```

Default weights: `w_trans=1.0, w_rot=1.0, w_tor=1.0, w_coord=0.5`

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Torsion (shared backbone + decomposed heads) |
| Hidden scalar/vector | 128 / 32 |
| Optimizer | Muon (lr=0.005) + AdamW (lr=3e-4) |
| Schedule | Warmup 5% + Plateau 80% + Cosine 15% |
| Loss | Trans MSE + Rot MSE + Circular Torsion + Coord Recon |
| EMA | decay=0.999 |
| Batch size | 32 |
| Epochs | 500 |
| ODE sampling | 20-step Euler |

---

## 5. File Map

```
src/models/flowmatching_torsion.py   # ProteinLigandFlowMatchingTorsion
src/data/dataset_torsion.py          # FlowFixTorsionDataset, collate_torsion_batch
src/utils/losses_torsion.py          # compute_se3_torsion_loss, reconstruct_coords
src/utils/sampling_torsion.py        # sample_trajectory_torsion
train_torsion.py                     # FlowFixTorsionTrainer
configs/train_torsion.yaml           # Training config
```

---

## 6. References

- DiffDock (Corso et al., 2023): Product space diffusion on R3 x SO(3) x T^M
- Torsional Diffusion (Jing et al., 2022): Diffusion on torsion angles
