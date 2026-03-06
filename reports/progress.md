# FlowFix Progress Report

> **SE(3)-Equivariant Flow Matching for Protein-Ligand Pose Refinement**
>
> Last updated: 2026-03-06

---

## Approach Comparison

| | Cartesian (v4) | SE(3) + Torsion |
|--|----------------|-----------------|
| **Output** | Per-atom velocity [N, 3] | Trans [3] + Rot [3] + Torsion [M] |
| **Dimension** | 3N (~132D) | 3+3+M (~14D) |
| **Status** | Trained (500 epochs) | Implemented, not yet trained |
| **Mean RMSD** | 3.20A -> 2.64A | - |
| **Success <2A** | 30.4% -> 44.6% | - |
| **Improved poses** | 75.2% | - |

---

## Cartesian (v4 Baseline)

- **Architecture**: Joint graph, 8x GatingEquivariantLayer, ~13M params
- **Results**: [cartesian/results.md](cartesian/results.md)
- **Architecture detail**: [cartesian/architecture.md](cartesian/architecture.md)
- **Inference data**: `output/cartesian_v4_baseline/`

### Key Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Mean RMSD | 3.20 A | 2.64 A | -0.56 A |
| Median RMSD | 3.00 A | 2.22 A | -0.78 A |
| Success rate (<2A) | 30.4% | 44.6% | +14.2%p |

---

## SE(3) + Torsion

- **Architecture**: Shared backbone + decomposed output heads (trans/rot/torsion)
- **Results**: [torsion/results.md](torsion/results.md)
- **Architecture detail**: [torsion/architecture.md](torsion/architecture.md)

### Expected Benefits

- 3-10x dimension reduction (물리적 자유도만 학습)
- Bond length/angle 자동 보존
- Interpretable decomposition (translation vs rotation vs torsion)

---

## TODO / Next Steps

- [ ] Torsion 모델 학습 및 Cartesian과 비교
- [ ] Success rate <2A 목표: 60%+
- [ ] Self-conditioning ablation
- [ ] Multi-step refinement
- [ ] Inference speed optimization

---

## Changelog

### 2026-03-06 - SE(3) + Torsion Implementation
- Translation [3] + Rotation [3] + Torsion [M] decomposition 구현
- 별도 model/dataset/loss/sampling/trainer 파일 분리
- Codebase cleanup (dead files, legacy configs 정리)

### 2026-02-18 - v4 Baseline Results
- Full validation on 200 PDBs (11,543 poses)
- 20-step Euler ODE with EMA model
- Mean RMSD: 3.20A -> 2.64A, Success rate <2A: 30.4% -> 44.6%

### 2026-02 - Joint Graph Architecture (v4)
- Joint protein-ligand graph with 4 edge types
- 8x GatingEquivariantLayer with time AdaLN
- cuEquivariance tensor product
- Muon + AdamW hybrid optimizer
