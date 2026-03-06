# FlowFix Model Architecture

> **SE(3)-Equivariant Flow Matching for Protein-Ligand Pose Refinement**
>
> Last updated: 2025-03-06

---

## 1. Overview

FlowFix는 docking pose를 crystal structure로 refinement하는 SE(3)-equivariant flow matching 모델입니다.
Linear interpolation path `x_t = (1-t)*x0 + t*x1`을 따라 per-atom velocity field `v(x_t)`를 학습합니다.

### Pipeline Summary

```mermaid
flowchart LR
    subgraph input["Input"]
        X0["Docked Pose\nx_0 (t=0)"]
    end

    subgraph model["FlowFix Model"]
        direction TB
        ENC["Protein/Ligand\nEncoding"]
        INT["Cross-Attention\nInteraction"]
        VEL["Velocity\nPrediction"]
        ENC --> INT --> VEL
    end

    subgraph output["Output"]
        X1["Refined Pose\nx_1 (t=1)"]
    end

    X0 -->|"ODE Integration\n(Euler/RK4)"| model -->|"v(x_t)"| X1
```

### Key Design Choices

| Component | Previous (v4) | Current | Rationale |
|-----------|--------------|---------|-----------|
| Graph structure | Joint protein-ligand graph | Separate encoders + cross-attention | Protein/ligand feature type이 다름 (vector vs scalar) |
| Interaction | Message passing on joint graph | Pair-bias attention (cuEquivariance) | 더 expressive한 long-range interaction |
| Time conditioning | Explicit sinusoidal embedding | Implicit (x_t coordinates) | Linear path에서 v = x1 - x0는 time-independent |
| Optimizer | Muon + AdamW hybrid | Adam | 단순화 |
| Velocity output | Single equivariant MLP | 4-layer conditioned GatingEquivariantLayer | Richer conditioning with protein context |

---

## 2. Model Architecture

### 2.1 Full Pipeline

```mermaid
flowchart TB
    subgraph stage1["Stage 1: Feature Encoding"]
        direction TB
        P["Protein Graph\nscalar: 76d, vector: 31x3d\nedge scalar: 39d, edge vector: 8x3d"]
        L["Ligand Graph\nscalar: 121d\nedge: 44d"]

        ESM["ESM Integration\nESMC 600M (1152d) + ESM3 (1536d)\nlearnable weighted sum\nresidual add to protein.x"]

        PN["ProteinNetwork\nUnifiedEquivariantNetwork\nEquivMLP + 3x GatingEquivLayer\nOutput: 128x0e + 32x1o + 32x1e"]

        LN["LigandNetwork\nUnifiedEquivariantNetwork\nEquivMLP + 3x GatingEquivLayer\nOutput: 128x0e + 16x1o + 16x1e"]

        P --> ESM --> PN
        L --> LN
    end

    subgraph stage2["Stage 2: Protein-Ligand Interaction"]
        direction TB
        PROJ["Equivariant -> Scalar Projection\nEquivariantMLP per type\n-> 256d scalars"]

        PAIR["Pair Bias Features\n32 RBF + 3 interaction types\n+ inv_dist + norm_dist + mask\nMLP -> 64d"]

        ATT["2x Pair-Bias Attention Block\n8 heads, pair_dim=64\ncuequivariance attention_pair_bias\n+ FFN (256->512->256) + LayerNorm"]

        POOL["Output\nlig_out: [N_l, 256] atom-wise\nprot_global: [B, 512] mean+std pool"]

        PROJ --> PAIR --> ATT --> POOL
    end

    subgraph stage3["Stage 3: Velocity Prediction"]
        direction TB
        COND["Condition Assembly\nprotein_global [B,512]\n+ lig_out [N_l,256]\n-> MLP -> atom_condition [N_l,256]"]

        VINP["Input: EquivariantMLP\nligand_output -> vel_hidden_irreps"]

        VBLK["4x GatingEquivariantLayer\nwith EquivariantAdaLN conditioning\n(input + output conditioning)"]

        VOUT["Output: EquivariantMLP (3-layer)\nvel_hidden -> 1x1o (3D vector)\nzero-init, scale=0.1"]

        COND --> VBLK
        VINP --> VBLK --> VOUT
    end

    PN --> PROJ
    LN --> PROJ
    LN -->|"ligand_output"| VINP
    ATT -->|"lig_out"| COND
    POOL -->|"prot_global"| COND

    subgraph out["Output"]
        V["Velocity: [N_ligand, 3]\nper-atom displacement vector"]
    end

    VOUT --> V

    style stage1 fill:#e8f5e9,stroke:#2E7D32
    style stage2 fill:#fff3e0,stroke:#E65100
    style stage3 fill:#fce4ec,stroke:#C62828
    style out fill:#f3e5f5,stroke:#6A1B9A
```

---

### 2.2 GatingEquivariantLayer (Core Block)

모든 equivariant processing의 기본 단위. Protein encoder, ligand encoder, velocity predictor 모두에서 사용.

```mermaid
flowchart TB
    subgraph input["Input"]
        H["node_features [N, irreps]"]
        C["condition [N, D]\n(velocity blocks only)"]
    end

    ADALN1["Input EquivariantAdaLN\nscalar: LayerNorm + scale/bias from condition\nvector: norm-based gating from condition"]

    subgraph mp["Message Passing"]
        SH["Spherical Harmonics Y_l(r_ij)\nl = 0, 1, 2"]
        EDGE["Edge Embedding\nMLP(edge_attr) -> hidden"]
        TP["Tensor Product\nnode[src] x (edge_emb, edge_sh)"]
        MSG["Message MLP"]
        IMP["Edge Importance\nsigmoid(MLP(src, dst, edge))"]
        SG["Scalar Gate\nsigmoid(MLP(edge_emb))"]
        VG["Vector Gate\nnorm features -> sigmoid(MLP)"]
        AGG["Scatter Sum\n[E] -> [N]"]
    end

    SELF["Self-Interaction\nTensorProduct(node, ones)"]

    ADD["Sum: aggregated + self_update"]
    NMLP["Node Update EquivariantMLP"]
    BN["Equivariant BatchNorm"]
    ADALN2["Output EquivariantAdaLN"]
    SKIP["Skip Connection: output + identity"]

    H --> ADALN1
    C --> ADALN1
    ADALN1 --> TP
    ADALN1 --> SELF
    SH --> TP
    EDGE --> TP
    TP --> MSG --> IMP --> SG --> VG --> AGG
    AGG --> ADD
    SELF --> ADD
    ADD --> NMLP --> BN --> ADALN2 --> SKIP
    C --> ADALN2

    subgraph output["Output"]
        OUT["updated node_features [N, irreps]"]
    end
    SKIP --> OUT

    style mp fill:#fff8e1,stroke:#F9A825
```

**핵심 특징:**
- **Tensor Product**: `cuequivariance_torch.FullyConnectedTensorProduct`로 SE(3) equivariance 보존
- **Dual Gating**: Scalar gate (element-wise) + Vector gate (norm-based adaptive)
- **Edge Importance**: 학습 가능한 message-level attention weight
- **Dual AdaLN**: Input/Output 양쪽에서 context conditioning (velocity blocks에서만)

---

### 2.3 Cross-Attention Interaction

Protein-ligand 간 상호작용을 pair-bias attention으로 모델링.

```mermaid
flowchart TB
    subgraph input["Inputs"]
        PF["Protein features\n128x0e + 32x1o + 32x1e"]
        LF["Ligand features\n128x0e + 16x1o + 16x1e"]
    end

    subgraph proj["Equivariant -> Scalar"]
        P2S["EquivariantMLP\n-> 256d scalars"]
        L2S["EquivariantMLP\n-> 256d scalars"]
    end

    subgraph seq["Sequence Format"]
        PAD["PyG -> Padded [B, N, D]\nDynamic padding"]
        CAT["Concatenate\n[B, N_p+N_l, 256]"]
    end

    subgraph pair["Pair Bias [B, N, N, 64]"]
        RBF["32 Gaussian RBF\ncenters: 0-20A, width: 2.5A"]
        TYPE["3 Interaction type flags\nPP / PL / LL"]
        EXTRA["inv_dist + norm_dist + mask"]
        PMLP["MLP: 38 -> 64 -> 64"]
    end

    subgraph attn["Attention Stack (x2)"]
        direction TB
        QKV["QKV Projection (no bias)\n[B, N, 256] -> [B, H, N, 32] x 3"]
        APB["cuequivariance attention_pair_bias\n8 heads + pair bias gating"]
        LN1["LayerNorm + Dropout + Residual"]
        FFN["FFN: 256 -> 512 (SiLU) -> 256"]
        LN2["LayerNorm + Dropout + Residual"]
        QKV --> APB --> LN1 --> FFN --> LN2
    end

    GSKIP["Global Skip: h + h_initial"]

    subgraph output["Outputs"]
        LO["lig_out [N_l, 256]\natom-wise interaction features"]
        PG["prot_global [B, 512]\nmean + std pooling"]
    end

    PF --> P2S --> PAD
    LF --> L2S --> PAD
    PAD --> CAT --> attn
    RBF --> PMLP
    TYPE --> PMLP
    EXTRA --> PMLP
    PMLP --> attn
    attn --> GSKIP --> output

    style pair fill:#fff3e0,stroke:#E65100
    style attn fill:#e3f2fd,stroke:#1565C0
```

---

### 2.4 ESM Embedding Integration

Pre-trained protein language model (PLM)의 residue-level embedding을 protein features에 통합.

```mermaid
flowchart LR
    ESMC["ESMC 600M\n[N, 1152]"] --> P1["MLP\n1152->128->128"]
    ESM3["ESM3\n[N, 1536]"] --> P2["MLP\n1536->128->128"]

    P1 --> WS["Weighted Sum\nsoftmax(w) * proj"]
    P2 --> WS

    WS --> B2I["MLP: 128->76"]
    PX["protein.x [N,76]"] --> ADD["Residual Add"]
    B2I --> ADD
    ADD --> OUT["enhanced protein.x [N,76]"]
```

- ESMC와 ESM3의 가중치는 학습 가능한 파라미터 (`nn.Parameter`)
- Softmax로 정규화 후 weighted sum
- 원래 protein feature에 residual connection으로 추가

---

## 3. Training

### 3.1 Flow Matching Loss

```mermaid
flowchart LR
    subgraph sampling["Timestep Sampling"]
        TS["Logistic-Normal\nmu=0.8, sigma=1.7\nmix_ratio=0.98 with uniform"]
    end

    subgraph interpolation["Linear Interpolation"]
        X0["x_0 (docked)"]
        X1["x_1 (crystal)"]
        XT["x_t = (1-t)*x_0 + t*x_1"]
    end

    subgraph loss["Loss Computation"]
        VP["v_pred = model(protein, ligand_t, t)"]
        VT["v_true = x_1 - x_0\n(constant for linear path)"]
        MSE["L_flow = MSE(v_pred, v_true)"]
        DG["L_dg = distance geometry\nbond length/angle constraints\ntime-weighted (stronger near t=1)"]
        TOTAL["L_total = L_flow + w * L_dg"]
    end

    TS --> XT
    X0 --> XT
    X1 --> XT
    XT --> VP --> MSE --> TOTAL
    VT --> MSE
    DG --> TOTAL
```

**Multi-timestep training**: 각 PDB system에 대해 `num_timesteps_per_sample`개의 서로 다른 timestep에서 loss 계산.
이를 위해 batch를 replicate하여 효율적으로 처리.

### 3.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4, eps=1e-8) |
| Scheduler | Cosine Annealing (min_lr=1e-6, epoch-based) |
| Gradient clipping | 1.0 |
| Batch size | config-dependent |
| Gradient accumulation | configurable |
| Distance geometry weight | 0.1 |
| Dropout | 0.1 |
| Early stopping | patience=50 on success rate <2A |

### 3.3 ODE Sampling (Inference/Validation)

```mermaid
flowchart LR
    X0["x_0\n(docked)"] --> LOOP

    subgraph LOOP["ODE Integration (N steps)"]
        direction TB
        T["t_i -> t_{i+1}"]
        V["v = model(protein, ligand_{t_i}, t_i)"]
        EU["Euler: x += dt * v"]
        RK["RK4: 4-stage weighted average"]
        T --> V
        V --> EU
        V --> RK
    end

    LOOP --> X1["x_1\n(refined)"]
```

**Timestep schedules:**
- `uniform`: 등간격
- `quadratic`: t=1 근처에서 dense (1-(1-t)^1.5)
- `root`: t^(2/3)
- `sigmoid`: 양 끝점 근처에서 dense

---

## 4. Dimension Reference

### Feature Dimensions

```
Protein:
  Node scalar:  76  ─┐
  Node vector:  31x3 ├─> UnifiedEquivNet ──> 128x0e + 32x1o + 32x1e (320d)
  Edge scalar:  39   │
  Edge vector:  8x3  ┘

Ligand:
  Node scalar:  121 ─┐
  Edge scalar:  44   ├─> UnifiedEquivNet ──> 128x0e + 16x1o + 16x1e (224d)
                     ┘

Interaction:
  Input:   320d (protein) + 224d (ligand)
  Hidden:  256d (scalar only, after equivariant projection)
  Pair:    64d (RBF + type features)
  Output:  256d per atom (ligand), 512d global (protein mean+std)

Velocity:
  Input:   224d (ligand irreps)
  Hidden:  128x0e + 16x1o + 16x1e (224d)
  Condition: 256d (protein_global 512 + lig_out 256 -> MLP -> 256)
  Output:  1x1o = 3d (velocity vector per atom)
```

### Irreps Notation Quick Reference

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| `Nx0e` | N scalar channels (even parity) | N |
| `Nx1o` | N true vector channels (odd parity) | N x 3 |
| `Nx1e` | N pseudo-vector channels (even parity) | N x 3 |

---

## 5. Module Dependency

```mermaid
graph TD
    FM["ProteinLigandFlowMatching\nsrc/models/flowmatching.py"]

    PN["UnifiedEquivariantNetwork\n(Protein Encoder)"]
    LN["UnifiedEquivariantNetwork\n(Ligand Encoder)"]
    IN["ProteinLigandInteractionNetwork"]

    FM --> PN
    FM --> LN
    FM --> IN

    subgraph network["src/models/network.py"]
        PN
        LN
        IN
    end

    subgraph cue["src/models/cue_layers.py"]
        GEL["GatingEquivariantLayer"]
        EMLP["EquivariantMLP"]
        PBA["PairBiasAttentionLayer"]
        ADALN_E["EquivariantAdaLN"]
    end

    subgraph torch_l["src/models/torch_layers.py"]
        MLP["MLP"]
        ADALN["AdaLN"]
        CTB["ConditionedTransitionBlock"]
    end

    subgraph nvidia["cuequivariance_torch (NVIDIA)"]
        CUE_TP["FullyConnectedTensorProduct"]
        CUE_SH["SphericalHarmonics"]
        CUE_LIN["Linear"]
        CUE_ATT["attention_pair_bias"]
    end

    PN --> GEL
    PN --> EMLP
    LN --> GEL
    LN --> EMLP
    IN --> EMLP
    IN --> PBA
    IN --> MLP
    FM --> GEL
    FM --> EMLP
    FM --> MLP

    GEL --> CUE_TP
    GEL --> CUE_SH
    GEL --> ADALN_E
    GEL --> MLP
    EMLP --> CUE_LIN
    PBA --> CUE_ATT

    style FM fill:#ffcdd2,stroke:#B71C1C
    style nvidia fill:#bbdefb,stroke:#0D47A1
    style cue fill:#c8e6c9,stroke:#1B5E20
    style torch_l fill:#fff9c4,stroke:#F57F17
```

---

## 6. Parameter Count Breakdown

| Module | Approx. Parameters | Role |
|--------|-------------------|------|
| ProteinNetwork | ~1.5M | Protein structure encoding |
| LigandNetwork | ~1.0M | Ligand structure encoding |
| ESM Projections | ~0.5M | PLM embedding integration |
| InteractionNetwork | ~2.5M | Cross-attention + pair bias |
| Velocity Blocks (x4) | ~3.0M | Conditioned velocity prediction |
| Velocity I/O MLPs | ~0.5M | Input/output projections |
| **Total** | **~9M** | |

> 정확한 수치는 config에 따라 다름. `train.py` 실행 시 출력됨.

---

## 7. File Map

```
src/models/
  flowmatching.py    ProteinLigandFlowMatching (top-level)
  network.py         UnifiedEquivariantNetwork, ProteinLigandInteractionNetwork
  cue_layers.py      GatingEquivariantLayer, EquivariantMLP, PairBiasAttention, ...
  torch_layers.py    MLP, AdaLN, SwiGLU, TimeEmbedding, ...

src/data/
  dataset.py         FlowFixDataset (lazy/hybrid/preload)
  protein_feat.py    Protein featurization + ESM
  ligand_feat.py     Ligand featurization

src/utils/
  losses.py          Distance geometry loss, clash loss
  sampling.py        ODE integration, timestep schedules
  training_utils.py  Optimizer, scheduler builders
  model_builder.py   Config -> model construction

train.py             Training loop (FlowFixTrainer)
inference.py         Inference script
```
