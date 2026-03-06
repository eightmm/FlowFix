# FlowFix Architecture Documentation

## Overview

FlowFix refines perturbed protein-ligand binding poses back to crystal structures using **SE(3)-equivariant flow matching**. The model learns velocity fields that move ligand atoms from docked poses (t=0) to crystal structures (t=1) via linear interpolation.

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        P["Protein Graph\n(scalar + vector features)"]
        L["Ligand Graph\n(scalar features + x_t coords)"]
        T["t (timestep)\n[B] in [0,1]"]
    end

    subgraph encoding["1. Feature Encoding"]
        ESM["ESM Embedding Integration\nESMC (1152d) + ESM3 (1536d)\n-> Weighted projection -> Residual add"]
        PN["Protein Network\nUnifiedEquivariantNetwork\n3 GatingEquivariantLayers"]
        LN["Ligand Network\nUnifiedEquivariantNetwork\n3 GatingEquivariantLayers"]
    end

    subgraph interaction["2. Protein-Ligand Interaction"]
        PROJ["Equivariant -> Scalar Projection\nEquivariantMLP per molecule type"]
        SEQ["PyG -> Padded Sequence\nDynamic padding per batch"]
        PAIR["Pair Bias Construction\nRBF (32) + interaction types + dist features"]
        ATT["Pair-Bias Attention Blocks x2\n+ FFN + LayerNorm + Residual"]
        POOL["Global Pooling\nMean + Std -> [B, D*2]"]
    end

    subgraph velocity["3. Velocity Prediction"]
        COND["Condition Assembly\nprotein_global [B,512] + lig_interaction [N,256]\n-> MLP -> atom_condition [N,256]"]
        VINP["Velocity Input Projection\nEquivariantMLP: ligand_irreps -> vel_hidden_irreps"]
        VBLK["Velocity Blocks x4\nGatingEquivariantLayer\nwith AdaLN conditioning"]
        VOUT["Velocity Output\nEquivariantMLP -> 1x1o (3D vector)\n* learnable scale (init=0.1)"]
    end

    subgraph output["Output"]
        V["Velocity Field\n[N_ligand, 3]"]
    end

    P --> ESM --> PN
    L --> LN
    T -.->|"implicit in x_t"| LN

    PN -->|"128x0e+32x1o+32x1e"| PROJ
    LN -->|"128x0e+16x1o+16x1e"| PROJ
    PROJ --> SEQ --> PAIR --> ATT
    ATT --> POOL

    LN -->|"ligand_output"| VINP
    POOL -->|"protein_global"| COND
    ATT -->|"lig_out (atom-wise)"| COND
    COND --> VBLK
    VINP --> VBLK
    VBLK --> VOUT --> V

    style inputs fill:#e8f4f8,stroke:#2196F3
    style encoding fill:#e8f5e9,stroke:#4CAF50
    style interaction fill:#fff3e0,stroke:#FF9800
    style velocity fill:#fce4ec,stroke:#E91E63
    style output fill:#f3e5f5,stroke:#9C27B0
```

---

## Detailed Module Diagrams

### 1. UnifiedEquivariantNetwork (Encoder)

Shared architecture for both protein and ligand encoding. Uses cuEquivariance for SE(3)-equivariant operations.

```mermaid
flowchart LR
    subgraph input["Input"]
        NF["Node Features\n[N, scalar_dim]\n(+ [N, vec_dim, 3])"]
        EF["Edge Features\n[E, edge_dim]\n(+ [E, edge_vec_dim, 3])"]
        POS["Positions\n[N, 3]"]
    end

    subgraph processing["Processing"]
        NP["Node Processor\nEquivariantMLP (3-layer)\nin_irreps -> hidden_irreps"]
        EP["Edge Processor\nEquivariantMLP (2-layer)\nedge_irreps -> edge_hidden"]

        subgraph layers["GatingEquivariantLayer x N"]
            direction TB
            L1["Layer 1"]
            L2["Layer 2"]
            L3["Layer N"]
            BN1["BatchNorm"]
            BN2["BatchNorm"]
            BN3["BatchNorm"]
            L1 --> BN1 --> L2 --> BN2 --> L3 --> BN3
        end

        OP["Output Projection\nEquivariantMLP (2-layer)\nhidden_irreps -> out_irreps"]
    end

    NF --> NP --> layers
    EF --> EP --> layers
    POS --> layers
    layers --> OP

    subgraph output["Output"]
        OUT["Node Features\n[N, out_irreps.dim]"]
    end

    OP --> OUT
```

**Protein config**: `76x0e + 31x1o` input, `128x0e + 32x1o + 32x1e` output, 3 layers
**Ligand config**: `121x0e` input (scalar only), `128x0e + 16x1o + 16x1e` output, 3 layers

---

### 2. GatingEquivariantLayer (Core Building Block)

The fundamental SE(3)-equivariant message passing layer used throughout the model.

```mermaid
flowchart TB
    subgraph input["Input"]
        H["node_features\n[N, irreps.dim]"]
        POS["positions [N, 3]"]
        EI["edge_index [2, E]"]
        EA["edge_attr [E, edge_dim]"]
        COND["condition [N, cond_dim]\n(optional)"]
    end

    subgraph conditioning["0. Input Conditioning"]
        ADALN_IN["EquivariantAdaLN\nscalar: LayerNorm + scale/bias\nvector: norm-based gating"]
    end

    subgraph edges["Edge Processing"]
        SH["Spherical Harmonics\nY_l(r_ij), l=0,1,2"]
        EE["Edge Embedding\nMLP(edge_attr) -> [E, hidden]"]
        EF["edge_features =\ncat(edge_emb, edge_sh)"]
    end

    subgraph message["Message Passing"]
        TP["Tensor Product (message)\nnode[src] x edge_features -> messages"]
        MLP_MSG["Message MLP\nEquivariantMLP"]
        IMP["Edge Importance\nMLP(src_scalar, dst_scalar, edge_emb)\n-> sigmoid -> weight"]
        SG["Scalar Gating\nMLP(edge_emb) -> sigmoid"]
        VG["Vector Gating\nnorm features + edge_emb\n-> MLP -> sigmoid"]
        AGG["Scatter Sum\n[E, D] -> [N, D]"]
    end

    subgraph self_int["Self-Interaction"]
        TP_SELF["Tensor Product (self)\nnode x ones -> self_update"]
    end

    subgraph update["Node Update"]
        ADD["aggregated + self_update"]
        NMLP["Node Update MLP\nEquivariantMLP"]
        BN["Equivariant BatchNorm"]
        ADALN_OUT["EquivariantAdaLN\n(output conditioning)"]
        DO["Equivariant Dropout"]
        SKIP["Skip Connection\noutput + identity"]
    end

    H --> ADALN_IN
    COND --> ADALN_IN
    ADALN_IN --> TP
    ADALN_IN --> TP_SELF
    POS --> SH
    EA --> EE
    SH --> EF
    EE --> EF
    EF --> TP
    TP --> MLP_MSG --> IMP --> SG --> VG --> AGG
    TP_SELF --> ADD
    AGG --> ADD
    ADD --> NMLP --> BN --> ADALN_OUT --> DO --> SKIP

    COND --> ADALN_OUT

    subgraph output["Output"]
        OUT["updated node_features\n[N, out_irreps.dim]"]
    end

    SKIP --> OUT

    style conditioning fill:#e3f2fd,stroke:#1565C0
    style message fill:#fff8e1,stroke:#F9A825
    style self_int fill:#f1f8e9,stroke:#558B2F
    style update fill:#fce4ec,stroke:#C62828
```

---

### 3. ProteinLigandInteractionNetwork

Cross-attention between protein and ligand using NVIDIA's cuEquivariance `attention_pair_bias`.

```mermaid
flowchart TB
    subgraph input["Input"]
        PO["Protein Output\n[N_p, prot_irreps]"]
        LO["Ligand Output\n[N_l, lig_irreps]"]
    end

    subgraph projection["Equivariant -> Scalar"]
        P2S["protein_to_scalar\nEquivariantMLP (2-layer)\nprot_irreps -> hidden_dim x 0e"]
        L2S["ligand_to_scalar\nEquivariantMLP (2-layer)\nlig_irreps -> hidden_dim x 0e"]
    end

    subgraph sequence["Sequence Conversion"]
        SEQ["PyG -> Padded Sequence\n[B, max_N, D] + mask"]
        CAT["Concatenate\ncombined = [prot_seq | lig_seq]\n[B, N_p+N_l, D]"]
    end

    subgraph pair["Pair Bias"]
        DIST["Pairwise Distance\ntorch.cdist"]
        RBF["RBF Features (32)\nexp(-(d-c)^2 / 2w^2)"]
        TYPE["Interaction Types\nPP / PL / LL flags"]
        FEAT["Additional Features\ninv_dist, norm_dist, node_mask"]
        PPROJ["Pair Projection\nMLP: (32+6) -> pair_dim"]
    end

    subgraph attention["Attention Stack x2"]
        direction TB
        ATTN["PairBiasAttentionLayer\nQKV projection (no bias)\ncuequivariance attention_pair_bias\n8 heads, pair_dim=64"]
        ANORM["LayerNorm + Dropout\n+ Residual"]
        FFN["FFN Block\nMLP: D -> 2D -> D (SiLU)"]
        FNORM["LayerNorm + Dropout\n+ Residual"]
        ATTN --> ANORM --> FFN --> FNORM
    end

    subgraph output["Output"]
        GSKIP["Global Skip Connection\nh + h_initial"]
        SPLIT["Split Protein / Ligand"]
        P2PYG["Sequence -> PyG\nprot_out [N_p, D]"]
        L2PYG["Sequence -> PyG\nlig_out [N_l, D]"]
        GPOOL["Mean+Std Pooling\nprot_global [B, D*2]\nlig_global [B, D*2]"]
    end

    PO --> P2S --> SEQ
    LO --> L2S --> SEQ
    SEQ --> CAT
    CAT --> DIST --> RBF --> PPROJ
    DIST --> TYPE --> PPROJ
    DIST --> FEAT --> PPROJ
    CAT --> attention
    PPROJ --> attention
    attention --> GSKIP --> SPLIT
    SPLIT --> P2PYG
    SPLIT --> L2PYG
    SPLIT --> GPOOL

    style projection fill:#e8f5e9,stroke:#2E7D32
    style pair fill:#fff3e0,stroke:#E65100
    style attention fill:#e3f2fd,stroke:#1565C0
```

---

### 4. Velocity Prediction Pipeline

How the final velocity vectors are computed from encoded features.

```mermaid
flowchart TB
    subgraph input["Inputs from Previous Stages"]
        LO["ligand_output\n[N_l, lig_irreps]\nfrom LigandNetwork"]
        PG["protein_global\n[B, hidden_dim*2]\nmean+std pooling"]
        LI["lig_out\n[N_l, hidden_dim]\nfrom InteractionNetwork"]
    end

    subgraph condition["Condition Assembly"]
        EXP["Expand protein_global\n[B, D*2] -> [N_l, D*2]\nvia batch indices"]
        CAT["Concatenate\n[N_l, D*2 + D] = [N_l, D*3]"]
        CPROJ["Condition MLP\nMLP: D*3 -> D*3 -> D\natom_condition [N_l, D]"]
    end

    subgraph velocity_net["Velocity Network"]
        VINP["Input Projection\nEquivariantMLP (2-layer)\nlig_irreps -> vel_hidden_irreps"]
        SAVE["Save h_initial\nfor global skip"]

        subgraph blocks["GatingEquivariantLayer x4"]
            B1["Block 1\n+ AdaLN(atom_condition)"]
            B2["Block 2"]
            B3["Block 3"]
            B4["Block 4"]
            B1 --> B2 --> B3 --> B4
        end

        GSKIP["Global Skip\nh + h_initial"]
        VOUT["Output MLP\nEquivariantMLP (3-layer)\nvel_hidden -> 1x1o"]
        SCALE["Learnable Scale\n* velocity_scale (init=0.1)"]
    end

    subgraph output["Output"]
        VEL["velocity [N_l, 3]"]
    end

    PG --> EXP --> CAT
    LI --> CAT
    CAT --> CPROJ
    LO --> VINP --> SAVE --> blocks
    CPROJ --> blocks
    blocks --> GSKIP --> VOUT --> SCALE --> VEL

    style condition fill:#f3e5f5,stroke:#6A1B9A
    style velocity_net fill:#fce4ec,stroke:#C62828
```

---

### 5. ESM Embedding Integration

How pre-trained protein language model embeddings are integrated.

```mermaid
flowchart LR
    subgraph input["Input"]
        PX["protein.x\n[N, 76]"]
        ESMC["esmc_embeddings\n[N, 1152]"]
        ESM3["esm3_embeddings\n[N, 1536]"]
    end

    subgraph projection["Projection"]
        P1["ESMC Projection\nMLP: 1152 -> 128 -> 128"]
        P2["ESM3 Projection\nMLP: 1536 -> 128 -> 128"]
    end

    subgraph combine["Weighted Combination"]
        W["Learnable Weights\nsoftmax([w_esmc, w_esm3])"]
        WS["Weighted Sum\nw1*esmc_proj + w2*esm3_proj"]
    end

    subgraph integrate["Integration"]
        B2I["Back to Input Dim\nMLP: 128 -> 76 -> 76"]
        RES["Residual Add\nprotein.x + esm_proj"]
    end

    ESMC --> P1 --> WS
    ESM3 --> P2 --> WS
    W --> WS
    WS --> B2I --> RES
    PX --> RES

    subgraph output["Output"]
        OUT["enhanced protein.x\n[N, 76]"]
    end

    RES --> OUT
```

---

## Training Pipeline

### Flow Matching Training

```mermaid
flowchart TB
    subgraph data["Data Loading"]
        DS["FlowFixDataset\n1 random pose per PDB per epoch"]
        DL["DataLoader + collate\nPyG Batch"]
    end

    subgraph train_step["Training Step"]
        TS["Sample Timesteps\nLogistic-Normal(mu=0.8, sigma=1.7)\nnum_timesteps_per_sample x batch_size"]
        REP["Replicate Batch\nEach PDB x num_timesteps"]
        INTERP["Linear Interpolation\nx_t = (1-t)*x0 + t*x1"]
        FWD["Model Forward\nv_pred = model(protein, ligand_t, t)"]
        LOSS["MSE Loss\n||v_pred - (x1-x0)||^2"]
        DG["Distance Geometry Loss\nBond length/angle constraints\nTime-weighted (stronger near t=1)"]
        TOTAL["Total Loss\nMSE + DG_weight * DG_loss"]
        BACK["Backward + Grad Clip (1.0)\n+ Gradient Accumulation"]
    end

    subgraph schedule["Learning Rate"]
        OPT["Adam (lr=1e-4, eps=1e-8)"]
        SCHED["Cosine Annealing\nmin_lr=1e-6, epoch-based"]
    end

    DS --> DL --> TS --> REP --> INTERP --> FWD --> LOSS --> TOTAL --> BACK
    DG --> TOTAL
    OPT --> BACK
    SCHED --> OPT
```

### Validation (ODE Sampling)

```mermaid
flowchart LR
    subgraph sampling["ODE Integration"]
        X0["x_0 (docked pose)"]
        STEP["For each timestep t_i:"]
        VEL["v = model(protein, ligand_t, t_i)"]
        EULER["Euler: x += dt * v\nor RK4: 4-stage update"]
        X1["x_1 (refined pose)"]
    end

    subgraph schedules["Timestep Schedules"]
        UNI["uniform: linspace(0,1,N)"]
        QUAD["quadratic: dense near t=1"]
        ROOT["root: t^(2/3)"]
        SIG["sigmoid: dense at endpoints"]
    end

    subgraph metrics["Metrics"]
        RMSD["RMSD to crystal"]
        SR2["Success Rate < 2.0 A"]
        SR1["Success Rate < 1.0 A"]
        SR05["Success Rate < 0.5 A"]
    end

    X0 --> STEP --> VEL --> EULER --> X1
    schedules --> STEP
    X1 --> metrics
```

---

## Tensor Dimensions Reference

### Feature Dimensions (Default Config)

| Component | Irreps / Dimension | Description |
|-----------|-------------------|-------------|
| Protein input (scalar) | 76 | Residue-level features |
| Protein input (vector) | 31 x 3 = 93 | Geometric vector features |
| Protein edge (scalar) | 39 | Edge scalar features |
| Protein edge (vector) | 8 x 3 = 24 | Edge vector features |
| Protein hidden | `128x0e + 32x1o + 32x1e` | Hidden representation |
| Protein output | `128x0e + 32x1o + 32x1e` | Encoded representation |
| Ligand input (scalar) | 121 | Atom-level features |
| Ligand edge (scalar) | 44 | Bond features |
| Ligand hidden | `128x0e + 16x1o + 16x1e` | Hidden representation |
| Ligand output | `128x0e + 16x1o + 16x1e` | Encoded representation |
| Interaction hidden_dim | 256 | Unified scalar dimension |
| Pair bias dim | 64 | Pairwise feature dimension |
| Attention heads | 8 | Multi-head attention |
| Velocity hidden | `128x0e + 16x1o + 16x1e` | Velocity network hidden |
| Velocity output | `1x1o` = 3 | Per-atom velocity vector |
| ESM-C embedding | 1152 | ESM-C 600M output |
| ESM-3 embedding | 1536 | ESM-3 output |

### Irreps Notation

| Symbol | Meaning |
|--------|---------|
| `Nx0e` | N scalar channels (even parity, l=0) |
| `Nx1o` | N vector channels (odd parity, l=1) - true vectors |
| `Nx1e` | N pseudo-vector channels (even parity, l=1) |
| `1x1o` | Single 3D vector - used for velocity output |

---

## Module Dependency Graph

```mermaid
graph TD
    FM["ProteinLigandFlowMatching\n(flowmatching.py)"]

    PN["UnifiedEquivariantNetwork\n(Protein)\n(network.py)"]
    LN["UnifiedEquivariantNetwork\n(Ligand)\n(network.py)"]
    IN["ProteinLigandInteractionNetwork\n(network.py)"]

    GEL["GatingEquivariantLayer\n(cue_layers.py)"]
    EMLP["EquivariantMLP\n(cue_layers.py)"]
    PBA["PairBiasAttentionLayer\n(cue_layers.py)"]
    ADALN_E["EquivariantAdaLN\n(cue_layers.py)"]
    EBN["EquivariantBatchNorm\n(cue_torch)"]

    MLP["MLP\n(torch_layers.py)"]
    ADALN["AdaLN\n(torch_layers.py)"]
    CTB["ConditionedTransitionBlock\n(torch_layers.py)"]
    SWIG["SwiGLU\n(torch_layers.py)"]

    CUE["cuequivariance_torch\nSphericalHarmonics\nFullyConnectedTensorProduct\nLinear\nattention_pair_bias"]

    FM --> PN
    FM --> LN
    FM --> IN
    FM --> GEL
    FM --> EMLP
    FM --> MLP

    PN --> GEL
    PN --> EMLP
    PN --> EBN

    LN --> GEL
    LN --> EMLP
    LN --> EBN

    IN --> EMLP
    IN --> PBA
    IN --> MLP

    GEL --> EMLP
    GEL --> ADALN_E
    GEL --> MLP
    GEL --> EBN
    GEL --> CUE

    EMLP --> CUE
    PBA --> CUE

    CTB --> ADALN
    CTB --> SWIG

    style FM fill:#ffcdd2,stroke:#B71C1C
    style CUE fill:#bbdefb,stroke:#0D47A1
    style GEL fill:#c8e6c9,stroke:#1B5E20
```

---

## File Structure

```
src/models/
  flowmatching.py          # ProteinLigandFlowMatching (top-level model)
  network.py               # UnifiedEquivariantNetwork, ProteinLigandInteractionNetwork
  cue_layers.py            # cuEquivariance layers (GatingEquivariantLayer, EquivariantMLP, etc.)
  torch_layers.py          # Pure PyTorch layers (MLP, AdaLN, SwiGLU, etc.)

src/utils/
  losses.py                # Distance geometry loss, clash loss
  sampling.py              # ODE integration (Euler, RK4), timestep schedules
  training_utils.py        # Optimizer, scheduler, timestep sampling
  model_builder.py         # Model construction from config
  data_utils.py            # Dataset/dataloader creation
  early_stop.py            # Early stopping with best model restoration

src/data/
  dataset.py               # FlowFixDataset (lazy/hybrid/preload modes)
  protein_feat.py          # Protein featurization + ESM embeddings
  ligand_feat.py           # Ligand featurization

train.py                   # FlowFixTrainer (main training loop)
inference.py               # Inference script
```

---

## Key Design Decisions

1. **Separate encoders + cross-attention** (not joint graph): Protein and ligand have different feature types (vectors vs scalar-only), processed separately then interact via attention.

2. **Time is implicit**: No explicit time embedding. Time information is encoded in `x_t` coordinates (linear interpolation position). The velocity field `v = x1 - x0` is time-independent for linear paths.

3. **cuEquivariance for SE(3)**: Uses NVIDIA's cuEquivariance library for hardware-accelerated equivariant operations (tensor products, spherical harmonics, attention).

4. **Zero-initialized velocity output**: Final velocity projection is zero-initialized with learnable scale (init=0.1) for stable early training.

5. **Dual conditioning in velocity blocks**: Each GatingEquivariantLayer receives atom-level conditioning via EquivariantAdaLN at both input (before message passing) and output (after batch norm).

6. **Multiple timesteps per sample**: Each PDB system is evaluated at `num_timesteps_per_sample` different timesteps per training step for efficiency.
