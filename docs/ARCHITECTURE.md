# FlowFix Architecture Documentation

## Overview

FlowFix refines perturbed protein-ligand binding poses back to crystal structures using **SE(3)-equivariant flow matching**. The model learns velocity fields that move ligand atoms from docked poses (t=0) to crystal structures (t=1) via linear interpolation.

**Trained model**: `rectified-flow-full-v4` (joint graph, 8-layer, ~13M params)

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        P["Protein Graph<br/>(scalar 76d + vector 31x3d)"]
        L["Ligand Graph<br/>(scalar 122d)"]
        T["t (timestep)<br/>in 0..1"]
    end

    subgraph preprocess["Preprocessing"]
        ESM["ESM Integration<br/>ESMC 600M + ESM3<br/>weighted projection<br/>gated concat to protein.x"]
        TIME["Time Embedding<br/>sinusoidal -> MLP<br/>-> condition [B, 384]"]
    end

    subgraph joint["Joint Graph Construction"]
        PPROJ["Protein Node Proj<br/>EquivariantMLP<br/>76x0e+31x1o -> hidden_irreps"]
        LPROJ["Ligand Node Proj<br/>EquivariantMLP<br/>122x0e -> hidden_irreps"]
        EDGES["4 Edge Types:<br/>PP (pre-computed)<br/>LL bonds (pre-computed)<br/>LL intra (dynamic radius)<br/>PL cross (dynamic radius)"]
        MERGE["Merge into Joint Graph<br/>joint_h = cat(protein_h, ligand_h)<br/>joint_pos, joint_edge_index"]
    end

    subgraph mp["Message Passing (x8)"]
        LAYER["GatingEquivariantLayer<br/>+ time AdaLN conditioning<br/>+ skip connection"]
    end

    subgraph output["Output"]
        EXTRACT["Extract ligand slice<br/>joint_h[N_p:]"]
        VEL["Velocity Output<br/>EquivariantMLP (2-layer)<br/>hidden_irreps -> 1x1o"]
        V["velocity [N_ligand, 3]"]
    end

    P --> ESM --> PPROJ
    L --> LPROJ
    T --> TIME
    PPROJ --> MERGE
    LPROJ --> MERGE
    EDGES --> MERGE
    MERGE --> LAYER
    TIME --> LAYER
    LAYER --> EXTRACT --> VEL --> V

    style inputs fill:#e8f4f8,stroke:#2196F3
    style preprocess fill:#e8f5e9,stroke:#4CAF50
    style joint fill:#fff3e0,stroke:#FF9800
    style mp fill:#fce4ec,stroke:#E91E63
    style output fill:#f3e5f5,stroke:#9C27B0
```

---

## Detailed Module Diagrams

### 1. Joint Graph Construction

Protein과 ligand를 하나의 그래프로 합쳐서 direct message passing.

```mermaid
flowchart TB
    subgraph protein["Protein"]
        PF["Node: scalar 76d + vector 31x3d<br/>Edge: scalar 39d + vector 8x3d<br/>Pre-computed graph"]
    end

    subgraph ligand["Ligand"]
        LF["Node: scalar 122d<br/>Edge: scalar 44d<br/>Pre-computed bond graph"]
    end

    subgraph proj["Input Projections"]
        PP["protein_node_proj<br/>EquivariantMLP<br/>76x0e + 31x1o -> 192x0e + 48x1o + 48x1e"]
        LP["ligand_node_proj<br/>EquivariantMLP<br/>122x0e -> 192x0e + 48x1o + 48x1e"]
    end

    subgraph edges["Edge Processing"]
        PPE["pp_edge_proj<br/>EquivariantMLP<br/>protein edge features -> 192d"]
        LLE["ll_edge_proj<br/>MLP<br/>ligand bond features -> 192d"]
        DYN["dynamic_edge_proj<br/>MLP(type_onehot) -> 192d<br/>+ RBF distance inside layer<br/>PL cross + LL intra"]
    end

    subgraph merge["Joint Graph"]
        JH["joint_h [N_p+N_l, irreps]"]
        JP["joint_pos [N_p+N_l, 3]"]
        JE["joint_edge_index<br/>(PP + LL + LL_intra + PL_cross)"]
    end

    PF --> PP --> JH
    LF --> LP --> JH
    PF --> PPE --> JE
    LF --> LLE --> JE
    DYN --> JE
    PP --> JP
    LP --> JP

    style edges fill:#fff8e1,stroke:#F9A825
    style merge fill:#e3f2fd,stroke:#1565C0
```

**Dynamic edge construction** (every forward):
- `PL cross`: protein-ligand radius graph (cutoff=6.0A, max_neighbors=16)
- `LL intra`: ligand intra-molecular radius graph (cutoff=6.0A, supplements bond edges)

---

### 2. GatingEquivariantLayer (Core Block)

Joint graph 위에서 8번 반복되는 SE(3)-equivariant message passing layer.

```mermaid
flowchart TB
    subgraph input["Input"]
        H["node_features [N, irreps]"]
        POS["positions [N, 3]"]
        EI["edge_index [2, E]"]
        EA["edge_attr [E, edge_dim]"]
        TC["time_condition [N, 384]"]
    end

    subgraph norm["Pre-Normalization"]
        PRE["pre_norm<br/>EquivariantBatchNorm"]
    end

    subgraph conditioning["Time Conditioning"]
        ADALN["context_adaln<br/>scalar: LayerNorm + scale/bias<br/>vector: norm-based gating"]
    end

    subgraph edgeproc["Edge Processing"]
        SH["Spherical Harmonics<br/>Y_l(r_ij), l=0,1,2"]
        EE["edge_embedding<br/>MLP(edge_attr) -> hidden"]
        EF["edge_features =<br/>cat(edge_emb, edge_sh)"]
    end

    subgraph message["Message Passing"]
        TP["tp_message<br/>TensorProduct<br/>node[src] x edge_features"]
        MSG["message_mlp<br/>EquivariantMLP"]
        NP["node_pair_proj<br/>Edge importance from<br/>src_scalar, dst_scalar"]
        SG["scalar_gate_net<br/>sigmoid gating"]
        VG["vector_gate_net<br/>norm-based adaptive gating"]
        AGG["Scatter Sum<br/>[E] -> [N]"]
    end

    subgraph self_int["Self-Interaction"]
        TP_SELF["tp_self<br/>TensorProduct(node, ones)"]
    end

    subgraph update["Node Update"]
        ADD["aggregated + self_update"]
        NMLP["node_update_mlp<br/>EquivariantMLP"]
        POST["post_adaln<br/>Time-conditioned AdaLN"]
        GATE["skip_gate<br/>Learnable gating"]
        SKIP["output = gate * update<br/>+ (1-gate) * identity"]
    end

    H --> PRE --> ADALN
    TC --> ADALN
    ADALN --> TP
    ADALN --> TP_SELF
    POS --> SH --> EF
    EA --> EE --> EF
    EF --> TP
    TP --> MSG --> NP --> SG --> VG --> AGG
    TP_SELF --> ADD
    AGG --> ADD
    ADD --> NMLP --> POST --> GATE --> SKIP
    TC --> POST

    subgraph output["Output"]
        OUT["updated node_features [N, irreps]"]
    end
    SKIP --> OUT

    style conditioning fill:#e3f2fd,stroke:#1565C0
    style message fill:#fff8e1,stroke:#F9A825
    style self_int fill:#f1f8e9,stroke:#558B2F
    style update fill:#fce4ec,stroke:#C62828
```

---

### 3. ESM Embedding Integration

```mermaid
flowchart LR
    ESMC["ESMC 600M<br/>[N, 1152]"] --> P1["esmc_projection<br/>MLP: 1152->128->128"]
    ESM3["ESM3<br/>[N, 1536]"] --> P2["esm3_projection<br/>MLP: 1536->128->128"]

    P1 --> WS["Weighted Sum<br/>softmax(esm_weight)"]
    P2 --> WS

    WS --> GATE["esm_gate<br/>MLP: 128->128<br/>sigmoid gating"]

    PX["protein.x [N, 76]"] --> CAT["Gated Concat<br/>[N, 76 + 128]"]
    GATE --> CAT
    CAT --> OUT["enhanced protein.x<br/>[N, 76 + 128]"]
```

---

### 4. Time Conditioning

```mermaid
flowchart LR
    T["t [B]"] --> SIN["Sinusoidal Embedding<br/>sin/cos frequencies"]
    SIN --> PROJ["time_embedding<br/>MLP: D -> 4D -> D"]
    PROJ --> COND["time_to_condition<br/>MLP -> hidden_dim (384)"]
    COND --> EXP["Expand to nodes<br/>[B, 384] -> [N, 384]<br/>via batch indices"]
    EXP --> ADALN["Applied via AdaLN<br/>in each of 8 layers"]
```

---

## Training Pipeline

### Flow Matching

```mermaid
flowchart TB
    subgraph data["Data"]
        DS["FlowFixDataset<br/>1 random pose per PDB per epoch<br/>~18k PDBs x 60 poses"]
    end

    subgraph train["Training Step"]
        TS["Sample Timesteps<br/>Uniform"]
        INTERP["x_t = (1-t)*x0 + t*x1"]
        FWD["v_pred = model(protein, ligand_t, t)"]
        VT["v_true = x1 - x0"]
        MSE["L_flow = MSE(v_pred, v_true)"]
        DG["L_dg = distance geometry<br/>time-weighted"]
        TOTAL["L = L_flow + 0.1 * L_dg"]
    end

    subgraph opt["Optimization"]
        MUON["Muon (lr=0.005)<br/>2D weights"]
        ADAM["AdamW (lr=3e-4)<br/>1D, bias, norm"]
        EMA["EMA decay=0.999"]
    end

    DS --> TS --> INTERP --> FWD --> MSE --> TOTAL
    VT --> MSE
    DG --> TOTAL
    TOTAL --> MUON
    TOTAL --> ADAM
    FWD --> EMA

    style opt fill:#e8f5e9,stroke:#2E7D32
```

### ODE Sampling (Inference)

```mermaid
flowchart LR
    X0["x_0 (docked)"] --> LOOP

    subgraph LOOP["ODE Integration (20 steps)"]
        direction TB
        VEL["v = model(protein, ligand_t, t_i)"]
        EU["Euler: x += dt * v"]
        VEL --> EU
    end

    LOOP --> X1["x_1 (refined)"]
    X1 --> M["RMSD, Success Rate"]
```

---

## Dimension Reference (v4)

| Component | Value | Description |
|-----------|-------|-------------|
| Protein node scalar | 76 | Residue features |
| Protein node vector | 31 x 3 | Geometric vectors |
| Protein edge scalar | 39 | Edge features |
| Protein edge vector | 8 x 3 | Edge vectors |
| Ligand node scalar | 122 | Atom features |
| Ligand edge scalar | 44 | Bond features |
| Hidden scalar | 192 | Joint hidden |
| Hidden vector | 48 | Joint hidden |
| Hidden edge | 192 | Edge embedding |
| Hidden irreps | `192x0e + 48x1o + 48x1e` | 480d total |
| Time condition | 384 | AdaLN conditioning |
| ESM projection | 128 | ESM feature dim |
| Cross-edge cutoff | 6.0 A | PL dynamic edges |
| RBF features | 32 | Distance encoding |
| Max neighbors | 16 | KNN cap |
| Joint layers | 8 | Message passing depth |
| Velocity output | 1x1o = 3 | Per-atom velocity |

---

## Module Structure (from checkpoint)

```
Top-level:
  esm_weight                    # Learnable [ESMC, ESM3] weights
  esmc_projection               # MLP: 1152 -> 128
  esm3_projection               # MLP: 1536 -> 128
  esm_gate                      # MLP: 128 -> 128 (sigmoid)
  time_embedding                # Sinusoidal + MLP
  time_to_condition             # MLP -> 384d

  joint_network/
    protein_node_proj           # EquivariantMLP
    ligand_node_proj            # EquivariantMLP
    pp_edge_proj                # EquivariantMLP (protein edges)
    ll_edge_proj                # MLP (ligand bond edges)
    dynamic_edge_proj           # MLP (PL cross + LL intra)

    layers[0..7]/               # 8x GatingEquivariantLayer
      pre_norm                  #   EquivariantBatchNorm
      context_adaln             #   Time AdaLN (input)
      edge_embedding            #   MLP(edge_attr)
      tp_message                #   TensorProduct (message)
      message_mlp               #   EquivariantMLP
      node_pair_proj            #   Edge importance
      scalar_gate_net           #   Scalar gating
      vector_norm_net           #   Vector norm features
      vector_gate_net           #   Vector gating
      tp_self                   #   TensorProduct (self)
      node_update_mlp           #   EquivariantMLP
      post_adaln                #   Time AdaLN (output)
      skip_gate                 #   Learnable skip
      rbf_centers, rbf_width    #   RBF params

    velocity_output             # EquivariantMLP -> 1x1o
```

**716 parameter tensors, ~13M trainable parameters**

---

## Training Config (v4)

| Parameter | Value |
|-----------|-------|
| Architecture | Joint graph |
| Joint layers | 8 |
| Hidden (scalar/vector) | 192 / 48 |
| Edge cutoff | 6.0 A |
| Optimizer | Muon (lr=0.005) + AdamW (lr=3e-4) |
| Schedule | Warmup 5% + Plateau 80% + Cosine 15% |
| Loss | Velocity MSE + DG (0.1) |
| EMA | decay=0.999 |
| Batch size | 32 |
| Epochs | 500 |
| Dropout | 0.1 |

---

## Key Design Decisions

1. **Joint graph** (not separate encoders + attention): Protein-ligand interaction via direct message passing on unified graph. Cross-edges propagate protein context to ligand nodes.

2. **4 edge types**: Pre-computed (PP, LL bonds) + dynamic (PL cross, LL intra). Dynamic edges recomputed every forward via `radius_graph`.

3. **Explicit time conditioning**: Sinusoidal -> MLP -> AdaLN at input and output of each layer.

4. **Muon + AdamW hybrid**: Muon for 2D weight matrices, AdamW for 1D params/bias/norm.

5. **EMA**: decay=0.999, used for inference.

6. **Gated skip connections**: Learnable gate balances new features vs identity, stabilizes deep (8-layer) network.
