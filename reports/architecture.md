# FlowFix Model Architecture

> **SE(3)-Equivariant Flow Matching for Protein-Ligand Pose Refinement**
>
> Last updated: 2026-03-06
>
> Trained model: `rectified-flow-full-v4` (joint graph, 8-layer, ~13M params)

---

## 1. Overview

FlowFix는 docking pose를 crystal structure로 refinement하는 SE(3)-equivariant flow matching 모델입니다.
Linear interpolation path `x_t = (1-t)*x0 + t*x1`을 따라 per-atom velocity field `v(x_t, t)`를 학습합니다.

### Pipeline Summary

```mermaid
flowchart LR
    subgraph input["Input"]
        X0["Docked Pose<br/>x_0 (t=0)"]
    end

    subgraph model["FlowFix (Joint Graph)"]
        direction TB
        PREP["ESM + Time<br/>Preprocessing"]
        JOINT["Joint Graph<br/>8x Message Passing"]
        VEL["Velocity<br/>Prediction"]
        PREP --> JOINT --> VEL
    end

    subgraph output["Output"]
        X1["Refined Pose<br/>x_1 (t=1)"]
    end

    X0 -->|"ODE Integration<br/>(20-step Euler)"| model -->|"v(x_t, t)"| X1
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Graph structure | Joint protein-ligand graph | Cross-edge로 protein context 직접 전달 |
| Equivariance | cuEquivariance tensor product | SE(3) symmetry 보존, GPU-accelerated |
| Interaction | Direct message passing (no attention) | 단순하고 효율적인 protein-ligand interaction |
| Time conditioning | Explicit sinusoidal + AdaLN | 각 layer에서 time-aware feature modulation |
| Optimizer | Muon + AdamW hybrid | 2D weight matrix에 Muon, 나머지 AdamW |
| Generative model | Flow matching (rectified flow) | Stable training, fast sampling |
| Protein embedding | ESMC 600M + ESM3 (weighted) | Pre-trained sequence representation |

---

## 2. Model Architecture

### 2.1 Full Pipeline

```mermaid
flowchart TB
    subgraph stage1["Preprocessing"]
        P["Protein Graph<br/>scalar: 76d, vector: 31x3d<br/>edge scalar: 39d, edge vector: 8x3d"]
        L["Ligand Graph<br/>scalar: 122d<br/>edge: 44d"]
        T["timestep t"]

        ESM["ESM Integration<br/>ESMC 600M (1152d) + ESM3 (1536d)<br/>softmax weighted sum<br/>gated concat to protein.x"]

        TIME["Time Embedding<br/>sinusoidal -> MLP<br/>-> condition [B, 384]"]

        P --> ESM
        T --> TIME
    end

    subgraph stage2["Joint Graph Construction"]
        PPROJ["protein_node_proj<br/>EquivariantMLP<br/>-> 192x0e + 48x1o + 48x1e"]
        LPROJ["ligand_node_proj<br/>EquivariantMLP<br/>-> 192x0e + 48x1o + 48x1e"]

        EDGES["4 Edge Types<br/>PP: pre-computed protein edges<br/>LL: pre-computed ligand bonds<br/>LL intra: dynamic radius graph<br/>PL cross: dynamic radius graph<br/>(cutoff=6.0A, max_neighbors=16)"]

        MERGE["Joint Graph<br/>joint_h = cat(prot_h, lig_h)<br/>+ merged edge_index"]

        ESM --> PPROJ --> MERGE
        L --> LPROJ --> MERGE
        EDGES --> MERGE
    end

    subgraph stage3["Message Passing (x8)"]
        LAYER["GatingEquivariantLayer<br/>pre_norm -> AdaLN(time)<br/>-> TensorProduct message<br/>-> scalar/vector gating<br/>-> scatter sum + self-interaction<br/>-> node_update -> post_adaln(time)<br/>-> gated skip connection"]
    end

    subgraph stage4["Velocity Output"]
        EXTRACT["Extract ligand nodes<br/>joint_h[N_p:]"]
        VOUT["velocity_output<br/>EquivariantMLP (2-layer)<br/>hidden_irreps -> 1x1o (3D)"]
        V["velocity [N_ligand, 3]"]
    end

    TIME --> LAYER
    MERGE --> LAYER
    LAYER --> EXTRACT --> VOUT --> V

    style stage1 fill:#e8f5e9,stroke:#2E7D32
    style stage2 fill:#fff3e0,stroke:#E65100
    style stage3 fill:#fce4ec,stroke:#C62828
    style stage4 fill:#f3e5f5,stroke:#6A1B9A
```

---

### 2.2 GatingEquivariantLayer (Core Block)

8번 반복되는 SE(3)-equivariant message passing layer. Joint graph의 모든 node (protein + ligand) 위에서 동작.

```mermaid
flowchart TB
    subgraph input["Input"]
        H["node_features [N, irreps]"]
        TC["time_condition [N, 384]"]
    end

    PRE["pre_norm (BatchNorm)"]
    ADALN1["context_adaln<br/>scalar: LN + scale/bias from time<br/>vector: norm-based gating"]

    subgraph mp["Message Passing"]
        SH["Spherical Harmonics<br/>Y_l(r_ij), l=0,1,2"]
        EDGE["edge_embedding<br/>MLP(edge_attr)"]
        TP["tp_message<br/>TensorProduct<br/>node[src] x (edge_emb, SH)"]
        MSG["message_mlp"]
        NP["node_pair_proj<br/>(edge importance)"]
        SG["scalar_gate_net<br/>sigmoid"]
        VG["vector_gate_net<br/>norm-based sigmoid"]
        AGG["Scatter Sum"]
    end

    SELF["tp_self<br/>TensorProduct(node, ones)"]

    ADD["aggregated + self_update"]
    NMLP["node_update_mlp"]
    ADALN2["post_adaln<br/>Time-conditioned AdaLN"]
    SKIP["skip_gate<br/>gate * update + (1-gate) * identity"]

    H --> PRE --> ADALN1
    TC --> ADALN1
    ADALN1 --> TP --> MSG --> NP --> SG --> VG --> AGG --> ADD
    ADALN1 --> SELF --> ADD
    SH --> TP
    EDGE --> TP
    ADD --> NMLP --> ADALN2 --> SKIP
    TC --> ADALN2

    subgraph output["Output"]
        OUT["updated features [N, irreps]"]
    end
    SKIP --> OUT

    style mp fill:#fff8e1,stroke:#F9A825
```

**핵심 특징:**
- **Tensor Product**: `cuequivariance_torch.FullyConnectedTensorProduct`로 SE(3) equivariance 보존
- **Dual Gating**: Scalar gate (element-wise sigmoid) + Vector gate (norm-based adaptive sigmoid)
- **Edge Importance**: src/dst scalar features로 학습 가능한 message weight
- **Dual AdaLN**: Input/Output 양쪽에서 time conditioning
- **Gated Skip**: 학습 가능한 gate로 identity와 update 비율 결정

---

### 2.3 ESM Embedding Integration

Pre-trained PLM의 residue-level embedding을 protein features에 통합.

```mermaid
flowchart LR
    ESMC["ESMC 600M<br/>[N, 1152]"] --> P1["MLP<br/>1152->128->128"]
    ESM3["ESM3<br/>[N, 1536]"] --> P2["MLP<br/>1536->128->128"]

    P1 --> WS["Weighted Sum<br/>softmax(esm_weight)"]
    P2 --> WS

    WS --> GATE["esm_gate<br/>sigmoid MLP"]

    PX["protein.x [N,76]"] --> CAT["Gated Concat"]
    GATE --> CAT
    CAT --> OUT["[N, 76+128]"]
```

- ESM weight: learnable `nn.Parameter`, softmax 정규화
- Gated concatenation: sigmoid gate로 ESM feature 영향도 학습

---

### 2.4 Time Conditioning

```mermaid
flowchart LR
    T["t [B]"] --> SIN["Sinusoidal<br/>sin/cos"]
    SIN --> TE["time_embedding<br/>MLP: D->4D->D"]
    TE --> TC["time_to_condition<br/>MLP -> 384d"]
    TC --> EXP["Expand<br/>[B,384] -> [N,384]"]
    EXP --> L["AdaLN in<br/>each of 8 layers"]
```

---

## 3. Training

### 3.1 Flow Matching Loss

```mermaid
flowchart LR
    subgraph sample["Sampling"]
        X0["x_0 (docked)"]
        X1["x_1 (crystal)"]
        T["t ~ Uniform(0,1)"]
    end

    INTERP["x_t = (1-t)*x0 + t*x1"]
    FWD["v_pred = model(prot, lig_t, t)"]
    VT["v_true = x1 - x0"]
    MSE["L_flow = MSE(v_pred, v_true)"]
    DG["L_dg = dist geometry<br/>bond constraints<br/>time-weighted"]
    TOTAL["L = L_flow + 0.1 * L_dg"]

    X0 --> INTERP
    X1 --> INTERP
    T --> INTERP
    INTERP --> FWD --> MSE --> TOTAL
    VT --> MSE
    DG --> TOTAL
```

### 3.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Joint graph (8 layers) |
| Hidden (scalar/vector) | 192 / 48 |
| Hidden irreps | `192x0e + 48x1o + 48x1e` (480d) |
| Edge cutoff | 6.0 A, max 16 neighbors |
| Optimizer | Muon (lr=0.005) + AdamW (lr=3e-4) |
| Schedule | Warmup 5% + Plateau 80% + Cosine 15% |
| Loss | Velocity MSE + DG loss (weight=0.1) |
| EMA | decay=0.999 (inference에 사용) |
| Batch size | 32 |
| Epochs | 500 |
| Dropout | 0.1 |
| ODE sampling | 20-step Euler, uniform schedule |

### 3.3 ODE Sampling (Inference)

```mermaid
flowchart LR
    X0["x_0 (docked)"] --> LOOP

    subgraph LOOP["Euler Integration (20 steps)"]
        direction TB
        V["v = model(prot, lig_t, t_i)"]
        U["x += dt * v"]
        V --> U
    end

    LOOP --> X1["x_1 (refined)"]
    X1 --> M["RMSD<br/>Success Rate"]
```

---

## 4. Dimension Reference

```
Protein:
  Node scalar:  76   ─┐
  Node vector:  31x3  │  + ESM 128d (gated concat)
  Edge scalar:  39    ├─> protein_node_proj ──> 192x0e + 48x1o + 48x1e
  Edge vector:  8x3   ┘

Ligand:
  Node scalar:  122 ─┐
  Edge scalar:  44   ├─> ligand_node_proj ──> 192x0e + 48x1o + 48x1e
                     ┘

Joint Graph:
  Nodes:  N_p + N_l, each 480d (hidden_irreps)
  Edges:  PP + LL + LL_intra + PL_cross, each 192d
  Time:   384d condition via AdaLN

Output:
  Velocity: 1x1o = 3d per ligand atom
```

### Irreps Quick Reference

| Symbol | Meaning | Flat dim |
|--------|---------|----------|
| `192x0e` | 192 scalars (even parity) | 192 |
| `48x1o` | 48 true vectors (odd parity) | 144 |
| `48x1e` | 48 pseudo-vectors (even parity) | 144 |
| Total hidden | `192x0e + 48x1o + 48x1e` | 480 |

---

## 5. Module Structure (from checkpoint)

```
Top-level:
  esm_weight                    # [2] learnable ESMC/ESM3 weights
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
      edge_embedding            #   MLP
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

## 6. File Map

```
src/models/
  flowmatching.py    Model definitions
  network.py         Network components
  cue_layers.py      GatingEquivariantLayer, EquivariantMLP, ...
  torch_layers.py    MLP, AdaLN, SwiGLU, TimeEmbedding, ...

src/data/
  dataset.py         FlowFixDataset

src/utils/
  losses.py          Distance geometry loss, clash loss
  sampling.py        ODE integration, timestep schedules
  model_builder.py   Config -> model construction

configs/
  train_rectified_flow_full.yaml   # v4 config (trained model)
  train_joint.yaml                 # Joint architecture template

train.py             Training loop
inference.py         Inference script
```
