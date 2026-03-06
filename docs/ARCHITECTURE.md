# FlowFix Architecture Diagram

## Visual Diagram

![FlowFix Architecture](../assets/flowfix_architecture_diagram.png)

## Mermaid Flowchart (editable)

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        P["Protein Batch<br/>(pos, x, edges)"]
        L["Ligand Batch<br/>(pos, x_t, edges)"]
        T["t (time)"]
        E["cross_edge_index<br/>intra_edge_index"]
        SC["x1_self_cond (optional)"]
    end

    subgraph prep["Preprocessing"]
        ESM["ESM Integration<br/>ESMC + ESM3 → project → gate → concat<br/>protein.x: 76 → 76+128"]
        SELF["Self-Conditioning (50%)<br/>x1_pred → MLP → + ligand.x"]
        TIME["Time Embedding<br/>t → sinusoidal → MLP<br/>time_condition [B, 256]"]
    end

    subgraph joint_in["Joint Network Input"]
        PN["Protein Node Proj<br/>EquivariantMLP<br/>scalar+vector → 128x0e+32x1o"]
        LN["Ligand Node Proj<br/>EquivariantMLP<br/>scalar → 128x0e+32x1o"]
    end

    subgraph graph["Joint Graph"]
        NODES["joint_h = concat(protein_h, ligand_h)<br/>joint_pos, joint_batch"]
        EDGES["4 edge types:<br/>PP (pre) | LL (pre) | LL_intra | PL_cross"]
    end

    subgraph mp["Message Passing (×6)"]
        LAYER["GatingEquivariantLayer<br/>PreNorm → AdaLN(t) → EdgeEmb(RBF, node_pair)<br/>→ TP message → gate → scatter<br/>→ self+node_update → PostAdaLN → gated skip"]
    end

    subgraph out["Output"]
        EXTRACT["Extract ligand slice<br/>joint_h[N_p:]"]
        VEL["Velocity Output<br/>EquivariantMLP (2-layer, zero-init)"]
        V["Velocity [N_ligand, 3]"]
    end

    P --> ESM
    L --> SELF
    SC --> SELF
    T --> TIME
    ESM --> PN
    L --> SELF
    SELF --> LN
    T --> TIME
    PN --> NODES
    LN --> NODES
    E --> EDGES
    NODES --> LAYER
    EDGES --> LAYER
    TIME --> LAYER
    LAYER --> EXTRACT
    EXTRACT --> VEL
    VEL --> V
```

## Data Flow Summary

| Stage | Input | Output |
|-------|--------|--------|
| ESM | protein.x [N,76], esmc [N,1152], esm3 [N,1536] | protein.x [N, 76+128] |
| Self-cond | ligand.x, x1_pred [N_l,3] | ligand.x + gate(x1_pred) |
| Time | t [B] | time_condition [B, 256] |
| Node proj | protein/ligand features | joint_h [N_p+N_l, hidden_irreps] |
| Edges | pos, batch, pre-computed edges | joint_edge_index, joint_edge_attr (4 types) |
| 6× Layer | joint_h, pos, edges, time_condition | updated joint_h |
| Velocity head | ligand_h | velocity [N_l, 3] |

## Edge Types

| Type | Source | Features |
|------|--------|----------|
| PP | Protein–protein (pre-computed) | EquivariantMLP(edge_attr + edge_vector) |
| LL | Ligand–ligand bonds (pre-computed) | EquivariantMLP(edge_attr) |
| LL_intra | Ligand radius_graph (dynamic) | MLP(type_onehot), RBF inside layer |
| PL_cross | Protein–ligand radius (dynamic) | MLP(type_onehot), RBF inside layer |
