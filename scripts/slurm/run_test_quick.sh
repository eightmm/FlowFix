#!/bin/bash
#SBATCH --job-name=flowfix_test
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_test_%j.out
#SBATCH --error=logs/slurm_test_%j.out

source ~/.bashrc
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine

# Quick test with small data
python -c "
import torch
from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from src.models.flowmatching import ProteinLigandFlowMatchingJoint
from src.utils.model_builder import build_model
from torch.utils.data import DataLoader
import yaml

print('='*60)
print('Testing Dataset + Model Integration')
print('='*60)

# Load config
with open('configs/train_joint.yaml') as f:
    config = yaml.safe_load(f)

# Create small dataset
print('\\n1. Creating dataset...')
ds = FlowFixDataset(
    data_dir='train_data',
    split='train',
    max_samples=4,
    cross_edge_cutoff=6.0,
    cross_edge_max_neighbors=16,
    intra_edge_cutoff=6.0,
    intra_edge_max_neighbors=16,
)
ds.set_epoch(0)
print(f'   Dataset size: {len(ds)}')

# Create dataloader
print('\\n2. Creating dataloader...')
loader = DataLoader(ds, batch_size=2, collate_fn=collate_flowfix_batch, num_workers=0)
batch = next(iter(loader))
print(f'   Batch keys: {list(batch.keys())}')
print(f'   t shape: {batch[\"t\"].shape}')
print(f'   cross_edge_index shape: {batch[\"cross_edge_index\"].shape}')
print(f'   intra_edge_index shape: {batch[\"intra_edge_index\"].shape}')

# Create model
print('\\n3. Building model...')
model = build_model(config['model']).cuda()
print(f'   Model type: {type(model).__name__}')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Move batch to GPU
print('\\n4. Moving batch to GPU...')
protein_batch = batch['protein_graph'].to('cuda')
ligand_batch = batch['ligand_graph'].to('cuda')
t = batch['t'].to('cuda')
cross_edge_index = batch['cross_edge_index'].to('cuda')
intra_edge_index = batch['intra_edge_index'].to('cuda')
ligand_coords_x0 = batch['ligand_coords_x0'].to('cuda')
ligand_coords_x1 = batch['ligand_coords_x1'].to('cuda')

# Forward pass
print('\\n5. Forward pass...')
velocity = model(protein_batch, ligand_batch, t, cross_edge_index, intra_edge_index)
print(f'   Velocity shape: {velocity.shape}')
print(f'   Expected: {ligand_batch.pos.shape}')

# Self-conditioning test
print('\\n6. Self-conditioning test...')
t_expanded = t[ligand_batch.batch].unsqueeze(-1)
x1_self_cond = ligand_batch.pos + (1 - t_expanded) * velocity
velocity2 = model(protein_batch, ligand_batch, t, cross_edge_index, intra_edge_index, x1_self_cond=x1_self_cond)
print(f'   Velocity2 shape: {velocity2.shape}')

# Loss computation
print('\\n7. Loss computation...')
true_velocity = ligand_coords_x1 - ligand_coords_x0
loss = torch.mean((velocity - true_velocity) ** 2)
print(f'   Loss: {loss.item():.6f}')

# Backward pass
print('\\n8. Backward pass...')
loss.backward()
print('   Backward: OK')

# Validation mode (no edges passed - should use fallback)
print('\\n9. Validation mode (edge fallback)...')
model.eval()
with torch.no_grad():
    velocity_val = model(protein_batch, ligand_batch, t)  # No edges - uses fallback
print(f'   Velocity shape: {velocity_val.shape}')

print('\\n' + '='*60)
print('All tests PASSED!')
print('='*60)
"
