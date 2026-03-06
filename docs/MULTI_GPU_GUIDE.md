# Multi-GPU Training Guide for FlowFix

This guide explains how to use the multi-GPU training implementation with PyTorch DistributedDataParallel (DDP).

## Overview

FlowFix now supports multi-GPU training using PyTorch's DistributedDataParallel (DDP). This allows you to:
- **Scale training** to multiple GPUs on a single node
- **Reduce training time** by distributing batches across GPUs
- **Maintain training quality** with synchronized gradient updates

## Files

- `train.py`: Multi-GPU training entrypoint (DDP 지원)
- `scripts/slurm/run_train_joint.sh`: Slurm script for multi-GPU training (예: 8 GPUs)
- `scripts/slurm/run_train_joint_test.sh`: Slurm script for a quick single-GPU smoke test
- `configs/train_joint.yaml`: Training configuration used by the Slurm scripts

## Key Features

### 1. **Distributed Data Parallelism (DDP)**
- Automatic gradient synchronization across all GPUs
- Each GPU processes a portion of the batch independently
- Model parameters are synchronized after each backward pass

### 2. **Efficient Data Loading**
- `DistributedSampler` ensures each GPU gets unique data samples
- No data duplication across GPUs
- Automatic epoch shuffling with proper seeding

### 3. **Smart Process Management**
- Only rank 0 (main process) handles:
  - WandB logging
  - Checkpoint saving
  - Visualization generation
  - Console output
- All processes participate in:
  - Training
  - Validation
  - Gradient computation

### 4. **Synchronized Validation**
- Metrics are gathered from all GPUs using `all_gather`
- Aggregated results computed on rank 0
- Early stopping decisions broadcasted to all processes

## Usage

### Quick Start

#### **Test Run** (Recommended First)
```bash
sbatch scripts/slurm/run_train_joint_test.sh
```

This test script will:
- Run a quick single-GPU job to verify the environment and data pipeline
- Use `configs/train_joint.yaml` (edit as needed)

#### **Full Training**
```bash
sbatch scripts/slurm/run_train_joint.sh
```

This full training script will:
- Use all available training data (~18,000 PDBs)
- Train for 5000 epochs
- Request 8 GPUs on a single node (6000ada_verylong partition)
- Automatically set up DDP environment variables

### Custom GPU Count

You can modify the slurm script to use different GPU counts:

```bash
#SBATCH --gres=gpu:N                   # Request N GPUs
#SBATCH --cpus-per-task=M              # M = 8 * N (8 CPUs per GPU)
```

Then update the distributed run command:
```bash
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=N \
    train.py \
    --config configs/train_joint.yaml
```

### Interactive Multi-GPU Training

For debugging or testing, you can run multi-GPU training interactively:

```bash
# Request an interactive session with 2 GPUs
srun --partition=6000ada --gres=gpu:2 --cpus-per-task=16 --pty bash

# Navigate to project directory
cd /home/jaemin/project/protein-ligand/pose-refine

# Set Python path
PYTHON=/home/jaemin/miniforge3/envs/torch-2.8/bin/python

# Run with torch.distributed.run
$PYTHON -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    train.py --config configs/train_joint.yaml
```

## Configuration

### Batch Size Per GPU

The effective batch size is automatically split across GPUs:

```python
# In config file
training:
  batch_size: 64  # Total batch size across all GPUs

# With 8 GPUs: each GPU processes 64/8 = 8 samples
# With 4 GPUs: each GPU processes 64/4 = 16 samples
# With 2 GPUs: each GPU processes 64/2 = 32 samples
```

**Note**: The batch size in your config file is the **total** batch size across all GPUs.

**Recommendation**: Increase total batch size when using more GPUs:
```yaml
# 8 GPUs: 256 total (32 per GPU)
# 4 GPUs: 128 total (32 per GPU)
# 2 GPUs: 64 total (32 per GPU)
```

### Learning Rate Scaling

**No learning rate scaling is needed** for DDP because:
- Each GPU processes a smaller batch (total_batch / num_gpus)
- Gradients are averaged (not summed) across GPUs
- This is equivalent to the single-GPU case with the same total batch size

### Data Workers

Adjust workers per GPU based on available CPUs:

```yaml
data:
  num_workers: 4  # Per GPU

# With 4 GPUs: 4 workers * 4 GPUs = 16 total workers
# Ensure SLURM allocates enough CPUs: --cpus-per-task=64 (16 workers * 4 threads)
```

## Environment Variables

The slurm scripts set important NCCL environment variables:

```bash
export NCCL_DEBUG=INFO                 # Debug output for troubleshooting
export NCCL_IB_DISABLE=0               # Enable InfiniBand (if available)
export NCCL_NET_GDR_LEVEL=2            # GPU Direct RDMA
export NCCL_P2P_LEVEL=NVL              # Enable NVLink (if available)
export NCCL_BLOCKING_WAIT=1            # Prevent hanging
export NCCL_ASYNC_ERROR_HANDLING=1     # Better error handling
```

## Monitoring

### WandB Logging

Only rank 0 logs to WandB to avoid conflicts:
- Training metrics (every 10 steps)
- Validation metrics (every validation epoch)
- Animations and visualizations
- Model architecture and gradients

### Console Output

Only rank 0 prints to console:
- Progress bars
- Epoch summaries
- Validation results
- Early stopping messages

## Checkpointing

### Saving Checkpoints

Only rank 0 saves checkpoints to prevent conflicts. The saved checkpoint contains:
- Unwrapped model state dict (DDP wrapper removed)
- Optimizer state
- Training progress (epoch, step)
- Configuration

### Resuming Training

All processes load the checkpoint:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    train.py \
    --config configs/train_joint.yaml \
    --resume save/experiment_name/checkpoints/latest.pt
```

## Performance Tips

### 1. **Optimal Batch Size**
- Increase total batch size when using more GPUs
- Example: Single GPU: 32 → 8 GPUs: 256
- Ensures each GPU still processes a reasonable batch

### 2. **Data Loading**
- Use enough workers: 4-8 per GPU
- Enable `pin_memory=True` (already set)
- Use `persistent_workers=True` (already set)

### 3. **NCCL Backend**
- Best for NVIDIA GPUs
- Automatically selected in `train.py` (uses `nccl` backend when running distributed)
- Distributed backend is initialized inside `train.py` when launched with `torch.distributed.run`
- Optimized for high-bandwidth interconnects (NVLink, InfiniBand)

### 4. **Gradient Accumulation**
- Can be combined with DDP
- Effective batch size = batch_per_gpu × num_gpus × accumulation_steps

```yaml
training:
  gradient_accumulation_steps: 2
  batch_size: 256  # Total across GPUs

# With 8 GPUs: effective batch = (256/8) * 8 * 2 = 512
# With 4 GPUs: effective batch = (256/4) * 4 * 2 = 512
```

## Troubleshooting

### Issue: NCCL Timeout

**Symptom**: Training hangs or times out during distributed initialization

**Solutions**:
```bash
# Increase timeout (in slurm script)
export NCCL_TIMEOUT=1800  # 30 minutes

# Or in train.py, add:
dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=1800))
```

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce batch size per GPU
2. Reduce model size (hidden_dim, num_layers)
3. Enable gradient checkpointing (if implemented)
4. Use gradient accumulation instead of larger batches

### Issue: Slow Data Loading

**Solutions**:
1. Increase `num_workers` in config
2. Ensure enough CPUs allocated in slurm script
3. Check disk I/O performance
4. Consider caching preprocessed data

### Issue: Different Results Across Runs

**Causes**:
- Random seed differs per process (intentional for data augmentation)
- Non-deterministic CUDA operations

**Solutions**:
```python
# For fully deterministic training (slower)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Comparison: Single-GPU vs Multi-GPU

| Aspect | Single GPU | Multi-GPU (8x) |
|--------|-----------|----------------|
| Script | `train.py` | `train.py` |
| Launch Command | `python train.py --config configs/train_joint.yaml` | `python -m torch.distributed.run --standalone --nproc_per_node=8 train.py --config configs/train_joint.yaml` |
| Batch Size (total) | 32 | 256 (recommended) |
| Batch Size (per GPU) | 32 | 32 |
| Training Speed | 1x | ~5.5-6.5x (depends on model/data) |
| Memory Usage (per GPU) | Same | Same |
| Learning Rate | α | α (no scaling needed) |
| Gradient Sync | N/A | Automatic |
| WandB Logging | All metrics | Only rank 0 |
| Checkpoint Saving | All checkpoints | Only rank 0 |

## Expected Speedup

Speedup is not perfectly linear due to:
- Communication overhead (gradient synchronization)
- Data loading bottlenecks
- Model architecture (some operations don't parallelize well)

**Typical speedups**:
- 2 GPUs: ~1.7-1.9x
- 4 GPUs: ~3.0-3.5x
- 8 GPUs: ~5.0-6.5x

## Validation and Metrics

### Metric Aggregation

During validation:
1. Each GPU processes its subset of validation data
2. Per-sample metrics collected locally
3. Metrics gathered to rank 0 using `all_gather`
4. Rank 0 computes final aggregated metrics
5. Early stopping decision broadcasted to all GPUs

### Reproducibility

With DDP, validation results should be identical to single-GPU:
- Same validation set split
- Same model weights (synchronized)
- Deterministic evaluation (no dropout)

## Best Practices

1. **Start Small**: Test with 2 GPUs before scaling to 4 or 8
2. **Monitor Utilization**: Use `nvidia-smi` to check GPU usage
3. **Profile Communication**: Check NCCL logs for bottlenecks
4. **Validate Correctness**: Compare single-GPU vs multi-GPU results
5. **Save Frequently**: DDP can be less stable, save checkpoints often

## Example Workflow

```bash
# 1. Quick smoke test (single GPU)
sbatch scripts/slurm/run_train_joint_test.sh

# 2. Monitor test training
tail -f logs/slurm_JOBID.out

# 3. Check GPU utilization
squeue -u $USER
# ssh to the compute node and monitor
watch -n 1 nvidia-smi

# 4. If test successful, run multi-GPU training
sbatch scripts/slurm/run_train_joint.sh

# 5. Monitor full training
tail -f logs/slurm_JOBID.out
```

## Additional Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
