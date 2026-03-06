# Slurm 가이드 (FlowFix)

## 실행 스크립트 위치

- Slurm 스크립트: `scripts/slurm/`
  - `scripts/slurm/run_train_joint_test.sh`: 단일 GPU 빠른 테스트
  - `scripts/slurm/run_train_joint.sh`: 멀티 GPU(예: 8 GPU) 학습
  - `scripts/slurm/run_inference.sh`: 추론(체크포인트 평가)

## 학습 (예시)

### 단일 GPU 테스트

```bash
sbatch scripts/slurm/run_train_joint_test.sh
```

### 멀티 GPU 학습 (DDP)

```bash
sbatch scripts/slurm/run_train_joint.sh
```


## Quick Commands

### Submit a job
```bash
sbatch scripts/slurm/run_train_joint.sh
```

### Check job status
```bash
squeue -u $USER
```

### View job details
```bash
scontrol show job <JOB_ID>
```

### Cancel a job
```bash
scancel <JOB_ID>
```

### View job output (real-time)
```bash
tail -f logs/slurm_<JOB_ID>.out
```

### View job errors (real-time)
```bash
tail -f logs/slurm_<JOB_ID>.err
```

### Check GPU usage on node
```bash
ssh <NODE_NAME>
nvidia-smi
```


## Monitoring

### 1. Slurm Logs
- Output: `logs/slurm_<JOB_ID>.out`
- Errors: `logs/slurm_<JOB_ID>.err`

### 2. Training Logs
- Experiment directory: `save/flowfix_YYYYMMDD_HHMMSS/`
- Training log: `save/flowfix_YYYYMMDD_HHMMSS/train.log`

### 3. WandB Dashboard
If enabled in config (`wandb.enabled: true`):
- Project: `protein-ligand-flowfix`
- Track metrics, animations, and model performance in real-time


## Customization

### Modify GPU count
Edit `#SBATCH --gres=gpu:N` where N is the number of GPUs (1-8)

### Modify CPU count
Edit `#SBATCH --cpus-per-task=N`
- Rule of thumb: 2× number of data workers + overhead
- Config has `num_workers: 8`, so 16-32 CPUs recommended

### Modify memory
Edit `#SBATCH --mem=XG`
- Single GPU: 64GB is usually sufficient
- Multi-GPU: 128GB+ recommended

### Modify runtime
Edit `#SBATCH --time=D-HH:MM:SS`
- Current: 7-00:00:00 (7 days)
- Adjust based on expected training time

### Change partition
Edit `#SBATCH --partition=<NAME>`
- Default: `gpu`
- Check available partitions: `sinfo`


## Training Configuration

학습 설정을 바꾸려면 `configs/train_joint.yaml`을 수정하세요:
- `batch_size`: Samples per batch (default: 8)
- `num_epochs`: Total epochs (default: 5000)
- `optimizer.muon.lr`: Muon LR for 2D weights (default: 0.02)
- `optimizer.adamw.lr`: AdamW LR for other params (default: 0.0003)
- `validation.frequency`: Validation every N epochs (default: 20)
- `checkpoint.save_freq`: Save checkpoint every N epochs (default: 10)


## Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size` in `configs/train_joint.yaml`
2. Reduce `num_timesteps_per_sample` (default: 16)
3. Increase `#SBATCH --mem` in Slurm script
4. Use gradient accumulation (increase `gradient_accumulation_steps`)

### Job pending for long time
```bash
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```
Check "NODELIST(REASON)" column for why job is waiting

### Job fails immediately
1. Check error log: `logs/slurm_<JOB_ID>.err`
2. Verify conda environment: `conda activate protein-ligand`
3. Test locally first: `python train.py --config configs/train_joint.yaml`

### CUDA out of memory
1. Reduce batch size
2. Reduce model dimensions in config
3. Use gradient accumulation (increase `gradient_accumulation_steps`)


## Resume Training

To resume from checkpoint:
```bash
python train.py \
    --config configs/train_joint.yaml \
    --resume save/flowfix_YYYYMMDD_HHMMSS/checkpoints/latest.pt \
    --device cuda
```

Update Slurm script to add `--resume` flag.


## Best Practices

1. **Test locally first** with CPU or small dataset
2. **Monitor early epochs** to ensure training is stable
3. **Use WandB** for remote monitoring
4. **Save checkpoints frequently** (every 10 epochs)
5. **Keep last N checkpoints** to save disk space
6. **Start with single GPU** before scaling to multi-GPU
7. **Set appropriate time limits** to avoid job termination


## Example Workflow

```bash
# 1. Prepare data (if not done)
ls train_data/ | head

# 2. Test configuration locally
python train.py --config configs/train_joint.yaml

# 3. Submit job
sbatch scripts/slurm/run_train_joint.sh

# 4. Monitor job
squeue -u $USER
tail -f logs/slurm_<JOB_ID>.out

# 5. Check WandB dashboard
# Visit: https://wandb.ai/<username>/protein-ligand-flowfix

# 6. After training
ls save/flowfix_YYYYMMDD_HHMMSS/checkpoints/
```
