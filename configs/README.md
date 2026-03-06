# `configs/` (실험 설정)

이 폴더에는 FlowFix 학습 실행에 필요한 YAML 설정 파일을 모아둡니다.

## 파일 목록

- `train.yaml`
  - **Cartesian 학습 설정** (per-atom velocity field)
  - Slurm 스크립트 `scripts/slurm/run_train_full.sh`에서 기본으로 사용합니다.
  - 추론 시에도 이 설정을 기반으로 모델을 로드합니다.

- `train_torsion.yaml`
  - **SE(3) + Torsion 학습 설정** (translation + rotation + torsion)
  - `train_torsion.py`에서 사용합니다.

## 공통 구조(개요)

설정 파일은 아래 섹션을 갖습니다.

- **`device`**: `cuda` 또는 `cpu`
- **`seed`**: 재현성 시드
- **`data`**: 데이터 경로/로딩 방식
  - `data_dir`: 전처리된 데이터 디렉토리 (예: `train_data`)
  - `split_file`: split 정의 파일 (예: `train_data/splits.json`)
  - `num_workers`: dataloader 워커 수
  - `loading_mode`: `lazy` 등
- **`model`**: 모델 아키텍처 및 hidden dim 등 하이퍼파라미터
- **`training`**: 학습 하이퍼파라미터
  - `batch_size`, `num_epochs`, `gradient_accumulation_steps`, `ema`, `validation` 등
  - `optimizer`: Muon + AdamW 하이브리드 설정 (2D 가중치: Muon, 나머지: AdamW)
  - `schedule`: 통합 LR 스케줄 (linear warmup + cosine decay)
- **`sampling`**: validation/inference 시 ODE integration 설정
  - `num_steps`, `method`(`euler`/`rk4`), `schedule`(`uniform`/`quadratic`/...)
- **`timestep_sampling`**: 학습 시 logit-normal 시간 샘플링 파라미터

## 빠른 실행 예시

### 학습 (로컬)

```bash
python train.py --config configs/train.yaml
```

### 멀티 GPU 학습 (DDP, 로컬/서버)

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  train.py --config configs/train.yaml
```

### 추론/평가

```bash
python inference.py \
  --config configs/train.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --device cuda
```
