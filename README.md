# FlowFix: Protein-Ligand Pose Refinement

FlowFix는 protein-ligand binding pose refinement를 위한 **SE(3)-equivariant flow matching** 기반 모델/학습 파이프라인입니다.

## 디렉토리 가이드

- `src/`: 모델/데이터 로더/유틸(패키지 코드)
- `configs/`: 학습 설정 YAML (`configs/README.md` 참고)
- `scripts/`: 데이터 준비/분석/Slurm 스크립트 (`scripts/README.md` 참고)
- `docs/`: 운영 가이드 문서 (`docs/README.md` 참고)
- `tests/`: 테스트
- `train.py`: 학습 엔트리포인트 (DDP 멀티 GPU 지원)
- `inference.py`: 추론/평가 엔트리포인트

## 환경 설정 (conda + pip)

> GPU/드라이버/클러스터 환경에 따라 PyTorch 및 PyG 설치 방법이 달라질 수 있습니다.

```bash
# 예시: 새 conda 환경 생성
conda create -n flowfix python=3.10 -y
conda activate flowfix

# (권장) 본인 CUDA 환경에 맞는 PyTorch 설치 후
pip install -r requirements.txt
```

## 빠른 실행

### 학습 (로컬)

```bash
python train.py --config configs/train_joint.yaml
```

### 멀티 GPU 학습 (DDP)

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  train.py --config configs/train_joint.yaml
```

### 추론/평가

```bash
python inference.py \
  --config configs/train_joint.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --device cuda
```

## 참고 문서

- 프로젝트 설계/구현 메모: `CLAUDE.md`
- 운영 가이드: `docs/`
