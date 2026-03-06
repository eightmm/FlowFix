# `scripts/` (데이터/분석/Slurm 유틸)

이 폴더는 **실험 운영에 필요한 보조 스크립트**들을 모아둡니다. 패키지 코드(모델/데이터 로더/유틸)는 `src/`에 둡니다.

## 구조

```
scripts/
├── analysis/   # 학습/추론 결과 분석 및 시각화
├── data/       # 데이터 생성/전처리/검증
└── slurm/      # 클러스터(job) 실행 스크립트
```

## `scripts/analysis/`

- `visualize_loss.py`: 학습 로그 기반 loss 시각화
- `visualize_trajectory.py`: 샘플링 trajectory 시각화
- `infer_full_validation.py`: 전체 validation set 추론
- `infer_small_train_valid.py`: 소규모 train/valid 추론

## `scripts/data/`

- `preprocess_pdbbind.py`: PDBbind 전처리
- `create_train_val_split.py`: train/val split 생성
- `generate_training_data.py`: 학습 데이터 생성
- `generate_test_data.py`: 테스트 데이터 생성
- `inspect_data.py`: 데이터 점검
- `verify_test_data.py`: 테스트 데이터 검증

## `scripts/slurm/`

환경(경로, 파티션, GPU 수)은 클러스터마다 다르므로, 본인 환경에 맞게 `PYTHON`, `PROJECT_DIR`, `#SBATCH ...`를 조정해서 사용하세요.

- `run_train_full.sh`: 멀티 GPU(DDP) 학습
- `run_inference.sh`: 체크포인트 추론/평가
- `run_full_val_inference.sh`: 전체 validation 추론
