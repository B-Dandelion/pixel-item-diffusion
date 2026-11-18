# Pixel Item Diffusion

32×32 픽셀 아트 게임 아이템을 생성하는 DDPM 프로젝트입니다. U-Net과 확산 과정을 PyTorch로 직접 구현합니다.

**진행 상태:** 개발 환경 구축 완료, 데이터 수집 중, 모델 미구현

## 개발 환경

- OS: Windows 11
- Python: 3.12
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (16GB)
- CPU: AMD Ryzen 7 7800X3D
- RAM: 32GB

## 설치

PowerShell에서 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts\check_gpu.py
```

GPU 검사 결과에 `CUDA available: True`와 `CUDA tensor test: OK`가 표시되어야 합니다.

## 데이터셋

게임 아이템 이미지 에셋을 출처별로 수집합니다. 원본 파일과 라이선스 정보를 보존하고, 학습 전 이미지 크기·형식·중복을 검사합니다.

- 데이터 출처 및 라이선스: `docs/sources.csv`
- 이미지별 메타데이터 양식: `docs/dataset_manifest_template.csv`
- 원본 이미지: `data/raw/<source>/` (Git 제외)
- 전처리 이미지: `data/processed/` (Git 제외)

```powershell
python scripts\audit_images.py --input data\raw --output results\raw_audit.csv
```

## 모델 설계안

아래 값은 초기 실험 설정이며 구현 및 검증 전입니다. 상세 설정은 `configs/base.json`을 참조합니다.

| 항목 | 초기 설정 |
|---|---|
| 이미지 | 32×32 |
| U-Net 채널 | 64 → 128 → 256 |
| 해상도 | 32 → 16 → 8 → 16 → 32 |
| Skip connection | 동일 해상도 특징 맵 연결 |
| 시간 조건 | Timestep embedding |
| 확산 단계 | 1,000 |
| 예측 대상 | 추가된 노이즈 ε |
| 손실 함수 | MSE |
| 초기 배치 크기 | 32 |
| 시드 | 42 |

이미지 채널(RGB/RGBA)과 클래스 조건부 생성 적용 여부는 데이터 검토 후 결정합니다. Attention 및 사전학습 가중치는 사용하지 않습니다.

## 프로젝트 구조

```text
configs/       실험 설정
data/          원본 및 전처리 이미지 (Git 제외)
docs/          데이터 출처 및 메타데이터
scripts/       환경·이미지 검사
src/           모델·학습·생성 코드 (구현 예정)
checkpoints/   학습 가중치 (Git 제외)
results/       학습 로그 및 생성 결과
```

## 실험 기록 항목

- 데이터 출처, 라이선스, 전처리 방식
- 모델 구조, 하이퍼파라미터, 시드
- Epoch별 손실 및 생성 이미지 변화
- 학습 시간, GPU 사용량, 최종 결과
- AI 도구 활용 범위
