# Pixel Item Diffusion — 시작 준비 자료

목표: 게임에서 사용할 수 있는 32×32 도트 아이템 이미지 생성 모델을 PyTorch로 **처음부터 직접 구현**합니다.

이 자료에는 사전학습 모델이나 U-Net 구현 완성본을 넣지 않았습니다. 수업 조건을 준수하며 데이터셋과 네트워크를 직접 검증하기 위한 시작 구조입니다.

## 확인된 로컬 환경

- Windows 11 Home
- NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB GDDR6X
- AMD Ryzen 7 7800X3D
- RAM 32 GB / SSD 1 TB

NVIDIA 공식 사양: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/

## Windows PowerShell에서 시작

1. Python 3.11 64-bit와 Git이 설치되어 있는지 확인합니다. `py -0p`, `git --version` 명령을 사용합니다.
2. 이 폴더에서 터미널을 열고 가상환경을 만듭니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

3. 공식 PyTorch 설치 페이지의 **Windows / Pip / Python / CUDA** 설정을 선택하여 표시된 명령으로 CUDA용 PyTorch를 설치합니다. CPU-only 버전을 잘못 설치하지 않도록 주의합니다.

   https://pytorch.org/get-started/locally/

4. 그 다음 나머지 패키지를 설치합니다.

```powershell
python -m pip install -r requirements.txt
nvidia-smi
python scripts\check_gpu.py
```

`CUDA available: True`와 GPU 이름/메모리가 출력되어야 다음 단계로 넘어갑니다.

5. 우선 아래 출처에서 **원본 ZIP**을 수동으로 받아 `data/raw/<source>/`에 해제합니다. 다운로드한 ZIP 원본과 LICENSE 문구도 보존합니다.

   - https://opengameart.org/content/496-pixel-art-icons-for-medievalfantasy-rpg
   - https://opengameart.org/content/idylwilds-arsenal
   - https://opengameart.org/content/idylwilds-inventory
   - https://opengameart.org/content/idylwilds-armory
   - https://opengameart.org/content/idylwilds-arcanum
   - https://opengameart.org/content/idylwilds-aerial-arsenal

6. 다운로드 후 원본을 변경하지 않은 채 파일 현황을 집계합니다.

```powershell
python scripts\audit_images.py --input data\raw --output results\raw_audit.csv
```

각 출처의 라이선스는 `docs/sources.csv`, 실제 사용한 이미지별 provenance는 `docs/dataset_manifest_template.csv` 형식으로 기록합니다. CSV를 실제 내용으로 채우기 전에 이미지가 자동으로 라이선스 검증된 것으로 간주하지 마세요.

## 폴더 구조

```text
data/raw/                 출처별 다운로드 원본 (Git 추적 제외)
data/processed/           실제 학습에 사용할 32×32 이미지 (Git 추적 제외)
src/                     직접 구현할 dataset, unet, diffusion, train, sample
configs/                  실험 설정
scripts/                  환경·데이터 검사
docs/                     데이터 출처, 진행 기록, 보고서 준비
checkpoints/              학습 중 가중치 (Git 추적 제외)
results/                  epoch별 loss / 고정 노이즈 샘플 / 실행 기록
```

## 과제 조건 (변경 불가)

- U-Net을 직접 설계·구현합니다.
- 사전학습 모델/가중치를 사용하지 않고 무작위 초기값에서 학습합니다.
- 수업에서 다루지 않은 Attention 등은 사용하지 않습니다.
- Python `random`, NumPy, PyTorch 등의 시드는 42로 고정합니다.
- epoch별 loss, 생성 이미지, 수렴 이후 변화, 총 학습 시간을 기록합니다.
- 직접 구현한 코드와 보고서의 재현성, 출처, AI 보조 활용 내역을 명시합니다.

### 초기 아키텍처 **설계안** (아직 미구현)

- 입력: `32×32`, 처음에는 투명 배경 이미지를 일관되게 처리할 방식 결정
- Down: `32→16→8`, 각 단계 채널 `64→128→256`
- Bottleneck: `8×8`, Time embedding 사용, **Attention 없음**
- Up: `8→16→32`, 대칭 U-Net + 같은 해상도 Skip concatenation
- 출력: 입력 노이즈의 예상값 `ε_θ(x_t, t)`
- 손실: `MSE(예측 노이즈, 실제 노이즈)`
- DDPM step: `T=1000`을 초기값으로 시험하고 실행 시간을 측정
- 배치: 32를 시작점으로 시험 (실측 GPU 사용량에 따라 변경)
- 라벨 조건부 학습은 강의 범위에 포함되는지 확인한 후 결정

강의 내용에 맞지 않는 블록이나 추가 기법은 적용하지 않습니다. `configs/base.json`은 확정 값이 아닌 최초 실험의 기준 값입니다.

## 작업 순서 및 종료 기준

1. 11/18: 원본 ZIP 확보, 라이선스 및 이미지 실사, 환경 검사, U-Net 설계
2. 11/19: DDPM/U-Net/데이터로더 직접 구현, 1배치 forward/backward 테스트, 체크포인트 저장 검증, 본 학습 시작
3. 11/20: Loss 및 고정 시드 생성 이미지 검토, 개선 실험
4. 11/21: 최종 체크포인트·epoch별 스냅샷 확보, 보고서 초안
5. 11/22: 재현성 검증, 그래프·표·이미지 일치 확인, 21:00 전 제출

**학습 시작 기준:** 정상적으로 로드되는 데이터셋, 유효한 U-Net 출력 shape, 1회 역전파, 체크포인트 저장/불러오기, 42 시드의 고정 노이즈 샘플링까지 통과해야 합니다.
