# Pixel Item Diffusion

32×32 픽셀 아트 게임 아이템을 생성하는 DDPM 프로젝트입니다. U-Net과 확산 과정을 PyTorch로 직접 구현합니다.

**진행 상태:** 28개 출처 조사 후 미사용 자료 정리 완료. 현재 14개 에셋 출처의 소형 후보 2,235개와 HF 2종의 합성 표본 256개를 검토합니다. 최종 데이터셋과 전처리는 미확정이며 본 학습은 시작하지 않았습니다.

목표 화풍은 Idylwild Inventory를 기준으로 채도가 아주 조금 높은 방향입니다. 원본 색상은 변경하지 않았습니다. 정제·검증셋 분리 후 본 학습에 사용할 이미지 5,000장 이상을 목표로 추가 수집합니다. 현재 수량은 그 목표를 충족하지 않습니다.

수작업 에셋을 우선 검토하며, 사용자 결정으로 Pixel-art 172K와 Krea 16 합성 자료도 학습 후보로 재검토합니다. PixelGPT는 사용자의 이미지 품질 판단에 따라 제외·삭제했습니다. 조건부 생성은 선택지이며 필수 모델로 정하지 않았습니다. 기존 496개 원본과 495개 RGB 전처리 결과도 검토용으로 보존했습니다.

현재 7Soul 495개·Idylwild 297개·DCSS 475개를 출처별 비교 후보로 제안합니다. 합계 1,267개는 최종 학습 수량이 아닙니다. 원본·시트·미리보기·변형을 구분한 전체 수치와 실제 실행 결과는 [데이터 조사 보고](docs/dataset_research_2026-11-18.md)에 있습니다.

사용자의 후속 요청에 따라 미사용 이미지·배포 파일과 이전 미리보기를 삭제했습니다. 두 HF 표본은 복원했고 보관 폴더는 제거했습니다. 출처·라이선스·수집/검사 기록은 유지합니다. 현재 수량과 정리 내역은 [정리 기록](docs/data_cleanup_2026-11-18.md)을 참조합니다.

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
- 현재 검토 대상 수량: `docs/active_dataset_candidates.csv`
- 전체 수집 이력·현재 경로·삭제 상태: `docs/dataset_candidates.csv`
- 다운로드 주소·저장 위치·해시: `docs/downloads.csv`
- 이미지별 메타데이터 양식: `docs/dataset_manifest_template.csv`
- 원본 이미지: `data/raw/<source>/` (Git 제외)
- 전처리 이미지: `data/processed/` (Git 제외)

```powershell
python scripts\audit_images.py --input data\raw --output results\all_raw_audit.csv
python scripts\analyze_candidates.py
```

현재 `data/raw`의 PNG 2,935개는 모두 읽기 성공했습니다. 소형 후보 2,491개는 기존 에셋 2,235개와 합성 표본 256개의 합계이며, 중복·변형·의미·권리 검토가 남아 있습니다. PixelGPT 이미지와 미리보기는 삭제하고 이전 검사 메타데이터만 `results/collection_history/`에 남겼습니다.

수집 예정 목록만 확인하려면 `python scripts\collect_candidates.py download --dry-run`을 사용합니다. 보류 출처와 하위 팩은 기본 수집에서 제외됩니다. 일반 수집기의 `--include-deferred`는 보류 OGA/Kenney 후보를 명시적으로 포함합니다. HF 전용 수집기는 현재 허용된 두 출처만 선택하며 PixelGPT 요청은 거부합니다.

## 모델 설계안

아래는 보존한 구현 초안의 초기 설정이며 데이터 구성에 따라 변경될 수 있습니다. 상세 설정은 `configs/base.json`, 구조·전처리 선택지는 [설계 문서](docs/model_design.md), 과제 원문 대조는 [요구사항 문서](docs/assignment_requirements.md)를 참조합니다.

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

전처리 초안은 어두운 배경에 RGB 합성하며 스킬 이미지를 포함합니다. 이는 검토용 설정이며 RGB/RGBA 처리와 최종 데이터 구성은 추가 분석 후 확정합니다. 클래스 조건부 생성은 적용하지 않았습니다. Attention 및 사전학습 가중치는 사용하지 않습니다.

단위 테스트와 임의 텐서의 GPU 순전파·역전파를 확인했습니다. optimizer step, 실제 데이터 학습, 학습된 샘플 생성은 실행하지 않았습니다. 과제에 필요한 0 epoch·매 epoch 샘플 기록과 학습 재개 통합 검증은 남아 있습니다.

```powershell
python -m unittest discover -s tests -v
python scripts\probe_model.py
```

## 프로젝트 구조

```text
configs/       실험 설정
data/          원본 및 전처리 이미지 (Git 제외)
docs/          데이터 출처 및 메타데이터
scripts/       환경·이미지 검사
src/           데이터로더 및 모델·학습·생성 코드 초안
checkpoints/   학습 가중치 (Git 제외)
results/       학습 로그 및 생성 결과
```

## 실험 기록 항목

- 데이터 출처, 라이선스, 전처리 방식
- 모델 구조, 하이퍼파라미터, 시드
- Epoch별 손실 및 생성 이미지 변화
- 학습 시간, GPU 사용량, 최종 결과
- AI 도구 활용 범위

현재 AI 사용 기록은 [docs/ai_usage.md](docs/ai_usage.md)에 있습니다. 커밋을 나누는 권고 명령은 데이터 조사 보고에 있으며, 커밋과 Push는 사용자가 직접 수행합니다.
