# 원본 검사 및 구현 중단 기록

이 문서는 추가 수집 전의 중단 시점 기록이다. 이후 작업을 재개한 현재 결과는 [데이터 조사 보고](dataset_research_2026-11-18.md), [모델 설계](model_design.md)를 따른다. 아래의 당시 수치와 미실행 상태는 이력으로 보존한다.

상태: 사용자 요청으로 일시 중단. 496개 원본 및 495개 후보 전처리 결과는 최종 데이터셋이 아니다. 추가 데이터 수집·구성 분석 후 전처리 및 모델 설계를 확정한다. 작성한 코드와 후보 산출물을 보존했으며 본 학습과 GPU 학습 스모크 테스트는 시작하지 않았다.

## 원본 검사

- PNG 496개 모두 읽기 성공. 전부 34×34 RGBA, 알파 채널 보유.
- SHA-256 중복 1개: `S_Buff06.png`와 `S_Buff07.png`가 동일. 고유 이미지 495개.
- 완전 투명 이미지 0개, 단색 이미지 0개, 읽기 오류 0개.
- 반투명 픽셀이 있는 이미지 2개. 나머지는 알파 0 또는 255.
- ZIP 1개는 PNG 집계에서 제외. 로컬 ZIP 내부 PNG 496개의 이름과 해시가 해제된 파일 전부와 일치한다. 이는 로컬 해제 일관성 확인이며 원격 배포본과의 동일성 검증은 아니다.
- 바깥쪽 가장자리에 보이는 픽셀이 있는 이미지가 149개다. 34×34에서 테두리 1픽셀을 일괄 제거하면 일부 이미지가 잘리므로 그렇게 처리해서는 안 된다.
- 보이는 영역의 비율은 최소 11.9%, 중앙값 38.3%, 최대 88.2%. 보이는 RGB 색상 수는 최소 3개, 중앙값 8개, 최대 40개다.
- 표본 및 전처리 비교 시트에서 아이템·스킬 아이콘을 확인했다. 모든 파일의 의미를 수동 검수한 결과는 아니며, 이름 기반 분류는 추정이다.

| 추정 분류 | 원본 수 | 중복 제외 후보 수 |
|---|---:|---:|
| 스킬 (`S_`) | 154 | 153 |
| 무기 (`W_`) | 138 | 138 |
| 포션 (`P_`) | 57 | 57 |
| 음식 (`I_C_`) | 25 | 25 |
| 방어구·방패 (`A_`, `C_`, `E_`) | 36 | 36 |
| 장신구 (`Ac_`) | 14 | 14 |
| 기타 인벤토리·재료 (`I_`, 음식 제외) | 72 | 72 |
| 합계 | 496 | 495 |

스킬과 무기는 후보의 58.8%로 비중이 크다. 사용자는 스킬을 포함하기를 선호하며, 이를 후보 전처리의 기본값에 반영했다. 스킬이 학습 품질에 미치는 영향은 아직 측정하지 않았다. 무조건부 모델은 아이템과 스킬을 모두 생성할 수 있다. 동일 해시뿐 아니라 색상·형태 변형에 의한 유사 중복과 분할 간 유사성도 추가 검토가 필요하다.

## 출처

[OpenGameArt 원문](https://opengameart.org/content/496-pixel-art-icons-for-medievalfantasy-rpg)에서 CC0 표기, 저자 Henrique Lazarini (7Soul1), 저작권 게임의 파생 아이콘을 제거했다는 설명을 확인했다. `docs/sources.csv`는 다운로드 완료로 갱신했으며 검토 상태는 `source_page_checked_file_provenance_pending`이다. 원격 배포본과 로컬 파일의 동일성 및 학습·배포 적합성은 최종 검증 완료로 표시하지 않았다.

## 후보 산출물과 구현

- `scripts/audit_images.py`: 해상도·모드·알파·빈 이미지·중복·파일명 분류와 JSON 요약으로 확장. 실행 확인.
- `scripts/prepare_dataset.py`: 전체 영역 Nearest Neighbor 축소, 종횡비 유지 및 중앙 패딩, `(24, 24, 24)` 배경 RGB 합성, 중복 제거, 시드 42 분할, 매니페스트, 비교 시트. 후보 데이터에서 실행 확인. `--exclude-skills` 비교 옵션도 작성했으나 실행하지 않았다.
- `src/dataset.py`: 매니페스트 기반 RGB 텐서 로더. 후보 495개 전체를 검사해 `(3, 32, 32)`, 유한값, `[-1, 1]` 범위를 확인했다. 후보 분할은 학습 445개·검증 50개이고 처리 후 동일 픽셀 해시의 분할 중복은 0개다.
- `src/unet.py`: 32→16→8→16→32, 잔차 블록, skip connection, timestep embedding 초안.
- `src/diffusion.py`: 선형 beta, forward noising, MSE 노이즈 예측, DDPM reverse sampling 초안.
- `src/train.py`: GPU 학습, epoch 손실, 체크포인트, 옵티마이저·난수·배치 위치 복원 초안.
- `src/sample.py`, `src/utils.py`, `src/__init__.py`: 고정 시드 샘플링, 설정·체크포인트·PNG 저장 지원 초안.
- 모델 파일들은 작성만 완료했고 출력 shape, GPU forward/backward, 체크포인트 왕복, reverse sampling은 아직 검증하지 않았다. 체크포인트 파일도 생성하지 않았다.
- `configs/base.json`에 데이터·모델·채널 결정이 잠정 상태임을 명시했다. 기존 채널 수 64/128/256, 1,000 확산 단계, 배치 32, 학습률 0.0002, 200 epoch는 초기값으로 보존했다.

후보 RGB 합성은 투명 배경을 고정 색상으로 단순화하지만 투명도 정보가 사라진다. 34→32 축소는 선명한 경계를 유지해도 세부 픽셀 일부를 잃을 수 있다. RGBA 모델링과 다른 전처리 방법은 최종 결정 전 비교 대상이다.

산출물: `results/raw_audit.csv`, `results/raw_audit.summary.json`, `results/data_composition_analysis.json`, `results/preparation.json`, `results/contact_sheets/before_after_01.png`부터 `07.png`, `data/processed/manifest.csv`, `data/processed/images/`. 모두 검토용이다. 원본은 수정하지 않았으며 데이터와 체크포인트를 Git에 추가하지 않았다. 커밋·푸시는 하지 않았다.

## 실행 검증 기록

프로젝트 루트 PowerShell에서 `.venv` Python을 사용했다.

| 실행 | 측정 결과 |
|---|---|
| `.\.venv\Scripts\python.exe scripts\audit_images.py --input data\raw --output results\raw_audit.csv` | 확장 검사 내부 측정 0.308초; 496개 정상 |
| `.\.venv\Scripts\python.exe scripts\prepare_dataset.py --config configs\base.json` | 내부 측정 3.102초; 후보 495개 및 비교 시트 7장 |
| `.\.venv\Scripts\python.exe scripts\check_gpu.py` | CUDA 사용 가능, RTX 4070 Ti SUPER 15.99 GiB, 행렬 연산 성공. 개별 소요 시간 미분리 |
| Python inline: `PixelItemDataset(split='all')` 전체 순회, `DataLoader(..., batch_size=32)`, 픽셀 해시 교집합 검사 | 프로세스 전체 약 3.31초; 495개 tensor/범위/유한값 검사 및 `(32, 3, 32, 32)` 배치 통과 |
| Python inline: 로컬 ZIP 엔트리 해시 비교 및 알파·색상 통계 | 프로세스 전체 약 1.13초; 해시 불일치 0개 |

GPU 행렬 연산은 환경 확인이며 모델 학습 스텝이 아니다. 모델 실행 테스트는 사용자의 중단 요청으로 미실행이다. 추가 데이터 수집·구성 검토 지시를 받을 때까지 후속 구현·테스트·학습을 진행하지 않는다.
