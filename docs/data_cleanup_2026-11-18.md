# 데이터 정리 최종 상태 — 2026-11-18

사용자 결정: Idylwild Inventory와 유사한 화풍에 채도는 아주 조금 높은 방향. 색상 변환은 아직 적용하지 않았다. 본 학습용 목표는 정제·검증셋 분리 후 5,000장 이상이다.

## 최종 처리

처음에는 제외/보류 자료를 보관 폴더로 옮겼다. 이후 사용자가 보관 대신 삭제를 요청했고, Pixel-art 172K와 Krea 16은 표본 보존·학습 후보 재검토를 선택했다. PixelGPT는 이미지 품질이 마음에 들지 않는다는 이유로 제외했다.

- **복원:** `data/raw/hf_pixelart172k/`, `data/raw/hf_krea16/` — 각 128개, 합계 256개. 학습 포함을 최종 승인한 상태는 아니다.
- **삭제:** PixelGPT·Atlas 표본, 보류한 Kenney 4종·OGA 6종, OCAL/Wikimedia/Glitch 계열 하위 팩 8종의 이미지·배포 파일, 이전 미리보기. 전체 4,524개 파일, 약 35.66 MiB를 삭제했다. 출처·라이선스 텍스트와 검사 메타데이터는 별도 보존했다.
- **보관 폴더:** `data/archive/`와 `results/archive/`는 제거했다.
- **유지:** 기존 496개·Idylwild 5종·DCSS·작은 아이템 팩·음식/재료 후보, 기존 495개 임시 전처리 결과, 사용자가 주석을 작성한 `src`.

초기 보관 대상에는 최종 품질 부적합 자료뿐 아니라 화풍·용도가 달라 보류한 자료도 있었다. 이번 삭제는 사용자의 로컬 정리 요청에 따른 것이며, 모든 자료가 기술적으로 학습 불가능하다는 판정은 아니다. 해당 출처의 URL·해시는 기록에 남겨 재수집 가능성을 보존한다.

## 현재 검토 대상

| 구분 | 출처 수 | 소형 후보 수 |
|---|---:|---:|
| 기존 에셋 후보 | 14 | 2,235 |
| 합성 재검토 표본 | 2 | 256 |
| 합계 | 16 | 2,491 |

raw PNG는 시트·미리보기 등을 포함해 2,935개이며 모두 읽기 성공했다. 표의 수량은 최종 학습 데이터 수가 아니다. 7Soul/Idylwild/DCSS의 기존 1,267개 비교 후보도 이 안에 포함된다.

## HF 선택 이유

Pixel-art 172K와 Krea 16은 데이터 카드에서 AI 생성임을 명시한다. 처음 제외한 주된 이유는 이전의 수작업 에셋만 사용한다는 방침이었다. 사용자는 이를 바꿔 두 자료를 학습 후보로 재검토하기로 했다.

- [Pixel-art 172K](https://huggingface.co/datasets/azyg0s/pixelart-172k): 학습 이미지 172,330개, 32×32 RGBA. 현재 128개만 보유한다. 캡션은 생성 프롬프트이며 이미지와의 일치가 전수 검증되지 않았다. 일부 magenta 잔여 픽셀과 격자 관련 경고가 있다.
- [Krea 16](https://huggingface.co/datasets/Angais/krea-2-pixelart-16x16-synthetic): 전체 550개, 16×16. 현재 네이티브 표본 128개를 보유한다. Krea 2 Turbo와 LoRA로 생성한 뒤 사람 검수를 거친 자료이며 수작업 그림은 아니다. 속성 캡션은 근사 라벨로 취급한다.
- PixelGPT: 사용자 품질 판단에 따라 삭제·수집 제외. 다른 HF 후보를 받더라도 포함하지 않는다.
- Atlas: 이번에 재선정한 두 자료에 포함되지 않아 표본을 삭제했다.

두 후보는 이미지 품질·아이템 종류·중복·라벨·이용 조건을 추가 검토해야 한다. 데이터 카드의 표시 라이선스만으로 모든 권리 검토가 완료되었다고 기록하지 않는다. 현재 추가 다운로드와 본 학습은 실행하지 않았다.

## 기록과 검증

- `docs/active_dataset_candidates.csv`: 현재 16개 출처의 후보 수량.
- `docs/dataset_candidates.csv`: 최초 수집 수량과 현재 활성/삭제 상태를 별도 필드로 기록.
- `docs/downloads.csv`: 다운로드 URL·해시 보존. 삭제된 데이터 경로는 비우고 `presence`와 `removed_archive_path`에 이력을 기록. 삭제된 HF 자료의 표본 manifest는 이미지 없이 조사 근거로만 유지.
- `docs/data_cleanup_moves.csv`: 초기 이동 기록과 최종 복원/삭제 상태.
- `data/research/removed_asset_provenance/`: 삭제된 배포본에서 보존한 라이선스·출처 텍스트. 이미지 없음.
- `results/collection_history/2026-11-18/`: 정리 전 CSV/JSON 검사 기록. 이미지 없음.
- `results/deletion_execution.json`, `results/final_cleanup_verification.json`: 실행·검증 결과. 초기 이동만 기록한 `cleanup_verification.json`과 구분한다.

남긴 데이터 파일 3,023개의 해시 변경 0개, `src/*.py` 해시 변경 0개. PixelGPT 이미지 및 보관 폴더 잔존 0개. 단위 테스트 9개 통과(0.124초), compileall 통과. HF 전용 수집기는 현재 재선정한 두 출처만 허용하며 PixelGPT 요청은 거부한다. 현재의 최종 학습 데이터 구성은 미확정이다.

## 사용자 커밋 명령

프로젝트 루트 PowerShell에서 순서대로 실행한다. 먼저 데이터 조사·정리와 문서를, 다음으로 모델 초안·주석·검증 코드를 기록한다. 각 `git diff --cached --stat`의 파일 목록을 확인한 뒤 커밋한다. 데이터 파일은 Git에서 제외된다.

```powershell
git add .gitignore configs/data_candidates.json docs scripts/audit_images.py scripts/analyze_candidates.py scripts/collect_candidates.py scripts/collect_dcss_items.py scripts/sample_hf_candidates.py
git diff --cached --stat
git commit -m "data: curate candidates and remove unused assets"

git add README.md configs/base.json src tests scripts/prepare_dataset.py scripts/probe_model.py
git diff --cached --stat
git commit -m "model: add annotated DDPM draft and validation checks"

git status --short
```

커밋·Push는 Codex가 실행하지 않았다.
