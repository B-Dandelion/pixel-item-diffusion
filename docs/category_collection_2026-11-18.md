# 카테고리별 추가 수집 — 2026-11-18

게임 아이템 후보 12,237개와 선택적 스킬·효과 300개를 확보했다. 최종 학습셋, 전처리, 모델 입력 채널과 조건부 생성 여부는 확정하지 않았다. 본 학습·optimizer step·Git Commit/Push는 실행하지 않았다.

## 출처와 저장 위치

- [Pixel-art 172K](https://huggingface.co/datasets/azyg0s/pixelart-172k): 원본 train 172,330개 중 24개 카테고리에서 12,055개를 선정. 32×32 RGBA PNG. 리비전 `69157ce36974a3afcaf2465531cfab4c2a0e36bb`.
- [Krea 16](https://huggingface.co/datasets/Angais/krea-2-pixelart-16x16-synthetic): 전체 550개 중 28개 아이템 그룹의 482개 선정. 16×16 RGBA PNG. 리비전 `a1d892dd9d800733a30e44c70ab5e3d270b38f75`. 기존 표본 119개 재사용, 신규 파일 363개.
- 파일: `data/raw/<source>/items/<class>/`. 재사용한 Krea 표본은 기존 `samples/` 경로 유지.
- 정확한 URL·원본 행·파일명·카테고리·캡션·생성 backend·추출 경고·SHA-256: `data/raw/<source>/item_manifest.json`.
- 종류별 미리보기 19개: `results/item_review/<source>/<class>.png`. 시드 42, 종류별 최대 48개. 미리보기에 표시된 숫자는 원본 행 번호다.
- `item_manifest.json`의 카테고리는 다운로드를 위한 거친 분류다. 검증된 학습 라벨이 아니다. 기존 viewer 표본의 행 번호와 원본 Parquet 행 번호를 같은 ID라고 가정하지 않는다.

## 수량과 용량

| 분류 | Pixel-art 172K | Krea 16 | 합계 |
|---|---:|---:|---:|
| 무기 | 1,800 | 95 | 1,895 |
| 방어구·방패 | 855 | 105 | 960 |
| 재료·광물·허브 | 1,800 | 53 | 1,853 |
| 포션 | 1,000 | 18 | 1,018 |
| 음식 | 1,800 | 90 | 1,890 |
| 책·스크롤 | 1,000 | 19 | 1,019 |
| 도구 | 1,000 | 18 | 1,018 |
| 보물·장신구 | 1,500 | 49 | 1,549 |
| RPG 기타 아이템 | 1,000 | 35 | 1,035 |
| 스킬·효과(선택 후보) | 300 | 0 | 300 |
| 합계 | 12,055 | 482 | 12,537 |

선별한 PNG 파일 내용은 15,479,998바이트(14.76MiB)다. 메타데이터 캐시와 두 manifest는 별도로 약 141.25MiB이며 검사 결과·미리보기·파일시스템 할당 공간도 추가된다. 작업 전 C 드라이브 가용 공간 약 267GiB를 확인했다.

전체 저장소 다운로드 대신 메타데이터 열을 먼저 읽고 선택 행이 들어 있는 이미지 열의 byte range를 읽었다. 172K 메타데이터 전송 약 10.1MiB, 이미지 열 전송 약 178.4MiB. Parquet 압축 단위에는 선택하지 않은 행도 있어 전송량이 저장한 PNG보다 크다. 선택하지 않은 이미지의 개별 파일과 전체 Parquet 파일은 저장하지 않았다. HF의 `/filter` 요청 실패 후 이 방식을 사용했다. 재현 가능한 선별 설정은 `configs/item_collection.json`에 있다.

기존 표본 256개는 보존했다. 선별분에 포함되지 않은 표본 137개는 조사용으로만 남아 있다. 전체 현황은 원본 PNG 15,353개, 소형 검토 이미지 14,909개(기존 에셋 2,235 + HF 선별분 12,537 + 추가 조사 표본 137)다. 기존 에셋에는 중복·변형이 있고 전체 수치는 학습 수량이 아니다. 현재 수량은 `active_dataset_candidates.csv`, 초기 수집 수량은 `dataset_candidates.csv`의 historical 열에서 구분한다.

## 품질과 의미 검토

- 전체 원본 PNG 15,353개 읽기 성공, 손상 0, 파일 바이트 중복 49개. 이 중복은 전체 보관 자료 기준이다.
- 이번 선별분 12,537개는 모두 예상 크기·RGBA·PNG, 빈 이미지 0, 반투명 알파 0. 투명 픽셀의 RGB를 무시한 완전 일치 중복도 선별분 안에서는 0개. 비슷한 디자인·색상 변형의 근접 중복은 아직 검수하지 않았다.
- Pixel-art 172K 258개에 원본 추출 경고가 있다(기타 RPG 232, 음식 26). 경고별 수는 겹칠 수 있다: weak_grid 210, anisotropic_grid 42, rescaled_1.06 12, rescaled_1.10 8, period_corrected 8. 경고가 곧 불합격이라는 뜻은 아니므로 자동 삭제하지 않았다.
- 스킬 300개와 추출 경고 258개를 제외해도 아이템 후보 11,979개가 남는다. 이것은 의미·화풍·권리 검토를 통과한 수량이 아니다.
- 7개 미리보기 시트의 336개 표본을 시각 검토했다. 172K 무기·방어구·보물·기타 RPG·도구·음식과 Krea 무기이며, 전체 전수 시각 검수는 아니다.
- 무기 행 116324의 `pike`는 물고기로 렌더링됐고, 행 89231은 투구다. 도구 행 88524는 바이킹 캐릭터다. 예시는 `hf_visual_review_examples.csv`에 기록했다. 원본 이동·삭제·최종 라벨 교정은 적용하지 않았다.
- 선명한 아이템 실루엣을 가진 표본도 많지만, 현대 가방·안경·산업 도구가 섞인다. Idylwild 계열과 동일 화풍이라고 단정할 수 없다. Krea 16은 픽셀 단위가 더 굵어 출처별로 유지한다.
- 정확한 `(150,0,140)` 색상 픽셀은 선별분에서 0개였다. 이것만으로 잔여 마젠타가 없다고 판정하지 않는다. 데이터 카드의 배경 제거 경고와 근접 색상은 여전히 검토 대상이다.

Pixel-art 172K 카드는 전부 AI 생성이며 캡션이 검증된 묘사가 아니라 생성 프롬프트라고 명시한다. 표시 라이선스는 MIT, Krea 16은 CC0다. 이는 데이터셋 표시를 기록한 것이며 생성 모델·개별 출력의 권리 검토 완료를 뜻하지 않는다. 수집 허용과 최종 학습·서비스 활용 확정을 구분한다. PixelGPT·Atlas는 수집하지 않았다.

## 시간 측정과 학습 시간

이미지 수집 107.5초(메타데이터 최초 조회 별도). 전체 원본 검사 27.492초, 전체 후보 분석 28.536초, 선별분 검사·미리보기·임시 변환 31.623초. 이 세 검사 작업은 병렬 실행했으므로 시간을 더한 값은 총 경과 시간이 아니다.

선별분 12,537개를 읽고 임시 RGB 합성 → nearest 32×32 → 정규화하는 측정은 **4.032초**였다. 검사 직후 파일 캐시가 있는 상태이며, 배열만 만들고 결과 파일은 저장하지 않았다. 최종 전처리 방식 선택이나 학습 성능 측정이 아니다.

작은 PNG 용량은 저장·전송·읽기에 유리하지만 GPU는 압축 파일이 아니라 텐서로 계산한다. 동일 모델·배치·epoch 기준으로 학습량은 이미지 수에 비례한다. 배치 32, drop_last=False라면 학습 이미지 5,000개는 epoch당 157 step, 12,000개는 375 step이다(약 2.39배). VRAM은 전체 데이터 개수보다 배치·모델·해상도에 크게 좌우된다. DDPM 샘플 생성 시간도 별도로 필요하므로 본 학습 소요 시간은 아직 확정할 수 없다.

## 다음 결정

추가 대량 다운로드보다 19개 미리보기에서 화풍·아이템 적합성을 먼저 비교한다. 사용자는 원하는 스타일과 제외할 현대적 요소를 판단하고, 스크립트로 손상·중복·알파·원본 경고를 검사한다. 원치 않는 캐릭터·배경 장면, 잘못 그린 사물, 찌꺼기 픽셀, 의미가 다른 라벨은 검토 목록으로 남긴다.

그다음 채택할 출처·카테고리와 제외 규칙을 정하고, 가까운 변형과 같은 sprite sheet를 묶어 분리한다. 10% 검증셋을 둔다면 단순 계산상 5,000개 train을 위해 약 5,556개 이상이 필요하며 그룹 분리와 추가 제외에 대비해 6,000개 이상의 검수 완료 풀을 목표로 한다. 현재 11,979개는 검수를 위한 여유분이지 학습 승인 수량이 아니다. Krea 16→32 변환 및 합성 자료와 수작업 에셋의 혼합은 이후 확정한다.

## 재현과 변경 파일

```powershell
python -m pip install -r requirements-data.txt
python scripts\collect_hf_items.py --preview
python scripts\collect_hf_items.py
python scripts\audit_images.py --input data\raw --output results\all_raw_audit.csv
python scripts\analyze_candidates.py
python scripts\review_hf_items.py
python -m unittest discover -s tests -v
```

기존 의존성을 확인한 뒤 데이터 수집용 선택 의존성 pyarrow 23.0.1을 설치했다. 모델 학습 의존성에 합치지 않고 `requirements-data.txt`로 분리했다.

새 파일: `configs/item_collection.json`, `requirements-data.txt`, `scripts/collect_hf_items.py`, `scripts/review_hf_items.py`, `tests/test_hf_item_collection.py`, 이 보고서와 `docs/hf_visual_review_examples.csv`. 수정: `.gitignore`, `README.md`, `scripts/analyze_candidates.py`, `docs/ai_usage.md`, 출처·수량·다운로드 CSV 4개. 모델 `src`와 사용자 주석은 변경하지 않았다. 기존 보관 이미지 2,935개는 과거 검사 해시와 모두 일치했다.

13개 단위 테스트 통과. 허용 카테고리·상한·재현성, 실제 원본 형식과 투명 픽셀 중복, 기존 표본 재사용, HTTP Range 무시·전송 예산 초과 거부 등을 포함한다. 최종 데이터 품질과 학습 결과는 이 테스트가 검증하지 않는다.

## Git 상태

실제 GitHub `refs/heads/main`과 로컬 HEAD는 `e27e1f4a7b2544825d2f314534e750cc77110312`로 일치했다. 앞선 두 커밋은 원격에 반영돼 있다. 데이터와 생성 검사 이미지는 `.gitignore`로 제외되므로 GitHub 파일 목록에 나타나지 않는다. 이번 추가 수집 코드·문서 변경은 아직 커밋하지 않았다.

사용자가 변경 내용을 확인한 뒤 다음 단위로 커밋할 수 있다. 데이터 원본은 포함되지 않는다.

```powershell
git status --short
git add .gitignore README.md requirements-data.txt configs/item_collection.json scripts/collect_hf_items.py scripts/review_hf_items.py scripts/analyze_candidates.py tests/test_hf_item_collection.py
git add docs/active_dataset_candidates.csv docs/dataset_candidates.csv docs/sources.csv docs/downloads.csv docs/ai_usage.md docs/category_collection_2026-11-18.md docs/hf_visual_review_examples.csv
git diff --cached --stat
git commit -m "data: collect and audit category-selected item candidates"
git push origin main
```
