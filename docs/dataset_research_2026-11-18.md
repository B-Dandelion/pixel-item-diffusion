# 데이터 수집·검사 및 구현 상태 — 2026-11-18

**최신 결정:** Idylwild와 유사하며 채도가 아주 조금 높은 방향. 사용자 요청으로 미사용 이미지를 삭제했으며 보관 폴더는 없다. Pixel-art 172K와 Krea 16은 표본 256개를 복원해 합성 학습 후보로 재검토한다. PixelGPT는 품질을 이유로 제외·삭제했다. 현재 경로·수량은 [정리 기록](data_cleanup_2026-11-18.md)과 [활성 후보 목록](active_dataset_candidates.csv)을 따른다. 아래 합성 제외 정책과 28개 출처·4,539개 파일·6,283개 후보 수치는 **정리 전 조사 이력**이다. 이전 검사 메타데이터는 `results/collection_history/2026-11-18/`에 있다.

수작업 에셋을 우선 검토하며, 후속 사용자 결정으로 Pixel-art 172K와 Krea 16도 합성 학습 후보로 재검토한다. PixelGPT는 이미지 품질을 이유로 제외했다. 조건부 생성은 선택지이며 모델을 조건부로 확정하지 않았다. 데이터 구성, RGB/RGBA, 최종 전처리와 학습 모델은 아직 미확정이다. 본 학습·커밋·Push는 실행하지 않았다.

## 오늘 목표와 다음 순서 — 사용자 추가 결정 반영

오늘 목표는 데이터 수집·검사·구성 분석, 전처리 결정, U-Net/DDPM 설계다. 현재 1차 수집과 기술 검사, 설계·구현 초안까지 진행했다. 최종 데이터 구성·전처리 결정은 남아 있다.

본 학습에 실제 투입할 목표는 **정제·중복 제거·검증셋 분리 후 5,000장 이상**이다. 테스트용 수량을 뜻하지 않는다. 검증 비율 10%를 사용한다면 정제 후 전체 약 5,600장 이상이 필요하며, 변형 그룹별 분리 시 실제 train 수를 다시 확인한다. 단순 복제나 증강으로 이 원본 목표 수량을 채우지 않는다.

1. Idylwild 기준 화풍 선호를 기록하고 미사용 자료를 별도 보관한다. 1차 정리 완료.
2. 현재 데이터 조사와 모델 초안을 커밋으로 기록한다. 아래의 두 단계 권고 명령을 사용자가 실행한다.
3. 그 기준으로 기존 자료를 선별하면서 수작업 에셋을 추가 수집한다. 다른 해상도·화풍은 별도 후보로 남긴다.
4. 정제한 수량·클래스·출처·변형 그룹을 재집계한 후 전처리와 조건부 여부를 결정한다.
5. 데이터로더와 학습 기록·재개 기능을 마무리하고 테스트한다. 본 학습은 현재 실행하지 않는다.

`small_review_images` 기준 HF 제외 6,283개 중 Kenney가 3,429개다. 여기에는 지형·캐릭터·UI·동일 물체 변형이 섞여 있어 유효 아이템 5,000장 확보를 의미하지 않는다. 나머지 출처의 소형 후보 2,854개도 추가 선별이 필요하다.

### 직접 볼 항목과 검사 분담

기술 검사와 출처 조사는 Codex가 계속 담당할 수 있다. 사용자는 먼저 결과물의 목표를 정하고 애매한 사례를 판단한다. 모든 파일의 해상도·라이선스를 직접 다시 조사할 필요는 없다.

| 확인 항목 | 판단 기준 | 처리 |
|---|---|---|
| 목표 대상 | 단일 아이템/재료/음식/스킬인가, 지형·캐릭터·UI 기호인가 | 대상 / 별도 후보 / 제외 사유 기록 |
| 원하는 화풍 | 선·음영·시점·색상·픽셀 밀도가 목표와 맞는가 | 함께 사용 검토 / 출처별 별도 유지 |
| 시각적 결함 | 물체가 잘리거나 여러 개 겹쳤는가, 시트가 단품으로 오인됐는가 | 문제 파일만 표시; 원본 삭제 금지 |
| 배경과 표현 | 고정 프레임·배경·글자·로고·투명 경계가 있는가 | 전처리 또는 제외 검토 대상으로 표시 |
| 의미상 변형 | 같은 물체의 색·방향·크기·애니메이션 변화인가 | 같은 그룹으로 묶어 split 누수 방지 |
| 클래스 | 실제 물체와 파일명 추정 라벨이 맞는가 | 검토된 라벨만 조건부 설계 판단에 사용 |

첫 검토는 `results/candidate_analysis/496_RPG_icons.png`, `oga_idylwild_inventory.png`, `dcss_cc0.png`, `oga_food_cc0.png`를 비교했다. 사용자는 Idylwild와 유사한 느낌에 채도가 아주 조금 높은 방향을 선택했다. `S_` 스킬은 이미 포함을 희망한 후보이므로 일괄 제거하지 않는다. 표본 검토는 스타일 기준을 정하는 단계이며 전체 품질 확인을 대신하지 않는다. 이후 기준에 따라 시트를 나눠 전수 선별하고 애매한 파일을 함께 검토한다.

기계적으로 확인할 해상도·실제 형식·채널·알파·손상·정확 중복·수량·분리 검증은 스크립트로 처리한다. 라이선스와 제작 경위는 원출처 자료로 별도 확인하며 이미지 외관으로 추정하지 않는다. 32×32 변환 후 잘림·픽셀 손실은 원본과 전처리 비교 시트에서 다시 확인한다.

## 수집 범위와 수량

기존 자료를 포함해 **28개 출처**를 로컬에서 검토했다. 24개 에셋 출처와 HF 4종 조사 표본이다. 출처별 라이선스는 [sources.csv](sources.csv), 수치·형식은 [dataset_candidates.csv](dataset_candidates.csv), 다운로드 주소·저장 위치·해시는 [downloads.csv](downloads.csv)에 기록했다. 기존 후보였던 Niko 및 crawl/crawl 행은 미수집 이력으로 유지하며 28개에 포함하지 않는다.

- `data/raw`의 `.png` 경로 4,539개: 실제 PNG 4,411개, **JPEG인 PixelGPT 뷰어 표본 128개**. 확장자만으로 PNG라고 집계하지 않는다.
- 위 4,539개 중 합성/제작 경위 불명확 HF 표본 512개는 학습 제외. 나머지 4,027개도 시트·미리보기·사진·벡터 축소본 등을 포함한 검토 파일 수이며 수작업 픽셀 아이템 수가 아니다.
- 그 밖에 GIF 6개·JPEG 2개 미리보기/참조 파일을 검사했다. ZIP·텍스트·SVG·작업 파일 등은 별도 보존했다.
- 시트에서 2,832개 검토용 셀을 `data/candidates`에 추출했다. 이는 원본의 파생물로, 원본 수에 더해 학습 데이터 규모를 주장하지 않는다.
- 원본 삭제·수정 및 최종 train/val split 생성 없음. 기존 495개 RGB 검토 결과와 445/50 임시 split도 그대로 보존했다.

표의 ‘파일’은 `.png` 경로의 원본/표본 수다. ‘소형 후보’는 별도 이미지 또는 추출 셀 중 크기 64 이하의 검토 대상 수로, 최종 아이템 수가 아니다. 괄호는 해당 출처 내부에서 완전히 같은 보이는 픽셀을 제외한 수다. 회전·색상·크기 변형의 의미상 중복은 아직 남는다.

| 출처 | 파일 | 소형 후보 (고유 픽셀) | 해상도·종류 및 검토 상태 |
|---|---:|---:|---|
| 496 RPG Icons | 496 | 496 (495) | 34×34 RGBA; 아이템+스킬. 최초 원본 |
| Idylwild Arsenal | 101 | 50 (50) | 32×32; 테두리 변형 50·시트 1 별도 |
| Idylwild Aerial Arsenal | 151 | 101 (101) | 32×32 아이콘 109 중 whoosh 8 별도; 폭발 96×96 40·시트 2 별도 |
| Idylwild Armory | 51 | 50 (50) | 32×32; 방어구·장신구, 시트 별도 |
| Idylwild Inventory | 51 | 50 (50) | 32×32; 재료·음식·도구, 시트 별도 |
| Idylwild Arcanum | 51 | 50 (50) | 32×32; 포션·책·지팡이·재료, 시트 별도 |
| DCSS CC0 export | 673 | 477 (475) | 32×32 P/RGBA; 식별용 `i-*` 기호 196개 별도 |
| OGA Icons 32×32 | 1 | 8 (8) | 128×64 시트; 포션·음식 |
| OGA RPG UI Icons | 108 | 108 (108) | 32×32 86, 16×16 22; 흰색 UI 기호·상태와 아이템 혼재 |
| OGA RPG Icon Base Pack | 40 | 0 | 물체 10종의 4개 해상도; 사진 기반으로 초기 픽셀 후보와 분리 |
| OGA 32×32 Items | 1 | 12 (12) | 16칸 중 단색 빈칸 4 제외; 고정 배경 |
| OGA Basic RPG Item Icons Remix | 59 | 29 (29) | 32/64 두 크기와 미리보기; 부드러운 음영, CC-BY |
| OGA CC0 Food Icons | 1,266 | 1,139 (1,138) | 16/24/32/34, 시트·strip·사진·벡터 혼재; 원제작자별 분리 필요 |
| OGA CC0 Tool Icons | 78 | 68 (68) | 16/24/32/64; 공구·생활용품, 파생 해상도 포함 |
| OGA CC0 Mineral Icons | 30 | 24 (24) | 광석·돌; 기존 출처 중복과 사진/벡터 검토 |
| OGA CC0 Gem Icons | 144 | 124 (124) | 16/24/32, 애니메이션 strip 및 렌더/벡터 혼재 |
| OGA CC0 Ingot Icons | 12 | 9 (9) | 32×32; 기존 7Soul 자료 중복 가능 |
| OGA 27 Items Icons | 1 | 27 (27) | 32×32 framed cells, 비어 있는 마지막 3칸 미추출 |
| OGA 20 Pixel Art Icons | 21 | 20 (20) | 실제 단품은 13–32px; 페이지 64×64 설명과 다름, 보조 후보 |
| OGA Inventory Filter Icons | 13 | 12 (12) | 16×16; 아이템보다는 필터/상태 UI 위주 |
| Kenney Roguelike/RPG | 5 | 1,704 (1,688) | 16×16+1px 간격 시트; 지형·가구·캐릭터 포함 |
| Kenney RPG Base | 233 | 229 (229) | 실제 64×64 RGB/RGBA; 주로 지형·오브젝트 |
| Kenney Monochrome RPG | 427 | 419 (416) | 16×16 중심, 팔레트·캐릭터 변형 포함; 별도 흑백 계열 |
| Kenney 1-Bit Pack | 14 | 1,077 (1,073) | 49×22의 16×16 격자 중 1칸 투명; 별도 1-bit 계열 |
| HF Pixel-art 172K | 128 | 128 (128) | 32×32 RGBA; 합성 표본, 학습 제외 |
| HF Pixel Art Atlas | 128 | 128 (128) | 16×16 RGBA; 합성 표본, 학습 제외 |
| HF Krea 2 16×16 | 128 | 128 (128) | 네이티브 sprite 16×16; 합성 표본, 학습 제외 |
| HF PixelGPT | 128 | 128 (128) | 24×24 RGB **JPEG 뷰어 표본**; 원본 PNG 미확인, 학습 제외 |

## 데이터 구성 제안과 종류별 수량

우선 **7Soul 495개 / Idylwild 297개 / DCSS 475개**를 출처별 하위 후보로 유지한다. 표본에서 7Soul은 비교적 단순한 선과 밝은 색, Idylwild는 어두운 외곽선과 질감, DCSS는 다양한 제작자·시점·음영을 보였다. 통합하면 1,267개지만, 품질 검토를 통과한 최종 데이터셋 수는 아니다. Idylwild 5종의 소형 후보 301개에서 4개 중복이 제거된다. 7Soul의 `S_` 153개는 사용자의 의향에 따라 후보에 유지한다. 스킬을 섞었을 때 아이템 형태 학습에 방해가 되는지는 이후 실제 비교로 판단한다.

`results/candidate_analysis/core_proposal.csv`는 원본 경로와 중복 관계를 기록한 **검토 목록**이다. 복사·정규화·분리를 수행하지 않는다. 아래는 그중 중복 대표를 결정적 순서로 하나씩 센 파일명/폴더 추정치다. 동일 이미지의 라벨 충돌, 책을 무기로 분류한 기존 접두어, 미분류 90개 등을 수동 검토해야 한다.

| 추정 분류 | 수 |
|---|---:|
| 무기 | 417 |
| 방어구·방패 | 191 |
| 스킬 | 154 |
| 포션 | 88 |
| 음식 | 50 |
| 장신구 | 60 |
| 책·스크롤 | 53 |
| 재료 | 27 |
| 도구 | 8 |
| 화폐 | 14 |
| 인벤토리·재료 혼합 | 72 |
| 기타 | 43 |
| 미분류 | 90 |
| 합계 | 1,267 |

무기 비중은 약 32.9%이며 음식·재료·도구가 상대적으로 적다. 소수 클래스를 단순 복제해 수량을 맞추지 않는다. Food/Tool/Gem 묶음에서 제작 경위·개별 라이선스·픽셀 품질을 확인한 하위 자료를 추가하는 것을 우선 권고한다. 20 Pixel Art Icons 및 작은 아이템 시트도 보조 후보로 남긴다. Kenney와 흑백 계열은 물체 선별 후 별도 실험 후보로 유지한다.

조건부 생성은 검토된 대분류 라벨과 충분한 클래스별 독립 원본이 있을 때 채택할 수 있다. 현 추정치를 그대로 클래스 라벨로 사용하지 않는다. 무조건부/조건부 어느 모델도 최종 선택하지 않았다.

## 품질·중복·라이선스 검토

28종의 출처별 contact sheet를 직접 확인했으며 큰 묶음은 최대 48개 seed 42 표본이다. 전수 의미 검수는 아니다. 깨진 이미지 디코딩은 발견되지 않았으나 PNG 엄격 검사에서 PixelGPT 128개가 확장자 불일치로 보고되었다. 실제 손상과 혼동하지 않는다. 수신 바이트와 기존 파일명은 이력으로 보존하고, 분석에 실제 형식과 미리보기 여부를 추가했다.

중복 검사는 원본 SHA-256, 투명 픽셀의 보이지 않는 RGB를 0으로 만든 RGBA, 투명 여백을 제외한 RGBA 세 가지다. 출처를 가로지르는 일치 그룹은 각각 1/25/58개였다. 그룹 수는 제외할 파일 수가 아니며, 크기/회전/색상/애니메이션 근접 중복을 모두 탐지하지 못한다. 원본은 삭제하지 않는다. 동일 물체 변형이 train과 val에 나뉘지 않도록 최종 split은 변형 그룹 단위로 해야 한다.

- [496 원본 페이지](https://opengameart.org/content/496-pixel-art-icons-for-medievalfantasy-rpg)는 CC0를 표시하고 배포자가 저작권 있는 게임에서 파생된 일부 이미지를 제거했다고 설명한다. 로컬 ZIP과 해제본의 일치는 확인했지만 원격 ZIP과의 동일성을 검증한 것은 아니다.
- Idylwild 5종과 Kenney 4종의 공식 페이지 라이선스를 확인했다. Kenney ZIP 내 `License.txt`도 보존했다. CC0 표시를 이미지 제작 경위에 대한 독립적인 보증으로 확대하지 않는다.
- [DCSS export](https://github.com/crawl/tiles)의 `README.md`, `ARTISTS.md`, `TILES_UNDER_UNKNOWN_LICENSE.md`를 함께 확인했다. 전체 게임 저장소가 아닌 Nov-2015 CC0 export의 item 경로만 수집했고, 고정 revision의 Git blob hash 673개 모두 일치했다. 목록의 불명확 라이선스 파일명과 일치한 수는 0이다. 목록 자체는 과거 기준이라는 제한이 있다.
- [32×32 Items](https://opengameart.org/content/32x32-items)는 CC-BY 3.0, 저작자 Kemono. [Basic RPG Item Icons Remix](https://opengameart.org/content/basic-rpg-item-icons-remix)는 CC-BY 3.0, 원저작자 yinakoSGA, 배경 제거·크기 변경 AntumDeluge. 사용할 경우 저작자·원문·라이선스와 변형 내역을 함께 기록해야 한다. [CC-BY 3.0 원문 요약](https://creativecommons.org/licenses/by/3.0/).
- [CC0 Food Icons](https://opengameart.org/content/cc0-food-icons)는 여러 출처를 모았고 다른 크기를 32px로 조정했다고 명시한다. 연결된 [LPC Food 원본](https://opengameart.org/node/98209)은 전체 묶음 CC-BY-SA/GPL 및 일부 개별 타일 CC0를 안내한다. `data/research/licenses/lpc-food-v1.zip`의 `CREDITS-food.txt`에서 항목별 라이선스가 실제로 다름을 확인했다. **수집한 하위 파일과 CC0 항목의 대응을 완료하기 전 전체를 CC0로 확정하지 않는다.** 다른 집계 팩도 TXT의 원저작자별 출처와 변형을 대조할 필요가 있다.
- 사진 기반 RPG Icon Base와 OCAL/Glitch/Wikimedia 파생물은 원본을 보존하되 픽셀 아트 기본 실험에서 분리한다. 해상도만 보고 수작업 픽셀 이미지로 인정하지 않는다.

## HF 조사의 범위와 제한

사용자 정책 확정 전에 4종을 각 128개씩 조사했다. 전체 shard 다운로드는 하지 않았다. Atlas/172K/PixelGPT는 seed 42로 고른 16행 블록 8개, Krea는 native sprite 무작위 128개다. 블록 표본은 클래스 분포·결함률의 불편추정량이 아니므로 전체 비율로 일반화하지 않는다. 현재 추가 합성 표본 수집은 기본 비활성화했다.

| 출처 | 공개 metadata/card의 규모·조건 | 검토 결과 |
|---|---|---|
| [Pixel-art Icons](https://huggingface.co/datasets/azyg0s/pixelart-172k) | train 172,330; validation 합계 1,634; 166 category; MIT 표시 | 무기·방어구·포션·음식·재료 category 존재. 캡션은 검증 라벨이 아니고 일부 magenta 점·불규칙 픽셀 격자 경고가 있음. category filter API 실패로 목표 클래스 표본 추출은 미완료 |
| [Pixel Art Atlas](https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m) | 합계 1,409,404; 6,194 subject; CC0 표시 | 아이템 subject와 비아이템이 혼재. 16→32 정수 확대는 픽셀 격자를 유지하나 32 원본과 밀도 차이. source model 권리 별도 |
| [PixelGPT](https://huggingface.co/datasets/unstonio/pixelgpt-24x24-20k) | 20,000; 133 category·19 family; CC-BY 4.0 표시 | 합성/curated 제작 경위 추가 검토 필요. 받아온 뷰어 JPEG는 원본 PNG·5색 팔레트 검증 근거가 될 수 없음 |
| [Krea 2 16×16](https://huggingface.co/datasets/Angais/krea-2-pixelart-16x16-synthetic) | 550; 321 object·32 group; CC0 표시 | 512px preview 대신 `sprites/`의 16×16 수집. Atlas와 표본 사이 동일 픽셀 0이나 전체 중복/독립성은 미확정 |

[Krea 모델 라이선스](https://www.krea.ai/krea-2-licensing)의 조건과 데이터셋 표시 라이선스가 동일한 권리 범위를 뜻한다고 가정하지 않는다. Atlas 카드도 저작자가 포기할 수 있는 권리 범위와 별도 계약을 구분한다. 현재 이들 자료는 학습에 사용하지 않으므로 추가 권리 해석·전수 추출을 진행하지 않았다.

## 구현 파일과 실행 검증

추가 수집/분석: `configs/data_candidates.json`, `scripts/collect_candidates.py`, `scripts/collect_dcss_items.py`, `scripts/sample_hf_candidates.py`, `scripts/analyze_candidates.py`.

보존한 초안: `scripts/prepare_dataset.py`, `src/{__init__,dataset,unet,diffusion,utils,train,sample}.py`. 현재 추가 데이터 전체를 처리하는 최종 학습 파이프라인으로 확장하지 않았다.

검증: `tests/test_data_and_model.py`, `scripts/probe_model.py`. 문서: 본 문서, `assignment_requirements.md`, `model_design.md`, `ai_usage.md`, `sources.csv`, `dataset_candidates.csv`, `downloads.csv`, README. 기존 `data_review_2026-11-18.md`에는 과거 중단 기록임을 명시했다. `.gitignore`, `configs/base.json`, `scripts/audit_images.py` 변경도 남아 있다.

| 실제 실행 | 결과 |
|---|---|
| 최초 원본 audit | 496개 정상, 34×34 RGBA, 정확 중복 1, 알파 부분값 2, 외곽 접촉 149 |
| 기존 후보 전처리 및 loader 검사 | 495개 임시 RGB tensor, [-1,1], 32×32; 보존, 최종화 안 함 |
| 전체 raw audit | 4,539개 `.png` 경로; PNG 4,411 통과, JPEG 128 형식 불일치. 내부 3.232초 |
| 후보 분석 최신 실행 | 4,539 decode 성공, 별도 참조 8개 성공; 2,832 파생 셀; 내부 13.589초 |
| DCSS 다운로드 | 673/673 Git blob hash 일치, 라이선스 파일명 대조 충돌 0 |
| 오프라인 단위 테스트 | audit의 손상/중복, 원본 보존, ZIP 경로 이탈 방지, 보이는 픽셀 중복, 합성 정책, 실제 형식, U-Net shape, DDPM 첫 역단계 검증 |
| GPU 임의 텐서 검사 | batch32 FP32, 4,668,739 parameters, forward/backward 0.471초, max allocated 385.12 MiB; 가중치 변경 없음 |

GPU 수치는 학습·샘플링 전체의 실측이 아니며 Adam 상태는 포함하지 않는다. 모델 학습, 실제 데이터 optimizer step, 학습 결과 생성, resume 통합 테스트는 미실행이다. 추가 패키지는 설치하지 않았다. PDF는 기존 번들 pypdf/Poppler로 4쪽 추출·렌더 확인했다. 최초 언급된 Downloads의 `CODEX_TASK_2026-11-18.md`는 현재 경로에서 찾지 못했으며, 대화에 제공된 작업 지시와 과제 PDF를 기준으로 정리했다.

최종 검증: 단위 테스트 **8개 통과(0.146초)**, Python compileall 통과. 최초 audit의 원본 PNG 496개는 모두 존재하고 SHA-256 변경 0개다. 다만 초기 검사 때 존재한 `data/raw/496_RPG_icons.zip`은 최종 확인 시 해당 경로에서 더 이상 발견되지 않았다. 이 작업에서 ZIP 삭제·이동 명령을 실행하지 않았으며 원인은 확인되지 않았다. 다운로드 목록의 해당 해시는 초기 측정 이력이다. 현재 압축파일은 52개이고, 이미 해제된 원본 496개는 변경 없이 남아 있다.

## 결정 상태와 다음 구현

전처리에서 확정된 것은 원본 보존·출처 분리·합성 데이터 제외 정책이다. RGB/RGBA, 배경색, 34→32 변환, 증강, 최종 split은 미확정이다. 선택지와 근거 및 U-Net Tensor Shape·시간 임베딩·skip·노이즈 스케줄·loss·메모리 검토는 [model_design.md](model_design.md)에 있다.

다음 작업은 소형 후보의 전수 의미 검토와 변형 그룹/라벨 작성, 추가 음식·재료의 파일별 출처 매핑, 데이터 구성과 전처리 확정, 데이터로더 수정이다. 그 후 무조건부/조건부를 선택하고 강의 범위를 대조한다. 과제 요구에 맞춘 0 epoch와 매 epoch 샘플·시간 기록, checkpoint/resume 통합 테스트를 마쳐야 한다. GPU 학습 시작은 별도 다음 단계이며 이번 작업에서는 진행하지 않았다.

## 재현 명령 및 커밋 권고

프로젝트 루트 PowerShell에서 실행한다. 아래 명령에는 모델 학습이 없다. 수집 파일은 Git에서 제외되므로 다른 PC에서는 다운로드 주소 또는 수집 스크립트를 이용한다. 원본과 다른 파일로 덮어쓰려 하면 수집 스크립트가 중단한다. Kenney는 공식 페이지의 Download 또는 쿠키/Referer를 사용하는 스크립트가 새 서명 URL을 얻는다. 저장 위치는 `data/raw/<source_id>/downloads`와 `extracted`다. DCSS는 고정 revision의 item 경로에 저장한다.

```powershell
.\.venv\Scripts\python.exe scripts\audit_images.py --input data\raw --output results\all_raw_audit.csv
.\.venv\Scripts\python.exe scripts\analyze_candidates.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\probe_model.py
# 필요한 출처만 다시 수집하는 예
.\.venv\Scripts\python.exe scripts\collect_candidates.py download --ids oga_idylwild_inventory
.\.venv\Scripts\python.exe scripts\collect_dcss_items.py
```

합성 표본의 기존 다운로드 이력은 `data/raw/hf_*/sample_manifest.json`과 `data/research/hf/`에 있다. 현재 사용 정책에서는 재수집할 필요가 없다. 각 실행의 기록이 덮이지 않도록 이후 수집 로그는 고유 파일명을 사용한다. DCSS 고정 export 보고서는 같은 고정 경로에 기록한다.

검토 후 1차로 데이터 조사·명세를, 이후 2차로 모델 구현·검증을 나누어 커밋하는 것을 권고한다. 실제 커밋·Push는 실행하지 않았다. 아래는 사용자가 실행할 명령 예이며, 아직 모델 초안임을 커밋 메시지에 남긴다.

```powershell
git add .gitignore configs/data_candidates.json docs scripts/audit_images.py scripts/analyze_candidates.py scripts/collect_candidates.py scripts/collect_dcss_items.py scripts/sample_hf_candidates.py
git diff --cached --stat
git commit -m "data: curate candidate assets and record dataset review"
# 위 변경을 검토한 다음 모델 초안과 설계 문서
git add README.md configs/base.json scripts/prepare_dataset.py scripts/probe_model.py src tests
git diff --cached --stat
git commit -m "model: add annotated DDPM draft and validation checks"
```
