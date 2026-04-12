# 2026-03-28 Dataset Download And Review

## User

ourafla는 이거야 https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset?utm_source=chatgpt.com

두가지 데이터셋 가져오고 cssr-s 매핑은 가져온 데이터 다시 확인하고 결정하자 가져오자

## Assistant

외부 데이터셋을 실제로 가져오는 순서가 맞는지 판단하려면, 먼저 저장소 안에 이미 잡아둔 `PrototypePack`/라벨셋 관련 구조와 `ourafla` 언급이 있는지 확인하고, 이어서 Kaggle 데이터셋 메타데이터를 공식 페이지 기준으로 보겠습니다.

확인 내용:

1. `data/README.md`
   - 원본 데이터는 Git 비추적 경로에 둔다.
2. `.gitignore`
   - `data/raw/`가 이미 비추적 경로다.
3. `shared/src/domain/entities/labeled_query.py`
   - `raw_label_scheme`, `raw_label`, `mapped_label_4` 구조가 이미 정의돼 있다.

## Assistant

원본을 Git 비추적 경로인 `data/raw/` 아래에 두는 게 저장소 규칙과 맞습니다. 이제 두 데이터셋을 실제로 받아오고, 바로 컬럼/라벨 구조만 확인해서 CSSR-S 매핑은 그 다음에 결정하겠습니다.

다운로드:

1. `data/raw/mental_heath_unbanlanced.csv`
2. `data/raw/mental_health_combined_test.csv`
3. `data/raw/labeled_rSuicidewatch_posts.csv`

비고:

1. Hugging Face 원본 다운로드는 sandbox 안에서 DNS가 막혀 실패했다.
2. 승인 경로로 재시도해 정상 다운로드했다.

## Assistant

다운로드 후 1차 구조 확인:

### ourafla

`mental_heath_unbanlanced.csv`

1. rows: 49612
2. labels:
   - Anxiety: 5503
   - Normal: 18391
   - Depression: 14506
   - Suicidal: 11212

`mental_health_combined_test.csv`

1. rows: 992
2. labels:
   - Anxiety: 248
   - Depression: 248
   - Normal: 248
   - Suicidal: 248

### CSSR-S Reddit

`labeled_rSuicidewatch_posts.csv`

1. rows: 1170
2. severity counts:
   - 0: 351
   - 1: 200
   - 2: 302
   - 3: 72
   - 4: 37
   - 5: 50
   - 6: 158

샘플을 몇 개 확인한 결과:

1. level 1~6은 자살 사고/행동 심각도 축으로 보인다.
2. level 0은 단순 normal이라고 보기 어렵고, distress/support-seeking 글도 섞여 있다.

현재 결론:

1. 두 데이터셋을 가져오는 결정은 맞았다.
2. `ourafla`는 4카테고리의 주 데이터셋으로 바로 사용 가능하다.
3. CSSR-S는 `raw_label_scheme=cssrs_7level.v1`, `raw_label=0..6`으로 먼저 보존해야 한다.
4. CSSR-S `0 -> normal` 자동 매핑은 아직 확정하면 안 된다.
