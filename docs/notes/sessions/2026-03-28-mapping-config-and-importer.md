# 2026-03-28 Mapping Config And Importer

## User

데이터 원본 가져온거에 대해서 매핑시키는 코드 따로 해야지 C:\Users\PC\Desktop\ko-embed 여기에서는 그렇게 했엇는데 이렇게 하는게 더 맞지 않아? 내가 임의로 매칭 조정 가능하도록?

## Assistant

맞다. raw CSV는 그대로 두고, 별도 매핑 설정 파일과 importer 스크립트로 `LabeledQuerySet`을 만드는 구조가 더 맞다. 이렇게 해야 `raw_label -> mapped_label_4`를 나중에 파일만 수정해서 다시 변환할 수 있다.

## Assistant

구현 내용:

1. `data/mappings/ourafla_to_4cat.v1.toml`
   - ourafla 4클래스 매핑 설정
2. `data/mappings/cssrs_to_4cat.v1.toml`
   - CSSR-S 7단계 매핑 설정
   - 기본값은 `0 -> __exclude__`, `1~6 -> suicidal`
3. `scripts/build_labeled_query_set.py`
   - raw CSV와 mapping config를 읽어 JSONL/manifest 산출물 생성

검증:

1. `python3.11 -m compileall scripts/build_labeled_query_set.py`
   - 성공
2. `python3.11 scripts/build_labeled_query_set.py --raw-csv data/raw/mental_heath_unbanlanced.csv --mapping-config data/mappings/ourafla_to_4cat.v1.toml`
   - 성공
3. `python3.11 scripts/build_labeled_query_set.py --raw-csv data/raw/labeled_rSuicidewatch_posts.csv --mapping-config data/mappings/cssrs_to_4cat.v1.toml`
   - 성공

출력 확인:

1. ourafla manifest
   - 49612행 모두 포함
   - anxiety/depression/normal/suicidal로 그대로 매핑
2. CSSR-S manifest
   - 1170행 중 351행 제외
   - 819행이 `suicidal`로 매핑

현재 의미:

1. raw 원본은 유지된다.
2. mapping config만 바꿔서 재생성 가능하다.
3. CSSR-S `0`을 나중에 포함시키거나 다른 클래스로 돌리는 것도 config 수정만으로 가능하다.
