# Data

이 디렉터리는 로컬 전용 데이터셋, manifest, 재사용 가능한 가공 산출물을 두는 용도다.

- 비공개 labeled query는 `.gitignore`에 지정한 Git 비추적 경로에 둔다.
- 레포에는 스키마, dataset card, 합성 fixture만 남긴다.
- dataset pipeline 실행 source of truth는 `scripts/conf/dataset/*.yaml`이다.
- 원시 CSV는 `data/raw/`, 가공 JSONL은 `data/processed/labeled_query_sets/`, split 결과는 `data/processed/splits/`, build state 결과는 `data/processed/prototype_build_states/`, centroid 결과는 `data/processed/prototype_packs/`에 저장한다.
- 각 실행의 상위 요약 manifest는 `data/processed/pipeline_runs/`에 저장한다.
- 실행별 평가 리포트, 비교 실험 결과, 시뮬레이션 산출물은 `data/`가 아니라 `runs/` 아래에 저장한다.
