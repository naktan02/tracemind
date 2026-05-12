# Data

이 디렉터리는 로컬 전용 데이터셋, manifest, 재사용 가능한 가공 산출물을 두는 용도다.

- 비공개 labeled query는 `.gitignore`에 지정한 Git 비추적 경로에 둔다.
- 레포에는 스키마, dataset card, 합성 fixture만 남긴다.
- dataset pipeline 실행 source of truth는 `conf/execution_context/dataset_asset/*.yaml`와
  `conf/entrypoints/data_pipeline/*.yaml`이다.
- 새 dataset asset은 가능하면 `data/datasets/<dataset_id>/` 아래에 `raw`,
  `mapped`, `splits`, `query_ssl`, `views`, `pipeline_runs`를 모은다.
- 기존 ourafla 같은 legacy 자산은 `data/raw/`, `data/processed/` 아래 경로를
  유지한다.
- 새 model/prototype/adapter 산출물은 `data/artifacts/`, cache성 파일은
  `data/cache/` 아래에 둔다.
- 실행별 평가 리포트, 비교 실험 결과, 시뮬레이션 산출물은 `data/`가 아니라 `runs/` 아래에 저장한다.
