# Experiment Dashboard

실험 결과를 비교하는 정적 research dashboard다. 별도 backend 서버를 실행하지 않고,
`runs/` 아래 report를 dashboard JSON으로 export한 뒤 정적 파일 서버로 연다.

이 앱은 metric 계산이나 실험 설정 기본값을 소유하지 않는다. 원본 결과의 source of
truth는 중앙 SSL/FSSL run report와 manifest다.

## What It Shows

- 중앙 SSL method와 supervised baseline 비교
- FL SSL run-level metric, round curve, client validation, client split 비교
- 선택한 run들의 metric bar/line chart
- per-class metric, confusion matrix, projection image
- method, model, checkpoint, label budget, split, seed 같은 faceted filter

## Build The Dashboard Cache

`runs/` 아래 report를 읽어 SQLite index와 정적 dashboard JSON을 만든다.

```bash
uv run python -m scripts.workflows.result_index.ingest \
  --runs-root runs \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

기본 ingest는 같은 `run_id`를 upsert하고 새 run을 누적한다. 실험 runner가 SQLite를
직접 쓰지 않기 때문에 새 실험이 끝난 뒤에는 ingest/export 명령을 다시 실행해야
dashboard에 반영된다.

기본 `--runs-root runs` ingest는 `runs/_smoke/**` 아래 smoke/test report를 제외한다.
smoke 결과를 따로 점검해야 할 때만 별도 runs root를 명시한다.

```bash
uv run python -m scripts.workflows.result_index.ingest \
  --runs-root runs/_smoke \
  --db data/processed/experiment_index/smoke_experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/smoke_experiment_dashboard.json
```

## Serve The Dashboard

```bash
python -m http.server 5175 -d apps/experiment_dashboard
```

브라우저에서 `http://127.0.0.1:5175`를 연다.

## Rebuild From Scratch

전체 SQLite index row를 비우고 `runs/` 아래 중앙 SSL/FL SSL report를 다시 훑어
dashboard cache를 만들 때는 `--reset`을 붙인다.

```bash
uv run python -m scripts.workflows.result_index.ingest \
  --runs-root runs \
  --reset \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

`--reset`은 SQLite index row만 비운다. `runs/` 원본 artifact는 삭제하지 않는다.

## Views

| View | Purpose |
|---|---|
| `Central SSL` | 중앙 SSL control run을 method, split, eval set, metric 기준으로 비교 |
| `Supervised` | 중앙 지도학습 baseline만 따로 필터링해 비교 |
| `전체` | 중앙 SSL/지도학습과 FL SSL 결과를 같은 표/그래프 화면에서 탐색 |
| `FL SSL` | FL SSL main comparison run, round, client validation, client split 확인 |

`Central SSL`과 `Supervised`에서는 per-class metric, confusion matrix, projection
image가 첫 번째 선택 run을 상세 대상으로 사용한다.

`전체` view는 표와 그래프 중심의 통합 탐색 화면이다. 중앙 SSL ranking과 FL SSL
ranking을 같은 의미로 합치지는 않는다.

## Source Layout

| Path | Role |
| --- | --- |
| `src/main.js` | data loading, event binding, track/page render order |
| `src/features/central_ssl/` | 중앙 SSL control 화면 상태, 필터, selector |
| `src/features/fl_ssl/` | FL SSL main comparison 화면 상태, 필터, selector |
| `src/ui/` | track에 묶이지 않는 chart, control, table primitive |
| `src/shared/` | HTML escape, metric/number formatting, faceted filter helper |
| `src/data/` | dashboard JSON loading과 old bundle field normalizer |

`localStorage`는 alias/color 같은 표시 preference만 저장한다. 실험 결과의 source of
truth는 항상 run report와 result-index export다.

## FL SSL Tables

FL SSL track은 result-index export가 만든 dashboard 전용 view-model을 읽는다.

| Table | Meaning |
|---|---|
| `fl_ssl_runs` | run-level `macro_f1`, worst-client macro-F1, ECE, communication cost, per-client variance, labeled exposure |
| `fl_ssl_rounds` | post-aggregation global validation curve와 round runtime/cost |
| `fl_ssl_client_rounds` | round별 client local update 통계. candidate/accepted/update norm/payload/time 상태를 본다 |
| `fl_ssl_client_validations` | final global model을 client validation split별로 평가한 macro-F1/loss/ECE |
| `fl_ssl_client_splits` | client별 non-IID labeled/unlabeled label distribution |

FL 상단의 `Run Filters`는 먼저 사용할 필터 축을 고르고, 선택한 축에서만 세부 값을
좁힌다. 기본 축은 round count와 method이며 regularizer, labeled/unlabeled, label
budget, local epochs, client count, adapter, aggregation, seed 등을 추가할 수 있다.

## Projection Artifacts

export는 report manifest의 `projection_artifacts`가 가리키는 UMAP/PCA PNG를
`apps/experiment_dashboard/data/artifacts/` 아래로 복사한다.

```text
apps/experiment_dashboard/data/experiment_dashboard.json
apps/experiment_dashboard/data/artifacts/
```

이 파일들은 재생성 가능한 cache라서 git에 올리지 않는다.

중앙 PEFT projection writer는 eval set별로 UMAP을 먼저 시도하고 실패하면 PCA로
fallback하는 단일 2D projection 이미지를 만든다. UMAP과 PCA 이미지를 동시에 만드는
구조가 아니며, 실제 reducer는 manifest와 dashboard 이미지 label의 `reducer` 값으로
확인한다.

## Data Ownership

- 중앙 SSL 원본 결과: `runs/**/reports/report.json`
- FL SSL 원본 결과: `runs/**/reports/fl_ssl_main_comparison.report.json`
- run 조건과 artifact 위치: 각 run manifest
- dashboard SQLite/JSON/cache: 재생성 가능한 비교용 cache
- metric 의미와 result schema: `scripts/workflows/result_index`
- 화면 조합과 local display preference: `apps/experiment_dashboard`

committed historical dashboard cache가 old field를 담을 수 있어 app loader에는 좁은
normalizer가 남아 있다. 새 export producer는 current field만 생산한다.
