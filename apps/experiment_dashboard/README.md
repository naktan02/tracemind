# Experiment Dashboard

실험 결과를 선택적으로 비교하는 정적 research dashboard다.

이 앱은 서버를 요구하지 않는다. 먼저 SQLite index에서 dashboard JSON을 만든 뒤
정적 파일 서버로 열면 된다.

```bash
uv run python -m scripts.experiments.result_index.ingest \
  --runs-root runs \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json

python -m http.server 5175 -d apps/experiment_dashboard
```

브라우저에서 `http://127.0.0.1:5175`를 연다.

전체 index를 버리고 `runs/` 아래 중앙 SSL/FL SSL report를 다시 훑어 만들 때는
`--reset`을 붙인다.

```bash
uv run python -m scripts.experiments.result_index.ingest \
  --runs-root runs \
  --reset \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

`--reset`은 SQLite index row만 비우고 `runs/` 원본 artifact는 삭제하지 않는다.
기본 ingest는 같은 `run_id`를 upsert하고 새 run을 누적한다. 실험 runner가
SQLite를 직접 쓰지 않기 때문에 새 실험이 끝난 뒤 ingest/export 명령을 다시
실행해야 화면에 반영된다.
기본 `--runs-root runs` ingest는 `runs/_smoke/**` 아래 smoke/test report를
제외한다. smoke 결과를 따로 점검해야 할 때만 `--runs-root runs/_smoke`로
명시해서 별도 index/cache에 적재한다.
index는 report의 `run_control.budget_name`과 output root도 보존하므로, artifact
경로가 바뀌어도 실행 tier를 필터에서 확인할 수 있다.

export는 `projection_artifacts`의 UMAP/PCA PNG를
`apps/experiment_dashboard/data/artifacts/` 아래로 복사한다. 이 디렉터리와
`experiment_dashboard.json`은 재생성 가능한 cache라서 git에 올리지 않는다.

왼쪽 track navigation은 `Central SSL`과 `FL SSL`을 분리한다. 현재 기존 화면은
`Central SSL` track이며 method, split, eval set을 고른 뒤 여러 run과 여러 metric을
동시에 선택해 bar/line chart로 비교할 수 있다. per-class metric과 confusion matrix,
projection image는 첫 번째 선택 run을 상세 대상으로 사용한다.

`FL SSL` track은 중앙 SSL ranking과 섞지 않는 별도 표면이다. result index export는
FL report를 dashboard 전용 view-model로 평탄화한다.

- `fl_ssl_runs`: run-level `macro_f1`, worst-client macro-F1, ECE,
  communication cost, per-client variance, labeled exposure/unique labeled counts.
- `fl_ssl_rounds`: post-aggregation global validation curve와 round runtime/cost.
- `fl_ssl_client_rounds`: round별 client local update 통계. per-client accuracy가
  아니라 candidate/accepted/update norm/payload/time 상태다.
- `fl_ssl_client_validations`: final global model을 client validation split별로 평가한
  macro-F1/loss/ECE.
- `fl_ssl_client_splits`: client별 non-IID labeled/unlabeled label distribution.

기존 LoRA-classifier FL run 중 validation scorer가 `prototype_similarity`인 결과는
round별 global validation curve가 평평할 수 있다. 이 scorer는 shared
LoRA/classifier state를 직접 읽지 않기 때문에, client update/aggregation artifact가
생성돼도 `macro_f1`, `accuracy_top_1`, `loss`가 전 라운드 동일하게 기록될 수 있다.
새 LoRA-classifier FL run은 `lora_classifier_eval` validation을 사용해야 global
LoRA/head artifact가 실제 성능 곡선에 반영된다.

## 경계

- 원본 source of truth는 중앙 SSL `runs/**/reports/report.json`, FL SSL
  `runs/**/reports/fl_ssl_main_comparison.report.json`, 각 run manifest다.
  `runs/_smoke/**`는 검증용 산출물이며 기본 dashboard ingest에서는 제외된다.
- SQLite와 `data/experiment_dashboard.json`은 재생성 가능한 비교용 cache다.
- 이 앱은 metric 계산이나 실험 설정 기본값을 소유하지 않는다.
