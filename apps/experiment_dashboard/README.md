# Experiment Dashboard

실험 결과를 선택적으로 비교하는 정적 research dashboard다.

이 앱은 서버를 요구하지 않는다. 먼저 SQLite index에서 dashboard JSON을 만든 뒤
정적 파일 서버로 열면 된다.

```bash
uv run python -m scripts.experiments.result_index.ingest \
  --runs-root runs/train_lora_ssl_classifier \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json

python -m http.server 5175 -d apps/experiment_dashboard
```

브라우저에서 `http://127.0.0.1:5175`를 연다.

`--reset`은 전체 cache를 다시 만들 때만 붙인다. 기본 ingest는 같은 `run_id`를
upsert하고 새 run을 누적한다. 실험 runner가 SQLite를 직접 쓰지 않기 때문에 새
실험이 끝난 뒤 위 ingest/export 명령을 다시 실행해야 화면에 반영된다.

export는 `projection_artifacts`의 UMAP/PCA PNG를
`apps/experiment_dashboard/data/artifacts/` 아래로 복사한다. 이 디렉터리와
`experiment_dashboard.json`은 재생성 가능한 cache라서 git에 올리지 않는다.

왼쪽 track navigation은 `Central SSL`과 `FL SSL`을 분리한다. 현재 기존 화면은
`Central SSL` track이며 method, split, eval set을 고른 뒤 여러 run과 여러 metric을
동시에 선택해 bar/line chart로 비교할 수 있다. per-class metric과 confusion matrix,
projection image는 첫 번째 선택 run을 상세 대상으로 사용한다.

`FL SSL` track은 중앙 SSL ranking과 섞지 않는 별도 표면이다. result index export는
`fl_ssl_runs`를 별도 view-model로 만들고, 이 패널은 run-level `macro_f1`,
worst-client macro-F1, ECE, communication cost, per-client variance를 소비한다.
round/client 시계열은 같은 방식으로 `fl_ssl_rounds`, `fl_ssl_client_rounds`를
추가해서 확장한다.

## 경계

- 원본 source of truth는 `runs/**/reports/report.json`과 각 run manifest다.
- SQLite와 `data/experiment_dashboard.json`은 재생성 가능한 비교용 cache다.
- 이 앱은 metric 계산이나 실험 설정 기본값을 소유하지 않는다.
