# Experiment Dashboard

실험 결과를 선택적으로 비교하는 정적 research dashboard다.

이 앱은 서버를 요구하지 않는다. 먼저 SQLite index에서 dashboard JSON을 만든 뒤
정적 파일 서버로 열면 된다.

```bash
uv run python -m scripts.experiments.result_index.ingest \
  --runs-root runs/train_lora_ssl_classifier \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --reset \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json

python -m http.server 5175 -d apps/experiment_dashboard
```

브라우저에서 `http://127.0.0.1:5175`를 연다.

## 경계

- 원본 source of truth는 `runs/**/reports/report.json`과 각 run manifest다.
- SQLite와 `data/experiment_dashboard.json`은 재생성 가능한 비교용 cache다.
- 이 앱은 metric 계산이나 실험 설정 기본값을 소유하지 않는다.
