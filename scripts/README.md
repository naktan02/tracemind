# Scripts

`scripts/`는 실험 실행, dataset workflow, result index, report/artifact helper를
제공하는 CLI/orchestration layer다.

알고리즘 core는 `methods/`, 공통 contract/domain은 `shared/`, production runtime
adapter는 `agent/`와 `main_server/`가 소유한다. `scripts`는 config를 읽고, runtime
context를 만들고, core를 호출하고, 산출물을 쓰는 얇은 실행 표면이다.

## What Belongs Here

- Hydra experiment entrypoint
- dataset materialization workflow
- run report ingest와 dashboard JSON export
- sweep, smoke, verification helper
- experiment-only artifact/report IO
- agent/main_server runtime을 실험에서 재사용하기 위한 bridge

## What Does Not Belong Here

- SSL/FSSL algorithm core
- local objective, pseudo-label policy, aggregation arithmetic
- shared payload schema 의미
- production FastAPI route, repository, persistent runtime state
- UI view-model 의미나 dashboard metric 계산

## Install

```bash
uv sync --extra dev --extra experiments
```

## Common Commands

Hydra compose 확인:

```bash
uv run python <entrypoint>.py --cfg job
```

기본 runtime override:

```bash
uv run python <entrypoint>.py execution_context/runtime_env=auto_local
```

Central SSL smoke:

```bash
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  run_controls/central_ssl/budget=smoke
```

Central PEFT supervised smoke:

```bash
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  run_controls/central_ssl/budget=smoke
```

Fixed-feature supervised baseline:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py
```

FL SSL smoke:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke
```

Dashboard index/export:

```bash
uv run python -m scripts.workflows.result_index.ingest \
  --runs-root runs \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

## Main Areas

| Path | Purpose |
|---|---|
| `experiments/` | central SSL, fixed-feature, FL SSL experiment entrypoints |
| `experiments/central/ssl_control/` | central PEFT/full text encoder supervised and SSL control |
| `experiments/central/fixed_feature_control/` | fixed-feature supervised and self-training baseline |
| `experiments/fl_ssl/` | FL SSL split, simulation, sweep entrypoint |
| `workflows/datasets/` | dataset asset, split, query SSL view materialization |
| `workflows/result_index/` | `runs` report ingest, SQLite index, dashboard JSON export |
| `support/query_ssl_text_encoder/` | central text encoder SSL runner/helper and artifact IO |
| `runtime_adapters/` | scripts에서 agent/main_server runtime을 재사용할 때 쓰는 bridge |
| `support/reporting/`, `support/artifacts/`, `codegen/` | 보조 helper와 generated type workflow |

## Experiment Guides

- [experiments/README.md](experiments/README.md)
  - 실험 entrypoint layer 전체 지도
- [experiments/central/ssl_control/README.md](experiments/central/ssl_control/README.md)
  - central SSL, PEFT/full text encoder supervised controls
- [experiments/central/fixed_feature_control/README.md](experiments/central/fixed_feature_control/README.md)
  - fixed-feature supervised/self-training baselines
- [experiments/fl_ssl/README.md](experiments/fl_ssl/README.md)
  - FL SSL simulation, materialized split, FedMatch/manual baseline
- [../apps/experiment_dashboard/README.md](../apps/experiment_dashboard/README.md)
  - result index와 dashboard cache 생성

## Read Path

1. 실행하려는 track의 README를 고른다.
2. 관련 `conf/entrypoints/**`와 `conf/strategy_axes/**`를 compose한다.
3. script entrypoint에서는 orchestration 흐름만 확인한다.
4. algorithm/core 의미는 `methods/`에서 본다.
5. payload schema 의미는 `shared/`에서 본다.
6. 결과 해석은 report artifact와 `scripts/workflows/result_index/` schema를 기준으로 한다.

## Boundary Rules

- 운영 후보 알고리즘을 `scripts`에 먼저 만들고 나중에 `methods`로 복사하지 않는다.
- `scripts`는 `agent.src`나 `main_server.src`를 직접 import하지 않는다.
- runtime 재사용은 `scripts/runtime_adapters/`의 capability bridge로 드러낸다.
- 실행 조합과 파라미터의 source of truth는 `conf/`다.
- script helper는 해당 entrypoint가 직접 쓰는 범위까지만 둔다.
- method, trainable surface, teacher 구현 이름으로 scripts runner가 분기하지 않는다.

현재 작업자는 이 README, 관련 experiment README, `conf/README.md`를 먼저 본다.
