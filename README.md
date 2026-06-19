# TraceMind

TraceMind는 로컬 웰빙 신호 처리와 semi-supervised learning 실험을 함께 다루는
연구용 프로토타입입니다. 로컬 agent, FastAPI 서버, 재사용 가능한 학습 method,
실험 실행 스크립트, 결과 대시보드, 가족용 확장 UI를 하나의 monorepo에서 관리합니다.

이 저장소의 중심 목표는 두 가지입니다.

- 개인 원문 텍스트와 해석 상태는 로컬 agent 경계에 남긴다.
- 공유 가능한 모델 개선과 실험 결과는 contract, method, artifact, dashboard로
  재현 가능하게 다룬다.

## What You Can Do

- 로컬 agent API를 실행해 captured text, inference, wellbeing summary를 확인할 수 있습니다.
- 가족용 extension UI에서 아이/보호자 화면 흐름을 확인할 수 있습니다.
- 중앙 SSL, fixed-feature baseline, FL SSL simulation 실험을 실행할 수 있습니다.
- `runs/` 아래 report를 SQLite index와 정적 dashboard JSON으로 변환할 수 있습니다.
- experiment dashboard에서 중앙/연합 실험 결과, metric, per-class 결과, projection artifact를 비교할 수 있습니다.

## Repository Layout

| Path | Purpose |
|---|---|
| `shared/` | 여러 runtime이 함께 쓰는 contract, domain entity, canonical payload 해석 |
| `methods/` | SSL, adaptation, classification, FL aggregation 같은 재사용 계산 core |
| `conf/` | Hydra 실행 조합, dataset/model/runtime/strategy 파라미터 |
| `agent/` | 로컬 inference, captured text, local state, local training, wellbeing API |
| `main_server/` | FL round lifecycle, aggregation, artifact publication |
| `scripts/` | dataset, central SSL, FL SSL, result index, report workflow entrypoint |
| `apps/family_extension/` | 가족용 Chrome extension UI와 collector shell |
| `apps/experiment_dashboard/` | 실험 결과 비교용 정적 dashboard |
| `tests/` | unit, integration, architecture guard, cross-boundary 검증 |
| `docs/` | architecture, operations, contracts, quality, governance 문서 |

## Quick Start

Python 의존성은 `pyproject.toml`과 `uv.lock`이 기준입니다.

```bash
uv sync --extra dev --extra experiments
uv run pytest
```

Lint와 format 확인:

```bash
uv run ruff check main_server/src agent/src shared/src scripts tests
uv run ruff format --check main_server/src agent/src shared/src scripts tests
```

현재 저장소에는 Docker Compose 또는 별도 `infra/` manifest가 없습니다. 로컬 실행은
Python `uv`, FastAPI/Uvicorn, Vite app 기준입니다.

## Run The APIs

Main server:

```bash
uv run uvicorn main_server.src.api.main:app --reload --port 8000
```

Agent:

```bash
uv run uvicorn agent.src.api.main:app --reload --port 8001
```

로컬 agent 설정이 필요하면 `agent/.env.example`을 `agent/.env`로 복사합니다.
`agent/.env`는 커밋하지 않습니다.

```bash
cp agent/.env.example agent/.env
```

## Run The Family Extension UI

Agent API를 먼저 실행한 뒤 frontend를 실행합니다.

```bash
cd apps/family_extension
npm install
npm run dev
```

기본 agent API target은 `http://127.0.0.1:8001`입니다. 다른 주소를 쓰려면:

```bash
cd apps/family_extension
VITE_AGENT_API_BASE_URL=http://127.0.0.1:9001 npm run dev
```

Chrome extension build:

```bash
cd apps/family_extension
npm run build
```

## Run The Experiment Dashboard

먼저 `runs/` 아래 report를 dashboard용 index로 변환합니다.

```bash
uv run python -m scripts.workflows.result_index.ingest \
  --runs-root runs \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

정적 파일 서버로 dashboard를 엽니다.

```bash
python -m http.server 5175 -d apps/experiment_dashboard
```

브라우저에서 `http://127.0.0.1:5175`를 엽니다.

기존 SQLite index를 비우고 `runs/` 아래 report를 다시 훑어 dashboard cache를
만들려면 `--reset`을 붙입니다. 원본 run artifact는 삭제하지 않습니다.

```bash
uv run python -m scripts.workflows.result_index.ingest \
  --runs-root runs \
  --reset \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

## Main Workflows

### Local Wellbeing Flow

```text
captured text
-> agent-local scoring and evidence summary
-> wellbeing signal projection
-> child / parent UI payload
-> family extension UI
```

관련 시작점:

- [agent/README.md](agent/README.md)
- [agent/src/features/wellbeing/README.md](agent/src/features/wellbeing/README.md)
- [apps/family_extension/README.md](apps/family_extension/README.md)

### Experiment Flow

```text
Hydra config
-> experiment script
-> methods core
-> run artifacts and reports
-> result index
-> experiment dashboard
```

관련 시작점:

- [conf/README.md](conf/README.md)
- [methods/README.md](methods/README.md)
- [scripts/README.md](scripts/README.md)
- [apps/experiment_dashboard/README.md](apps/experiment_dashboard/README.md)

### Experiment Guides

| Experiment | Guide |
|---|---|
| 중앙 SSL / PEFT text encoder SSL | [scripts/experiments/central/ssl_control/README.md](scripts/experiments/central/ssl_control/README.md) |
| 중앙 PEFT / full text encoder 지도학습 | [scripts/experiments/central/ssl_control/README.md](scripts/experiments/central/ssl_control/README.md#supervised-controls) |
| 중앙 fixed-feature 지도학습과 classical self-training | [scripts/experiments/central/fixed_feature_control/README.md](scripts/experiments/central/fixed_feature_control/README.md) |
| FL SSL simulation / FedMatch 비교 | [scripts/experiments/fl_ssl/README.md](scripts/experiments/fl_ssl/README.md) |

## Documentation

처음 읽는 경우에는 [docs/README.md](docs/README.md)에서 필요한 문서만 고르는 편이
가장 빠릅니다.

| Document | Description |
|---|---|
| [docs/architecture/system-overview.md](docs/architecture/system-overview.md) | 현재 runtime 구성과 코드 소유 경계 |
| [docs/operations/local-runbook.md](docs/operations/local-runbook.md) | 로컬 실행, GPU preflight, smoke 절차 |
| [docs/quality/test-strategy.md](docs/quality/test-strategy.md) | 테스트 층과 보호 범위 |
| [docs/api/api-surface.md](docs/api/api-surface.md) | agent/main_server FastAPI endpoint 지도 |
| [shared/src/contracts/README.md](shared/src/contracts/README.md) | shared payload contract 해석 |

내부 작업자나 AI agent를 위한 상세 작업 지도는
[docs/execution_index.md](docs/execution_index.md)와 [AGENTS.md](AGENTS.md)에 있습니다.

## Safety And Scope

TraceMind는 연구/프로토타입 코드베이스입니다. 웰빙 신호와 child-support UI는
로컬 상태를 바탕으로 한 보조 안내를 목표로 하며, 의료 진단이나 긴급 대응 시스템을
대체하지 않습니다. 원문 텍스트와 개인 해석 상태는 agent-local boundary에 남기는
구조를 기본으로 합니다.
