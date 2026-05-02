# TraceMind Server Monorepo

TraceMind는 `personalized local inference + federated shared model improvement`
를 함께 다루는 monorepo다. 이 저장소는 단일 앱이 아니라 공용 계약,
agent 로컬 runtime, server orchestration, 실험층을 역할별로 분리해 둔다.

처음 읽을 때는 코드보다 문서 진입점을 먼저 보는 편이 빠르다.

1. [docs/ai_context_manifest.yaml](docs/ai_context_manifest.yaml)
2. [docs/execution_index.md](docs/execution_index.md)
3. [docs/architecture/system-overview.md](docs/architecture/system-overview.md)
4. [docs/operations/local-runbook.md](docs/operations/local-runbook.md)
5. [plan.md](plan.md)
6. [docs/project_execution_plan.md](docs/project_execution_plan.md)
7. [shared/src/contracts/README.md](shared/src/contracts/README.md)

## Current Runtime Shape

| 구성요소 | 역할 |
|---|---|
| `shared/` | agent, main_server, scripts가 함께 읽는 contract와 canonical rule |
| `agent/` | 로컬 추론, query buffer, local training, wellbeing/family output |
| `main_server/` | FL round lifecycle, aggregation, artifact publication, experiment workspace backend |
| `scripts/` | dataset/prototype/classifier/LoRA/FL simulation 실행 조합 |
| `apps/` | experiment web과 family extension UI shell |
| `tests/` | package unit, cross-boundary integration/e2e, architecture guard |

현재 의존성 source of truth는 `pyproject.toml`과 `uv.lock`이다.
현재 저장소에는 Docker Compose나 `infra/` manifest가 없다.

## AI Harness Quick Start

Codex CLI와 VS Code Codex extension을 같이 쓸 때는 아래 계층을 기준으로
운영한다.

- project-scoped 실행 설정: [.codex/config.toml](.codex/config.toml)
- repo-wide 규칙: [AGENTS.md](AGENTS.md)
- task별 문맥 라우팅: [docs/ai_context_manifest.yaml](docs/ai_context_manifest.yaml)
- 반복 workflow: [.codex/skills](.codex/skills)
- 하네스 자체 유지보수 시만: [docs/ai_harness_operating_model.md](docs/ai_harness_operating_model.md)

## Quick Start

```bash
uv sync --extra dev --extra experiments
uv run pytest
uv run ruff check main_server/src agent/src shared/src scripts tests
```

로컬 agent 설정이 필요하면 `agent/.env.example`을 `agent/.env`로 복사해서
수정한다. `agent/.env`는 커밋하지 않는다.

```bash
cp agent/.env.example agent/.env
```

Main server:

```bash
uv run uvicorn main_server.src.api.main:app --reload --port 8000
```

Agent:

```bash
uv run uvicorn agent.src.api.main:app --reload --port 8001
```

Health check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/api/v1/system/health
```

## Code Reading Quick Start

- agent 경로: [agent/src/services/README.md](agent/src/services/README.md)
- main_server 경로: [main_server/src/services/README.md](main_server/src/services/README.md)
- round orchestration 경로: [main_server/src/services/federation/rounds/README.md](main_server/src/services/federation/rounds/README.md)
- experiment 경로: [scripts/experiments/README.md](scripts/experiments/README.md)

## Documentation Map

| 문서 | 역할 |
|---|---|
| [docs/architecture/system-overview.md](docs/architecture/system-overview.md) | 현재 런타임, 활성 레일, 코드 소유 경계 |
| [docs/api/api-surface.md](docs/api/api-surface.md) | agent/main_server FastAPI endpoint 표면 |
| [docs/operations/local-runbook.md](docs/operations/local-runbook.md) | 로컬 실행, GPU preflight, smoke 절차 |
| [docs/quality/test-strategy.md](docs/quality/test-strategy.md) | 테스트 층과 보호 범위 |
| [docs/governance/document-governance.md](docs/governance/document-governance.md) | 문서 class, source of truth, 갱신 규칙 |
| [shared/src/contracts/README.md](shared/src/contracts/README.md) | payload 해석과 contract 파일 지도 |
