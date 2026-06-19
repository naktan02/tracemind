# Main Server

TraceMind의 중앙 federation runtime이다. FL round lifecycle, update acceptance,
aggregation, model manifest activation, server-owned artifact publication을 소유한다.

main server는 개인 raw text나 개인 해석 상태를 소유하지 않는다. agent가 local state와
local training을 소유하고, `methods/`가 aggregation/update 계산 core와 method policy를
소유한다. main server는 선택된 capability를 live round lifecycle에 연결한다.

## What It Does

- 현재 active `ModelManifest`를 기준으로 training task를 만든다.
- agent update를 수집하고 round 상태와 payload compatibility를 검증한다.
- selected aggregation backend를 실행해 다음 shared adapter/model state를 만든다.
- 다음 `ModelManifest`와 optional auxiliary artifact를 server-owned ref로 publish한다.
- 첫 round 전에 선택된 update family의 initial shared state를 publish한다.
- 다음 round에 적용할 active federation strategy pointer를 관리한다.

## Run The API

repo root에서 실행한다.

```bash
uv run uvicorn main_server.src.api.main:app --reload --host 127.0.0.1 --port 8000
```

## API Surface

| Area | Route module |
|---|---|
| Health | `main_server/src/api/health.py` |
| FL rounds | `main_server/src/api/fl_rounds.py` |
| Admin/runtime profile | `main_server/src/api/admin.py`, `agent_runtime_profile.py` |
| Prototypes | `main_server/src/api/prototypes.py` |
| App wiring | `main_server/src/api/main.py` |

Endpoint 목록과 owner 지도는 [../docs/api/api-surface.md](../docs/api/api-surface.md)를
기준으로 본다.

## Main Responsibilities

| Responsibility | Owner path |
|---|---|
| Round open/update/finalize orchestration | `src/services/federation/rounds/` |
| Active manifest storage and activation | `src/services/federation/rounds/active_manifest_service.py` |
| Initial shared state publication | `src/services/federation/rounds/initial_publication_service.py` |
| Server-owned auxiliary artifact refs | `src/services/federation/rounds/initial_state_artifact_publication.py` |
| Aggregation runtime adapter | `src/services/federation/rounds/aggregation/` |
| Payload adapter registry/wiring | `src/services/federation/rounds/payload_adapters/` |
| Acceptance checks | `src/services/federation/rounds/acceptance/` |
| Active strategy pointer | `src/services/federation/strategy/` |

## Round Lifecycle

Round runtime의 핵심 경로는 [src/services/federation/rounds/README.md](src/services/federation/rounds/README.md)다.

```text
active manifest
-> training task generation
-> agent update collection
-> acceptance and compatibility checks
-> aggregation
-> next state artifact publication
-> next ModelManifest activation
```

사람이 코드를 읽을 때는 아래 순서가 짧다.

```text
src/services/federation/rounds/round_lifecycle_service.py
-> src/services/federation/rounds/round_manager_service.py
-> src/services/federation/rounds/payload_adapters/registry.py
-> src/services/federation/rounds/aggregation/registry.py
```

API payload와 canonical request shape는 아래 순서로 본다.

```text
src/services/federation/rounds/boundary/models.py
-> src/services/federation/rounds/boundary/payloads.py
-> src/services/federation/rounds/boundary/mappers.py
```

## Strategy And Method Boundaries

`main_server`는 method-specific server policy를 직접 소유하지 않는다.

| Meaning | Owner |
|---|---|
| Reusable aggregation arithmetic | `methods/federated/aggregation/` |
| Adapter family delta interpretation and next-state projection | `methods/adaptation/<family>/` |
| FL SSL method identity and method-only policy | `methods/federated_ssl/<method>/` |
| Shared payload format and compatibility meaning | `shared/src/contracts/` |
| Live round lifecycle and server-owned refs | `main_server/src/services/federation/rounds/` |

새 FL SSL method를 추가하기 위해 `main_server`에 method 이름을 가진 round,
aggregation, server policy 파일을 만들지 않는다. method package가 의미를 소유하고,
main server는 generic runtime capability와 registry로 호출한다.

## State And Artifact Ownership

| State or artifact | Owner |
|---|---|
| Round state and submission status | main server |
| Active strategy pointer | main server |
| Active `ModelManifest` pointer | main server |
| Server-owned aggregate artifact ref | main server |
| Raw text, personal threshold, personal interpretation | agent |
| Local training source and usage ledger | agent |
| Payload schema and canonical meaning | `shared/src/contracts` |
| Algorithm objective and aggregation policy meaning | `methods/` |

`artifact_ref`와 `payload_ref`는 파일 경로가 아니라 server-owned ref로 다룬다. 실제
저장소 해석은 infrastructure repository에 위임한다.

## Developer References

- [src/services/README.md](src/services/README.md)
  - main server service package의 전체 경계
- [src/services/federation/rounds/README.md](src/services/federation/rounds/README.md)
  - FL round runtime 핵심 경로
- [src/services/federation/strategy/README.md](src/services/federation/strategy/README.md)
  - 다음 round에 적용할 active strategy pointer
- [../shared/src/contracts/README.md](../shared/src/contracts/README.md)
  - shared payload contract 해석
- [../methods/README.md](../methods/README.md)
  - method, aggregation, adaptation core 소유 경계

서버 내부 단위 검증은 `main_server/tests`에 둔다. multi-agent HTTP와 end-to-end
federation 검증은 root `tests/` integration 시나리오로 둔다.
