# TraceMind API Surface

이 문서는 현재 FastAPI route 표면을 빠르게 찾기 위한 문서다.

요청/응답 payload 필드의 최종 source of truth는 각 route의 Pydantic 모델과 `shared/src/contracts/*.py`다. 이 문서는 endpoint 위치와 책임 경계만 요약한다.

## 1. App Entry Points

| 앱 | import path | 기본 역할 |
|---|---|---|
| Agent API | `agent.src.api.main:app` | 로컬 수집, inference, query/local state, training participation, family/wellbeing output |
| Main Server API | `main_server.src.api.main:app` | FL round orchestration, prototype publication |

로컬 실행 예시:

```bash
uv run uvicorn main_server.src.api.main:app --reload --port 8000
uv run uvicorn agent.src.api.main:app --reload --port 8001
```

## 2. API Boundary Rules

| 규칙 | 의미 |
|---|---|
| raw text는 agent local boundary에 남긴다 | 서버 API는 query buffer 원문을 읽지 않는다 |
| shared contract는 코드가 source of truth다 | API 문서가 payload 필드를 복제해 정본이 되지 않는다 |
| route는 orchestration edge다 | domain rule과 training/inference mechanism은 services/domain 계층에 둔다 |
| CORS/auth는 local app 노출 경계에서 별도 확인한다 | production auth/security hardening 문서는 아직 별도 canonical 문서가 없다 |

현재 route-level 인증 dependency는 명시적으로 두껍게 걸려 있지 않다. 외부 노출 전에는 API key, local-only binding, reverse proxy policy, family PIN/session 경계를 별도 security pass로 닫아야 한다.

## 3. Agent API

Agent app은 `agent/src/api/main.py`에서 router를 조합한다.

### Health

| Method | Path | 역할 | Source |
|---|---|---|---|
| GET | `/health` | agent health probe | `agent/src/api/health.py` |
| GET | `/api/v1/system/health` | family extension이 로컬 프로그램 상태 확인 | `agent/src/api/wellbeing.py` |

### Ingest

| Method | Path | 역할 | Source |
|---|---|---|---|
| POST | `/api/v1/ingest/event` | 단일 텍스트 이벤트를 inference pipeline으로 처리 | `agent/src/api/ingest.py` |
| POST | `/api/v1/ingest/batch` | 최대 100개 이벤트 일괄 처리 | `agent/src/api/ingest.py` |
| GET | `/api/v1/ingest/status` | 저장된 scored event 수 조회 | `agent/src/api/ingest.py` |
| POST | `/api/v1/typing-segments` | 브라우저 확장 typing segment를 agent-local inference pipeline으로 처리 | `agent/src/api/typing_segments.py` |
| POST | `/api/v1/typing-segments/batch` | 최대 100개 typing segment 일괄 처리 | `agent/src/api/typing_segments.py` |

주요 payload:

| Payload | Source |
|---|---|
| `IngestEventRequest` | `agent/src/api/ingest.py` |
| `IngestEventResponse` | `agent/src/api/ingest.py` |
| `QueryEvent` | `shared/src/domain/entities/inference/events.py` |
| `TypingSegmentPayload` | `shared/src/contracts/typing_segment_contracts.py` |
| `TypingSegmentIngestResponsePayload` | `shared/src/contracts/typing_segment_contracts.py` |

`TypingSegmentPayload`는 extension/collector -> local agent 전용 raw segment 계약이다.
`final_text`, `deleted_text`, `field_hint`, page context는 main_server API나 FL update
envelope으로 전달하지 않는다.

### Agent Sync

| Method | Path | 역할 | Source |
|---|---|---|---|
| GET | `/api/v1/sync/prototypes/current` | agent local active prototype pack 조회 | `agent/src/api/sync.py` |
| POST | `/api/v1/sync/prototypes/pull` | main server의 current prototype pack을 local로 pull | `agent/src/api/sync.py` |
| GET | `/api/v1/sync/shared-adapters/current` | agent local active shared adapter state 조회 | `agent/src/api/sync.py` |
| POST | `/api/v1/sync/shared-adapters/pull` | main server의 current shared adapter state를 local로 pull | `agent/src/api/sync.py` |

### Training

| Method | Path | 역할 | Source |
|---|---|---|---|
| POST | `/api/v1/training/run-current-task` | server active task를 읽어 local training 후 update upload | `agent/src/api/training.py` |
| GET | `/api/v1/training/status` | server active task 존재 여부 조회 | `agent/src/api/training.py` |

`run-current-task` route는 HTTP 요청/응답 변환만 맡고, active task 조회부터
shared/prototype sync, example build, update upload까지의 실행 흐름은
`agent/src/services/training/execution/agent_training_task_runner_service.py`가
소유한다.

주요 payload:

| Payload | Source |
|---|---|
| `RunCurrentTaskRequest` | `agent/src/api/training.py` |
| `RunCurrentTaskResponse` | `agent/src/api/training.py` |
| `TrainingTaskPayload` | `shared/src/contracts/training_contracts.py` |
| `TrainingUpdateEnvelopePayload` | `shared/src/contracts/training_contracts.py` |

### Family Access and Wellbeing

| Method | Path | 역할 | Source |
|---|---|---|---|
| POST | `/api/v1/child-support/messages` | 아이용 지원 대화 응답 생성 및 agent-local conversation 저장 | `agent/src/api/child_support.py` |
| GET | `/api/v1/child-support/proactive-prompt` | 아이 화면 진입 시 선제 발화 필요 여부 조회 | `agent/src/api/child_support.py` |
| GET | `/api/v1/family/setup/status` | 최초 setup 완료 여부 조회 | `agent/src/api/family_access.py` |
| POST | `/api/v1/family/setup` | child/parent PIN 최초 설정 | `agent/src/api/family_access.py` |
| POST | `/api/v1/family/unlock` | role별 PIN 잠금 해제 | `agent/src/api/family_access.py` |
| GET | `/api/v1/wellbeing/summary` | 현재 wellbeing summary 조회 | `agent/src/api/wellbeing.py` |
| GET | `/api/v1/wellbeing/timeseries` | 부모용 추이 조회 | `agent/src/api/wellbeing.py` |
| POST | `/api/v1/parent/unlock` | 부모 상세 화면용 PIN 검증 | `agent/src/api/wellbeing.py` |

주요 contract:

| Contract | Source |
|---|---|
| `ChildSupportConversation*` | `shared/src/contracts/child_support_contracts.py` |
| `ChildSupportProactivePromptPayload` | `shared/src/contracts/child_support_contracts.py` |
| `FamilySetup*`, `FamilyUnlock*` | `shared/src/contracts/family_access_contracts.py` |
| `WellbeingSignalSummaryPayload` | `shared/src/contracts/wellbeing_signal_contracts.py` |
| `WellbeingSignalTimeseriesPayload` | `shared/src/contracts/wellbeing_signal_contracts.py` |

`ChildSupportConversation*` raw message와 conversation 저장은 agent-local
SQLite 경계다. main_server API는 child-support 원문을 읽지 않는다.

## 4. Main Server API

Main server app은 `main_server/src/api/main.py`에서 router를 조합한다.

### Health

| Method | Path | 역할 | Source |
|---|---|---|---|
| GET | `/health` | main server health probe | `main_server/src/api/health.py` |

### FL Rounds

| Method | Path | 역할 | Source |
|---|---|---|---|
| GET | `/api/v1/fl/rounds/current` | 현재 active round 조회 | `main_server/src/api/fl_rounds.py` |
| GET | `/api/v1/fl/rounds/active-manifest/current` | 서버 current model manifest 조회 | `main_server/src/api/fl_rounds.py` |
| POST | `/api/v1/fl/rounds/active-manifest` | 초기/수동 model manifest 활성화 | `main_server/src/api/fl_rounds.py` |
| GET | `/api/v1/fl/rounds/active-state/current` | 서버 current manifest와 shared adapter state 조회 | `main_server/src/api/fl_rounds.py` |
| POST | `/api/v1/fl/rounds` | 서버 current manifest 기준 새 round open | `main_server/src/api/fl_rounds.py` |
| GET | `/api/v1/fl/rounds/{round_id}` | 특정 round 조회 | `main_server/src/api/fl_rounds.py` |
| POST | `/api/v1/fl/rounds/{round_id}/updates` | agent update submission accept | `main_server/src/api/fl_rounds.py` |
| POST | `/api/v1/fl/rounds/{round_id}/finalize` | round finalize와 aggregation/publication | `main_server/src/api/fl_rounds.py` |

주요 payload:

| Payload | Source |
|---|---|
| `RoundOpenRequestPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `RoundRecordPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `RoundFinalizeRequestPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `ModelManifestPayload` | `shared/src/contracts/model_contracts.py` |
| `CurrentSharedAdapterStatePayload` | `shared/src/contracts/adapter_contract_families/base.py` |
| `TrainingUpdateSubmissionPayload` | `shared/src/contracts/training_contracts.py` |

### Prototype Packs

| Method | Path | 역할 | Source |
|---|---|---|---|
| GET | `/api/v1/prototypes/current` | current prototype pack pointer와 payload 조회 | `main_server/src/api/prototypes.py` |
| GET | `/api/v1/prototypes/{prototype_version}` | version별 prototype pack 조회 | `main_server/src/api/prototypes.py` |
| POST | `/api/v1/prototypes/activate` | prototype pack 활성화 | `main_server/src/api/prototypes.py` |

주요 contract:

| Contract | Source |
|---|---|
| `PrototypePackPayload` | `shared/src/contracts/prototype_contracts.py` |
| `PrototypePackActivation*` | `shared/src/contracts/prototype_contracts.py` |

## 5. API 변경 시 갱신 기준

| 변경 | 같이 확인할 것 |
|---|---|
| shared payload 변경 | `shared/src/contracts/*.py`, contract tests, producer/consumer, 관련 `docs/contracts/*` |
| agent route 변경 | `agent/tests/unit/*_api.py`, `docs/api/api-surface.md`, relevant app consumer |
| main_server route 변경 | `main_server/tests/unit/*_api.py`, root integration tests, `docs/api/api-surface.md` |
| family/wellbeing route 변경 | `apps/family_extension`, generated types, `docs/family_extension_wellbeing_signal_mvp_plan.md` |
