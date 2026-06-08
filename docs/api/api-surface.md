# TraceMind API Surface

이 문서는 현재 FastAPI route 표면을 빠르게 찾기 위한 문서다.

요청/응답 payload 필드의 최종 source of truth는 각 route의 Pydantic 모델,
`shared/src/contracts/*.py`, 그리고 agent-local raw text 계약의 경우
`agent/src/contracts/*.py`다. 이 문서는 endpoint 위치와 책임 경계만 요약한다.

## 1. App Entry Points

| 앱 | import path | 기본 역할 |
|---|---|---|
| Agent API | `agent.src.api.main:app` | 로컬 수집, inference, captured/local state, training participation, family/wellbeing output |
| Main Server API | `main_server.src.api.main:app` | FL round orchestration, aggregation/publication |

로컬 실행 예시:

```bash
uv run uvicorn main_server.src.api.main:app --reload --port 8000
uv run uvicorn agent.src.api.main:app --reload --port 8001
```

## 2. API Boundary Rules

| 규칙 | 의미 |
|---|---|
| raw text는 agent local boundary에 남긴다 | 서버 API는 captured text 원문과 generated view를 직접 읽지 않는다 |
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
| POST | `/api/v1/captured-text/events` | 브라우저 확장 captured text event를 agent-local raw 저장소에 저장 | `agent/src/api/captured_text.py` |
| POST | `/api/v1/captured-text/batch` | 최대 100개 captured text event 일괄 raw 저장 | `agent/src/api/captured_text.py` |
| GET | `/api/v1/captured-text/status` | captured text raw 저장/view generation/analysis job 상태 조회 | `agent/src/api/captured_text.py` |
| GET | `/api/v1/captured-text/debug-job/status` | captured text view generation/debug job과 analysis 상태 조회 | `agent/src/api/captured_text.py` |
| POST | `/api/v1/captured-text/debug-job/config` | captured text debug job 주기 실행 on/off 설정 | `agent/src/api/captured_text.py` |
| POST | `/api/v1/captured-text/debug-job/run-view-generation` | pending captured text weak/strong view 생성 후 weak text 분류를 즉시 실행 | `agent/src/api/captured_text.py` |

주요 payload:

| Payload | Source |
|---|---|
| `IngestEventRequest` | `agent/src/api/ingest.py` |
| `IngestEventResponse` | `agent/src/api/ingest.py` |
| `QueryEvent` | `shared/src/domain/entities/inference/events.py` |
| `TypingSegmentPayload` | `agent/src/contracts/typing_segment_contracts.py` |
| `TypingSegmentIngestResponsePayload` | `agent/src/contracts/typing_segment_contracts.py` |
| `CapturedTextEventPayload` | `agent/src/contracts/captured_text_contracts.py` |
| `CapturedTextDebugJob*Payload` | `agent/src/contracts/captured_text_contracts.py` |

`TypingSegmentPayload`는 extension/collector -> local agent 전용 raw segment 계약이다.
`final_text`, `deleted_text`, `field_hint`, page context는 main_server API나 FL update
envelope으로 전달하지 않는다.

`CapturedTextEventPayload`와 debug job payload도 agent-local 계약이다. 원문,
page context, generated weak/strong view는 main_server나 FL update envelope으로
전달하지 않는다.
captured text ingest는 raw 저장만 수행하며, main_server current shared adapter나
agent FL active cache를 직접 읽지 않는다. debug job은 view generation 이후 generated
weak text를 agent-local inference pipeline에 넣어 `analysis_events`,
`analysis_category_scores`에 분류 결과를 저장한다. captured text 학습 입력은
generated weak/strong view source에서 시작한다.
학습 사용 여부와 사용 시각은 `training_usage_rows`의 source_kind/source_id/recorded_at
기록으로 추적한다.
Agent-local runtime DB는 기본적으로 하나의 SQLite 파일(`agent_local.db`)을 사용한다.
Captured text DB는 테스트 단계 destructive migration을 허용한다. 현재 정규화 구조는
`captured_text_events` raw 원문, `captured_text_view_generation_jobs` 처리 상태,
`captured_text_generated_views` weak/strong 산출물, `captured_text_analysis_jobs`
분석 대기/완료 상태를 분리하고, completed analysis job은 `analysis_events`를 참조한다.

### Agent Sync

| Method | Path | 역할 | Source |
|---|---|---|---|
| GET | `/api/v1/sync/shared-adapters/current` | agent local active shared adapter state 조회 | `agent/src/api/sync.py` |
| POST | `/api/v1/sync/shared-adapters/pull` | main server의 current shared adapter state를 local로 pull | `agent/src/api/sync.py` |

### Training

| Method | Path | 역할 | Source |
|---|---|---|---|
| POST | `/api/v1/training/run-current-task` | server active task를 읽어 local training 후 update upload | `agent/src/api/training.py` |
| GET | `/api/v1/training/status` | server active task 존재 여부 조회 | `agent/src/api/training.py` |

`run-current-task` route는 HTTP 요청/응답 변환만 맡고, active task 조회부터
shared adapter sync, Query SSL/FSSL local update upload까지의 실행 흐름은
`agent/src/services/training_runtime/current_task/agent_training_task_runner_service.py`가
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
| `ChildSupportConversation*` | `agent/src/contracts/child_support_contracts.py` |
| `ChildSupportProactivePromptPayload` | `agent/src/contracts/child_support_contracts.py` |
| `FamilySetup*`, `FamilyUnlock*` | `agent/src/contracts/family_access_contracts.py` |
| `WellbeingSignalSummaryPayload` | `agent/src/contracts/wellbeing_signal_contracts.py` |
| `WellbeingSignalTimeseriesPayload` | `agent/src/contracts/wellbeing_signal_contracts.py` |

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
| POST | `/api/v1/fl/rounds/active-manifest/initialize` | 선택된 shared adapter family의 initial state 생성 및 active manifest publish | `main_server/src/api/fl_rounds.py` |
| GET | `/api/v1/fl/rounds/active-state/current` | 서버 current manifest와 shared adapter state 조회 | `main_server/src/api/fl_rounds.py` |
| POST | `/api/v1/fl/rounds` | 서버 current manifest 기준 새 round open | `main_server/src/api/fl_rounds.py` |
| GET | `/api/v1/fl/rounds/{round_id}` | 특정 round 조회 | `main_server/src/api/fl_rounds.py` |
| POST | `/api/v1/fl/rounds/{round_id}/updates` | agent update submission accept | `main_server/src/api/fl_rounds.py` |
| POST | `/api/v1/fl/rounds/{round_id}/finalize` | round finalize와 aggregation/publication | `main_server/src/api/fl_rounds.py` |

주요 payload:

| Payload | Source |
|---|---|
| `RoundOpenRequestPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `RoundStrategyPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `RoundRecordPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `RoundFinalizeRequestPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `InitialSharedArtifactPublicationRequestPayload` | `main_server/src/services/federation/rounds/boundary/payloads.py` |
| `ModelManifestPayload` | `shared/src/contracts/model_contracts.py` |
| `CurrentSharedAdapterStatePayload` | `shared/src/contracts/adapter_contract_families/base.py` |

`RoundOpenRequestPayload.strategy`는 운영 round에서 local update profile, composed
SSL method, method-owned FSSL method, server update policy, aggregation backend
이름을 받는 일반 입력 표면이다. composed SSL 모드에서는 `ssl_method`를 사용하고,
method-owned FSSL 모드에서는 `fssl_method`를 사용한다. active strategy 전환 API는
`fssl_method`가 설정된 동안 `ssl_method`를 사용자 선택값으로 저장하지 않는다.
`strategy.parameter_overrides`의 bare key는 composed SSL method 파라미터로 해석되어
`query_ssl.*` scope로 정규화된다.
`objective_config`는 debug/advanced override용 raw task objective이며, 둘을 동시에
제공하면 거부한다.
| `TrainingUpdateSubmissionPayload` | `shared/src/contracts/training_contracts.py` |

## 5. API 변경 시 갱신 기준

| 변경 | 같이 확인할 것 |
|---|---|
| shared payload 변경 | `shared/src/contracts/*.py`, contract tests, producer/consumer, 관련 `docs/contracts/*` |
| agent-local raw text payload 변경 | `agent/src/contracts/*.py`, agent contract/API tests, generated app types |
| agent route 변경 | `agent/tests/unit/*_api.py`, `docs/api/api-surface.md`, relevant app consumer |
| main_server route 변경 | `main_server/tests/unit/*_api.py`, root integration tests, `docs/api/api-surface.md` |
| family/wellbeing route 변경 | `apps/family_extension`, generated types, 관련 shared wellbeing/family contract |
