# TraceMind System Overview

이 문서는 2026-04-25 기준 TraceMind의 현재 런타임, 코드 경계, 활성 연구/시스템 레일을 설명하는 canonical 개요다.

세부 payload 필드의 source of truth는 문서가 아니라 `shared/src/contracts/*.py`와 `shared/src/domain/entities/*`다.

## 1. 현재 목표

TraceMind는 `personalized local inference + federated shared model improvement`를 다루는 monorepo다.

현재 활성 순서는 아래와 같다.

```text
central fixed embedding + classifier seed
-> central SSL pooled/offline control
-> FL SSL non-IID main comparison
-> FL/runtime translation
```

핵심 원칙:

- 원문 텍스트와 개인 해석 상태는 agent 로컬에 남긴다.
- 전역 서버는 round lifecycle, aggregation, artifact publication을 소유한다.
- 공통 payload와 canonical payload 해석 규칙은 `shared`가 소유한다.
- 교체 가능한 알고리즘/method 계산 core는 `methods`가 소유한다.
- 논문 방법론은 `methods/federated_ssl/<method>/`가 사람이 읽는 시작점이고,
  method-only 변형은 이 폴더에 둔다. 두 개 이상 방법론에서 공유되는 계산은 축별
  methods 패키지로 승격한다.
- `agent`와 `main_server`는 method-specific 구현이 아니라 선택된 method core를
  실행하는 capability adapter만 소유한다.
- `scripts`는 운영 후보 코어를 소유하지 않고, 실험 조합과 실행 표면만 소유한다.
- `apps`는 contract/API consumer이며 source of truth가 아니다.

## 2. 시스템 구성요소

| 구성요소 | 역할 | 주요 코드 |
|---|---|---|
| Shared contract/domain | agent, main_server, scripts가 함께 읽는 canonical payload와 domain entity | `shared/src/contracts/*`, `shared/src/domain/entities/*` |
| Methods | SSL, adaptation, prototype, evaluation, FL aggregation의 교체 가능한 계산 core | `methods/*` |
| Hydra config | 실행 조합, strategy axis, track preset | `conf/*` |
| Agent API/runtime | 로컬 inference, query buffer, local training, wellbeing/family extension output | `agent/src/api/*`, `agent/src/services/*` |
| Main server API/runtime | FL round, aggregation, prototype publication | `main_server/src/api/*`, `main_server/src/services/*` |
| Scripts | dataset/prototype/classifier/LoRA/FL simulation entrypoint와 thin wrapper | `scripts/experiments/*`, `scripts/prototypes/*` |
| Apps | family extension UI shell과 future 제품 UI shell | `apps/family_extension/*` |
| Tests | package unit, cross-boundary integration, architecture guard | `shared/tests`, `agent/tests`, `main_server/tests`, `tests/*` |

## 3. 활성 레일

### 3.1 Local Inference Rail

```text
Raw Event
-> Preprocess / Translation
-> Embedding
-> Global Classifier
-> PersonalizationState
-> Time-Series Accumulator / Persistence
-> DecisionPolicy
-> AssessmentResult
```

주요 코드:

| 책임 | 파일 |
|---|---|
| API 수집 | `agent/src/api/ingest.py` |
| pipeline 조합 | `agent/src/services/inference/pipeline_service.py` |
| prototype scoring core | `methods/prototype/scoring/*` |
| scoring backend adapter | `agent/src/services/inference/scoring_backends/*` |
| final decision | `agent/src/services/inference/decision_service.py` |
| wellbeing projection | `agent/src/services/wellbeing/*` |

### 3.2 Child Support Rail

```text
Child Message
-> Agent-local Conversation Store
-> LocalContextProvider
-> ConversationState / SafetyIntent
-> SafetyPolicy / Scope Redirect
-> ResponsePolicy Plan / Required Moves
-> Local Guarded Reply or Local LLM Provider Execution
-> Plan Validation / Fallback
-> Child UI Response
```

주요 코드:

| 책임 | 파일 |
|---|---|
| API route | `agent/src/api/child_support.py` |
| service 조합 | `agent/src/services/wellbeing/child_support_service.py` |
| local conversation store | `agent/src/infrastructure/repositories/child_support_repository.py` |
| local context provider | `agent/src/services/wellbeing/child_support_context_provider.py` |
| conversation state extractor | `agent/src/services/wellbeing/child_support_conversation_state.py` |
| agent-local safety intent | `agent/src/services/wellbeing/child_support_safety_intent.py` |
| safety/scope policy | `agent/src/services/wellbeing/child_support_safety_policy.py` |
| response plan/validation policy | `agent/src/services/wellbeing/child_support_response_policy.py` |
| local LLM adapter | `agent/src/services/wellbeing/child_support_llm_provider.py` |
| UI panel | `apps/family_extension/src/components/ChildSupportCoachPanel.tsx` |

중요:

- child-support raw message와 query context는 agent-local boundary에 남긴다.
- 같은 `conversation_id`에서는 agent-local conversation store의 최근 메시지를 읽어
  폭력 사건 후속 대화를 감정 정리나 친구 대응 계획으로 이어간다.
- 타인 위해 intent 직후의 일반 힘듦 표현은 일반 check-in으로 리셋하지 않고,
  감정 수용과 위해 행동 경계를 함께 담은 de-escalation 응답으로 이어간다.
- safety routing은 shared contract의 화면 노출용 `safety_level`과 agent 내부용
  typed `SafetyIntent`를 분리해서, 타인 위해 의도 같은 새 케이스를 UI 계약 변경
  없이 확장할 수 있게 한다.
- 기본값은 deterministic `local_guarded`이고, Ollama를 켠 경우에도 agent가
  `ResponsePolicy` plan과 required move를 먼저 정한다.
- LLM은 plan의 required move를 순서대로 수행하는 실행 adapter이며, 응답이 필수
  의미를 잃거나 종료/회피 문구로 흐르면 plan validation에서 버리고 guarded
  fallback을 쓴다.
- main_server는 child-support 원문을 읽지 않고 FL aggregation 경계만 소유한다.

### 3.3 Query Adaptation Rail

```text
Reddit Labeled Data
-> Fixed Embedding
-> Classifier Seed
-> Local Deployment
-> Query Buffer
-> Threshold / Policy Selection
-> Accepted Query-derived Rows
-> Continue LoRA + Classifier Adaptation
-> Central or Federated Evaluation
```

주요 코드:

| 책임 | 파일 |
|---|---|
| LoRA supervised entrypoint | `scripts/experiments/central_ssl_control/train_lora_supervised_classifier.py` |
| LoRA SSL entrypoint | `scripts/experiments/central_ssl_control/train_lora_ssl_classifier.py` |
| 중앙/FL 공통 LoRA SSL harness | `scripts/experiments/query_lora_ssl/*` |
| trainer core | `methods/adaptation/query_classifier_adaptation/*`, `methods/ssl/*`, `methods/adaptation/*` |
| evaluation metric core | `methods/evaluation/*` |
| query buffer repository | `agent/src/infrastructure/repositories/query_buffer_repository.py` |
| query buffer selection | `agent/src/services/training/selection/query_buffer_selection_service.py` |

주의:

- 이 레일의 중앙 SSL 비교는 pooled/offline control table이다.
- prototype 기반 pseudo-label/SSL도 SSL 비교군 중 하나다.
- `FedMatch`, `FedLGMatch`, `(FL)^2`처럼 non-IID client 제약이 핵심인 방법은
  FL runtime rail에서 메인 논문 비교로 다룬다.

### 3.4 FL Runtime Rail

```text
Raw Event / Local Signal
-> Local Training
-> SharedClassifierUpdate or SharedAdapterUpdate
-> Central Aggregation
-> New ModelManifest / PrototypePack pair
```

논문 비교 관점에서는 이 레일이 `FL SSL under non-IID`의 메인 비교 위치다.

주요 코드:

| 책임 | 파일 |
|---|---|
| round lifecycle | `main_server/src/services/federation/rounds/round_lifecycle_service.py` |
| round manager | `main_server/src/services/federation/rounds/round_manager_service.py` |
| diagonal-scale local update core | `methods/adaptation/diagonal_scale/*` |
| shared adapter scoring core | `methods/adaptation/<family>/scoring.py` |
| shared adapter privacy guard core | `methods/adaptation/privacy_guards/*` |
| prototype training input core | `methods/prototype/training_inputs/*` |
| FL shard policy core | `methods/federated/shard_policy/*` |
| aggregation backend adapter | `main_server/src/services/federation/rounds/aggregation/*` |
| FedAvg generic core | `methods/federated/aggregation/fedavg/*` |
| adapter-family FedAvg core/materialization | `methods/adaptation/<family>/aggregation/fedavg.py`, 필요 시 `server_preflight.py` |
| FL SSL method descriptor/recipe metadata/policy | `methods/federated_ssl/*` |
| method-only aggregation variant | `methods/federated_ssl/<method>/aggregation.py` |
| FL simulation runtime adapter | `scripts/experiments/fl_ssl/federated_simulation/adapters/method_runtime.py` |
| FL report/evaluation payload | `methods/evaluation/*`, `scripts/experiments/fl_ssl/federated_simulation/io/*` |
| adapter family wiring | `main_server/src/services/federation/rounds/families/registry.py`, `families/models.py` |
| agent round client/runtime | `agent/src/services/federation/rounds/*` |
| agent current-task application flow | `agent/src/services/training/execution/agent_training_task_runner_service.py` |
| server-owned prototype rebuild input | `main_server/src/infrastructure/repositories/prototype_rebuild_input_repository.py` |
| prototype scoring core | `methods/prototype/scoring/*` |
| prototype evidence core | `methods/prototype/evidence/*` |

## 4. 코드 계층과 소유권

| 경로 | 소유 책임 | 금지 사항 |
|---|---|---|
| `shared/` | 공통 contract, domain entity, canonical payload 해석 규칙 | 실험 편의 로직을 공통 계층으로 승격하지 않는다 |
| `methods/` | 교체 가능한 SSL, adaptation, prototype, FL aggregation 계산 core와 method-local recipe metadata/policy | FastAPI, repository, Hydra entrypoint, runtime state를 소유하지 않는다 |
| `conf/` | Hydra 실행 조합과 파라미터 | Python 구현, 복잡한 계산 로직, runtime state를 소유하지 않는다 |
| `agent/` | local inference, local training, private/local state, server participation | method identity/local objective와 서버 round orchestration/aggregation policy를 소유하지 않는다 |
| `main_server/` | round lifecycle, aggregation, publication | method-specific server policy, raw text, 개인 threshold, 개인 해석 상태를 소유하지 않는다 |
| `scripts/` | Hydra 기반 실험 entrypoint, sweep, report, compatibility wrapper | 운영 후보 알고리즘 코어를 먼저 만들고 나중에 복사하지 않는다 |
| `apps/` | UI shell, wellbeing output consumer | 계약 의미, 전략 이름, 실행 기본값을 재정의하지 않는다 |
| `tests/` | cross-boundary integration/e2e, architecture 검증 | package 내부 단위 테스트를 불필요하게 루트로 올리지 않는다 |

## 5. 상태와 산출물 위치

| 위치 | 의미 | Git 기준 |
|---|---|---|
| `data/datasets/` | 새 dataset별 raw/mapped/split/query_ssl/view artifact | ignore |
| `data/artifacts/` | 새 model/prototype/adapter artifact | ignore |
| `data/cache/` | 새 model/translation/query cache | ignore |
| `data/processed/` | legacy dataset/model artifact | ignore |
| `runs/` | 실험 1회 실행 결과와 report. 신규 FL SSL은 `runs/fl_ssl/...` 계층을 쓴다 | ignore |
| `agent/state/` | 로컬 agent runtime state | ignore |
| `main_server/state/` | server runtime state와 publication artifact | ignore |
| `hf_cache/` | legacy Hugging Face/model cache | ignore |
| `apps/*/dist`, `apps/*/node_modules` | frontend build/dependency output | ignore |

## 6. 현재 운영 상태

현재 저장소는 Python package와 family extension Vite app 중심으로 구성되어 있다.

- Python dependency source of truth: `pyproject.toml`, `uv.lock`
- Python API app: `agent.src.api.main:app`, `main_server.src.api.main:app`
- Frontend apps: `apps/family_extension`
- 현재 repo에는 Docker Compose나 `infra/` manifest가 없다.

로컬 실행 절차는 `docs/operations/local-runbook.md`를 기준으로 본다.

## 7. 관련 문서

| 문서 | 역할 |
|---|---|
| `docs/execution_index.md` | 짧은 진입점과 문서 지도 |
| `docs/project_execution_plan.md` | 활성 연구/시스템 계획과 현재 Phase |
| `docs/architecture/method-owned-runtime-refactor-plan.md` | method-owned core와 runtime adapter 경계 리팩터링 계획 |
| `shared/src/contracts/README.md` | contract 파일 가까운 payload 해석 |
| `docs/api/api-surface.md` | 현재 FastAPI endpoint 표면 |
| `docs/operations/local-runbook.md` | 로컬 실행, GPU preflight, smoke 절차 |
| `docs/quality/test-strategy.md` | 테스트 층과 보호 범위 |
| `docs/governance/document-governance.md` | 문서 class와 갱신 규칙 |
