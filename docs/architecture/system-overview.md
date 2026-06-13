# TraceMind System Overview

이 문서는 TraceMind의 현재 런타임, 코드 경계, 활성 연구/시스템 레일을 설명하는 canonical 개요다.

세부 payload 필드의 source of truth는 문서가 아니라 `shared/src/contracts/*.py`,
`shared/src/domain/entities/*`, agent-local API/UI payload의 경우
`agent/src/contracts/*.py`, agent-local inference state/result의 경우
`agent/src/features/inference/interpretation/*`다.

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
- 최종 method/runtime 구조 판단은
  `docs/architecture/target-method-runtime-structure.md`를 우선한다. 이 문서는 현재
  런타임 개요이고, target 문서는 future-facing 용어와 migration plan을 소유한다.
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
| Methods | SSL, adaptation, evaluation, FL aggregation의 교체 가능한 계산 core | `methods/*` |
| Hydra config | 실행 조합, strategy axis, track preset | `conf/*` |
| Agent API/runtime | 로컬 inference, captured text, local training, wellbeing/family extension output | `agent/src/api/*`, `agent/src/features/*`, `agent/src/runtime/*` |
| Main server API/runtime | FL round, aggregation, publication | `main_server/src/api/*`, `main_server/src/services/*` |
| Scripts | dataset/PEFT/FL simulation entrypoint와 thin wrapper | `scripts/experiments/*`, `scripts/workflows/*` |
| Apps | family extension UI, experiment dashboard, future 제품 UI shell | `apps/family_extension/*`, `apps/experiment_dashboard/*` |
| Tests | package unit, cross-boundary integration, architecture guard | `shared/tests`, `agent/tests`, `main_server/tests`, `tests/*` |

## 3. 활성 레일

### 3.1 Local Inference Rail

```text
Raw Event
-> Preprocess / Translation
-> Embedding
-> Global Classifier
-> Time-Series Accumulator / Persistence
-> Agent-local Rule Decision
-> AssessmentResult
```

주요 코드:

| 책임 | 파일 |
|---|---|
| API 수집 | `agent/src/api/ingest.py` |
| pipeline 조합 | `agent/src/features/inference/pipeline_service.py` |
| scoring backend adapter | `agent/src/features/inference/scoring_backends/*` |
| local baseline/time-series state | `agent/src/features/inference/interpretation/state.py`, `agent/src/features/inference/interpretation/time_series.py` |
| final decision | `agent/src/features/inference/interpretation/decision.py` |
| wellbeing projection | `agent/src/features/wellbeing/signal/*`, `agent/src/features/wellbeing/space_web/*` |

### 3.2 Child Support Rail

```text
Child Message
-> Agent-local Conversation Store
-> LocalContextProvider
-> SafetyPolicy / Scope Redirect
-> Local Context Prompt
-> Local LLM Provider Execution
-> LLM Reply Normalization
-> Child UI Response
```

주요 코드:

| 책임 | 파일 |
|---|---|
| API route | `agent/src/api/child_support.py` |
| service 조합 | `agent/src/features/wellbeing/child_support/service.py` |
| local conversation store | `agent/src/features/wellbeing/storage/child_support_repository.py` |
| local context provider | `agent/src/features/wellbeing/child_support/context_provider.py` |
| safety/scope policy | `agent/src/features/wellbeing/child_support/safety_policy.py` |
| local LLM prompt builder | `agent/src/features/wellbeing/child_support/llm_prompt.py` |
| local evidence summary | `agent/src/features/wellbeing/child_support/evidence_summary.py` |
| local LLM adapter | `agent/src/features/wellbeing/child_support/llm_provider.py` |
| UI panel | `apps/family_extension/src/components/ChildSupportCoachPanel.tsx` |

중요:

- child-support raw message와 query context는 agent-local boundary에 남긴다.
- 같은 `conversation_id`에서는 agent-local conversation store의 최근 메시지를 prompt
  history로 전달해 LLM이 대화 흐름을 이어가게 한다.
- safety routing은 shared contract의 화면 노출용 `safety_level`과 agent-local
  `reason` 문자열을 함께 저장한다. 세부 reason은 UI 계약을 늘리지 않고 대화
  history와 LLM prompt hint에만 쓴다.
- child-support 답변 기본값은 LLM-first다. LLM provider가 없거나 LLM 응답을 만들지
  못하면 agent API는 정적 상담 답변을 대신 만들지 않고 실패를 반환한다.
- agent service는 wellbeing summary, recent conversation, evidence summary와
  최근 원문 일부를 local context prompt로 묶어 LLM에 전달한다. safety policy는
  routing hint에 쓰고, 응답 문장 구조를 required move로 강제하지 않는다.
- main_server는 child-support 원문을 읽지 않고 FL aggregation 경계만 소유한다.

### 3.3 Query Adaptation Rail

```text
Reddit Labeled Data
-> Fixed Embedding
-> Classifier Seed
-> Local Deployment
-> Generated View Source or legacy Query Buffer
-> Threshold / Policy Selection
-> Accepted Query-derived Rows
-> Continue PEFT text encoder + linear head adaptation
-> Central or Federated Evaluation
```

주요 코드:

| 책임 | 파일 |
|---|---|
| PEFT supervised entrypoint | `scripts/experiments/central/ssl_control/run_peft_supervised_control.py` |
| Full text encoder supervised entrypoint | `scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py` |
| PEFT SSL entrypoint | `scripts/experiments/central/ssl_control/run_peft_ssl_control.py` |
| 중앙/FL 공통 text encoder SSL runtime support | `scripts/support/query_ssl_text_encoder/*` |
| trainer core | `methods/adaptation/query_text_views/*`, `methods/ssl/*`, `methods/adaptation/*` |
| evaluation metric core | `methods/evaluation/*` |
| captured generated view source | `agent/src/features/captured_text/training_source/service.py` |
| training source usage ledger | `agent/src/features/training_runtime/storage/training_usage_ledger_repository.py` |

주의:

- 이 레일의 중앙 SSL 비교는 pooled/offline control table이다.
- FedMatch처럼 non-IID client 제약이 핵심인 방법은 FL runtime rail에서 다룬다.

### 3.4 FL Runtime Rail

```text
Raw Event / Local Signal
-> Local Training
-> SharedClassifierUpdate or SharedAdapterUpdate
-> Central Aggregation
-> New ModelManifest / optional auxiliary artifacts
```

논문 비교 관점에서는 이 레일이 `FL SSL under non-IID`의 메인 비교 위치다.

주요 코드:

| 책임 | 파일 |
|---|---|
| round lifecycle | `main_server/src/services/federation/rounds/round_lifecycle_service.py` |
| round manager | `main_server/src/services/federation/rounds/round_manager_service.py` |
| shared adapter scoring core | `methods/classification/linear_head/scoring.py` |
| shared adapter privacy guard core | `methods/adaptation/privacy_guards/*` |
| FL shard policy core | `methods/federated/shard_policy/*` |
| aggregation backend adapter | `main_server/src/services/federation/rounds/aggregation/*` |
| FedAvg generic core | `methods/federated/aggregation/fedavg/*` |
| update-family FedAvg projection/materialization | `methods/adaptation/<family>/aggregation/*` 또는 `methods/classification/<family>/aggregation/*`, 필요 시 `server_preflight.py` |
| FL SSL method descriptor/recipe metadata/policy | `methods/federated_ssl/*` |
| method-only aggregation variant | `methods/federated_ssl/<method>/aggregation.py` |
| FL simulation runtime adapter | `scripts/experiments/fl_ssl/federated_simulation/adapters/method_runtime.py` |
| FL report/evaluation payload | `methods/evaluation/*`, `scripts/experiments/fl_ssl/federated_simulation/io/*` |
| payload adapter wiring | `main_server/src/services/federation/rounds/payload_adapters/registry.py`, `payload_adapters/models.py` |
| agent round client/runtime | `agent/src/features/federation/rounds/*` |
| agent current-task application flow | `agent/src/features/training_runtime/current_task/runner.py` |

## 4. 코드 계층과 소유권

| 경로 | 소유 책임 | 금지 사항 |
|---|---|---|
| `shared/` | 공통 contract, domain entity, canonical payload 해석 규칙 | 실험 편의 로직을 공통 계층으로 승격하지 않는다 |
| `methods/` | 교체 가능한 SSL, adaptation, FL aggregation 계산 core와 method-local recipe metadata/policy | FastAPI, repository, Hydra entrypoint, runtime state를 소유하지 않는다 |
| `conf/` | Hydra 실행 조합과 파라미터 | Python 구현, 복잡한 계산 로직, runtime state를 소유하지 않는다 |
| `agent/` | local inference, local training, private/local state, server participation | method identity/local objective와 서버 round orchestration/aggregation policy를 소유하지 않는다 |
| `main_server/` | round lifecycle, aggregation, publication | method-specific server policy, raw text, 개인 threshold, 개인 해석 상태를 소유하지 않는다 |
| `scripts/` | Hydra 기반 실험 entrypoint, sweep, report/artifact orchestration, runtime bridge | 운영 후보 알고리즘 코어와 method policy를 소유하지 않는다 |
| `apps/` | UI shell, wellbeing output consumer | 계약 의미, 전략 이름, 실행 기본값을 재정의하지 않는다 |
| `tests/` | cross-boundary integration/e2e, architecture 검증 | package 내부 단위 테스트를 불필요하게 루트로 올리지 않는다 |

## 5. 상태와 산출물 위치

| 위치 | 의미 | Git 기준 |
|---|---|---|
| `data/datasets/` | 새 dataset별 raw/mapped/split/query_ssl/view artifact | ignore |
| `data/artifacts/` | 새 model/adapter artifact | ignore |
| `data/cache/` | 새 model/translation/query cache | ignore |
| `data/processed/` | legacy dataset/model artifact | ignore |
| `runs/` | 실험 1회 실행 결과와 report. 신규 FL SSL은 `runs/fl_ssl/...` 계층을 쓴다 | ignore |
| `agent/state/` | 로컬 agent runtime state | ignore |
| `main_server/state/` | server runtime state와 publication artifact | ignore |
| `hf_cache/` | legacy Hugging Face/model cache | ignore |
| `apps/*/dist`, `apps/*/node_modules` | frontend build/dependency output | ignore |

## 6. 현재 운영 상태

현재 저장소는 Python package, family extension Vite app, 정적 experiment dashboard로 구성되어 있다.

- Python dependency source of truth: `pyproject.toml`, `uv.lock`
- Python API app: `agent.src.api.main:app`, `main_server.src.api.main:app`
- Frontend apps: `apps/family_extension`, `apps/experiment_dashboard`
- 현재 repo에는 Docker Compose나 `infra/` manifest가 없다.

로컬 실행 절차는 `docs/operations/local-runbook.md`를 기준으로 본다.

## 7. 관련 문서

| 문서 | 역할 |
|---|---|
| `docs/execution_index.md` | 짧은 진입점과 문서 지도 |
| `docs/project_execution_plan.md` | 활성 연구/시스템 계획과 현재 Phase |
| `shared/src/contracts/README.md` | contract 파일 가까운 payload 해석 |
| `docs/api/api-surface.md` | 현재 FastAPI endpoint 표면 |
| `docs/operations/local-runbook.md` | 로컬 실행, GPU preflight, smoke 절차 |
| `docs/quality/test-strategy.md` | 테스트 층과 보호 범위 |
| `docs/governance/document-governance.md` | 문서 class와 갱신 규칙 |
