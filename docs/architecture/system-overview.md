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
- 공통 payload와 canonical 계산 규칙은 `shared`가 소유한다.
- `scripts`는 운영 후보 코어를 소유하지 않고, 실험 조합과 실행 표면만 소유한다.
- `apps`는 contract/API consumer이며 source of truth가 아니다.

## 2. 시스템 구성요소

| 구성요소 | 역할 | 주요 코드 |
|---|---|---|
| Shared contract/domain | agent, main_server, scripts가 함께 읽는 canonical payload와 domain entity | `shared/src/contracts/*`, `shared/src/domain/entities/*` |
| Agent API/runtime | 로컬 inference, query buffer, local training, wellbeing/family extension output | `agent/src/api/*`, `agent/src/services/*` |
| Main server API/runtime | FL round, aggregation, prototype publication, experiment workspace backend | `main_server/src/api/*`, `main_server/src/services/*` |
| Scripts | dataset/prototype/classifier/LoRA/FL simulation 실행 조합 | `scripts/experiments/*`, `scripts/prototypes/*`, `scripts/conf/*` |
| Apps | developer experiment UI와 family extension UI shell | `apps/experiment_web/*`, `apps/family_extension/*` |
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
| scoring backend/policy | `agent/src/services/inference/scoring_backends.py`, `agent/src/services/inference/scoring_policies.py` |
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
| LoRA supervised entrypoint | `scripts/experiments/train_lora_classifier.py` |
| pseudo-label bootstrap entrypoint | `scripts/experiments/train_lora_bootstrap_classifier_teacher.py` |
| pseudo-label self-training entrypoint | `scripts/experiments/train_lora_pseudo_label_classifier.py` |
| FixMatch entrypoint | `scripts/experiments/train_lora_fixmatch.py` |
| trainer core | `agent/src/services/training/query_adaptation/*` |
| query buffer repository | `agent/src/infrastructure/repositories/query_buffer_repository.py` |
| query buffer selection | `agent/src/services/training/selection/query_buffer_selection_service.py` |

주의:

- 이 레일의 중앙 SSL 비교는 pooled/offline control table이다.
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
| aggregation backend | `main_server/src/services/federation/rounds/aggregation/*` |
| adapter family wiring | `main_server/src/services/federation/rounds/families/*` |
| agent round client/runtime | `agent/src/services/federation/rounds/*` |

## 4. 코드 계층과 소유권

| 경로 | 소유 책임 | 금지 사항 |
|---|---|---|
| `shared/` | 공통 contract, domain entity, canonical 계산 규칙 | 실험 편의 로직을 공통 계층으로 승격하지 않는다 |
| `agent/` | local inference, local training, private/local state, server participation | 서버 round orchestration과 aggregation policy를 소유하지 않는다 |
| `main_server/` | round lifecycle, aggregation, publication, experiment workspace backend | raw text, 개인 threshold, 개인 해석 상태를 소유하지 않는다 |
| `scripts/` | Hydra 기반 실험 entrypoint, sweep, report, compatibility wrapper | 운영 후보 알고리즘 코어를 먼저 만들고 나중에 복사하지 않는다 |
| `apps/` | UI shell, compile/run consumer, wellbeing output consumer | 계약 의미, 전략 이름, 실행 기본값을 재정의하지 않는다 |
| `tests/` | cross-boundary integration/e2e, architecture 검증 | package 내부 단위 테스트를 불필요하게 루트로 올리지 않는다 |

## 5. 상태와 산출물 위치

| 위치 | 의미 | Git 기준 |
|---|---|---|
| `data/processed/` | 재사용 가능한 dataset/model artifact | ignore |
| `runs/` | 실험 1회 실행 결과와 report | ignore |
| `agent/state/` | 로컬 agent runtime state | ignore |
| `main_server/state/` | server runtime state와 publication artifact | ignore |
| `hf_cache/` | Hugging Face/model cache | ignore |
| `apps/*/dist`, `apps/*/node_modules` | frontend build/dependency output | ignore |

## 6. 현재 운영 상태

현재 저장소는 Python package와 Vite app 중심으로 구성되어 있다.

- Python dependency source of truth: `pyproject.toml`, `uv.lock`
- Python API app: `agent.src.api.main:app`, `main_server.src.api.main:app`
- Frontend apps: `apps/experiment_web`, `apps/family_extension`
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
