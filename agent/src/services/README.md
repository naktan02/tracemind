# Agent Services

이 디렉터리는 feature module 전환 중 남은 legacy 안내 표면이다. 현재 운영 기능의
source of truth는 `agent/src/features/`이며, 프로세스 시작 시 service graph를
조립하는 책임은 `agent/src/runtime/`이 소유한다. 남은 `runtime_registry.py` 계열은
Phase 8에서 feature/runtime 공통 primitive 위치로 정리한다.

핵심 원칙:

- `shared` 계약을 읽는 local inference/training/server participation 구현은
  `agent/src/features/` 아래 feature module이 소유한다.
- FastAPI shell과 service graph 조립은 `api/`와 `runtime/`에 둔다.
- `services/*/__init__.py`는 기본적으로 marker만 두고 direct-file import를 우선한다.
- newcomer는 `__init__.py`보다 아래 읽기 순서로 시작하는 편이 빠르다.
- 구조 리팩터링은 `agent/REFACTOR_ROADMAP.md`의 phase gate를 따른다.
- 이동 중에도 같은 의미의 로직을 옛 위치와 새 위치에 복사하지 않는다.
- 너무 작은 pass-through Module이나 단일 caller 전용 helper는 새로 만들지 않는다.

## 이동 완료 feature:

- `agent/src/features/assets/`
  - 서버 current shared adapter state 동기화, local runtime helper, composition boundary
- `agent/src/features/captured_text/`
  - raw text ingest, generated view, training source projection, 전용 storage
- `agent/src/features/federation/`
  - agent와 서버 사이 round orchestration rail, current round fetch/upload client
- `agent/src/features/inference/`
  - 로컬 추론 pipeline, scorer backend adapter, agent-local 개인 기준 해석
- `agent/src/features/language/`
  - preprocess, translation, backtranslation helper
- `agent/src/features/training_runtime/`
  - 로컬 학습 runtime rail, current TrainingTask 실행, Query SSL/FSSL adapter, 전용 storage
- `agent/src/features/typing_segments/`
  - typing segment ingest use case
- `agent/src/features/wellbeing/`
  - 가족용 확장 프로그램이 읽는 전체 상태/추이/잠금용 local output과 전용 storage

## 먼저 읽을 파일

현재 v1에서 권장하는 읽기 관점:

- scorer backend는 같은 analysis event로 투영한다.
- `local interpretation`이 final decision owner다.
- shared adapter와 scorer asset은 비교/확장 경로로 유지한다.

### 1. 로컬 추론 흐름을 보고 싶을 때

1. `agent/src/features/inference/pipeline_service.py`
2. `agent/src/features/inference/scoring_service.py`
3. `agent/src/features/inference/interpretation/decision.py`
4. `agent/src/features/inference/interpretation/time_series.py`
5. `agent/src/features/language/translation_service.py`

### 2. 로컬 학습 흐름을 보고 싶을 때

1. `agent/src/features/training_runtime/README.md`
2. `agent/src/features/training_runtime/current_task/runner.py`
3. `agent/src/features/training_runtime/query_ssl/task_service.py`
4. `agent/src/features/training_runtime/query_ssl_peft/local_training_service.py`
5. `agent/src/features/captured_text/training_source/service.py`

### 3. agent가 서버 round에 참여하는 흐름을 보고 싶을 때

1. `agent/src/features/federation/rounds/round_client.py`
2. `agent/src/features/training_runtime/current_task/runner.py`
3. `agent/src/features/training_runtime/query_ssl/task_service.py`
4. `agent/src/features/assets/shared_adapters/sync_service.py`
5. `agent/src/features/assets/adapters/composition_service.py`

### 4. 가족용 확장 출력 surface를 보고 싶을 때

1. `agent/src/features/wellbeing/README.md`
2. `agent/src/features/wellbeing/family_access/service.py`
3. `agent/src/features/wellbeing/signal/summary_service.py`
4. `agent/src/features/wellbeing/signal/timeseries_service.py`
5. `agent/src/features/wellbeing/signal/projection_service.py`
6. `agent/src/features/wellbeing/space_web/README.md`
7. `agent/src/features/wellbeing/family_access/parent_auth_adapter.py`
8. `agent/src/features/wellbeing/child_support/service.py`
9. `agent/src/features/wellbeing/child_support/context_provider.py`
10. `agent/src/features/wellbeing/child_support/safety_policy.py`
11. `agent/src/features/wellbeing/child_support/llm_provider.py`
12. `agent/src/api/family_access.py`
13. `agent/src/api/child_support.py`
14. `agent/src/api/wellbeing.py`

## 파일 역할 빠른 맵

- `agent/src/features/inference/scoring_backends/`
  - scorer backend registry와 agent runtime adapter 구현
  - backend 구현 옆 catalog entry와 decorator 등록을 둔다
- `agent/src/features/language/backtranslation_service.py`
  - 운영 translation 코어와 같은 층에서 재사용하는 backtranslation service
  - strict USB NLP input용 `aug_0`, `aug_1` strong candidate 생성에 재사용한다
- `agent/src/features/captured_text/view_generation/provider_factory.py`
  - captured text weak/strong view generation provider를 agent env에서 조립한다
  - 기본은 identity fallback이며, NLLB provider를 켜면 모델은 실제 view generation
    실행 시 lazy-load된다
  - captured text DB 상태는 raw event row에 섞지 않고 view generation job,
    generated view, analysis job 상태 테이블로 분리한다
  - debug job은 generated weak text를 inference pipeline에 넣어 analysis event까지
    저장한다. captured text 학습 입력은 generated weak/strong view source에서
    시작한다
- `agent/src/features/captured_text/training_source/service.py`
  - agent-local `CapturedTextGeneratedViewRecord`를 training backend가 읽는
    Query SSL unlabeled row로 정규화한다
  - captured text는 raw string이나 임의 JSON으로 학습에 직접 들어가지 않고,
    captured event -> generated view -> source row 단계를 거친다
- `agent/src/features/training_runtime/storage/training_usage_ledger_repository.py`
  - generated view나 analysis event가 어떤 round/task/update의 학습 입력으로
    사용됐는지 source id와 recorded_at 기준으로 기록한다
- `agent/src/features/training_runtime/query_ssl_peft/local_training_service.py`
  - Query SSL raw-row local training을 agent-local artifact 저장과 submission envelope에 연결
- `agent/src/features/training_runtime/current_task/runner.py`
  - active task 조회, shared adapter sync, Query SSL task 실행, update upload까지의
    agent application flow 소유
  - stored-event self-training rebuild는 지원하지 않는다
- local update backend registry는 `methods/adaptation/local_update_registry.py`가 소유한다
  - `training/` old path는 재도입하지 않는다
  - 새 local update backend는 `methods/adaptation/<family>/training_backend.py`에 둔다
- `methods/adaptation/peft_text_encoder/`
  - PEFT text encoder update family와 local update backend core
  - raw text를 agent-local 입력으로 요구하고 shared payload에는 artifact ref만 남긴다.

## 전략 추가 시 출발점

- local update/adaptation 계산 추가: `methods/adaptation/<family>/`
  - `agent`에는 raw text 접근, local artifact materialization, payload upload 같은
    runtime capability adapter만 둔다.
  - fixed embedding을 쓰는 family와 raw text/tokenized batch를 쓰는 family를 같은
    adapter 내부에서 섞지 않는다.
- Query SSL/FSSL 학습 입력 추가: 해당 method core와
  `agent/src/features/training_runtime/training_sources/`
- scorer backend 추가: method core를 먼저 추가하고,
  `inference/scoring_backends/`에는 agent runtime adapter만 둔다
- pseudo-label acceptance/selection 정책 추가: 해당 SSL/FSSL method package
- privacy guard 추가: `methods/adaptation/privacy_guards/`

FedMatch, FreeMatch 같은 method 이름을 가진 파일은 `agent`에 만들지
않는다. 해당 method의 local objective, hook, selection/threshold 의미는
`methods/`가 소유하고, `agent`는 선택된 method core를 local data와 contract
payload에 연결한다.

확장 전에 `shared/src/contracts/README.md`, `methods/README.md`, `conf/README.md`를
함께 읽는 것을 권장한다.
