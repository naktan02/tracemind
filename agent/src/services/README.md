# Agent Services

이 디렉터리는 agent가 소유하는 로컬 runtime 서비스 모음이다.

핵심 원칙:

- `shared` 계약을 읽어 local inference와 local training을 실행한다.
- 서버와 통신하는 orchestration은 `federation/`에 두고,
  실제 scoring/training adapter는 `inference/`, `training_runtime/`에 둔다.
- asset cache/sync와 language helper는 각각 `assets/`, `language/`로 분리한다.
- `services/*/__init__.py`는 기본적으로 marker만 두고 direct-file import를 우선한다.
- newcomer는 `__init__.py`보다 아래 읽기 순서로 시작하는 편이 빠르다.

## 하위 패키지 역할

- `inference/`
  - 로컬 추론 rail
  - scorer backend별 분석 계산, 의사결정, 시계열 누적 담당
- `training_runtime/`
  - 로컬 학습 runtime rail
  - current TrainingTask 실행, Query SSL/FSSL local objective adapter,
    captured text generated view source projection 담당
- `federation/`
  - agent와 서버 사이 round orchestration rail
  - current round fetch/upload 담당
- `assets/shared_adapters/`
  - 서버 current shared adapter state 동기화/로컬 runtime helper
- `assets/adapters/`
  - global shared state와 future agent-private local state의 조합 경계
- `language/`
  - preprocess, translation, backtranslation helper
- `wellbeing/`
  - 가족용 확장 프로그램이 읽는 전체 상태/추이/잠금용 local output service

## 먼저 읽을 파일

현재 v1에서 권장하는 읽기 관점:

- scorer backend는 같은 analysis event로 투영한다.
- `local interpretation`이 final decision owner다.
- shared adapter와 scorer asset은 비교/확장 경로로 유지한다.

### 1. 로컬 추론 흐름을 보고 싶을 때

1. `inference/pipeline_service.py`
2. `inference/scoring_service.py`
3. `inference/decision_service.py`
4. `inference/time_series_service.py`
5. `language/translation_service.py`

### 2. 로컬 학습 흐름을 보고 싶을 때

1. `training_runtime/README.md`
2. `training_runtime/current_task/agent_training_task_runner_service.py`
3. `training_runtime/current_task/query_ssl_training_task_service.py`
4. `training_runtime/query_ssl_peft/local_training_service.py`
5. `training_runtime/training_sources/captured_text_source.py`

### 3. agent가 서버 round에 참여하는 흐름을 보고 싶을 때

1. `federation/rounds/round_client.py`
2. `training_runtime/current_task/agent_training_task_runner_service.py`
3. `training_runtime/current_task/query_ssl_training_task_service.py`
4. `assets/shared_adapters/sync_service.py`
5. `assets/adapters/composition_service.py`

### 4. 가족용 확장 출력 surface를 보고 싶을 때

1. `wellbeing/family_access_service.py`
2. `wellbeing/summary_service.py`
3. `wellbeing/timeseries_service.py`
4. `wellbeing/projection_service.py`
5. `wellbeing/auth_service.py`
6. `wellbeing/child_support_service.py`
7. `wellbeing/child_support_context_provider.py`
8. `wellbeing/child_support_safety_policy.py`
9. `wellbeing/child_support_llm_provider.py`
10. `agent/src/api/family_access.py`
11. `agent/src/api/child_support.py`
12. `agent/src/api/wellbeing.py`

## 파일 역할 빠른 맵

- `inference/scoring_backends/`
  - scorer backend registry와 agent runtime adapter 구현
  - backend 구현 옆 catalog entry와 decorator 등록을 둔다
- `language/backtranslation_service.py`
  - 운영 translation 코어와 같은 층에서 재사용하는 backtranslation service
  - strict USB NLP input용 `aug_0`, `aug_1` strong candidate 생성에 재사용한다
- `ingest/captured_text_view_provider_factory.py`
  - captured text weak/strong view generation provider를 agent env에서 조립한다
  - 기본은 identity fallback이며, NLLB provider를 켜면 모델은 실제 view generation
    실행 시 lazy-load된다
  - captured text DB 상태는 raw event row에 섞지 않고 view generation job,
    generated view, analysis job 상태 테이블로 분리한다
  - debug job은 generated weak text를 inference pipeline에 넣어 analysis event까지
    저장한다. captured text 학습 입력은 generated weak/strong view source에서
    시작한다
- `training_runtime/training_sources/captured_text_source.py`
  - agent-local `CapturedTextGeneratedViewRecord`를 training backend가 읽는
    Query SSL unlabeled row로 정규화한다
  - captured text는 raw string이나 임의 JSON으로 학습에 직접 들어가지 않고,
    captured event -> generated view -> source row 단계를 거친다
- `infrastructure/repositories/training_usage_ledger_repository.py`
  - generated view나 analysis event가 어떤 round/task/update의 학습 입력으로
    사용됐는지 source id와 recorded_at 기준으로 기록한다
- `training_runtime/query_ssl_peft/local_training_service.py`
  - Query SSL raw-row local training을 agent-local artifact 저장과 submission envelope에 연결
- `training_runtime/current_task/agent_training_task_runner_service.py`
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
  `training_runtime/training_sources/`
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
