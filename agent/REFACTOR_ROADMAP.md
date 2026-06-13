# Agent Refactor Roadmap

이 문서는 `agent/` 폴더 구조 리팩터링의 단계별 기준선이다. 현재 source of truth를
대체하지 않고, 각 phase를 시작하기 전 scope와 검증 gate를 맞추기 위한 작업 문서로
사용한다.

## 목표

최종 방향은 agent를 feature-oriented runtime으로 정리하는 것이다. 단, 단순 파일 이동이
목표가 아니다. 각 Module은 작은 Interface 뒤에 충분한 Implementation을 숨겨야 하며,
caller가 알아야 할 사실을 줄이는 쪽으로 이동한다.

핵심 목표:

- `api/main.py`는 FastAPI shell과 router 등록에 집중한다.
- agent runtime 조립은 `runtime/` Module이 소유한다.
- local inference, local update, captured text lifecycle, wellbeing output이 각자의
  읽기 시작점을 가진다.
- `contracts/`는 agent-local payload 계약의 source of truth로 유지한다.
- `methods/`가 소유해야 할 알고리즘 의미를 agent로 끌어오지 않는다.
- raw text와 개인 해석 상태는 agent-local boundary에 남긴다.

## 최종 목표 구조

```text
agent/src/
  api/
    main.py
    routers/
    dependencies/

  runtime/
    composition.py
    state.py
    env.py

  contracts/

  features/
    captured_text/
    wellbeing/
    inference/
    training_runtime/
    federation/
    shared_assets/
    language/
    typing_segments/

  infrastructure/
    persistence/
    repositories/
    model_adapters/
    transport/
```

`features/` 도입은 최종 목표다. 초반 phase에서는 기존 `services/`와
`infrastructure/` 안에서 의미를 먼저 정리하고, import churn과 compatibility 부담이
낮아진 뒤 feature-oriented layout으로 이동한다.

## Migration 원칙

- 한 phase는 한 concern만 다룬다.
- 한 phase가 끝날 때마다 테스트와 문서 gate를 통과시킨다.
- 같은 의미의 로직을 새 위치와 옛 위치에 동시에 복사하지 않는다.
- 임시 compatibility Module을 만들면 제거 조건을 같은 phase에 기록한다.
- 단일 caller만 쓰는 얕은 helper나 pass-through Module은 만들지 않는다.
- 큰 파일을 쪼갤 때는 table, lifecycle, policy, adapter처럼 독립 변경 이유가 있는
  조각만 분리한다.
- `__init__.py` re-export로 새 구조를 감추지 않는다. caller는 direct-file import를 쓴다.
- 이동 중에도 payload field 의미는 계약 파일 가까이에 남긴다.

## Phase Plan

### Phase 0. 기준선 고정

목표:

- 현재 구조와 목표 구조의 차이를 명시한다.
- phase별 검증 gate를 정한다.
- 코드 이동 전 baseline test set을 확인한다.

Gate:

- `git status -sb`로 작업 범위를 확인한다.
- 관련 README와 roadmap이 현재 phase를 설명한다.
- 아래 smoke set이 통과해야 다음 phase로 넘어간다.

```bash
uv run ruff check agent/src agent/tests/unit
uv run pytest agent/tests/unit/test_wellbeing_api.py agent/tests/unit/test_captured_text_repository.py agent/tests/unit/test_query_ssl_training_task_service.py agent/tests/unit/test_inference_pipeline.py
```

### Phase 1. runtime composition 분리

목표:

- `api/main.py`에서 repository/service/provider 조립을 분리한다.
- FastAPI shell과 agent runtime 조립의 Interface를 나눈다.

목표 구조:

```text
agent/src/runtime/
  composition.py
  state.py
  env.py
```

Gate:

- `create_app()` caller가 기존처럼 동작한다.
- app state 이름이 `runtime/state.py`에서 한 번에 확인된다.
- route smoke test가 통과한다.

### Phase 2. wellbeing 하위 Module 정리

목표:

- wellbeing signal, space-web, family access, child support를 서로 다른 Module로
  정리한다.

목표 구조:

```text
agent/src/services/wellbeing/
  signal/
  space_web/
  family_access/
  child_support/
```

Gate:

- family extension payload type generation이 통과한다.
- 부모 화면에 raw text/category detail이 새지 않는다.
- child support와 signal projection test가 통과한다.

### Phase 3. captured text lifecycle 정리

목표:

- `services/ingest/`를 captured text lifecycle 중심으로 정리한다.
- raw event, view generation, analysis job, training source 흐름을 한 읽기 경로로
  만든다.

목표 구조:

```text
agent/src/services/captured_text/
  ingest.py
  lifecycle.py
  debug_jobs.py
  view_generation/
  training_source/
```

Gate:

- captured text ingest/API/repository tests가 통과한다.
- training source caller가 raw text를 직접 읽지 않는다.

### Phase 4. captured text storage 분해

목표:

- `CapturedTextRepository`의 public Interface는 유지하고 내부 table store를 분리한다.

목표 구조:

```text
agent/src/infrastructure/repositories/captured_text/
  repository.py
  schema.py
  event_store.py
  view_job_store.py
  generated_view_store.py
  analysis_job_store.py
  retention.py
```

Gate:

- 기존 repository caller import가 모두 새 direct import로 전환된다.
- SQL/table 책임이 한 파일에 다시 누적되지 않는다.
- compatibility Module이 있으면 제거 조건이 기록된다.

### Phase 5. inference interpretation 정리

목표:

- scoring과 local interpretation을 이름과 위치로 분리한다.

목표 구조:

```text
agent/src/features/inference/
  pipeline.py
  scoring.py
  embedding.py
  interpretation/
    baseline.py
    time_series.py
    decision.py
    decision_policy.py
    state.py
    result.py
  scoring_backends/
```

Gate:

- pipeline test와 local interpretation tests가 통과한다.
- wellbeing projection은 interpretation Module을 통해 개인 기준 해석을 읽는다.

### Phase 6. training runtime 분해

목표:

- current task 실행, Query SSL source selection, method request build, upload, usage
  recording을 분리한다.

목표 구조:

```text
agent/src/features/training_runtime/
  current_task/
    runner.py
    dispatch.py
    result.py
  query_ssl/
    task_service.py
    source_selection.py
    method_request_builder.py
    usage_recording.py
    upload_flow.py
  local_updates/
```

Gate:

- `methods/`가 agent-local repository를 import하지 않는다.
- Query SSL task test와 training API test가 통과한다.
- usage ledger 변경이 method request build와 분리되어 있다.

### Phase 7. infrastructure layout 정리

목표:

- persistence, repository, model adapter, transport Adapter를 역할별로 정리한다.

목표 구조:

```text
agent/src/infrastructure/
  persistence/
  repositories/
  model_adapters/
  transport/
```

Gate:

- repository는 storage lifecycle만 소유한다.
- runtime policy가 infrastructure Adapter로 내려가지 않는다.

### Phase 8. architecture guard 고정

목표:

- 리팩터링 후 import drift를 테스트로 막는다.

Guard 후보:

- `api/`는 `methods/`를 직접 import하지 않는다.
- `apps/`는 agent service Implementation을 import하지 않는다.
- `shared/`는 agent-local contract를 import하지 않는다.
- `methods/`는 agent repository를 import하지 않는다.
- `training_runtime` 외 agent Module은 method-owned training core를 직접 import하지
  않는다.
- wellbeing UI payload는 `agent/src/contracts`가 소유한다.

Gate:

- guard test가 CI/local test에서 실행된다.
- phase 중 남긴 compatibility Module이 제거되었거나 제거 조건이 남아 있다.

## Phase별 공통 검증

각 phase 종료 시 최소 아래를 확인한다.

```bash
uv run ruff check agent/src agent/tests/unit
uv run pytest agent/tests/unit
```

family extension 계약이나 UI payload를 건드렸다면 추가로 실행한다.

```bash
./.venv/bin/python scripts/codegen/generate_family_extension_types.py
cd apps/family_extension && npm run build
```

전체 format check는 기존 미포맷 파일이 있는 동안 phase 변경 파일 범위로 먼저 닫고,
별도 cleanup phase에서 전체 format을 맞춘다.

## Feature Module Migration Plan

이 섹션은 위 Phase 0-8 cleanup 이후 `agent/src/features/`를 실제 운영 구조로 여는
후속 계획이다. 기존 cleanup phase는 `services/`와 `infrastructure/` 안의 의미를 먼저
정리했고, 이 계획은 그 결과를 일반 백엔드형 feature module 구조로 옮기는 단계다.

공통 gate:

- 한 phase는 하나의 feature 또는 하나의 wiring concern만 다룬다.
- 기존 위치와 새 위치에 같은 의미의 구현을 복사하지 않는다.
- route, service, repository를 한 번에 옮길 수 없으면 source of truth를 한 곳으로
  유지하고 남은 import만 다음 phase의 제거 대상으로 기록한다.
- `__init__.py` re-export facade를 만들지 않고 direct-file import를 유지한다.
- phase 종료 시 변경 범위 `ruff`, 관련 unit test, architecture guard를 실행한다.

### Feature Phase 1. 기준선과 목표 경계 고정

목표:

- `agent/src/features/`를 열고, feature module에 들어갈 것과 남길 것을 명시한다.
- 후속 2-8단계의 순서와 검증 gate를 code-adjacent 문서로 남긴다.
- 운영 import와 런타임 wiring은 아직 변경하지 않는다.

Gate:

- `agent/src/features/README.md`가 migration 원칙과 순서를 설명한다.
- 기존 runtime test가 그대로 통과한다.

### Feature Phase 2. captured_text pilot 이동

Status: 완료. 현재 source of truth는 `agent/src/features/captured_text/`이며, 전용
SQLite storage는 `agent/src/features/captured_text/storage/`가 소유한다.

목표:

- raw text lifecycle의 첫 pilot을 `features/captured_text/`로 이동한다.
- ingest, lifecycle, debug jobs, view generation, training source, captured text
  repository를 한 읽기 경로로 묶는다.
- raw text를 읽는 caller가 feature boundary 밖에서 storage 내부를 직접 보지 않게 한다.

Gate:

- captured text API, repository, training source tests가 통과한다.
- compatibility import가 남으면 제거 phase와 제거 조건을 같은 변경에 기록한다.

### Feature Phase 3. wellbeing 이동

Status: 완료. 현재 source of truth는 `agent/src/features/wellbeing/`이며, family access,
child-support, wellbeing settings/snapshot 전용 storage는
`agent/src/features/wellbeing/storage/`가 소유한다.

목표:

- wellbeing signal, space-web, family access, child support를
  `features/wellbeing/`으로 이동한다.
- 부모 UI payload와 child support 내부 상태가 같은 feature 안에 있더라도 계약과
  private state 경계를 유지한다.

Gate:

- family extension payload generation과 wellbeing API/unit tests가 통과한다.
- 부모 surface에 raw text나 category detail이 새지 않는다.

### Feature Phase 4. inference 이동

Status: 완료. 현재 source of truth는 `agent/src/features/inference/`이며, model
adapter mechanism은 `agent/src/infrastructure/model_adapters/`에 남긴다.

목표:

- `features/inference/`를 local inference pipeline feature로 연다.
- preprocess/translation/embedding/scoring/interpretation 흐름을 유지하되, model
  adapter mechanism은 `infrastructure/model_adapters/`에 남긴다.
- wellbeing은 inference storage 내부가 아니라 interpretation result boundary를 읽는다.

Gate:

- inference pipeline, scoring backend, interpretation tests가 통과한다.
- scorer backend 교체가 feature route나 wellbeing projection을 건드리지 않는다.

### Feature Phase 5. training_runtime 이동

Status: 완료. 현재 source of truth는 `agent/src/features/training_runtime/`이며,
training artifact와 usage ledger 저장소는
`agent/src/features/training_runtime/storage/`가 소유한다.

목표:

- `features/training_runtime/`로 current task runner, Query SSL task, method request
  build, upload, usage recording을 옮긴다.
- local update 실행은 agent-local capability로 남기고, objective/method 의미는
  `methods/`가 계속 소유한다.

Gate:

- training task/API tests가 통과한다.
- `methods/`가 agent repository를 import하지 않는 guard가 유지된다.

### Feature Phase 6. federation, assets, language, typing_segments 이동

Status: 완료. 현재 source of truth는 `agent/src/features/{assets,federation,language,typing_segments}/`다.

목표:

- 서버 라운드 참여, scorer asset sync/composition, language helper, typing segment ingest를
  각각 feature module로 옮긴다.
- 여러 feature가 공유하는 transport/model adapter mechanism은 `infrastructure/`에 둔다.

Gate:

- federation upload/round client tests와 asset/language 관련 tests가 통과한다.
- method-specific runtime 파일이 agent feature 아래에 새로 생기지 않는다.

### Feature Phase 7. FastAPI router와 dependency wiring 정리

Status: 완료. `agent/src/api/dependencies.py`가 FastAPI dependency glue와
`app.state` runtime lookup을 소유하고, route module은 endpoint payload 변환과 feature
service 호출 흐름만 남긴다.

목표:

- `api/routers/`와 `api/dependencies/`를 feature route registration 중심으로 정리한다.
- `dependencies.py` 계열은 FastAPI dependency glue와 runtime object lookup만 맡기고
  business rule을 소유하지 않는다.

Gate:

- `api/main.py`는 app shell과 router 등록 흐름만 보여준다.
- route smoke tests가 통과한다.

### Feature Phase 8. legacy 표면 제거와 guard 고정

목표:

- 남은 `services/` compatibility import, 비어 있는 repository placeholder, 중복 README를
  제거한다.
- feature module 의존 방향을 architecture guard로 고정한다.

Gate:

- `services/`가 source of truth로 남지 않는다.
- import guard가 feature-to-infrastructure, feature-to-feature, methods-to-agent
  의존 방향을 검증한다.
- 전체 agent unit test와 architecture guard가 통과한다.
