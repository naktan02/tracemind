# FL Runtime Implementation Checklist

## 목적

이 문서는 대화 기록이 아니라 실제 구현 작업표다.
목표는 아래 두 가지를 동시에 만족하는 것이다.

1. `agent`와 `main_server`가 실제 운영용 FL runtime 메커니즘을 소유한다.
2. `scripts`는 synthetic simulation, benchmark, evaluation harness로만 남긴다.

이 문서는 "다음에 무엇을 할지"만 적는 메모가 아니라,
"어떤 순서로, 왜, 어느 파일에, 어떤 완료 기준으로 구현할지"를 고정하는 체크리스트다.

중요:

- 이 체크리스트는 `시스템 FL 트랙`의 작업표다.
- 현재 전체 로드맵에서는 `central fixed+classifier seed`와 query 적응 설계를 먼저 닫고,
  그 다음에 이 문서 순서대로 FL translation을 진행한다.
- 즉 이 문서는 repo의 유일한 다음 단계 문서가 아니라, `논문 트랙 이후`에
  따라갈 시스템 구현 체크리스트다.

## 현재 상태 요약

현재 코드 기준으로 이미 있는 것:

- `agent/src/services/training/`
  - pseudo-label selection
  - local update 생성
  - training backend / privacy guard 교체 지점
- `agent/src/services/federation/training_example_service.py`
  - runtime training example preparation
  - example-generation backend 교체 지점
- `agent/src/services/inference/`
  - scorer backend / score policy 교체 지점
- `main_server/src/services/rounds/`
  - round lifecycle
  - update acceptance policy
  - adapter-family 기준 aggregation
  - server-owned config axis (`adapter_family`, `aggregation_backend`)
  - next manifest/state publication
- `scripts/experiments/federated_simulation/`
  - synthetic shard split
  - runtime core 조합 실험
  - evaluation / artifact dump
- `main_server/src/services/prototypes/`
  - runtime rebuild/publication
  - canonical rebuild input repository 연계

현재 비어 있거나 아직 약한 것:

- classifier-first baseline의 live agent 경로 확장
- 실제 두 번째 aggregation backend 구현
- learned scorer / scorer artifact lifecycle
- 논문 winner, 특히 future `lora` family를 시스템 FL로 옮기는 translation 설계
- secure aggregation runtime 자체
- 일부 integration test 인프라 안정화

## 최종 구조 원칙

### runtime이 소유할 것

- `main_server`
  - active round lifecycle
  - task publication
  - update ingestion
  - aggregation
  - manifest/state/prototype publication
- `agent`
  - active pair/task fetch
  - local event -> training example preparation
  - pseudo-label selection
  - local update generation
  - update upload

### scripts가 소유할 것

- synthetic shard split
- smoke / standard preset
- simulation harness
- evaluation / report dump
- exploratory benchmark
- runtime 코어를 감싸는 thin experiment wrapper

### 절대 섞지 않을 것

- training logic와 privacy enforcement
- aggregation logic와 HTTP lifecycle
- production runtime과 synthetic simulation helper
- server-owned canonical rebuild input과 agent private raw event
- 운영 후보 알고리즘 구현과 experiment-only 조합 로직

## 구현 순서

## Phase 0. 선행 결정 고정

이 phase에 들어오기 전에 `fixed embedding + classifier` seed baseline과 query 적응 단계의 `LoRA + classifier` 비교가 먼저 끝나야 한다.

### 왜 먼저 해야 하나

이 phase를 먼저 고정하지 않으면 이후 구현이 다 흔들린다.
특히 `PrototypePack`을 새 revision에서 어떻게 다시 만드는지가 정해지지 않으면
server runtime을 끝까지 닫을 수 없다.

### 결정해야 할 항목

- [x] prototype rebuild source를 확정한다.
  - 옵션 A: server-owned canonical bootstrap corpus
  - 옵션 B: server-owned canonical base-embedding cache
  - 권장: v1은 canonical bootstrap corpus부터 시작
- [x] v1 round participation 모델을 확정한다.
  - 현재 구현: polling 기반 agent self-check
  - `RoundClient.fetch_current_round()` / `fetch_current_task()`로 현재 active round를 조회
  - server push는 v1 범위 밖
- [x] update upload idempotency 정책을 확정한다.
  - 같은 `update_id` 재전송 허용 여부
  - 같은 `task_id`에 agent당 1회만 허용할지 여부
- [x] round close 조건을 확정한다.
  - 현재 구현: manual finalize 기반
  - `deadline_at`은 task 메타데이터로 전달되지만 서버가 자동 close를 트리거하지는 않음
  - deadline/min update 기반 자동 finalize는 future extension으로 둠

### 문서/코드 반영 위치

- [x] [docs/project_execution_plan.md](project_execution_plan.md)
- [x] [docs/contracts/training_task_v1.md](contracts/training_task_v1.md)
- [ ] 새 runtime service contract 문서가 필요하면 `docs/contracts/` 아래에 추가

### 완료 기준

- [x] 서버가 prototype rebuild에 사용할 canonical input이 무엇인지 한 문장으로 설명 가능하다.
- [x] round 하나가 어떤 조건에서 열리고 닫히는지 ambiguity가 없다.
- [x] 같은 update 재전송 시 서버 행동이 결정돼 있다.

## Phase 1. Main Server FL Runtime 닫기

### 목표

`main_server`가 "라운드 하나를 운영하는 중앙 coordinator"로 실제 동작하게 만든다.

### 새로 만들 것

- [x] `main_server/src/services/rounds/round_lifecycle_service.py`
  - 역할: round open / task publish / update accept / finalize / publish orchestration
- [x] `main_server/src/infrastructure/repositories/round_repository.py`
  - 역할: active round 상태 저장
- [x] `main_server/src/services/rounds/models.py` 또는 동등 경로
  - 역할: round status, participant summary, finalize input/output

### 기존 코드와 연결할 것

- [x] [round_manager_service.py](../main_server/src/services/rounds/round_manager_service.py)
  - domain primitive로 유지
  - lifecycle orchestration의 하위 구성 요소로 사용
- [x] [adapter_family_service.py](../main_server/src/services/rounds/adapter_family_service.py)
  - family-based aggregation 확장 지점으로 유지
- [x] [aggregation_service.py](../main_server/src/services/rounds/aggregation_service.py)
  - aggregation backend 교체 지점으로 유지

### API로 닫을 것

- [x] [main_server/src/api/fl_rounds.py](../main_server/src/api/fl_rounds.py)
  - `GET /api/v1/fl/rounds/current`
  - `POST /api/v1/fl/rounds`
  - `GET /api/v1/fl/rounds/{round_id}`
  - `POST /api/v1/fl/rounds/{round_id}/updates`
  - `POST /api/v1/fl/rounds/{round_id}/finalize`
- [x] [main_server/src/api/main.py](../main_server/src/api/main.py)
  - FL router include

### 검증할 것

- [x] active round가 없을 때의 응답
- [x] 이미 닫힌 round에 update 업로드 시 거부
- [x] base revision mismatch 거부
- [x] duplicate update 거부 또는 idempotent accept
- [x] finalize 전/후 state transition 확인

### 완료 기준

- [x] 서버만 띄워도 round 생성, 조회, finalize가 된다.
- [x] update ingestion이 파일 저장이 아니라 round lifecycle 문맥으로 연결된다.
- [x] aggregation 후 next manifest/state publication이 API 경로에서 일어난다.

## Phase 2. Agent Federated Runtime 닫기

### 목표

`agent`가 실제 FL participant로 동작하게 만든다.

### 새로 만들 것

- [x] `agent/src/services/federation/round_client.py`
  - 역할: current round/task fetch, update upload
- [x] `agent/src/services/federation/runtime_service.py`
  - 역할: active pair/task 기준 local training orchestration
- [x] `agent/src/services/federation/training_example_service.py`
  - 역할: local event/scored event를 `EmbeddedTrainingExample`으로 변환

### 기존 코드와 연결할 것

- [x] [local_training_service.py](../agent/src/services/training/local_training_service.py)
  - selection + update generation 코어로 유지
- [x] [pseudo_label_service.py](../agent/src/services/training/pseudo_label_service.py)
  - acceptance policy 교체 지점으로 유지
- [x] [training_backends/__init__.py](../agent/src/services/training/training_backends/__init__.py)
  - backend 교체 지점으로 유지

### API로 닫을 것

- [x] [agent/src/api/training.py](../agent/src/api/training.py)
  - `POST /api/v1/training/run-current-task`
  - `GET /api/v1/training/status`
  - 필요 시 `POST /api/v1/training/pull-task`
- [x] [agent/src/api/main.py](../agent/src/api/main.py)
  - training router include

### simulation에서 agent runtime으로 내릴 것

- [x] `scripts/experiments/federated_simulation/simulation.py`의 training example preparation helper
  - `agent/src/services/federation/training_example_service.py`로 이동 완료
- [ ] `scripts/experiments/federated_simulation/simulation.py`의 local runtime orchestration helper
  - runtime 코어 일부는 `FederationRuntimeService`로 이동했지만, simulation loop 자체는 아직 scripts에 남아 있음

### 검증할 것

- [x] active manifest/task가 없을 때 안전하게 종료
- [x] local accepted examples가 부족할 때 no-update 처리
- [x] upload 성공/실패 상태 분리
- [x] 같은 task에 대한 중복 실행 방지 정책
- [x] stored-event 경로가 지원하지 않는 runtime 조합은 `unsupported_runtime`으로 조기 거부

### 완료 기준

- [x] agent가 current task를 읽고 local update를 만들어 업로드할 수 있다.
- [x] local training example preparation이 더 이상 script 전용 helper가 아니다.
- [x] `TrainingTask`만 보면 local training 실행 가능 여부가 결정된다.
- [ ] classifier-first baseline이 live agent stored-event 경로까지 닫힌다.

## Phase 3. Prototype Rebuild Runtime 경로 닫기

### 목표

새 adapter revision이 발행될 때 `PrototypePack`도 운영 경로에서 같이 갱신되게 만든다.

### 왜 중요하나

지금 simulation은 bootstrap rows를 들고 있기 때문에 rebuild가 가능하다.
하지만 실제 production에서는 서버가 agent private raw event에 의존하면 안 된다.

### 새로 만들 것

- [x] `main_server/src/services/prototypes/prototype_rebuild_service.py`
  - 역할: canonical rebuild input으로 next prototype 생성
- [x] `main_server/src/infrastructure/repositories/prototype_rebuild_input_repository.py`
  - 역할: canonical bootstrap corpus 또는 base-embedding cache 관리

### 기존 코드와 연결할 것

- [ ] [prototype_build_state_service.py](../main_server/src/services/prototypes/prototype_build_state_service.py)
  - 운영 경로에서 build state를 유지할지 여부 결정
- [x] [scripts/prototypes/seeding.py](../scripts/prototypes/seeding.py)
  - production logic을 그대로 재사용할지, server service 전용 core로 분리할지 결정
- [x] [round_lifecycle_service.py](../main_server/src/services/rounds/round_lifecycle_service.py)
  - next pair publication 직후 rebuild 호출 흐름과 연결
- [x] `scripts/experiments/prototype_strategy`의 single/kmeans
  - shared canonical builder를 재사용

### 결정해야 할 세부사항

- [ ] `PrototypeBuildState`를 운영에서 계속 저장할지
- [ ] single builder only exact incremental을 v1 운영 경로에 포함할지
- [ ] multi-prototype runtime을 rebuild에 같이 열지, 나중에 열지
- [ ] prototype rebuild를 classifier-first baseline의 bootstrap/comparison artifact로 계속 유지할지

### 완료 기준

- [x] simulation helper 없이도 server runtime이 next prototype을 발행할 수 있다.
- [ ] rebuild 입력이 private raw event에 의존하지 않는다.
- [x] next manifest와 next prototype version이 항상 일관되게 맞물린다.

## Phase 4. End-to-End HTTP Federation Integration Test 추가

### 목표

unit test와 script simulation이 아니라 실제 HTTP runtime으로 federation loop를 닫아 본다.

### 만들 것

- [x] `tests/integration/test_fl_round_e2e.py`
  - server HTTP round lifecycle 기본 완주 시나리오
- [ ] server 1개 + agent 2개 이상 시나리오 확장
- [ ] fixture
  - bootstrap canonical input
  - active manifest/state
  - local sample events

### 검증 시나리오

- [x] current round fetch
- [x] current task fetch
- [ ] local update generation
- [x] update upload
- [x] finalize
- [x] next manifest/prototype publication
- [ ] integration test infra를 최신 httpx/transport 방식에 맞게 안정화

### 실패 시나리오

- [ ] deadline 지난 round에 upload
- [ ] stale base revision upload
- [ ] duplicate update
- [ ] accepted examples 0건
- [ ] finalize without enough updates

### 완료 기준

- [x] `scripts` simulation 없이 runtime API만으로 1 round가 돈다. (server 중심 기본 경로)
- [ ] 최소 2-agent integration test가 안정적으로 통과한다.

## Phase 5. Policy / Backend 교체성 강화

### 목표

지금 있는 "교체 가능해 보이는 구조"를 실제로 확장 가능한 구조로 만든다.

### local 쪽

- [ ] heuristic 외 `DiagonalScaleGradientTrainingBackend` 추가
- [x] backend 선택 규칙이 `TrainingTask.objective_config.training_backend_name`와 일관되게 매핑되도록 정리
- [x] acceptance policy registry를 명시적으로 정리
- [x] scorer backend를 독립 축으로 분리
- [x] example-generation backend를 독립 축으로 분리
- [x] local runtime compatibility validator 추가

### server 쪽

- [x] aggregation backend registry 또는 family 확장 규칙 정리
- [x] adapter family별 state/update/payload compatibility 검증 강화
- [x] server-owned config axis (`adapter_family`, `aggregation_backend`) 도입
- [ ] real second aggregation backend 추가

### config 쪽

- [x] FL runtime config에 한해 typed config 도입
- [ ] task payload와 runtime config의 drift 방지 테스트 추가
- [x] secure aggregation을 bool이 아니라 typed contract로 승격

### 완료 기준

- [x] acceptance policy, training backend, aggregation backend를 각각 독립적으로 교체 가능하다.
- [x] 새 backend 하나 추가 시 lifecycle/API 계층을 뜯지 않는다.
- [ ] 계약 자체가 달라지는 family까지 새 구현체 추가로 닫혔는지 확인한다.

## Phase 5-1. Classifier-First Baseline 정리

### 목표

`embedding -> global classifier -> local interpretation`을 v1의 기본 실험선으로
명확히 하고, shared adapter는 비교 실험 축으로 내린다.

### 해야 할 일

- [ ] classifier-head family의 live agent state fetch / manifest sync 경로 추가
- [ ] stored-event 경로에서 classifier-first baseline이 실제로 도는지 검증
- [ ] classifier-only baseline과 shared adapter baseline의 ablation 실험 추가
- [ ] single prototype baseline으로 충분한지 error analysis 수행

### 완료 기준

- [ ] classifier-first baseline이 simulation과 live agent 양쪽에서 같은 계약으로 돈다.
- [ ] shared adapter를 기본선으로 둘 근거가 있는지, 비교축으로 둘 근거가 있는지 문서화된다.

## Phase 6. Privacy / Robustness Hardening

### 목표

training과 aggregation이 먼저 닫힌 뒤, privacy와 robustness를 분리된 계층으로 추가한다.

### 추가할 것

- [x] secure aggregation 적용 지점 확정
- [ ] DP/noise mechanism 적용 지점 확정
- [ ] stale revision / malformed payload / oversized norm 방어
- [ ] robust aggregation 후보 실험
  - weighted mean 외 옵션 검토

### 주의사항

- [ ] privacy 코드를 training backend 안에 섞지 않는다.
- [ ] aggregation backend 안에 transport 검증을 섞지 않는다.

### 완료 기준

- [ ] insecure path와 hardened path가 분리돼 있다.
- [ ] privacy layer를 꺼도 training/aggregation 코어는 독립적으로 유지된다.

## 지금 당장 하지 않을 것

- [ ] full encoder FL
- [ ] private adapter/head를 shared runtime에 조기 통합
- [ ] `scripts` simulation을 통째로 production runtime으로 승격
- [ ] secure aggregation/DP를 lifecycle 완성 전에 먼저 구현

## 문서 정합성 체크

아래 문서가 서로 같은 이야기를 해야 한다.

- [ ] [plan.md](../plan.md)
- [ ] [docs/project_execution_plan.md](project_execution_plan.md)
- [ ] [docs/staged_execution_roadmap.md](staged_execution_roadmap.md)
- [ ] [docs/contracts/training_task_v1.md](contracts/training_task_v1.md)
- [ ] [shared/src/contracts/README.md](../shared/src/contracts/README.md)

## 가장 먼저 시작할 실제 작업 3개

- [x] 두 번째 real family(`classifier_head`) 추가
- [ ] 두 번째 real aggregation backend 추가
- [ ] classifier-first baseline live agent 확장
