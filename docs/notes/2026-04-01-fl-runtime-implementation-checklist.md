# FL Runtime Implementation Checklist

## 배경

현재 TraceMind의 FL simulation은 `scripts/experiments/federated_simulation/`에 있고,
실제 메커니즘의 핵심 일부는 이미 `agent`와 `main_server`에 있다.

확인된 현재 상태:

- `agent`에는 local pseudo-label selection과 update 생성 서비스가 있다.
- `main_server`에는 round task 생성과 aggregation/pair publication 서비스가 있다.
- 하지만 실제 HTTP runtime 경로와 round lifecycle, prototype rebuild 운영 경로는 아직 닫히지 않았다.

핵심 원칙은 아래로 고정한다.

- runtime mechanism은 `agent` / `main_server`가 소유한다.
- `scripts`는 synthetic dataset, smoke/benchmark, evaluation harness를 소유한다.
- policy, backend, privacy는 task/config 기반으로 바꿔낄 수 있어야 한다.

## 결정 내용

앞으로 구현 순서는 아래 순서로 고정한다.

### 0. 선행 결정 고정

- [ ] prototype rebuild source를 확정한다.
  - 후보 A: server-owned canonical bootstrap corpus
  - 후보 B: server-owned canonical base-embedding cache
- [ ] v1의 실제 round participation 모델을 고정한다.
  - 예: polling 기반 단순 participation부터 시작
- [ ] update upload idempotency 정책을 정한다.
  - 같은 `task_id`/`update_id` 재전송 시 처리 규칙
- [ ] round close 조건을 정한다.
  - deadline 기반 / 최소 update 수 기반 / manual finalize 기반

완료 기준:

- [ ] 위 4개가 문서와 코드에서 같은 의미로 보인다.
- [ ] 서버가 prototype을 어떤 canonical input으로 재생성하는지 애매함이 없다.

### 1. Main Server FL Runtime 닫기

- [ ] `main_server/src/services/rounds/` 위에 lifecycle orchestration 서비스를 추가한다.
  - 예: `RoundLifecycleService` 또는 `FederationCoordinatorService`
- [ ] active round 저장/조회 repository를 추가한다.
- [ ] round status model을 추가한다.
  - created / open / closed / published / failed 같은 상태
- [ ] update deduplication 규칙을 추가한다.
- [ ] round close와 pair publication을 한 흐름으로 묶는다.
- [ ] `main_server/src/api/routers/fl_rounds.py`를 실제 endpoint로 구현한다.
  - 현재 active round/task 조회
  - update 업로드
  - round 상태 조회
  - finalize/publish
- [ ] `main_server/src/api/main.py`에 FL router를 연결한다.

완료 기준:

- [ ] 서버만 띄워도 round를 열고 상태를 조회할 수 있다.
- [ ] task publication과 update ingestion이 API로 닫힌다.
- [ ] aggregation 후 새 manifest/state 발행이 runtime 경로에서 재현된다.

### 2. Agent Federated Runtime 닫기

- [ ] `agent/src/services/federation/` 계층을 만든다.
- [ ] active manifest/prototype/task를 가져오는 client/service를 만든다.
- [ ] local event를 training example로 바꾸는 runtime service를 만든다.
  - 현재 simulation 내부 helper를 agent runtime 계층으로 이동
- [ ] `LocalTrainingService`를 실제 runtime에서 호출하는 orchestration을 만든다.
- [ ] update upload client를 만든다.
- [ ] `agent/src/api/routers/training.py`를 실제 endpoint로 구현한다.
  - task pull 또는 수동 trigger
  - local run trigger
  - last training result/status 조회
- [ ] `agent/src/api/main.py`에 training router를 연결한다.

완료 기준:

- [ ] agent 단독으로 active pair를 받아 local update를 생성할 수 있다.
- [ ] training example preparation이 더 이상 script 전용 helper에 묶여 있지 않다.
- [ ] local runtime이 `TrainingTask`만 보고 동작 가능하다.

### 3. Prototype Rebuild Runtime 경로 닫기

- [ ] `main_server/src/services/prototypes/` 아래에 `PrototypeRebuildService`를 추가한다.
- [ ] rebuild input source를 service contract로 고정한다.
- [ ] 새 model revision 발행 시 prototype rebuild가 같은 transaction 흐름으로 이어지게 한다.
- [ ] manifest/state/prototype publication의 순서를 고정한다.
- [ ] `PrototypeBuildState`를 운영 경로에서 어떻게 쓸지 결정한다.
  - exact incremental을 계속 유지할지
  - rebuild 전용 metadata로만 둘지
- [ ] runtime rebuild와 script rebuild의 역할을 분리한다.

완료 기준:

- [ ] simulation helper 없이도 서버가 새 revision용 prototype pair를 운영 경로에서 발행할 수 있다.
- [ ] “서버가 원문을 가져야 하나?” 같은 프라이버시 충돌이 구조적으로 제거된다.

### 4. End-to-End HTTP Federation 검증 추가

- [ ] integration test용 server 1개 + agent N개 시나리오를 추가한다.
- [ ] 실제 HTTP 경로로 task publish -> local update -> upload -> aggregation -> republish를 검증한다.
- [ ] artifact consistency를 검증한다.
  - model revision
  - prototype version
  - base revision compatibility
- [ ] failure path를 검증한다.
  - deadline 초과
  - duplicate update
  - base revision mismatch
  - empty accepted examples

완료 기준:

- [ ] `scripts` simulation 없이도 runtime API 기준 E2E federation이 재현된다.
- [ ] 최소 2-agent integration test가 안정적으로 통과한다.

### 5. Policy / Backend 교체성 강화

- [ ] local training backend를 heuristic 하나 이상으로 확장한다.
  - 예: `DiagonalScaleGradientTrainingBackend`
- [ ] pseudo-label selection policy registry를 현재보다 더 명시적으로 정리한다.
- [ ] aggregation backend도 family별로 교체 가능하게 유지한다.
- [ ] config와 runtime contract 사이의 drift를 줄이기 위해 typed config를 필요한 범위에 도입한다.
  - 우선 FL runtime 관련 config부터
- [ ] 각 backend/policy 교체 시 필요한 테스트 축을 분리한다.

완료 기준:

- [ ] acceptance policy, training backend, aggregation backend를 독립적으로 바꿔도 round runtime이 안 깨진다.
- [ ] 새 backend 추가 시 기존 round lifecycle 코드를 뜯지 않는다.

### 6. Privacy / Robustness Hardening

- [ ] secure aggregation 도입 지점을 명시한다.
- [ ] clipping과 DP를 training logic과 분리된 계층으로 유지한다.
- [ ] malicious / low-quality update 방어 규칙을 추가한다.
  - shape 검증
  - norm 검증
  - stale revision 거부
- [ ] robust aggregation 후보를 분리한다.
  - weighted mean 외 후보 검토

완료 기준:

- [ ] privacy 계층이 training/aggregation 코드와 섞이지 않는다.
- [ ] insecure default와 hardened path의 차이가 명확하다.

## 지금 당장 하지 않을 것

- [ ] full encoder FL
- [ ] private adapter/head를 shared runtime에 성급히 섞기
- [ ] `scripts` simulation을 전부 production runtime으로 옮기기
- [ ] secure aggregation/DP를 round lifecycle 완성 전에 먼저 붙이기

## scripts vs runtime 분리 규칙

`scripts`에 남길 것:

- synthetic shard split
- smoke/benchmark preset
- 실험 결과 dump
- simulation/evaluation harness

`agent` / `main_server`로 옮길 것:

- 실제 round/task/update transport
- 실제 training example preparation
- 실제 prototype rebuild publication
- 실제 round lifecycle state machine

## 다음 액션

가장 먼저 시작할 구현 단위는 아래 3개다.

1. prototype rebuild source와 publication policy 확정
2. `main_server` round lifecycle service + FL router 구현
3. `agent` federation client/orchestration + training router 구현
