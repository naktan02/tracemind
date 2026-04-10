# TraceMind Execution Plan

## 1. 현재 활성 목표

현재 TraceMind의 활성 목표는 아래 네 가지다.

1. 로컬에서 원문을 처리한다.
2. 공통 의미 표현 공간은 전역 모델과 shared artifact로 유지한다.
3. 해석과 최종 판단은 로컬 개인화 상태로 수행한다.
4. 필요 시 FL로 shared representation 계층만 점진적으로 개선한다.

중요:

- `WindowSummary`, `NormPack`은 활성 경로가 아니다.
- `PrototypePack`은 유지한다.
- `v1`은 `same global representation + different local interpretation`이다.
- `v2`에서만 private adapter/head 기반 표현 개인화를 연다.

## 2. 활성 계약

현재 source of truth는 문서보다 코드 계약이 우선이다.

1. `shared/src/contracts/README.md`
2. `shared/src/contracts/adapter_contracts.py`
3. `shared/src/contracts/training_contracts.py`
4. `shared/src/domain/entities/training/`

현재 활성 객체는 아래다.

1. `PrototypePack`
2. `PrototypeBuildState`
3. `ModelManifest`
4. `SharedAdapterState / SharedAdapterUpdate`
5. `TrainingTask`
6. `TrainingUpdateEnvelope`
7. `DecisionFeedbackSignal`
8. `PersonalizationState`
9. `AssessmentResult`

## 3. 활성 아키텍처

### 3-1. 로컬 추론 레일

```text
Raw Event
-> Preprocess / Translation
-> Embedding
-> Prototype Scoring or Personal Head
-> PersonalizationState
-> Time-Series Accumulator / Persistence
-> DecisionPolicy
-> AssessmentResult
```

### 3-2. 연합 학습 레일

```text
Raw Event / Local Signal
-> Pseudo-label or Feedback Signal
-> Local Training
-> SharedAdapterUpdate
-> Central Aggregation
-> New ModelManifest / PrototypePack pair
```

설명:

1. `v1`에서는 backbone은 고정하고 shared adapter/head family를 다룬다.
2. 현재 활성 family는 `diagonal_scale`, `classifier_head` 두 개다.
3. `classifier_head` family는 simulation과 row-source multiview 경로까지 닫혔고,
   stored scored event를 쓰는 real agent 경로는 아직 `diagonal_scale`만 안전하다.
   현재 live agent API는 지원하지 않는 조합이 오면 `unsupported_runtime`으로 조기 종료한다.
4. 현재 로컬 update는 heuristic 기반과 classifier-head FixMatch-style consistency
   두 경로가 있으며, local training / example generation / scorer / acceptance /
   privacy 축은 각각 독립적으로 교체 가능하게 정리됐다.
5. prototype은 직접 FL 파라미터라기보다 shared state로부터 재생성되는 artifact에 가깝다.

## 4. 구현 배치 규칙

1. 공용 계산 규칙과 canonical 알고리즘은 `shared`에 둔다.
2. agent-owned local training/inference runtime은 `agent`에 둔다.
3. server-owned round/rebuild/publication orchestration은 `main_server`에 둔다.
4. `scripts`는 위 코어를 조합하는 실험층으로만 유지한다.
5. 운영 후보 로직을 `scripts`에 먼저 만들고 나중에 복사하는 흐름은 허용하지 않는다.

## 5. 현재 구현 상태

완료 또는 정리된 방향:

1. shared adapter 계약을 일반화했다.
2. training backend와 privacy guard를 분리했다.
3. 서버 aggregation을 family object 기준으로 조합하게 정리했다.
4. 계약 필드 의미를 `shared/src/contracts` 가까이에 두기 시작했다.
5. `main_server`는 round lifecycle, update acceptance policy, prototype rebuild runtime을 소유한다.
6. `agent`는 runtime training example builder를 소유한다.
7. `agent`는 `RoundClient`, `FederationRuntimeService`, training API(`run-current-task`, `status`)를 가진다.
8. `scripts/experiments/federated_simulation`은 `RoundLifecycleService`와 prototype rebuild runtime을 직접 조합한다.
9. `scripts/experiments/prototype_strategy`의 single/kmeans는 shared canonical builder를 재사용한다.
10. 기본 local training objective/selection fallback은 `shared/src/contracts/training_contracts.py`에 canonical builder로 모였다.
11. scorer backend와 example-generation backend를 독립 축으로 분리했다.
12. 서버는 `adapter_family`, `aggregation_backend`를 server-owned config axis로 고른다.
13. secure aggregation 메타데이터는 typed contract로 승격했다.

아직 남은 핵심:

1. 두 번째 real family(`classifier_head`)를 simulation/runtime 레일에 추가했다.
2. 아직 두 번째 real aggregation backend는 없다.
3. integration test infra를 안정화하고 multi-agent HTTP 시나리오를 확대한다.
4. secure aggregation / DP / robust aggregation의 실제 runtime 구현을 붙인다.
5. stored scored event 기반 real agent 경로에도 classifier-head family를 연결할지 결정한다.
6. 필요하면 learned scorer 또는 richer example-generation backend를 구현한다.

## 6. Phase 요약

### Phase 0. 계약 정리

- 활성 contract와 보관 contract를 분리한다.

### Phase 1. 로컬 개인화 추론 MVP

- `AssessmentResult`를 안정적으로 생성한다.

### Phase 2. 로컬 update 생성 MVP

- pseudo-label 또는 feedback 기반 local update를 생성한다.

### Phase 3. 중앙 FL coordinator MVP

- task publication, update 수집, aggregation, revision 발행을 닫는다.

### Phase 4. end-to-end federation

- 여러 agent가 참여하는 배포/학습 루프를 검증한다.

### Phase 5. richer shared adapter

- 필요 시 diagonal scale보다 표현력 있는 shared adapter로 확장한다.

### Phase 6. privacy hardening

- clipping, secure aggregation, DP, 필요 시 HE를 붙인다.

## 7. 다음 우선순위

가장 자연스러운 다음 작업은 아래 순서다.

1. 두 번째 real `aggregation_backend` 하나를 추가
2. end-to-end HTTP integration test를 multi-agent/agent API 기준으로 확장
3. `DiagonalScaleGradientTrainingBackend` 추가
4. classifier-head family의 real agent stored-event 경로 확장 여부 결정
5. 필요 시 learned scorer 또는 richer example-generation backend 추가
6. classifier-head bootstrap을 multi-prototype까지 확장할지 결정

## 8. 검증 기준

### 로컬 추론

1. 같은 입력과 같은 shared artifact에 대해 raw score가 재현된다.
2. 다른 `PersonalizationState`를 적용하면 다른 해석 결과가 나올 수 있다.
3. 단발 high score와 지속 high score가 다른 최종 판단으로 이어진다.

### shared FL

1. update가 base revision과 호환된다.
2. aggregation 후 새 revision이 일관되게 발행된다.
3. shared adapter drift가 과도하지 않다.
4. prototype 재생성이 새 revision과 일치한다.

### privacy

1. 원문이 서버로 올라가지 않는다.
2. clipping/secure aggregation/DP는 training logic과 분리된 계층으로 붙는다.

## 9. 사용자 확인이 필요한 결정

1. pseudo-label을 정식 학습 신호로 얼마나 신뢰할지
2. FL 범위를 adapter/head에서 어디까지 열지
3. private adapter/head를 언제 도입할지
4. secure aggregation과 DP 도입 시점
5. multi-prototype를 runtime까지 확장할지
