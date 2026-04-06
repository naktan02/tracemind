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

1. `v1`에서는 backbone은 고정하고 shared adapter만 다룬다.
2. 현재 adapter family는 `diagonal_scale` 하나다.
3. 현재 로컬 update는 heuristic 기반이고, 다음 단계는 gradient 기반 shared adapter FL이다.
4. prototype은 직접 FL 파라미터라기보다 shared state로부터 재생성되는 artifact에 가깝다.

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
7. `scripts/experiments/federated_simulation`은 `RoundLifecycleService`와 prototype rebuild runtime을 직접 조합한다.
8. `scripts/experiments/prototype_strategy`의 single/kmeans는 shared canonical builder를 재사용한다.

아직 남은 핵심:

1. ~~`agent`의 real round client/runtime 닫기~~ ✅ (Phase 2 완료)
2. `diagonal_scale` heuristic update를 gradient backend로 교체
3. runtime의 multi-prototype 지원 확대 여부 결정
4. privacy hardening의 실제 프로토콜 추가

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

1. ~~`agent` round client/runtime 추가~~ ✅ 완료
2. Phase 3: inference pipeline 연결 (scored events → training_examples 주입)
3. Phase 4: end-to-end HTTP integration test (server + agent 실제 round 완주)
4. `DiagonalScaleGradientTrainingBackend` 추가 (Phase 5 선행)
5. local objective에 drift correction 항 추가
6. multi-prototype runtime과 FL 연결 여부 결정

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
