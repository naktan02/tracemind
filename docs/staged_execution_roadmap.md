# TraceMind Staged Execution Roadmap

## 1. 목적

이 문서는 TraceMind를
`personalized local adaptation + FL runtime`
구조로 단계적으로 닫아 가기 위한 실행 계획을 정리한다.

2026-03-29 전환 결정 이후, 이 문서는 더 이상
`WindowSummary -> NormPack`
경로를 활성 기본 레일로 취급하지 않는다.

이 문서의 역할은 아래와 같다.

1. 지금 당장 구현할 것과 나중 확장할 것을 분리한다.
2. 로컬 해석과 전역 학습의 책임을 분리한다.
3. full encoder FL과 adapter/head FL의 우선순위를 분리한다.
4. privacy hardening을 구조 바깥 별도 단계로 명확히 둔다.
5. 사용자 확인이 필요한 제품/연구 판단 지점을 문서화한다.

---

## 2. 문서 사용 순서

이 문서는 아래 문서들과 함께 읽는 것을 전제로 한다.

1. [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
   - 연구 비전, 문제 정의, 전환 사유
2. [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
   - 구현 순서, 책임 분리, 유지/폐기 범위
3. [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
   - 실제 단계별 산출물과 검증 기준
4. 활성 계약 문서
   - [`docs/contracts/model_manifest_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/model_manifest_v1.md)
   - [`docs/contracts/training_task_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/training_task_v1.md)
   - [`docs/contracts/training_update_envelope_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/training_update_envelope_v1.md)
   - [`docs/contracts/decision_feedback_signal_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/decision_feedback_signal_v1.md)
   - [`docs/contracts/personalization_state_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/personalization_state_v1.md)
   - [`docs/contracts/prototype_pack_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_pack_v1.md)

---

## 3. 최종 목표

TraceMind의 최종 목표는 아래 두 축을 모두 포함한다.

1. 원문을 로컬에 남긴 채 개인화된 해석을 수행하는 추론 구조
2. 필요 시 공통 표현 모델 또는 그 일부를 FL로 개선하는 학습 구조

그리고 이 둘 위에서 변하지 않는 제품 목표는 다음이다.

1. 시계열적으로 누적되는 마음 이상 신호를 로컬에서 감지한다.
2. 단발 이벤트가 아니라 변화와 지속성을 기준으로 판단한다.
3. 최종적으로는 사용자에게 지원 리소스나 보호적 조치를 제안한다.

중요한 구분:

- `PrototypePack`, `ModelManifest`, `TrainingTask`, `TrainingUpdateEnvelope`는 활성 shared artifact다.
- `PersonalizationState`, `AssessmentResult`, 로컬 체크포인트는 로컬 상태다.
- `WindowSummary`, `NormPack`은 보관 경로다.
- 서버는 `peer norm`을 만들지 않는다.

---

## 4. 핵심 원칙

1. 도메인과 계약을 먼저 고정한다.
2. 원문은 서버로 보내지 않는다.
3. 개인화 상태는 기본적으로 서버로 보내지 않는다.
4. 중앙은 모델/라운드/배포를 담당하고, 개인 판정은 하지 않는다.
5. 먼저 로컬 추론 폐회로를 닫고 그 다음 학습 루프를 연다.
6. full encoder FL은 마지막 확장 단계로 둔다.
7. 모델 update는 privacy-safe가 자동으로 보장되지 않는다는 점을 항상 전제로 둔다.

사용자 확인이 필요한 대표 항목:

1. pseudo-label을 정식 학습 신호로 쓸지 여부
2. self-report / support action / delayed outcome 중 어떤 feedback을 도입할지
3. FL 범위를 head, adapter, selected encoder block, full encoder 중 어디까지 열지
4. secure aggregation과 DP 도입 시점

---

## 5. 실행 레일

TraceMind는 아래 두 레일로 이해하는 것이 가장 안전하다.

### 5-1. Personalized Inference 레일

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

설명:

1. 현재 MVP의 본체다.
2. 중앙 서버가 없어도 로컬 단독으로 성립한다.
3. 개인화된 해석과 변화 감지를 담당한다.
4. 최종 조치 판단은 시계열 누적 결과를 포함해 내려진다.

### 5-2. Federated Training 레일

```text
Local Features / Signals
-> DecisionFeedbackSignal or Pseudo-label
-> Local Training
-> TrainingUpdateEnvelope
-> Central Aggregation
-> New ModelManifest / PrototypePack
```

설명:

1. 로컬 추론 레일 위에 얹히는 확장 계층이다.
2. 초기에는 작은 param subset부터 시작한다.
3. full encoder FL은 후반 확장 항목이다.

---

## 6. 단계별 상세 계획

### Phase 0. 전환 계약 문서 정리

목표:

- 활성 계약과 보관 계약을 분리해 이후 구현 기준을 고정한다.

선행조건:

- 사용자 전환 결정 완료

구현:

1. `ModelManifest v1` 문서 작성
2. `TrainingTask v1` 문서 작성
3. `TrainingUpdateEnvelope v1` 문서 작성
4. `DecisionFeedbackSignal v1` 문서 작성
5. `PersonalizationState v1` 문서 작성
6. `WindowSummary` 문서 보관 표시
7. 실행 인덱스 갱신

산출물:

1. `docs/contracts/model_manifest_v1.md`
2. `docs/contracts/training_task_v1.md`
3. `docs/contracts/training_update_envelope_v1.md`
4. `docs/contracts/decision_feedback_signal_v1.md`
5. `docs/contracts/personalization_state_v1.md`

검증:

1. 활성/보관 contract 구분이 명확하다.
2. 앞으로 구현할 API와 domain object 이름이 문서에서 일관된다.
3. `WindowSummary/NormPack`이 활성 계획에 다시 등장하지 않는다.

사용자 확인 필요:

- 없음

### Phase 1. `shared` 도메인 계층 재정렬

목표:

- `agent`와 `main-server`가 동일한 FL/runtime 객체를 참조하게 만든다.

선행조건:

1. Phase 0

구현:

1. `shared/src/domain/entities/model_manifest.py`
2. `shared/src/domain/entities/training_task.py`
3. `shared/src/domain/entities/training_update.py`
4. `shared/src/domain/entities/personalization_state.py`
5. `shared/src/domain/entities/decision_feedback_signal.py`
6. 새 contract 직렬화 유틸 초안

산출물:

1. 직렬화/역직렬화 가능한 새 도메인 객체
2. fixture 기반 contract test

검증:

1. 같은 fixture를 `agent`와 `main-server`가 동일하게 해석한다.
2. summary/norm 객체를 활성 경로에서 참조하지 않는다.

사용자 확인 필요:

- 없음

### Phase 2. 로컬 개인화 추론 MVP

목표:

- 입력 이벤트에서 개인화된 `AssessmentResult`를 안정적으로 생성한다.

선행조건:

1. Phase 0
2. Phase 1

구현:

1. ingest
2. preprocess
3. optional translation
4. embedding adapter
5. `PrototypePack` scoring
6. `BaselineService`
7. `PersonalizationState`
8. time-series accumulator
9. `DecisionService`

산출물:

1. fixture 입력
2. personalized inference result
3. service-level unit test
4. ingest route에서 시작되는 end-to-end fixture test

검증:

1. 같은 입력과 같은 global artifact에 대해 같은 raw score가 재현된다.
2. 서로 다른 `PersonalizationState`를 적용하면 다른 해석 결과가 나올 수 있다.
3. 단발 high score와 지속 high score가 다른 최종 판단으로 이어져야 한다.
4. 로컬 해석 차이가 문서의 문제 정의와 맞아떨어진다.

사용자 확인 필요:

1. 개인 threshold 기본 정책
2. warm-up 길이
3. personal prototype 허용 여부

### Phase 3. 로컬 학습/update 생성 MVP

목표:

- 로컬이 학습 후보를 선별하고 update를 안정적으로 생성한다.

선행조건:

1. Phase 2

구현:

1. pseudo-label 후보 생성
2. confidence / margin filter
3. `DecisionFeedbackSignal` 저장
4. local checkpoint store
5. local training service
6. `TrainingUpdateEnvelope` serializer

산출물:

1. 로컬 학습 시뮬레이션
2. update fixture
3. training 관련 unit test

검증:

1. 같은 seed와 task에서 같은 update shape가 재현된다.
2. update가 base model revision과 정확히 연결된다.
3. drift 방지를 위한 최소 filter가 존재한다.

사용자 확인 필요:

1. pseudo-label 허용 범위
2. 학습 objective 종류
3. 로컬 epochs 상한

### Phase 4. 중앙 FL Coordinator MVP

목표:

- 중앙이 모델 버전과 라운드를 관리한다.

선행조건:

1. Phase 1
2. Phase 3

구현:

1. `POST /agents/register`
2. `POST /agents/heartbeat`
3. `GET /fl/rounds/current`
4. `POST /fl/rounds/{round_id}/join`
5. `POST /fl/rounds/{round_id}/updates`
6. `GET /models/current`
7. `GET /prototypes/current`

산출물:

1. 라운드 메타데이터
2. update 수집 경로
3. model publication 경로

검증:

1. 중앙이 update를 round 단위로 구분한다.
2. 새 `ModelManifest`가 발행되면 agent가 pull 가능한 상태가 된다.
3. 중앙이 개인 판정 결과를 저장하지 않는다.

사용자 확인 필요:

1. 라운드 운영 정책
2. agent 최소 참여 수
3. aggregation 전략

### Phase 5. End-to-End Federation 폐회로

목표:

- 여러 agent를 이용해 FL 루프를 한 바퀴 닫는다.

선행조건:

1. Phase 2
2. Phase 3
3. Phase 4

구현:

1. model/prototype pull
2. task join
3. local train
4. update upload
5. aggregation
6. republish
7. repull and reuse

산출물:

1. `tests/federation`
2. synthetic scenario set
3. e2e smoke path

검증:

1. `model publish -> local train -> update upload -> aggregate -> republish` 전체 사이클이 동작한다.
2. payload에 원문이 포함되지 않는다.
3. 실패 위치를 `local inference`, `local training`, `aggregation`, `publication`으로 분리해 설명할 수 있다.

사용자 확인 필요:

- 없음

### Phase 6. 입력 어댑터 연결

목표:

- 브라우저 입력 채널을 실제 로컬 파이프라인에 연결한다.

선행조건:

1. Phase 2
2. Phase 5

구현:

1. `agent/chrome-extension`
2. local bridge
3. payload sanitization
4. source metadata 최소화

산출물:

1. extension -> local bridge path
2. adapter integration test

검증:

1. 입력 채널이 바뀌어도 `agent/src/services` 핵심 추론/학습 로직은 유지된다.
2. 확장 프로그램이 privacy 경계를 깨지 않는다.

사용자 확인 필요:

1. 우선 지원할 입력 채널
2. 수집 허용 범위

### Phase 7. 모델/데이터셋 고도화

목표:

- 구조를 흔들지 않고 embedding model, prototype dataset, objective를 개선한다.

선행조건:

1. Phase 2
2. Phase 5

구현:

1. embedding adapter 교체 실험
2. prototype dataset 버전 실험
3. pseudo-label threshold 실험
4. adapter/head objective 실험

산출물:

1. benchmark 결과
2. candidate model report

검증:

1. 구조를 바꾸지 않고 모델만 바꿔도 실험 가능해야 한다.
2. regression 지표가 유지된다.

사용자 확인 필요:

1. 기준 모델 승격 정책
2. evaluation 우선 지표

### Phase 8. Selective Encoder FL

목표:

- full encoder FL을 포함한 확장 범위를 점진적으로 검토한다.

선행조건:

1. Phase 5
2. Phase 7

구현:

1. head-only FL
2. adapter/LoRA FL
3. selected encoder block FL
4. 필요 시 full encoder FL

산출물:

1. param subset별 실험 결과
2. 통신량/안정성 비교표

검증:

1. 작은 param subset보다 명확한 이득이 있을 때만 범위를 넓힌다.
2. drift와 불안정성이 허용 범위를 넘으면 rollback 기준이 있다.

사용자 확인 필요:

1. full encoder FL 개시 여부
2. param subset 확대 기준

### Phase 9. Privacy Hardening

목표:

- update 경로에 privacy 보호 계층을 붙인다.

선행조건:

1. Phase 5

구현:

1. transport security
2. update clipping
3. secure aggregation
4. client-level DP
5. 필요 시 추가 암호화 계층

산출물:

1. privacy layer 설계 문서
2. 관련 integration test

검증:

1. privacy 계층이 학습 루프를 깨지 않는다.
2. 보호 수준과 성능 저하의 tradeoff가 문서화된다.

사용자 확인 필요:

1. 초기 운영에 필수로 넣을 보호 단계
2. DP noise budget 정책

---

## 7. 단계별 검증 전략

각 단계는 아래 질문에 답할 수 있어야 닫힌다.

### Phase 2까지

1. 개인화 해석이 실제로 추론 결과 차이를 만들 수 있는가
2. 그 차이가 arbitrary가 아니라 `baseline/threshold`로 설명 가능한가

### Phase 3까지

1. 로컬 update가 재현 가능한가
2. update가 어떤 base revision에서 나왔는지 추적 가능한가

### Phase 4까지

1. 중앙이 모델과 task를 버전 단위로 관리할 수 있는가
2. 잘못된 update를 round 밖으로 격리할 수 있는가

### Phase 5까지

1. end-to-end 학습 루프가 실제로 닫히는가
2. 실패 분석이 가능한가

### Phase 8까지

1. full encoder FL이 정말 필요한가
2. adapter/head보다 얻는 이득이 충분한가

---

## 8. 지금 바로 착수할 우선순위

1. `ModelManifest`, `TrainingTask`, `TrainingUpdateEnvelope`, `DecisionFeedbackSignal`, `PersonalizationState` 문서 확정
2. `shared` 도메인 우선순위 재조정
3. personalized inference fixture 작성
4. `BaselineService`와 `DecisionService` 실구현 범위 확정
5. `training` 라우터와 `round_manager`를 활성 경로로 승격

지금 하지 않을 것:

1. `WindowSummary` 생성 경로 보강
2. `NormPack` 생성 경로 보강
3. feedback 신호 정의 없이 full encoder FL부터 구현

---

## 9. 현재 계획에서 부족한 부분

이 전환 이후 새로 구체화가 필요한 항목은 아래와 같다.

1. `ModelManifest`의 artifact 참조 방식
2. `TrainingTask`의 objective 타입
3. `TrainingUpdateEnvelope`의 payload 형식
4. `DecisionFeedbackSignal`의 taxonomy
5. `PersonalizationState`의 로컬 저장 형식
6. rollback 가능한 model publication 정책
7. pseudo-label 품질 관리 기준
8. param subset별 FL 운영 전략

---

## 10. 권장 결론

TraceMind의 다음 구현 순서는 아래로 고정하는 것이 맞다.

1. 활성 계약 재정의
2. 로컬 개인화 추론 폐회로 완성
3. 로컬 update 생성
4. 중앙 FL coordinator
5. federated e2e
6. selective encoder FL
7. privacy hardening

즉 이제 프로젝트는
`summary/norm analytics MVP`
를 닫는 것이 아니라,
`personalized inference + federated update runtime`
을 닫는 방향으로 진행한다.
