# TraceMind Execution Plan

## 1. 방향 정리

2026-03-29 기준 TraceMind의 활성 목표는 아래와 같다.

1. 로컬에서 원문을 처리한다.
2. 공통 의미 표현 공간은 전역 모델로 공유한다.
3. 위험 해석은 로컬 개인화 상태에 기반해 수행한다.
4. 단일 점수 대신 시계열 누적과 persistence를 통해 최종 로컬 판단을 만든다.
5. 중앙은 `NormPack`을 만드는 서버가 아니라, 모델/라운드/배포를 조정하는 서버가 된다.
6. 필요 시 FL로 공통 표현 모델 또는 그 일부를 점진적으로 개선한다.

중요한 구조적 결정:

- `WindowSummary`는 더 이상 활성 analytics 산출물이 아니다.
- `NormPack`도 더 이상 활성 중앙 산출물이 아니다.
- `PrototypePack`은 유지한다.
- 전역 공유 산출물의 중심은 `ModelManifest`, `TrainingTask`, `TrainingUpdateEnvelope`, `PrototypePack`이다.
- `v1` 개인화는 `same global embedding + different local interpretation`을 사용한다.
- `v1`에서는 학습용 candidate model과 추론용 active pair를 분리하지 않는다.
- `v2` 확장에서만 private adapter 기반 표현 개인화를 연다.

즉 이 프로젝트는 이제
`cohort parameter learning + local decision`
중심 구조가 아니라,
`personalized local inference + federated model improvement`
중심 구조로 해석한다.

중요:

이 전환은 "마음 이상 탐지와 조치" 목표를 버리는 것이 아니다.
달라진 것은 중앙에서 규범을 계산하던 경로를 없애고,
로컬에서 `baseline + 시계열 누적 + persistence + personalization`으로
최종 조치 판단을 만드는 구조를 택했다는 점이다.

---

## 2. 왜 이 방향이 맞는가

### 2-1. 같은 표현의 의미는 개인별로 다르다

심리·정서·distress 신호는 고양이/개 분류처럼 절대적인 해석만으로 처리하기 어렵다.
같은 문장이라도:

1. 평소 표현 강도가 높은 사용자에게는 경미한 상태일 수 있고
2. 평소 거의 그런 표현을 쓰지 않는 사용자에게는 중요한 변화일 수 있다

따라서 공통 임베딩 공간은 유지하되,
최종 해석은 로컬 `baseline`, `threshold`, `personal prototype`, `persistence`가 담당하는 편이 더 자연스럽다.

여기서 `persistence`는 단순 부가 feature가 아니라,
시간에 따라 누적된 상태를 바탕으로
false positive를 줄이고 실제 지원 필요 상황을 더 안정적으로 잡기 위한 핵심 축이다.

### 2-2. 요약 통계 경로보다 모델 중심 경로가 더 직접적이다

기존 normative 경로는 아래 문제를 가진다.

1. `peer norm`을 만들기 위한 별도 중앙 집계 단계가 필요하다.
2. `WindowSummary -> NormPack -> local decision` 경로가 길다.
3. 연구 메시지가 개인화 해석보다는 집단 기준 학습에 더 가깝다.

이번 전환에서는 아래를 택한다.

1. 공통 표현 모델과 bootstrap semantic artifact를 중앙이 배포한다.
2. 로컬은 개인화 상태를 이용해 해석한다.
3. 중앙은 학습 라운드와 모델 publication을 담당한다.

### 2-3. 현재 저장소 자산을 버리지 않아도 된다

유지 가능한 자산:

1. 데이터셋 다운로드/정제/매핑 파이프라인
2. 임베딩/번역 어댑터
3. `PrototypePack` 빌드 및 배포 경로
4. `agent / main-server / shared` 폴더 구조

즉 지금 전환은 "모든 것을 다시 시작"하는 것이 아니라,
문제 정의와 활성 계약을 재배치하는 작업에 가깝다.

---

## 3. 활성 계약 재정의

### 3-1. 먼저 고정할 계약

현재 활성 contract source of truth는 아래로 재정의한다.

1. `PrototypePack`
2. `PrototypeBuildState`
3. `ModelManifest`
4. `TrainingTask`
5. `TrainingUpdateEnvelope`
6. `DecisionFeedbackSignal`
7. `PersonalizationState`
8. `AssessmentResult`

### 3-2. 각 객체의 의미

- `PrototypePack`
  - 공통 의미 공간에서 bootstrap scoring에 쓰는 semantic artifact

- `ModelManifest`
  - 현재 활성 전역 모델, adapter/head, artifact revision, 호환성 정보를 나타내는 배포 메타데이터

- `TrainingTask`
  - 중앙이 로컬에 내려주는 학습 작업 정의
  - 어떤 파라미터 subset을 어떤 objective로 학습할지 담는다

- `TrainingUpdateEnvelope`
  - 로컬이 중앙에 보내는 업데이트 단위
  - round/task/base revision/metrics/payload 메타데이터를 포함한다

- `DecisionFeedbackSignal`
  - self-report, support action, delayed outcome, pseudo-label 등 로컬 학습 신호

- `PersonalizationState`
  - 로컬에만 유지하는 개인화 상태
  - `baseline`, `threshold`, `personal prototype`, calibration metadata, warm-up 상태를 포함한다

- `AssessmentResult`
  - 로컬 최종 판단과 explanation을 담는 결과 객체
  - 지원 필요 여부와 조치 레벨 판단에 쓰인다

### 3-3. 비활성/보관 계약

아래는 보관 상태로 전환한다.

1. `WindowSummary`
2. `NormPack`
3. summary upload 중심 sync envelope

문서와 코드에서 즉시 완전 삭제할 필요는 없지만,
앞으로의 구현 우선순위에서는 제외한다.

---

## 4. 최종 아키텍처 목표

### 4-1. 로컬 에이전트

로컬 에이전트는 최종적으로 아래 역할을 가진다.

1. 이벤트 수집
2. 전처리/번역
3. 임베딩
4. `PrototypePack` 기반 bootstrap scoring
5. 로컬 개인화 해석
6. 시계열 누적과 persistence 계산
7. `AssessmentResult` 생성
8. `DecisionFeedbackSignal` 관리
9. 로컬 학습 후보 생성
10. 로컬 학습 수행
11. `TrainingUpdateEnvelope` 업로드
12. privacy 계층 적용

즉 로컬 에이전트는
`ingest worker + personalized inference engine + local training runtime`
역할을 모두 가진다.

### 4-2. 중앙 서버

중앙 서버는 최종적으로 아래 역할을 가진다.

1. agent registry
2. model registry
3. prototype publication
4. training task publication
5. FL round orchestration
6. update aggregation
7. model publication
8. privacy orchestration
9. 운영/감사 로그

즉 중앙 서버는
`model coordinator + FL coordinator + publication server`
가 된다.

중앙 서버는 아래를 하지 않는다.

1. 원문 텍스트 분석
2. 개인 상태 판정
3. cohort norm 계산

---

## 5. 단계별 전략

## Phase 0. 전환 계약 고정

목표:

- 활성 계약과 보관 계약을 명확히 분리한다.

먼저 만들 것:

1. `ModelManifest`
2. `TrainingTask`
3. `TrainingUpdateEnvelope`
4. `DecisionFeedbackSignal`
5. `PersonalizationState`
6. `AssessmentResult v1` 문서 보강
7. `WindowSummary` 보관 표시

완료 조건:

- 새 문서 기준으로 로컬/중앙이 무엇을 주고받는지 혼동이 없어야 한다.

## Phase 1. Local Personalized Inference MVP

목표:

- 로컬에서 입력 이벤트를 받아 개인화된 `AssessmentResult`를 생성한다.

포함:

1. ingest
2. preprocess
3. optional translation
4. embedding adapter
5. `PrototypePack` scoring
6. `BaselineProfile`
7. `PersonalizationState`
8. time-series state / persistence feature
9. `DecisionService`

제외:

1. 중앙 cohort 집계
2. `NormPack`
3. 실제 FL round

완료 조건:

- 같은 입력 fixture에 대해 deterministic score는 유지되면서,
  개인 `baseline/threshold` 차이로 다른 `AssessmentResult`를 생성할 수 있어야 한다.
- 시계열 누적에 따라 단발성 표현과 지속 변화가 구분되어야 한다.

### Phase 1 현재 기준 세부 실행 순서

1. `QueryEvent -> ScoredEvent` orchestration을 안정화한다.
2. `PrototypePack` 기준 bootstrap scoring을 유지한다.
3. `BaselineService`를 실제 동작 수준으로 올린다.
4. `PersonalizationState`를 직렬화 가능한 로컬 객체로 정의한다.
5. time-series 누적 상태와 persistence feature를 정의한다.
6. `DecisionService`가 `score + personalization state + persistence`를 받아 `AssessmentResult`를 만들게 한다.
7. fixture 기반으로 사용자별 해석 차이와 시간 누적 효과를 검증한다.

### 현재 보류 결정

현재 시점에서는 full encoder FL을 바로 runtime 기본 경로로 승격하지 않는다.

이 결정의 이유는 아래와 같다.

1. 먼저 추론 폐회로를 닫아야 학습 회귀를 검증할 수 있다.
2. pseudo-label만으로 encoder 전체를 계속 업데이트하면 drift 위험이 크다.
3. 초기 FL 범위는 adapter/head 또는 제한된 param subset이 더 통제 가능하다.
4. `v1`에서는 같은 global embedding을 유지한 채 `baseline + threshold + persistence`로 해석을 개인화하는 편이 더 안정적이다.

## Phase 2. Local Update Generation MVP

목표:

- 로컬에서 학습 후보를 만들고, 이를 `TrainingUpdateEnvelope`로 패키징한다.

포함:

1. pseudo-label candidate builder
2. confidence / margin filter
3. feedback signal store
4. local checkpoint store
5. local training service
6. update serializer

완료 조건:

- 로컬이 `TrainingTask`를 받아 update를 재현 가능하게 생성할 수 있어야 한다.
- pseudo-label selection, local training, update의 `base_model_revision`이 같은 active pair에 묶여 있어야 한다.

## Phase 3. Central FL Coordinator MVP

목표:

- 중앙이 agent와 라운드를 관리하고 모델을 다시 배포한다.

포함:

1. `AgentRegistry`
2. `ModelRegistry`
3. `RoundManager`
4. `TrainingTask` publication
5. update aggregation
6. new synchronized `ModelManifest` / `PrototypePack` publication

완료 조건:

- 여러 로컬 update를 받아 새 모델 버전을 발행할 수 있어야 한다.
- 새 `model_revision`이 배포될 때 호환되는 `prototype_version`도 함께 republish되어야 한다.

## Phase 4. Federation End-to-End MVP

목표:

- `agent -> main-server -> agent` FL 루프를 한 번 닫는다.

포함:

1. model fetch
2. task join
3. local train
4. update upload
5. aggregate
6. republish
7. pull new manifest

완료 조건:

- synthetic multi-agent 환경에서 한 라운드가 성공적으로 재현된다.
- `train -> bootstrap subset + client shard`, `validation holdout` 구성으로 round active pair가 교체되어야 한다.

## Phase 5. Selective Encoder FL

목표:

- adapter/head를 넘어 encoder 일부까지 FL 범위를 확장할지 검증한다.

권장 순서:

1. head only
2. adapter or low-rank update
3. selected encoder block
4. full encoder는 마지막 선택지

이 단계 전까지는 같은 query에 대한 raw embedding은 global revision 기준으로 동일하고,
agent별 차이는 주로 로컬 interpretation state에서 발생한다.

완료 조건:

- 성능, drift, 통신량, 회귀 안정성에서 확장 가치가 확인된다.

## Phase 6. Privacy Hardening

목표:

- update 경로에 privacy 보호 계층을 추가한다.

권장 순서:

1. transport security
2. clipping
3. secure aggregation
4. client-level DP
5. 필요 시 추가 암호화 계층

---

## 6. 추천 폴더 구조

현재 폴더 구조는 큰 틀을 유지한다.
다만 책임 해석을 아래처럼 바꾼다.

```text
main-server/
  src/
    api/
      routers/
        health.py
        sync.py
        prototypes.py
        agents.py
        fl_rounds.py
        models.py
    services/
      prototype_pack_service.py
      model_registry_service.py
      round_manager_service.py
      update_aggregation_service.py
      publication_service.py
      privacy_orchestration_service.py

agent/
  src/
    api/
      routers/
        ingest.py
        assessment.py
        sync.py
        training.py
    services/
      preprocess_service.py
      embedding_service.py
      scoring_service.py
      baseline_service.py
      decision_service.py
      prototype_runtime_service.py
      prototype_sync_service.py
      local_training_service.py
      checkpoint_service.py
      privacy_guard_service.py

shared/
  src/
    domain/
      entities/
        query_event.py
        scored_event.py
        prototype_pack.py
        baseline_profile.py
        assessment_result.py
        training_task.py
        training_update.py
        model_manifest.py
        personalization_state.py
        decision_feedback_signal.py
    contracts/
      prototype_contracts.py
      training_contracts.py
      model_contracts.py
```

해석상 내려갈 우선순위:

1. `windowing_service.py`
2. `norm_pack_service.py`
3. summary upload 중심 `sync_contracts.py`

해석상 올라갈 우선순위:

1. `training.py`
2. `round_manager_service.py`
3. `prototype_sync_service.py`
4. `local_training_service.py`
5. `model registry` 계층

---

## 7. 현재 저장소에 대한 전환 해석

### 그대로 재사용할 것

1. `data/`
2. `scripts/datasets/`
3. `scripts/prototypes/`
4. `agent/src/infrastructure/model_adapters/embedding/`
5. `agent/src/infrastructure/model_adapters/translation/`
6. `agent/src/services/scoring_service.py`
7. `agent/src/services/prototype_runtime_service.py`
8. `agent/src/services/prototype_sync_service.py`

### 우선순위를 낮출 것

1. `agent/src/services/windowing_service.py`
2. `main-server/src/services/norm_pack_service.py`
3. `shared/src/domain/entities/window_summary.py`
4. `shared/src/domain/entities/norm_pack.py`

### 새로 채워야 할 것

1. `ModelManifest` 도메인/contract
2. `TrainingTask` contract 세부 필드
3. `TrainingUpdateEnvelope` contract 세부 필드
4. `DecisionFeedbackSignal`
5. `PersonalizationState`
6. `local checkpoint store`
7. `update aggregation logic`

---

## 8. 지금 바로 다음 액션

우선순위는 아래로 고정한다.

1. 새 active contract 문서 작성
2. 보관 문서와 활성 문서 구분 표시
3. `plan.md`, `docs/staged_execution_roadmap.md` 정렬
4. `shared` 도메인 객체 우선순위 재정렬
5. `agent` 쪽 personalized inference fixture 설계
6. 그 다음에만 `training task`와 `update envelope` 코드 초안 작성

하지 말아야 할 것:

1. `WindowSummary` 경로를 더 구현하는 것
2. `NormPack` 생성을 더 보강하는 것
3. full encoder FL부터 바로 구현하는 것
4. privacy 보강 없이 update 업로드를 실제 운영 가정으로 다루는 것

결론적으로, 앞으로의 구현은
`summary/norm analytics completion`
이 아니라
`local personalized inference closure -> local update generation -> central FL coordination`
순서로 진행하는 것이 맞다.
