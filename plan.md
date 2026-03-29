# TraceMind Plan

## 1. 문제 정의

TraceMind는 아동·청소년의 온라인 위험 감지 문제를 다룬다.
기존 접근의 핵심 문제는 두 가지다.

1. 중앙 서버 기반 감시
   - 원문 텍스트나 대화 로그를 서버에 모은다.
   - 프라이버시 침해와 윤리적 부담이 크다.

2. 절대적 위험 판정
   - 특정 표현이나 특정 점수 구간을 모든 사용자에게 동일하게 해석한다.
   - 심리 상태, 감정 표현, distress 신호는 개인별 평소 표현 습관에 크게 좌우된다.

이 도메인에서는 같은 문장도 사용자 맥락에 따라 의미가 달라질 수 있다.
즉 공통 표현 공간은 가능하지만, 최종 해석 기준까지 절대적으로 고정하기는 어렵다.

---

## 2. 연구 가설

이 프로젝트의 핵심 가설은 아래와 같다.

1. 텍스트를 위한 공통 의미 표현 공간은 전역 모델로 공유할 수 있다.
2. 그러나 위험 신호 해석은 로컬 개인화 계층에서 수행해야 한다.
3. 따라서 중앙은 집단 규범을 계산하는 서버가 아니라, 전역 모델과 학습 라운드를 조정하는 서버가 되는 편이 더 자연스럽다.

이 전환 이후에도 최종 제품 목표는 변하지 않는다.

1. 들어온 쿼리의 의미를 공통 임베딩 공간에서 표현한다.
2. 이를 개인 기준선과 시계열 누적으로 해석한다.
3. 로컬에서 마음 이상 신호 또는 지원 필요 신호를 판단한다.
4. 필요 시 사용자에게 지원 리소스나 보호적 조치를 제안한다.

즉 TraceMind는 이제
`normative modeling + peer norm`
중심 구조가 아니라,
`privacy-preserving personalized local adaptation`
중심 구조를 목표로 한다.

---

## 3. 핵심 설계 결정

2026-03-29 기준 구조 결정은 아래와 같다.

1. `WindowSummary`와 `NormPack`은 활성 아키텍처에서 제외한다.
2. 전역에서 공유하는 주된 산출물은 `global model parameter`, `ModelManifest`, `PrototypePack`, `TrainingTask`다.
3. 로컬에만 유지하는 주된 상태는 `PersonalizationState`, `BaselineProfile`, `personal threshold`, `personal prototype`, `local checkpoint`다.
4. 중앙은 개인 상태를 판정하지 않는다.
5. 로컬은 원문 처리, 개인화 해석, 로컬 학습, 최종 지원 판단을 담당한다.

중요:

- `PrototypePack`은 유지한다.
- 다만 이는 `NormPack` 대체물이 아니라, 공통 의미 공간에서의 bootstrap semantic artifact다.
- 개인화는 `baseline`, `threshold`, `personal prototype`, 필요 시 `adapter/head`에서 일어난다.
- `v1`은 `same global embedding + different local interpretation`으로 시작한다.
- `v2`에서 필요 시 private adapter를 붙여 embedding 자체도 약하게 개인화한다.

---

## 4. 전체 구조

### 로컬 추론 경로

```text
Raw Query/Event
-> Preprocess / Translation
-> Embedding (global backbone)
-> Prototype Scoring or Personal Head
-> PersonalizationState / Time-Series Accumulator
-> AssessmentResult
-> Local Support Suggestion
```

### 로컬 학습 경로

```text
Raw Query/Event
-> Embedding / Score
-> Pseudo-label or Feedback Signal
-> Local Training Step
-> TrainingUpdateEnvelope
-> Central Aggregation
-> New ModelManifest / PrototypePack
```

---

## 5. 전역과 로컬의 역할 분리

### 전역에서 유지하는 것

1. 임베딩 백본 또는 adapter/head의 공통 파라미터
2. `PrototypePack`
3. `ModelManifest`
4. `TrainingTask`
5. 라운드 상태와 모델 버전

### 로컬에서만 유지하는 것

1. 원문 텍스트와 입력 이벤트
2. `PersonalizationState`
3. 개인 `baseline`
4. 개인 `threshold`
5. 개인 `prototype`
6. 시계열 누적 상태와 persistence feature
7. 로컬 학습용 후보 샘플과 체크포인트
8. 최종 `AssessmentResult`

핵심은 아래 한 줄이다.

> 공통 의미 공간은 전역에서 관리하고,
> 해석과 적응은 로컬에서 수행한다.

---

## 6. 프라이버시 원칙

반드시 지킬 원칙:

1. 원문 텍스트는 로컬에 남긴다.
2. 서버는 개인 위험 판정을 하지 않는다.
3. 로컬 개인화 상태는 기본적으로 서버에 보내지 않는다.
4. 서버로 올라가는 것은 모델 업데이트와 최소 메타데이터다.
5. 모델 업데이트는 privacy-safe가 자동으로 보장되지 않으므로, 이후 단계에서 clipping, secure aggregation, 필요 시 DP를 붙인다.

즉 `summary를 안 보내니까 자동으로 안전하다`는 식의 해석은 금지한다.
이 구조에서도 privacy hardening은 별도 과제다.

---

## 7. 왜 이 방향으로 전환하는가

이 전환의 이유는 단순히 FL을 써보고 싶어서가 아니다.

1. 개인 해석의 중요성
   - 동일 문장의 의미가 사용자 과거 맥락에 따라 달라진다.

2. 변화 감지의 중요성
   - 절대 점수보다 개인 대비 변화량과 지속성이 더 중요하다.

3. 중앙 역할 축소
   - cohort norm server를 유지하지 않아도, 공통 모델과 라운드 orchestration만으로 구조를 더 단순하게 만들 수 있다.

4. 현재 자산 재사용 가능성
   - 이미 확보한 데이터셋, 임베딩 어댑터, `PrototypePack` 빌드 경로를 버리지 않고 전환할 수 있다.

---

## 8. 유지하는 것과 버리는 것

### 유지하는 것

1. 모노레포 구조(`agent`, `main-server`, `shared`)
2. 데이터셋 다운로드/정제/매핑 파이프라인
3. 임베딩 모델 어댑터와 번역 어댑터
4. `PrototypePack` 기반 bootstrap scoring
5. 로컬 `baseline` 개념
6. 로컬 최종 판단 엔진

### 버리는 것

1. `WindowSummary`
2. `NormPack`
3. cohort normative aggregation
4. summary upload 중심 sync 계약
5. `peer norm` 기반 연구 메시지

---

## 9. 새 연구 메시지

TraceMind는 더 이상
“집단 규범을 학습해 또래 대비 이례성을 판단하는 시스템”
을 1차 목표로 두지 않는다.

새 메시지는 아래와 같다.

> 본 연구는 원문을 중앙 서버에 보내지 않고,
> 공통 의미 표현 공간과 로컬 개인화 상태를 결합하여
> 사용자별 변화와 시계열 맥락에 맞는 해석을 수행하고,
> 필요 시 연합학습으로 공통 표현 모델을 점진적으로 개선하는
> privacy-preserving personalized local adaptation 시스템을 제안한다.

---

## 10. 구현 원칙

1. 도메인과 계약을 먼저 고정한다.
2. full encoder FL을 첫 단계로 강제하지 않는다.
3. 먼저 로컬 추론 폐회로를 안정화한다.
4. 그다음 로컬 update 생성과 중앙 aggregation을 붙인다.
5. 초기 FL 범위는 작은 adapter/head 또는 제한된 param subset부터 시작한다.
6. full encoder FL은 충분한 검증 이후의 확장 옵션으로 둔다.
7. 초기 개인화는 해석 계층에서 처리하고, 표현 계층 개인화는 후속 단계로 미룬다.

이 순서를 택하는 이유는 다음과 같다.

1. 학습 신호 정의가 먼저 명확해야 한다.
2. pseudo-label만으로 encoder 전체를 바로 돌리면 drift 위험이 크다.
3. 작은 범위부터 시작해야 통신량, 안정성, 회귀 검증을 통제하기 쉽다.

---

## 11. 구현 로드맵

### Phase 0. 계약 재정의

먼저 아래 활성 계약을 정의한다.

1. `PrototypePack`
2. `ModelManifest`
3. `TrainingTask`
4. `TrainingUpdateEnvelope`
5. `DecisionFeedbackSignal`
6. `PersonalizationState`
7. `AssessmentResult`

### Phase 1. 로컬 개인화 추론 MVP

목표:

- `QueryEvent -> Embedding -> Score -> PersonalizationState -> Time-Series State -> AssessmentResult`
  폐회로를 안정화한다.

이 단계의 핵심은 단일 이벤트 점수 자체가 아니라,
반복되는 쿼리 흐름에서 개인 대비 변화량과 지속성을 누적해
최종 조치 판단에 쓸 수 있는 로컬 평가를 만드는 것이다.

### Phase 2. 로컬 학습/update 생성 MVP

목표:

- pseudo-label 또는 feedback 기반으로 로컬 update를 만든다.

### Phase 3. 중앙 FL coordinator MVP

목표:

- 모델 버전 관리, round orchestration, update aggregation을 구현한다.

### Phase 4. federation end-to-end 검증

목표:

- 여러 agent가 참여하는 학습/배포 루프를 재현한다.

### Phase 5. selective encoder FL

목표:

- 충분한 검증 이후 필요한 범위에서 encoder 일부까지 FL 범위를 넓힌다.

### Phase 6. privacy hardening

목표:

- clipping, secure aggregation, 필요 시 DP를 추가한다.

---

## 12. 지금 바로 해야 할 일

1. 비전 문서와 실행 계획 문서를 새 방향으로 재작성한다.
2. `WindowSummary/NormPack` 문서를 비활성 또는 보관 상태로 표시한다.
3. 새로운 active contract 문서를 작성한다.
4. `shared`의 차세대 도메인 객체 우선순위를 재정렬한다.
5. 그 다음에만 코드 정리와 API 재배치를 시작한다.

현재 시점의 중요한 판단은 이미 끝났다.
이제부터는 `summary/norm server`를 보강하는 것이 아니라,
`personalized local inference + FL runtime`를 중심으로 실제 구현 순서를 다시 잡으면 된다.
