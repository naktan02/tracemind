# 2026-03-22 Architecture Notes

## 배경

`plan.md`를 기준으로 프로젝트의 최종 목표를 구현 가능한 단계로 나누는 논의를 진행했다.

현재까지 확정된 전제:

1. 임베딩 모델 선정 완료
2. 번역 모델 선정 완료
3. 이후 목표는 중앙 서버 + 개인 로컬 서버 기반 구조
4. 최종적으로는 파라미터 연합학습과 privacy 기술(`DP`, secure aggregation, 필요 시 HE`)까지 포함

---

## 결정 내용

### 1. MVP 시작점은 중앙 서버가 아니라 계약 정의와 로컬 파이프라인이다

초기 구현 순서는 아래로 정리했다.

1. `WindowSummary` 계약 정의
2. `NormPack` 계약 정의
3. 로컬의 `Embedding -> Scoring -> Windowing` 구현
4. 중앙의 cohort 집계 및 `NormPack` 배포
5. 로컬의 `Self-baseline + Peer norm + Persistence` 판단 연결

### 2. 현재 MVP는 고전적인 FL 서버보다 federated analytics에 가깝다

초기 단계의 핵심은 모델 파라미터 교환이 아니라:

- 로컬에서 원문 처리
- 요약 통계 생성
- 중앙에서 집단 기준 계산
- 로컬에서 최종 판단

따라서 초기 중앙 서버는 `analytics + norms server`로 두고,
`FL rounds`, `model updates`, `agent heartbeat`는 후속 단계로 미룬다.

### 3. 파라미터 연합학습은 나중 단계에서 포함한다

이 프로젝트는 최종적으로 parameter FL을 포함한다.

다만 구현 순서는 아래로 정리했다.

1. analytics MVP
2. FL-ready runtime
3. parameter FL MVP
4. privacy hardening(`DP`, secure aggregation, optional HE)

즉 FL을 버리는 것이 아니라,
FL이 올라갈 수 있는 구조를 먼저 만드는 방식으로 진행한다.

### 4. 네가 초안으로 적은 중앙 서버 API는 폐기 대상이 아니다

아래 API들은 틀린 방향이 아니라 시점이 조금 이른 설계로 판단했다.

- `POST /api/v1/agents/register`
- `POST /api/v1/agents/heartbeat`
- `GET /api/v1/fl/rounds/current`
- `POST /api/v1/fl/rounds/{roundId}/join`
- `GET /api/v1/fl/rounds/{roundId}/config`
- `GET /api/v1/fl/models/{modelVersion}/manifest`
- `POST /api/v1/fl/rounds/{roundId}/updates`
- `GET /api/v1/fl/rounds/{roundId}/status`

이 API들은 `Phase 4+`에서 의미가 커진다.

### 5. MVP의 중앙 서버 API는 더 작게 시작한다

초기 중앙 서버는 아래 API 정도로 시작하는 것이 적절하다고 정리했다.

- `POST /api/v1/sync/window-summaries`
- `GET /api/v1/norms/{cohort_key}`
- `GET /api/v1/policies/current`
- `GET /api/v1/health`

### 6. 폴더 구조는 API-first보다 domain-first가 더 적합하다

기존 초안은 기능별 `router/service/repository/schemas/models.py` 반복 구조였다.

논의 결과, 이 프로젝트는 아래 방향이 더 적합하다고 정리했다.

- `src/api`
- `src/core/domain`
- `src/core/ports`
- `src/services/local`
- `src/services/central`
- `src/infrastructure`

핵심은:

1. 도메인 객체를 먼저 정의하고
2. 로컬/중앙 유스케이스를 서비스로 분리하고
3. 모델, DB, 네트워크 구현은 어댑터로 뺀다

### 7. 핵심 계약 객체는 세 가지로 정리했다

가장 먼저 정의할 계약 객체:

1. `WindowSummary`
2. `NormPack`
3. `TrainingUpdateEnvelope`

추가로 초안 단계에서 함께 잡아둘 객체:

1. `AssessmentResult`
2. `TrainingTask`
3. `AgentCapabilities`

### 8. 동형암호는 목표에서 제외하지 않되 뒤로 미룬다

privacy 계층 우선순위는 아래와 같이 정리했다.

1. transport security
2. secure aggregation
3. client-level DP
4. 필요 시 부분적 HE 검토

즉 HE는 연구 목표에서는 열어두되,
초기 구조는 HE 없이도 완전히 동작해야 한다고 정리했다.

---

## 이유

이번 논의에서 위 결정을 내린 이유는 아래와 같다.

1. 프로젝트의 핵심 가치가 "원문을 보내지 않는 구조"에 있기 때문이다.
2. 먼저 로컬이 어떤 요약을 만들지 정해져야 중앙 서버 설계가 안정된다.
3. 처음부터 FL/DP/HE를 모두 넣으면 구현과 디버깅 복잡도가 너무 높아진다.
4. 계약이 먼저 고정되면 임베딩 모델, 번역 모델, 프레임워크, 저장소 구현을 교체하기 쉬워진다.
5. 장기적으로 AI 기능이나 모델 학습 확장을 하더라도 도메인 구조를 유지할 수 있다.

---

## 보류 사항

아직 확정하지 않은 항목:

1. `WindowSummary v1`의 정확한 필드 정의
2. `NormPack v1`의 정확한 필드 정의
3. `TrainingTask`와 `TrainingUpdateEnvelope` 스키마
4. cohort key 구성 방식(연령대, 지역, 언어, 기타 속성 포함 여부)
5. secure aggregation 도입 시점과 구체 프레임워크
6. DP 적용 위치(local clipping only vs local clipping + noise)
7. HE 적용 필요성과 범위

---

## 참고 자료

논의 중 참고한 공개 자료:

1. Google Research의 Federated Analytics 소개
2. TensorFlow Federated
3. `google-parfait/federated-compute`
4. `google-research/federated`
5. Flower

정확히 동일한 공개 구현은 찾지 못했지만,
`federated analytics`, `cross-device FL`, `secure aggregation` 관점에서 참고 가능한 자료로 정리했다.

---

## 다음 액션

우선순위는 아래로 정리했다.

1. `WindowSummary v1` 계약 문서 작성
2. `NormPack v1` 계약 문서 작성
3. 로컬 `windowing_service` 설계
4. 중앙 `sync/norms` API 명세 작성
5. `TrainingTask` 및 `TrainingUpdateEnvelope` 초안 작성
