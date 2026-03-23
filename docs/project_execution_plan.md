# TraceMind Execution Plan

## 1. 방향 정리

이 프로젝트의 최종 목표는 두 축을 모두 포함한다.

1. 로컬에서 원문을 처리하고 중앙에는 최소 통계만 보내는 privacy-preserving analytics 구조
2. 이후 확장으로 multi-agent privacy-preserving FL과 privacy 기술(`DP`, secure aggregation, 필요 시 동형암호)까지 수용 가능한 구조

다만 여기서 중요한 구분이 있다.

- `WindowSummary`, `NormPack` 같은 cohort 기준은 analytics 산출물이다.
- 모델 파라미터 업데이트는 FL 산출물이다.

즉, 최종적으로는 "로컬 분석 시스템"만 만드는 것이 아니라,
"로컬 분석 시스템을 기반으로 cohort parameter learning과 필요 시 decision-model FL까지 수용 가능한 에이전트/서버 구조"를 만드는 것이다.

다만 구현 순서는 최종 목표와 다르게 가져가는 것이 맞다.

처음부터 `FL rounds`, `model updates`, `secure aggregation`, `HE`를 동시에 잡으면
시스템이 너무 빨리 복잡해지고, 무엇을 주고받아야 하는지 계약도 불안정해진다.

그래서 초기에는 다음 순서를 따른다.

1. 로컬 데이터 처리 파이프라인 확정
2. 중앙 서버와의 최소 계약 확정
3. analytics 기반 MVP 완성
4. 그 위에 FL-ready runtime 계층 추가
5. 로컬 feedback/self-report 신호가 확보될 때만 decision-model FL 추가
6. 마지막에 privacy hardening(`DP`, secure aggregation, HE`) 추가

이 방식은 "최종 목표를 낮추는 것"이 아니라 "복잡도를 통제하는 방식"이다.

---

## 2. 왜 로컬과 계약부터 시작하는가

### 2-1. 계약 정의란 무엇인가

계약 정의(contract definition)는 로컬과 중앙이 주고받는 데이터 구조를 먼저 고정하는 것이다.

이 프로젝트에서 최소한 먼저 고정해야 하는 계약은 아래 세 가지다.

1. `WindowSummary`
2. `NormPack`
3. `AssessmentResult`

의미는 다음과 같다.

- `WindowSummary`
  - 로컬이 중앙으로 보내는 analytics용 요약 통계
- `NormPack`
  - 중앙이 로컬에게 내려주는 cohort 기준
- `AssessmentResult`
  - 로컬이 최종 판단 후 내부적으로 사용하는 결과 객체

그리고 FL 확장을 고려한다면 아래 계약 초안도 함께 잡아두는 것이 좋다.

1. `TrainingTask`
2. `TrainingUpdateEnvelope`
3. `DecisionFeedbackSignal`

의미는 다음과 같다.

- `TrainingTask`
  - 나중에 decision-model FL 단계에서 중앙이 로컬에 내려주는 학습 작업 정의
- `TrainingUpdateEnvelope`
  - 로컬이 중앙으로 보내는 모델 업데이트 단위
- `DecisionFeedbackSignal`
  - self-report, support action, delayed outcome 같은 로컬 학습 신호

이 계약을 먼저 정의하면, 나중에 FastAPI 라우터, DB 스키마, 시뮬레이터, 테스트를 안정적으로 쌓을 수 있다.

### 2-2. 왜 요약 생성이 먼저인가

현재 프로젝트의 핵심 연구 가치는 "원문을 서버로 보내지 않으면서 집단 기준을 학습하고, 최종 판단은 로컬에서 한다"는 점이다.

즉 중앙 서버보다 먼저 중요한 것은:

1. 로컬에서 어떤 정보를 계산할 것인지
2. 그 중 무엇만 중앙에 보낼 것인지
3. 절대로 보내지 말아야 하는 것은 무엇인지

이 세 가지다.

이 경계가 먼저 서야 나중에 FL과 DP/HE를 붙여도 privacy 원칙이 무너지지 않는다.

---

## 3. 최종 아키텍처 목표

### 3-1. 로컬 에이전트

로컬 에이전트는 최종적으로 아래 역할을 가진다.

1. 이벤트 수집
2. 전처리/번역
3. 임베딩
4. 프로토타입 기반 카테고리 점수 계산
5. 시간 윈도우 요약 생성
6. 개인 기준선 계산
7. 중앙 기준(`NormPack`) 반영
8. 최종 로컬 판단
9. 연합학습 참여
10. privacy 계층 적용(`DP`, secure aggregation client, optional HE client)

즉 로컬 에이전트는 단순 수집기가 아니라,
`analytics worker + decision engine + FL client runtime`의 역할을 모두 가진다.

### 3-2. 중앙 서버

중앙 서버는 최종적으로 아래 역할을 가진다.

1. cohort 기반 집계
2. `NormPack` 생성 및 배포
3. FL round orchestration
4. 모델 버전 관리
5. 업데이트 집계
6. secure aggregation coordinator
7. 정책 배포
8. 감사/운영 모니터링

즉 초기에는 analytics server에 가깝지만,
최종적으로는 `analytics coordinator + FL coordinator + policy registry`가 된다.

---

## 4. 단계별 전략

## Phase 0. 공유 계약과 도메인 객체 고정

목표:

- analytics와 FL 모두에 공통으로 쓰일 핵심 도메인 정의

먼저 만들 것:

1. `WindowSummary`
2. `NormPack`
3. `AssessmentResult`
4. `TrainingTask`
5. `TrainingUpdateEnvelope`
6. `DecisionFeedbackSignal`
7. `AgentCapabilities`

여기서 `TrainingTask`, `TrainingUpdateEnvelope`, `DecisionFeedbackSignal`, `AgentCapabilities`까지 초안만 잡아두면
나중에 FL로 확장할 때 구조를 다시 뜯지 않아도 된다.

---

## Phase 1. Local Analytics MVP

목표:

- 로컬에서 원문을 처리하고 `WindowSummary`를 만드는 것

포함:

1. 이벤트 ingest
2. 번역/정규화
3. 임베딩
4. 프로토타입 유사도 점수 계산
5. 7일 윈도우 통계 생성
6. 개인 기준선 계산의 단순 버전
7. 입력 채널은 우선 API/fixture 기반으로 시작하고, 브라우저 확장 프로그램은 후속 입력 어댑터로 붙인다

제외:

1. FL training
2. secure aggregation
3. HE
4. 고급 정책 배포

완료 조건:

- 샘플 입력에서 `WindowSummary`가 안정적으로 생성된다.

---

## Phase 2. Central Normative Server MVP

목표:

- 중앙 서버가 `WindowSummary`를 받아 cohort 기준을 계산한다.

포함:

1. summary 업로드 API
2. cohort별 집계
3. robust aggregation
4. `NormPack` 배포
5. 최소 정책 배포

이 단계의 중앙 서버는 아직 FL coordinator가 아니다.
우선은 `analytics + norms server`다.

완료 조건:

- 로컬 여러 개를 시뮬레이션해 `NormPack`이 생성되고 로컬 판단에 반영된다.

---

## Phase 3. Local Decision MVP

목표:

- `Self-baseline + Peer norm + Persistence`를 결합해 로컬 최종 판단 완성

포함:

1. warm-up period
2. slope/persistence
3. single-spike filtering
4. support resource selection

완료 조건:

- 단발성 이벤트에는 과민반응하지 않고, 지속 변화에서만 판단이 뜬다.

중요:

이 단계의 판단은 우선 `rule-based decision` 또는 명시적 `DecisionPolicy`로 구현하는 것이 맞다.
`NormPack`이 바뀌면 결과는 달라질 수 있지만, 그것은 입력 컨텍스트 변화이지 FL 학습이 아니다.

---

## Phase 4. FL-ready Runtime 설계

목표:

- 현재 로컬 에이전트와 중앙 서버를 나중에 decision-model FL로 확장 가능하게 만든다.

추가할 것:

1. `AgentRegistry`
2. `RoundManager`
3. `ModelRegistry`
4. `TrainingTask`
5. `TrainingUpdateEnvelope`
6. `ClientCheckpointStore`

중요:

이 단계에서 처음으로 네가 초안으로 적은 아래 API들이 의미를 갖기 시작한다.

- `POST /agents/register`
- `POST /agents/heartbeat`
- `GET /fl/rounds/current`
- `POST /fl/rounds/{round_id}/join`
- `POST /fl/rounds/{round_id}/updates`

즉 이 API들은 틀린 게 아니라, MVP보다 한 단계 뒤에 오는 것이다.

다만 이 단계는 "FL이 이미 성립했다"는 뜻이 아니다.
self-report나 feedback 신호 없이 round orchestration만 추가하면 여전히 학습 루프는 닫히지 않는다.

---

## Phase 5. Decision-Model FL MVP

목표:

- 로컬 최종 판단 모델의 파라미터 업데이트를 교환하는 FL 루프 구현

권장 범위:

1. `WindowSummary`, baseline, `NormPack` 기반 feature를 입력으로 쓰는 작은 decision head부터 시작
2. 임베딩 백본이나 프로토타입 자체보다 최종 판단 계층을 우선 학습
3. 라운드 기반 집계
4. basic FedAvg 또는 weighted averaging
5. local self-report, support action, delayed outcome 중 최소 하나를 학습 신호로 사용

왜 이렇게 하냐면:

- 현재 TraceMind의 핵심은 cohort parameter learning과 로컬 판단 분리다.
- 학습 신호가 없는 상태에서 최종 판단을 FL로 돌리면 규칙을 다시 근사하는 수준에 머물 가능성이 크다.
- 작은 decision head부터 시작해야 통신량, 검증 난이도, 해석 가능성을 통제할 수 있다.

완료 조건:

- 중앙이 `TrainingTask`를 배포하고, 로컬이 `DecisionFeedbackSignal`을 바탕으로 업데이트를 올리며, 중앙이 새 모델 버전을 발행한다.
- `NormPack`과 FL 모델 파라미터가 역할상 분리되어 동작한다.

하지 말아야 할 것:

1. `NormPack`을 그대로 "글로벌 모델 파라미터"라고 취급하는 것
2. cohort rarity와 semantic classifier 학습을 한 단계에 섞는 것
3. feedback 신호 없이 decision-model FL을 먼저 도입하는 것

---

## Phase 6. Privacy Hardening

목표:

- FL 단계에 privacy 보강 계층을 추가

권장 순서:

1. transport security
2. secure aggregation
3. client-level DP
4. 필요 시 homomorphic encryption 또는 MPC 계층 검토

이 순서를 권장하는 이유:

- `DP`와 `HE`는 매우 비싸다.
- MVP 이전에 도입하면 디버깅이 거의 불가능해진다.
- 실제 구현에서는 `secure aggregation + 제한적 DP`가 먼저 실용적이다.

### 동형암호에 대한 현실적 판단

동형암호는 "할 수 있으면 좋다"가 아니라 "명확한 이유가 있을 때 선택"하는 기술이다.

초기에는 아래 우선순위를 추천한다.

1. secure aggregation
2. client-side clipping + noise
3. 필요한 경우에만 부분적 HE 검토

즉, "동형암호를 목표에서 빼라"는 뜻은 아니지만,
초기 구조는 HE 없이도 완전히 동작해야 한다.

---

## 5. 추천 폴더 구조

```text
main-server/
  Dockerfile
  src/
    api/
      main.py
      routers/
        health.py
        sync.py
        norms.py
        policies.py
        agents.py
        fl_rounds.py
        models.py
    services/
      ingestion_service.py
      aggregation_service.py
      robust_aggregation_service.py
      norm_pack_service.py
      agent_registry_service.py
      round_manager_service.py
      update_aggregation_service.py
      model_publication_service.py
      privacy_orchestration_service.py
    infrastructure/
      persistence/
        postgres/
      repositories/
      transport/
      privacy/
  tests/
    unit/
    integration/

agent/
  Dockerfile
  chrome-extension/
    manifest.json
    src/
      content/
      background/
      popup/
      options/
      bridge/
  src/
    api/
      main.py
      routers/
        ingest.py
        assessment.py
        sync.py
        training.py
    services/
      preprocess_service.py
      embedding_service.py
      scoring_service.py
      windowing_service.py
      baseline_service.py
      decision_service.py
      sync_service.py
      local_training_service.py
      privacy_guard_service.py
    infrastructure/
      persistence/
        sqlite/
      repositories/
      transport/
      model_adapters/
        embedding/
        translation/
        classification/
      privacy/
  tests/
    unit/
    integration/

shared/
  src/
    domain/
      entities/
        query_event.py
        scored_event.py
        window_summary.py
        baseline_profile.py
        norm_pack.py
        assessment_result.py
        training_task.py
        training_update.py
        agent_profile.py
      value_objects/
        category.py
        cohort_key.py
        schema_version.py
        round_id.py
        model_version.py
      policies/
        decision_policy.py
        cohort_policy.py
        training_policy.py
    contracts/
      sync_contracts.py
      norm_contracts.py
      assessment_contracts.py
      training_contracts.py
    ports/
      embedding_port.py
      translation_port.py
      summary_repository_port.py
      norm_repository_port.py
      sync_client_port.py
      training_client_port.py
      model_registry_port.py
  tests/
    unit/

scripts/
tests/
  federation/
    e2e/

docker-compose.yml
```

이 구조의 포인트는:

1. `한 레포` 안에서 개발 속도를 유지하면서도 `앱 경계`를 처음부터 분리한다.
2. `main-server`와 `agent`는 나중에 별도 레포로 옮기기 쉬운 실행 단위가 된다.
3. `shared`는 루트에 두는 공통 도메인/계약 계층이며, 컨테이너 서비스가 아니라 코드 공유 패키지다.
4. `main-server/Dockerfile`과 `agent/Dockerfile`은 각 실행 단위의 컨테이너 정의를 서비스 폴더 안에 둬서 역할 경계를 유지한다.
5. 루트 `docker-compose.yml`은 여러 실행 단위를 함께 띄우는 조합 정의만 담당한다.
6. 지금 단계에서는 `infra/`를 별도로 두지 않는다. 배포 자산이 늘어나기 전까지는 구조를 평평하게 두는 편이 MVP 속도와 이해도에 유리하다.
7. analytics MVP와 FL 확장을 같은 구조 안에서 수용하되, 서로 다른 책임이 한 트리에 뒤섞이지 않게 한다.
8. `agent/chrome-extension`은 브라우저 입력 수집 전용 어댑터로 두고, 분석 로직은 `agent/src/services`에만 둔다.

---

## 6. 네가 초안으로 만든 중앙 서버 API의 위치

네가 적은 아래 API들은 대부분 나중 단계에서 맞다.

- `POST /api/v1/agents/register`
- `POST /api/v1/agents/heartbeat`
- `GET /api/v1/fl/rounds/current`
- `POST /api/v1/fl/rounds/{roundId}/join`
- `GET /api/v1/fl/rounds/{roundId}/config`
- `GET /api/v1/fl/models/{modelVersion}/manifest`
- `POST /api/v1/fl/rounds/{roundId}/updates`
- `GET /api/v1/fl/rounds/{roundId}/status`

이건 "버릴 설계"가 아니라 `Phase 4+`에 어울리는 설계다.

반대로 지금 먼저 필요한 API는 아래다.

- `POST /api/v1/sync/window-summaries`
- `GET /api/v1/norms/{cohort_key}`
- `GET /api/v1/policies/current`
- `GET /api/v1/health`

즉 네 초안은 시점이 조금 앞서 있을 뿐, 방향이 틀린 것은 아니다.

---

## 7. 문서 저장 전략

이 프로젝트는 문서를 세 층으로 나누는 것이 좋다.

### 7-1. 연구/비전 문서

- 파일 예: `plan.md`
- 용도: 문제 정의, 연구 기여, 발표용 메시지

### 7-2. 구현 계획 문서

- 파일 예: `docs/project_execution_plan.md`
- 용도: MVP 범위, 단계별 구현 순서, 아키텍처 결정

### 7-3. 작업 메모/대화 요약

- 디렉터리: `docs/notes/`
- 파일 예:
  - `docs/notes/2026-03-22-architecture-notes.md`
  - `docs/notes/2026-03-22-api-decisions.md`

이렇게 분리하는 이유:

1. `plan.md`는 계속 깔끔하게 유지할 수 있다.
2. 구현용 의사결정은 별도 추적이 가능하다.
3. 대화 메모를 코드 구조와 섞지 않을 수 있다.

---

## 8. 지금 바로 다음 액션

추천 순서는 아래다.

1. `WindowSummary v1` 문서 작성
2. `NormPack v1` 문서 작성
3. `shared/src/domain/entities/window_summary.py` 작성
4. 모델/데이터셋 리소스 경로와 메타데이터 정리
5. `agent/src/services/windowing_service.py` 구현
6. 중앙 `sync/norms` API 구현
7. `TrainingTask`, `TrainingUpdateEnvelope`, `DecisionFeedbackSignal` 초안 작성
8. 크롬 확장 프로그램 어댑터 연결

즉, 지금은 analytics MVP를 닫되,
FL 확장 포인트는 미리 설계에 심어두되, 실제 decision-model FL은 feedback 신호 확보 이후에 여는 방식으로 가는 것이 맞다.

중요한 순서 원칙은 아래와 같다.

1. 계약을 먼저 고정한다.
2. 모델과 데이터셋은 병렬로 준비하되, 계약이 없는데 파이프라인에 먼저 박아 넣지 않는다.
3. 초기 입력 채널은 테스트 fixture나 로컬 API로 충분하다.
4. 크롬 확장 프로그램은 입력 어댑터이므로, 핵심 분석 로직이 안정된 뒤 붙여도 늦지 않다.
5. `NormPack`과 decision-model parameter는 역할이 다르므로 한 계약으로 합치지 않는다.
