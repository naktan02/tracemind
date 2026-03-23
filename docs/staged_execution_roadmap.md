# TraceMind Staged Execution Roadmap

## 1. 목적

이 문서는 TraceMind를 한 번에 크게 만들지 않고,
가장 작은 검증 가능한 폐회로부터 단계적으로 닫아 가기 위한 실행 계획을 정리한다.

이 문서의 역할은 아래와 같다.

1. 지금 당장 구현할 것과 나중 확장할 것을 분리한다.
2. 각 단계의 입력, 출력, 검증 기준을 명확히 한다.
3. 사용자 판단이 필요한 지점을 문서에 표시한다.
4. analytics 산출물과 FL 산출물을 혼동하지 않게 한다.
5. 이 프로젝트는 단기적인 임시 해결보다, 교체 가능하고 모듈화된 장기 확장형 구조를 우선한다.
한 번 돌아가는 구현보다 이후 모델 교체, 입력 채널 확장, FL, privacy 계층 추가에도 재작업이 적은 구조를 목표로 한다.

---

## 2. 문서 사용 순서

이 문서는 아래 문서들과 함께 읽는 것을 전제로 한다.

1. [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
   - 연구 비전, 문제 정의, 발표 메시지
2. [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
   - 구현 순서, 아키텍처 결정, 단계 해석
3. [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
   - 실제 실행 단위, 단계별 상세 작업, 검증과 확인 게이트
4. [`docs/execution_index.md`](/home/jmgjmg102/tracemind_server/docs/execution_index.md)
   - 문서 진입점과 참조 경로

---

## 3. 최종 목표

TraceMind의 최종 목표는 두 축을 모두 포함한다.

1. 로컬에서 원문을 처리하고 중앙에는 최소 통계만 보내는 privacy-preserving analytics 구조
2. 이후 multi-agent privacy-preserving FL까지 수용 가능한 구조

중요한 구분:

- `WindowSummary`, `NormPack`은 analytics 산출물이다.
- `DecisionTask`, `TrainingUpdateEnvelope`, `global decision-model parameter`는 FL 산출물이다.
- `NormPack`은 글로벌 모델 파라미터가 아니다.
- feedback 신호가 없으면 decision-model FL 단계는 열지 않는다.

---

## 4. 핵심 원칙

1. 도메인과 계약을 먼저 고정한다.
2. 원문은 서버로 보내지 않는다.
3. 중앙은 개인 판정을 하지 않는다.
4. 로컬 판단과 중앙 집계를 같은 계층에 섞지 않는다.
5. analytics 경로와 FL 경로를 분리한다.
6. 사용자의 제품 판단이나 연구 범위 판단이 필요한 변경은 임의로 확정하지 않고 확인 후 진행한다.

사용자 확인이 필요한 대표 항목:

1. cohort 분할 기준 변경
2. feedback/self-report 수집 도입
3. 판단 정책의 임계값 또는 리소스 제안 정책 변경
4. FL 단계 활성화 여부
5. privacy 원칙에 영향을 주는 데이터 필드 추가

---

## 5. 실행 레일

TraceMind는 아래 두 레일로 이해하는 것이 가장 안전하다.

### 5-1. Analytics 레일

```text
Raw Event
-> Preprocess / Translation
-> Embedding
-> Prototype Scoring
-> WindowSummary
-> Central Aggregation
-> NormPack
-> Local DecisionPolicy
-> AssessmentResult
```

설명:

1. 현재 MVP의 본체다.
2. feedback 신호가 없어도 성립한다.
3. `또래 평균`, `분산`, `prevalence` 같은 cohort parameter를 학습한다.

### 5-2. Optional FL 레일

```text
Local Features
-> DecisionFeedbackSignal
-> Local Decision Model Update
-> TrainingUpdateEnvelope
-> Central Aggregation
-> Global Decision Model
```

설명:

1. feedback 또는 self-report 신호가 있을 때만 열린다.
2. `NormPack`과 별개로 최종 판단 계층의 파라미터를 교환한다.
3. analytics 레일 위에 얹히는 확장 계층이다.

---

## 6. 단계별 상세 계획

### Phase 0. 계약 우선 정리

목표:

- 로컬과 중앙이 주고받는 핵심 객체의 의미를 먼저 고정한다.

선행조건:

- 없음

구현:

1. `WindowSummary v1` 필드 정의
2. `NormPack v1` 필드 정의
3. `AssessmentResult v1` 필드 정의
4. 최소 sync contract 초안 작성
5. `cohort_key` 기본 규칙 정의

산출물:

1. `docs/contracts/window_summary_v1.md`
2. `docs/contracts/norm_pack_v1.md`
3. 필요 시 `docs/contracts/assessment_result_v1.md`
4. 예제 payload

검증:

1. 예제 payload가 문서와 코드 타입에서 동일하게 해석된다.
2. 필수 필드와 nullable 필드가 명확하다.
3. 원문이 계약에 포함되지 않는다.

사용자 확인 필요:

1. cohort 분할 기준
2. category 집합
3. privacy-safe optional field 범위

### Phase 1. `shared` 도메인 계층 작성

목표:

- `agent`와 `main-server`가 동일한 도메인 객체를 참조하도록 만든다.

선행조건:

- Phase 0 문서 확정

구현:

1. `shared/src/domain/entities/window_summary.py`
2. `shared/src/domain/entities/norm_pack.py`
3. `shared/src/domain/entities/assessment_result.py`
4. `shared/src/contracts/*`
5. `shared/src/value_objects/*`

산출물:

1. 직렬화/역직렬화 가능한 도메인 객체
2. fixture 기반 contract test

검증:

1. `dict` 중심 처리 대신 명시적 객체를 사용한다.
2. 같은 fixture를 `agent`와 `main-server`에서 동일하게 읽는다.

사용자 확인 필요:

- 없음

### Phase 2. 로컬 분석 파이프라인 MVP

목표:

- 로컬에서 입력 이벤트를 받아 안정적으로 `WindowSummary`를 생성한다.

선행조건:

1. Phase 0
2. Phase 1

구현:

1. ingest
2. preprocess
3. optional translation
4. embedding adapter 연결
5. prototype scoring
6. window aggregation

산출물:

1. 로컬 fixture 입력
2. deterministic `WindowSummary`
3. service-level unit test

검증:

1. 같은 입력 fixture에 대해 같은 `WindowSummary`가 재현된다.
2. scoring과 windowing이 분리된 테스트로 검증된다.
3. 원문이 중앙 payload 타입으로 새지 않는다.

사용자 확인 필요:

1. 초기 embedding 모델 선택
2. prototype dataset 버전 선택

### Phase 3. 로컬 self-baseline MVP

목표:

- 로컬에서 개인 변화량을 계산할 수 있게 만든다.

선행조건:

1. Phase 2

구현:

1. warm-up period
2. rolling baseline 또는 moving stats
3. slope
4. persistence
5. spike filtering

산출물:

1. `BaselineProfile`
2. baseline 관련 unit test
3. fallback policy

검증:

1. 단발 이벤트와 지속 변화를 구분한다.
2. 과거 데이터 부족 상황의 fallback이 명확하다.

사용자 확인 필요:

1. warm-up 길이
2. persistence window
3. false positive 억제 수준

### Phase 4. 중앙 Normative Server MVP

목표:

- 중앙이 `WindowSummary`를 받아 cohort 기준 `NormPack'을 생성한다.

선행조건:

1. Phase 0
2. Phase 1
3. Phase 2

구현:

1. summary upload API
2. cohort grouping
3. robust aggregation
4. `NormPack` publication
5. version / TTL / cohort minimum size 정책

산출물:

1. `POST /api/v1/sync/window-summaries`
2. `GET /api/v1/norms/{cohort_key}`
3. `GET /api/v1/policies/current`
4. cohort fixture 기반 aggregation test

검증:

1. 여러 로컬 summary fixture를 넣으면 cohort별 `NormPack`이 생성된다.
2. 원문이나 개인 판정 결과가 중앙 payload에 포함되지 않는다.
3. cohort sample size가 부족하면 안전하게 fallback한다.

사용자 확인 필요:

1. cohort 세분화 수준
2. 최소 cohort size
3. robust aggregation 방식

### Phase 5. 로컬 최종 판단 MVP

목표:

- `Self-baseline + Peer norm + Persistence`를 결합한 로컬 최종 판단을 완성한다.

선행조건:

1. Phase 3
2. Phase 4

구현:

1. `NormPack` 다운로드 및 캐시
2. `DecisionPolicy`
3. support resource selection
4. explanation field 정의

산출물:

1. `AssessmentResult`
2. local decision service
3. policy test

검증:

1. 중앙 기준이 있을 때와 없을 때의 정책 차이가 명확하다.
2. 단순 threshold가 아니라 설명 가능한 판단 흐름이 존재한다.
3. `NormPack` 변화로 결과가 달라져도, 그것은 정책 입력 변화이지 학습된 모델 변경이 아니다.

사용자 확인 필요:

1. support resource 정책
2. 사용자에게 노출할 explanation 수준
3. threshold 또는 decision level 해석

### Phase 6. End-to-End 폐회로 검증

목표:

- `agent -> main-server -> agent` 한 바퀴를 실제로 닫는다.

선행조건:

1. Phase 2
2. Phase 3
3. Phase 4
4. Phase 5

구현:

1. summary upload
2. `NormPack` generation
3. `NormPack` fetch
4. local decision update
5. retry / backoff

산출물:

1. 루트 `tests/federation`
2. synthetic scenario set
3. e2e smoke path

검증:

1. `의미 점수 -> 윈도우 통계 -> 중앙 기준 -> 로컬 판단` 전체 사이클이 동작한다.
2. payload에 원문이 포함되지 않는다.
3. 장애가 나도 실패 위치를 분리해 설명할 수 있다.

사용자 확인 필요:

- 없음

### Phase 7. 입력 어댑터 연결

목표:

- 브라우저 입력 채널을 실제 로컬 파이프라인에 연결한다.

선행조건:

1. Phase 6

구현:

1. `agent/chrome-extension`
2. local bridge
3. payload sanitization
4. source metadata 최소화

산출물:

1. extension -> local bridge path
2. adapter integration test

검증:

1. 입력 채널이 바뀌어도 `agent/src/services` 핵심 분석 로직은 바뀌지 않는다.
2. 확장 프로그램이 privacy 경계를 깨지 않는다.

사용자 확인 필요:

1. 어떤 입력 채널을 우선 지원할지
2. extension이 수집 가능한 범위

### Phase 8. 모델/데이터셋 고도화

목표:

- 임베딩 모델, 번역 모델, prototype 데이터셋 선택을 구조를 흔들지 않고 개선한다.

선행조건:

1. Phase 2

구현:

1. embedding adapter 교체 실험
2. translation adapter 실험
3. prototype sentence curation
4. synthetic 또는 공개 데이터셋 기반 평가

산출물:

1. 모델 메타데이터 관리 문서
2. 비교 실험 결과

검증:

1. 모델 교체가 서비스 구조 변경 없이 가능하다.
2. 품질 비교 지표가 존재한다.

사용자 확인 필요:

1. 기본 모델 채택
2. 데이터셋 출처와 사용 범위

### Phase 9. Docker 기반 실행 표준화

목표:

- 구조를 바꾸지 않고 실행 환경만 표준화한다.

선행조건:

1. Phase 6

구현:

1. `main-server/Dockerfile`
2. `agent/Dockerfile`
3. 루트 `docker-compose.yml`

산출물:

1. 표준 실행 경로
2. 개발자 온보딩 절차

검증:

1. 개발자 환경 차이와 무관하게 동일한 방식으로 서비스를 띄울 수 있다.

사용자 확인 필요:

1. 배포 우선순위
2. 로컬/중앙 분리 실행 방식

### Phase 10. FL-ready Runtime

목표:

- 현재 analytics 구조 위에 decision-model FL orchestration 계층을 얹을 준비를 한다.

선행조건:

1. Phase 6
2. feedback 신호 설계 초안

구현:

1. `AgentRegistry`
2. `RoundManager`
3. `ModelRegistry`
4. `TrainingTask`
5. `TrainingUpdateEnvelope`
6. `DecisionFeedbackSignal`
7. `ClientCheckpointStore`

산출물:

1. `POST /api/v1/agents/register`
2. `POST /api/v1/agents/heartbeat`
3. `GET /api/v1/fl/rounds/current`
4. `POST /api/v1/fl/rounds/{roundId}/join`
5. `POST /api/v1/fl/rounds/{roundId}/updates`

검증:

1. analytics 경로를 깨지 않고 runtime API가 추가된다.
2. `NormPack` 경로와 FL update 경로가 명확히 분리된다.

사용자 확인 필요:

1. feedback/self-report를 실제로 도입할지
2. FL 라운드 운영 여부

### Phase 11. Optional Decision-Model FL MVP

목표:

- 실제 decision-model 파라미터 업데이트를 교환하는 FL 루프를 구현한다.

선행조건:

1. Phase 10
2. usable feedback 또는 self-report 신호

구현:

1. `WindowSummary`, baseline, `NormPack` 기반 feature를 입력으로 쓰는 작은 decision head 설계
2. local self-report, support action, delayed outcome 중 최소 하나를 label 또는 weak signal로 사용
3. local training loop
4. basic FedAvg 또는 weighted averaging
5. model versioning

산출물:

1. decision model manifest
2. FL simulation
3. update aggregation path

검증:

1. 중앙이 `TrainingTask`를 배포하고, 로컬이 업데이트를 올리고, 중앙이 새 모델 버전을 발행한다.
2. `NormPack`과 FL 모델 파라미터가 역할상 분리되어 동작한다.
3. feedback 신호가 없을 때 이 단계가 자동으로 열리지 않는다.

사용자 확인 필요:

1. feedback 신호 채택
2. decision model 구조
3. 학습 목표 정의

### Phase 12. FL 신뢰도/평판 고도화

목표:

- 반복 라운드에서 튀는 노드나 불량 업데이트를 덜 반영하는 신뢰도 계층을 추가한다.

선행조건:

1. Phase 11

구현:

1. update quality 평가 기준
2. 라운드별 reputation 또는 trust score
3. aggregation weight 동적 조정
4. 익명/가명 식별 전략 검토

산출물:

1. trust 정책 문서
2. aggregation 전략 비교

검증:

1. 단순 weighted averaging보다 안정적인 집계가 가능하다.
2. privacy 원칙과 노드 추적 필요성이 충돌할 때의 정책이 명시된다.

사용자 확인 필요:

1. reputation 허용 여부
2. 익명성 수준과 운영성 tradeoff

### Phase 13. Privacy Hardening

목표:

- analytics와 FL 경로 모두에 privacy 보강 계층을 붙인다.

선행조건:

1. Phase 6 또는 Phase 11

구현:

1. transport security
2. secure aggregation
3. client-level DP
4. 필요 시 HE 또는 MPC 검토

산출물:

1. privacy threat model
2. privacy regression test
3. secure aggregation integration plan

검증:

1. privacy 계층이 핵심 도메인 로직을 오염시키지 않는다.
2. analytics-only 경로도 독립적으로 유지된다.

사용자 확인 필요:

1. secure aggregation 도입 시점
2. DP budget 허용 수준
3. HE 또는 MPC 검토 필요성

---

## 7. 단계별 검증 전략

각 단계는 아래 기준을 만족해야 다음으로 넘어간다.

1. 입력과 출력 계약이 명확하다.
2. 반복 가능한 테스트 또는 시뮬레이션이 있다.
3. 실패 원인을 다음 단계와 분리해 설명할 수 있다.

권장 검증 순서:

1. schema/unit test
2. service-level test
3. adapter integration test
4. end-to-end federation simulation

---

## 8. 지금 바로 착수할 우선순위

가장 먼저 해야 할 일:

1. `docs/contracts/window_summary_v1.md` 작성
2. `docs/contracts/norm_pack_v1.md` 작성
3. 필요 시 `docs/contracts/assessment_result_v1.md` 작성
4. `shared/src/domain/entities/window_summary.py` 작성
5. `shared/src/domain/entities/norm_pack.py` 작성
6. fixture 기반 summary example 정의

그 다음:

1. `agent`의 `Embedding -> Scoring -> Windowing` 파이프라인 구현
2. `main-server`의 cohort 집계 및 `NormPack` 생성 구현
3. `DecisionService` 연결
4. `tests/federation` e2e 최소 폐회로 작성

feedback 신호가 없는 현재 기준에서 하지 말아야 할 것:

1. decision-model FL 라운드부터 먼저 구현
2. `NormPack`을 글로벌 모델 파라미터라고 가정하고 설계
3. semantic classifier 학습과 cohort rarity 판단을 한 단계에 섞는 것

---

## 9. 현재 계획에서 부족한 부분

아래 항목들은 아직 문서화 또는 설계가 더 필요하다.

1. `WindowSummary` 필드 정의
2. `NormPack` 필드 정의
3. `cohort_key` 설계
4. baseline 수학 정의
5. support suggestion 정책
6. 모델 평가 전략
7. `DecisionFeedbackSignal` 정의
8. `TrainingTask`와 `TrainingUpdateEnvelope`의 decision-model 기준 필드
9. privacy regression test 계획
10. FL 신뢰도 계층의 privacy tradeoff
11. monorepo 내부 패키지 관리 방식
12. 관측 가능성 설계

---

## 10. 권장 결론

지금 가장 먼저 닫아야 하는 것은 아래 한 바퀴다.

1. 로컬이 어떤 요약을 안정적으로 만들 것인가
2. 중앙이 그 요약으로 어떤 cohort 기준을 만들 것인가
3. 로컬이 그 기준을 받아 어떻게 최종 판단할 것인가

즉 우선순위는 아래가 맞다.

1. 계약
2. `shared`
3. 로컬 분석
4. baseline
5. 중앙 `NormPack`
6. 로컬 판단
7. end-to-end 검증
8. 입력 어댑터
9. Docker
10. FL-ready runtime
11. optional decision-model FL
12. trust/reputation 고도화
13. privacy hardening

이 순서가 가장 재작업이 적고, 각 단계의 실패 원인을 분리해 검증하기 쉽다.
