# TraceMind Staged Execution Roadmap

## 1. 목적

이 문서는 지금까지 논의한 내용을 바탕으로 TraceMind를 한 번에 크게 만들지 않고,
가장 낮은 단계부터 검증 가능한 단위로 닫아 가기 위한 실행 계획을 정리한다.

핵심 원칙은 아래와 같다.

이 프로젝트는 단기적인 임시 해결보다, 교체 가능하고 모듈화된 장기 확장형 구조를 우선한다.
한 번 돌아가는 구현보다 이후 모델 교체, 입력 채널 확장, FL, privacy 계층 추가에도 재작업이 적은 구조를 목표로 한다.

1. 최종 목표는 크게 유지하되, 구현은 가장 작은 폐회로부터 닫는다.
2. 계약과 데이터 경계를 먼저 고정하고, 구현체는 그 뒤에 붙인다.
3. 로컬 처리와 중앙 집계를 먼저 안정화한 뒤, FL과 privacy hardening을 올린다.
4. 입력 채널, Docker, 배포 구조는 도메인과 계약이 선 뒤에 붙인다.

---

## 2. 최종 목표

TraceMind의 최종 목표는 두 축을 모두 포함한다.

1. 로컬에서 원문을 처리하고 중앙에는 최소 통계만 보내는 privacy-preserving analytics 구조
2. 이후 parameter federated learning, secure aggregation, DP, 필요 시 HE까지 확장 가능한 구조

핵심 가치:

- 원문은 로컬에 남긴다.
- 중앙은 개인 판정을 하지 않는다.
- 중앙은 cohort 기준만 학습해 `NormPack`으로 배포한다.
- 최종 판단은 다시 로컬에서 수행한다.

---

## 3. 사용자 관점 데이터 흐름

1. 사용자가 브라우저나 앱에서 텍스트 입력 또는 이벤트를 만든다.
2. 입력은 `agent/chrome-extension` 같은 수집 어댑터를 통해 로컬 `agent`로 전달된다.
3. 로컬 `agent`가 전처리, 번역 선택 적용, 임베딩, prototype scoring을 수행한다.
4. 로컬 `agent`가 이벤트들을 시간 윈도우로 묶어 `WindowSummary`를 생성한다.
5. 로컬 `agent`가 과거 자기 패턴으로 `Self-baseline`을 계산한다.
6. 중앙 `main-server`에는 원문이 아니라 `WindowSummary` 같은 요약 통계만 전달된다.
7. 중앙 `main-server`가 cohort 기준으로 집계해 `NormPack`을 생성한다.
8. `NormPack`이 다시 로컬 `agent`로 배포된다.
9. 로컬 `agent`가 `Self-baseline + Peer norm + Persistence`를 결합해 최종 판단을 수행한다.
10. 최종 결과는 사용자 본인에게 지원 리소스 제안 형태로 제공된다.

중앙 서버가 하지 않는 것:

- 원문 저장
- 개인 위험 판정
- 개별 사용자 추적

---

## 4. 현재 합의된 구조

```text
tracemind_server/
  main-server/
    Dockerfile
    src/
    tests/

  agent/
    Dockerfile
    src/
    chrome-extension/
    tests/

  shared/
    src/
      domain/
      contracts/
      value_objects/
      policies/
      ports/
    tests/

  scripts/
  tests/
    federation/

  docker-compose.yml
  docs/
  plan.md
  README.md
```

구조 원칙:

1. `main-server`와 `agent`는 실행 단위다.
2. `shared`는 실행 서비스가 아니라 공통 도메인/계약 계층이다.
3. `Dockerfile`은 각 실행 단위 폴더 안에 둔다.
4. 여러 실행 단위를 함께 띄우는 조합은 루트 `docker-compose.yml`에 둔다.
5. 현재 단계에서는 별도 `infra/` 폴더를 두지 않는다.

---

## 5. 단계별 구현 순서

### Phase 0. 계약 우선 정리

목표:

- 로컬과 중앙이 주고받는 핵심 객체를 먼저 고정한다.

산출물:

1. `WindowSummary v1`
2. `NormPack v1`
3. `AssessmentResult v1`
4. 최소 sync contract 초안

완료 조건:

- 예제 payload가 문서와 코드 타입에서 동일하게 해석된다.
- 직렬화/역직렬화 테스트가 안정적으로 통과한다.
- 필수 필드와 nullable 필드가 명확하다.

이 단계에서 하지 않을 것:

- 실제 API 구현 확대
- FL runtime
- Chrome extension 연결

### Phase 1. `shared` 도메인 계층 작성

목표:

- `agent`와 `main-server`가 같은 도메인 타입을 참조하도록 만든다.

산출물:

1. `shared/src/domain/entities/window_summary.py`
2. `shared/src/domain/entities/norm_pack.py`
3. `shared/src/domain/entities/assessment_result.py`
4. `shared/src/contracts/*`
5. `shared/src/value_objects/*`

완료 조건:

- `dict` 중심 처리 대신 명시적 객체를 쓴다.
- 같은 fixture를 `agent`와 `main-server`에서 동일하게 읽는다.

### Phase 2. 로컬 분석 파이프라인 MVP

목표:

- 로컬에서 입력 이벤트를 받아 안정적으로 `WindowSummary`를 생성한다.

포함:

1. ingest
2. preprocess
3. optional translation
4. embedding
5. prototype scoring
6. window aggregation

완료 조건:

- 같은 입력 fixture에 대해 같은 `WindowSummary`가 재현된다.
- scoring과 windowing이 분리된 테스트로 검증된다.

이 단계에서 하지 않을 것:

- 중앙 집계
- 최종 사용자 판단 정책 고도화
- 실제 브라우저 확장 연결

### Phase 3. 로컬 self-baseline MVP

목표:

- 로컬에서 개인 변화량을 계산할 수 있게 만든다.

포함:

1. warm-up period
2. moving baseline 또는 rolling stats
3. slope
4. persistence
5. spike filtering

완료 조건:

- 단발 이벤트와 지속 변화가 구분된다.
- 과거 데이터가 부족할 때의 fallback 정책이 정의된다.

### Phase 4. 중앙 Normative Server MVP

목표:

- 중앙이 `WindowSummary`를 받아 cohort 기준 `NormPack`을 생성한다.

포함:

1. summary upload API
2. cohort grouping
3. robust aggregation
4. `NormPack` publication

완료 조건:

- 여러 로컬 summary fixture를 넣으면 cohort별 `NormPack`이 생성된다.
- 원문이나 개인 판정 결과가 중앙 payload에 포함되지 않는다.

중요:

- 연령대별 cohort는 이 단계부터 들어간다.
- `cohort_key`는 FL 이후가 아니라 analytics MVP에서 이미 핵심이다.

### Phase 5. 로컬 최종 판단 MVP

목표:

- `Self-baseline + Peer norm + Persistence`를 결합한 로컬 최종 판단을 완성한다.

포함:

1. `NormPack` 다운로드 및 캐시
2. decision policy
3. support resource selection

완료 조건:

- 중앙 기준이 있을 때와 없을 때의 정책 차이가 명확하다.
- 단순 threshold가 아니라 설명 가능한 로컬 판단 흐름이 존재한다.

### Phase 6. End-to-End 폐회로 검증

목표:

- `agent -> main-server -> agent` 한 바퀴를 실제로 닫는다.

포함:

1. summary upload
2. `NormPack` generation
3. `NormPack` fetch
4. local decision update

완료 조건:

- `의미 점수 -> 윈도우 통계 -> 중앙 기준 -> 로컬 판단` 전체 사이클이 동작한다.
- 루트 `tests/federation`에서 최소 e2e 시뮬레이션이 가능하다.

### Phase 7. 입력 어댑터 연결

목표:

- 브라우저 입력 채널을 실제 로컬 파이프라인에 연결한다.

포함:

1. `agent/chrome-extension`
2. local bridge
3. payload sanitization

완료 조건:

- 입력 채널이 바뀌어도 `agent/src/services` 핵심 분석 로직은 바뀌지 않는다.

중요:

- Chrome extension은 초기 우선순위가 아니다.
- 먼저 분석 파이프라인을 닫고 나서 붙인다.

### Phase 8. 모델/데이터셋 고도화

목표:

- 임베딩 모델, 번역 모델, prototype 데이터셋 선택을 구조를 흔들지 않고 개선한다.

포함:

1. embedding adapter 교체 실험
2. translation adapter 실험
3. prototype sentence curation
4. synthetic 또는 공개 데이터셋 기반 평가

완료 조건:

- 모델 교체가 서비스 구조 변경 없이 가능하다.
- 품질 비교 지표가 존재한다.

중요:

- 준비는 병렬로 가능하지만, 계약보다 먼저 파이프라인에 박아 넣지 않는다.

### Phase 9. Docker 기반 실행 표준화

목표:

- 구조를 바꾸지 않고 실행 환경만 표준화한다.

포함:

1. `main-server/Dockerfile`
2. `agent/Dockerfile`
3. 루트 `docker-compose.yml`

완료 조건:

- 개발자 환경 차이와 무관하게 동일한 방식으로 서비스를 띄울 수 있다.

중요:

- Docker는 구조를 만드는 도구가 아니라 이미 정한 구조를 재현하는 도구다.

### Phase 10. FL-ready Runtime

목표:

- 현재 analytics 구조 위에 FL orchestration 계층을 얹을 준비를 한다.

포함:

1. `AgentRegistry`
2. `RoundManager`
3. `ModelRegistry`
4. `TrainingTask`
5. `TrainingUpdateEnvelope`
6. `ClientCheckpointStore`

완료 조건:

- `agents/register`, `heartbeat`, `round join` 같은 runtime API가 analytics 경로를 깨지 않고 추가된다.

### Phase 11. Parameter Federated Learning MVP

목표:

- 실제 모델 파라미터 업데이트를 교환하는 FL 루프를 구현한다.

권장 범위:

1. 작은 projection head 또는 얕은 분류기부터 시작
2. basic FedAvg 또는 weighted averaging부터 시작
3. 라운드 기반 집계

완료 조건:

- 중앙이 `TrainingTask`를 배포하고, 로컬이 업데이트를 올리고, 중앙이 새 모델 버전을 발행한다.

### Phase 12. FL 신뢰도/평판 고도화

목표:

- 반복 라운드에서 튀는 노드나 불량 업데이트를 덜 반영하는 신뢰도 계층을 추가한다.

포함:

1. update quality 평가 기준
2. 라운드별 reputation 또는 trust score
3. aggregation weight 동적 조정
4. 익명/가명 식별 전략 검토

완료 조건:

- 단순 weighted averaging보다 안정적인 집계가 가능하다.
- privacy 원칙과 노드 추적 필요성이 충돌할 때의 정책이 명시된다.

중요:

- 이 단계는 초기 cohort 집계와 다르다.
- summary 집계용 robust aggregation보다 훨씬 뒤에 온다.

### Phase 13. Privacy Hardening

목표:

- FL과 동기화 경로에 privacy 보강 계층을 붙인다.

권장 순서:

1. transport security
2. secure aggregation
3. client-level DP
4. 필요 시 HE 또는 MPC 검토

완료 조건:

- privacy 계층이 핵심 도메인 로직을 오염시키지 않는다.

---

## 6. 단계별 검증 전략

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

## 7. 지금 바로 착수할 우선순위

가장 먼저 해야 할 일:

1. `docs/contracts/window_summary_v1.md` 작성
2. `docs/contracts/norm_pack_v1.md` 작성
3. 필요 시 `docs/contracts/sync_contracts_v1.md` 작성
4. `shared/src/domain/entities/window_summary.py` 작성
5. `shared/src/domain/entities/norm_pack.py` 작성
6. fixture 기반 summary example 정의

그 다음:

1. `agent`의 `Embedding -> Scoring -> Windowing` 파이프라인 구현
2. `main-server`의 cohort 집계 및 `NormPack` 생성 구현
3. `DecisionService` 연결

---

## 8. 현재 계획에서 부족한 부분

아래 항목들은 아직 문서화 또는 설계가 더 필요하다.

1. `WindowSummary` 필드 정의가 아직 구체적이지 않다.
   - 카테고리별 score summary
   - time window semantics
   - locale/source metadata
   - privacy-safe optional fields

2. `NormPack` 필드 정의가 아직 충분히 세밀하지 않다.
   - 평균과 분산 외에 어떤 robust stats를 보낼지
   - 버전 정책과 TTL
   - 최소 cohort size 정책

3. `cohort_key` 설계가 더 필요하다.
   - 연령대 버킷 정의
   - 성별/지역/언어 같은 추가 분할 여부
   - 과도한 cohort 세분화에 따른 privacy risk

4. baseline 정책이 아직 수학적으로 덜 고정되어 있다.
   - warm-up 길이
   - slope 계산 방식
   - persistence window
   - spike filtering 기준

5. support suggestion 정책이 아직 약하다.
   - 어떤 수준에서 어떤 리소스를 제안할지
   - 사용자에게 어떤 설명 문구를 줄지

6. 모델 평가 전략이 아직 부족하다.
   - 임베딩 품질 지표
   - prototype dataset 검증 방법
   - synthetic data와 실제 데이터의 경계

7. privacy regression test 계획이 더 필요하다.
   - 서버 payload에 원문이 없는지 자동 검증
   - 로그에 민감 데이터가 남지 않는지 검증

8. FL 신뢰도 계층의 privacy tradeoff가 아직 열려 있다.
   - reputation을 유지하려면 식별성이 일부 필요할 수 있다.
   - 이 부분은 가명 토큰, 회차 제한, TTL 정책과 함께 설계해야 한다.

9. Docker 이후 저장소/의존성 전략이 아직 없다.
   - `shared`를 각 서비스에 어떤 방식으로 설치할지
   - monorepo 내부 패키지 관리 방식

10. 관측 가능성 설계가 아직 없다.
   - audit log 범위
   - privacy-safe metrics
   - debugging용 trace boundary

---

## 9. 내가 보는 권장 결론

지금 가장 중요한 것은 "입력 채널"도 아니고 "FL orchestration"도 아니다.
가장 먼저 닫아야 하는 것은 아래 한 바퀴다.

1. 로컬이 어떤 요약을 안정적으로 만들 것인가
2. 중앙이 그 요약으로 어떤 cohort 기준을 만들 것인가
3. 로컬이 그 기준을 받아 어떻게 최종 판단할 것인가

즉 우선순위는 다음과 같다.

1. 계약
2. `shared`
3. 로컬 분석
4. baseline
5. 중앙 `NormPack`
6. 로컬 판단
7. end-to-end 검증
8. 입력 어댑터
9. Docker
10. FL runtime
11. parameter FL
12. trust/reputation 고도화
13. privacy hardening

이 순서가 가장 재작업이 적고, 각 단계의 실패 원인을 분리해 검증하기 쉽다.
