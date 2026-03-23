## 2️⃣ 왜 이 주제를 하려고 하나? (문제 정의)

현재 아동·청소년의 온라인 위험 감지 연구에는 다음과 같은 문제가 있습니다.

1. **중앙 서버 기반 감시**
    
    - 메시지, 대화, 콘텐츠 원문을 서버로 수집
        
    - 프라이버시 침해 및 윤리적 문제
        
2. **절대적 위험 판정**
    
    - 특정 키워드(자살 등)를 사용하면 곧바로 위험으로 간주
        
    - 오탐, 낙인 문제 발생

---

### ②  임베딩

- 쿼리를  모델로 임베딩
    
- 벡터 공간에서 의미 표현 획득
    

---

### ③ 카테고리 점수 계산 (B안 방식)

카테고리 예:

	**Anxiety**
	**Depression**
	**Suicidal**
	**Normal**
    

각 카테고리별 프로토타입 문장 세트와  
코사인 유사도 계산:

score_distress = max cosine(query, proto_distress_j)

→ 문장 단위 카테고리 점수 생성

---

### ④ 시간 윈도우 집계 (예: 7일)

윈도우 요약 벡터 xₜ 생성:

x_t = {  
}

---

### ⑤ 개인 기준선(Self-baseline) 계산

- 과거 4~6주 평균과 비교
    
- 추세(slope) + 지속성 판단
    
- 단발 이벤트 필터링
    
- 워밍업 기간 설정
    

---

### ⑥ 로컬 판단

조건:

- 개인 변화 발생
    
- 또래 기준 대비 드묾
    
- 일정 기간 지속
    

→ 아이에게만 지원 리소스 제안

📌 부모 통보, 서버 통보 없음

---

# 🔹 3. 서버에서 하는 일

### 서버는 절대 하지 않는 것

- 개인 원문 분석 ❌
    
- 개인 위험 판정 ❌
    
- 개별 사용자 추적 ❌
    

---

### 서버가 받는 데이터

- 원문 ❌
    
- 개별 점수 ❌
    
- 판단 결과 ❌
    

✔ 오직:

윈도우 요약 통계 (x_t)  
또는 sum / sumsq / count

---

### 서버에서 수행하는 작업

1. 같은 연령 코호트 집계
    
2. 평균(μ), 분산(σ) 계산
    
3. robust aggregation 적용
    
4. 신뢰도 기반 가중치 업데이트
    

---

### 서버가 내려주는 것

NormPack {  
	**Anxiety**
	**Depression**
	**Suicidal**
	**Normal**
}

→ 개인 기준이 아니라 **집단 기준**

---

# 5️⃣ 핵심 아이디어: 두 가지 기준의 분리

## ① 개인 기준선 (Self-baseline)

- 이 아이의 과거 대비 변화 감지
    
- 상승 추세 + 지속성
    

## ② 또래 기준 (Peer norm)

- 같은 연령 집단의 정상 분포
    
- z-score 기반 드묾 판단
    

📌 두 조건이 동시에 만족될 때만 개입

---

# 6️⃣ 기술적 특징

## 🔹 NLP 구조

- BGE-M3-ko = 의미 임베딩 엔진
    
- 카테고리 점수 = 프로토타입 유사도 기반
    
- 필요 시 약한 멀티라벨 학습(A안) 적용
    

---

## 🔹 서버 모델

- 딥러닝 ❌
    
- 신뢰도 기반 집계 알고리즘
    
- robust aggregation
    
- normative modeling
    

---

## 🔹 프라이버시 구조

- 원문 로컬 처리
    
- 서버에는 통계만 전송
    
- 개인 outlier score 전송 없음
    
- 메신저 데이터 미수집
    

---

# 7️⃣ 이 연구의 차별성 / 기여

1. 감시 없는 조기 지원 구조
    
2. 위험 탐지가 아닌 변화 감지로 재정의
    
3. 로컬 판단 + 중앙 기준 학습 분리
    
4. 한국어 검색 기반 모델 설계
    
5. 아동·청소년 환경에 적합한 윤리적 설계
    

---

# 🔷  마무리 문장

> 본 연구는 “위험한 아이를 찾는 시스템”이 아니라,  
> 개인이 남긴 검색 신호의 의미적 분포가  
> 시간에 따라 어떻게 달라지는지를 분석하여  
> 프라이버시를 보존한 상태에서 조기 지원을 가능하게 하는  
> 분산 기준 학습 시스템입니다.

---

# 8️⃣ 구현 로드맵: MVP부터 확장형 구조까지

## 8-1. 어디부터 시작할 것인가

가장 먼저 해야 할 일은 모델을 더 붙이는 것이 아니라,  
로컬 서버와 중앙 서버가 어떤 데이터를 주고받는지 **계약(contract)** 을 고정하는 것이다.

이 프로젝트의 기본 경로는 일반적인 FedAvg 기반 연합학습이 아니라,
로컬에서 의미 점수와 시간 요약을 만들고 중앙에서는 집단 기준만 학습하는 구조다.

다만 최종 목표는 여기서 멈추지 않는다.
로컬 feedback 또는 self-report 신호를 확보할 수 있다면,
이후에는 로컬 최종 판단 계층에 대해 multi-agent privacy-preserving FL을 얹을 수 있어야 한다.

중요한 구분은 아래와 같다.

- `WindowSummary`, `NormPack`은 cohort parameter learning 산출물이다.
- decision-model parameter는 FL 산출물이다.

즉, 구현 순서는 아래가 맞다.

1. `WindowSummary(x_t)` 구조 정의
2. `NormPack` 구조 정의
3. 로컬 서버가 `x_t`를 안정적으로 생성하도록 구현
4. 중앙 서버가 `x_t`를 집계해 `NormPack`을 생성하도록 구현
5. 마지막에 로컬 판단 엔진이 `Self-baseline + Peer norm`을 결합
6. feedback 또는 self-report 신호가 확보되면 decision-model FL을 확장 계층으로 추가

이 순서로 가야 구조가 단단해지고, 이후 AI 기능을 추가해도 중심 설계가 흔들리지 않는다.

---

## 8-2. 설계 원칙

### 반드시 지킬 원칙

이 프로젝트는 단기적인 임시 해결보다, 교체 가능하고 모듈화된 장기 확장형 구조를 우선한다.
한 번 돌아가는 코드보다 이후 모델 교체, 입력 채널 확장, FL, privacy 계층 추가에도 재작업이 적은 구조를 목표로 한다.

1. **도메인 우선 설계**
   - 서버 프레임워크보다 먼저 `WindowSummary`, `NormPack`, `AssessmentResult` 같은 핵심 객체를 정의한다.

2. **포트-어댑터(헥사고날) 구조**
   - 임베딩 모델, 번역 모델, DB, API 전송 로직은 모두 교체 가능한 어댑터로 둔다.

3. **조합 가능한 작은 서비스**
   - 하나의 거대한 `AIService`를 만들지 말고,
     `EmbeddingService`, `ScoringService`, `WindowingService`, `BaselineService`, `DecisionService`
     같은 식으로 역할을 쪼갠다.

4. **판단과 모델을 분리**
   - 임베딩/번역은 표현 학습 계층이다.
   - 실제 판단은 정책 계층(`DecisionPolicy`)에서 한다.
   - 나중에 LLM이나 추천 AI를 붙여도 판단 로직이 오염되지 않도록 한다.
   - `NormPack` 같은 cohort 통계 파라미터와 FL 모델 파라미터를 같은 것으로 취급하지 않는다.

5. **스키마 버전 관리**
   - `WindowSummary v1`, `NormPack v1`처럼 명시적 버전을 둔다.
   - 미래 확장 시 필드 추가가 쉬워진다.

6. **개인 추적 최소화**
   - 서버는 개인 식별자를 다루지 않는다.
   - 신뢰도 가중치는 MVP에서는 보류하거나, 이후 프라이버시 검토를 거친 익명/가명 토큰 방식으로 분리 설계한다.

---

## 8-3. 추천 아키텍처

### 전체 흐름

`Raw Query/Event`
-> `Translation(optional)`
-> `Embedding`
-> `Prototype Scoring`
-> `Window Summary Builder`
-> `Self Baseline Analyzer`
-> `Federated Sync Client`
-> `Central Aggregator`
-> `NormPack`
-> `Local Decision Engine`
-> `Support Suggestion`

### 서버 역할 분리

#### 로컬 서버

- 원문 입력 수집
- 전처리/번역/임베딩
- 카테고리 점수 계산
- 시간 윈도우 통계 생성
- 개인 기준선 계산
- 중앙 서버와 동기화
- 최종 지원 리소스 제안

#### 중앙 서버

- cohort별 통계 집계
- robust aggregation
- 집단 기준(`NormPack`) 생성
- 버전 관리 및 배포
- 품질 모니터링
- 필요 시 decision-model FL coordinator 역할 추가

중앙 서버는 절대로 원문 해석이나 개인 판정 로직을 가지지 않는다.

---

## 8-4. 추천 폴더 구조

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
      publication_service.py
      agent_registry_service.py
      round_manager_service.py
      update_aggregation_service.py
      privacy_orchestration_service.py
    infrastructure/
      persistence/
        postgres/
      repositories/
      transport/
        http/
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
        http/
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
        window_range.py
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
      prototype_repository_port.py
      event_repository_port.py
      summary_repository_port.py
      norm_repository_port.py
      sync_client_port.py
      training_client_port.py
      model_registry_port.py
  tests/
    unit/

scripts/
  seed_prototypes.py
  run_local_demo.py
  run_federated_simulation.py

tests/
  federation/
    e2e/

docker-compose.yml
```

### 구조 해설

- `main-server`는 중앙 서버 실행 단위다. cohort 집계, `NormPack`, 정책 배포, FL orchestration 책임을 가진다.
- `agent`는 로컬 에이전트 실행 단위다. ingest, 임베딩, scoring, windowing, baseline, local decision, local training 책임을 가진다.
- `agent/chrome-extension`은 브라우저 입력 수집 어댑터다. 텍스트를 가져와 로컬 에이전트로 전달하지만, 핵심 분석 로직은 두지 않는다.
- `shared`는 공통 계약과 도메인 규칙만 담는 루트 계층이다. 양쪽이 모두 써야 하는 객체만 두며, 독립 서비스로 띄우지 않는다.
- 각 앱은 자기 `src/`, 자기 `tests/`를 가진다. 나중에 별도 레포로 분리할 때 폴더를 거의 그대로 옮기면 된다.
- `main-server/Dockerfile`과 `agent/Dockerfile`은 각 실행 단위의 컨테이너 정의만 담당한다. 서비스 경계를 Dockerfile 경계와 맞춘다.
- 루트 `docker-compose.yml`은 여러 실행 단위를 함께 띄우는 조합 정의다. 단일 서비스의 내부 구조를 대신하지 않는다.
- 현재 단계에서는 별도 `infra/` 폴더를 두지 않는다. 운영 자산이 커지기 전까지는 Docker 정의를 각 실행 단위 가까이에 두는 편이 더 단순하고 확장에도 무리가 없다.
- 루트 `tests/federation`은 `agent + main-server + shared`가 함께 도는 end-to-end 검증만 둔다.

이 구조를 택하는 이유는 "한 레포로 빠르게 개발"하면서도 "앱 경계는 처음부터 분명하게 유지"하기 위해서다.
즉 모노레포이지만 단일 앱 트리는 피하고, `main-server / agent / shared` 3축으로 분리하는 방식이다.

---

## 8-5. 핵심 클래스와 책임

### 도메인 엔티티

- `QueryEvent`
  - 로컬에서 발생한 원본 이벤트
  - timestamp, locale, source_type, normalized_text 등을 가짐

- `ScoredEvent`
  - 한 이벤트의 카테고리 점수 결과
  - `category_scores`, `embedding_meta`, `quality_flags`

- `WindowSummary`
  - 특정 기간의 요약 통계
  - 서버로 보내는 최소 단위
  - `sum`, `sumsq`, `count`, `max`, `slope_hint`, `coverage` 등을 포함

- `BaselineProfile`
  - 개인의 과거 4~6주 기준선
  - 이동평균, 분산, 지속성, 워밍업 상태 포함

- `NormPack`
  - cohort별 집단 기준
  - `mu`, `sigma`, `median`, `mad`, `min_support`, `pack_version`

- `AssessmentResult`
  - 최종 로컬 판단 결과
  - `self_shift`, `peer_deviation`, `persistence`, `decision`, `explanation`

### 포트 인터페이스

- `EmbeddingPort`
- `TranslationPort`
- `PrototypeRepositoryPort`
- `SummaryRepositoryPort`
- `NormRepositoryPort`
- `SyncClientPort`

이 포트들은 `Protocol` 또는 `ABC`로 정의하고,
실제 구현은 infrastructure 계층에서 연결한다.

### 서비스 클래스

- `EmbeddingService`
  - 텍스트를 임베딩으로 변환

- `PrototypeScoringService`
  - 임베딩과 프로토타입 유사도를 기반으로 카테고리 점수 계산

- `WindowingService`
  - 이벤트들을 7일 윈도우로 요약해 `WindowSummary` 생성

- `BaselineService`
  - 개인 기준선 계산

- `AggregationService`
  - 중앙 서버에서 cohort별 집계 수행

- `RobustAggregationService`
  - 이상치에 강한 평균/분산 계산

- `NormPackService`
  - 집계 결과를 배포 가능한 `NormPack`으로 변환

- `DecisionService`
  - `Self-baseline + Peer norm + Persistence`를 결합해 최종 판단

### 중요한 설계 기준

- 클래스 상속보다 조합(composition)을 우선한다.
- 서비스는 가능한 한 stateless하게 유지한다.
- 설정은 생성자 주입으로 넣고, 전역 상태를 피한다.

---

## 8-6. 가장 먼저 고정해야 할 데이터 계약

MVP 전에 아래 세 가지는 먼저 문서로 확정하는 것이 좋다.

### 1. `WindowSummary`

권장 필드:

- `schema_version`
- `cohort_key`
- `window_start`
- `window_end`
- `category_stats`
- `total_events`
- `coverage_ratio`
- `quality_flags`
- `client_manifest`

`category_stats` 내부 예시:

- `sum`
- `sumsq`
- `count`
- `max`
- `mean`

### 2. `NormPack`

권장 필드:

- `schema_version`
- `cohort_key`
- `generated_at`
- `pack_version`
- `category_norms`
- `support_size`
- `aggregation_method`

`category_norms` 내부 예시:

- `mu`
- `sigma`
- `median`
- `mad`
- `lower_bound`
- `upper_bound`

### 3. `AssessmentResult`

권장 필드:

- `window_id`
- `self_shift_score`
- `peer_z_score`
- `persistence_score`
- `decision_level`
- `reasons`
- `resource_bundle_id`

이 셋이 안정되면 나머지 구현은 갈아끼워도 된다.

---

## 8-7. MVP 범위

### MVP에서 반드시 포함할 것

1. 로컬에서 텍스트를 임베딩하고 카테고리 점수를 계산할 수 있어야 함
2. 7일 윈도우 기준 `WindowSummary`를 만들 수 있어야 함
3. 중앙 서버가 cohort별 `NormPack`을 계산할 수 있어야 함
4. 로컬 서버가 `NormPack`을 받아 최종 판단할 수 있어야 함
5. 원문이 서버로 전송되지 않는다는 것을 코드 레벨에서 보장해야 함

### MVP에서 일부러 빼야 할 것

1. 딥러닝 가중치 연합학습
2. 실시간 알림 시스템
3. 부모/학교 통보 플로우
4. 복잡한 신뢰도 가중치 시스템
5. LLM 기반 설명 생성
6. feedback 신호 없이 수행하는 decision-model FL

MVP는 "의미 점수 -> 윈도우 통계 -> 중앙 기준 -> 로컬 판단"의 완전한 한 바퀴를 도는 것이 목표다.

---

## 8-8. 단계별 개발 계획

## Phase 0. 명세 고정

목표:

- `WindowSummary`, `NormPack`, `AssessmentResult` 스키마 문서화
- cohort 기준 정의
- 카테고리 정의 및 프로토타입 버전 관리 방식 확정

산출물:

- `docs/architecture.md`
- `docs/contracts/window_summary_v1.md`
- `docs/contracts/norm_pack_v1.md`

이 단계가 끝나야 팀이 같은 말을 하게 된다.

## Phase 1. 로컬 서버 도메인 MVP

목표:

- 로컬에서 이벤트를 입력받고 요약 벡터 `x_t`를 만들 수 있어야 함

구현:

1. 이벤트 저장소(SQLite)
2. 전처리/번역/임베딩 파이프라인
3. 프로토타입 기반 카테고리 점수 계산
4. `WindowingService`
5. `BaselineService`의 단순 버전
6. 입력 채널은 우선 파일/테스트 fixture/API ingest로 시작하고, 크롬 확장 프로그램은 후속 어댑터로 붙인다

완료 기준:

- 샘플 데이터셋으로 `WindowSummary`가 재현 가능하게 생성됨
- 같은 입력에 대해 항상 같은 요약 결과가 나옴

## Phase 2. 중앙 서버 통계 MVP

목표:

- 업로드된 `WindowSummary`를 cohort별로 집계해 `NormPack` 생성

구현:

1. `POST /sync/windows`
2. `GET /norms/{cohort_key}`
3. 평균/분산 계산
4. simple clipping 또는 median/MAD 기반 robust aggregation
5. `NormPack` 버전 저장

완료 기준:

- 중앙 서버가 cohort별 집단 기준을 안정적으로 반환함

## Phase 3. 연동 MVP

목표:

- 로컬과 중앙이 실제로 한 사이클을 교환

구현:

1. `SyncService`
2. 업로드 배치 생성
3. NormPack 캐시
4. 동기화 실패 시 재시도/백오프

완료 기준:

- 로컬 샘플 10~100개를 시뮬레이션해 중앙 기준이 생성되고 다시 로컬 판단에 반영됨

## Phase 4. 판단 엔진 고도화

목표:

- 연구 아이디어의 핵심인 "개인 변화 + 또래 대비 드묾 + 지속성"을 실제 정책으로 구현

구현:

1. `DecisionPolicy`
2. slope / persistence 계산
3. warm-up period 처리
4. false positive 억제를 위한 minimum evidence 규칙

완료 기준:

- 단발 이벤트에는 반응하지 않고, 지속적 변화에만 판단이 발생함

중요:

- 이 단계의 판단 엔진은 우선 `DecisionPolicy` 기반의 rule-driven 구조로 둔다.
- `NormPack` 변화로 결과가 달라지는 것은 입력 기준 변화이지 FL 학습이 아니다.

## Phase 5. 품질/윤리/운영 안정화

목표:

- 연구용 프로토타입에서 운영 가능한 백엔드로 가기 위한 기반 마련

구현:

1. 감사 로그(원문 제외)
2. 스키마 마이그레이션 전략
3. 데이터 보존 정책
4. cohort 최소 샘플 수 제한
5. privacy regression test

## Phase 6. 확장형 AI 기능 추가

가능한 확장:

1. LLM 기반 지원 문구 추천
2. weak supervision 기반 카테고리 보정
3. 적응형 cohort 분할
4. 로컬 개인화 추천
5. feedback 신호 확보 이후의 decision-model FL

중요:

이 단계의 AI 기능은 반드시 `AssessmentResult` 이후에 붙여야 한다.
탐지 엔진 자체에 바로 결합하면 전체 구조가 금방 무너진다.

---

## 8-9. 저장소/인프라 선택 권장안

### 로컬 서버

- DB: SQLite
- 이유: 배포 단순성, 오프라인 대응, 디버깅 용이

### 중앙 서버

- DB: PostgreSQL
- 이유: cohort 집계, 버전 관리, 운영 안정성

### 캐시

- 초기에 없음 또는 파일 캐시
- 필요 시 Redis 추가

### 메시지 큐

- MVP에서는 도입하지 않음
- 배치/스케줄러로 충분

### 임베딩 저장

- 원문/벡터 전체를 영구 저장하지 말고,
  가능하면 이벤트 처리 후 필요한 요약 통계만 보존하는 방향을 우선 고려한다.

---

## 8-10. 테스트 전략

### 단위 테스트

- `WindowingService`
- `BaselineService`
- `RobustAggregationService`
- `DecisionPolicy`

### 통합 테스트

- 로컬 ingest -> summary 생성
- summary upload -> norm pack 생성
- norm pack download -> local decision 반영

### 시뮬레이션 테스트

- 연령 cohort별 synthetic data 생성
- 이상치/희소 데이터/불균형 데이터 시나리오 검증

### 프라이버시 테스트

- 서버 요청 payload에 원문이 없는지 검증
- 개인 판단 결과가 서버로 전달되지 않는지 검증

---

## 8-11. 가장 현실적인 시작 순서

실제로는 아래 순서로 진행하는 것이 가장 안전하다.

1. `WindowSummary`와 `NormPack` 스키마 문서부터 작성
2. 임베딩 모델/번역 모델/데이터셋을 프로젝트 리소스로 정리하되 계약에 종속되지 않게 보관
3. 로컬 서버의 `Embedding -> Scoring -> Windowing` 파이프라인 구현
4. SQLite 기반 로컬 저장소 구현
5. 중앙 서버의 cohort 집계 API 구현
6. `NormPack` 다운로드/캐시 구현
7. `DecisionService`로 최종 로컬 판단 연결
8. synthetic federation simulation 작성
9. feedback/self-report 수집 전략이 정해지면 `DecisionTask`, `TrainingUpdateEnvelope`, `DecisionFeedbackSignal` 초안 작성
10. 마지막에 크롬 확장 프로그램, robust aggregation 고도화, 신뢰도 가중치, AI 추천 기능 추가

핵심은 "중앙 서버를 먼저 크게 만드는 것"이 아니라
"로컬 서버가 어떤 요약을 안정적으로 만들 것인지 먼저 고정하는 것"이다.

또한 "모델과 데이터셋을 먼저 파이프라인에 박아 넣는 것"보다
"계약과 도메인 구조를 먼저 고정하고, 모델/데이터셋은 병렬로 준비한 뒤 연결하는 것"이 더 안전하다.

---

## 8-12. 지금 바로 다음 작업으로 추천하는 것

지금 당장 시작한다면 다음 세 가지를 먼저 만드는 것이 좋다.

1. `docs/contracts/window_summary_v1.md`
2. `shared/src/domain/entities/window_summary.py`
3. `agent/src/services/windowing_service.py`

그 다음 순서는 아래가 자연스럽다.

1. `shared/src/domain/entities/norm_pack.py`
2. `shared/src/contracts/sync_contracts.py`
3. 모델/데이터셋 로딩 경로와 메타데이터 정의
4. 중앙 서버 집계 API
5. 로컬 `agent/src/services/decision_service.py`

이 순서면 MVP를 빠르게 닫을 수 있으면서도,
나중에 임베딩 모델 교체, 다국어 추가, LLM 추천, 모바일 클라이언트 확장까지 모두 버틸 수 있다.
