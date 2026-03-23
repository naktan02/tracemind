# WindowSummary v1

## 1. 목적

`WindowSummary`는 로컬 `agent`가 원문 이벤트를 직접 중앙으로 보내지 않고,
로컬에서 임베딩과 카테고리 점수 계산을 마친 뒤 생성하는 privacy-safe 요약 객체다.

중앙 `main-server`는 여러 `WindowSummary`를 cohort 기준으로 집계해 `NormPack`을 만든다.

즉 `WindowSummary`는:

1. 원문이 아니다.
2. 임베딩 벡터 원본이 아니다.
3. 개인 최종 판단 결과가 아니다.
4. 로컬 micro-batch를 대표하는 최소 요약 통계다.

---

## 2. v1 설계 방향

`WindowSummary v1`은 최종 시계열 윈도우 구조로 가기 전,
초기 MVP에서 빠르게 폐회로를 닫기 위한 micro-batch summary 계약이다.

v1 원칙:

1. 로컬에서 생성한다.
2. 중앙은 이를 받아 cohort 기준만 계산한다.
3. `self-baseline`이나 개인 변화 해석은 포함하지 않는다.
4. 카테고리별 요약 통계만 담고, 원문과 개별 임베딩은 담지 않는다.
5. 향후 rolling window나 시계열 확장으로 자연스럽게 넘어갈 수 있어야 한다.

---

## 3. 생성 시점

`WindowSummary`는 개별 이벤트 1개마다 생성하지 않는다.
로컬 `agent`는 이벤트들을 짧은 micro-batch로 묶은 뒤 flush 시점에 하나의 `WindowSummary`를 생성한다.

v1 flush 규칙:

1. 이벤트 개수 기준과 시간 기준을 함께 사용한다.
2. 둘 중 먼저 도달한 조건에서 flush한다.
3. 구체 임계값은 구현 설정으로 둔다.

즉 계약은 flush 방식만 고정하고,
구체 숫자는 구현 단계에서 설정 가능하게 둔다.

---

## 4. 역할 경계

### 로컬 `agent`가 하는 일

1. 입력 수집
2. 전처리
3. 필요 시 번역
4. 임베딩
5. prototype scoring
6. micro-batch 집계
7. `WindowSummary` 생성

### 중앙 `main-server`가 하는 일

1. `WindowSummary` 수신
2. `age_band` 기준 cohort grouping
3. robust aggregation
4. `NormPack` 생성 및 배포

중앙은 `WindowSummary`를 해석해 개인 상태를 판정하지 않는다.

---

## 5. 포함할 필드

v1은 아래 필드를 기본으로 한다.

### 필수 필드

1. `schema_version`
   - 스키마 버전
   - 예: `window_summary.v1`

2. `summary_id`
   - 요약 객체 식별자
   - 로컬에서 생성하는 UUID 계열 식별자

3. `age_band`
   - cohort 계산용 연령대 버킷
   - v1에서는 cohort 축을 연령대로만 제한한다

4. `batch_started_at`
   - micro-batch 시작 시각

5. `batch_ended_at`
   - micro-batch 종료 시각

6. `event_count`
   - batch 안에 포함된 이벤트 수

7. `category_stats`
   - 카테고리별 요약 통계 집합
   - v1 카테고리: `anxiety`, `depression`, `suicidal`, `normal`

### `category_stats` 내부 필드

각 카테고리마다 아래 값을 가진다.

1. `mean`
   - batch 내 해당 카테고리 점수 평균

2. `max`
   - batch 내 해당 카테고리 최대 점수

3. `count`
   - 해당 카테고리 점수 계산에 반영된 이벤트 수

---

## 6. 포함하지 않을 필드

v1에서는 아래를 포함하지 않는다.

1. 원문 텍스트
2. 개별 이벤트 원문 목록
3. 개별 이벤트 임베딩 벡터
4. 개별 이벤트별 카테고리 점수 배열
5. 개인 최종 판단 결과
6. `self-baseline` 기반 변화량
7. agent 식별자
8. `locale`
9. `source_type`

이유:

1. privacy 경계를 단순하게 유지하기 위해
2. 중앙이 개인 해석을 하지 않도록 하기 위해
3. v1 계약을 최소 크기로 유지하기 위해
4. 식별 책임을 요약 객체가 아니라 전송 계층에 두기 위해

---

## 7. 왜 변화량을 보내지 않는가

v1에서 중앙으로 보내는 것은 개인 변화량이 아니라 요약 통계다.

이유:

1. 변화량은 개인 과거 이력에 대한 해석이 이미 포함된 값이다.
2. 그 값을 중앙으로 보내면 로컬 판단 책임이 중앙 계약에 섞인다.
3. 중앙은 집단 기준 생성만 담당해야 한다.

따라서:

- 로컬은 `WindowSummary`를 보낸다.
- 중앙은 여러 summary를 집계해 `NormPack`을 만든다.
- 로컬은 `NormPack`과 자기 `self-baseline`을 결합해 최종 판단한다.

식별자가 필요한 경우에는 `WindowSummary` 본문이 아니라,
추후 정의할 `sync/upload envelope`에서 `agent_id` 같은 전송 메타데이터로 관리한다.

---

## 8. JSON 예시

```json
{
  "schema_version": "window_summary.v1",
  "summary_id": "5a34b4e0-6e84-4df1-b1f2-4f31a84e2f90",
  "age_band": "13_15",
  "batch_started_at": "2026-03-23T12:00:00Z",
  "batch_ended_at": "2026-03-23T12:04:30Z",
  "event_count": 8,
  "category_stats": {
    "anxiety": {
      "mean": 0.41,
      "max": 0.77,
      "count": 8
    },
    "depression": {
      "mean": 0.16,
      "max": 0.33,
      "count": 8
    },
    "suicidal": {
      "mean": 0.02,
      "max": 0.08,
      "count": 8
    },
    "normal": {
      "mean": 0.63,
      "max": 0.88,
      "count": 8
    }
  }
}
```

---

## 9. 검증 기준

`WindowSummary v1` 계약이 유효하려면 아래를 만족해야 한다.

1. 같은 입력 fixture에 대해 안정적으로 재현된다.
2. 중앙 업로드 payload에 원문이 포함되지 않는다.
3. 개별 임베딩 벡터가 포함되지 않는다.
4. `age_band` 기준 cohort 집계에 바로 사용 가능하다.
5. 이후 `NormPack v1` 계약으로 자연스럽게 연결된다.

---

## 10. 향후 v2 확장 후보

아래는 v1에 넣지 않고, 필요 시 v2에서 검토한다.

1. `locale`
2. `source_type`
3. 분산 또는 percentile 계열 통계
4. quality/confidence metadata
5. rolling window identifier
6. privacy-safe provenance metadata

별도 계약에서 다룰 후보:

1. `sync/upload envelope`
2. `agent_id` 또는 pseudonymous transport identifier
3. 업로드 시각과 전송 메타데이터

v1의 목적은 미래 확장을 막지 않으면서도,
가장 작은 계약으로 로컬-중앙 폐회로를 먼저 닫는 것이다.
