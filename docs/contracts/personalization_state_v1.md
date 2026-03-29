# PersonalizationState v1

## 1. 목적

`PersonalizationState`는 로컬 `agent`가 사용자별 해석 기준을 유지하기 위한 상태 객체다.

이 객체는 아래를 담는다.

1. 개인 기준선
2. 개인 threshold
3. 개인 prototype
4. warm-up 및 calibration 상태

중요:

`PersonalizationState`는 기본적으로 로컬 전용 상태다.
서버 업로드를 전제로 설계하지 않는다.

---

## 2. v1 설계 방향

v1 원칙:

1. 로컬 inference와 local training이 동일한 상태 객체를 참조하게 한다.
2. 공통 score와 개인 해석을 분리한다.
3. 직렬화 가능해야 하지만, 기본 사용처는 로컬 persistence다.

---

## 3. 포함할 필드

### 필수 필드

1. `schema_version`
   - 예: `personalization_state.v1`

2. `state_version`

3. `baseline_by_category`
   - 카테고리별 개인 평균 또는 moving stat

4. `threshold_by_category`

5. `warmup_status`
   - 예: `cold_start`, `warming_up`, `ready`

6. `updated_at`

### 선택 필드

1. `personal_prototype_refs`
   - 로컬 prototype 저장 위치나 참조 키

2. `persistence_features`
   - slope, streak, rolling variance 같은 값

3. `calibration_notes`

---

## 4. 포함하지 않을 필드

1. raw text corpus
2. 중앙 서버용 round metadata
3. 전역 모델 전체 가중치

---

## 5. JSON 예시

```json
{
  "schema_version": "personalization_state.v1",
  "state_version": "ps_2026_03_29_001",
  "baseline_by_category": {
    "anxiety": 0.31,
    "depression": 0.22,
    "suicidal": 0.03,
    "normal": 0.71
  },
  "threshold_by_category": {
    "anxiety": 0.62,
    "depression": 0.58,
    "suicidal": 0.30
  },
  "warmup_status": "ready",
  "updated_at": "2026-03-29T14:00:00Z",
  "personal_prototype_refs": {
    "depression": "local://personal_prototypes/depression_v3"
  },
  "persistence_features": {
    "depression_slope": 0.12,
    "anxiety_streak": 4
  }
}
```

---

## 6. 검증 기준

1. 동일한 raw score에 대해 서로 다른 personalization state가 다른 해석을 만들 수 있어야 한다.
2. 로컬 persistence와 복원이 가능해야 한다.
3. 서버 전송이 기본 경로가 아니어야 한다.
