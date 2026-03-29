# DecisionFeedbackSignal v1

## 1. 목적

`DecisionFeedbackSignal`은 로컬 학습에 사용할 수 있는 신호를 표현하는 계약이다.

이 객체는 아래를 표현한다.

1. 어떤 종류의 신호인지
2. 신호의 방향성이 무엇인지
3. 신뢰도와 발생 시점이 언제인지

---

## 2. v1 설계 방향

v1 원칙:

1. 로컬 학습 신호를 명시적으로 모델링한다.
2. pseudo-label과 사용자/시스템 feedback을 같은 객체 계열로 다룬다.
3. 원문 텍스트는 포함하지 않는다.

---

## 3. signal 종류

v1에서 고려하는 대표 signal:

1. `pseudo_label`
2. `self_report`
3. `support_action`
4. `delayed_outcome`

---

## 4. 포함할 필드

### 필수 필드

1. `schema_version`
   - 예: `decision_feedback_signal.v1`

2. `signal_id`
3. `signal_type`
4. `label`
5. `confidence`
6. `occurred_at`

### 선택 필드

1. `source_event_ref`
2. `task_context`
3. `notes`

---

## 5. 포함하지 않을 필드

1. raw text
2. 개인 신상 식별자
3. 서버 판정 결과

---

## 6. JSON 예시

```json
{
  "schema_version": "decision_feedback_signal.v1",
  "signal_id": "signal_2026_03_29_001",
  "signal_type": "pseudo_label",
  "label": "depression_rising",
  "confidence": 0.91,
  "occurred_at": "2026-03-29T13:10:00Z",
  "source_event_ref": "local_event_00031"
}
```

---

## 7. 검증 기준

1. signal type taxonomy가 애매하지 않아야 한다.
2. 로컬 학습 service가 signal만으로 objective branch를 고를 수 있어야 한다.
3. 원문 없이도 사용 가능해야 한다.
