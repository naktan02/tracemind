# TrainingUpdateEnvelope v1

## 1. 목적

`TrainingUpdateEnvelope`는 로컬 `agent`가 중앙 `main_server`로 보내는 학습 업데이트 단위다.

이 객체는 payload 그 자체보다,
payload를 안전하게 집계하고 추적하기 위한 메타데이터를 정의한다.

---

## 2. v1 설계 방향

v1 원칙:

1. update payload와 메타데이터를 분리한다.
2. 어떤 base model revision에서 생성된 update인지 반드시 남긴다.
3. 로컬 개인화 상태를 직접 포함하지 않는다.
4. 개인정보성 원문이나 개별 샘플을 포함하지 않는다.
5. 이 문서는 현재 시스템/FL runtime update 봉투를 설명한다. 중앙집중형
   논문 트랙의 PEFT text encoder trainer checkpoint/gradient 저장 형식은 범위 밖이다.

---

## 3. 포함할 필드

### 필수 필드

1. `schema_version`
   - 예: `training_update_envelope.v1`

2. `update_id`
3. `round_id`
4. `task_id`

5. `model_id`
6. `base_model_revision`
   - pseudo-label selection과 local training에 사용한 active pair의 model revision

7. `training_scope`

8. `payload_ref`
   - 실제 update tensor/blob 위치 또는 조회 키

9. `payload_format`
   - 예: `classifier_head_update`, `peft_classifier_update`
   - `diagonal_scale_update`, `lora_classifier_update`는 active producer에서 제거된
     legacy format이며, 필요하면 old-run reader fallback에서만 해석한다

10. `example_count`

11. `client_metrics`
    - loss, acceptance ratio, confidence 평균 등

12. `created_at`

### 선택 필드

1. `clipped`
2. `dp_applied`
3. `checksum`
4. `agent_id`
   - 기기가 직접 생성한 pseudonymous UUID
   - 실제 사용자 신원 아님
   - q-합의 알고리즘에서 per-agent 신뢰도 추적과 중복 제출 차단에 사용
   - None이면 완전 익명 모드
5. `notes`

---

## 4. 포함하지 않을 필드

1. raw text
2. 개별 pseudo-label 목록
3. `PersonalizationState` 본문
4. `AssessmentResult` 본문

---

## 5. JSON 예시

```json
{
  "schema_version": "training_update_envelope.v1",
  "update_id": "update_2026_03_29_001",
  "round_id": "round_0001",
  "task_id": "task_2026_03_29_001",
  "model_id": "tracemind-embed",
  "base_model_revision": "tm_embed_2026_03_29_001",
  "training_scope": "head_only",
  "payload_ref": "updates/round_0001/update_2026_03_29_001",
  "payload_format": "classifier_head_update",
  "example_count": 64,
  "client_metrics": {
    "accepted_ratio": 0.31,
    "mean_confidence": 0.87,
    "mean_margin": 0.12,
    "delta_l2_norm": 0.19,
    "selected_examples": 64.0
  },
  "created_at": "2026-03-29T14:20:00Z",
  "clipped": false,
  "dp_applied": false,
  "agent_id": "3f4a2b1c-9e8d-4f7e-a6b5-2c1d0e9f8a7b"
}
```

위 예시는 시스템/FL runtime의 `classifier_head` update 봉투 예시다.
논문 트랙의 중앙집중형 PEFT text encoder trainer 결과물은 이 envelope을 직접 쓰지 않는다.

---

## 6. 검증 기준

1. update의 base revision을 추적할 수 있어야 한다.
2. aggregation에 필요한 최소 메타데이터가 있어야 한다.
3. 개인 상태나 원문이 포함되지 않아야 한다.
4. `agent_id`는 서버가 발급하지 않는다. 기기가 직접 생성한 UUID여야 한다.

---

## 7. 관련 가이드

- [aggregation_algorithm_transition_guide.md](algorithm_extension_guide.md#7-aggregation-backend-main_server)
  — q-합의 등 aggregation 알고리즘 추가 시 교체 지점과 체크리스트
