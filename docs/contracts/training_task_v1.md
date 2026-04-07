# TrainingTask v1

## 1. 목적

`TrainingTask`는 중앙 `main-server`가 로컬 `agent`에 내려주는 학습 작업 정의다.

이 객체는 아래를 고정한다.

1. 어떤 round에 참여하는지
2. 어떤 model revision을 기준으로 학습하는지
3. 어떤 파라미터 범위를 업데이트할지
4. 어떤 objective와 하이퍼파라미터를 사용할지

---

## 2. v1 설계 방향

v1 원칙:

1. 중앙이 발행한다.
2. 로컬은 task를 수락할지 결정할 수 있다.
3. task는 model artifact 자체를 담지 않고, `ModelManifest`를 참조한다.
4. 초기에는 작은 param subset 학습을 우선한다.
5. `task.model_revision`은 현재 active pair의 `model_revision`과 일치해야 한다.

---

## 3. 포함할 필드

### 필수 필드

1. `schema_version`
   - 예: `training_task.v1`

2. `task_id`
   - 작업 식별자

3. `round_id`
   - 라운드 식별자

4. `model_id`
5. `model_revision`

6. `task_type`
   - 예: `pseudo_label_self_training`, `feedback_supervised`

7. `training_scope`
   - 예: `head_only`, `adapter_only`, `selected_encoder_block`

8. `local_epochs`
9. `batch_size`
10. `learning_rate`
11. `max_steps`

12. `objective_config`
    - loss 종류, margin threshold 등 objective 관련 설정

13. `selection_policy`
    - 어떤 로컬 샘플을 학습에 사용할지

14. `deadline_at`

### 선택 필드

1. `gradient_clip_norm`
2. `min_required_examples`
3. `secure_aggregation_required`
4. `notes`

---

## 4. 포함하지 않을 필드

1. 개인화 상태
2. 개별 샘플 본문
3. 학습 결과 payload

---

## 5. JSON 예시

```json
{
  "schema_version": "training_task.v1",
  "task_id": "task_2026_03_29_001",
  "round_id": "round_0001",
  "model_id": "tracemind-embed",
  "model_revision": "tm_embed_2026_03_29_001",
  "task_type": "pseudo_label_self_training",
  "training_scope": "adapter_only",
  "local_epochs": 1,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "max_steps": 50,
  "objective_config": {
    "training_backend_name": "diagonal_scale_heuristic",
    "loss_name": "contrastive",
    "confidence_threshold": 0.8,
    "margin_threshold": 0.15
  },
  "selection_policy": {
    "max_examples": 128,
    "require_feedback": false
  },
  "deadline_at": "2026-03-30T00:00:00Z",
  "gradient_clip_norm": 1.0,
  "min_required_examples": 20,
  "secure_aggregation_required": false
}
```

---

## 6. 검증 기준

1. task만 보고 로컬이 학습 가능 여부를 판단할 수 있어야 한다.
2. `training_scope`와 `model_revision`이 빠지지 않아야 한다.
3. objective와 selection policy가 분리돼 있어야 한다.
