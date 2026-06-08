# TrainingTask v1

## 1. 목적

`TrainingTask`는 중앙 `main_server`가 로컬 `agent`에 내려주는 학습 작업 정의다.

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
6. 이 문서는 현재 시스템/FL runtime task 계약을 설명한다. 논문 트랙의
   query-domain PEFT text encoder 적응은 별도 중앙 실험 레일 config를 사용할 수 있다.

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
    - local update backend 식별자, loss 이름, threshold, score/acceptance/privacy 정책 등
    - canonical key는 `training_backend_name`
    - 입력 호환을 위해 legacy key `loss`를 받을 수 있지만, 직렬화와 문서 기준 키는 `training_backend_name`

13. `selection_policy`
    - 어떤 로컬 샘플을 학습에 사용할지

14. `deadline_at`

### 선택 필드

1. `gradient_clip_norm`
2. `min_required_examples`
3. `secure_aggregation`
4. `notes`

---

## 4. 포함하지 않을 필드

1. 개인화 상태
2. 개별 샘플 본문
3. 학습 결과 payload

---

## 4-1. `training_scope` 의미

`training_scope`는 "모델의 어느 파라미터 범위를 업데이트 대상으로 삼는가"를
나타낸다. 이것은 `adapter_family`나 `training_backend`와는 다른 축이다.

1. `adapter_only`
   - adapter 파라미터만 학습한다.
   - 현재 TraceMind v1에서는 비교/확장 축에서 주로 사용한다.

2. `head_only`
   - backbone/encoder는 고정하고 최종 분류 head만 학습한다.
   - 고정 feature 위 linear head baseline에 쓰는 범위다.
   - shared adapter family와는 별개 축이다.

3. `selected_encoder_block`
   - encoder 전체가 아니라 미리 정한 일부 블록만 학습한다.
   - PEFT text encoder처럼 특정 transformer block에 trainable module을 삽입하는
     경우와 잘 맞는다.

4. `full_encoder`
   - encoder 전체를 학습한다.
   - 현재 논문 트랙의 기본 대상은 아니며, upper-bound 비교나 별도 benchmark용에 가깝다.

정리하면:

- `training_scope`는 "어디를 학습하느냐"
- `update_family_name`은 "그 범위를 어떤 공유 가능 trainable state로 해석하느냐"

예를 들어 `adapter_only` 범위 안에서도 `peft_text_encoder`,
`linear_head`는 서로 다른 선택이다.

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
    "training_backend_name": "peft_classifier_trainer",
    "algorithm_profile_name": "peft_classifier_update_v1",
    "example_generation_backend_name": "peft_classifier_raw_rows",
    "privacy_guard_name": "noop",
    "query_ssl.method_name": "fixmatch_usb_v1",
    "query_ssl.algorithm_name": "fixmatch",
    "query_ssl.strong_view_policy": "first_aug",
    "query_ssl.unlabeled_batch_size": 8,
    "query_ssl.temperature": 0.5,
    "query_ssl.p_cutoff": 0.95,
    "query_ssl.hard_label": true,
    "query_ssl.lambda_u": 1.0,
    "query_ssl.supervised_loss_weight": 1.0,
    "peft_classifier.delta_format": "inline_delta"
  },
  "selection_policy": {
    "max_examples": 128,
    "require_feedback": false
  },
  "deadline_at": "2026-03-30T00:00:00Z",
  "gradient_clip_norm": 1.0,
  "min_required_examples": 20,
  "secure_aggregation": {
    "required": false
  }
}
```

`secure_aggregation`은 secure aggregation 계층 요구사항을 나타내는 task 축이다.
server round runtime이 내부적으로 어떤 aggregation backend를 선택하는지와는 별도다.
예를 들어 현재 round runtime의 기본 aggregation backend 이름은 `fedavg`지만,
이 값은 `TrainingTaskPayload.secure_aggregation` 필드와 같은 의미가 아니다.

위 JSON은 시스템/FL runtime의 `adapter_only + peft_classifier_update_v1` 예시다.
논문 트랙의 중앙 PEFT text encoder 적응 비교는 별도 중앙 trainer
config를 사용하며, 이 예시를 그대로 재현 기준으로 삼지 않는다.

명시 objective 없이 live/API round를 열 때의 runtime fallback은
`methods/federated_ssl/runtime_fallbacks.py`가 소유한다. 현재 기본 profile은
`fixmatch_fedavg.v1`이며 `query_ssl.method_name=fixmatch_usb_v1`,
`query_ssl.algorithm_name=fixmatch`, `update_family=peft_text_encoder`,
`aggregation=fedavg` 조합을 가리킨다. live `run-current-task`는 서버가 바로
집계할 수 있도록 `peft_classifier.delta_format=inline_delta`를 기본으로 사용한다.
agent는 `query_ssl.*` objective를 legacy pseudo-label runner로 흘려보내지 않고
Query SSL raw-row runner로 분기한다.

---

## 6. 검증 기준

1. task만 보고 로컬이 학습 가능 여부를 판단할 수 있어야 한다.
2. `training_scope`와 `model_revision`이 빠지지 않아야 한다.
3. objective와 selection policy가 분리돼 있어야 한다.
4. `training_backend_name`, `example_generation_backend_name`,
   `scorer_backend_name`, `score_policy_name`,
   `pseudo_label_algorithm_name`, `acceptance_policy_name`,
   `privacy_guard_name`은 서로 다른 교체 축으로 해석돼야 한다.
   단, PEFT Query SSL raw-row runtime은 scorer backend를 쓰지 않으므로
   `scorer_backend_name`을 기본값으로 채우지 않는다. stored-event scoring
   runtime처럼 scorer가 실제로 필요한 경로만 명시한다.
