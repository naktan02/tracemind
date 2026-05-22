# ModelManifest v1

## 1. 목적

`ModelManifest`는 현재 활성 전역 모델 구성을 로컬 `agent`에 배포하기 위한 메타데이터 계약이다.

이 문서는 아래를 명확히 한다.

1. 어떤 모델 revision이 활성인지
2. 어떤 artifact가 함께 배포되어야 하는지
3. 필요한 부속 artifact가 있는지
4. 해당 모델이 추론 전용인지, 학습 참여 가능한지

즉 `ModelManifest`는 모델 가중치 자체가 아니라,
가중치와 부속 artifact를 해석하기 위한 배포 메타데이터다.

---

## 2. v1 설계 방향

v1 원칙:

1. 중앙 `main_server`가 발행한다.
2. 로컬 `agent`는 이를 pull해서 현재 활성 모델 구성을 동기화한다.
3. artifact 자체와 manifest를 분리한다.
4. 추론 호환성과 학습 가능 범위를 함께 명시한다.
5. prototype은 manifest의 필수 의미가 아니다. 필요한 방법론만 선택적
   auxiliary artifact로 `prototype_pack` version을 기록한다.
   top-level `prototype_version`은 canonical 필드가 아니다.
6. 이 문서는 현재 시스템/FL runtime의 배포 메타데이터를 설명한다. 논문 트랙의
   중앙집중형 LoRA trainer checkpoint/optimizer state는 범위 밖이다.

---

## 3. 포함할 필드

### 필수 필드

1. `schema_version`
   - 예: `model_manifest.v1`

2. `model_id`
   - 전역 모델 식별자

3. `model_revision`
   - 현재 활성 revision

4. `published_at`
   - 발행 시각

5. `artifact_kind`
   - 예: `embedding_backbone`, `adapter`, `decision_head`

6. `artifact_ref`
   - 실제 artifact 위치 또는 조회 키

7. `training_scope`
   - 허용된 학습 범위
   - 예: `head_only`, `adapter_only`, `selected_encoder_block`, `full_encoder`

8. `training_enabled`
   - 현재 revision이 학습 참여 가능한지 여부

9. `compatible_task_types`
    - 예: `pseudo_label_self_training`, `feedback_supervised`

### 선택 필드

1. `base_model_id`
2. `base_model_revision`
3. `translation_model_id`
4. `translation_model_revision`
5. `auxiliary_artifact_versions`
   - 예: `{ "prototype_pack": "proto_2026_03_28_163056" }`
6. `notes`

---

## 4. 포함하지 않을 필드

v1에서는 아래를 포함하지 않는다.

1. 개인화 상태
2. agent별 설정
3. 개별 학습 샘플
4. raw gradient나 update payload
5. top-level `prototype_version`

구형 manifest JSON에 `prototype_version`이 있으면 runtime parser가
`auxiliary_artifact_versions["prototype_pack"]`로 승격한다. 새 manifest는 이 필드를
출력하지 않는다.

---

## 5. JSON 예시

```json
{
  "schema_version": "model_manifest.v1",
  "model_id": "tracemind-embed",
  "model_revision": "tm_head_2026_03_29_001",
  "published_at": "2026-03-29T12:00:00Z",
  "artifact_kind": "decision_head",
  "artifact_ref": "models/tracemind-embed/heads/tm_head_2026_03_29_001",
  "training_scope": "head_only",
  "training_enabled": true,
  "compatible_task_types": [
    "pseudo_label_self_training",
    "feedback_supervised"
  ],
  "base_model_id": "mixedbread-ai/mxbai-embed-large-v1",
  "base_model_revision": "main"
}
```

위 예시는 현재 시스템/FL runtime의 shared classifier head 배포 예시다.
query-domain 적응 단계의 `central LoRA classifier`는 frozen backbone에 LoRA adapter와 classifier를 함께 두므로,
checkpoint 배치 방식이 이 예시와 다를 수 있다.

---

## 6. 검증 기준

1. manifest만 보고 로컬이 필요한 모델 artifact와 optional auxiliary artifact를
   결정할 수 있어야 한다.
2. 어떤 revision에서 나온 update인지 추적 가능해야 한다.
3. `training_scope`가 비어 있지 않아야 한다.
4. 같은 round의 추론과 pseudo-label 생성이 동일한 active model revision을
   참조할 수 있어야 한다.
