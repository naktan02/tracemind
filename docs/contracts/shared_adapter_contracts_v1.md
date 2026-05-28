# Shared Adapter Contracts v1

## 1. 목적

이 문서는 TraceMind의 전역 shared state/update payload 계약을 설명한다.
이름은 historical하게 `shared adapter`를 유지하지만, 현재 concrete family는
generic linear head family인 `classifier_head`와 PEFT text encoder family의
payload kind인 `peft_classifier`다. `diagonal_scale`/`vector_adapter_*` parser와
factory, v1 `lora_classifier` shared parser/factory는 target 구조에서 제거됐다.

이 문서가 다루는 범위는 두 가지다.

1. 서버가 배포하는 전역 adapter 상태
2. 로컬 agent가 학습 후 서버로 보내는 adapter update

즉 이 문서는 `TrainingUpdateEnvelope`의 메타데이터가 아니라,
그 안에 실제로 담기는 shared adapter payload의 의미를 설명한다.

중요:

- 이 문서는 현재 시스템/FL runtime의 shared state/update 계약을 다룬다.
- query-domain 적응 단계의 `central PEFT classifier` checkpoint 전체를 설명하는
  문서는 아니다.

---

## 2. 왜 별도 계약이 필요한가

`ModelManifest`, `TrainingTask`, `TrainingUpdateEnvelope`만으로는 아래가 명확하지 않다.

1. adapter가 실제로 어떤 파라미터를 가지는지
2. 그 파라미터가 임베딩에 어떤 수학적 변환을 적용하는지
3. 로컬 update가 "가중치 전체"인지 "일부 delta"인지
4. 서버 집계 후 어떤 값이 다음 revision으로 발행되는지

`shared adapter`는 linear head, PEFT text encoder, prototype-derived state 같은
서로 다른 trainable state family로 확장될 수 있으므로 payload 자체도 별도
계약으로 관리한다.

---

## 3. 큰 구조

shared adapter 계약은 두 레이어로 나뉜다.

1. `SharedAdapterStatePayload`
   - 현재 활성 전역 adapter 상태
   - 서버가 보관/배포

2. `SharedAdapterUpdatePayload`
   - 로컬 agent가 생성한 update
   - 서버가 round 단위로 집계

현재 concrete family는 `adapter_kind = classifier_head`, `peft_classifier` 두 개다.

---

## 4. 공통 필드

### 4-1. `SharedAdapterStatePayload`

필드 의미:

1. `schema_version`
   - payload 자체의 스키마 버전
   - 예: `vector_adapter_state.v1`

2. `adapter_kind`
   - 어떤 family인지 나타낸다.
   - 현재 값 예: `classifier_head`, `peft_classifier`

3. `model_id`
   - 이 adapter 상태가 속한 전역 모델 식별자

4. `model_revision`
   - 현재 active adapter revision

5. `training_scope`
   - 이 adapter가 어떤 범위 학습을 전제로 하는지
   - 예: `adapter_only`

6. `updated_at`
   - 이 상태가 발행된 시각

### 4-2. `SharedAdapterUpdatePayload`

필드 의미:

1. `schema_version`
   - update payload 스키마 버전
   - 예: `vector_adapter_delta.v1`

2. `adapter_kind`
   - 어떤 family용 update인지
   - 현재 값 예: `classifier_head`, `peft_classifier`

3. `model_id`
   - 어떤 전역 모델에 대한 update인지

4. `base_model_revision`
   - 이 update를 만들 때 agent가 사용한 기준 revision

5. `training_scope`
   - task가 허용한 학습 범위

6. `example_count`
   - update 생성에 실제 사용된 accepted example 수

7. `created_at`
   - 로컬 update 생성 시각

현재 FL runtime의 기본 update family는 `peft_text_encoder`이고, payload kind는
`peft_classifier` v2를 사용한다. `classifier_head`는 고정 feature 위 linear head
family를 위한 shared value object로 남아 있으며 top-level FL 기본 축은 아니다.
과거 `diagonal_scale`와 `lora_classifier` payload는 shared parser/factory가 아니라
old artifact/report reader compatibility에서만 다룬다.

---

## 5. 제거된 v1 compatibility payload: `diagonal_scale`

이 절은 historical reference다. `diagonal_scale` shared parser/factory와 methods/runtime
구현은 제거됐고, 새 canonical 표면은 없다.

이 v1 adapter payload는 임베딩 차원마다 scale 하나씩 갖는 가장 단순한 형태다.

상태 payload:

```json
{
  "schema_version": "vector_adapter_state.v1",
  "adapter_kind": "diagonal_scale",
  "model_id": "tracemind-embed",
  "model_revision": "rev_001",
  "training_scope": "adapter_only",
  "dimension_scales": [1.0, 0.98, 1.03],
  "updated_at": "2026-03-31T00:00:00Z"
}
```

적용식:

```text
x' = normalize(x ⊙ s)
```

설명:

1. `x`
   - backbone이 만든 원래 임베딩

2. `s = dimension_scales`
   - 각 차원별 scale 벡터

3. `⊙`
   - 차원별 곱

4. `normalize`
   - L2 정규화

중요:

- 이것은 prototype 쪽으로 점을 직접 끌어당기는 translation이 아니다.
- 공간 전체에 공통으로 적용되는 차원별 rescaling이다.
- 따라서 "A prototype으로 직접 이동"이라기보다,
  "accepted example 분포에 맞게 각 축을 조금 늘리고 줄여서
   이후 임베딩과 prototype의 상대 위치가 달라지게 만드는 방식"에 가깝다.

---

## 6. 현재 concrete 구현 2: `classifier_head`

현재 시스템/FL v1의 우선 baseline은 backbone을 고정한 채 category별 선형
classifier head를 공유하는 방식이다.

state payload:

```json
{
  "schema_version": "classifier_head_state.v1",
  "adapter_kind": "classifier_head",
  "model_id": "tracemind-embed",
  "model_revision": "rev_001",
  "training_scope": "head_only",
  "label_weights": {
    "anxiety": [0.1, 0.0, -0.2],
    "depression": [-0.1, 0.3, 0.2]
  },
  "label_biases": {
    "anxiety": 0.0,
    "depression": 0.0
  },
  "updated_at": "2026-03-31T00:00:00Z"
}
```

적용식:

```text
z_c = w_c^T x + b_c
p(c | x) = softmax(z)_c
```

설명:

1. `x`
   - backbone이 만든 임베딩
2. `w_c`, `b_c`
   - category `c`의 classifier weight와 bias
3. 이 family는 임베딩 위치를 바꾸지 않고,
   현재 임베딩 공간 위에서 category decision boundary를 학습한다.
4. prototype은 main scoring 파라미터가 아니라 classifier bootstrap/comparison
   artifact로 남는다.
5. 이 family는 시스템 translation용 shared head를 뜻하며, 논문 트랙의
   frozen backbone + PEFT encoder classifier checkpoint 전체와 동일한 개념은 아니다.

update payload:

```json
{
  "schema_version": "classifier_head_delta.v1",
  "adapter_kind": "classifier_head",
  "model_id": "tracemind-embed",
  "base_model_revision": "rev_001",
  "training_scope": "head_only",
  "label_weight_deltas": {
    "anxiety": [0.01, 0.0, -0.01],
    "depression": [-0.02, 0.01, 0.0]
  },
  "label_bias_deltas": {
    "anxiety": 0.001,
    "depression": -0.001
  },
  "example_count": 12,
  "mean_confidence": 0.91,
  "mean_margin": 0.34,
  "label_counts": {
    "anxiety": 7,
    "depression": 5
  },
  "created_at": "2026-03-31T01:00:00Z"
}
```

서버는 같은 label order와 embedding dim을 가진 client delta를 가중 평균해
다음 classifier head state를 만든다.

---

## 7. 제거된 v1 compatibility payload: `lora_classifier`

이 절은 historical reference다. `lora_classifier` shared parser/factory와 golden
fixture는 제거됐고, 현재 active payload는 `peft_classifier` v2다. 과거 artifact를
읽어야 하는 흐름은 old artifact/report/materialization reader가 자기 경계에서
legacy 값을 읽고 canonical PEFT 표면으로 정규화한다.

state payload의 핵심 의미:

1. backbone/tokenizer reference
   - 어떤 frozen encoder와 tokenizer 위에서 학습된 state인지 나타낸다.
2. PEFT adapter config
   - LoRA/DoRA 같은 mechanism 이름과 rank, alpha, dropout, target module 같은
     scaffold 고정값을 담는다.
3. label schema
   - classifier head weight/bias의 label order와 의미를 고정한다.
4. PEFT adapter/head artifact ref
   - 다음 global state가 참조하는 server-owned 누적 snapshot이다.
   - client update delta artifact와 같은 의미가 아니다.

update payload의 핵심 의미:

1. `base_model_revision`
   - client가 local update를 계산한 기준 global revision이다.
2. PEFT adapter/head delta artifact ref
   - 큰 weight delta는 `peft_adapter_delta_artifact_ref`,
     `classifier_head_delta_artifact_ref` 같은 artifact-ref 경로를 기본으로 한다.
3. optional inline delta
   - 작은 smoke나 deterministic 단위 검증에서만 쓴다.
   - runtime은 artifact-ref와 inline delta를 명시적으로 구분해야 한다.
4. raw text exclusion
   - shared update payload에는 raw text나 agent-local query state를 넣지 않는다.

현재 PEFT text encoder payload shape의 source of truth는
`shared/src/contracts/adapter_contract_families/peft_classifier.py`와
`shared/src/contracts/README.md`다.

---

## 8. Historical update: `DiagonalScaleAdapterUpdatePayload`

update payload:

```json
{
  "schema_version": "vector_adapter_delta.v1",
  "adapter_kind": "diagonal_scale",
  "model_id": "tracemind-embed",
  "base_model_revision": "rev_001",
  "training_scope": "adapter_only",
  "dimension_deltas": [0.01, -0.02, 0.0],
  "example_count": 12,
  "mean_confidence": 0.84,
  "mean_margin": 0.11,
  "label_counts": {
    "anxiety": 7,
    "depression": 5
  },
  "created_at": "2026-03-31T01:00:00Z"
}
```

필드 의미:

1. `dimension_deltas`
   - 현재 revision의 `dimension_scales`에 더할 변화량
   - 임베딩 자체를 이동시키는 좌표값이 아니라, scale 파라미터의 delta다.

2. `mean_confidence`
   - accepted example들의 평균 confidence

3. `mean_margin`
   - accepted example들의 평균 margin

4. `label_counts`
   - 어떤 pseudo-label 분포로 update가 만들어졌는지 요약하는 메타데이터

---

## 9. v1 diagonal_scale heuristic 기준 update 의미

삭제된 `diagonal_scale` methods-level 구현은 gradient 학습이 아니라 heuristic
방식이었다. 아래 내용은 과거 payload 의미를 읽기 위한 compatibility 설명이다.

로컬에서 하는 일:

1. 쿼리를 backbone으로 임베딩한다.
2. 현재 adapter state를 적용한다.
3. prototype scoring으로 pseudo-label을 고른다.
4. threshold를 넘은 accepted example만 남긴다.
5. accepted example 임베딩의 confidence 가중 평균을 구한다.
6. 그 평균 방향으로 `dimension_deltas`를 만든다.

`diagonal_scale` methods-level heuristic core와 agent runtime adapter는 제거됐다.
이 절은 v1 payload를 읽을 때의 의미 해석을 설명하는 compatibility 기록이며, 새
runtime/update-family 구현 위치로 해석하지 않는다.

즉 현재 update는
"특정 prototype으로 직접 당기기"
가 아니라,
"선택된 example 분포가 자주 활성화시키는 차원을 조금 더 키우고,
 덜 맞는 차원은 줄이는 전역 보정"
으로 보는 편이 정확하다.

---

## 10. v1 서버 집계 의미

서버 집계 payload compatibility는 같은 `adapter_kind`끼리만 집계한다.

과거 `diagonal_scale` 집계식은:

```text
next_scale = clamp(base_scale + weighted_mean(delta), min_scale, max_scale)
```

설명:

1. client update를 `example_count` 가중 평균한다.
2. 현재 scale에 더한다.
3. 안정성을 위해 범위를 clamp한다.

그 다음 새 adapter state로 bootstrap rows를 다시 임베딩해서
새 `PrototypePack`을 다시 만든다.

즉 이 v1 구조에서 바뀌는 것은:

1. adapter state
2. adapter가 적용된 query 위치
3. 새 state로 재생성된 prototype 위치

---

## 11. 현재 코드에서의 격리 방식

현재 코드는 아래 세 축을 분리한다.

1. `adapter family`
   - `adapter_kind`로 구분

2. `local training backend`
   - 새 backend는 method-owned core와 agent runtime adapter를 분리한다.

3. `server aggregation backend`
   - 예: `fedavg` reusable backend와 main_server generic executor

adapter family별 payload 해석과 projection은 해당 owner package가 소유하고,
재사용 aggregation backend는 `methods/federated/aggregation/`가 소유한다.
특정 논문 method에만 종속된 aggregation 변형은 `methods/federated_ssl/<method>/`에
둘 수 있다. `main_server`는 round lifecycle과 publication adapter로 남긴다.

---

## 12. v1 한계

현재 v1 한계는 분명하다.

1. `lora_classifier`는 열렸지만 full encoder, projection head, 추가 PEFT family는
   아직 열지 않았다.
2. `diagonal_scale`은 active methods-level update family가 아니라 v1 contract
   compatibility 표면이다.
3. live agent stored-event 경로는 classifier multiview/runtime 조합이 아직 약하다.
4. projection head, full encoder FL, 다른 PEFT family를 열려면 payload 타입과
   backend 구현을 더 추가해야 한다.

즉 이 문서는 "최종형 계약"보다는
"shared adapter를 family 단위로 확장하기 위한 첫 계약"으로 읽는 것이 맞다.
