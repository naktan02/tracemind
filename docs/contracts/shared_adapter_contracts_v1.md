# Shared Adapter Contracts v1

## 1. 목적

이 문서는 TraceMind의 공통 표현 보정 계층인 `shared adapter` 관련 계약을 설명한다.

이 문서가 다루는 범위는 두 가지다.

1. 서버가 배포하는 전역 adapter 상태
2. 로컬 agent가 학습 후 서버로 보내는 adapter update

즉 이 문서는 `TrainingUpdateEnvelope`의 메타데이터가 아니라,
그 안에 실제로 담기는 shared adapter payload의 의미를 설명한다.

---

## 2. 왜 별도 계약이 필요한가

`ModelManifest`, `TrainingTask`, `TrainingUpdateEnvelope`만으로는 아래가 명확하지 않다.

1. adapter가 실제로 어떤 파라미터를 가지는지
2. 그 파라미터가 임베딩에 어떤 수학적 변환을 적용하는지
3. 로컬 update가 "가중치 전체"인지 "일부 delta"인지
4. 서버 집계 후 어떤 값이 다음 revision으로 발행되는지

`shared adapter`는 앞으로 diagonal scale, LoRA, projection head 등 여러 family로 확장될 수 있으므로,
payload 자체도 별도 계약으로 관리하는 편이 낫다.

---

## 3. 큰 구조

shared adapter 계약은 두 레이어로 나뉜다.

1. `SharedAdapterStatePayload`
   - 현재 활성 전역 adapter 상태
   - 서버가 보관/배포

2. `SharedAdapterUpdatePayload`
   - 로컬 agent가 생성한 update
   - 서버가 round 단위로 집계

현재 concrete 구현은 둘 다 `adapter_kind = diagonal_scale` 하나만 지원한다.

---

## 4. 공통 필드

### 4-1. `SharedAdapterStatePayload`

필드 의미:

1. `schema_version`
   - payload 자체의 스키마 버전
   - 예: `vector_adapter_state.v1`

2. `adapter_kind`
   - 어떤 adapter family인지 나타낸다.
   - 현재 값: `diagonal_scale`

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
   - 어떤 adapter family용 update인지
   - 현재 값: `diagonal_scale`

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

---

## 5. 현재 concrete 구현: `diagonal_scale`

현재 adapter는 임베딩 차원마다 scale 하나씩 갖는 가장 단순한 형태다.

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

## 6. 현재 concrete update: `DiagonalScaleAdapterUpdatePayload`

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

## 7. 현재 heuristic 기준 update 생성 방식

현재 `diagonal_scale` 구현은 gradient 학습이 아니라 heuristic 방식이다.

로컬에서 하는 일:

1. 쿼리를 backbone으로 임베딩한다.
2. 현재 adapter state를 적용한다.
3. prototype scoring으로 pseudo-label을 고른다.
4. threshold를 넘은 accepted example만 남긴다.
5. accepted example 임베딩의 confidence 가중 평균을 구한다.
6. 그 평균 방향으로 `dimension_deltas`를 만든다.

즉 현재 update는
"특정 prototype으로 직접 당기기"
가 아니라,
"선택된 example 분포가 자주 활성화시키는 차원을 조금 더 키우고,
 덜 맞는 차원은 줄이는 전역 보정"
으로 보는 편이 정확하다.

---

## 8. 서버 집계 방식

현재 서버는 같은 `adapter_kind`끼리만 집계한다.

현재 `diagonal_scale` 집계식은:

```text
next_scale = clamp(base_scale + weighted_mean(delta), min_scale, max_scale)
```

설명:

1. client update를 `example_count` 가중 평균한다.
2. 현재 scale에 더한다.
3. 안정성을 위해 범위를 clamp한다.

그 다음 새 adapter state로 bootstrap rows를 다시 임베딩해서
새 `PrototypePack`을 다시 만든다.

즉 현재 구조에서 바뀌는 것은:

1. adapter state
2. adapter가 적용된 query 위치
3. 새 state로 재생성된 prototype 위치

---

## 9. 현재 코드에서의 결합 방식

현재 코드는 아래 세 축을 분리하려는 방향으로 정리돼 있다.

1. `adapter family`
   - `adapter_kind`로 구분

2. `local training backend`
   - 예: `DiagonalScaleHeuristicTrainingBackend`

3. `server aggregation backend`
   - 예: `DiagonalScaleAggregationService`

즉 같은 `diagonal_scale` family 안에서는
`heuristic -> gradient` 교체가 비교적 쉽고,
나중에는 `lora`, `projection_head` 같은 다른 family도 추가할 수 있다.

다만 현재는 첫 단계이므로,
registry가 완전히 일반화된 상태는 아니고 `diagonal_scale` 하나를 concrete 구현으로 둔 구조다.

---

## 10. v1 한계

현재 v1 한계는 분명하다.

1. adapter family는 아직 `diagonal_scale` 하나뿐이다.
2. local update는 gradient가 아니라 heuristic이다.
3. scoring/runtime은 여전히 single-centroid 중심 구조다.
4. richer adapter family를 넣으려면 payload 타입과 backend 구현을 더 추가해야 한다.

즉 이 문서는 "최종형 계약"보다는
"shared adapter를 family 단위로 확장하기 위한 첫 계약"으로 읽는 것이 맞다.
