# Shared Contracts

이 디렉터리는 agent, main-server, script가 공통으로 읽는 payload 계약의 source of truth다.

문서 우선순위는 이렇게 본다.

1. 이 폴더의 Python contract 파일
2. 이 README
3. `docs/contracts/*` 설계 메모

`docs/contracts/*`는 배경과 설계 이유를 설명하는 보조 문서이고, 실제 필드 의미와 포맷은 이 폴더 파일이 기준이다.

## 주요 파일

### `adapter_contracts.py`

Shared adapter 상태와 update payload를 정의한다.

- `SharedAdapterStatePayload`
  - 서버가 현재 배포하는 전역 shared adapter 상태 공통 필드
- `DiagonalScaleAdapterStatePayload`
  - 현재 concrete 구현
  - `dimension_scales`는 임베딩 차원별 전역 scale 벡터
- `SharedAdapterUpdatePayload`
  - agent가 서버로 올리는 shared adapter update 공통 필드
- `DiagonalScaleAdapterUpdatePayload`
  - 현재 concrete 구현
  - `dimension_deltas`는 차원별 scale 변화량
  - `label_counts`는 drift 관찰용 메타데이터이며, 직접 gradient 자체는 아니다

### `training_contracts.py`

FL orchestration과 로컬 학습 제어용 envelope을 정의한다.

- `TrainingTaskPayload`
  - 서버가 agent에 내려주는 학습 task
  - 로컬 학습 하이퍼파라미터, threshold, selection policy 포함
- `TrainingUpdateEnvelopePayload`
  - agent가 올리는 update 메타데이터 봉투
  - 실제 파라미터 payload는 `payload_ref`가 가리키는 별도 파일에 있음
- `DecisionFeedbackSignalPayload`
  - pseudo-label, 사용자 피드백, 후속 결과 등 로컬 학습용 signal 단위 계약

### `prototype_contracts.py`

Prototype runtime이 직접 읽는 semantic artifact 계약을 정의한다.

- `PrototypePackPayload`
  - category마다 하나 이상의 prototype을 가진다
  - single prototype도 길이 1 리스트로 정규화해서 해석한다
- `CategoryPrototypePayload`
  - `prototype_id`, `centroid`, `sample_count`를 담는다
- `extract_category_prototypes(...)`
  - runtime scoring용 `category -> prototype vectors` 변환 helper
- `extract_category_centroids(...)`
  - single-prototype pack에서만 쓰는 legacy helper

### `prototype_build_state_contracts.py`

Prototype exact incremental merge용 build-state 계약을 정의한다.

- `PrototypeBuildStatePayload`
  - 현재 v1은 category별 `embedding_sum`, `sample_count`만 담는다
  - 따라서 exact incremental merge는 single mean-centroid builder 전용이다
  - multi-prototype builder는 build-state 없이 pack만 생성할 수 있다

## 해석 규칙

- `adapter_kind`
  - adapter family discriminator
  - 예: `diagonal_scale`
- `payload_format`
  - 동일 family 안에서도 state/update envelope 해석에 쓰는 포맷 식별자
- `training_scope`
  - 어느 수준까지 학습하는지 나타내는 범위 식별자
  - 현재는 주로 `adapter_only`
- `model_revision` / `base_model_revision`
  - `model_revision`: 서버가 현재 배포 중인 revision
  - `base_model_revision`: 로컬 update가 계산된 기준 revision

## 현재 diagonal scale adapter 의미

현재 runtime 적용식은 아래와 같다.

```text
x' = normalize(x ⊙ s)
```

- `x`: backbone embedding
- `s`: `dimension_scales`
- `dimension_deltas`: `s`에 더해질 변화량

즉 현재 update는 특정 prototype 좌표로 직접 끌어당기는 값이 아니라, 전체 임베딩 공간의 차원별 비율을 전역적으로 재조정하는 값이다.
