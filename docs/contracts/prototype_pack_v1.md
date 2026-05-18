# PrototypePack v1

## 1. 목적

`PrototypePack`은 로컬 `agent`와 실험층이 사용하는
배포용 semantic 기준 객체다.

이 객체는 관리자 라벨 데이터셋인 `LabeledQuerySet`을 직접 담지 않고,
특정 `embedding model + translation model + mapping rule` 조합으로부터 계산된
카테고리별 대표 벡터만 담는다.

즉 `PrototypePack`은:

1. 원본 라벨 데이터셋이 아니다.
2. 개별 query 임베딩 모음이 아니다.
3. 또래 기준인 `NormPack`이 아니다.
4. FL 모델 파라미터가 아니다.
5. classifier-first v1에서는 bootstrap/comparison에 쓰는 semantic layer 산출물이다.

---

## 2. v1 설계 방향

`PrototypePack v1`은 prototype scoring comparison rail과 classifier bootstrap을
안정적으로 닫기 위한 최소 계약이다.

v1 원칙:

1. `main_server` 또는 별도 빌드 경로가 생성하고 배포한다.
2. 로컬 `agent`는 이를 내려받아 bootstrap/comparison artifact로 사용한다.
3. `LabeledQuerySet` 원문이나 예시 query는 포함하지 않는다.
4. 어떤 임베딩 공간에서 생성됐는지 추적 가능한 메타데이터를 반드시 가진다.
5. 향후 centroid 외 다른 prototype 방식으로 자연스럽게 확장 가능해야 한다.

현재 v1 classifier-first와의 관계:

1. 라벨된 데이터셋은 prototype build뿐 아니라 supervised classifier seed와
   validation/calibration split source로도 직접 사용한다.
2. main inference path의 category 판정은 `global classifier`가 맡는다.
3. `PrototypePack`은 classifier bootstrap, semantic reference, comparison/ablation
   artifact로 남는다.

---

## 3. 생성 시점

`PrototypePack`은 매 query마다 생성하지 않는다.
아래 중 하나가 발생할 때 새 버전을 빌드한다.

1. `LabeledQuerySet` 버전이 바뀔 때
2. `mapping_version`이 바뀔 때
3. `embedding_model_id` 또는 revision이 바뀔 때
4. `translation_model_id` 또는 revision이 바뀔 때
5. `build_method`가 바뀔 때

즉 semantic 기준이 달라질 수 있는 요인이 바뀌면,
새 `PrototypePack`을 다시 만들어야 한다.

---

## 4. 역할 경계

### `LabeledQuerySet`이 하는 일

1. 관리자 라벨 query를 보관
2. `raw_label`과 `mapped_label_4`를 보존
3. prototype 생성과 평가의 근거 제공
4. supervised classifier seed와 calibration split의 원천 데이터 제공

### `PrototypePack`이 하는 일

1. 카테고리별 대표 벡터 제공
2. classifier bootstrap과 comparison scoring에 필요한 최소 메타데이터 제공
3. 로컬 `agent`가 내려받아 즉시 사용할 수 있는 semantic reference 형태 제공

### 로컬 `agent`가 하는 일

1. 입력 query 전처리
2. 필요 시 번역
3. 임베딩 생성
4. main path에서는 global classifier로 category evidence 계산
5. comparison/ablation 또는 bootstrap 검증에서는 `PrototypePack` 기준 점수 계산

### 중앙 `main_server`가 하는 일

1. 현재 활성 `PrototypePack` 버전 관리
2. agent에 배포
3. 필요 시 새 pack 발행

중앙은 `PrototypePack`만 배포하며,
개별 query의 의미를 중앙에서 직접 판정하지 않는다.

---

## 5. 포함할 필드

v1은 아래 필드를 기본으로 한다.

### 필수 필드

1. `schema_version`
   - 스키마 버전
   - 예: `prototype_pack.v1`

2. `prototype_version`
   - pack 자체의 버전 식별자
   - 예: `proto_2026_03_26_001`

3. `embedding_model_id`
   - 대표 벡터 생성에 사용한 임베딩 모델 식별자
   - 예: `mixedbread-ai/mxbai-embed-large-v1`

4. `embedding_model_revision`
   - 임베딩 모델 revision 또는 checkpoint 식별값

5. `translation_model_id`
   - 번역을 사용했다면 해당 번역 모델 식별자
   - 번역 없이 바로 임베딩하면 `null`

6. `translation_model_revision`
   - 번역 모델 revision 또는 checkpoint 식별값
   - 번역 없이 바로 임베딩하면 `null`

7. `translation_direction`
   - 번역 방향
   - 예: `kor_Hang->eng_Latn`
   - 번역 없이 바로 임베딩하면 `null`

8. `mapping_version`
   - `raw_label -> mapped_label_4` 변환 규칙 버전

9. `build_method`
   - 카테고리 대표 벡터 생성 방식
   - 예: `mean_centroid_l2_normalized`

10. `distance_metric`
    - scoring 시 사용할 거리 또는 유사도 기준
    - 예: `cosine`

11. `built_at`
    - pack 생성 시각

12. `categories`
    - 카테고리별 prototype 집합
    - v1 카테고리: `anxiety`, `depression`, `suicidal`, `normal`

### `categories` 내부 필드

각 카테고리마다 아래 값을 가진다.

1. `centroid`
   - 해당 카테고리를 대표하는 벡터
   - 개별 query 임베딩이 아니라 집계된 대표 벡터

2. `sample_count`
   - 해당 카테고리 prototype 계산에 반영된 query 수

---

## 6. 포함하지 않을 필드

v1에서는 아래를 포함하지 않는다.

1. 원본 query 텍스트
2. 대표 query 예시
3. 개별 query 임베딩 목록
4. category별 label histogram 상세
5. 평가 metric
6. build log 전체
7. source dataset 파일 경로
8. `LabeledQuerySet` 본문 전체

이유:

1. runtime에서 필요한 최소 정보만 유지하기 위해
2. 민감한 query 원문이 배포 객체에 섞이지 않게 하기 위해
3. scoring 기준과 생성 근거를 역할상 분리하기 위해
4. `PrototypePack`을 가볍고 배포 가능한 객체로 유지하기 위해

---

## 7. 왜 `LabeledQuerySet`을 직접 넣지 않는가

`PrototypePack`은 배포용 객체이고,
`LabeledQuerySet`은 생성 근거이자 관리용 데이터셋이다.

이 둘을 섞지 않는 이유는 다음과 같다.

1. 원본 query 데이터는 더 민감할 수 있다.
2. agent runtime에는 대표 벡터만 있으면 된다.
3. 생성 근거와 배포 산출물을 분리해야 버전 관리가 쉬워진다.
4. 나중에 동일한 `LabeledQuerySet`으로 다른 `build_method`를 실험하기 쉽다.

따라서:

- `LabeledQuerySet`은 build/eval 경로에서 사용한다.
- `PrototypePack`은 runtime scoring 경로에서 사용한다.

---

## 8. JSON 예시

```json
{
  "schema_version": "prototype_pack.v1",
  "prototype_version": "proto_2026_03_26_001",
  "embedding_model_id": "mixedbread-ai/mxbai-embed-large-v1",
  "embedding_model_revision": "main",
  "translation_model_id": "facebook/nllb-200-distilled-600M",
  "translation_model_revision": "main",
  "translation_direction": "kor_Hang->eng_Latn",
  "mapping_version": "cssrs_to_4cat.v1",
  "build_method": "mean_centroid_l2_normalized",
  "distance_metric": "cosine",
  "built_at": "2026-03-26T13:00:00Z",
  "categories": {
    "anxiety": {
      "centroid": [0.014, -0.227, 0.101, 0.442],
      "sample_count": 214
    },
    "depression": {
      "centroid": [0.088, -0.190, 0.154, 0.401],
      "sample_count": 187
    },
    "suicidal": {
      "centroid": [0.132, -0.311, 0.284, 0.519],
      "sample_count": 96
    },
    "normal": {
      "centroid": [-0.041, 0.083, -0.019, 0.147],
      "sample_count": 402
    }
  }
}
```

주의:

예시의 `centroid` 길이는 설명용으로 축약했다.
실제 구현에서는 임베딩 차원 전체가 들어간다.

---

## 9. 검증 기준

`PrototypePack v1` 계약이 유효하려면 아래를 만족해야 한다.

1. 같은 입력 `LabeledQuerySet`과 같은 build 설정에서 재현 가능하다.
2. `embedding_model_id`, `translation_model_id`, `mapping_version`가 누락되지 않는다.
3. 모든 카테고리에 `centroid`와 `sample_count`가 존재한다.
4. agent scoring 경로에서 바로 사용할 수 있다.
5. `LabeledQuerySet v1`과 자연스럽게 연결된다.

---

## 10. 향후 v2 확장 후보

아래는 v1에 넣지 않고, 필요 시 v2에서 검토한다.

1. `vector_dim`
2. `prototype_confidence`
3. category별 `build_method` override
4. `median` 또는 `medoid` 기반 보조 prototype
5. 대표 query 예시의 privacy-safe 요약
6. 평가 metric 요약

별도 계약에서 다룰 후보:

1. `LabeledQuerySet v1`
2. `PrototypeBuildReport`
3. `PrototypePublishEnvelope`

v1의 목적은
semantic runtime이 어떤 기준 벡터를 쓰는지 흔들림 없이 고정하면서도,
생성 근거와 배포 산출물을 분리해 이후 확장을 쉽게 만드는 것이다.
