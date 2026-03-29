# PrototypeBuildState v1

## 1. 목적

`PrototypeBuildState`는 `PrototypePack`을 정확하게 다시 만들기 위한
빌드용 누적 상태 객체다.

이 객체는 runtime scoring에 직접 배포되지 않고,
신규 labeled data가 순차적으로 들어올 때
전체 원시 데이터를 매번 다시 임베딩하지 않고도
exact incremental update를 가능하게 하는 용도로 사용한다.

즉 `PrototypeBuildState`는:

1. 로컬 `agent` runtime 객체가 아니다.
2. 원문 query 집합이 아니다.
3. `PrototypePack`을 다시 계산하기 위한 누적 통계 상태다.

---

## 2. v1 설계 원칙

1. 카테고리별 `embedding_sum`과 `sample_count`를 유지한다.
2. 최종 centroid는 이 상태에서 평균을 구한 뒤 L2 정규화해 생성한다.
3. 동일한 `embedding backend + model + mapping rule` 조합에서만 incremental update를 허용한다.
4. runtime 배포용 `PrototypePack`과 build용 `PrototypeBuildState`는 분리한다.

---

## 3. 포함 필드

1. `schema_version`
2. `prototype_version`
3. `embedding_backend`
4. `embedding_model_id`
5. `embedding_model_revision`
6. `normalize_embeddings`
7. `task_prefix`
8. `translation_model_id`
9. `translation_model_revision`
10. `translation_direction`
11. `mapping_version`
12. `build_method`
13. `distance_metric`
14. `built_at`
15. `categories`

`categories` 내부:

1. `embedding_sum`
2. `sample_count`

---

## 4. exact incremental update 수식

카테고리 `c`에 대해 기존 누적 상태가 `S_old`, `N_old`,
신규 데이터 임베딩 합이 `S_new`, 개수가 `N_new`이면:

`S_total = S_old + S_new`

`N_total = N_old + N_new`

`mu_c = S_total / N_total`

`centroid_c = mu_c / ||mu_c||_2`

즉 평균 centroid 기준에서는
신규 데이터 임베딩만 추가 계산해도
전체 데이터로 다시 평균을 낸 것과 같은 결과를 얻는다.

---

## 5. full rebuild가 필요한 경우

아래 경우에는 incremental update가 아니라 full rebuild를 수행한다.

1. `embedding_backend` 변경
2. `embedding_model_id` 또는 revision 변경
3. `task_prefix` 변경
4. `normalize_embeddings` 변경
5. `mapping_version` 변경
6. 과거 데이터 삭제 또는 재라벨
7. 전처리/번역 경로 변경
