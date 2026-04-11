# TraceMind Experiment Results

## 목적

이 문서는 현재 저장소에서 실제로 실행했거나 산출물이 남아 있는 실험 결과를 한곳에 정리한 문서다.
코드 구조나 계약 문서가 아니라, baseline 비교와 의사결정에 필요한 숫자와 해석을 모은다.
경로 표기는 현재 canonical layout 기준으로 적고, 정리 과정에서 제거된 historical report는
남아 있는 artifact 또는 현재 기준 output root로 표기한다.

중요:

- 이 문서에 적힌 현재 결과는 대부분 `fixed embedding` 또는 시스템 트랙 기준 실험이다.
- 앞으로 논문용 `central LoRA classifier` 결과는 별도 섹션으로 추가한다.
- 즉 현재 숫자를 `논문 fidelity baseline`과 동일하게 읽으면 안 된다.

---

## 1. PrototypePack single-centroid comparison baseline

관련 경로:

- `data/processed/prototype_packs/proto_2026_03_28_163056.json`
- `data/processed/prototype_packs/proto_2026_03_28_163056.manifest.json`
- 평가 리포트 canonical root: `runs/prototype_pack_eval/<run_id>/reports/{validation,test}.json`

설정 요약:

- embedding backend: `transformers_mxbai`
- embedding model: `mixedbread-ai/mxbai-embed-large-v1`
- build method: `mean_centroid_l2_normalized`
- class당 prototype 수: 1

결과:

- validation accuracy: `0.742794`
- test accuracy: `0.711694`

해석:

- single centroid만으로도 comparison baseline은 성립한다.
- `normal`은 비교적 강하고, `depression`과 `suicidal` 사이 혼동이 크다.
- classifier-first 체계에서는 semantic bootstrap/comparison 기준점 역할을 한다.

---

## 2. Fixed Embedding + Softmax Classifier Head Baseline

관련 경로:

- `data/processed/classifier_heads/clf_2026_03_28_172617.pt`
- `data/processed/classifier_heads/clf_2026_03_28_172617.manifest.json`
- 평가 리포트 canonical root: `runs/train_classifier/<classifier_version>/reports/report.json`

설정 요약:

- embedding backend: `transformers_mxbai`
- embedding model: `mixedbread-ai/mxbai-embed-large-v1`
- device: `cuda`
- classifier: linear head + softmax
- epoch: `12`

결과:

- validation accuracy: `0.789155`
- test accuracy: `0.734879`

해석:

- 현재 저장소에 남아 있는 static baseline 중 가장 높은 정확도다.
- 현재 v1 계획의 classifier-first 연구선에서 가장 직접적인 supervised seed 기준선이다.
- 이후 논문 트랙의 LoRA scaffold에서 FixMatch -> FreeMatch -> PabLO를 올릴 때
  가장 먼저 비교할 reference baseline이다.

---

## 3. Prototype Strategy Comparison

관련 경로:

- `runs/prototype_strategy/20260329T152634Z/summary.json`
- `runs/prototype_strategy/20260329T152634Z/projections/train_pca.png`
- `runs/prototype_strategy/20260329T152634Z/projections/train_umap.png`

비교 전략:

- `single`
- `kmeans`
- `dbscan`

validation 결과:

- `single`: `0.7427937915742794`
- `kmeans`: `0.759322717194114`
- `dbscan`: `0.7397702076194316`

선택 전략:

- `kmeans`

선택 전략 test accuracy:

- `0.7096774193548387`

세부 해석:

- `kmeans`가 validation 기준 가장 좋았다.
- 이 실험에서는 모든 클래스에서 `k=2`가 선택되어 총 prototype 수가 `8`개였다.
- `dbscan`은 총 prototype 수가 `19`개로 많이 늘었지만 정확도 이득이 없었다.
- 특히 `depression`이 과하게 쪼개지는 경향이 있어 현재 데이터와 설정에선 불리했다.

현재 결론:

- multi-prototype이 필요하다면 첫 선택은 `kmeans`
- `dbscan`은 비교군 가치는 있지만 현재 기본 전략으로는 부적합

---

## 4. Threshold Sweep

threshold sweep은 pseudo-label 채택 기준을 바꿔 가며
`accepted_ratio`, `accepted_accuracy`, `accepted_correct_ratio`를 비교하는 실험이다.

용어:

- `accepted_ratio`
  - 전체 샘플 중 pseudo-label로 채택된 비율
- `accepted_accuracy`
  - 채택된 샘플만 놓고 봤을 때 실제 라벨과 맞는 비율
- `accepted_correct_ratio`
  - 전체 샘플 중 “채택도 되었고 정답이기도 한” 비율

### 4-1. Coverage-first 성향 결과

#### kmeans

관련 경로:

- `runs/prototype_threshold_sweep/20260329T155803Z/summary.json`

선택 threshold:

- `confidence >= 0.60`
- `margin >= 0.00`

결과:

- validation
  - `accepted_count = 4598`
  - `accepted_ratio = 0.926829268292683`
  - `accepted_accuracy = 0.7461939973901697`
  - `accepted_correct_ratio = 0.6915944366055231`
- test
  - `accepted_count = 966`
  - `accepted_ratio = 0.9737903225806451`
  - `accepted_accuracy = 0.7039337474120083`
  - `accepted_correct_ratio = 0.6854838709677419`

해석:

- 거의 대부분을 채택한다.
- coverage는 크지만 purity는 상대적으로 낮다.
- 초기 self-training을 빨리 열고 싶을 때는 유리하지만 drift 위험이 더 크다.

#### single

관련 경로:

- `runs/prototype_threshold_sweep/20260329T162700Z/summary.json`

선택 threshold:

- `confidence >= 0.60`
- `margin >= 0.00`

결과:

- validation
  - `accepted_count = 4532`
  - `accepted_ratio = 0.9135254988913526`
  - `accepted_accuracy = 0.7255075022065314`
  - `accepted_correct_ratio = 0.6627696029026406`
- test
  - `accepted_count = 961`
  - `accepted_ratio = 0.96875`
  - `accepted_accuracy = 0.7023933402705516`
  - `accepted_correct_ratio = 0.6804435483870968`

해석:

- coverage-first 기준에서도 `single`은 `kmeans`보다 더 약했다.
- 따라서 coverage를 크게 우선하는 정책에서는 `kmeans` 쪽이 더 낫다.

### 4-2. Conservative 성향 결과

현재 기본 selection policy는 다음 성격을 가진다.

- `minimum_accepted_ratio >= 0.5`를 만족하는 후보를 우선 고려
- 그 범위 안에서 `accepted_accuracy`를 우선
- 그다음 `accepted_correct_ratio`, `accepted_ratio` 순

즉 너무 적은 샘플만 받는 극단적 threshold는 제외하면서,
coverage보다 precision을 우선하는 절충형 보수 정책이다.

#### kmeans

관련 경로:

- `runs/prototype_threshold_sweep/20260329T163955Z/summary.json`

선택 threshold:

- `confidence >= 0.60`
- `margin >= 0.02`

결과:

- validation
  - `accepted_count = 2492`
  - `accepted_ratio = 0.50231808103205`
  - `accepted_accuracy = 0.8808186195826645`
  - `accepted_correct_ratio = 0.4424511187260633`
- test
  - `accepted_count = 462`
  - `accepted_ratio = 0.4657258064516129`
  - `accepted_accuracy = 0.8744588744588745`
  - `accepted_correct_ratio = 0.40725806451612906`

해석:

- 채택량은 줄지만 pseudo-label purity가 크게 올라간다.
- non-IID 환경에서 초기 drift를 줄이려는 목적과 잘 맞는다.

#### single

관련 경로:

- `runs/prototype_threshold_sweep/20260329T164225Z/summary.json`

선택 threshold:

- `confidence >= 0.60`
- `margin >= 0.02`

결과:

- validation
  - `accepted_count = 2694`
  - `accepted_ratio = 0.5430356782906672`
  - `accepted_accuracy = 0.844097995545657`
  - `accepted_correct_ratio = 0.45837532755492844`
- test
  - `accepted_count = 507`
  - `accepted_ratio = 0.5110887096774194`
  - `accepted_accuracy = 0.8579881656804734`
  - `accepted_correct_ratio = 0.43850806451612906`

해석:

- `single`은 `kmeans`보다 더 많이 채택한다.
- 하지만 purity는 `kmeans`보다 낮다.
- 즉 `single`은 volume 쪽, `kmeans`는 precision 쪽 성격이 더 강하다.

### 4-3. Threshold Sweep 종합 결론

1. coverage-first로 가면 `kmeans + 0.60 / 0.00`이 가장 자연스럽다.
2. 보수적으로 가면 `0.60 / 0.02`가 안정적이다.
3. 현재 사용자 결정은 보수적 방향이므로, 기본 추천 조합은 `kmeans + 0.60 / 0.02`다.
4. 단, pseudo-label 물량을 조금 더 늘리고 싶다면 `single + 0.60 / 0.02`도 실험 후보가 될 수 있다.

---

## 5. Smoke / 구조 검증 실험

### 5-1. Prototype strategy smoke

관련 경로:

- `runs/prototype_strategy_smoke/output/<run_id>/summary.json`
- `runs/prototype_strategy/<run_id>/summary.json`

메모:

- 이전 `tmp/prototype_strategy_*` smoke 산출물은 정리 과정에서 제거됐다.

역할:

- 전체 데이터셋을 돌리기 전에 경로와 산출물 구조가 맞는지 확인
- refactor 이후에도 동일 흐름이 깨지지 않는지 확인

핵심:

- 전략 비교 코드 자체는 hash_debug smoke, real backend smoke, refactor smoke를 모두 통과했다.

### 5-2. Threshold sweep smoke

관련 경로:

- `runs/prototype_threshold_sweep/<run_id>/summary.json`

역할:

- grid search 로직과 summary/grid 산출물 형식 검증

### 5-3. Federated simulation smoke

산출물 예시:

- `runs/federated_simulation_smoke/20260331T155147Z/main_server/model_manifests/sim_rev_0000.json`
- `runs/federated_simulation_smoke/20260331T155147Z/main_server/model_manifests/sim_rev_0001.json`
- `runs/federated_simulation_smoke/20260331T155147Z/main_server/prototype_packs/proto_sim_0000.json`
- `runs/federated_simulation_smoke/20260331T155147Z/main_server/prototype_packs/proto_sim_0001.json`
- `runs/federated_simulation_smoke/20260331T155147Z/agents/agent_01/shared_adapter_updates/update_round_0001_08779ed1233e.json`

역할:

- `train -> bootstrap + client shard`
- `pseudo-label -> local update`
- `aggregation -> new model/prototype pair republish`

확인한 사항:

- `sim_rev_0000 -> sim_rev_0001`
- `proto_sim_0000 -> proto_sim_0001`
- client update 파일이 실제로 생성됨
- `round active pair only` 구조가 코드상으로 닫힘

---

## 6. 현재 권장안

현재까지 실험을 종합하면, TraceMind v1의 추천 출발점은 아래다.

1. prototype 전략: `kmeans`
2. prototype 수: 현재 실험 기준 클래스당 `2개`
3. pseudo-label 기본 threshold: `confidence >= 0.60`, `margin >= 0.02`
4. selection policy: `minimum_accepted_ratio >= 0.5`를 만족하는 후보 중 precision 우선

이 조합을 추천하는 이유:

1. 전략 비교에서 `kmeans`가 가장 좋은 validation accuracy를 보였다.
2. conservative threshold 기준에서 `kmeans`가 `single`보다 더 높은 pseudo-label purity를 보였다.
3. 현재 제품 방향이 빠른 확장보다 안정적 학습 시작에 더 가깝다.

---

## 7. 다음 실험 후보

현재 문맥에서 다음으로 자연스러운 후보는 아래 세 가지다.

1. `kmeans + 0.60 / 0.02`를 기본값으로 둔 실제 FL loop 반복 실험
2. `single`과 `kmeans`의 accepted volume 차이가 실제 update 품질 차이로 이어지는지 비교
3. synthetic vector adapter 대신 실제 adapter/LoRA backend로 교체한 뒤 동일 실험 재실행
