# Scripts Guide

이 디렉터리는 데이터 준비, `PrototypePack` 관리, 분류 baseline, prototype 전략 비교, threshold sweep, federated simulation용 스크립트를 모아 둔 곳이다.

## 구조 인덱스

- `scripts/datasets/*.py`: 직접 실행하는 dataset CLI entrypoint
- `scripts/datasets/lib/`: dataset CLI가 공유하는 재사용 함수
- `scripts/prototypes/*.py`: 직접 실행하는 prototype CLI entrypoint
- `scripts/prototypes/io.py`, `evaluation.py`, `seeding.py`: prototype CLI가 직접 쓰는 활성 helper
- `scripts/experiments/*.py`: 직접 실행하는 experiment Hydra entrypoint
- `scripts/experiments/federated_simulation/`: federated simulation 전용 조합/덤프/sharding
- `scripts/experiments/prototype_strategy/`: prototype 전략 비교 실험 전용 모듈
- `scripts/conf/dataset`, `embedding`, `runtime`, `prototype_builder`, `federated_run_preset`: 재사용 Hydra config group
- `scripts/conf/datasets`, `experiments`, `prototypes`: 각 entrypoint가 읽는 top-level Hydra job config
- `scripts/classification_report.py`, `scripts/run_artifacts.py`: 여러 스크립트가 공유하는 공통 helper

원칙:

- production/runtime에서 재사용해야 하는 코어는 `shared`, `agent`, `main_server`로 올린다.
- `scripts`는 그 코어를 조합해서 실행하는 entrypoint와 experiment 전용 helper만 둔다.
- `scripts/prototypes/lib` 같은 compatibility indirection은 유지하지 않는다.

현재 활성 실행 방식은 **Hydra config group + override** 기준이다.  
즉 예전처럼 `--dataset`, `--runtime-profile`를 길게 넘기기보다 아래처럼 실행한다.

```bash
uv run python <script>.py dataset=ourafla runtime=gpu_online
```

## 사전 준비

```bash
uv sync --extra dev --extra experiments
```

## 공통 실행 규칙

### 기본값

대부분의 활성 스크립트는 아래 기본 group를 사용한다.

- `dataset=ourafla`
- `embedding=mxbai`
- `runtime=gpu_online`

즉 별도 override 없이 실행하면:

- ourafla split 경로 사용
- `mixedbread-ai/mxbai-embed-large-v1` 사용
- `device=cuda`
- `local_files_only=false`

### 실행 산출물 위치

- 재사용 가능한 데이터 산출물은 `data/processed/` 아래에 둔다.
- 서버/에이전트 runtime 저장소는 `main_server/state/`, `agent/state/` 아래에 둔다.
- 실험 실행 1회의 결과는 `runs/<job>/<run_id>/` 아래에 둔다.

로 동작한다.

### `gpu_online`이란?

- `runtime=gpu_online`
  - `device=cuda`
  - `local_files_only=false`
  - GPU를 기본으로 쓰되, 캐시에 모델이 없으면 다운로드도 허용한다.

- `runtime=gpu_local`
  - `device=cuda`
  - `local_files_only=true`
  - GPU를 쓰되, 로컬 캐시에 이미 있는 파일만 사용한다.

- `runtime=cpu_local`
  - `device=cpu`
  - `local_files_only=true`

- `runtime=auto_online`
  - `device=auto`
  - `local_files_only=false`

### 자주 쓰는 override 예시

같은 backend 계열에서 모델만 바꾸고 싶다면:

```bash
uv run python <script>.py \
  dataset=ourafla \
  embedding=mxbai \
  embedding.model_id=intfloat/e5-large-v2 \
  embedding.revision=main
```

빠른 smoke/debug용 해시 임베딩으로 바꾸고 싶다면:

```bash
uv run python <script>.py \
  dataset=ourafla \
  embedding=hash_debug \
  runtime=cpu_local \
  embedding.hash_dim=64
```

최종 조합된 Hydra 설정만 보고 싶다면:

```bash
uv run python <script>.py --cfg job
```

---

## 1. Dataset 준비

### classifier-first 자산 필드

`scripts/conf/dataset/*.yaml`은 classifier-first 실험의 데이터 자산 source of truth다.
현재 활성 필드 의미는 아래로 고정한다.

- `train_jsonl`: supervised classifier seed 학습용 labeled train split
- `validation_jsonl`: labeled validation split
- `test_jsonl`: labeled held-out test split
- `prototype_input_jsonl`: prototype bootstrap/comparison artifact 생성 입력
- `query_dev_jsonl`: 나중에 추가할 실제 query 도메인용 소량 labeled dev set
- `query_calibration_jsonl`: threshold/calibration 용 labeled query set
- `unlabeled_query_pool_jsonl`: SSL/FixMatch류에 넣을 unlabeled 일반 query pool

현재 `ourafla`는 labeled split과 prototype 입력만 채워져 있고,
query-domain 자산은 아직 `null` placeholder로 둔다. 실제 query 세트가 준비되면
해당 dataset alias에서 위 세 필드만 채우면 된다.

### 가장 짧은 경로

dataset Hydra group 기준으로 `download -> map -> split -> prototype`를 한 번에 실행한다.

```bash
uv run python scripts/datasets/run_dataset_pipeline.py
```

다른 dataset alias나 stage 제한을 쓰려면:

```bash
uv run python scripts/datasets/run_dataset_pipeline.py \
  dataset=ourafla \
  only_stages=[download,map,split]
```

offline cache만 쓰고 싶다면 prototype stage runtime만 이렇게 바뀐다.

```bash
uv run python scripts/datasets/run_dataset_pipeline.py \
  runtime=gpu_local
```

`scripts/conf/dataset/*.yaml`에 등록된 dataset alias만 보고 싶다면:

```bash
uv run python scripts/datasets/run_dataset_pipeline.py list_datasets=true
```

### 단계별 수동 실행

1. raw CSV 다운로드

```bash
uv run python scripts/datasets/download_dataset.py \
  --dataset-id ourafla/Mental-Health_Text-Classification_Dataset \
  --split train \
  --data-file mental_heath_unbanlanced.csv
```

2. raw CSV -> labeled JSONL 변환

```bash
uv run python scripts/datasets/build_labeled_query_set.py \
  --raw-csv data/raw/ourafla_train.csv \
  --mapping-config data/mappings/ourafla_to_4cat.v1.toml
```

3. labeled JSONL -> stratified train/validation split

```bash
uv run python scripts/datasets/split_labeled_query_set.py \
  --input-jsonl data/processed/labeled_query_sets/ourafla_mental_health_text_classification.v1.jsonl \
  --split-name ourafla_train_split.v1 \
  --validation-ratio 0.1
```

---

## 2. PrototypePack 생성

train split으로 production용 `PrototypePack`을 만든다.

기본 실행:

```bash
uv run python scripts/prototypes/seed_prototypes.py
```

`kmeans` multi-prototype로 바꾸려면:

```bash
uv run python scripts/prototypes/seed_prototypes.py \
  prototype_builder=kmeans \
  prototype_builder.candidate_ks=[2,3,4,5]
```

명시적 GPU local 실행:

```bash
uv run python scripts/prototypes/seed_prototypes.py runtime=gpu_local
```

출력 기본 경로:

- `data/processed/prototype_packs/`
- `data/processed/prototype_build_states/`

설명:

- 기본 builder는 `single`이다.
- `prototype_builder=kmeans`로 바꾸면 category마다 여러 prototype을 저장할 수 있다.
- exact incremental용 `prototype_build_state`는 현재 single builder에서만 생성된다.

---

## 3. PrototypePack baseline 평가

생성된 `PrototypePack`을 validation/test에 평가한다.

```bash
uv run python scripts/prototypes/evaluate_prototype_pack.py \
  prototype_pack=data/processed/prototype_packs/<prototype_version>.json
```

offline cache만 쓰고 싶다면:

```bash
uv run python scripts/prototypes/evaluate_prototype_pack.py \
  prototype_pack=data/processed/prototype_packs/<prototype_version>.json \
  runtime=gpu_local
```

출력 기본 경로:

- `runs/prototype_pack_eval/<run_id>/reports/`

---

## 4. Prototype 전략 비교 실험

`single / kmeans / dbscan` 세 전략을 같은 train/validation/test에서 비교한다.

기본 실행:

```bash
uv run python scripts/experiments/prototype_strategy_experiment.py
```

`single` centroid만 보고 싶다면:

```bash
uv run python scripts/experiments/prototype_strategy_experiment.py \
  strategy.name=single
```

`kmeans`만 `k=2`로 보고 싶다면:

```bash
uv run python scripts/experiments/prototype_strategy_experiment.py \
  strategy.name=kmeans \
  strategy.kmeans_candidate_ks=[2]
```

`hash_debug` smoke 예시:

```bash
uv run python scripts/experiments/prototype_strategy_experiment.py \
  embedding=hash_debug \
  runtime=cpu_local \
  embedding.hash_dim=64
```

산출물:

- `summary.json`
- `validation/`
- `strategies/`
- `projections/`
- `projections/train_{pca|umap}.{strategy}_prototypes.jsonl`
- `projections/train_{pca|umap}.{strategy|label}_visual_centers.jsonl`
- `logs/`

설명:

- projection 그림은 `train` split 기준이다.
- 전략 선택은 `validation` 기준이다.
- 최종 요약 accuracy는 `test` 기준이다.
- `*_prototypes.jsonl`은 runtime scoring에 쓰는 prototype을 2D로 투영한 점이다.
- `*_visual_centers.jsonl`은 실제 2D 그림 위 점들의 산술평균 중심이다.
- `kmeans`일 때 visual center는 원공간 k-means cluster 할당을 유지한 뒤, 각 cluster의 2D 점 평균으로 계산한다.

출력 기본 경로:

- `runs/prototype_strategy/<run_id>/`

---

## 5. Prototype threshold sweep

선택한 전략 위에서 static pseudo-label threshold policy를 비교한다.
현재 classifier-first 연구선의 주 경로라기보다 prototype comparison용 실험이다.

기본 포함 policy:

- `fixmatch_fixed_confidence`
  - FixMatch 논문식 전역 confidence cutoff 후보 비교
- `validation_target_error_confidence`
  - validation target error를 만족하는 최대 coverage cutoff 탐색
- `classwise_static_confidence`
  - predicted label별 정적 confidence cutoff를 validation에서 따로 fit

기본 실행:

```bash
uv run python scripts/experiments/prototype_threshold_sweep.py
```

`single` 전략으로 바꾸려면:

```bash
uv run python scripts/experiments/prototype_threshold_sweep.py \
  strategy.name=single
```

smoke 예시:

```bash
uv run python scripts/experiments/prototype_threshold_sweep.py \
  embedding=hash_debug \
  runtime=cpu_local \
  strategy.name=single \
  threshold_policies[0].thresholds=[0.8,0.95] \
  threshold_policies[1].target_errors=[0.05,0.1] \
  threshold_policies[2].target_errors=[0.1]
```

출력 기본 경로:

- `runs/prototype_threshold_sweep/<run_id>/`

---

## 6. Softmax classifier head baseline

고정 임베딩 위에 linear classifier head를 학습한다.
현재 classifier-first 연구선의 supervised seed baseline으로 본다.
라벨된 데이터셋은 prototype build 전용이 아니라 이 baseline을 직접 학습하는
source로도 사용한다.

```bash
uv run python scripts/experiments/train_softmax_classifier.py
```

선택 기준을 test로 바꾸려면:

```bash
uv run python scripts/experiments/train_softmax_classifier.py \
  selection_set=test
```

출력 기본 경로:

- 평가 리포트: `runs/train_classifier/<classifier_version>/reports/`
- 모델 아티팩트: `data/processed/classifier_heads/`

---

## 7. Federated simulation smoke

bootstrap train subset으로 prototype semantic artifact를 만들고, 나머지 train을
client shard로 나눠 `pseudo-label -> local update -> aggregation -> republish`
루프를 검증한다. classifier_head family를 쓰는 경우 prototype centroid는 초기
classifier bootstrap에도 사용할 수 있다.

기본 smoke 실행:

```bash
uv run python scripts/experiments/run_federated_simulation.py
```

주의:

- 현재 Hydra 기본 preset은 `prototype_pseudo_label_v1`다.
- 현재 v1 권장 연구선은 classifier-first이므로, classifier 실험에서는
  `training_algorithm_profile=fixmatch_v1` 또는 classifier-head 관련 override를
  명시적으로 주는 편이 맞다.

라운드와 client 수를 바꾸려면:

```bash
uv run python scripts/experiments/run_federated_simulation.py \
  federated_run_preset=standard \
  federated_run_preset.client_count=4 \
  federated_run_preset.rounds=3
```

FL 재빌드도 `kmeans` multi-prototype로 보려면:

```bash
uv run python scripts/experiments/run_federated_simulation.py \
  federated_run_preset=standard \
  federated_run_preset.client_count=8 \
  federated_run_preset.rounds=3 \
  prototype_builder=kmeans \
  prototype_builder.candidate_ks=[2]
```

빠른 해시 smoke:

```bash
uv run python scripts/experiments/run_federated_simulation.py \
  embedding=hash_debug \
  runtime=cpu_local \
  federated_run_preset=smoke
```

client shard 분배 비율이나 scoring policy 같은 세부 전략을 바꾸려면:

```bash
uv run python scripts/experiments/run_federated_simulation.py \
  shard_policy.dominant_ratio=0.6 \
  training_task.objective.score_policy_name=topk_mean_cosine \
  training_task.objective.score_top_k=2 \
  validation.score_policy_name=topk_mean_cosine \
  validation.score_top_k=2
```

prototype 재빌드 메타데이터나 dump 경로도 leaf override로 바꿀 수 있다:

```bash
uv run python scripts/experiments/run_federated_simulation.py \
  prototype_rebuild.mapping_version=custom_mapping.v1 \
  diagnostics.dump_dir_name=custom_selection_dumps
```

출력 예시:

- `federated_run_preset=smoke`: `runs/federated_simulation_smoke/<run_id>/...`
- `federated_run_preset=standard`: `runs/federated_simulation/<run_id>/...`

현재 수준:

- 실제 `train/validation` JSONL을 사용한다.
- 여러 client shard를 가상 agent처럼 돌린다.
- 각 agent는 pseudo-label 선별과 local update 생성을 수행한다.
- `selection_dumps/`에는 row별 `confidence`, `margin`,
  `threshold_accepted`, `selected_by_cap`, `selection_stage`가 저장된다.
- 중앙은 update를 집계해 새 `model_revision + prototype_version` pair를 발행한다.
- prototype 재빌드는 `prototype_builder` 설정을 그대로 따른다.
- 현재 default smoke는 diagonal scale preset이지만,
  classifier-head + `fixmatch_v1` simulation path도 지원한다.
- `FreeMatch`, `PabLO`는 아직 direct preset이 없으므로 classifier-first 확장으로
  추가할 대상이다.

---

## 8. Local demo

현재 상태:

- `scripts/experiments/run_local_demo.py`는 아직 미구현이다.
- 실행하면 종료 메시지만 출력한다.

---

## 참고

실험 결과 요약은 아래 문서를 본다.

- `docs/experiment_results.md`
