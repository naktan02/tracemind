# Scripts Guide

이 디렉터리는 데이터 준비, `PrototypePack` 관리, 중앙집중형 classifier/SSL baseline, prototype 전략 비교, threshold sweep, federated simulation용 스크립트를 모아 둔 곳이다.

현재 실행 순서:

1. seed 단계: `central fixed embedding + classifier`
2. 적응 단계: `query accumulation -> threshold/policy -> LoRA + classifier`
3. 시스템 트랙: winner를 `FL/runtime` 제약에 맞게 translation

즉 `federated_simulation`은 현재도 중요하지만, 논문용 중앙 비교선을 먼저 닫은 뒤에 따라오는 후행 단계로 본다.

## 구조 인덱스

- `scripts/datasets/*.py`: 직접 실행하는 dataset CLI entrypoint
- `scripts/datasets/lib/`: dataset CLI가 공유하는 재사용 함수
- `scripts/prototypes/*.py`: 직접 실행하는 prototype CLI entrypoint
- `scripts/prototypes/io.py`, `evaluation.py`, `seeding.py`: prototype CLI가 직접 쓰는 활성 helper
- `scripts/experiments/*.py`: 직접 실행하는 experiment Hydra entrypoint
- `scripts/experiments/federated_simulation/`: federated simulation 전용 조합/덤프/sharding
- `scripts/experiments/prototype_strategy/`: prototype 전략 비교 실험 전용 모듈
- `scripts/conf/dataset`, `embedding`, `runtime`, `prototype_builder`, `federated_run_preset`: 재사용 Hydra config group
- `scripts/conf/federated_round_runtime`, `federated_training_task`, `federated_validation`: federated simulation이 `training_algorithm_profile`을 runtime/task/validation shape로 번역하는 bridge group
- `scripts/conf/datasets`, `experiments`, `prototypes`: 각 entrypoint가 읽는 top-level Hydra job config
- `scripts/classification_report.py`, `scripts/labeled_query_rows.py`: shared canonical utility를 다시 노출하는 compatibility wrapper
- `scripts/run_artifacts.py`: 여러 스크립트가 공유하는 실행 산출물 경로 helper

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

- `runtime=auto_local`
  - `device=auto`
  - `local_files_only=true`
  - GPU가 보이면 GPU를 쓰고, 없으면 CPU로 fallback한다.
  - smoke나 로컬 캐시 기반 검증에 권장한다.

- `runtime=auto_online`
  - `device=auto`
  - `local_files_only=false`

### GPU Preflight

LoRA나 transformer 실험을 GPU로 돌리기 전에 아래를 먼저 확인한다.

```bash
nvidia-smi
..venv/bin/python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

- sandbox 안에서 GPU가 안 보여도 실제 머신에 GPU가 있을 수 있다.
- GPU 의존 smoke/run은 실제 실행 환경에서 위 두 명령을 먼저 확인한 뒤 시작한다.

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

`scripts/conf/dataset/*.yaml`은 논문 트랙과 시스템 트랙이 공유하는 데이터 자산 source of truth다.
현재 활성 필드 의미는 아래로 고정한다.

- `train_jsonl`: supervised classifier seed 학습용 labeled train split
- `validation_jsonl`: labeled validation split
- `test_jsonl`: labeled held-out test split
- `prototype_input_jsonl`: prototype bootstrap/comparison artifact 생성 입력
- `query_dev_jsonl`: 나중에 추가할 실제 query 도메인용 소량 labeled dev set
- `query_calibration_jsonl`: threshold/calibration 용 labeled query set
- `unlabeled_query_pool_jsonl`: 향후 query-domain SSL 적응 후보에 넣을 unlabeled 일반 query pool

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
현재 classifier-first staged 구조의 보조 실험이며, prototype comparison용으로 유지한다.

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
현재 staged 구조의 `seed baseline`으로 본다.
이 baseline을 먼저 닫은 뒤 query-domain 적응 단계에서만 `LoRA + classifier`를 연다.
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

## 7. Query-domain LoRA adaptation baseline

이 레일은 첫 seed baseline이 아니라 query-domain 적응 단계용 scaffold다.
초기 `fixed embedding + classifier` seed를 먼저 닫고, query가 충분히 쌓인 뒤에만
같은 scaffold 위에 `supervised adaptation`, `pseudo-label self-training`,
`FixMatch`, `R-Drop`, `MixText`를 올리고, 필요 시 `TAPT`를 앞단 preadaptation phase로 둔다.

실행 예시:

```bash
uv run python scripts/experiments/train_lora_classifier.py
```

첫 실행에서 scaffold와 산출물만 빠르게 검증하려면:

```bash
uv run python scripts/experiments/train_lora_classifier.py \
  runtime=gpu_local \
  epochs=1
```

step 로그를 더 자주 보고 싶다면 직접 override를 주거나,
`lora_run_preset=smoke_verbose_e1`를 선택한다.

```bash
uv run python scripts/experiments/train_lora_classifier.py \
  runtime=gpu_local \
  lora_run_preset=smoke_verbose_e1
```

현재 LoRA runner 로그는 `log_every_steps` 간격으로 `running_train_loss`만 출력한다.
즉 진행률, 전체 step, ETA는 아직 기본 로그에 포함하지 않는다. 대략적인 진행량은
`학습 row 수 / train_batch_size`로 epoch당 총 step 수를 계산해서 해석한다.

pseudo-label self-training 실행:

```bash
uv run python scripts/experiments/train_lora_pseudo_label_classifier.py \
  lora_run_preset=smoke_verbose_e1 \
  pseudo_label_algorithm=margin_threshold_v1 \
  query_adaptation_initial_checkpoint.manifest_path=/abs/path/to/initial_lora_seed.manifest.json \
  pseudo_label_jsonl=/abs/path/to/pseudo_label_rows.jsonl
```

USB FixMatch 실행:

```bash
uv run python scripts/experiments/train_lora_fixmatch.py \
  runtime=gpu_local \
  query_ssl_train_source=bootstrap_teacher_split30_2026_04_14 \
  query_adaptation_initial_checkpoint.manifest_path=/abs/path/to/initial_lora_seed.manifest.json \
  lora_run_preset=smoke_verbose_e1
```

첫 진입 bootstrap 실행:

```bash
uv run python scripts/experiments/train_lora_bootstrap_classifier_teacher.py \
  lora_run_preset=smoke_verbose_e1 \
  teacher_unlabeled_jsonl=/abs/path/to/unlabeled_pool.jsonl
```

중요:

- `train_lora_bootstrap_classifier_teacher.py`의 기본 `teacher_unlabeled_jsonl`은 `null`이다.
- 즉 아래 둘 중 하나를 반드시 해야 한다.
  - `teacher_unlabeled_jsonl=...`를 직접 넘긴다.
  - `bootstrap_split.enabled=true`로 켜고 train split 일부를 hidden unlabeled pool로 자동 분리한다.
- 경로 override를 직접 길게 넘길 수도 있지만, 지금은 아래처럼 preset group 선택으로 줄일 수 있다.
- 재사용 split은 `bootstrap_teacher_source=<preset>`과 `lora_train_source=<preset>`으로 고른다.
- student 학습 배치/epoch/log 간격은 `lora_run_preset=<preset>`으로 고른다.
- teacher bootstrap selection rule과 이후 self-training provenance preset은 `pseudo_label_algorithm=<preset>`으로 고른다.
- 적응 단계 initial checkpoint source of truth는 `query_adaptation_initial_checkpoint=<preset>`이다.
- LoRA manifest를 넘기면 adapter+classifier를 함께 warm-start하고,
  fixed classifier manifest를 넘기면 classifier head만 warm-start한다.
- FixMatch consistency baseline의 unlabeled source는 `query_ssl_train_source=<preset>`으로 고른다.
- FixMatch method hyperparameter source of truth는 `query_ssl_method=<preset>`이다.
- FixMatch NLP strong view 생성/caching source of truth는 `query_ssl_augmenter=<preset>`이다.

bootstrap split을 이미 만들어 둔 경우의 짧은 실행 예시:

```bash
uv run python scripts/experiments/train_lora_bootstrap_classifier_teacher.py \
  runtime=gpu_local \
  bootstrap_teacher_source=bootstrap_teacher_split30_2026_04_14 \
  pseudo_label_algorithm=fixed_confidence_095 \
  lora_run_preset=smoke_verbose_e1 \
  bootstrap_version=bootstrap_teacher_split30_2026_04_14_rerun \
  teacher_classifier_version=fixed_teacher_split30_2026_04_14_rerun \
  trainer_version=lora_student_bootstrap_split30_2026_04_14_rerun
```

같은 split 기준 supervised LoRA baseline 비교:

```bash
uv run python scripts/experiments/train_lora_classifier.py \
  runtime=gpu_local \
  lora_train_source=bootstrap_teacher_split30_2026_04_14 \
  lora_run_preset=smoke_verbose_e1 \
  trainer_version=supervised_seed_split30_2026_04_14_rerun
```

복사해서 바로 쓰는 현재 warm-start 재실행 레시피:

- labeled subset:
  `data/processed/query_ssl_smoke_subsets/2026_04_20_warmstart/seed_train_stratified_4096.jsonl`
- unlabeled subset:
  `data/processed/query_ssl_smoke_subsets/2026_04_20_warmstart/unlabeled_pool_stratified_1024.jsonl`
- supervised seed manifest:
  `data/processed/lora_classifier_heads/stage3_supervised_seed4096_2026_04_20.manifest.json`

실행 전에는 stale process와 GPU를 먼저 확인한다.

```bash
ps aux | rg 'train_lora_classifier|train_lora_bootstrap_classifier_teacher|train_lora_fixmatch'
nvidia-smi
..venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
```

1. supervised seed LoRA baseline

```bash
uv run python scripts/experiments/train_lora_classifier.py \
  runtime=gpu_online \
  train_jsonl=data/processed/query_ssl_smoke_subsets/2026_04_20_warmstart/seed_train_stratified_4096.jsonl \
  lora_run_preset=default \
  trainer_version=stage3_supervised_seed4096_2026_04_20
```

2. pseudo-label baseline
   - teacher는 기존 canonical fixed classifier seed를 그대로 재사용한다.
   - student는 위 supervised LoRA seed manifest에서 warm-start한다.

```bash
uv run python scripts/experiments/train_lora_bootstrap_classifier_teacher.py \
  runtime=gpu_online \
  teacher_train_jsonl=data/processed/query_ssl_smoke_subsets/2026_04_20_warmstart/seed_train_stratified_4096.jsonl \
  teacher_unlabeled_jsonl=data/processed/query_ssl_smoke_subsets/2026_04_20_warmstart/unlabeled_pool_stratified_1024.jsonl \
  pseudo_label_algorithm=fixed_confidence_095 \
  query_adaptation_initial_checkpoint=required \
  query_adaptation_initial_checkpoint.manifest_path=data/processed/lora_classifier_heads/stage3_supervised_seed4096_2026_04_20.manifest.json \
  lora_run_preset=default \
  bootstrap_version=stage3_pseudolabel_seed4096_unl1024_2026_04_20 \
  trainer_version=stage3_pseudolabel_seed4096_unl1024_2026_04_20
```

3. FixMatch baseline
   - 같은 supervised LoRA seed manifest에서 warm-start한다.
   - strict USB형 `text + aug_0 + aug_1` backtranslation cache를 생성한 뒤 학습한다.
   - GPU headroom을 위해 학습 batch를 낮춘 보수적 실행값을 기본으로 둔다.
   - 이미 만든 augmentation cache를 다시 쓰려면 `query_ssl_augmenter.batch_size=32`를 유지한다.

```bash
uv run python scripts/experiments/train_lora_fixmatch.py \
  runtime=gpu_online \
  train_jsonl=data/processed/query_ssl_smoke_subsets/2026_04_20_warmstart/seed_train_stratified_4096.jsonl \
  unlabeled_jsonl=data/processed/query_ssl_smoke_subsets/2026_04_20_warmstart/unlabeled_pool_stratified_1024.jsonl \
  query_adaptation_initial_checkpoint=required \
  query_adaptation_initial_checkpoint.manifest_path=data/processed/lora_classifier_heads/stage3_supervised_seed4096_2026_04_20.manifest.json \
  query_ssl_augmenter=backtranslation_nllb_en_de_fr_usb_v1 \
  query_ssl_augmenter.batch_size=32 \
  lora_run_preset=default \
  train_batch_size=8 \
  eval_batch_size=16 \
  log_every_steps=20 \
  trainer_version=stage3_fixmatch_seed4096_unl1024_2026_04_20_b8
```

리포트 경로:

- supervised: `runs/train_lora_classifier/stage3_supervised_seed4096_2026_04_20/reports/report.json`
- pseudo-label: `runs/train_lora_pseudo_label_classifier/stage3_pseudolabel_seed4096_unl1024_2026_04_20/reports/report.json`
- FixMatch: `runs/train_lora_fixmatch/stage3_fixmatch_seed4096_unl1024_2026_04_20_b8/reports/report.json`

현재 제공하는 preset:

- `bootstrap_teacher_source=dataset_default`
  - dataset alias가 가리키는 기본 teacher 입력 사용
- `bootstrap_teacher_source=bootstrap_teacher_split30_2026_04_14`
  - 이미 만든 split30 bootstrap 산출물 재사용
- `lora_train_source=dataset_train`
  - dataset alias의 기본 train split 사용
- `lora_train_source=bootstrap_teacher_split30_2026_04_14`
  - split30 teacher seed train 재사용
- `query_ssl_train_source=dataset_default`
  - dataset alias의 labeled train + unlabeled query pool 경로 사용
  - 현재 `ourafla`처럼 `dataset.unlabeled_query_pool_jsonl=null`인 alias에서는 직접 override가 필요하다
- `query_ssl_train_source=bootstrap_teacher_split30_2026_04_14`
  - split30 labeled train / hidden unlabeled pool 재사용
- `lora_run_preset=default`
  - `train_batch_size=16`, `eval_batch_size=32`, `epochs=5`, `log_every_steps=100`
- `lora_run_preset=smoke_verbose_e1`
  - `train_batch_size=16`, `eval_batch_size=32`, `epochs=1`, `log_every_steps=20`
- `pseudo_label_algorithm=margin_threshold_v1`
  - 현재 bootstrap baseline
  - `confidence_threshold=0.6`, `margin_threshold=0.02`
- `pseudo_label_algorithm=fixed_confidence_095`
  - confidence-only 비교선
  - `confidence_threshold=0.95`, `margin_threshold=0.0`
- `query_ssl_method=fixmatch_usb_v1`
  - USB FixMatch core baseline
  - `temperature=0.5`, `p_cutoff=0.95`, `hard_label=true`, `lambda_u=1.0`, `supervised_loss_weight=1.0`
  - 현재 USB semilearn 경로와 동일하게 pseudo label target 생성에서는 `temperature`가 실제로 쓰이지 않는다
  - `query_ssl_method.supervised_loss_weight=0.0`으로 override하면 warm-start checkpoint 위 `unlabeled-only` ablation을 돌릴 수 있다
- `query_adaptation_initial_checkpoint=none`
  - supervised LoRA seed baseline용 fresh-start
- `query_adaptation_initial_checkpoint=canonical_fixed_classifier_seed`
  - canonical fixed classifier seed manifest를 읽어 classifier head 초기값을 재사용한다
- `query_adaptation_initial_checkpoint=required`
  - warm-start artifact를 명시적으로 넘겨야 하는 strict preset
- `query_ssl_augmenter=backtranslation_nllb_en_de_fr_usb_v1`
  - strict USB형 NLP input baseline
  - `weak=text`, `strong=random(aug_0, aug_1)`
  - `aug_0`, `aug_1`는 `EN->DE->EN`, `EN->FR->EN` backtranslation으로 자동 생성 후 cache한다
  - 기본 `torch_dtype=auto`이며, CUDA에서는 `float16`, CPU에서는 `float32`로 해석한다
- `query_ssl_augmenter=precomputed_usb_candidates_v1`
  - 이미 `aug_0`, `aug_1`가 들어 있는 JSONL을 그대로 소비한다

hidden unlabeled split을 지금 바로 자동 생성하려면:

```bash
uv run python scripts/experiments/train_lora_bootstrap_classifier_teacher.py \
  runtime=gpu_local \
  lora_run_preset=smoke_verbose_e1 \
  bootstrap_split.enabled=true \
  bootstrap_split.unlabeled_ratio=0.3 \
  bootstrap_version=bootstrap_teacher_split30_2026_04_14
```

bootstrap runner에서도 student LoRA 단계 로그를 더 자주 보려면
같은 `lora_run_preset=smoke_verbose_e1`를 쓰면 된다.

```bash
uv run python scripts/experiments/train_lora_bootstrap_classifier_teacher.py \
  runtime=gpu_local \
  lora_run_preset=smoke_verbose_e1 \
  bootstrap_split.enabled=true \
  bootstrap_split.unlabeled_ratio=0.3 \
  epochs=1
```

canonical 경로:

- CLI entrypoint: `scripts/experiments/train_lora_classifier.py`
- scripts baseline runner: `scripts/experiments/lora_classifier/runner.py`
- FixMatch entrypoint: `scripts/experiments/train_lora_fixmatch.py`
- Query SSL consistency runner: `scripts/experiments/lora_classifier/query_ssl/consistency_runner.py`
- agent 학습 코어: `agent/src/services/training/query_adaptation/`
- USB FixMatch core mapping: `agent/src/services/training/query_adaptation/algorithms/fixmatch.py`
- teacher bootstrap entrypoint: `scripts/experiments/train_lora_bootstrap_classifier_teacher.py`
- teacher bootstrap helper: `scripts/experiments/lora_classifier/bootstrap_runner.py`
- pseudo-label entrypoint: `scripts/experiments/train_lora_pseudo_label_classifier.py`
- pseudo-label helper: `scripts/experiments/lora_classifier/pseudo_label_runner.py`

주요 산출물:

- `runs/train_lora_classifier/<run_id>/reports/report.json`
- `data/processed/lora_adapters/<run_id>/`
- `data/processed/lora_classifier_heads/<run_id>.pt`

주의:

- 이 실험은 `peft`가 필요하므로 `experiments` extra 설치가 필요하다.
- 기본 LoRA target은 공식 PEFT 문서의 shorthand인 `all-linear`를 사용한다.
- 현재 기본 backbone은 `mixedbread-ai/mxbai-embed-large-v1`이지만,
  실제 최종 적응 비교 표에서는 backbone과 LoRA spec을 먼저 고정한 뒤 비교해야 한다.
- 이 레일을 쓰려면 raw query text가 로컬에 남아 있어야 한다. LoRA 재학습과 weak/strong augmentation은 embedding만으로는 닫히지 않는다.
- agent-local `QueryAdaptationDataset`은 `scripts/experiments/lora_classifier/query_adaptation_io.py`로
  현재 `train_lora_classifier.py`가 읽는 `labeled_query_rows` JSONL shape로 export할 수 있다.
- query-buffer selection 결과는 `agent/src/services/training/query_buffer_selection_diagnostics.py`로
  canonical summary/trace shape를 만들고,
  `scripts/query_buffer_selection_diagnostics.py`로 JSON/JSONL dump를 남길 수 있다.
- adaptation dataset의 canonical provenance (`locale`, `source_type`, `model_revision`)는
  free-form metadata key가 아니라 typed field로 유지한다.
- export 시 JSONL/manifest와 함께 dataset summary JSON도 같이 기록한다.
- concrete helper는 `scripts.experiments.lora_classifier` package re-export가 아니라
  각 모듈을 직접 import하는 것을 기준으로 본다.
- `scripts/experiments/lora_classifier/query_adaptation_runner.py`는
  labeled row를 메모리에서 바로 `run_supervised_lora_baseline()`에 넘기고,
  exported dataset path는 trace/audit 산출물로만 함께 남긴다.
- 첫 bootstrap은 `fixed embedding + classifier` teacher가 unlabeled pool에 pseudo-label을 붙이고,
  그 accepted pseudo-label을 `LoRA + classifier` student가 학습하는 teacher-student 구조다.
- `train_lora_fixmatch.py`는 USB `fixmatch.py::train_step`의 수식 코어를 가져오되,
  USB `AlgorithmBase`가 맡던 iterator/hook orchestration은 현재 TraceMind epoch trainer adapter로 둔다.
- scripts 쪽 실행 껍데기는 `query_ssl/common.py`와 `query_ssl/consistency_runner.py`로 나눠
  family 공통 scaffolding과 알고리즘별 wiring을 분리한다.
- 현재 `FixMatch`는 `text + aug_0 + aug_1` canonical unlabeled shape를 쓰고,
  `query_ssl_augmenter`가 strict USB형 strong candidate를 먼저 준비한다.
- `train_lora_pseudo_label_classifier.py`는 첫 bootstrap 이후 같은-family self-training loop에 쓴다.
- `scripts/experiments/lora_classifier/pseudo_label_runner.py`는
  현재 offline union retraining helper로서 seed labeled rows와 pseudo-labeled rows를 합쳐
  self-training용 combined train JSONL을 남기고, 실제 학습은 메모리 row를 baseline runner에 직접 넘긴다.
- 다만 central canonical 비교 규약은 `seed checkpoint 1회 생성 -> 이후 new accepted query-derived rows only continual adaptation`으로 본다.
- 따라서 pseudo-label / FixMatch / 후속 방법론은 가능한 한 같은
  `query_adaptation_initial_checkpoint`에서 출발하도록 맞춘다.
- `FedMatch`, `FedLGMatch`, `(FL)^2`는 이 중앙 규약으로 흡수하지 않고, 이후 FL stage 비교선으로 남긴다.
- `scripts/experiments/lora_classifier/query_adaptation_multiview_io.py`는
  family-agnostic multiview dataset을 기존 `labeled_query_rows` shape의
  `weak_text` / `strong_text` 필드가 채워진 JSONL로 export하고,
  그 산출물을 `train_lora_fixmatch.py` 입력으로 바로 사용할 수 있다.
- weak/strong augmentation recipe는 이 단계에서 고정하지 않고,
  agent의 `QueryAdaptationMultiviewService`에 pluggable augmenter로 주입한다.
- 로컬 smoke 검증은 `runtime=auto_local`이 권장된다.
- epoch 로그는 기본적으로 `log_every_steps=100` 간격으로 출력된다.
  긴 run을 닫기 전에 step-level loss가 정상인지 먼저 확인하는 용도로 둔다.

## 8. Federated simulation smoke

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
- 현재 `run_federated_simulation`은 시스템 FL 트랙용 entrypoint다.
- 현재 v1 권장 시스템 baseline은 `prototype_pseudo_label_v1` 기반 adapter-first다.
- 단, query-domain 적응의 핵심 비교선은 이 entrypoint가 아니라 별도 중앙 trainer 레일에서 진행한다.

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
- 현재 default smoke는 diagonal scale preset이다.
- `FreeMatch`, `PabLO`는 아직 direct system-FL preset이 없으므로 classifier-first 확장으로
  추가할 대상이다.
- staged 구조에서는 먼저 `fixed embedding + classifier` seed를 닫고, 그 다음 query-domain LoRA 적응과 시스템 translation 여부를 판단한다.

---

## 9. Local demo

현재 상태:

- `scripts/experiments/run_local_demo.py`는 아직 미구현이다.
- 실행하면 종료 메시지만 출력한다.

---

## 참고

실험 결과 요약은 아래 문서를 본다.

- `docs/experiment_results.md`
