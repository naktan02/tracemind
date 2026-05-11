# Central SSL Control

이 폴더는 중앙집중형 LoRA + classifier control 실행 entrypoint만 둔다.
알고리즘 core는 `methods/ssl`, 실행 조합과 파라미터는 `conf/` Hydra config가
소유한다.

## 기본 실행

실제 학습 실행 전에는 먼저 compose 결과를 확인한다.

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py --cfg job
```

공통 Query SSL control 기본 실행은 현재 FixMatch다.

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py
```

공통 기본값:

```text
runtime=gpu_local
query_ssl_method=fixmatch_usb_v1
query_source=ourafla_ssl_labeled1024_per_class_seed42_nllb_views_v1
augmentation=precomputed_usb_candidates_v1
initial_checkpoint=none
max_train_steps=3000
train_batch_size=12
query_ssl_method.unlabeled_batch_size=12
eval_batch_size=32
train_jsonl=data/processed/query_ssl_views/.../labeled_train.with_views.jsonl
unlabeled_jsonl=data/processed/query_ssl_views/.../unlabeled_pool.with_views.jsonl
```

`precomputed_usb_candidates_v1`는 실행 중 역번역을 다시 만들지 않고,
row에 strict USB형 `text + aug_0 + aug_1`이 없으면 실패하게 하는 설정이다.
기본 실행은 USB식 cold-start 비교에 맞춰 기존 classifier seed를 로드하지 않고,
LoRA adapter와 classifier head를 새로 초기화한다.
학습 예산도 USB처럼 전체 데이터 epoch replay가 아니라 `max_train_steps` 총
optimizer update 수로 고정한다. `epochs`는 selection 평가/history cadence를
나누는 단위이며, 기본값은 `3000` steps다.
16GB급 GPU 기준 기본 batch는 labeled `12`, unlabeled `12`로 둔다. FixMatch는
한 step에서 labeled/weak/strong forward를 수행하므로 VRAM 여유를 우선한다.

## 방법론별 실행

### FixMatch

기본 실행:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py
```

명시 실행:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  strategy_axes/ssl/consistency_method=fixmatch_usb_v1 \
  track_presets/central_ssl_control/query_source=ourafla_ssl_labeled1024_per_class_seed42_nllb_views_v1 \
  strategy_axes/ssl/augmentation=precomputed_usb_candidates_v1
```

입력 view:

- `text`: weak view
- `aug_0`: strong view
- `aug_1`: 저장은 유지하지만 FixMatch single strong-view 경로에서는 소비하지 않는다.

자주 쓰는 override:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  query_ssl_method.p_cutoff=0.9 \
  query_ssl_method.lambda_u=1.0 \
  query_ssl_method.supervised_loss_weight=1.0
```

Budget ablation:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  max_train_steps=1000

uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  max_train_steps=3000

uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  max_train_steps=10000
```

### USB PseudoLabel

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  strategy_axes/ssl/consistency_method=pseudolabel_usb_v1
```

PseudoLabel은 weak view만 필요하므로 unlabeled `text`를 사용하고,
strong augmentation 설정을 소비하지 않는다. 공통 entrypoint 기본값에 남아 있는
augmentation axis는 multiview method에서만 runner manifest와 row 준비에 반영된다.

입력 view:

- `text`: weak view
- `aug_0/aug_1`: 소비하지 않는다.

자주 쓰는 override:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  strategy_axes/ssl/consistency_method=pseudolabel_usb_v1 \
  query_ssl_method.p_cutoff=0.9 \
  query_ssl_method.unsup_warm_up=0.2
```

### FlexMatch

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  strategy_axes/ssl/consistency_method=flexmatch_usb_v1
```

FlexMatch는 USB 원본처럼 FixMatch의 weak/strong objective 위에
`idx_ulb` 기반 classwise adaptive threshold state를 추가한다. 실행 입력은
FixMatch와 동일하게 precomputed USB 후보를 사용한다.

입력 view:

- `text`: weak view
- `aug_0`: strong view
- `aug_1`: 저장은 유지하지만 FlexMatch single strong-view 경로에서는 소비하지 않는다.

자주 쓰는 override:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  strategy_axes/ssl/consistency_method=flexmatch_usb_v1 \
  query_ssl_method.p_cutoff=0.9 \
  query_ssl_method.thresh_warmup=true \
  query_ssl_method.lambda_u=1.0
```

### FreeMatch

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  strategy_axes/ssl/consistency_method=freematch_usb_v1
```

FreeMatch는 USB 원본처럼 FixMatch의 weak/strong objective 위에 `time_p`,
`p_model`, `label_hist` 기반 self-adaptive threshold와 entropy regularization을
추가한다. 실행 입력은 FixMatch와 동일하게 precomputed USB 후보를 사용한다.

자주 쓰는 override:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_query_ssl.py \
  strategy_axes/ssl/consistency_method=freematch_usb_v1 \
  query_ssl_method.ema_p=0.999 \
  query_ssl_method.ent_loss_ratio=0.01 \
  query_ssl_method.use_quantile=false
```

Supervised LoRA seed control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_classifier.py
```

## 산출물과 metric

실행이 끝나면 stdout에 아래 경로가 출력된다.

```text
output_dir=runs/train_lora_query_ssl/<method_name>/<run_id>
adapter_dir=...
classifier_path=...
manifest=...
report_json=...
projection_manifest=...
```

주요 파일:

- `adapter_dir`: 학습된 LoRA adapter와 tokenizer.
- `classifier_path`: classifier head `.pt`.
- `manifest`: 실행 설정, method config, history, best selection report.
- `report_json`: validation/test 결과 전체.
- `projection_manifest`: eval set별 최종 representation 분포도 artifact 목록.

`report_json.results.<eval_set>`의 논문 비교용 주요 metric:

- `accuracy_top_1`
- `macro_precision`, `macro_recall`, `macro_f1`
- `weighted_precision`, `weighted_recall`, `weighted_f1`
- `balanced_accuracy`
- `worst_category_f1`, `worst_category_f1_value`
- `worst_category_precision`, `worst_category_recall`
- `expected_calibration_error`, `max_calibration_error`
- `overconfidence_gap`
- `mean_correct_top_1_probability`, `mean_incorrect_top_1_probability`
- `loss`, `correct_top_1`, `rows_total`
- `mean_true_label_probability`, `mean_top_1_probability`, `mean_margin_top1_top2`
- `confusion_matrix`, `per_category`

`per_category`에는 class별 `support`, `predicted`, `correct`, `precision`,
`recall`, `f1`, confidence/margin 평균이 들어간다.

`manifest.runtime_metrics`에는 run-level 학습 비용 지표가 들어간다.

- `train_seconds`: 학습 loop에 걸린 wall-clock 초.
- `training_example_count`: throughput 계산에 사용한 학습 row 수.
- `examples_per_second`: `training_example_count / train_seconds`.
- `trainable_param_ratio`: 전체 parameter 중 학습 대상 parameter 비율.

`projection_manifest`는 eval set마다 아래 파일을 가리킨다.

- `<eval_set>.projection.jsonl`: `x`, `y`, 실제 label, 예측 label, 정오답,
  top-1 probability.
- `<eval_set>.projection.png`: 최종 LoRA pooled backbone feature의 2D 분포도.

projection은 UMAP을 우선 사용하고, UMAP 실행이 불가능하면 PCA 또는 zero-pad
fallback으로 저장하며 fallback 이유를 manifest에 남긴다.

## 알고리즘 추가 방식

새 Query SSL 알고리즘을 추가할 때 기본 변경 위치는 아래다.

- `methods/ssl/algorithms/<method>/`: 알고리즘 objective core
- `conf/strategy_axes/ssl/consistency_method/<method>_*.yaml`: method identity와 파라미터
- `tests/unit/test_methods_<method>.py`: tensor-level objective 검증
- 필요 시 `conf/strategy_axes/ssl/augmentation/*.yaml`: weak/strong view 준비 방식

방법론이 늘어날 때마다 `scripts/experiments/central_ssl_control/train_lora_<method>.py`
파일을 계속 추가하는 방식은 장기 구조로 보지 않는다. method 차이는 Hydra
`strategy_axes/ssl/consistency_method` 교체로 표현하고, Python entrypoint는 공통
Query SSL runner를 호출하는 얇은 wrapper로 수렴시키는 것이 맞다.

새 알고리즘을 추가할 때 별도 entrypoint가 꼭 필요한지는 먼저 확인하고, 실행
의미가 같다면 Hydra config 추가만 우선한다.
