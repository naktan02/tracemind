# Central Fixed-Feature Control

`central/fixed_feature_control/`은 encoder를 학습하지 않는 중앙 지도학습 baseline
entrypoint다. TF-IDF 또는 frozen embedding feature를 만든 뒤 scikit-learn
classifier만 학습한다.

PEFT/full text encoder를 학습하는 중앙 지도학습과 neural SSL control은
[../ssl_control/README.md](../ssl_control/README.md)를 사용한다.

## Entrypoints

| 목적 | 명령 |
| --- | --- |
| fixed-feature supervised baseline | `uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py` |
| fixed-feature self-training baseline | `uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py` |
| Hydra compose 확인 | 위 명령에 `--cfg job` 추가 |

## Standard Run Shape

기본 supervised run은 아래 조합이다.

```text
feature space: tfidf_word
estimator: logistic_regression
labeled budget: per_class1024
runtime leaf: gpu_local
selection/eval set: test
```

TF-IDF와 scikit-learn classifier 학습은 CPU에서 실행된다. `gpu_local`은 중앙 실험의
runtime axis와 맞추기 위한 기본 leaf다. `frozen_embedding_mxbai` feature를 쓸 때는
mxbai embedding 추출 때문에 cache/GPU 설정의 영향을 받는다.

## Combination Axes

| Axis | Common values |
| --- | --- |
| Feature space | `tfidf_word`, `frozen_embedding_mxbai` |
| Estimator | `logistic_regression`, `multinomial_nb`, `decision_tree`, `linear_svc` |
| Labeled budget | `per_class1024`, `labeled100_per_class_seed42_nllb_views_v1` |
| Runtime | `gpu_local`, `gpu_online`, `cpu_local` |

`multinomial_nb`는 non-negative sparse TF-IDF feature용이다. dense
`frozen_embedding_mxbai` feature와 조합하지 않는다. `linear_svc`는
`predict_proba`가 없어 threshold self-training에서는 제외한다.

## Quick Examples

기본 TF-IDF + logistic regression:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py
```

TF-IDF + 다른 estimator:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=multinomial_nb

uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=linear_svc
```

pc100 labeled budget:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1
```

frozen mxbai embedding + classifier:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/feature_space=frozen_embedding_mxbai \
  strategy_axes/classification/estimator=logistic_regression
```

## Self-Training Baseline

`run_fixed_feature_self_training_baseline.py`는 scikit-learn
`SelfTrainingClassifier` 기반 classical 준지도 baseline이다. FixMatch, AdaMatch,
FreeMatch 같은 neural consistency SSL objective와는 다른 family다.

기본값은 `tfidf_word + logistic_regression + threshold=0.95`다.

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py

uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py \
  strategy_axes/classification/estimator=decision_tree

uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py \
  fixed_feature_self_training.max_unlabeled_rows=8000
```

unlabeled source는 `query_source.unlabeled_jsonl`을 사용하되 학습 입력 label은
`fixed_feature_self_training.unlabeled_label` 기본값 `-1`로 마스킹한다.
`unlabeled_cap_policy=step_budget`은 중앙 실험 budget의 step 수에 맞춰 unlabeled
exposure를 줄인다.

## Outputs

supervised baseline:

```text
runs/central/supervised/fixed_feature/{feature_space}/{estimator}/{labeled_budget}/{run_id}/
```

self-training baseline:

```text
runs/central/ssl/fixed_feature_self_training/{feature_space}/{estimator}/{labeled_budget}/{run_id}/
```

대표 산출물:

```text
artifacts/model.joblib
artifacts/feature_space.joblib
artifacts/label_schema.json
artifacts/predictions.test.jsonl
reports/report.json
logs/training_log.jsonl
```

self-training run은 `artifacts/pseudo_labels.train_unlabeled.jsonl`도 저장한다.

## Ownership

- 이 폴더는 Hydra entrypoint와 artifact 저장 orchestration만 맡는다.
- 계산 core는 [../../../../methods/classification/fixed_feature/README.md](../../../../methods/classification/fixed_feature/README.md)가 소유한다.
- 실행 조합과 기본값은 [../../../../conf/README.md](../../../../conf/README.md)와
  `conf/entrypoints/central/fixed_feature_control/*.yaml`이 소유한다.
