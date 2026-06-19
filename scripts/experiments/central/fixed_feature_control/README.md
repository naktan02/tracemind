# Central Fixed Feature Control

중앙 지도학습 fixed-feature baseline entrypoint다. 이 트랙은 encoder/backbone을
학습하지 않고, 고정 feature 위의 scikit-learn classifier만 학습한다.

## 기본 실행

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py
```

기본값은 `tfidf_word + logistic_regression + pc1024 + gpu_local`이다. TF-IDF와
scikit-learn 학습 자체는 CPU에서 실행되지만, runtime leaf는 중앙 실험 기본 축과
맞추기 위해 `gpu_local`을 사용한다.

## Supervised Combination Examples

지도학습 fixed-feature baseline은 feature space, estimator, labeled budget을 조합한다.

| Axis | Example values |
|---|---|
| Feature space | `strategy_axes/classification/feature_space={tfidf_word,frozen_embedding_mxbai}` |
| Estimator | `strategy_axes/classification/estimator={logistic_regression,multinomial_nb,decision_tree,linear_svc}` |
| Labeled budget | `execution_context/query_labeled_budget={per_class1024,labeled100_per_class_seed42_nllb_views_v1}` |
| Runtime | `execution_context/runtime_env={gpu_local,gpu_online,cpu_local}` |

```bash
# TF-IDF + logistic regression + pc1024.
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py

# TF-IDF + multinomial Naive Bayes + pc1024.
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=multinomial_nb

# TF-IDF + decision tree + pc1024.
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=decision_tree

# TF-IDF + linear SVC + pc1024.
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=linear_svc

# TF-IDF + logistic regression + pc100.
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1

# Frozen mxbai embedding + logistic regression.
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/feature_space=frozen_embedding_mxbai \
  strategy_axes/classification/estimator=logistic_regression

# Frozen mxbai embedding + linear SVC.
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/feature_space=frozen_embedding_mxbai \
  strategy_axes/classification/estimator=linear_svc
```

`multinomial_nb`는 non-negative sparse TF-IDF feature에는 맞지만 dense embedding과는
맞지 않으므로 `frozen_embedding_mxbai`와 조합하지 않는다.

## Self-training 준지도 baseline

`run_fixed_feature_self_training_baseline.py`는 기존 fixed-feature classifier를
scikit-learn `SelfTrainingClassifier`로 감싼 classical 준지도 baseline이다.
이는 논문식 `Self-Training Classifier` 비교선이며, FixMatch/AdaMatch/UDA 같은
neural consistency SSL objective와는 별도 family다.

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py \
  strategy_axes/classification/estimator=decision_tree
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py \
  strategy_axes/classification/estimator=multinomial_nb
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py \
  fixed_feature_self_training.max_unlabeled_rows=8000
```

unlabeled source는 `query_source.unlabeled_jsonl`을 사용하지만, 학습 입력 label은
항상 `fixed_feature_self_training.unlabeled_label` 기본값 `-1`로 마스킹한다.
기본 pseudo-label threshold는 FixMatch 계열과 비교하기 쉽도록 `0.95`다.
기본 `unlabeled_cap_policy=step_budget`은
`central_ssl_budget.max_train_steps * train_batch_size`만큼 unlabeled row를
seed 고정 sampling해서 쓴다. main budget에서는 현재 pool 전체가 들어가지만,
`max_train_steps=2000` 같은 reduced 비교에서는 unlabeled exposure가 함께 줄어든다.
`linear_svc`는 `predict_proba`가 없어 threshold self-training에서 제외한다.

## Frozen Embedding 실행

`frozen_embedding_mxbai`는 TF-IDF를 쓰지 않는다. mxbai encoder로 dense embedding을
만든 뒤, encoder는 고정하고 classifier만 학습한다.
모델 ID/cache/task prefix는 중앙 supervised와 같은
`strategy_axes/model_architecture/backbone=mxbai_encoder`에서 오고, 임베딩 추출
batch size는 `central_ssl_budget.eval_batch_size`를 따른다.

기본 `gpu_local`에서는 `hf_cache`에 mxbai가 이미 있어야 한다. 최초 다운로드가
필요하면 `execution_context/runtime_env=gpu_online`으로 한 번 실행한다.
`multinomial_nb`는 dense embedding과 맞지 않으므로 이 feature space에서는 제외한다.

## 산출물

```text
runs/central/supervised/fixed_feature/
  tfidf_word/
    logistic_regression/
      labeled-.../
        fixed_feature_tfidf_word_logistic_regression_YYYY_MM_DD_HHMMSS/
          artifacts/
            model.joblib
            feature_space.joblib
            label_schema.json
            predictions.test.jsonl
          reports/
            report.json
          logs/
            training_log.jsonl
  frozen_embedding_mxbai/
    linear_svc/
      labeled-.../
        fixed_feature_frozen_embedding_mxbai_linear_svc_YYYY_MM_DD_HHMMSS/
          artifacts/
            model.joblib
            feature_space.joblib
            label_schema.json
            predictions.test.jsonl
          reports/
            report.json
          logs/
            training_log.jsonl
runs/central/ssl/fixed_feature_self_training/
  tfidf_word/
    logistic_regression/
      labeled-.../
        fixed_feature_self_training_tfidf_word_logistic_regression_YYYY_MM_DD_HHMMSS/
          artifacts/
            model.joblib
            feature_space.joblib
            label_schema.json
            predictions.test.jsonl
            pseudo_labels.train_unlabeled.jsonl
          reports/
            report.json
          logs/
            training_log.jsonl
```

계산 core는 `methods/classification/fixed_feature/`가 소유하고, 이 폴더는 Hydra
config 조합과 artifact 저장 orchestration만 맡는다.
