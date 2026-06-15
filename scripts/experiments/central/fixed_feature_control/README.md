# Central Fixed Feature Control

중앙 지도학습 fixed-feature baseline entrypoint다. 이 트랙은 encoder/backbone을
학습하지 않고, 고정 feature 위의 scikit-learn classifier만 학습한다.

## 기본 실행

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=multinomial_nb
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=decision_tree
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/estimator=linear_svc
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  execution_context/query_labeled_budget=labeled100_per_class_seed42_nllb_views_v1
```

기본값은 `tfidf_word + logistic_regression + pc1024 + gpu_local`이다. TF-IDF와
scikit-learn 학습 자체는 CPU에서 실행되지만, runtime leaf는 중앙 실험 기본 축과
맞추기 위해 `gpu_local`을 사용한다.

## Frozen Embedding 실행

`frozen_embedding_mxbai`는 TF-IDF를 쓰지 않는다. mxbai encoder로 dense embedding을
만든 뒤, encoder는 고정하고 classifier만 학습한다.
모델 ID/cache/task prefix는 중앙 supervised와 같은
`strategy_axes/model_architecture/backbone=mxbai_encoder`에서 오고, 임베딩 추출
batch size는 `central_ssl_budget.eval_batch_size`를 따른다.

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/feature_space=frozen_embedding_mxbai \
  strategy_axes/classification/estimator=logistic_regression
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/feature_space=frozen_embedding_mxbai \
  strategy_axes/classification/estimator=decision_tree
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py \
  strategy_axes/classification/feature_space=frozen_embedding_mxbai \
  strategy_axes/classification/estimator=linear_svc
```

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
```

계산 core는 `methods/classification/fixed_feature/`가 소유하고, 이 폴더는 Hydra
config 조합과 artifact 저장 orchestration만 맡는다.
