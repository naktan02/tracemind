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
```

계산 core는 `methods/classification/fixed_feature/`가 소유하고, 이 폴더는 Hydra
config 조합과 artifact 저장 orchestration만 맡는다.
