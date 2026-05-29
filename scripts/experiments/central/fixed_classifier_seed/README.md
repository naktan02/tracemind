# Fixed Classifier Seed

이 폴더는 고정 임베딩 위에 linear classifier head를 학습하는 첫 seed 단계를 둔다.
PEFT text encoder SSL control 자체가 아니라, 이후 control/ablation에서 참조할 수
있는 안정된 label schema, classifier head, 평가 기준점을 만든다.

즉 `fixed_classifier_seed`는 실행 가능한 entrypoint다. 다만 중앙 SSL run이 매번
이 단계를 자동으로 실행하지는 않는다. 한 번 실행해서 classifier artifact를 만들고,
필요한 중앙 SSL bootstrap 또는 ablation run이 그 artifact를 입력으로 참조한다.

## 왜 필요한가

- 중앙/FL SSL 비교에서 label order와 category 해석을 고정하는 기준 artifact를 남긴다.
- PEFT SSL이 아직 학습되지 않은 bootstrap 단계에서 fixed embedding teacher로
  pseudo-label 후보를 만들 수 있다.
- `canonical_fixed_classifier_seed` warm-start ablation이 같은 classifier provenance에서
  시작했는지 검증할 수 있다.
- seed 자체 성능을 남겨 PEFT text encoder adaptation이 실제로 이득을 내는지 비교한다.

현재 canonical seed artifact는 `clf_2026_04_11_143138`이다. 중앙 SSL method 비교의
기본 initial checkpoint는 `none`이며, fixed seed는 기본값이 아니라 bootstrap,
continual adaptation, warm-start ablation에서 명시적으로 선택한다.

## 실행

compose 확인:

```bash
uv run python scripts/experiments/central/fixed_classifier_seed/train_softmax_classifier.py --cfg job
```

기본 실행:

```bash
uv run python scripts/experiments/central/fixed_classifier_seed/train_softmax_classifier.py
```

runtime이나 dataset을 바꾸는 경우에도 실행 조합은 `conf/entrypoints/central/fixed_classifier_seed/`
와 `conf/execution_context/**`가 source of truth다.
