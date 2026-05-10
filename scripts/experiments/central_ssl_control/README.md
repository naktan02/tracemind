# Central SSL Control

이 폴더는 중앙집중형 LoRA + classifier control 실행 entrypoint만 둔다.
알고리즘 core는 `methods/ssl`, 실행 조합과 파라미터는 `conf/` Hydra config가
소유한다.

## 기본 실행

실제 학습 실행 전에는 먼저 compose 결과를 확인한다.

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_fixmatch.py --cfg job
```

FixMatch 중앙 SSL control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_fixmatch.py
```

현재 FixMatch 기본값은 materialized view 파일을 사용한다.

```text
query_source=ourafla_ssl_labeled1024_per_class_seed42_nllb_views_v1
augmentation=precomputed_usb_candidates_v1
train_jsonl=data/processed/query_ssl_views/.../labeled_train.with_views.jsonl
unlabeled_jsonl=data/processed/query_ssl_views/.../unlabeled_pool.with_views.jsonl
```

`precomputed_usb_candidates_v1`는 실행 중 역번역을 다시 만들지 않고,
row에 strict USB형 `text + aug_0 + aug_1`이 없으면 실패하게 하는 설정이다.
FixMatch는 single strong-view 알고리즘이므로 `text`를 weak, `aug_0`을 strong으로
사용한다.

USB PseudoLabel 중앙 SSL control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_pseudolabel.py
```

PseudoLabel은 weak view만 필요하므로 unlabeled `text`를 사용하고,
strong augmentation 설정을 요구하지 않는다.

Supervised LoRA seed control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_classifier.py
```

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

다만 현재는 기존 비교 명령의 안정성을 위해 `train_lora_pseudolabel.py`와
`train_lora_fixmatch.py`를 유지한다. 새 알고리즘을 추가할 때 별도 entrypoint가 꼭
필요한지는 먼저 확인하고, 실행 의미가 같다면 Hydra config 추가만 우선한다.
