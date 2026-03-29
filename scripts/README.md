# Scripts Guide

이 디렉터리는 데이터 준비, prototype 관리, 실험 실행용 스크립트를 모아 둔 곳이다.

## 사전 준비

실험 스크립트 실행 전 기본 의존성을 맞춘다.

```bash
uv sync --extra dev --extra experiments
```

모델을 이미 캐시에 받아 둔 상태에서 GPU로 실행하려면 보통 아래 옵션을 같이 쓴다.

```bash
--device cuda --local-files-only
```

## 공통 실행 규칙

대부분의 임베딩 기반 스크립트는 이제 `--embedding-profile`을 공통으로 받는다.

- `mxbai`
  - 실제 transformer 임베딩 실행용 기본 preset
- `hash_debug`
  - 빠른 smoke/debug용 preset

기본 사용법:

```bash
--embedding-profile mxbai
```

같은 backend 안에서 모델만 바꾸고 싶다면 low-level override만 추가한다.

```bash
--embedding-profile mxbai \
--embedding-model-id intfloat/e5-large-v2 \
--embedding-model-revision main
```

즉 보통은 `profile`만 고르고,
정말 필요할 때만 `backend/model_id/revision`을 개별 override한다.

YAML 기반 실험 스크립트는 같은 개념을 `embedding.profile=...` override로 쓴다.

---

## 1. PrototypePack 생성

train split으로 single centroid 기반 `PrototypePack`을 만든다.

```bash
uv run python scripts/prototypes/seed_prototypes.py \
  --input-jsonl data/processed/splits/ourafla_train_split.v1.train.jsonl \
  --embedding-profile mxbai \
  --device cuda \
  --local-files-only
```

출력 기본 경로:

- `data/processed/prototype_packs/`
- `data/processed/prototype_build_states/`

---

## 2. PrototypePack baseline 평가

생성된 `PrototypePack`을 validation/test에 평가한다.

```bash
uv run python scripts/prototypes/evaluate_prototype_pack.py \
  --prototype-pack data/processed/prototype_packs/<prototype_version>.json \
  --eval-set validation=data/processed/splits/ourafla_train_split.v1.validation.jsonl \
  --eval-set test=data/processed/labeled_query_sets/ourafla_mental_health_text_classification_test.v1.jsonl \
  --embedding-profile mxbai \
  --device cuda \
  --local-files-only
```

출력 기본 경로:

- `data/processed/evaluations/prototype_packs/`

---

## 3. Prototype 전략 비교 실험

`single / kmeans / dbscan` 세 전략을 같은 train/validation/test에서 비교한다.

기본 설정 파일:

- `scripts/experiments/configs/prototype_strategy/default.yaml`

기본 실행:

```bash
uv run python scripts/experiments/prototype_strategy_experiment.py
```

GPU 실험:

```bash
uv run python scripts/experiments/prototype_strategy_experiment.py \
  --override embedding.device=cuda \
  --override embedding.local_files_only=true
```

주요 override 예시:

```bash
uv run python scripts/experiments/prototype_strategy_experiment.py \
  --override dataset.train_jsonl=tmp/prototype_strategy_smoke/train.jsonl \
  --override dataset.validation_jsonl=tmp/prototype_strategy_smoke/validation.jsonl \
  --override dataset.test_jsonl=tmp/prototype_strategy_smoke/test.jsonl \
  --override embedding.profile=hash_debug \
  --override embedding.hash_dim=64
```

출력 기본 경로:

- `data/processed/evaluations/prototype_strategy_experiments/`

---

## 4. Prototype threshold sweep

선택한 전략 위에서 pseudo-label 채택 threshold를 grid search한다.

기본 설정 파일:

- `scripts/experiments/configs/prototype_threshold_sweep/default.yaml`

기본 실행:

```bash
uv run python scripts/experiments/prototype_threshold_sweep.py
```

`kmeans` + GPU:

```bash
uv run python scripts/experiments/prototype_threshold_sweep.py \
  --override embedding.device=cuda \
  --override embedding.local_files_only=true
```

`single` + GPU:

```bash
uv run python scripts/experiments/prototype_threshold_sweep.py \
  --override strategy.name=single \
  --override embedding.device=cuda \
  --override embedding.local_files_only=true
```

smoke 예시:

```bash
uv run python scripts/experiments/prototype_threshold_sweep.py \
  --override embedding.profile=hash_debug \
  --override embedding.hash_dim=64 \
  --override embedding.device=cpu \
  --override embedding.local_files_only=true \
  --override dataset.train_jsonl=tmp/prototype_strategy_smoke/train.jsonl \
  --override dataset.validation_jsonl=tmp/prototype_strategy_smoke/validation.jsonl \
  --override dataset.test_jsonl=tmp/prototype_strategy_smoke/test.jsonl
```

출력 기본 경로:

- `data/processed/evaluations/prototype_threshold_sweeps/`

---

## 5. Softmax classifier head baseline

고정 임베딩 위에 linear classifier head를 학습한다.

```bash
uv run python scripts/experiments/train_softmax_classifier.py \
  --train-jsonl data/processed/splits/ourafla_train_split.v1.train.jsonl \
  --eval-set validation=data/processed/splits/ourafla_train_split.v1.validation.jsonl \
  --eval-set test=data/processed/labeled_query_sets/ourafla_mental_health_text_classification_test.v1.jsonl \
  --embedding-profile mxbai \
  --device cuda \
  --local-files-only \
  --selection-set validation
```

출력 기본 경로:

- 평가 리포트: `data/processed/evaluations/classifier_heads/`
- 모델 아티팩트: `data/processed/classifier_heads/`

---

## 6. Federated simulation smoke

bootstrap train subset으로 prototype을 만들고, 나머지 train을 client shard로 나눠
`pseudo-label -> local update -> aggregation -> republish` 루프를 검증한다.

```bash
uv run python scripts/experiments/run_federated_simulation.py \
  --train-jsonl data/processed/splits/ourafla_train_split.v1.train.jsonl \
  --validation-jsonl data/processed/splits/ourafla_train_split.v1.validation.jsonl \
  --output-dir tmp/federated_simulation_manual \
  --client-count 4 \
  --rounds 1 \
  --embedding-profile hash_debug \
  --published-model-id tracemind-embed-sim \
  --confidence-threshold 0.6 \
  --margin-threshold 0.02 \
  --max-examples 32 \
  --min-required-examples 4
```

출력 예시:

- `main_server/model_manifests/`
- `main_server/prototype_packs/`
- `main_server/vector_adapter_states/`
- `agents/<agent_id>/training_updates/`

---

## 7. Local demo

현재 상태:

- `scripts/experiments/run_local_demo.py`는 아직 미구현이다.
- 실행하면 종료 메시지만 출력한다.

---

## 참고

실험 결과 요약은 아래 문서를 본다.

- `docs/experiment_results.md`
