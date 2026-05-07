# Scripts Guide

`scripts/`는 Hydra 실험 entrypoint, sweep, report, visualization thin wrapper만
소유한다. 알고리즘 core는 `methods/`, 공통 contract/domain은 `shared/`,
runtime/service adapter는 `agent/`와 `main_server/`가 소유한다.

## 구조

- `scripts/datasets/`: dataset asset 생성용 CLI.
- `scripts/prototypes/`: prototype pack 생성, 평가, publication 보조 CLI.
- `scripts/experiments/`: track별 실험 entrypoint와 실험 전용 harness.
- `scripts/experiments/fl_ssl/federated_simulation/`: FL SSL synthetic harness와 artifact dump.
- `scripts/experiments/query_lora_ssl/`: 중앙/FL에서 공유 가능한 query-domain LoRA SSL harness와 adaptation IO.
- `scripts/runtime_adapters/`: scripts가 불가피하게 agent/main_server runtime을 재사용할 때 쓰는 명시 bridge.
- `scripts/io/`: scripts 전용 row IO compatibility wrapper.
- `scripts/reporting/`: report/diagnostics helper.
- `scripts/artifacts/`: run output 경로 helper.
- `scripts/codegen/`: app type 생성 entrypoint.

## 불변 규칙

- 운영 후보 알고리즘을 `scripts`에 먼저 만들고 나중에 복사하지 않는다.
- `scripts`는 `agent.src`, `main_server.src`를 직접 import하지 않는다.
- runtime 재사용이 필요하면 `scripts/runtime_adapters/`에 역할이 드러나는 bridge를 둔다.
- `conf/`가 Hydra 실행 조합과 파라미터의 source of truth다.
- `scripts` 하위 helper는 해당 entrypoint가 직접 쓰는 범위까지만 둔다.

## Config 지도

- `conf/entrypoints/`: 사람이 실행하는 top-level Hydra config.
- `conf/execution_context/`: dataset asset, embedding adapter, runtime env.
- `conf/strategy_axes/ssl/`: pseudo-label selection, consistency method, augmentation.
- `conf/strategy_axes/adaptation/`: PEFT adapter, transformer backbone, initial checkpoint.
- `conf/strategy_axes/fl/`: shard policy, method descriptor, client training profile.
- `conf/strategy_axes/prototype/`: prototype build strategy.
- `conf/track_presets/`: central SSL control과 FL SSL track별 preset.

기본 실행 방식:

```bash
uv run python <entrypoint>.py execution_context/runtime_env=auto_local
```

최종 조합 확인:

```bash
uv run python <entrypoint>.py --cfg job
```

## 자주 쓰는 명령

설치:

```bash
uv sync --extra dev --extra experiments
```

Dataset pipeline:

```bash
uv run python scripts/datasets/run_dataset_pipeline.py
```

Prototype pack 생성:

```bash
uv run python scripts/prototypes/seed_prototypes.py
```

Prototype pack 평가:

```bash
uv run python scripts/prototypes/evaluate_prototype_pack.py
```

중앙 classifier seed:

```bash
uv run python scripts/experiments/central_classifier_seed/train_softmax_classifier.py
```

중앙 LoRA supervised control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_classifier.py
```

중앙 LoRA pseudo-label control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_pseudo_label_classifier.py
```

중앙 LoRA FixMatch control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_fixmatch.py
```

FL SSL simulation smoke:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_simulation.py \
  track_presets/fl_ssl/simulation_preset=smoke
```

FL SSL seed sweep:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_seed_sweep.py \
  track_presets/fl_ssl/simulation_preset=standard \
  strategy_axes/fl/shard_policy=dirichlet_alpha03
```

FL SSL 실험 기본값은 `execution_context/runtime_env=gpu_local`과
`execution_context/embedding_adapter=mxbai`다. `gpu_online`은 cache warm-up/최초
다운로드용이고, `cpu_local + hash_debug`는 entrypoint wiring smoke나 빠른 디버그에만 쓴다.

Prototype strategy analysis:

```bash
uv run python scripts/experiments/prototype_analysis/prototype_strategy_experiment.py
```

Type generation:

```bash
uv run python scripts/codegen/generate_experiment_web_types.py
uv run python scripts/codegen/generate_family_extension_types.py
```

## Runtime Adapters

`scripts/runtime_adapters/`는 허용된 예외 지점이다. scripts가 runtime을 실행하거나
runtime payload를 읽어야 할 때 이 폴더를 통해서만 연결한다.

- `embedding_runtime.py`: agent embedding adapter factory/device resolver bridge.
- `backtranslation_runtime.py`: agent backtranslation service bridge.
- `federated_agent_runtime.py`: FL simulation에서 agent local runtime 호출.
- `federated_server_runtime.py`: FL simulation server bridge의 compatibility facade.
- `federated_server/`: FL simulation에서 main_server round/aggregation 호출을 책임별로
  나눈 실제 runtime adapter package.
  - `runtime.py`: `SimulationServerRuntime` orchestration.
  - `repositories.py`: simulation output root 기준 main_server repository wiring.
  - `prototype_rebuild_bridge.py`: prototype rebuild runtime/service bridge.
  - `initial_state_factory.py`: adapter family별 initial shared state 생성.
  - `round_request_mapper.py`: experiment task config -> round open request 변환.
- `prototype_publication_runtime.py`: prototype pack publication/sync bridge.
- `experiment_web_schema_runtime.py`: app type 생성용 main_server payload bridge.

새 bridge를 추가하기 전에 먼저 확인한다.

- 해당 로직이 `methods` core인가.
- 해당 계약이 `shared`로 올라가야 하는 cross-boundary payload인가.
- scripts 전용 harness 조합인지, production runtime 동작인지.

## 산출물

- `data/processed/`: 재사용 가능한 dataset/model/prototype artifact.
- `runs/<job>/<run_id>/`: 실험 1회 실행 결과, report, dump.
- `agent/state/`: agent runtime state.
- `main_server/state/`: server runtime state와 publication artifact.

GPU/모델 캐시가 필요한 실험 전에는 실제 실행 환경에서 `nvidia-smi`와
`torch.cuda.is_available()`를 먼저 확인한다.
