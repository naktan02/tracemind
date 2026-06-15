# Scripts Guide

`scripts/`는 실험 실행, sweep, report, visualization entrypoint만 소유한다.
알고리즘 core는 `methods/`, 공통 contract/domain은 `shared/`, production runtime
adapter는 `agent/`와 `main_server/`가 소유한다.

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-scripts-guide.md`에 보관했다. 현재 작업자는
아래 경계와 `conf/` 조합을 먼저 본다.

## 경계

- `scripts/workflows/datasets/`: dataset asset 생성 CLI.
- `scripts/experiments/`: track별 실험 entrypoint와 실험 전용 runtime support.
- `scripts/experiments/fl_ssl/`: FL SSL split, simulation, sweep entrypoint.
- `scripts/experiments/central/fixed_feature_control/`: 고정 feature 기반 중앙
  지도학습 baseline entrypoint.
- `scripts/support/query_ssl_text_encoder/`: query-domain text encoder SSL runtime
  support와 IO. PEFT-specific runner/artifact 이름은 PEFT leaf에만 남긴다.
- `scripts/workflows/result_index/`: `runs` report를 index/dashboard JSON으로 정규화.
- `scripts/runtime_adapters/`: scripts가 agent/main_server runtime을 재사용할 때 쓰는 bridge.
- `scripts/support/reporting/`, `scripts/support/artifacts/`, `scripts/codegen/`: 보조 entrypoint/helper.

## 불변 규칙

- 운영 후보 알고리즘을 `scripts`에 먼저 만들고 나중에 `methods`로 복사하지 않는다.
- `scripts`는 `agent.src`, `main_server.src`를 직접 import하지 않는다.
- runtime 재사용은 `scripts/runtime_adapters/`의 capability bridge로 드러낸다.
- 실행 조합과 파라미터의 source of truth는 루트 `conf/`다.
- script helper는 해당 entrypoint가 직접 쓰는 범위까지만 둔다.

## 기본 확인 명령

설치:

```bash
uv sync --extra dev --extra experiments
```

Hydra compose:

```bash
uv run python <entrypoint>.py --cfg job
```

기본 실행:

```bash
uv run python <entrypoint>.py execution_context/runtime_env=auto_local
```

FL SSL smoke:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_simulation.py \
  run_controls/fl_ssl/budget=smoke
```

중앙 PEFT SSL control smoke:

```bash
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py \
  run_controls/central_ssl/budget=smoke
```

중앙 fixed-feature 지도학습 baseline:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py
```

중앙 fixed-feature self-training 준지도 baseline:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_self_training_baseline.py
```

## Read Path

1. `docs/ai_context_manifest.yaml`에서 task route를 고른다.
2. `docs/execution_index.md`에서 필요한 active 문서만 읽는다.
3. 관련 `conf/entrypoints/**`와 `conf/strategy_axes/**`를 compose한다.
4. script entrypoint는 orchestration만 확인하고 core 의미는 `methods/`에서 본다.
5. 결과 해석은 report artifact와 `scripts/workflows/result_index/` schema를 기준으로 한다.
