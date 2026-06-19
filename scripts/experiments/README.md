# Experiments

`scripts/experiments/`는 TraceMind의 실험 entrypoint 계층이다. Hydra
config를 compose하고, dataset/runtime resource를 준비한 뒤 `methods`,
`shared`, `agent`, `main_server`의 core를 호출해 중앙 실험과 FL SSL simulation을
실행한다.

알고리즘 의미와 계약은 이 디렉터리가 소유하지 않는다. 실행 조합은 `conf/`,
method core는 `methods/`, payload 의미는 `shared/`, 운영 runtime adapter는
`agent`와 `main_server`가 소유한다.

## What You Can Run

| 목적 | 시작점 | 자세한 안내 |
| --- | --- | --- |
| 중앙 SSL pooled/offline control | `central/ssl_control/run_query_ssl_control.py` | [central/ssl_control/README.md](central/ssl_control/README.md) |
| 중앙 PEFT/full text encoder 지도학습 control | `central/ssl_control/run_peft_supervised_control.py`, `central/ssl_control/run_full_text_encoder_supervised_control.py` | [central/ssl_control/README.md](central/ssl_control/README.md) |
| fixed-feature 지도학습 baseline | `central/fixed_feature_control/run_fixed_feature_baseline.py` | [central/fixed_feature_control/README.md](central/fixed_feature_control/README.md) |
| fixed-feature self-training baseline | `central/fixed_feature_control/run_fixed_feature_self_training_baseline.py` | [central/fixed_feature_control/README.md](central/fixed_feature_control/README.md) |
| FL SSL client split materialization | `fl_ssl/materialize_fl_client_split.py` | [fl_ssl/README.md](fl_ssl/README.md) |
| FL SSL simulation | `fl_ssl/run_federated_simulation.py` | [fl_ssl/README.md](fl_ssl/README.md), [fl_ssl/federated_simulation/README.md](fl_ssl/federated_simulation/README.md) |

## Common Commands

설치:

```bash
uv sync --extra dev --extra experiments
```

Hydra가 실제로 compose한 job 확인:

```bash
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py --cfg job
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation --cfg job
```

중앙 SSL control 실행:

```bash
uv run python scripts/experiments/central/ssl_control/run_query_ssl_control.py
```

fixed-feature 지도학습 baseline 실행:

```bash
uv run python scripts/experiments/central/fixed_feature_control/run_fixed_feature_baseline.py
```

FL SSL split materialization:

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split
```

FL SSL simulation 실행:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation
```

트랙별 method, seed, dataset, output 조합 예시는 하위 README에 둔다. 이 파일은
실험 layer의 시작점과 경계만 설명한다.

## How Execution Is Split

| 계층 | 소유 책임 |
| --- | --- |
| `conf/entrypoints/**` | entrypoint별 기본 실행 조합 |
| `conf/strategy_axes/**` | method, model architecture, local update, data source 같은 교체 축 |
| `scripts/experiments/**` | Hydra compose, resource 준비, run context 구성, artifact 저장 |
| `scripts/support/**` | 실험 entrypoint가 공유하는 runner/helper |
| `scripts/workflows/**` | dataset 준비, cache, report, index 같은 작업형 CLI |
| `methods/**` | SSL, FSSL, aggregation, adaptation, prototype core |
| `shared/**` | 경계 payload와 canonical contract |
| `agent/**`, `main_server/**` | production/runtime adapter와 round lifecycle |

## Read Path

처음 보는 사람은 아래 순서로 읽으면 된다.

1. 실행하려는 트랙의 README를 고른다.
   - 중앙 SSL/지도학습: [central/ssl_control/README.md](central/ssl_control/README.md)
   - fixed-feature baseline: [central/fixed_feature_control/README.md](central/fixed_feature_control/README.md)
   - FL SSL: [fl_ssl/README.md](fl_ssl/README.md)
2. 같은 entrypoint를 `--cfg job`으로 preview한다.
3. `conf/entrypoints/**`와 `conf/strategy_axes/**`에서 조합 가능한 축을 확인한다.
4. 실제 algorithm 의미는 `methods/README.md`에서 해당 method package로 따라간다.
5. 산출물 확인은 `apps/experiment_dashboard/README.md` 또는 `scripts/workflows/`의
   report/index CLI를 본다.

## Boundary Rules

- 이 디렉터리에 새 method core를 두지 않는다.
- shared schema, contract 의미, runtime policy의 source of truth를 만들지 않는다.
- script-local preset으로 `conf/`의 기본값을 복제하지 않는다.
- 실험 helper가 운영 경로에서도 필요해지면 `methods`, `shared`, `agent`,
  `main_server` 중 실제 소유 경계로 옮긴다.
- secure aggregation/encryption처럼 config와 contract만 있고 runtime이 아직 붙지
  않은 축은 실행 가능 기능으로 문서화하지 않는다.
