# FL SSL Experiments

`scripts/experiments/fl_ssl/`는 non-IID client split에서 federated
semi-supervised learning 방법을 비교하는 simulation entrypoint다. 중앙
pooled/offline SSL control은 [../central/ssl_control/README.md](../central/ssl_control/README.md)를
사용한다.

이 트랙은 production runtime이 아니라 `agent`, `main_server`, `methods`,
`shared` core를 한 프로세스에서 조합해 FL round를 재현하는 실험 layer다.

## Entrypoints

| 목적 | 명령 |
| --- | --- |
| FL SSL simulation 실행 | `uv run python -m scripts.experiments.fl_ssl.run_federated_simulation` |
| Hydra compose 확인 | `uv run python -m scripts.experiments.fl_ssl.run_federated_simulation --cfg job` |
| FL client split materialization | `uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split` |
| report artifact 검증 | `uv run python -m scripts.experiments.fl_ssl.verify_federated_report_artifacts` |

## Standard Run Shape

기본 실행은 materialized pc1024 split에서 manual FixMatch/FedAvg baseline을 짧게
확인하는 smoke run이다.

```text
split: shared_general_reddit_pc1024_alpha03_clients10
composition_mode: manual
local SSL objective: fixmatch_usb_v1
update family: peft_text_encoder
aggregation backend: fedavg
runtime: gpu_local + mxbai
budget: smoke
```

논문 비교용 기본 축은 `10 clients`, `Dirichlet alpha=0.3`, `FedAvg`,
`PEFT text encoder update family`다. 새 조합은 `smoke` 또는 `reduced`로 먼저
확인한다.

| Budget | 용도 |
| --- | --- |
| `smoke` | wiring 확인용 짧은 실행 |
| `reduced` | 작은 비교 run |
| `main` | 본 비교 run |

## Choose A Run

| 하고 싶은 일 | 사용 |
| --- | --- |
| FL loop만 빠르게 확인 | `run_controls/fl_ssl/budget=smoke` |
| 같은 runtime에서 baseline 직접 조합 | `fl_method.composition_mode=manual` |
| FedMatch method policy 사용 | `fl_method.composition_mode=method_owned strategy_axes/fssl_method=fedmatch` |
| pc100 materialized split 사용 | `execution_context/fl_client_split=shared_general_reddit_pc100_alpha03_clients10` |
| 중앙 supervised checkpoint warm-start | `strategy_axes/model_architecture/initial_checkpoint=supervised_20260612_step2000` |
| non-IID 강도 변경 | `strategy_axes/fl_topology/shard_policy=dirichlet_alpha01` |
| 로컬 GPU/cache 사용 | `execution_context/runtime_env=gpu_local` |

`manual`은 SSL objective, topology, checkpoint를 직접 조합하는 baseline/debug 경로다.
`method_owned`는 FedMatch처럼 method package가 local/server policy와 report metadata를
소유하는 경로다.

## Combination Axes

| Axis | Common values |
| --- | --- |
| Split | `shared_general_reddit_pc1024_alpha03_clients10`, `shared_general_reddit_pc100_alpha03_clients10` |
| Method mode | `manual`, `method_owned` |
| FSSL method | `fedmatch` |
| SSL objective | `fixmatch_usb_v1`, `freematch_usb_v1`, `pseudolabel_usb_v1`, `adamatch_usb_v1` |
| Shard policy | `dirichlet_alpha03`, `dirichlet_alpha01`, `label_dominant` |
| Initial checkpoint | `none`, `supervised_20260612_step2000` |
| Runtime | `gpu_local`, `gpu_online`, `cpu_local` |

더 넓은 override 목록은 [../../../conf/README.md](../../../conf/README.md)와
`conf/strategy_axes/**` leaf를 기준으로 본다.

## Quick Examples

기본 smoke:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke
```

pc100 split으로 reduced baseline:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  execution_context/fl_client_split=shared_general_reddit_pc100_alpha03_clients10
```

manual FixMatch/FedAvg baseline:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  fl_method.composition_mode=manual \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha03
```

FedMatch method-owned run:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  fl_method.composition_mode=method_owned \
  strategy_axes/fssl_method=fedmatch \
  ssl_method.scenario=labels-at-client
```

supervised checkpoint warm-start:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=main \
  strategy_axes/model_architecture/initial_checkpoint=supervised_20260612_step2000
```

## Client Splits

본 비교는 materialized client split을 우선 사용한다. 새 split이 필요하면 아래
entrypoint로 manifest를 만들고, 생성된 `manifest_json` 값을 `fl_data.split_manifest`
또는 `execution_context/fl_client_split` leaf에 연결한다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  run_controls/fl_ssl/budget=smoke \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10
```

`shared_client_seed`, `client_local_split`, `server_only_seed`는 labeled exposure
policy다. non-IID 분포는 `strategy_axes/fl_topology/shard_policy`가 정한다.

## Outputs

새 run은 아래 형태로 저장된다.

```text
runs/fl_ssl/{split}/{condition}/{surface}/{method}/{run_id}/
```

canonical report:

```text
reports/fl_ssl_main_comparison.report.json
```

final projection:

```text
projections/validation.projection.jsonl
projections/validation.projection.png
projections/projection_manifest.json
```

대시보드 cache 생성과 `--reset` 의미는
[../../../apps/experiment_dashboard/README.md](../../../apps/experiment_dashboard/README.md)가
소유한다.

## Code Reading Path

실행 흐름을 코드로 따라갈 때는 아래 순서가 가장 짧다.

```text
conf/entrypoints/fl_ssl/run_federated_simulation.yaml
-> scripts/experiments/fl_ssl/run_federated_simulation.py
-> scripts/experiments/fl_ssl/federated_simulation/config_request.py
-> scripts/experiments/fl_ssl/federated_simulation/simulation.py
-> scripts/experiments/fl_ssl/federated_simulation/flow/
```

round lifecycle, adapter, report serialization 세부 구조는
[federated_simulation/README.md](federated_simulation/README.md)를 본다.

## Boundary Rules

- scripts runner에 새 method core를 추가하지 않는다.
- FL method identity와 local/server policy는 `methods/federated_ssl/<method>/`가
  소유한다.
- SSL objective core는 `methods/ssl/`, update-family core는
  `methods/adaptation/**`, aggregation core는 `methods/federated/**`가 소유한다.
- config 기본값과 조합 가능성은 `conf/`에 둔다.
- smoke/test artifact는 `runs/_smoke` 아래에 둔다.
