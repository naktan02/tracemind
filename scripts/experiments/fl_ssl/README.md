# FL SSL Experiments

FL SSL 실험 entrypoint와 support helper를 둔다. 이 트랙은 non-IID client split에서
federated semi-supervised learning 방법을 비교하는 simulation track이다.

FL method identity와 method-only 정책은 `methods/federated_ssl/`, SSL objective core는
`methods/ssl`, update-family 계산 core는 `methods/adaptation/*`, 실행 조합과
파라미터는 `conf/` Hydra config가 소유한다.

## When To Use

- non-IID client split에서 SSL/FSSL 방법을 비교할 때
- FedMatch처럼 method package가 policy와 metadata를 소유하는 방법을 실행할 때
- manual FixMatch/FedAvg baseline을 같은 FL runtime에서 확인할 때
- 중앙 supervised checkpoint를 warm-start로 사용해 FL round를 시작할 때
- client split, round metric, communication cost, per-client variance를 함께 보고 싶을 때

중앙 pooled/offline SSL control은
`scripts/experiments/central/ssl_control/README.md`를 사용한다.

## Entrypoint

메인 실행은 하나다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation
```

실행 전 compose 결과를 먼저 확인한다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation --cfg job
```

## Standard Run Shape

기본 smoke는 manual baseline이다.

```text
composition_mode: manual
query_ssl_method: fixmatch_usb_v1
update_family: peft_text_encoder
aggregation_backend: fedavg
budget: smoke
output_dir: runs/_smoke/fl_ssl
```

기본 FL 실행은 materialized split
`shared_general_reddit_pc1024_alpha03_clients10`을 사용한다. 논문용 비교는 보통
`10 clients`, `Dirichlet alpha=0.3`, `FedAvg`, `PEFT text encoder update family`를
기준으로 시작한다.

| Budget | Shape |
|---|---|
| `smoke` | wiring 확인용 짧은 실행 |
| `reduced` | `10 clients x 5 rounds` |
| `main` | `10 clients x 30 rounds` |

새 wiring은 `smoke` 또는 `reduced`로 먼저 확인한다.

## Choose A Run

| If you want to... | Use |
|---|---|
| 가장 빠르게 FL loop를 확인한다 | `run_controls/fl_ssl/budget=smoke` |
| 동일 split에서 baseline을 비교한다 | `fl_method.composition_mode=manual` |
| FedMatch method policy를 사용한다 | `fl_method.composition_mode=method_owned` + `strategy_axes/fssl_method=fedmatch` |
| pc100 split으로 실행한다 | `execution_context/fl_client_split=shared_general_reddit_pc100_alpha03_clients10` |
| 즉석 split smoke를 만든다 | `execution_context/fl_client_split=null` + `fl_data.*` override |
| 중앙 supervised checkpoint에서 시작한다 | `strategy_axes/model_architecture/initial_checkpoint=supervised_20260612_step2000` |
| cache가 준비된 로컬 GPU로 실행한다 | `execution_context/runtime_env=gpu_local` |

`manual`은 SSL objective, topology, checkpoint를 직접 조합하는 baseline/debug 경로다.
`method_owned`는 FedMatch처럼 method package가 policy와 report metadata를 소유하는
경로다.

## Combination Axes

| Axis | Common values | When to change |
|---|---|---|
| Budget | `smoke`, `reduced`, `main` | 실행 크기와 비용을 바꿀 때 |
| Split | `shared_general_reddit_pc100_alpha03_clients10`, `shared_general_reddit_pc1024_alpha03_clients10` | 라벨 예산이나 source 조건을 바꿀 때 |
| Method mode | `manual`, `method_owned` | baseline 직접 조합과 method-owned 실행을 구분할 때 |
| FSSL method | `fedmatch` | method-owned FedMatch를 실행할 때 |
| SSL objective | `fixmatch_usb_v1`, `freematch_usb_v1`, `pseudolabel_usb_v1` | manual baseline의 local SSL objective를 바꿀 때 |
| Shard policy | `dirichlet_alpha03`, `dirichlet_alpha01`, `label_dominant` | non-IID 강도를 바꿀 때 |
| Initial checkpoint | `none`, `supervised_20260612_step2000` | 중앙 지도학습 checkpoint warm-start를 쓸 때 |
| Runtime | `gpu_local`, `gpu_online`, `cpu_local` | cache/network/GPU 조건을 바꿀 때 |

## Quick Examples

### Baseline And Split Checks

```bash
# 고정 split smoke.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  execution_context/fl_client_split=shared_general_reddit_pc1024_alpha03_clients10 \
  federated_run_budget.rounds=1

# pc100 materialized split으로 같은 manual baseline 실행.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  execution_context/fl_client_split=shared_general_reddit_pc100_alpha03_clients10

# manual FixMatch/FedAvg baseline.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  fl_method.composition_mode=manual \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha03
```

### Method-Owned Runs

```bash
# FedMatch reduced run.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  fl_method.composition_mode=method_owned \
  strategy_axes/fssl_method=fedmatch \
  ssl_method.scenario=labels-at-client

# FedMatch main run with supervised warm-start.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=main \
  fl_method.composition_mode=method_owned \
  strategy_axes/fssl_method=fedmatch \
  strategy_axes/model_architecture/initial_checkpoint=supervised_20260612_step2000
```

### Warm-Start And Stress Checks

```bash
# Manual FixMatch run from supervised checkpoint.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=main \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  strategy_axes/model_architecture/initial_checkpoint=supervised_20260612_step2000

# non-IID stress split smoke.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  execution_context/fl_client_split=null \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha01 \
  federated_run_budget.rounds=1

# cache가 준비된 로컬 GPU 실행.
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  execution_context/runtime_env=gpu_local \
  execution_context/fl_client_split=shared_general_reddit_pc100_alpha03_clients10
```

최초 model download/cache warm-up이 필요할 때만 `gpu_online`을 사용한다.

## Client Split

논문 비교용 FL 실행은 materialized client split을 우선한다.

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

생성된 `manifest_json=...` 값을 `fl_data.split_manifest`로 넘긴다.
`shared_client_seed`, `client_local_split`, `server_only_seed`는 labeled exposure
policy이며 shard policy가 아니다.

## Sweep

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  sweep.axis=seed \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha03
```

`materialized_client_split` 기반 client-count sweep은
`sweep.client_count.split_manifest_by_client_count`에 count별 manifest를 명시한다.

## Outputs

새 FL SSL run은 같은 split과 숫자 조건 아래에서 방법론을 빠르게 비교하도록 아래
구조로 저장한다.

```text
runs/fl_ssl/{split}/{condition}/{surface}/{method}/{run_id}/
```

예:

```text
runs/fl_ssl/sz4_ourafla_shared_s42/c10_r30_e1_b8_s50/peft_text_encoder_lora/fixmatch_fedavg/20260605T101139Z/
```

각 run의 canonical report는 아래 파일이다.

```text
reports/fl_ssl_main_comparison.report.json
```

FL SSL는 run 종료 시 final projection을 `projections/`에 자동 생성한다.

```text
projections/validation.projection.jsonl
projections/validation.projection.png
projections/projection_manifest.json
```

대시보드 cache 생성과 `--reset` 의미는
`apps/experiment_dashboard/README.md`가 소유한다.

## Internal References

사람이 코드를 읽을 때는 아래 순서가 가장 짧다.

```text
conf/entrypoints/fl_ssl/run_federated_simulation.yaml
-> run_federated_simulation.py
-> federated_simulation/config_request.py
-> federated_simulation/simulation.py
-> federated_simulation/flow/bootstrap.py
-> federated_simulation/flow/round_loop.py
-> federated_simulation/flow/result_builder.py
```

`run_federated_simulation.py`는 sweep 처리, output dir 결정, request 생성, runner
호출, 결과 출력만 맡는다. `flow/round_loop.py`는 server step부터 summary assembly까지
round lifecycle phase를 보여준다.

긴 과거 runbook과 특정 시점 실행 예시는
`docs/notes/decisions/2026-05-28-archived-fl-ssl-runbook.md`에 보관했다. 현재 실행
구조 판단은 `docs/architecture/target-method-runtime-structure.md`와 이 README를
우선한다.

주의:

- scripts runner에 새 method core를 추가하지 않는다.
- 논문용 산출물은 `report.protocol.embedding_adapter`와
  `local_trainer_runtime`으로 `gpu_local + mxbai` 여부를 확인한다.
- smoke/test artifact는 `runs/_smoke` 아래에 둔다.
