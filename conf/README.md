# Hydra Config

`conf/`는 TraceMind 실험 entrypoint와 실행 조합 파라미터의 source of truth다.
YAML은 조합과 값만 소유한다. method 계산 의미는 `methods/`, runtime 구현은
`agent`, `main_server`, `scripts/runtime_adapters`가 소유한다.

## What Belongs Here

- entrypoint가 compose할 config tree
- dataset, split, embedding adapter, runtime environment 같은 execution context
- SSL objective, trainable surface, update family, FL topology 같은 strategy selector
- budget, output root, sweep, long-run guard 같은 run control
- runtime callable path나 owner metadata를 선언하는 얇은 leaf

## What Does Not Belong Here

- Python 계산 로직
- algorithm/method 의미와 기본값 해석
- repository, HTTP transport, artifact writer 구현
- report/dashboard schema 의미
- raw data artifact나 run artifact

## Config Groups

| Group | Purpose | Examples |
| --- | --- | --- |
| `entrypoints/` | `@hydra.main(config_name=...)`이 읽는 실행 시작점 | central SSL, fixed-feature, FL SSL, dataset pipeline |
| `execution_context/` | 실행 재료와 환경 | dataset asset, query source/view, FL split, runtime env |
| `strategy_axes/` | 교체 가능한 method/objective/runtime capability 축 | SSL objective, FSSL method, topology, trainable surface |
| `run_controls/` | 실행 크기와 안전장치 | smoke/main/reduced budget, sweep, output root |

대표 구조:

```text
conf/
├── entrypoints/
├── execution_context/
├── strategy_axes/
└── run_controls/
```

## Common Overrides

| Track | Common overrides |
| --- | --- |
| Central SSL | `strategy_axes/ssl_objective/consistency_method=<method>`, `strategy_axes/model_architecture/trainable_surface=<surface>`, `execution_context/query_labeled_budget=<budget>`, `run_controls/central_ssl/budget=<smoke|main>` |
| Fixed-feature supervised | `strategy_axes/classification/feature_space=<feature_space>`, `strategy_axes/classification/estimator=<estimator>`, `execution_context/query_labeled_budget=<budget>` |
| FL SSL | `run_controls/fl_ssl/budget=<smoke|reduced|main>`, `execution_context/fl_client_split=<split>`, `fl_method.composition_mode=<manual|method_owned>`, `strategy_axes/fssl_method=fedmatch`, `strategy_axes/fl_topology/shard_policy=<policy>` |

실행 가능한 조합 예시는 각 track README에서 본다.

- Central SSL: `scripts/experiments/central/ssl_control/README.md`
- Fixed-feature: `scripts/experiments/central/fixed_feature_control/README.md`
- FL SSL: `scripts/experiments/fl_ssl/README.md`

## Group Guide

### `entrypoints/`

Entrypoint config는 실행 스크립트의 시작점이다. 실행 조합만 소유하고 계산 의미는
소유하지 않는다.

주요 entrypoint:

- `entrypoints/central/ssl_control/run_query_ssl_control.yaml`
- `entrypoints/central/ssl_control/run_peft_supervised_control.yaml`
- `entrypoints/central/ssl_control/run_full_text_encoder_supervised_control.yaml`
- `entrypoints/central/fixed_feature_control/run_fixed_feature_baseline.yaml`
- `entrypoints/central/fixed_feature_control/run_fixed_feature_self_training_baseline.yaml`
- `entrypoints/fl_ssl/run_federated_simulation.yaml`
- `entrypoints/fl_ssl/materialize_fl_client_split.yaml`

### `execution_context/`

Execution context는 방법론 비교축이 아니라 실행 재료다.

| Group | Purpose |
| --- | --- |
| `dataset_asset/` | dataset pipeline에서 쓸 dataset asset |
| `query_data_source/` | query-domain JSONL source 주소록 |
| `query_labeled_budget/` | selected labeled source의 label budget artifact |
| `query_split/`, `query_view/` | central SSL query split과 weak/strong/original view surface |
| `fl_client_split/` | FL SSL materialized client split manifest preset |
| `embedding_adapter/` | embedding adapter runtime selection |
| `runtime_env/` | local/online/GPU/CPU runtime environment |
| `security_policy/` | plaintext/secure aggregation transport policy |

`data/**`와 `runs/**`는 config cleanup 과정에서 수정하지 않는다.

### `strategy_axes/`

Strategy axis는 실제로 교체 가능한 계산/정책 축이다.

| Group | Purpose |
| --- | --- |
| `ssl_objective/consistency_method/` | FixMatch, FreeMatch, AdaMatch 등 SSL objective 선택 |
| `ssl_objective/local_update_profile/` | local update backend, example surface, privacy guard 조합 |
| `model_architecture/{backbone,trainable_surface,peft,update_family,initial_checkpoint}/` | model/update family 관련 실행 축 |
| `classification/{feature_space,estimator}/` | fixed-feature supervised baseline 조합 |
| `fssl_method/` | method-owned FL SSL identity |
| `fl_topology/` | shard policy, labeled exposure, participation, server step, peer context 등 |

Method identity나 local SSL update recipe를 `fl_topology`에 두지 않는다. FL SSL method
의미는 `strategy_axes/fssl_method`와 `methods/federated_ssl/<method>/`가 소유한다.

### `run_controls/`

Run control은 특정 비교표 안에서 쓰는 실행 조건 묶음이다.

| Group | Purpose |
| --- | --- |
| `run_controls/central_ssl/budget/` | central SSL smoke/main budget |
| `run_controls/fl_ssl/budget/` | FL SSL smoke/reduced/main budget |
| `run_controls/fl_ssl/safety_and_sweeps/` | accidental long-run guard, seed/client-count sweep, resume/persistence control |

## Naming Rules

- `payload_adapter_kind`는 shared payload compatibility kind다.
- `round_runtime.update_family_name`은 실행 trainable state/update family다.
- `adapter_family_name`, `lora_classifier`, `diagonal_scale`은 active config 입력이 아니다.
- PEFT adapter mechanism은 `strategy_axes/model_architecture/peft`가 고르고, compose 후 namespace는 `cfg.peft_adapter`다.
- 중앙 SSL의 학습 가능한 모델 표면은 `strategy_axes/model_architecture/trainable_surface`가 고른다.
- 중앙 fixed-feature baseline은 `strategy_axes/classification/{feature_space,estimator}`로 고른다.

최종 vocabulary는
[../docs/architecture/target-method-runtime-structure.md](../docs/architecture/target-method-runtime-structure.md)를
우선한다.

## Current FL SSL Defaults

| Axis | Default |
| --- | --- |
| Manual baseline | `FixMatch + peft_text_encoder + FedAvg` |
| Payload/update family | `payload_adapter_kind=peft_classifier`, `update_family_name=peft_text_encoder` |
| Aggregation | `aggregation_backend_name=fedavg` |
| Materialized split | `shared_general_reddit_pc1024_alpha03_clients10` |
| Main split shape | `10 clients`, Dirichlet `alpha=0.3`, labeled `1024/class`, split seed `42` |
| Full budget | `30 rounds`, `local_epochs=1`, `batch_size=8`, `max_steps=50` |
| Smoke artifacts | `runs/_smoke/fl_ssl` |
| Comparison artifacts | `runs/fl_ssl` |

## Adding Or Changing A Strategy

1. 변경 축이 method, SSL algorithm, trainable state, aggregation, runtime context,
   report/view 중 어디인지 먼저 정한다.
2. 계산 의미는 `methods/`에 둔다.
3. 실행 조합 leaf와 파라미터만 `conf/strategy_axes/**`에 둔다.
4. runtime 차이가 필요하면 method 이름이 아니라 capability 이름의 adapter를 둔다.
5. entrypoint에는 leaf들을 조합하고 canonical request surface로 넘기는 값만 둔다.
6. Hydra compose test와 architecture guard를 함께 갱신한다.

YAML leaf는 얇아도 된다. 단, 기본값과 계약 의미를 여러 leaf가 중복 소유하면 안 된다.
빈 placeholder leaf는 만들지 않는다. 구현된 runtime callable이나 명확한 source of
truth가 있을 때만 leaf를 둔다.
