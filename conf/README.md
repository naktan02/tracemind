# Hydra Config Layout

`conf/`는 TraceMind 실험 entrypoint와 실행 조합 파라미터의 source of truth다.
YAML은 조합과 값만 소유하고, method 계산 의미나 runtime 구현은 `methods/`,
`agent`, `main_server`, `scripts/runtime_adapters`로 둔다.

## Groups

- `entrypoints/`: `@hydra.main(config_name=...)`이 읽는 실행 시작점.
- `execution_context/`: 데이터 자산, embedding adapter, runtime 환경.
- `strategy_axes/`: 중앙 SSL/FL SSL/adaptation에서 교체 가능한 전략 축.
- `run_controls/`: 논문 비교 track의 budget, output root, safety guard.

대표 구조:

```text
conf/
├── entrypoints/
├── execution_context/
│   ├── fl_client_split/
├── strategy_axes/
│   ├── classification/
│   ├── fl_topology/
│   ├── fssl_method/
│   ├── model_architecture/
│   └── ssl_objective/
└── run_controls/
```

## Naming Rules

- entrypoint config는 실행 스크립트의 시작점이다.
- execution context는 방법론 비교가 아니라 실행 재료다.
- strategy axis는 실제로 교체 가능한 계산/정책 축이다.
- run control은 특정 비교표 안에서 쓰는 실행 조건 묶음이다.
- `payload_adapter_kind`는 shared payload compatibility kind다.
- `round_runtime.update_family_name`은 실행 trainable state/update family다.
- `adapter_family_name`, `lora_classifier`, `diagonal_scale`은 active config 입력이 아니다.
- PEFT adapter mechanism은 `strategy_axes/model_architecture/peft`가 고르고,
  compose 후 namespace는 `cfg.peft_adapter`다.
- 중앙 SSL의 학습 가능한 모델 표면은
  `strategy_axes/model_architecture/trainable_surface`가 고른다. 중앙 Query SSL은
  `peft_text_encoder`와 `full_text_encoder` surface를 모두 실행할 수 있다. LoRA
  여부는 surface 이름이 아니라 PEFT adapter mechanism 축에서 결정한다.
- 중앙 fixed-feature 지도학습 baseline은
  `strategy_axes/classification/{feature_space,estimator}`로 고정 feature와 얕은
  classifier를 고른다. 계산 의미는 `methods/classification/fixed_feature/`가
  소유한다.
- `strategy_axes/fl_topology`는 client split/topology leaf와 공통 round capability
  leaf를 함께 둔다. method identity나 local SSL update recipe는 여기에 두지 않는다.

최종 vocabulary는
`docs/architecture/target-method-runtime-structure.md`를 우선한다.

## FL SSL Contract

- 논문 method identity와 method-owned policy는
  `strategy_axes/fssl_method`와 `methods/federated_ssl/<method>/`가 소유한다.
- manual baseline은 method descriptor 없이 `query_ssl_method`,
  `local_update_profile`, `round_runtime.*` 조합으로 표현한다.
- local SSL algorithm은 `strategy_axes/ssl_objective/consistency_method`가 고르고,
  실제 objective core는 `methods/ssl/algorithms/*`가 소유한다.
- local update recipe는 `strategy_axes/ssl_objective/local_update_profile`이
  소유한다. 이 leaf는 update backend, example surface, privacy guard를 고르고,
  pseudo-label/selection/scoring 세부값은 SSL/FSSL method가 소유한다.
  특정 논문 method를 뜻하는 값은 아니므로 `fssl_method` 아래에 두지 않는다.
- `strategy_axes/ssl_objective/local_ssl_policy`는 manual baseline/ablation에서만 직접
  고르는 축이다. Method-owned 실행은 descriptor required capability와 method surface
  default에서 local SSL policy를 파생한다.
- FL SSL simulation의 query multiview source는 현재 materialized row만 지원하므로
  Hydra leaf로 노출하지 않고 capability plan 기본값으로 둔다. live agent나 다른
  source가 실제 구현될 때만 별도 config 축을 연다.
- update family와 runtime callable path는
  `strategy_axes/model_architecture/update_family`가 선언한다.
- family-specific client/server round callable은 같은 leaf의
  `client_round_runtime` / `server_round_runtime` 아래에 선언한다. scripts bridge는
  family 이름으로 module path를 조립하지 않는다.
- update family가 runtime payload 일부를 training objective extra로 넘겨야 하면,
  entrypoint가 family 이름으로 분기하지 않고 update-family leaf의
  `training_objective_payload_scope` 선언을 따른다.
- server-side step 여부는 `strategy_axes/fl_topology/server_step`가 고르고,
  update family별 executor mapping은 `round_runtime.server_step_executors`가 소유한다.
- aggregation backend는 `round_runtime.aggregation_backend_name`으로 고른다.
- PEFT text encoder runtime 값은 `strategy_axes/model_architecture/backbone`과
  `strategy_axes/model_architecture/peft`에서 온다.
- client split 방식은 `strategy_axes/fl_topology/shard_policy`가 소유하고, 이미
  materialize된 FL client split artifact preset 선택은 `execution_context/fl_client_split`이
  소유한다.
- labeled rows를 server/client 어디에 노출할지는
  `strategy_axes/fl_topology/labeled_exposure`가 소유한다.
- 기본 client 수, round budget, output root는 `run_controls/fl_ssl/budget`이 소유한다.
  단, `execution_context/fl_client_split` preset은 manifest와 맞추기 위해
  `client_count`와 `bootstrap_ratio`를 함께 고정할 수 있다.
- accidental long-run guard, seed/client-count sweep, resume/persistence control은
  `run_controls/fl_ssl/safety_and_sweeps`가 소유한다.
- plaintext/secure aggregation 같은 transport/privacy policy는
  `execution_context/security_policy`가 소유한다.

## Current Defaults

- FL SSL default/manual baseline:
  `FixMatch + peft_text_encoder + FedAvg`.
- `payload_adapter_kind=peft_classifier`
- `update_family_name=peft_text_encoder`
- `aggregation_backend_name=fedavg`
- default materialized split:
  `shared_general_reddit_pc1024_alpha03_clients10`.
- main split: `10 clients`, Dirichlet `alpha=0.3`, labeled `1024/class`,
  split seed `42`.
- full budget: `30 rounds`, `local_epochs=1`, `batch_size=8`, `max_steps=50`.
- smoke artifacts: `runs/_smoke/fl_ssl`
- comparison artifacts: `runs/fl_ssl`

## Adding Or Changing A Strategy

1. 변경 축이 method, SSL algorithm, trainable state, aggregation, runtime context,
   report/view 중 어디인지 먼저 정한다.
2. 계산 의미는 `methods/`에 둔다.
3. 실행 조합 leaf와 파라미터만 `conf/strategy_axes/**`에 둔다.
4. runtime 차이가 필요하면 method 이름이 아니라 capability 이름의 adapter를 둔다.
5. entrypoint에는 leaf들을 조합하고 canonical request surface로 넘기는 값만 둔다.
6. Hydra compose test와 architecture guard를 함께 갱신한다.

## Dataset And View Context

- `execution_context/query_data_source`는 source별 JSONL 주소와 선택값을 소유한다.
- `query_data_selection.labeled`, `unlabeled`, `validation`, `test`가 실행 source를 고른다.
- `execution_context/query_labeled_budget`은 선택된 labeled source의 라벨 예산
  artifact만 바꾼다. unlabeled/eval source 조합은 `query_data_source`가 계속 소유한다.
- `execution_context/query_split`은 central SSL query split 선택을 소유한다.
- `execution_context/query_view`는 weak/strong/original view JSONL surface를 고른다.
- `execution_context/fl_client_split`은 FL SSL simulation이 읽을 materialized client
  split manifest와 그 manifest에 맞춰야 하는 source/policy/budget preset을 고른다.
- FL materialized split은 data artifact를 읽기만 한다. `data/**`와 `runs/**`는
  config cleanup 과정에서 수정하지 않는다.

## Cleanup Rules

- 실행 조건 묶음이 하나뿐이고 entrypoint 전용이면 새 group을 만들지 않는다.
- YAML leaf는 얇아도 된다. 단, 기본값과 계약 의미를 여러 leaf가 중복 소유하면 안 된다.
- 빈 placeholder leaf는 만들지 않는다. 구현된 runtime callable이나 명확한 source of
  truth가 있을 때만 leaf를 둔다.
- namespace 이동은 Hydra config test, docs, architecture guard를 같이 닫는다.
- 긴 과거 판단 기록은 `docs/notes/**`로 archive하고, active README에는 현재 규칙만 남긴다.
