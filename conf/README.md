# Hydra Config Layout

`conf/`는 TraceMind 실험 entrypoint와 실행 조합 파라미터의 source of truth다.
YAML은 조합과 값만 소유하고, method 계산 의미나 runtime 구현은 `methods/`,
`agent`, `main_server`, `scripts/runtime_adapters`로 둔다.

## Groups

- `entrypoints/`: `@hydra.main(config_name=...)`이 읽는 실행 시작점.
- `execution_context/`: 데이터 자산, embedding adapter, runtime 환경.
- `strategy_axes/`: 중앙 SSL/FL SSL/prototype/adaptation에서 교체 가능한 전략 축.
- `run_controls/`: 논문 비교 track의 budget, output root, safety guard.

대표 구조:

```text
conf/
├── entrypoints/
├── execution_context/
├── strategy_axes/
│   ├── adaptation/
│   ├── fl/
│   ├── prototype/
│   ├── ssl/
│   └── trainable_state/
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
- PEFT adapter mechanism은 `strategy_axes/adaptation/peft_adapter`가 고르고,
  compose 후 namespace는 `cfg.peft_adapter`다.

최종 vocabulary는
`docs/architecture/target-method-runtime-structure.md`를 우선한다.

## FL SSL Contract

- 논문 method identity와 method-owned policy는
  `strategy_axes/fl/method_descriptor`와 `methods/federated_ssl/<method>/`가 소유한다.
- manual baseline은 method descriptor 없이 `query_ssl_method`,
  `local_update_profile`, `round_runtime.*` 조합으로 표현한다.
- local SSL algorithm은 `strategy_axes/ssl/consistency_method`가 고르고,
  실제 objective core는 `methods/ssl/algorithms/*`가 소유한다.
- update family와 runtime callable path는
  `strategy_axes/trainable_state/update_family`가 선언한다.
- server-side step 여부는 `strategy_axes/fl/server_step_policy`가 고르고,
  update family별 executor mapping은 `round_runtime.server_step_executors`가 소유한다.
- aggregation backend는 `round_runtime.aggregation_backend_name`으로 고른다.
- PEFT text encoder runtime 값은 `strategy_axes/adaptation/transformer_backbone`과
  `strategy_axes/adaptation/peft_adapter`에서 온다.
- client split 방식은 `strategy_axes/fl/shard_policy`, materialized split 선택은
  `strategy_axes/fl/materialized_split`이 소유한다.
- labeled rows를 server/client 어디에 노출할지는
  `strategy_axes/fl/labeled_exposure_policy`가 소유한다.
- client 수, round budget, output root는 `run_controls/fl_ssl/budget`이 소유한다.
- accidental long-run guard는 `run_safety`가 소유한다.

## Current Defaults

- FL SSL default/manual baseline:
  `FixMatch + peft_text_encoder + FedAvg`.
- `payload_adapter_kind=peft_classifier`
- `update_family_name=peft_text_encoder`
- `aggregation_backend_name=fedavg`
- main split: `10 clients`, Dirichlet `alpha=0.3`, split seed `42`.
- full budget: `30 rounds`, `local_epochs=1`, `max_steps=20`.
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
- `execution_context/query_split`은 central SSL query split 선택을 소유한다.
- `execution_context/query_view`는 weak/strong/original view JSONL surface를 고른다.
- FL materialized split은 data artifact를 읽기만 한다. `data/**`와 `runs/**`는
  config cleanup 과정에서 수정하지 않는다.

## Cleanup Rules

- 실행 조건 묶음이 하나뿐이고 entrypoint 전용이면 새 group을 만들지 않는다.
- YAML leaf는 얇아도 된다. 단, 기본값과 계약 의미를 여러 leaf가 중복 소유하면 안 된다.
- 빈 placeholder leaf는 만들지 않는다. 구현된 runtime callable이나 명확한 source of
  truth가 있을 때만 leaf를 둔다.
- namespace 이동은 Hydra config test, docs, architecture guard를 같이 닫는다.
- 긴 과거 판단 기록은 `docs/notes/**`로 archive하고, active README에는 현재 규칙만 남긴다.
