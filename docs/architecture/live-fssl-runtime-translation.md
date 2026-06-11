# Live FSSL Runtime Translation

이 문서는 FL SSL simulation에서 검증한 method/capability 구조를 live
agent/main_server runtime으로 옮기는 진행 지도를 둔다. 필드 shape의 source of
truth는 `shared/src/contracts/training_contracts.py`이고, capability vocabulary와
기본값 해석은 `methods/federated_ssl/`가 소유한다.

## 목표

Simulation과 live는 transport, local data source, process lifecycle이 다르다.
하지만 아래 의미는 같은 계약을 타야 한다.

```text
method descriptor
execution plan snapshot
capability plan snapshot
local objective core
update payload contract
server aggregation/publication semantics
```

새 FL SSL method를 추가할 때 agent/main_server가 method 이름으로 새 분기를 만들지
않는 것이 성공 기준이다.

## 현재 차이

| 축 | simulation | live | 상태 |
|---|---|---|---|
| method descriptor | `methods/federated_ssl` descriptor 사용 | active strategy의 `fssl_method`로 descriptor 검증 | 공유됨 |
| execution plan | Hydra에서 typed plan resolve | task에 snapshot 추가 시작 | 진행 중 |
| capability plan | Hydra/runtime config에서 resolve | task에 snapshot 추가 시작 | 진행 중 |
| client data source | materialized split/shard | agent-local analysis events/captured text | 의도적 차이 |
| local objective core | config callable bridge | Query SSL/PEFT current-task service | 일부 공유 |
| peer context | simulation round loop가 구성 | server가 `fssl_context`로 제공 | 일부 공유 |
| server lifecycle | `SimulationServerRuntime`이 main_server service 사용 | live main_server service 직접 사용 | 공유됨 |
| transport | in-process repository/runtime | HTTP/API + local repositories | 의도적 차이 |
| compatibility validation | bootstrap 전 검증 | task snapshot 기반 검증 추가 시작 | 진행 중 |

## Migration 순서

1. simulation/live capability gap을 active 문서로 고정한다.
2. `TrainingTaskPayload`가 FSSL execution/capability snapshot을 싣게 한다.
3. main_server가 method-owned round task를 만들 때 descriptor 기반 snapshot을
   생산한다.
4. agent가 snapshot이 있는 task를 local training 전에 검증한다.
5. live agent dispatcher를 method 이름이 아니라 capability와 update family 기준으로
   넓힌다.
6. simulation/live 공통 compatibility test를 추가한다.
7. architecture guard로 agent/main_server/scripts의 method-specific 분기를 막는다.
8. legacy context-only 경로는 artifact audit 뒤 compatibility layer로 격리하거나
   제거한다.
9. live smoke는 manual Query SSL -> method-owned no-peer -> peer context ->
   partitioned update 순서로 연다.

## 현 단계

`TrainingTaskPayload`에는 하위 호환용 `fssl_method`/`fssl_context`가 남아 있다.
새 live producer는 `fssl_execution`과 `fssl_capability_plan`을 함께 기록한다. 기존
task는 계속 읽을 수 있지만, 새 snapshot이 있는 task는 agent가 method/capability
drift를 먼저 검증한다. `fssl_execution.runtime_surface`는 live dispatcher가
사용하는 `update_family_name`, `payload_adapter_kind`, `aggregation_backend_name`을
담는다.
