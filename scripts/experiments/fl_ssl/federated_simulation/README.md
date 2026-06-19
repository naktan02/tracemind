# Federated Simulation

`federated_simulation/`은 FL SSL 실험 entrypoint가 호출하는 simulation package다.
`agent`, `main_server`, `methods`, `shared`의 core를 한 프로세스에서 조합해 FL
round를 재현하고 report artifact를 만든다.

직접 실행 명령과 조합 예시는 상위
[../README.md](../README.md)를 먼저 본다. 이 문서는 코드를 읽거나 수정할 때의
내부 지도다.

## What It Owns

- `SimulationRunRequest` 실행
- bootstrap, round loop, final result 조립
- simulation 전용 repository/cache/runtime resource wiring
- method descriptor, local update, server step, aggregation runtime bridge 호출
- report, diagnostics, projection, resume checkpoint serialization

## What It Does Not Own

- FL SSL method identity와 local/server policy 의미
- Query SSL objective, PEFT training, aggregation algorithm core
- shared payload contract 의미
- live `agent`/`main_server` runtime 기본값
- Hydra config 기본 조합

위 의미는 각각 `methods/`, `shared/`, `agent/`, `main_server/`, `conf/`의 owner
문서를 기준으로 본다.

## Read Path

실행 흐름을 따라갈 때는 아래 순서가 가장 짧다.

1. `../run_federated_simulation.py`: Hydra entrypoint, sweep 처리, output dir 결정
2. `config_request.py`: Hydra config를 typed `SimulationRunRequest`로 변환
3. `simulation.py`: runtime 준비, round loop 실행, final result 조립
4. `flow/bootstrap.py`: initial state, dataset split, runtime context 준비
5. `flow/round_loop.py`: communication round lifecycle
6. `flow/result_builder.py`: final evaluation과 report 입력 조립
7. `adapters/`: runtime boundary bridge
8. `io/`: report/artifact/diagnostics serialization

`simulation.py`의 첫 화면은 전체 orchestration을 보여준다. round 단계를 자세히
볼 때만 `flow/round_loop.py`로 내려간다.

## Package Map

| Path | 역할 |
| --- | --- |
| `models.py` | simulation request, shard, metric, config value object |
| `simulation_result_models.py` | 최종 simulation result shape |
| `config_request.py` | Hydra config에서 typed request 생성 |
| `data_source_request.py` | runtime split과 materialized split loading |
| `runtime_resources.py` | simulation resource cache |
| `sweep.py` | seed/client-count sweep request 생성 |
| `flow/bootstrap.py` | 초기 model revision, split, runtime state 준비 |
| `flow/round_loop.py` | server step, client selection, local training, aggregation, validation |
| `flow/result_builder.py` | run 종료 후 report 입력 조립 |
| `flow/state.py` | round 사이에 이어지는 active state |
| `adapters/client_training.py` | agent-side local training bridge |
| `adapters/server_step_execution.py` | server-side pre-round step bridge |
| `adapters/method_runtime.py` | manual/method-owned runtime plan bridge |
| `adapters/runtime_compatibility.py` | 실행 전 compatibility check |
| `adapters/evaluation.py` | simulation validation bridge |
| `io/` | split manifest, round report, final report, diagnostics, projection, resume checkpoint |

## Main Execution Flow

```text
Hydra config
-> SimulationRunRequest
-> method/runtime compatibility validation
-> bootstrap_simulation
-> run_one_round 반복
-> build_simulation_result
-> report/artifact writer
```

한 round 안에서는 대략 아래 순서로 흐른다.

```text
server step
-> round open
-> client selection
-> peer context 준비
-> client local training
-> update publication/aggregation
-> validation
-> round summary
```

## Runtime Modes

`fl_method.composition_mode=manual`은 SSL objective, update family, aggregation 같은
lower axis를 직접 조합하는 baseline/debug 경로다.

`fl_method.composition_mode=method_owned`는 FedMatch처럼
`methods/federated_ssl/<method>/`의 descriptor와 capability plan이 method policy를
소유하는 경로다.

두 모드 모두 이 package에서는 runtime bridge와 orchestration만 맡는다. method
policy를 이 디렉터리에 추가하지 않는다.

## Boundary Rules

- 새 method를 위해 이 패키지에 `<method>_*.py`를 추가하지 않는다.
- `adapters/`는 runtime 차이만 bridge한다.
- method identity와 policy 의미는 `methods/federated_ssl/<method>/`에 둔다.
- row sampling, aggregation weight, label 해석 같은 공통 의미는 각각
  `methods/federated_ssl`, `methods/federated`, `shared` helper를 사용한다.
- report payload 의미가 바뀌면 producer, consumer, `shared` contract 문서를 함께
  확인한다.
