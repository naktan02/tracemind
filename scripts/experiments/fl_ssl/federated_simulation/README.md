# Federated Simulation

이 패키지는 `agent`와 `main_server` core를 직접 조합해 FL SSL loop를 재현하는
실험층이다. Production runtime이 아니라 production service와 method core를 연결해
검증하는 simulation package다.

긴 과거 설명은
`docs/notes/decisions/2026-05-28-archived-fl-simulation-package-readme.md`에
보관했다.

## 책임

- typed `SimulationRunRequest` 실행
- bootstrap, round loop, result/report 조립
- simulation 전용 runtime repository/cache wiring
- config가 고른 method/update-family/runtime capability를 호출
- report/artifact/diagnostics JSON 작성

## 제외

- FL SSL method identity와 local/server policy 의미
- Query SSL algorithm, PEFT text encoder training, aggregation core
- shared payload contract 의미
- live `agent`/`main_server` runtime 기본값

위 의미는 각각 `methods/`, `shared/`, `conf/`, runtime adapter package가 소유한다.

## 읽기 순서

1. `../run_federated_simulation.py` - Hydra entrypoint
2. `config_request.py` - config를 typed request로 변환
3. `simulation.py` - public simulation API
4. `flow/bootstrap.py` - initial state, split, runtime context
5. `flow/round_loop.py` - server step부터 summary assembly까지 round lifecycle
6. `flow/result_builder.py` - final evaluation/report input 조립
7. `adapters/` - method descriptor, local execution, server step/update bridge
8. `io/` - report/artifact/diagnostics serialization

`simulation.py`의 첫 화면은 runtime 준비, round loop 실행, result 조립 순서로
읽는다. `round_loop.py`의 첫 화면은 server step, round open, client selection,
peer context, client training, sync state, publication, validation, cleanup,
summary assembly phase를 따라간다.

## 경계 규칙

- 새 method를 위해 이 패키지에 `<method>_*.py`를 추가하지 않는다.
- `adapters/`는 runtime 차이만 bridge한다. method identity와 policy 의미는
  `methods/federated_ssl/<method>/`에 둔다.
- row sampling, aggregation weight, labeled row label 해석 같은 공통 의미는 각각
  `methods/federated_ssl`, `methods/federated`, `shared` helper를 사용한다.
- prototype 기반 방법론이 확정되면 runner 분기가 아니라 update-family leaf와
  `methods/` capability/core를 추가한다.

## 주요 축

- `fl_method.composition_mode=manual`: Query SSL lower axes를 직접 조합하는 baseline
- `fl_method.composition_mode=method_owned`: FedMatch 같은 descriptor-owned method
- `fl_data.source_mode=materialized_client_split`: 논문 비교용 고정 split
- `run_controls/fl_ssl/budget`: smoke/reduced/main runtime budget
- `strategy_axes/fl_topology/*`: shard, labeled exposure, participation, server
  step/update, peer context 등 FL split/topology와 round capability
- `strategy_axes/ssl_objective/local_update_profile`: local update backend,
  scoring/evidence, privacy guard recipe

`local_update_profile`과 `consistency_method`는 manual baseline/ablation에서는
의미 있는 lower axis다. `method_owned` 실행에서는 descriptor/capability plan이
local objective와 method protocol을 소유하므로, FedMatch request는 Query SSL
objective payload를 싣지 않는다.

Report protocol의 `runtime_selection`은 위 실행 축을 app/report가 읽기 쉬운
canonical summary로 남긴다. manual 실행은 `local_ssl_algorithm + update_family +
aggregation` 조합으로 기록하고, method-owned 실행은 descriptor 이름으로 기록한다.

상세 실행 명령은 상위 `scripts/experiments/fl_ssl/README.md`를 먼저 본다.
