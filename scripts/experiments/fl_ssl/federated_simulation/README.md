# Federated Simulation

이 패키지는 `agent`와 `main_server` 코어를 직접 조합해 synthetic FL loop를
재현하는 실험층이다.

production runtime이 아니라, production service를 조립해서 검증하는
experiment package로 이해하면 된다.

중요:

- 이 패키지는 현재 로드맵에서 `시스템 FL 트랙`에 해당한다.
- `central LoRA classifier` 논문 비교선을 먼저 닫은 뒤, winner를 FL로 옮길 때
  이 패키지가 직접적인 기준 entrypoint가 된다.

현재 v1에서 이 패키지의 기본 비교선은
`embedding -> global classifier -> local interpretation`이다.
shared adapter와 prototype scoring은 비교 실험 축으로 함께 유지한다.

## 읽기 순서

1. `../run_federated_simulation.py`
   - Hydra entrypoint와 top-level config -> `SimulationRunRequest` 조립
2. `simulation.py`
   - `SimulationRunRequest` -> bootstrap -> round loop -> result 조립 흐름
3. `flow/bootstrap.py`
   - 초기 shared state, prototype, manifest active pair 생성
4. `flow/round_loop.py`
   - round open, client local training, update submit, publication
5. `flow/result_builder.py`
   - final/client validation과 report 조립
6. `scripts/runtime_adapters/federated_server/`
   - main_server runtime adapter package
   - `runtime.py`: `SimulationServerRuntime`, active state load, round family 조립
   - `repositories.py`: simulation output root 기준 repository wiring
   - `prototype_rebuild_bridge.py`: prototype rebuild runtime과 adapter seam
   - `initial_state_factory.py`: initial shared state 생성
   - `task_config_surface.py`: simulation이 server round draft로 넘길 task config port
   - `round_request_mapper.py`: round task/open request 변환
7. `scripts/runtime_adapters/federated_agent/`
   - agent local runtime adapter package
   - `training_example_mapper.py`: simulation row를 agent training example request로 변환
   - `row_validator.py`: `weak_strong_pair` 같은 row-source 요구사항 검증
   - `training_runtime.py`: local training request/service bridge
   - `backend_resolver.py`: profile compatibility 검증용 backend resolve
8. `adapters/method_runtime.py`
   - `methods/federated_ssl/` descriptor를 simulation runtime adapter로 연결
9. `adapters/evaluation.py`
   - validation scoring과 training example 재구성
10. `adapters/sharding.py`
   - `methods/federated/shard_policy/`의 row adapter
11. `io/`
   - `run_artifact_writer.py`: prototype pack과 model manifest 저장
   - `selection_diagnostics_writer.py`: selection diagnostics 저장
   - `report_metrics.py`: report/seed sweep 공통 metric payload와 통계 helper
   - `simulation_report_builder.py`: simulation report payload 조립
   - `simulation_report_writer.py`: simulation report JSON 저장

## 파일 역할

- `models.py`
  - simulation 전용 request/summary/config dataclass
- `simulation.py`
  - public `run_simulation_request`와 legacy `run_simulation` wrapper
- `flow/`
  - FL simulation bootstrap, round loop, result/report 조립, 내부 active state context
- `adapters/`
  - method descriptor, round task config, sharding, validation scorer 연결
- `io/`
  - JSONL row load, artifact writer, diagnostics writer, report builder/writer

`flow/`는 FL simulation 전용이다. 중앙 SSL과 공유될 수 있는 algorithm core나
contract가 생기면 이 패키지 안에서 공통화하지 않고 `methods/`, `shared/`,
`scripts/runtime_adapters/` 중 의미에 맞는 계층으로 올린다.

## 바로 조절 가능한 실험 축

- `local_update_profile`
  - `strategy_axes/fl/local_update_profile`에서 compose된다.
  - agent local update를 만드는 training/evidence/scoring/privacy 조합 profile이다.
- `round_runtime_profile`
  - `strategy_axes/fl/round_runtime_profile`에서 compose된다.
  - server round runtime의 adapter family와 aggregation backend 조합을 소유한다.
- `ssl_method`
  - `strategy_axes/fl/method_descriptor`에서 compose된다.
  - method identity/report metadata와 `methods/federated_ssl/` method spec을
    선택한다.
  - descriptor config만 추가해도 새 논문 method runtime이 생기는 것은 아니다.
- `shard_policy`
- `federated_run_budget`
- `seed_sweep.seeds`
- `seed_sweep.output_dir`
- `client_pool_split.labeled_ratio`
- `client_pool_split.unlabeled_ratio`
- `round_runtime.adapter_family_name`
- `round_runtime.aggregation_backend_name`
- `round_runtime.classifier_head_bootstrap_logit_scale`
- `training_task.objective.confidence_threshold`
- `training_task.objective.margin_threshold`
- `training_task.objective.training_backend_name`
- `training_task.objective.algorithm_profile_name`
- `training_task.objective.example_generation_backend_name`
- `training_task.objective.evidence_backend_name`
- `training_task.objective.scorer_backend_name`
- `training_task.objective.score_policy_name`
- `training_task.objective.score_top_k`
- `training_task.objective.acceptance_policy_name`
- `training_task.objective.privacy_guard_name`
- `training_task.selection_policy.max_examples`
- `validation.scorer_backend_name`
- `validation.score_policy_name`
- `validation.score_top_k`

예시:

```bash
python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=main \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  strategy_axes/fl/method_descriptor=fedavg_pseudo_label \
  strategy_axes/fl/experiment_profile=none \
  strategy_axes/fl/local_update_profile=prototype_top1_confidence_v1 \
  training_task.objective.privacy_guard_name=noop
```

Seed sweep:

```bash
python -m scripts.experiments.fl_ssl.run_federated_seed_sweep \
  run_controls/fl_ssl/budget=main \
  strategy_axes/fl/shard_policy=dirichlet_alpha03
```

주의:

- `aggregation_backend_name`과 `adapter_family_name`은 `round_runtime.*`로 노출된다.
  기본값은 `round_runtime_profile`에서 파생된다.
- `local_update_profile` 또는 `round_runtime_profile`을 직접 바꿀 때는
  `strategy_axes/fl/experiment_profile=none`을 함께 지정한다. preset 이름표까지
  함께 바꾸려면 `strategy_axes/fl/experiment_profile=<profile>`로 시작한다.
- FL SSL main split은 `run_controls/fl_ssl/budget=main`과
  `strategy_axes/fl/shard_policy=dirichlet_alpha03`를 기본 조합으로 본다.
  stress split은 `strategy_axes/fl/shard_policy=dirichlet_alpha01`로 바꾼다.
- `run_controls/fl_ssl/budget=main`은 `10 clients`, `50 rounds`를 main budget으로
  쓰고, 기본 smoke preset은 `4 clients`, `3 rounds`를 쓴다.
- `strategy_axes/fl/method_descriptor=fedavg_pseudo_label`는 현재 구현된 baseline method다.
- method spec source of truth는 `methods/federated_ssl/`이다.
  이 package의 `adapters/method_runtime.py`는 method spec을 simulation runtime으로 연결하는
  adapter만 둔다.
- 후보 논문 method는 확정 전까지 descriptor/config/runtime 파일을 미리 만들지 않는다.
  실제 method core는 `methods/federated_ssl/<method>/`를 시작점으로 두고,
  재사용 가능한 계산만 축별 `methods` 패키지로 승격한다. `agent`와 `main_server`에는
  필요한 capability adapter만 둔다.
- `report.track=fl_ssl_main_comparison`은 중앙 SSL control과 분리된
  `reports/fl_ssl_main_comparison.report.json`을 남긴다. 현재 report shape는
  entrypoint-local section이므로 별도 Hydra group이 아니다.
- report는 global validation `macro_f1`, client validation shard 기준
  `worst_client_macro_f1`, ECE, client update envelope 수 기반 communication
  cost proxy, per-client macro-F1 variance를 포함한다.
- `client_pool_split`은 각 client shard 안에서 `10% labeled / 90% unlabeled`
  pool을 deterministic하게 나눈다. 현재 `fedavg_pseudo_label` baseline은
  `unlabeled` partition만 pseudo-label training 후보로 사용한다.
- `seed_sweep.seeds` 기본값은 `[42, 43, 44]`이며 `report.seed_count=3`과
  일치해야 한다. seed sweep runner는 seed별 report와
  `reports/fl_ssl_seed_sweep.summary.json`을 남긴다.
- `weak_strong_pair` example backend는 source row에 weak/strong view가 이미 있어야 한다.
  현재 기본 JSONL row shape는 그 view를 따로 저장하지 않으므로, 별도 multiview row 공급이 없으면
  `prototype_rescore`를 계속 써야 한다.
- `weak_strong_pair`는 generic multiview input backend다.
  real agent의 stored scored event 경로는 아직 weak/strong view를 저장하지 않으므로
  현재는 simulation/row-source 경로가 우선이다.
- validation의 accepted_ratio는 raw score threshold가 아니라
  runtime과 같은 `evidence backend -> acceptance policy` 경로로 계산한다.
## newcomer 메모

- task shape를 바꾸고 싶으면 이 패키지부터 고치기보다
  `main_server/src/services/federation/rounds/boundary/models.py`와
  `scripts/runtime_adapters/federated_server/task_config_surface.py`,
  `scripts/runtime_adapters/federated_server/round_request_mapper.py`를 같이 본다.
- validation scorer를 바꾸고 싶으면 `adapters/evaluation.py`와
  `shared/src/contracts/training_contracts.py`의 objective 축을 함께 본다.
- threshold, scorer, privacy knob의 현재 노출 범위는
  [docs/strategy_surface_map.md](../../../../docs/strategy_surface_map.md)를
  먼저 보는 편이 빠르다.
