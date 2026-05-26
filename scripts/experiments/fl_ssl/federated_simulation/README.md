# Federated Simulation

이 패키지는 `agent`와 `main_server` 코어를 직접 조합해 synthetic FL loop를
재현하는 실험층이다.

production runtime이 아니라, production service를 조립해서 검증하는
experiment package로 이해하면 된다.

중요:

- 이 패키지는 현재 로드맵에서 `시스템 FL 트랙`에 해당한다.
- `central PEFT-classifier` 논문 비교선을 먼저 닫은 뒤, winner를 FL로 옮길 때
  이 패키지가 직접적인 기준 entrypoint가 된다.

현재 v1에서 이 패키지의 기본 비교선은
`raw query rows -> local Query SSL PEFT-classifier -> server delta aggregation`이다.
기존 embedding/prototype scorer fallback은 제거했다. 나중에 prototype 기반
방법론이 실제 비교 method로 확정되면 method-owned capability로 다시 붙인다.

## 읽기 순서

1. `../run_federated_simulation.py`
   - Hydra entrypoint와 top-level config -> `SimulationRunRequest` 조립
2. `simulation.py`
   - `SimulationRunRequest` -> bootstrap -> round loop -> result 조립 흐름
3. `flow/bootstrap.py`
   - 초기 PEFT-classifier shared state와 manifest active pair 생성
4. `flow/round_loop.py`
   - round open, client execution 호출, publication, post-aggregation validation
5. `flow/result_builder.py`
   - final/client validation과 report 조립
6. `scripts/runtime_adapters/federated_server/`
   - main_server runtime adapter package
   - `runtime.py`: `SimulationServerRuntime`, active state load, round family 조립
   - `repositories.py`: simulation output root 기준 repository wiring
   - `initial_state_factory.py`: adapter family별 initial shared state factory.
     family-specific payload 의미는 `methods/adaptation/*`에 위임한다.
   - `task_config_surface.py`: simulation이 server round draft로 넘길 task config port
   - `round_request_mapper.py`: round task/open request 변환
7. `scripts/runtime_adapters/federated_agent/`
   - agent local runtime adapter package
   - `local_training.py`: local training request/service bridge
   - `artifact_store.py`: agent-local/server-owned artifact ref bridge
   - `training_example_mapper.py`: simulation row를 agent training example request로 변환
   - `row_validator.py`: `weak_strong_pair` 같은 row-source 요구사항 검증
   - `training_runtime.py`: local training backend resolve와 호출
   - `backend_resolver.py`: profile compatibility 검증용 backend resolve
8. `adapters/method_runtime.py`
   - `methods/federated_ssl/` descriptor를 simulation runtime adapter로 연결
9. `adapters/client_training.py`
   - client local training 실행, update 제출, selection quality summary 조립
10. `adapters/runtime_compatibility.py`
   - `methods/adaptation/runtime_objective_compatibility.py` dispatcher를 통해
     method-owned runtime/objective compatibility rule을 simulation request에 적용
11. `adapters/evaluation.py`
   - PEFT-classifier validation evaluator 실행 wiring
12. `adapters/sharding.py`
   - `methods/federated/shard_policy/`의 row adapter
13. `io/`
   - `run_artifact_writer.py`: model manifest 저장
   - `selection_diagnostics_writer.py`: selection diagnostics 저장
   - `report_metrics.py`: report/seed sweep 공통 metric payload와 통계 helper
   - `simulation_report_builder.py`: simulation report payload 조립
   - `simulation_report_writer.py`: simulation report JSON 저장

## 파일 역할

- `models.py`
  - simulation 전용 request/summary/config dataclass
- `simulation.py`
  - typed `SimulationRunRequest`를 실행하는 public `run_simulation_request`
- `flow/`
  - FL simulation bootstrap, round lifecycle, result/report 조립, 내부 active state context
- `adapters/`
  - method descriptor, client local execution, runtime compatibility, sharding,
    validation scorer 연결
- `io/`
  - JSONL row load, artifact writer, diagnostics writer, report builder/writer

`flow/`는 FL simulation 전용이다. 중앙 SSL과 공유될 수 있는 algorithm core나
contract가 생기면 이 패키지 안에서 공통화하지 않고 `methods/`, `shared/`,
`scripts/runtime_adapters/` 중 의미에 맞는 계층으로 올린다.

## 바로 조절 가능한 실험 축

- `local_update_profile`
  - `strategy_axes/fl/local_update_profile`에서 compose된다.
  - agent local update를 만드는 training/evidence/scoring/privacy 조합 profile이다.
  - `validation_*` 필드는 local update scorer와 분리된 validation evaluator를
    고른다. PEFT-classifier profile은 `peft_classifier_eval`을 사용하고,
    `lora_classifier_eval`은 old-run/config compatibility 이름으로만 남긴다.
- `ssl_method`
  - `strategy_axes/fl/method_descriptor`에서 compose된다.
  - method-owned 논문 method identity/report metadata와
    `methods/federated_ssl/` method spec을 선택한다.
  - 기본 manual baseline은 이 group을 compose하지 않는다.
  - descriptor config만 추가해도 새 논문 method runtime이 생기는 것은 아니다.
- `fl_method`
  - entrypoint-local section이며 `FederatedSslExecutionPlan`으로 해석된다.
  - `composition_mode=method_owned`에서는 상위 method가 client/server 정책을 소유한다.
  - `composition_mode=manual`에서는 lower-axis 조합 baseline/ablation임을 명시한다.
- `query_ssl_method`
  - `strategy_axes/ssl/consistency_method`에서 compose된다.
  - manual baseline과 `local_ssl_policy=query_ssl_method` 조합의 client SSL objective
    parameter source of truth다.
  - 기본은 `fixmatch_usb_v1`이며, FlexMatch/FreeMatch/PseudoLabel 비교는 이
    Hydra group 이름만 바꿔야 한다.
- `fl_data`
  - 기본 `source_mode=runtime_split_from_train`은 기존 debug 경로로 train JSONL을
    즉석 split한다.
  - 논문 비교는 먼저 `materialize_fl_client_split.py`로 client별 labeled/unlabeled
    JSONL과 manifest를 만든 뒤 `source_mode=materialized_client_split`으로 실행한다.
  - manifest에는 source selection, split seed, shard policy, client count,
    labeled source 선택 정책, `weak=text`, `strong=[aug_0, aug_1]` view schema와
    sha256 provenance가 report에 남는다.
- `labeled_exposure_policy`
  - 현재 entrypoint 기본값은 `shared_client_seed`다.
  - `client_local_split`은 legacy/ablation으로 유지한다.
  - `server_only_seed`는 materialized artifact와 request metadata를 보존하고,
    `server_step_policy=supervised_seed_step`과 함께 쓰면 round open 전에 server
    bootstrap rows로 supervised seed step을 실행한다.
- `client_participation_policy`
  - 기본은 `all_clients`다.
  - `fraction_random`, `fixed_count_random`은 FedMatch류 partial participation
    실험을 method와 무관하게 round loop에서 적용한다.
- `aggregation_weight_policy`
  - 기본은 기존 FedAvg와 같은 `example_count`다.
  - method가 요구하면 `uniform`이나 `accepted_count`를 capability plan으로 고른다.
- `server_step_policy`, `server_update_policy`, `peer_context_policy`,
  `update_partition_policy`, `local_ssl_policy`, `query_multiview_source`
  - method 전용 파일명이 아니라 공통 capability axis다.
  - manual baseline의 현재 실행 기본은 `server_step=none`, `peer_context=none`,
    `server_update=fedavg_merged_delta`, `update_partition=unified`,
    `local_ssl_policy=query_ssl_method`, `query_multiview_source=materialized_rows`다.
  - FedMatch method-owned slice는 `peer_context=fixed_probe_output_knn`와
    `server_update=fedmatch_partitioned`를 실행할 수 있다. 이때 local runtime이
    `partitioned_deltas`를 생산하고, server runtime이 PEFT-classifier
    `partitioned_delta_average` backend로 소비한다.
  - `server_update_policy`는 server가 merged/partitioned update payload를 어떤
    의미로 해석할지 나타내며, server-side supervised seed step 여부인
    `server_step_policy`와 분리한다.
  - `local_ssl_policy=query_ssl_method`는 `query_ssl_method.algorithm_name`을
    canonical local SSL policy 이름으로 쓴다. FixMatch류 파라미터를 FL capability
    config에 복제하지 않는다.
  - `peer_context=fixed_probe_output_knn`는 method `effective_parameters`의
    helper 개수와 refresh interval을 읽어 client별 helper context를 만든다.
    현재 slice는 이전 round client-local PEFT snapshot과 validation probe vector로
    KDTree 우선 nearest-neighbor helper client를 고르고, 선택된 helper snapshot의
    weak-view probability를 method-owned trainer에 주입한다.
  - `agent_generated_or_cached`는 live agent stored-event 경로가 weak/strong view를
    만들거나 캐시할 때 열 후속 축이다.
- `fl_client_split_materialization.labeled_policy`
  - split 생성 entrypoint 전용 값이다.
  - 기본 `mode=all`은 선택된 labeled source 전체를 bootstrap/client labeled pool로
    분배한다.
  - 라벨 데이터 일부만 쓰는 ablation은 `mode=count_per_class` 또는
    `mode=fraction`으로 manifest 생성 시 명시한다.
  - 같은 source와 seed에서 `count_per_class` budget은 class별 deterministic prefix를
    쓰므로 `25/class ⊂ 100/class ⊂ 400/class ⊂ 1024/class`처럼 nested subset으로
    해석한다. 여러 budget manifest를 만들 때는 overwrite를 피하도록 split id에
    `labels_pc<N>`를 명시한다.
- `security_policy`
  - 현재 simulation은 `plaintext`만 지원한다.
  - secure aggregation, DP, 암호화 artifact ref는 method가 아니라 runtime capability
    축으로 추가한다.
- `shard_policy`
- `federated_run_budget`
  - `run_controls/fl_ssl/budget=smoke` 산출물은 `runs/_smoke/fl_ssl` 아래에 둔다.
  - `run_controls/fl_ssl/budget=reduced`는 10 clients, 5 rounds 검증용 preset이다.
  - `run_controls/fl_ssl/budget=main`은 10 clients, 30 rounds full-budget preset이다.
- `seed_sweep.seeds`
- `seed_sweep.output_dir`
- `client_count_sweep.client_counts`
- `client_count_sweep.output_dir`
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
- `training_task.local_epochs`
- `training_task.batch_size`
- `training_task.max_steps`
- `query_ssl_method.unlabeled_batch_size`

`protocol.labeled_unlabeled_split`의 `actual_*` count/ratio는 client가 실제로 받은
row exposure 기준이다. `shared_client_seed`처럼 같은 labeled seed가 모든 client에
반복 노출되는 설정은 `unique_*` 필드를 함께 보고 source query 기준 규모를 해석한다.

예시:

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10 \
  fl_client_split_materialization.labeled_policy.mode=all
```

위 materialization 예시는 client count와 split policy를 맞추기 위한 것이며
round loop를 실행하지 않는다.

고정 split 실행:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<exposure_group>/<split_id>/manifest.json \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

위 smoke 실행은 `runs/_smoke/fl_ssl/...` 아래에 report를 남긴다. 웹/dashboard 기본
ingest는 `runs/_smoke/**`를 제외한다. 성능 방향 확인용 검증 실험은
`run_controls/fl_ssl/budget=reduced`를 사용하고, full-budget 비교가 필요할 때만
`run_controls/fl_ssl/budget=main`을 명시한다.

Seed sweep:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_seed_sweep \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

Client-count sweep:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_client_count_sweep \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.rounds=1
```

주의:

- `aggregation_backend_name`과 `adapter_family_name`은 `round_runtime.*`로 노출된다.
  기본값은 entrypoint leaf의 `peft_classifier` / `fedavg`다. `lora_classifier`
  leaf는 기존 override와 run artifact 호환을 위한 fallback이다.
- `local_update_profile`은 local update 조합을 고르고, server round 조합은
  `round_runtime.adapter_family_name`과 `round_runtime.aggregation_backend_name`을
  직접 override한다. high-level compose preset은 중복 source-of-truth를 피하기 위해
  두지 않는다.
- `diagonal_scale`/prototype scorer fallback은 현재 FL SSL simulation에서 제거했다.
  실제 방법론으로 다시 필요해지면 `methods/`의 method/capability 의미를 먼저
  정의하고, 그때 simulation adapter를 추가한다.
- FL SSL archived main split은 `run_controls/fl_ssl/budget=main`과
  `strategy_axes/fl/shard_policy=dirichlet_alpha03` 조합이었다. `alpha=0.1`은
  기본값이 아니라 마지막 stress/robustness 확인이 필요할 때만
  `strategy_axes/fl/shard_policy=dirichlet_alpha01`로 바꾼다.
- `run_controls/fl_ssl/budget=main`은 `10 clients`, `30 rounds`를 담는
  full-budget preset이다. 새 wiring/method 검증은 먼저 `1-round` smoke 또는 필요 시
  `5-round` reduced run으로 제한한다. 기본 smoke preset은 `4 clients`, `3 rounds`다.
- runner는 accidental long run을 막기 위해 `run_safety.max_total_rounds_without_ack`
  초과 총 예정 round를 시작 전에 차단한다. 총 예정 round는 단일 run이면
  `federated_run_budget.rounds`, seed/client-count sweep이면
  `federated_run_budget.rounds * sweep 항목 수`다. 장시간 실행이 명시 승인된 경우에만
  `run_safety.allow_long_run=true`와
  `run_safety.long_run_ack=ALLOW_FL_SSL_LONG_RUN`을 함께 override한다. 단일 main
  `30-round` 실행은 기본 guard 안에 들어오며, 그보다 큰 실행이나 sweep은 명시
  승인 대상이다.
- 현재 기본 `fl_method.composition_mode`는 `manual`이며 lower axes는
  `query_ssl_method.algorithm_name`, `round_runtime.aggregation_backend_name`,
  `round_runtime.adapter_family_name`에서 자동 파생된다. 기본 조합은
  `FixMatch + FedAvg + PEFT-classifier`이고 method descriptor를 참조하지 않는다.
  일회성 ablation은
  `strategy_axes/ssl/consistency_method=...`,
  `round_runtime.adapter_family_name=...`,
  `round_runtime.aggregation_backend_name=...`로 직접 고른다.
- manual `Query SSL + PEFT-classifier` 경로의 local optimizer step 수는
  `min(training_task.max_steps, training_task.local_epochs * full_epoch_steps)`로
  계산된다. `full_epoch_steps`는 labeled/unlabeled loader step 수의 max이며,
  loader step 수는 `training_task.batch_size`와
  `query_ssl_method.unlabeled_batch_size`로 바뀐다.
- 같은 경로의 기본 delta 전송은 `server_uploaded_artifact_ref`다. client update
  payload에는 LoRA/head inline weight를 싣지 않고 `aggregation_artifact::...` ref를
  남긴다. `inline_delta`는 legacy/debug compatibility 확인용으로만 쓴다.
- FedMatch처럼 client/server 정책을 함께 소유하는 상위 method는
  `fl_method.composition_mode=method_owned`와
  `strategy_axes/fl/method_descriptor=<method>`로 고르고, 하위
  FixMatch/FedAvg 축을 따로 고르지 않는다.
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
- report protocol에는 `embedding_adapter`와 `local_trainer_runtime` metadata를
  기록한다. 논문용 산출물은 `gpu_local + mxbai` 경로인지 이 필드로 확인하고,
  `hash_debug`/CPU smoke 결과를 성능 근거로 섞지 않는다.
- `client_pool_split`은 `fl_data.source_mode=runtime_split_from_train` fallback에서만
  train row를 `10% labeled / 90% unlabeled`로 다시 나누는 debug 값이다.
  `materialized_client_split`에서는 manifest의 client별 `labeled.jsonl`과
  `unlabeled.jsonl`을 그대로 사용하고 실제 비율은 report count로 확인한다.
- `seed_sweep.seeds` 기본값은 `[42, 43, 44]`이며 `report.seed_count=3`과
  일치해야 한다. seed sweep runner는 seed별 report와
  `reports/fl_ssl_seed_sweep.summary.json`을 남긴다.
- `client_count_sweep.client_counts` 기본값은 `[1, ..., 10]`이다. client-count
  sweep runner는 같은 seed/config에서 client 수만 바꾼 report와
  `reports/fl_ssl_client_count_sweep.summary.json`을 남긴다.
- `weak_strong_pair` example backend는 source row에 weak/strong view가 이미 있어야 한다.
  FL SSL v1은 scripts data pipeline에서 materialize한 `.with_views.jsonl`의
  `text + aug_0 + aug_1`를 `materialized_rows` source로 쓴다. live agent가
  weak/strong view를 생성하거나 캐시하는 경로는 후속 작업이다.
- `weak_strong_pair`는 generic multiview input backend다.
  real agent의 stored scored event 경로는 아직 weak/strong view를 저장하지 않으므로
  현재는 simulation/row-source 경로가 우선이다.
- validation은 `peft_classifier_eval`을 기본으로 사용하고 legacy
  `lora_classifier_eval`도 읽는다. 모든 validation row를 classifier
  forward로 평가하므로 accepted ratio는 성능 판단 metric이 아니다.
- 기본 manual `FixMatch + FedAvg + PEFT-classifier` 경로는
  `methods/ssl/algorithms/*` Query SSL algorithm과 실제 PEFT LoRA/classifier
  trainer를 호출해 artifact-ref delta update를 만든다. `agent-local://` ref는
  simulation round loop에서 server-owned `aggregation_artifact::` ref로
  upload/materialize한 뒤 서버에 제출한다. simulation output에는 다음 state가
  참조하는 누적 global LoRA/head parameter snapshot JSON도 저장된다.
  legacy pseudo-label selection 기반 inline executor는 `query_ssl_method`가 없는
  compatibility/debug 경로로만 유지한다. direct production API submission은 아직
  server-owned ref 또는 inline debug payload만 수락한다.

## newcomer 메모

- task shape를 바꾸고 싶으면 이 패키지부터 고치기보다
  `main_server/src/services/federation/rounds/boundary/models.py`와
  `scripts/runtime_adapters/federated_server/task_config_surface.py`,
  `scripts/runtime_adapters/federated_server/round_request_mapper.py`를 같이 본다.
- validation scorer를 바꾸고 싶으면 `adapters/evaluation.py`와
  `local_update_profile.validation_*` config를 함께 본다. training objective의
  `scorer_backend_name`은 pseudo-label/evidence selection 경로이고,
  PEFT-classifier 성능 평가는 `peft_classifier_eval`이 맡는다.
- threshold, scorer, privacy knob의 현재 노출 범위는
  [docs/strategy_surface_map.md](../../../../docs/strategy_surface_map.md)를
  먼저 보는 편이 빠르다.
