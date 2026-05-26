# FL Runtime Implementation Checklist

이 문서는 FL 트랙의 현재 구현 상태와 다음 작업만 남기는 짧은 체크리스트다.
연구 순서는 `docs/project_execution_plan.md`, 코드 경계는
`docs/architecture/system-overview.md`, 전략 축은 `docs/strategy_surface_map.md`를
source of truth로 본다.

## 현재 상태

- `main_server`는 round lifecycle, update ingest, aggregation, publication
  scaffold를 갖고 있다.
- `agent`는 active round fetch, local training, pseudo-label selection, update
  upload scaffold를 갖고 있다.
- `scripts/experiments/fl_ssl`는 FL SSL simulation, seed sweep, client-count
  sweep, report dump, 기존 report artifact metadata verifier를 갖고 있다.
- 기본 FL SSL baseline은 descriptor 없는 `manual` mode다. report/index에는
  `execution_role=manual_baseline`으로 남기고 `descriptor_name`은 비워 둔다.
- `diagonal_scale`와 `lora_classifier` adapter family의 FedAvg core/projection은
  `methods/adaptation/<family>/`에 있다.
- 공통 분류 metric 계산은 `methods/evaluation`이 소유하고, FL report는 중앙 SSL과
  같은 metric shape를 재사용한다.

## Report / Evaluation

- [x] final/initial/round/client validation에 `loss`, accuracy, macro/weighted F1,
  balanced accuracy, worst-category metric, ECE/max-ECE를 남긴다.
- [x] round progression, best macro-F1 round, best loss round, round delta를 남긴다.
- [x] client split label distribution, entropy, labeled/unlabeled count와
  split skew summary를 남긴다.
- [x] client별 validation 요약에 train/labeled/unlabeled count, accepted ratio,
  update 생성 여부, update norm 진단을 함께 남긴다.
- [x] accepted-count 기반 aggregation weight proxy, zero-update client, update norm,
  communication proxy를 남긴다.
- [x] round index와 early-stop 후보 진단을 남긴다.
- [x] validation curve, primary 기준 best round, round/client time과 payload byte
  계측 상태를 report에 남긴다.
- [x] 중앙 SSL control report와 FL SSL main comparison report를 같은 ranking으로
  합치지 않는다.
- [x] `theta` 같은 method 내부 파라미터는 기본 report에 노출하지 않는다.
- [x] 기존 FL SSL report와 client-count sweep summary가 기대한 round budget,
  client count, split seed, shard policy/alpha, SSL method, adapter family,
  aggregation, delta format, round record/update count를 담는지
  `scripts/experiments/fl_ssl/verify_federated_report_artifacts.py`로 재검증할 수 있다.
  runtime metadata 도입 뒤 생성된 report는 같은 verifier로 GPU/mxbai metadata까지
  기대값으로 고정할 수 있다. LoRA-classifier artifact-ref run은 shared update 수,
  `aggregation_artifact::` ref, `agent-local://` ref 미노출, 최종 LoRA/head aggregate
  snapshot 존재까지 같은 verifier로 확인한다. FedMatch physical-partition run은
  `fedmatch_partitioned`, `partitioned`, `uniform`, `fixed_probe_output_knn`,
  `fedmatch_agreement`, `partitioned_deltas_artifact_ref`를 verifier expectation으로
  고정한다. Posthoc communication sidecar를 병합한 뒤에는 같은 verifier에서
  `fl_ssl_posthoc_communication_cost.v1`와 partitioned sparse S2C 추정 필드도
  고정한다.
- [x] 실제 FL report 산출물 shape를 result index 샘플로 고정하고 dashboard/index
  소비 필드를 확정했다. `result_index`는 `fl_ssl_main_comparison.report.json`에서
  track, method/algorithm, split/source, seed, client/round budget, shard alpha,
  adapter/aggregation, delta format, final/initial validation metric을 정규화한다.
  report에 GPU/mxbai runtime metadata가 있으면 embedding/trainer runtime 필드도
  dashboard filter로 노출한다.
- [x] client-local pseudo-label 품질 진단은 전체 unlabeled pool을 매번 평가하지
  않고 `diagnostic_view` deterministic subset으로 계산할 수 있다. 성능 평가는 계속
  validation/test row를 쓰며, report는 full pool 기준 `candidate_count`와 진단 subset
  기준 `diagnostic_candidate_count`를 분리해 기록한다.
- [x] FL SSL simulation report에 client round별 `timing_breakdown`을 남긴다.
  이는 batch/step마다 로그를 찍지 않는 wall-clock 구간 계측이며, model prepare,
  training loop, pseudo-label diagnostics, delta/update materialization, server submit
  같은 병목을 reduced run report에서 확인하기 위한 metadata다.
- [x] Simulation 기본 artifact 저장 정책은 server-owned aggregation artifact를
  canonical update source로 둔다. agent-local update 사본은 디버그용이므로
  `artifact_persistence.persist_agent_local_updates=false` 기본값에서는 저장하지
  않는다. 학습/aggregation 의미는 바꾸지 않고 local 사본 저장 비용만 제거한다.

주의: FL prototype score의 `loss`는 현재 raw score를 softmax 분포로 바꾼 NLL
proxy다. report의 `loss_kind`와 `score_distribution_kind`를 같이 읽어야 한다.

## Method Extension

- [x] method identity와 recipe metadata는 `methods/federated_ssl/<method>/`가
  소유한다.
- [x] registry는 `<method>/descriptor.py` convention import를 사용한다. 같은
  convention을 따르면 새 method 추가 시 registry 목록을 수정하지 않는다.
- [x] Hydra 실행 조합은 `conf/strategy_axes/fl/*`가 소유한다.
- [x] incompatible method/profile/runtime 조합은 simulation bootstrap 전에
  compatibility validator에서 실패한다.
- [x] FedMatch/FedLGMatch/(FL)^2 중 실제 구현할 첫 method를 확정한다.
  첫 method는 FedMatch이며, 현재 descriptor, capability surface, 원본 core/config
  snapshot, tensor local objective core, LoRA partitioned step core를 열었다.
- [x] 확정 method의 custom round-state exchange나 server policy capability가 필요한지
  먼저 문서화한다. FedMatch v1은 공통 `partitioned` update capability와 `uniform`
  aggregation weight를 요구하고, `sigma/psi` scheme, confidence filter/agreement
  pseudo-label/KDTree 우선 helper nearest-neighbor selection,
  supervised/unsupervised tensor loss는
  FedMatch method core에 고정했다. PEFT text-classifier trainer 한 step의 logical
  partition delta split과 method-owned local simulation wiring은
  `methods/adaptation/text_classifier/peft_encoder/federated_ssl/`의 method-neutral
  adapter-family slice가 소유한다. 기존 `lora_classifier/federated_ssl` path는
  compatibility shim이다.
  `fixed_probe_output_knn` helper client context 주입 seam과 KDTree 우선
  helper index, 이전 round client-local LoRA snapshot 기반 helper weak-probability
  provider는 열렸다. labels-at-server는 `server_only_seed + supervised_seed_step`
  server runtime과 client-local `psi` upload slice로 열렸다. sparse S2C/C2S는
  client-local previous partition snapshot과 partitioned global state 기준 simulation
  slice로 열렸고, 통신량은 posthoc estimate로 기록한다.
- [x] FedMatch method-owned smoke로 helper injection과 partitioned delta path를
  확인했다. 1-round smoke는 previous client snapshot이 없어 helper count 0이 정상이고,
  2-client 2-round smoke에서는 round 2에서 helper count/refreshed가 1.0으로 기록됐다.
  report verification CLI도 PASS했다.
- [x] FedMatch reduced run 전에 LoRA-classifier simulation 병목을 줄인다.
  현재 병목은 client/round마다 frozen transformer backbone/tokenizer를 재로딩하는 것,
  helper snapshot마다 helper model을 materialize하는 것, 전체 validation rows를
  fixed probe처럼 사용하는 것이었다. `fixed_probe_output_knn` probe는
  `peer_probe.selection_policy=label_balanced`, `max_rows=128` 기본값의 deterministic
  subset + manifest/hash로 계약화했다. 공통 runtime resource cache seam도
  `methods.common` protocol과 simulation run-scoped in-memory cache로 열었고,
  LoRA-classifier model builder는 cache가 있으면 tokenizer와 frozen backbone base를
  재사용한다. Helper snapshot별 materialized helper model도 같은 cache로 재사용한다.
  client-local pseudo-label quality 진단은 `diagnostic_view.max_rows=512` 기본값의
  deterministic subset으로 줄였다. 이는 manual Query SSL 경로와 FedMatch
  method-owned LoRA-classifier 경로가 같이 쓰는 simulation runtime capability다.
  FedMatch method-owned local budget은 main fair comparison에서
  `local_budget_policy=iteration_capped`와 `max_steps=20`을 쓴다. 원본
  labels-at-client budget은 `ssl_method.local_budget_policy=original_method`를
  명시한 별도 faithful run에서만 공통 labeled-anchored SSL budget primitive로
  계산한다.
  2026-05-26에는 helper/backbone cache를 client 경계에서 회수하고, round 간
  보존되는 peer/client partition snapshot vector를 float32 array로 압축했다.
  이후 `shared_general_reddit_pc100_alpha03_clients10` materialized split에서
  FedMatch method-owned `10 clients x 5 rounds` reduced run을 완료했고,
  posthoc communication backfill과 report verifier가 PASS했다.
  산출물은
  `runs/fl_ssl/fedmatch/fedmatch__lora_classifier__fedmatch_partitioned/labeled-szegeelim_general4_unlabeled-ourafla_reddit_labels_pc100_shared_client_seed42/clients10_rounds5/20260526T120100Z`다.
  최종 macro-F1은 `0.138327`로 초기 `0.265190`보다 낮아, 실행 경로 검증은
  닫혔지만 성능 해석은 별도 method/ablation 비교에서 다룬다.
- [x] server update/delta 해석 축과 local SSL objective 축을 분리했다.
  `server_update_policy=fedavg_merged_delta`는 현재 merged delta/FedAvg runtime이고,
  `fedmatch_partitioned`는 LoRA-classifier `partitioned_delta_average` simulation backend로
  shared update의 inline `partitioned_deltas` 또는 server-owned
  `partitioned_deltas_artifact_ref`를 소비한다. runtime 기본 경로는 큰 partitioned
  material을 artifact로 저장하고 payload에는 ref/metadata만 남긴다. 이 effective backend 해석은
  `methods/adaptation/<family>/federated_ssl/server_update_policy.py`가 소유한다.
  `local_ssl_policy=query_ssl_method`는 FixMatch/FlexMatch/FreeMatch 파라미터를
  기존 `query_ssl_method`에서 읽고, `fedmatch_agreement`는 FedMatch method package가
  소유한다. 현재 validator는 `fedmatch_partitioned + unified`를 막고,
  `fedmatch_partitioned + fixmatch + partitioned`를 capability surface에서 표현
  가능하게 열며 method-owned simulation에서는 FixMatch를 `psi` partition objective로
  실행할 수 있고,
  FlexMatch/FreeMatch처럼 state surface가 필요한 조합은 실행 전에 막는다.
- [x] 선택 전 capability matrix는
  `docs/contracts/fl_ssl_method_capability_matrix.md`에 정리했다. 현재 권장 첫 후보는
  payload family를 바꾸지 않는 FedMatch method-owned local objective다.
  FedMatch 외 method는 선택 전에는 method placeholder config나 production method
  폴더를 만들지 않는다.
- [x] `tests/architecture/test_layer_dependencies.py`가 method descriptor YAML과
  실제 `methods/federated_ssl/<method>/` 구현 파일 일치를 검증해 선택 전
  placeholder config를 막는다.

새 method 기본 변경 위치:

```text
methods/federated_ssl/<method>/
conf/strategy_axes/fl/method_descriptor/<method>.yaml
conf/strategy_axes/fl/local_update_profile/*.yaml      # 필요할 때만
conf/entrypoints/fl_ssl/run_federated_simulation.yaml  # round_runtime.* leaf override
tests/unit/test_methods_federated_ssl.py
tests/unit/test_scripts_hydra_configs.py
```

`agent`나 `main_server`에 method 이름 파일을 추가해야 한다면 먼저 capability seam이
부족한지 점검한다.

## Prototype / Scoring Extension

- [x] prototype build/scoring/evidence/training input core는 `methods/prototype/*`에
  분리되어 있다.
- [x] FL validation은 scoring/evidence 결과를 공통 classification report payload로
  변환한다.
- [ ] prototype-only 또는 prototype-SSL 평가 파일이 필요하면 `scripts`에
  entrypoint/thin wrapper로 추가한다.
- [ ] 두 개 이상 실험에서 안정적으로 공유되는 prototype 평가 metric만
  `methods/evaluation`으로 승격한다.

새 prototype 평가 기본 위치:

```text
scripts/experiments/prototype_analysis/        # prototype-only 분석
scripts/experiments/fl_ssl/federated_simulation/
methods/evaluation/                            # stable metric helper만
```

## Runtime Translation

- [x] `lora_classifier` family의 state/update shape와 inline/server-owned artifact-ref
  FedAvg core를 smoke로 검증했다.
- [x] FL simulation에서 `lora_pseudo_label_v1` local profile과
  `round_runtime.adapter_family_name=lora_classifier`,
  `round_runtime.aggregation_backend_name=fedavg` leaf 조합을 compose할 수 있다.
- [x] LoRA-classifier FedAvg는 두 라운드에서
  `previous global snapshot + round aggregated delta = next global snapshot`
  수식을 테스트로 고정했다.
- [x] FL simulation inline-delta 경로도 `sim_rev_0002 = sim_rev_0001 +
  round2 applied delta` 수식을 테스트로 고정했다.
- [x] manual `Query SSL + LoRA-classifier` simulation 경로는
  `methods/ssl/algorithms/*`와 실제 PEFT LoRA/classifier local trainer를 호출한다.
- [x] method-owned FedMatch LoRA simulation 경로는 manual Query SSL trainer를 우회해
  FedMatch local objective를 LoRA-classifier family slice에서 호출하고, merged delta와
  logical `sigma`/`psi` partition delta를 함께 제출한다.
- [x] `fedmatch_partitioned` server update adapter를 simulation에 연결해 partitioned
  LoRA-classifier delta를 aggregate하고 published state를 `sigma_plus_psi`로 만든다.
- [x] FixMatch 같은 stateless Query SSL local objective를 같은 partitioned sigma/psi
  loop의 `psi` objective로 주입하는 hybrid local trainer를 연다.
- [x] manual Query SSL client별 local optimizer step 수는
  `training_task.local_epochs`, `training_task.batch_size`,
  `training_task.max_steps`, `query_ssl_method.unlabeled_batch_size`로 동적으로
  바뀐다. FedMatch method-owned path도 기본은 같은 `iteration_capped` budget을 쓰며,
  원본 labels-at-client budget은 `original_method` policy로 선택 가능하다.
- [x] simulation adapter에서 agent-local LoRA artifact ref를 server-owned
  `aggregation_artifact::` ref로 upload/materialize하는 경로를 닫았다. 서버 direct
  submission은 여전히 server-owned ref만 수락한다.
- [x] 실제 PEFT executor 기준 LoRA 1-round smoke를 실행했다.
- [ ] winner method가 요구하는 shared family/state/update payload를 확정한다.
- [x] server accept 단계에서 update payload의 `model_id`, `base_model_revision`,
  `training_scope`, LoRA backbone/config/label schema가 active state와 맞는지
  family별 compatibility preflight로 확인한다.

## Main Comparison Gate

- [x] main split: `10 clients`, Dirichlet `alpha=0.3`, split `seed=42`,
  `30 rounds`. 기존 50-round report는 round/split/method/delta 기준으로 검증됐지만
  runtime metadata 도입 전 산출물이므로 현재 main preset source of truth는 아니다.
  현재 full-budget preset은 `30 rounds`다.
- [ ] final stress split: Dirichlet `alpha=0.1`, `30 rounds`. 이 항목은 마지막
  stress 확인 요소다. 기본/main 비교 조건은 `alpha=0.3`이며, 실행 시 후보와
  비교 조건을 별도로 명시한다.
- [x] accidental long run 방지: FL SSL runner는 총 예정 communication round가
  `run_safety.max_total_rounds_without_ack`를 넘으면 시작 전에 실패한다. 단일 run은
  `rounds`, seed/client-count sweep은 `rounds * sweep 항목 수`로 계산한다.
- [x] materialized split: 선택된 labeled source 전체와 unlabeled source 전체를
  client에 분배하고, 실제 labeled/unlabeled ratio는 report count로 기록한다.
- [x] `client_count=1..10` sweep runner와 summary JSON을 추가했다.
- [x] `client_count=1..10` 1-round summary는 report artifact verifier로
  `FixMatch + FedAvg + LoRA-classifier` metadata를 재검증했다.
- [ ] `gpu_local + mxbai` runtime metadata가 있는 main/sweep 산출물을 남긴다.
  현재는 `alpha=0.3` 같은 split의 1-round smoke와 5-round reduced ablation에서
  runtime metadata를 확인했다. runtime metadata 도입 전 50-round report와 1-round
  client-count sweep은 archive/reference로만 보고, final stress/ablation/sweep과
  full-budget main 실행은 후보와 비교 조건이 확정된 뒤 현재 `30-round` preset으로
  별도 실행한다. 2026-05-26 `clients10_rounds30/20260526T122807Z`는 사용자 중단으로
  초반 partial artifact만 남은 비검증 산출물이며 main evidence로 사용하지 않는다.
- [x] 새 FL simulation report protocol은 `embedding_adapter`와
  `local_trainer_runtime` metadata를 기록한다. 이후 논문용 산출물은 이 필드로
  `gpu_local + mxbai` 여부를 확인하고, `hash_debug`/CPU smoke 결과를 성능
  근거로 섞지 않는다.

## 완료 기준

- raw text와 개인 해석 상태는 agent-local boundary에 남는다.
- server는 round, aggregation, publication만 소유한다.
- scripts simulation은 production core를 복사하지 않고 호출한다.
- 새 method 추가 위치가 `methods/federated_ssl/<method>/`, `conf`, 필요한 capability
  adapter, test로 분명히 나뉜다.
- report 파일만으로 split, round progression, client variance, calibration,
  communication proxy를 확인할 수 있다.
