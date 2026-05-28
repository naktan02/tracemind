# FL SSL 실행

이 폴더는 FL SSL 실험 entrypoint만 둔다. FL method identity와 method-only
정책은 `methods/federated_ssl/`, SSL objective core는 `methods/ssl`,
PEFT text encoder 계산 core는
`methods/adaptation/peft_text_encoder`, 실행 조합과 파라미터는
`conf/` Hydra config가 소유한다.

현재 기본 실행은 논문 method가 아니라 manual baseline이다.

```text
FixMatch USB objective + PEFT text encoder local update + FedAvg aggregation
```

FedMatch/FedLGMatch 같은 method-owned FL SSL method는 선택/구현 전까지
descriptor/config placeholder를 만들지 않는다.
현재 FedMatch는 첫 method로 선택되어 `methods/federated_ssl/fedmatch/`의
capability surface만 열려 있다.

## 먼저 확인

실제 실행 전에 compose 결과를 확인한다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation --cfg job
```

기본값:

```text
runtime_env=gpu_local
embedding_adapter=mxbai
budget=smoke
client_count=4
rounds=3
shard_policy=label_dominant
labeled_exposure_policy=shared_client_seed
client_participation_policy=all_clients
composition_mode=manual
query_ssl_method=fixmatch_usb_v1
local_update_profile=peft_pseudo_label_v1
update_family=peft_text_encoder
aggregation_backend=fedavg
output_dir=runs/_smoke/fl_ssl
```

`run_controls/fl_ssl/budget=smoke`는 wiring 검증용 산출물을
`runs/_smoke/fl_ssl` 아래에 둔다. `budget=reduced`는 `10 clients`, `5 rounds`
확인용 preset이고, `budget=main`은 `10 clients`, `30 rounds` full-budget
preset이다. 성능 방향 확인은 `budget=reduced`, full-budget 실행이 필요할 때만
`budget=main`을 명시한다.

## 데이터 선택

FL도 중앙 SSL과 같은 `execution_context/query_data_source` catalog를 사용한다.
라벨/논라벨/validation/test는 `query_data_selection.*`으로 고른다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10
```

중앙 SSL과의 차이는 소비 방식이다. 중앙 SSL은 선택된 JSONL을 바로 학습에
사용하지만, FL 논문 비교는 선택된 labeled/unlabeled/validation/test JSONL을 먼저
client split manifest로 materialize한다.

```text
query_data_selection
-> query_source.train_jsonl / unlabeled_jsonl / validation_jsonl / test_jsonl
-> materialize_fl_client_split
-> data/datasets/fl_client_splits/<exposure_group>/<split_id>/manifest.json
-> run_federated_simulation fl_data.source_mode=materialized_client_split
```

`<exposure_group>`은 실행자가 고르는 labeled exposure 표면이다.
`client_local_labeled`는 client-local labeled split,
`shared_client_labeled`는 모든 client가 같은 public labeled seed를 보는 split,
`server_only_labeled`는 server/bootstrap boundary에만 labeled seed를 두는 split이다.
manifest 내부에는
canonical policy name(`client_local_split`, `shared_client_seed`, `server_only_seed`)이
남는다.

`query_data_selection.labeled`는 client labeled pool과 bootstrap labeled pool의
원본이다. `query_data_selection.unlabeled`는 client unlabeled pool의 원본이다.
`query_data_selection.validation`과 `query_data_selection.test`는 global/client
평가용 row 원본으로 manifest에 고정된다.

## 고정 Client Split 생성

논문 비교용 FL 실행은 먼저 client별 labeled/unlabeled split을 materialize한다.
기본 비교 split은 Dirichlet alpha 0.3, seed 42를 기준으로 둔다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  run_controls/fl_ssl/budget=smoke \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10 \
  fl_client_split_materialization.labeled_policy.mode=all
```

출력되는 `manifest_json=...` 값을 다음 실행의 `fl_data.split_manifest`로 넘긴다.

라벨 budget ablation은 이미 materialize된 labeled-with-views source에서
`fl_client_split_materialization.labeled_policy.mode=count_per_class`와
`count_per_class=<N>`만 바꿔 만든다. 새 역번역은 하지 않는다. 같은 seed에서
class별 deterministic prefix를 쓰므로 `25/class`, `100/class`, `400/class`,
`1024/class` budget은 nested subset이다. split id에는 `labels_pc<N>`를 넣어 서로
덮어쓰지 않게 한다.

모든 client가 같은 public labeled seed를 보고, unlabeled만 client별 non-IID로
나누는 `shared_client_seed` split은 labeled exposure policy만 바꿔 별도
manifest로 materialize한다. split id에는 `shared_client_seed_`가 들어가 기존
`client_local_split` manifest를 덮어쓰지 않는다.
`server_only_seed`는 materialization과 request metadata를 지원한다. method-owned
descriptor, `server_step_policy=supervised_seed_step`, client-unlabeled regime을 함께
고르면 simulation이 round open 전에 bootstrap labeled rows로 supervised seed step을
실행한다.

생성된 shared-client main 후보 split은 simulation에서 `materialized_split` 축으로
고를 수 있다. 이 축은 manifest path뿐 아니라 `query_data_selection.*`도 같이
고정한다.

```bash
uv run python scripts/experiments/fl_ssl/run_federated_simulation.py \
  run_controls/fl_ssl/budget=reduced \
  strategy_axes/fl/materialized_split=shared_general_reddit_pc100_alpha03_clients10
```

현재 열린 selector는 10 clients, alpha=0.3, `shared_client_seed` exposure의
`pc25/pc100/pc400/pc1024`다. source pair는 `shared_reddit_reddit_*`와
`shared_general_reddit_*`를 쓴다. `shared_general_reddit_*`는 labeled source가
`szegeelim_general4`, unlabeled pool과 validation/test가 `ourafla_reddit`인
조합이다. `client_local_labeled`는 현재 main 후보가 아니므로 별도 8세트를 만들지
않는다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  run_controls/fl_ssl/budget=main \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  strategy_axes/fl/labeled_exposure_policy=shared_client_seed
```

## 실행 명령 구조

FL SSL 실행 명령은 보통 다섯 축을 동시에 고른다.

```text
1. method/composition: manual lower-axis 조합인지, FedMatch 같은 method-owned인지
2. data source: labeled/unlabeled/validation/test source 선택
3. data split: runtime split인지, materialized client split인지
4. run condition: client_count, rounds, seed, shard policy
5. runtime family: update_family, aggregation_backend, local_update_profile
```

현재 기본값은 `fl_method.composition_mode=manual`이다. 즉 별도 override가 없으면
상위 FL method를 고르는 것이 아니라 아래 lower axes를 조합한다.

```text
query_ssl_method + round_runtime.update_family_name + round_runtime.aggregation_backend_name
```

기본 조합은 `fixmatch_usb_v1 + peft_text_encoder + fedavg`다.

## 단일 Simulation 실행

고정 split을 쓰는 1-round smoke 예시:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<exposure_group>/<split_id>/manifest.json \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

이 명령은 방법론을 새로 고르는 예시가 아니라, 기본 manual 조합
`FixMatch + PEFT text encoder + FedAvg`를 고정 client split에서 `10 clients`,
`1 round`로 실행하는 예시다. `federated_run_budget.client_count`와
`federated_run_budget.rounds`는 임의 override 가능하다. 단, long-run guard를
넘는 실행은 현재 정책상 피한다.

짧은 reduced run 예시:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<exposure_group>/<split_id>/manifest.json
```

산출 경로는 아래 구조로 쌓인다.

```text
runs/fl_ssl/
  manual_baselines/
    fixmatch_usb_v1__peft_text_encoder_lora__fedavg/
      labeled-ourafla_reddit_unlabeled-ourafla_reddit_shared_client_seed42/
        clients10_rounds1/
        clients10_rounds5/
  fedmatch/
    fedmatch__peft_text_encoder_lora__fedmatch_partitioned/
      labeled-szegeelim_general4_unlabeled-ourafla_reddit_labels_pc100_shared_client_seed42/
        clients10_rounds5/
```

manual baseline의 composition 폴더는 `query_ssl_method + update family +
aggregation_backend`를 쓴다. `peft_text_encoder`처럼 PEFT mechanism이 별도 축인
family는 `peft_text_encoder_lora`처럼 실제 `peft_adapter_name`을 붙인다. FedMatch 같은
`method_owned` run은 Query SSL lower axis가 아니라
`method_descriptor + update family + method-derived server update policy`를 쓴다.

split 폴더명은 labeled/unlabeled source, labeled exposure, seed만 사람이 읽는
이름으로 남긴다. Dirichlet alpha, manifest hash, 전체 source JSONL 경로 같은 세부
조건은 `reports/fl_ssl_main_comparison.report.json`의 `protocol.fl_data_source`와
materialized split manifest를 source of truth로 본다.

`budget=smoke` 산출물은 같은 하위 구조를 `runs/_smoke/fl_ssl/` 아래에 만든다.

각 run의 canonical report는
`reports/fl_ssl_main_comparison.report.json`이다.

FedMatch physical partition smoke/reduced report는 preset을 새로 만들지 않고 leaf
override 조합을 verifier로 고정한다. 최소 검증 축은
`ssl_method=fedmatch`, `update_partition_policy=partitioned`,
`aggregation_weight_policy=uniform`,
`peer_context_policy=fixed_probe_output_knn`,
`ssl_method.implementation_status=partitioned_trainable_state_slice_v1`,
`ssl_method.local_budget_policy=iteration_capped`,
`ssl_method.parameter_override_status=original`,
`partitioned_deltas_artifact_ref`다. 정상 report 생성 경로는 artifact 기반
communication estimate를 report에 함께 쓰며,
`expected_communication_estimate_schema_version`과
`expect_partitioned_sparse_s2c_estimates`로 sparse S2C 추정 필드도 고정한다.

```bash
uv run python -m scripts.experiments.fl_ssl.verify_federated_report_artifacts \
  --report <run-dir>/reports/fl_ssl_main_comparison.report.json \
  --expected-fl-method-name fedmatch \
  --expected-fl-method-descriptor-name fedmatch \
  --expected-fl-method-execution-role method_owned \
  --expected-federated-ssl-method fedmatch \
  --expected-ssl-method-implementation-status partitioned_trainable_state_slice_v1 \
  --expected-ssl-method-scenario labels-at-client \
  --expected-ssl-method-local-budget-policy iteration_capped \
  --expected-ssl-method-parameter-override-status original \
  --expected-server-update-policy fedmatch_partitioned \
  --expected-update-partition-policy partitioned \
  --expected-aggregation-weight-policy uniform \
  --expected-peer-context-policy fixed_probe_output_knn \
  --expected-local-ssl-policy fedmatch_agreement \
  --expect-partitioned-update-artifact-refs \
  --expect-no-agent-local-update-refs \
  --expected-communication-estimate-schema-version fl_ssl_artifact_communication_cost.v1 \
  --expect-partitioned-sparse-s2c-estimates
```

## Sweep

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

`materialized_client_split` 기반 client-count sweep은 client count별 manifest가
필요하다. `client_count_sweep.split_manifest_by_client_count`에 count별
manifest를 지정한다.

## 다른 Manual Baseline 조합

manual baseline에서는 lower axes를 직접 고른다. `composition_mode=manual`은
기본값이지만, 문서/로그 가독성을 위해 명시해도 된다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  fl_method.composition_mode=manual \
  strategy_axes/ssl/consistency_method=flexmatch_usb_v1 \
  strategy_axes/trainable_state/update_family=peft_text_encoder \
  round_runtime.aggregation_backend_name=fedavg \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<exposure_group>/<split_id>/manifest.json \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1 \
  training_task.max_steps=20
```

위 명령은 `FlexMatch + PEFT text encoder + FedAvg` manual 조합이다.
client 수와 round 수는 `federated_run_budget.client_count`,
`federated_run_budget.rounds`로 바꾼다. client가 중앙으로 update를 보내기 전
local optimizer step 상한은 `training_task.max_steps`로 조절한다. 고정 split을
쓰려면 예시처럼 `fl_data.source_mode=materialized_client_split`과
`fl_data.split_manifest=...`를 함께 넘긴다.

`diagnostic_view`는 학습 입력이 아니라 pseudo-label 품질 진단 입력을 제한하는
공통 runtime 설정이다. 기본값은 `diagnostic_view.max_rows=512`이며, client full
unlabeled pool 크기는 report의 `candidate_count`로 유지하고 진단 subset 크기는
`diagnostic_candidate_count`로 따로 남긴다. global/client 성능 평가는 이 설정과
무관하게 validation/test split으로 수행한다.

각 client round report에는 `timing_breakdown`도 남는다. 이 값은 stdout 진행 로그가
아니라 report metadata이며, `core_model_prepare_seconds`,
`core_training_loop_seconds`, `core_pseudo_label_diagnostics_seconds`,
`update_upload_materialize_seconds`, `server_update_submit_seconds` 같은 구간별
wall-clock 시간을 기록한다. batch/step 단위 로그나 GPU sync를 추가하지 않는다.
Round summary에는 별도 `round_timing_breakdown`도 남긴다. 이 값은 client timing
밖에 있는 `round_finalize_publication_seconds`, `round_validation_seconds`,
`round_peer_state_build_seconds` 같은 round-level gap을 분리하기 위한 metadata다.
최종 report 조립 단계는 `diagnostics.result_timing_breakdown`에
`result_client_evaluation_seconds`, `result_report_build_seconds`처럼 별도로 남긴다.

Simulation artifact 저장 정책은 `artifact_persistence.persist_agent_local_updates`
가 소유한다. 기본값은 `false`이며, server-owned aggregation artifact를 canonical
update source로 보고 agent-local update 사본은 저장하지 않는다. 디버그 목적으로
client별 local update 사본이 필요할 때만 `true`로 override한다.
FedMatch partitioned path처럼 server update policy가 partition별 material을 직접
소비하는 경우에는 큰 `partitioned_deltas`를 update payload에 inline으로 남기지 않고
`partitioned_deltas_artifact_ref`로 server-owned artifact를 가리킨다. 새 runtime
artifact는 safetensors 기반 binary tensor format을 우선 사용한다. JSON partitioned
artifact는 agent-local debug copy와 이전 run 호환 fallback으로만 남긴다.

`fl_method.composition_mode=method_owned`는 FedMatch처럼 상위 FL SSL method가
client/server 정책을 함께 소유할 때 사용한다. 이 경우
`strategy_axes/fl/method_descriptor=<method>`와 실제
`methods/federated_ssl/<method>/` 구현이 함께 있어야 한다.

예시 형태:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  fl_method.composition_mode=method_owned \
  strategy_axes/fl/method_descriptor=fedmatch \
  strategy_axes/fl/update_partition_policy=partitioned \
  strategy_axes/fl/aggregation_weight_policy=uniform \
  strategy_axes/fl/peer_context_policy=fixed_probe_output_knn
```

현재 FedMatch는 descriptor, capability surface, 원본 core/config snapshot,
method-owned PEFT text encoder local objective, partitioned update 제출, peer helper
context injection, labels-at-server supervised seed step까지 simulation slice에서
실행된다. sparse S2C/C2S는 client-local previous partition snapshot과 partitioned
global state 기준 simulation slice로 실행되며, 통신량은 report 생성 시 artifact
estimate로 기록한다.

원본 기본값은 YAML에 복제하지 않고
`methods/federated_ssl/fedmatch/original_spec.py`에서 report protocol로 주입된다.
trace/report metadata와 `fedmatch_agreement` local objective 선택은
`methods/federated_ssl/fedmatch/descriptor.py`에서 파생된다. ablation은 필요한
method parameter 값만 override한다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  fl_method.composition_mode=method_owned \
  strategy_axes/fl/method_descriptor=fedmatch \
  strategy_axes/fl/update_partition_policy=partitioned \
  strategy_axes/fl/aggregation_weight_policy=uniform \
  +ssl_method.parameter_overrides.confidence_threshold=0.85 \
  +ssl_method.parameter_overrides.num_helpers=4 \
  +ssl_method.parameter_overrides.helper_refresh_interval=1
```

`fixed_probe_output_knn` runtime은 helper 개수와 refresh 주기를
`ssl_method.effective_parameters`에서 먼저 읽고, refresh round에서는 KDTree 우선
nearest-neighbor index로 helper를 고른다. 따라서 smoke처럼 짧은 round에서 helper
injection을 검증할 때는 `helper_refresh_interval=1`만 override하고, 논문 기본
실행에서는 원본 `h_interval=10`을 유지한다.

주의: helper injection은 previous round client snapshot이 있어야 보인다. 1-round
smoke에서는 helper count가 0인 것이 정상이고, 최소 2 rounds가 필요하다. 2026-05-22
기준 `2 clients x 2 rounds x max_steps=1` smoke에서 round 2 helper injection은
확인했지만 약 10분이 걸렸다. 10-client 5-round reduced로 올리기 전에는
PEFT text encoder simulation의 frozen backbone/tokenizer 재로딩과 helper model
materialization 병목을 줄였고, 비교용 reduced는 `10 clients`, `5 rounds`,
`max_steps=20`으로 검증했다. FedMatch도 main fair comparison에서는
`ssl_method.local_budget_policy=iteration_capped` 기본값을 사용한다. 원본
labels-at-client local budget을 확인해야 할 때만
`ssl_method.local_budget_policy=original_method`를 명시한다. 이 original mode는
`client_batch_size/client_epochs`를 읽어 labeled batch 수를 epoch step으로 삼고,
unlabeled batch size를 같은 epoch 안에서 unlabeled pool을 지나가도록 동적으로 정한다.
manual Query SSL 경로와 FedMatch method-owned 경로 모두 `diagnostic_view`를 통과하므로
보고용 pseudo-label diagnostics 비용은 같은 방식으로 제한된다.

2026-05-26 기준 FedMatch method-owned reduced는
`shared_general_reddit_pc100_alpha03_clients10` split에서 `10 clients x 5 rounds`로
완주했고, artifact communication estimate를 포함한 verifier가 PASS했다. 해당 run은 실행
경로 검증용이며 최종 macro-F1은 초기보다 낮았다. main/full-budget은 별도 요청이
있기 전까지 실행하지 않는다.

## Report Index 갱신

실험 웹/대시보드용 index는 중앙 SSL과 FL SSL report를 같은 ingest로 읽는다.
대시보드 cache 생성 명령과 `--reset` 의미는
[experiment dashboard README](../../../apps/experiment_dashboard/README.md)를 기준으로
본다. 이 폴더의 README는 FL 실행 방법만 소유하고, 웹 데이터 생성 절차는
`apps/experiment_dashboard`가 설명한다.

## 주의

- 현재 FL SSL simulation은 classifier 기반 경로만 지원한다. prototype scorer/rebuild
  scaffold는 제거했으며, prototype 기반 방법론은 실제 method로 확정될 때 다시 붙인다.
- `report.protocol.embedding_adapter`, `local_trainer_runtime`이 논문용 실행
  환경인지 확인한다. CPU/hash debug smoke 결과를 성능 비교에 섞지 않는다.
- scripts는 Hydra entrypoint, sweep, report/index wrapper만 소유한다. 새 method
  core는 scripts에 추가하지 않는다.

## 예시: FlexMatch Shared Labeled 5라운드 Reduced Run

이미 materialize된 `shared_client_seed` 10-client Dirichlet alpha=0.3 split을
고정 입력으로 사용해 `FlexMatch + PEFT text encoder + FedAvg` manual 조합을
5라운드 실행한다. 모든 client는 같은 labeled seed를 보고, unlabeled shard만
client별 non-IID로 나뉜다. 각 client의 round당 local optimizer step 상한은
`training_task.max_steps=20`이고, central SSL main과 맞춰 labeled/unlabeled
batch size는 `12`로 둔다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  fl_method.composition_mode=manual \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  strategy_axes/ssl/consistency_method=flexmatch_usb_v1 \
  strategy_axes/trainable_state/update_family=peft_text_encoder \
  round_runtime.aggregation_backend_name=fedavg \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/shared_client_labeled/labeled-ourafla_reddit_unlabeled-ourafla_reddit_validation-ourafla_reddit_test-ourafla_reddit_shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_clients10_seed42/manifest.json \
  training_task.batch_size=12 \
  training_task.max_steps=20
```
