# FL SSL Method Capability Matrix

이 문서는 FedMatch/FedLGMatch/(FL)^2의 FL SSL capability 차이를 정리한다.
FedMatch는 첫 method로 선택되어 capability surface와 원본 core/config snapshot,
method-owned tensor local objective core를 추가했다. 현재 LoRA-classifier logical
partition step, physical trainable partition step, PEFT-backed partitioned model
builder, partitioned global state 보존, C2S sparse upload projection,
method-owned local simulation bridge는
`methods/adaptation/lora_classifier/federated_ssl/`의 method-neutral 실행 primitive가
소유한다. 현재 helper peer context simulation slice, labels-at-server supervised seed
server step, client-local previous partition snapshot accounting, sparse S2C/C2S sync
slice가 simulation 경로에 열려 있다.
각 method의 source of truth는
`methods/federated_ssl/<method>/`의 descriptor, local objective, server policy, round
policy가 된다.

## 현재 Runtime Surface

이미 열린 capability:

- `methods/federated_ssl.FederatedSslMethodDescriptor`
  - method identity, required views, local step, server step, recipe, runtime support를
    표현한다.
- `FederatedSslRoundStateExchangeSpec`
  - `none`
  - `client_metric_summary`
  - custom exchange는 descriptor에서 선언할 수 있지만 default live runtime에서는
    bootstrap/finalize 전에 실패한다.
  - simulation에서는 `peer_context=fixed_probe_output_knn`가 이전 round
    client-local LoRA snapshot과 validation probe vector 기반 helper context를 만든다.
- `DefaultServerPolicyExecutor`
  - 현재는 `round_runtime_aggregation_backend` + `round_active_pair_only`만 지원한다.
  - custom server policy는 method-specific 분기가 아니라 capability executor 추가로
    열어야 한다.
- `manual` composition
  - FixMatch/FlexMatch/FreeMatch/PseudoLabel + FedAvg + LoRA-classifier 같은 lower-axis
    baseline/ablation용이다.
- `method_owned` composition
  - FedMatch/FedLGMatch/(FL)^2처럼 client objective와 server/round policy를 함께
    소유하는 논문 method용이다.
- `FederatedSslCapabilityPlan`
  - labeled exposure, local supervision regime, server step, peer context,
    server update, update partition, local SSL policy, aggregation weight,
    query multiview source,
    client participation을 공통 capability 축으로 기록한다.
  - 기본값은 `shared_client_seed`, `all_clients`, `client_labeled_and_unlabeled`,
    `server_step=none`, `server_update=fedavg_merged_delta`, `peer_context=none`,
    `update_partition=unified`, `local_ssl_policy=profile_pseudo_label`,
    `aggregation_weight=example_count`, `query_multiview_source=materialized_rows`다.
- `partitioned_trainable_state` planned capability
  - frozen backbone은 공유하고 trainable adapter/head state만 partition별로 보존한다.
  - FedMatch의 `sigma/psi`는 이 capability 위에 올라가는 method-owned partition
    scheme이며, LoRA/DoRA 같은 concrete PEFT composition은 adapter-family가 소유한다.
  - runner와 runtime은 FedMatch 이름을 판단하지 않고 `partitioned_update`와
    `composition_policy`를 통해 materialization/evaluation을 연결한다.
  - 초기 구현 위치는 가까운 owner인 `methods/adaptation/lora_classifier/` 아래지만,
    type과 primitive 이름은 `TrainableAdapterPartitionPlan`,
    `AdapterClassifierDeltaBundle`, `PartitionedAdapterStateProjector`처럼
    adapter-neutral하게 둔다. PEFT adapter 축이 `lora`/`dora`로 재정립되면 이
    primitive를 상위 축으로 옮긴다.
- `methods/federated_ssl/fedmatch/`
  - FedMatch descriptor, 원본 설정 snapshot, local objective/server/round policy,
    recipe, sigma/psi partition metadata를 소유한다.
  - 현재 status는 `lora_local_runtime_slice_v1`이다. 원본 설정값, confidence
    filter, agreement pseudo-label vote, KDTree 우선 helper nearest-neighbor selection,
    supervised/unsupervised FedMatch tensor loss는 method package에 고정했다.
    현재 status는 `partitioned_trainable_state_slice_v1`로, LoRA concrete slice를 넘어
    DoRA 같은 PEFT adapter 교체를 허용하는 partitioned trainable state primitive를
    단계적으로 열고 있다.
  - `methods/adaptation/lora_classifier/federated_ssl/`는 method-owned objective를
    LoRA-classifier model/loaders, logical/physical partition delta, shared update
    payload로 실행하는 adapter-family slice다. FedMatch method 의미는
    `methods/federated_ssl/fedmatch/`에서 읽는다.
  - helper prediction exchange는 이전 round client-local LoRA snapshot과 validation
    probe vector 기반 simulation slice로 실행된다. labels-at-server는
    `server_only_seed + supervised_seed_step` server runtime과 client-local `psi`
    upload slice로 실행된다. 현재 labels-at-client slice는 기존 LoRA-classifier
    FedAvg merged delta와 `fedmatch_partitioned`에서 쓰는 `partitioned_deltas`를
    함께 제출한다. C2S/S2C는 원본 `delta_threshold`/`l1_threshold` 의미를 반영해
    sparse projection을 적용한다. S2C는 round 사이 client-local partition snapshot을
    기준으로 server-client diff를 만든다.

현재 구현하지 않을 것:

- `agent/src/**/fedmatch*.py`, `main_server/src/**/fedmatch*.py` 같은 method-name runtime
  파일.
- method descriptor YAML만 먼저 추가하는 placeholder config.
- 검증 없이 바로 full-budget 실행으로 들어가는 것. 총 예정 communication round가
  30을 넘으면 runner guard가 막으므로, 필요한 경우 long-run ack를 명시한다.

## Candidate Matrix

| 후보 | 논문 setting과 핵심 아이디어 | 현재 TraceMind fit | 필요한 capability | 구현 난도 | 권장 순서 |
|---|---|---|---|---|---|
| FedMatch | labels-at-clients FSSL. inter-client consistency와 labeled/unlabeled parameter decomposition 중심. | `shared_client_seed` 또는 client-labeled regime에서 가장 가깝다. | descriptor, 원본 core/config snapshot, tensor local objective core는 method package에 있고, LoRA-classifier partitioned runtime slice, helper peer-context simulation slice, labels-at-server supervised seed server step, partitioned global state, sparse S2C/C2S projection, communication accounting은 열림. full 원본 parity에는 reduced/main run 관측 검증이 추가로 필요하다. | 중간 | 1순위, partitioned trainable state slice opened |
| FedLGMatch | local/global pseudo-label을 함께 쓰는 FSSL. global pseudo-label state를 round마다 활용할 가능성이 높다. | 현재 global model/prototype은 있으나 global pseudo-label cache/state는 별도 policy로 고정되지 않았다. | method-owned descriptor, local objective, `round_state_exchange`로 global/local pseudo-label statistics, custom server/round policy 가능성. | 높음 | 2순위 |
| (FL)^2 | labels-at-server setting. server에 소량 labeled data, client는 unlabeled data 중심. | 현재 main split은 client에 labeled source도 분배한다. 논문 setting을 맞추려면 dataset/split policy부터 바꿔야 한다. | server-labeled seed regime, client unlabeled-only local objective, server-owned threshold/calibration state, custom round policy 가능성. | 높음 | 3순위 |

## First Method Recommendation

첫 구현 후보는 FedMatch로 확정했다. 현재 완료된 범위는 capability surface,
원본 core/config snapshot, tensor local objective core, LoRA-classifier partitioned
runtime slice다.

이유:

- 현재 TraceMind main comparison은 `shared_client_seed + client별 unlabeled` split을
  기본으로 두고, `client_local_split`도 legacy/ablation으로 유지한다.
- 현재 기본 조합인 `FixMatch + FedAvg + LoRA-classifier`에서 local objective만 더
  깊게 method-owned로 바꾸는 경로가 가장 짧다.
- FedMatch descriptor는 공통 `partitioned` update capability와 `uniform`
  aggregation weight를 요구하도록 capability validator에 고정했다. `sigma/psi`
  scheme 이름과 loss routing 의미는 FedMatch method package가 소유한다.
- FedMatch 원본 snapshot은 `wyjeong/FedMatch`
  `4947aa255d59bd37915e25a719763aaaf5d7e067`로 고정했다.
- 원본 full ResNet9 parameter decomposition은 TraceMind에서 frozen backbone을 제외한
  trainable adapter + classifier head tensor의 `sigma/psi` partition으로 매핑한다.
  현재 concrete adapter는 LoRA지만, 새 runtime primitive는 DoRA 같은 PEFT adapter로
  바뀌어도 FedMatch method core와 runner를 바꾸지 않는 구조여야 한다.
- FedLGMatch와 (FL)^2는 global pseudo-label cache 또는 labels-at-server regime 때문에
  dataset/split/report 의미까지 같이 바뀔 가능성이 크다.

FedMatch 다음 구현 결정:

- parameter decomposition은 우선 기존 `lora_classifier` family 위에서
  generic `partitioned` update capability로 표현한다. `sigma/psi` partition scheme은
  FedMatch-local metadata로 두고, shared contract를 바꾸는 payload split은 실제 필요가
  확인될 때만 연다.
- server update/delta 해석은 `server_update_policy`로 분리했다. 현재 실행되는 FedMatch
  slice는 `fedavg_merged_delta`로 merged LoRA-classifier delta를 기존 FedAvg path에
  제출하거나, `fedmatch_partitioned`로 shared update의 `partitioned_deltas`를
  LoRA-classifier `partitioned_delta_average` simulation backend에서 소비한다. 이
  backend는 원본 sparse S2C/C2S sync 전체가 아니라 logical partition delta 평균
  simulation slice다.
  FixMatch 같은 stateless local SSL policy는 method-owned simulation runtime에서
  `psi` partition objective로 주입할 수 있다. FlexMatch/FreeMatch처럼 state surface가
  필요한 Query SSL policy는 아직 실행 전에 막는다.
- local pseudo-label/consistency objective는 `local_ssl_policy`로 분리했다.
  FixMatch/FlexMatch/FreeMatch 파라미터는 기존 `query_ssl_method`가 계속 소유하고,
  `fedmatch_agreement`는 FedMatch method package가 소유한다. FlexMatch/FreeMatch처럼
  algorithm state 저장 surface가 필요한 조합은 실행 전에 validator가 막는다.
- inter-client consistency는 `peer_context=none` baseline을 유지하면서,
  `peer_context_policy=fixed_probe_output_knn` runtime adapter로 helper client
  선택과 method-owned trainer 주입 seam을 열었다. helper selection은 원본
  `KDTree.query(num_helpers + 1)` 의미를 보존해 KDTree를 우선 사용하고, experiments
  dependency가 없는 실행에서는 같은 Euclidean nearest 기준의 full-scan으로 fallback한다.
  실제 helper weak-view probability는 이전 round client-local LoRA snapshot/probe
  vector를 이용해 FedMatch KL loss에 연결한다.
- labels-at-server variant는 `server_only_seed + supervised_seed_step` capability로
  simulation에서 열었다. server step은 round open 전에 bootstrap labeled rows로
  active LoRA-classifier state를 발행하고, client side는 unlabeled-only `psi`
  partition update를 제출한다.

FedMatch 원본에서 보존한 기본값:

- `num_clients=100`, `num_rounds=200`, script client fraction `0.05`
- `num_helpers=2`, `confidence=0.75`, `psi_factor=0.2`, `h_interval=10`
- labels-at-client: `lambda_s=10`, `lambda_i=1e-2`, `lambda_a=1e-2`,
  `lambda_l2=10`, `lambda_l1=1e-4`, `l1_thres=5e-6`, `delta_thres=5e-5`
- labels-at-server: 같은 `lambda_s/i/a/l2`, `lambda_l1=1e-5`,
  `l1_thres=1e-5`, `delta_thres=1e-5`

위 값의 source of truth는 `methods/federated_ssl/fedmatch/original_spec.py`다.
`conf/strategy_axes/fl/method_descriptor/fedmatch.yaml`은 `scenario`,
`use_original_parameters`, `parameter_overrides`와 trace/report wiring metadata만 노출하고,
원본 numeric 기본값은 복제하지 않는다. runner가 report protocol에
`original_parameters`, `effective_parameters`, `parameter_override_status`를 주입한다.

## Open Selection Gate

FedMatch 이후 새 method를 시작하려면 아래 결정을 먼저 확정한다.

- FedLGMatch를 선택하면 global/local pseudo-label state를 어떤 artifact나
  `round_state_exchange`로 주고받을지 먼저 정한다.
- `(FL)^2`를 선택하면 labels-at-server regime을 맞추기 위해 split/source policy부터
  새로 정한다.
- 어떤 method를 선택해도 먼저 `1-round` smoke와 필요 시 `5-round` reduced run으로
  wiring과 metadata를 확인한 뒤 full-budget 비교로 올린다.

FedMatch 외 method는 선택 전에는 `methods/federated_ssl/<method>/` 구현 파일과
`conf/strategy_axes/fl/method_descriptor/<method>.yaml` placeholder를 만들지 않는다.

## Implementation Gate

선택된 method는 아래 순서로만 추가한다.

1. `methods/federated_ssl/<method>/README.md`
2. `methods/federated_ssl/<method>/descriptor.py`
3. `methods/federated_ssl/<method>/local_objective.py`
4. `methods/federated_ssl/<method>/server_policy.py`
5. `methods/federated_ssl/<method>/round_policy.py`
6. 필요할 때만 `recipe.py` 또는 `aggregation.py`
7. descriptor가 실제로 resolve된 뒤에만
   `conf/strategy_axes/fl/method_descriptor/<method>.yaml`
8. capability가 부족하면 `agent`/`main_server`에 method 이름이 아니라 capability 이름의
   adapter를 추가한다.

## Verification Gate

새 method의 최소 검증:

- descriptor registry resolve.
- Hydra method descriptor compose.
- compatibility validator가 unsupported local profile/runtime pair를 bootstrap 전에 실패.
- `fl_method.composition_mode=method_owned` report metadata.
- 1-round smoke에서 method 이름 변경과 실제 local objective 변경이 동시에 기록.
- artifact-ref delta path 유지.
- long-run guard 유지.

새 method 검증은 먼저 `1-round` smoke와 필요 시 `5-round` reduced run으로 닫고,
full-budget comparison은 후보와 비교 조건을 확정한 뒤 별도 실행한다.

## Paper Anchors

- FedMatch: `Federated Semi-Supervised Learning with Inter-Client Consistency &
  Disjoint Learning`, arXiv `2006.12097`
  (`https://arxiv.org/abs/2006.12097`).
- FedLGMatch: `Federated semi-supervised learning via joint local and global pseudo
  labeling`, Knowledge-Based Systems, 2025
  (`https://www.sciencedirect.com/science/article/pii/S0950705125006884`).
- (FL)^2: `(FL)^2: Overcoming Few Labels in Federated Semi-Supervised Learning`,
  NeurIPS 2024
  (`https://papers.nips.cc/paper_files/paper/2024/hash/4d2aa4c034745f558bfea34643c8d6a6-Abstract-Conference.html`).
