# FedMatch

`fedmatch/`는 FedMatch 논문 method 의미를 소유한다.

- `original_spec.py`: 원본 repository/commit과 config.py 기반 hyperparameter snapshot
- `descriptor.py`: method identity, required views, runtime capability surface,
  method-owned local runtime entrypoint
- `local_objective.py`: sigma/psi loss routing metadata, confidence filter,
  agreement pseudo-label vote helper, tensor-level supervised/unsupervised loss core
- `server_policy.py`: labels-at-client / labels-at-server policy metadata
- `round_policy.py`: helper context policy metadata
- `compatibility.py`: FedMatch 전용 capability 조합 검증
- `runtime_requirements.py`: FedMatch local runtime이 요구하는 helper probability
  provider 같은 method-local runtime requirement
- `helper_selection.py`: 원본 KDTree helper selection을 TraceMind fixed-probe
  vector nearest-neighbor로 보존한 core
- `parameter_routing.py`: 원본 full parameter sigma/psi를 frozen-backbone 위
  trainable adapter/head partition scope로 매핑하는 metadata
- `partitioned_runtime_plan.py`: scenario, local supervision, sigma/psi routing,
  upload partition, objective, `psi_factor`를 update-family runtime plan으로 정규화
- `partitioned_local_training.py`: FedMatch descriptor가 호출하는 partitioned local
  training entrypoint. 현재는 PEFT text encoder runtime bridge로 위임한다.
- `server_step_parameters.py`: labels-at-server supervised seed step budget 해석

PEFT text-classifier에서 FedMatch를 실행하는 family-specific bridge와 partitioned
optimizer loop의 source of truth는
`methods/adaptation/peft_text_encoder/federated_ssl/`다. 기존
`methods/adaptation/lora_classifier/federated_ssl/` direct import path는 삭제됐다.
FedMatch 원본 의미와 policy는 이 폴더가 계속 소유한다.

현재 상태는 `partitioned_trainable_state_slice_v1`이다. 이 상태는 LoRA concrete
runtime 위에서 FedMatch 의미를 실행하지만, 최종 목표는 adapter 종류가 LoRA/DoRA로
바뀌어도 `sigma/psi` method 의미가 유지되는 `partitioned_trainable_state`
capability다. 원본 FedMatch snapshot은
`https://github.com/wyjeong/FedMatch.git`
`4947aa255d59bd37915e25a719763aaaf5d7e067`로 고정한다.

완료된 것:

- 원본 labels-at-client/server 설정값 보존
- confidence filter와 agreement-based pseudo labeling의 framework 독립 helper
- `loss_fn_s`의 supervised CE를 `sigma` partition loss로 계산
- `loss_fn_u`의 confident row filter, helper KL, agreement CE, psi L1,
  sigma/psi L2 regularization을 PyTorch tensor core로 계산
- PEFT encoder FedMatch step은 원본 `loss_fn_u` 순서에 맞춰 weak/original view를
  full unlabeled batch로 먼저 평가하고, confidence를 통과한 row에 대해서만
  strong/backtranslated view forward와 helper weak-view probability를 계산한다.
- PEFT encoder trainable tensor에 supervised step과 unsupervised step을 순차
  적용하고, sub-step delta를 각각 logical `sigma`/`psi` partition으로 기록
- PEFT text encoder family slice에서 FedMatch local objective를 호출하고, 기존
  merged PEFT encoder delta와 logical `sigma`/`psi` partition delta를 함께 제출
- main fair comparison의 local budget은 `local_budget_policy=iteration_capped`를
  기본으로 하며 `training_task.max_steps`를 따른다. 원본 labels-at-client budget은
  `local_budget_policy=original_method`를 명시할 때만 공통 labeled-anchored SSL
  budget primitive로 계산한다. 이때 입력값은 FedMatch `effective_parameters`의 원본
  `client_batch_size/client_epochs`에서 읽는다.
- report 검증은 `implementation_status`, `scenario`, `local_budget_policy`,
  `parameter_override_status`를 함께 고정해, FedMatch가 metadata-only placeholder나
  원본 budget ablation으로 잘못 실행된 경우를 즉시 실패 처리한다.
- helper refresh와 KDTree 우선 nearest-neighbor helper selection
- 공통 `peer_context=fixed_probe_output_knn` mechanism은
  `methods/federated_ssl/peer_context.py`가 실행하고, FedMatch
  `effective_parameters`의 `num_helpers`/`helper_refresh_interval` 해석은
  method-owned parameter surface로 보존
- 이전 round client-local LoRA snapshot과 validation probe vector 기반 helper
  weak-view probability provider
- `labels-at-server` client-local slice: client labeled rows를 금지하고 unlabeled
  batch로 `psi` objective만 학습해 `psi` partition만 업로드
- `labels-at-server` supervised seed server step: `server_only_seed` bootstrap rows를
  round open 전에 server-side supervised 학습으로 반영하고 다음 active state를 발행
- full ResNet9 sigma/psi decomposition을 frozen backbone을 제외한 trainable
  adapter + classifier head의 logical sigma/psi partition으로 매핑하는 metadata
- `server_update_policy`와 `local_ssl_policy` 축에서 FedMatch-style partitioned
  server update와 FixMatch local SSL policy 조합을 표현/검증하는 capability surface
- PEFT-backed physical partition model builder와 partitioned global state 보존.
  서버는 merged published state와 함께 partition별 trainable adapter/head snapshot을
  artifact metadata에 보존하고, 다음 round client는 해당 partition state에서 시작한다.
- C2S sparse upload projection. 원본 `cal_c2s`의 `delta_threshold` cut과 `psi`
  `l1_threshold` sparsify 의미를 PEFT encoder partition delta에 적용한다.
- S2C sparse download projection. 원본 `cal_s2c`의 sparse mask는 transport 여부만
  결정하고, 적용 값은 raw server partition value로 유지한다.
- round 사이 client-local previous partition snapshot accounting. C2S 후 snapshot은
  `server partition + uploaded sparse delta` 상태로 저장해 upload되지 않은 local
  변화가 다음 round 기준으로 새지 않게 한다.
- C2S/S2C sparse projection은 client metric과 report summary에 non-zero transport
  value count로 남겨 실제 sparse path 사용 여부를 관측한다.
- Helper context는 선택된 helper 수, 실제 provider에 들어간 helper snapshot 수,
  missing snapshot 수, lazy materialized helper model 수를 report에 남겨 peer context
  drift를 관측한다.

아직 원본 FedMatch의 full server/runtime 동작은 열지 않는다. helper weak-view
probability provider, `labels-at-server` client-local `psi` upload slice,
`server_step_policy=supervised_seed_step` simulation path, sparse S2C/C2S projection,
client-local previous partition snapshot accounting은 simulation slice로 열렸지만
실제 네트워크 packet 측정은 posthoc estimate로 남긴다.
현재 실행 server path는 `server_step_policy=none`에서
`server_update_policy=fedavg_merged_delta`면 PEFT encoder FedAvg가 merged
delta를 aggregation하고, `server_update_policy=fedmatch_partitioned`면 simulation
runtime이 PEFT encoder `partitioned_delta_average` backend로 `partitioned_deltas`를
소비한다. 이 backend는 client sparse projection 이후 제출된 PEFT encoder
partition delta를 평균하는 simulation slice다.
`fedmatch_agreement`는 원본 FedMatch agreement objective 의미다. generic
`local_ssl_policy` leaf로 고르지 않고 `method_descriptor=fedmatch`의 method config에서
파생하며, `server_update_policy=fedmatch_partitioned`와 함께 써야 한다.
`server_step_policy=supervised_seed_step`은 round open 전에 server bootstrap rows로
PEFT encoder active state를 한 번 더 발행한 뒤 client round를 시작한다.

## Partitioned State Direction

FedMatch에서 보존해야 하는 의미는 특정 LoRA 구현이 아니라 parameter decomposition이다.

- `sigma`는 labeled/supervised objective가 업데이트하는 partition이다.
- `psi`는 unlabeled agreement, helper consistency, L1/L2 regularization이 업데이트하는
  partition이다.
- `lambda_l1`은 `psi` partition에만 적용한다.
- `lambda_l2`는 같은 trainable adapter/head scope의 `sigma`와 `psi` 차이에 적용한다.
- `sigma/psi`는 한 local step 안의 delta label만이 아니라 round를 넘어 보존되는 global
  partitioned state가 되어야 한다.
- physical partition forward는 `psi` 단독 logits가 아니라 trainable parameter를
  `sigma + psi`로 합성한 effective state를 기준으로 weak/strong confidence,
  pseudo-label, helper agreement를 계산한다. optimizer 대상만 supervised는 `sigma`,
  unsupervised는 `psi`로 제한한다.
- partition별 server state가 아직 없는 첫 physical round에서 published state가
  `sigma + psi`라고 해석되면 원본 `psi_factor`에 따라
  `sigma = published / (1 + psi_factor)`,
  `psi = published * psi_factor / (1 + psi_factor)`로 초기 분해한다.

TraceMind에서 이 의미는 frozen backbone + partitioned trainable adapter/head state로
해석한다. FedMatch package는 partition 이름, loss routing, 원본 parameter snapshot,
helper/server policy를 소유한다. Adapter-family package는 LoRA/DoRA 같은 concrete
PEFT mechanism의 physical partition materialization과 composed forward를 소유한다.
runner, agent, main_server는 FedMatch 이름이 아니라 capability를 통해 이 상태를
전달해야 한다.

## Stepwise Parity Plan

FedMatch 원본 의미 보존은 한 번에 열지 않는다.

1. Adapter-family guard를 먼저 세운다. `sigma/psi` 이름은 FedMatch package가 소유하고,
   LoRA/DoRA 같은 PEFT 구현은 update-family partition mechanism으로 격리한다.
2. Local physical partition을 연다. client 안에서 `sigma`와 `psi` adapter/head
   parameter를 실제로 분리하고, supervised objective는 `sigma`, unsupervised objective와
   L1/L2는 `psi`로 routing한다.
3. Global partitioned state를 연다. server는 `sigma`와 `psi`를 round 간 따로 보존하고,
   다음 round client materialization도 partitioned state에서 시작한다.
4. Sparse C2S upload projection을 연다. client upload는 원본 `delta_threshold`와
   `psi` `l1_threshold`를 반영해 의미 있게 변한 partition delta만 남긴다.
5. Composition policy를 연결한다. evaluation, pseudo-label diagnostics, UMAP은
   method가 선언한 partition 조합을 update-family composed forward로 실행한다.
6. Sparse S2C sync와 communication accounting을 연다. server-to-client transport는
   client-local previous partition snapshot 대비 changed element ratio를 기록하고,
   helper payload까지 포함한 S2C 비용을 report에 남긴다.

각 단계는 unit/integration/reduced run으로 닫은 뒤 다음 단계로 넘어간다. 특히 global
state 단계에서 shared contract를 열더라도 `shared`에는 `sigma/psi`나 `fedmatch` 의미를
넣지 않고, partition mapping과 artifact reference 같은 canonical shape만 둔다.
