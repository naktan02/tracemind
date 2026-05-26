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
- `parameter_routing.py`: 원본 full parameter sigma/psi를 LoRA-classifier trainable
  scope로 매핑하는 metadata
- `server_step_parameters.py`: labels-at-server supervised seed step budget 해석

LoRA-classifier에서 FedMatch를 실행하는 family-specific bridge와 partitioned optimizer
loop는 `methods/adaptation/lora_classifier/federated_ssl/`가 소유한다. 이 폴더는
FedMatch의 원본 의미와 policy를 읽는 시작점이고, adapter family 구현 파일을 누적하지
않는다.

현재 상태는 `lora_local_runtime_slice_v1`이다. 원본 FedMatch snapshot은
`https://github.com/wyjeong/FedMatch.git`
`4947aa255d59bd37915e25a719763aaaf5d7e067`로 고정한다.

완료된 것:

- 원본 labels-at-client/server 설정값 보존
- confidence filter와 agreement-based pseudo labeling의 framework 독립 helper
- `loss_fn_s`의 supervised CE를 `sigma` partition loss로 계산
- `loss_fn_u`의 confident row filter, helper KL, agreement CE, psi L1,
  sigma/psi L2 regularization을 PyTorch tensor core로 계산
- LoRA-classifier FedMatch step은 원본 `loss_fn_u` 순서에 맞춰 weak/original view를
  full unlabeled batch로 먼저 평가하고, confidence를 통과한 row에 대해서만
  strong/backtranslated view forward와 helper weak-view probability를 계산한다.
- LoRA-classifier trainable tensor에 supervised step과 unsupervised step을 순차
  적용하고, sub-step delta를 각각 logical `sigma`/`psi` partition으로 기록
- LoRA-classifier family slice에서 FedMatch local objective를 호출하고, 기존 merged
  LoRA-classifier delta와 logical `sigma`/`psi` partition delta를 함께 제출
- main fair comparison의 local budget은 `local_budget_policy=iteration_capped`를
  기본으로 하며 `training_task.max_steps`를 따른다. 원본 labels-at-client budget은
  `local_budget_policy=original_method`를 명시할 때만 공통 labeled-anchored SSL
  budget primitive로 계산한다. 이때 입력값은 FedMatch `effective_parameters`의 원본
  `client_batch_size/client_epochs`에서 읽는다.
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
- full ResNet9 sigma/psi decomposition을 LoRA adapter + classifier head의 logical
  sigma/psi partition으로 매핑하는 metadata
- `server_update_policy`와 `local_ssl_policy` 축에서 FedMatch-style partitioned
  server update와 FixMatch local SSL policy 조합을 표현/검증하는 capability surface

아직 원본 FedMatch의 full server/runtime 동작은 열지 않는다. helper weak-view
probability provider, `labels-at-server` client-local `psi` upload slice,
`server_step_policy=supervised_seed_step` simulation path는 열렸지만 sparse S2C/C2S
delta sync는 다음 단계 capability로 남긴다.
현재 실행 server path는 `server_step_policy=none`에서
`server_update_policy=fedavg_merged_delta`면 기존 LoRA-classifier FedAvg가 merged
delta를 aggregation하고, `server_update_policy=fedmatch_partitioned`면 simulation
runtime이 LoRA-classifier `partitioned_delta_average` backend로 `partitioned_deltas`를
소비한다. 이 backend는 원본 FedMatch sparse S2C/C2S sync 전체가 아니라
LoRA-classifier logical partition delta를 평균하는 simulation slice다.
`server_step_policy=supervised_seed_step`은 round open 전에 server bootstrap rows로
LoRA-classifier active state를 한 번 더 발행한 뒤 client round를 시작한다.
