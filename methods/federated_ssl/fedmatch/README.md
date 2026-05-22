# FedMatch

`fedmatch/`는 FedMatch 논문 method 의미를 소유한다.

- `original_spec.py`: 원본 repository/commit과 config.py 기반 hyperparameter snapshot
- `descriptor.py`: method identity, required views, runtime capability surface
- `local_objective.py`: sigma/psi loss routing metadata, confidence filter,
  agreement pseudo-label vote helper, tensor-level supervised/unsupervised loss core
- `lora_classifier_training.py`: LoRA-classifier local training에서 FedMatch
  supervised/unsupervised step을 실행하고 logical `sigma`/`psi` delta를 누적해
  shared LoRA-classifier update의 `partitioned_deltas`에 싣는 method-owned core
- `server_policy.py`: labels-at-client / labels-at-server policy metadata
- `round_policy.py`: helper context policy metadata
- `helper_selection.py`: 원본 KDTree helper selection을 generic vector top-k로 보존한 core
- `parameter_routing.py`: 원본 full parameter sigma/psi를 LoRA-classifier trainable
  scope로 매핑하는 metadata

현재 상태는 `lora_local_runtime_slice_v1`이다. 원본 FedMatch snapshot은
`https://github.com/wyjeong/FedMatch.git`
`4947aa255d59bd37915e25a719763aaaf5d7e067`로 고정한다.

완료된 것:

- 원본 labels-at-client/server 설정값 보존
- confidence filter와 agreement-based pseudo labeling의 framework 독립 helper
- `loss_fn_s`의 supervised CE를 `sigma` partition loss로 계산
- `loss_fn_u`의 confident row filter, helper KL, agreement CE, psi L1,
  sigma/psi L2 regularization을 PyTorch tensor core로 계산
- LoRA-classifier trainable tensor에 supervised step과 unsupervised step을 순차
  적용하고, sub-step delta를 각각 logical `sigma`/`psi` partition으로 기록
- method-owned FL simulation client runtime에서 FedMatch local objective를 호출하고,
  기존 merged LoRA-classifier delta와 logical `sigma`/`psi` partition delta를 함께 제출
- helper refresh/top-k selection helper
- 공통 `peer_context=prediction_similarity_topk` simulation adapter에서 FedMatch
  descriptor의 `num_helpers`/`refresh_interval`을 읽어 helper client context를 만들고
  method-owned LoRA trainer까지 전달하는 seam
- full ResNet9 sigma/psi decomposition을 LoRA adapter + classifier head의
  logical sigma/psi partition으로 매핑
- `server_update_policy`와 `local_ssl_policy` 축에서 FedMatch-style partitioned
  server update와 FixMatch local SSL policy 조합을 표현/검증하는 capability surface

아직 원본 FedMatch의 full server/runtime 동작은 열지 않는다. helper client id
선택 seam은 열렸지만 helper model prediction tensor 생성, sparse S2C/C2S delta
sync, labels-at-server supervised server step은 다음 단계 capability로 남긴다.
현재 실행 server path는 `server_step_policy=none`에서
`server_update_policy=fedavg_merged_delta`면 기존 LoRA-classifier FedAvg가 merged
delta를 aggregation하고, `server_update_policy=fedmatch_partitioned`면 simulation
runtime이 LoRA-classifier `partitioned_delta_average` backend로 `partitioned_deltas`를
소비한다. 이 backend는 원본 FedMatch sparse S2C/C2S sync 전체가 아니라
LoRA-classifier logical partition delta를 평균하는 simulation slice다.
