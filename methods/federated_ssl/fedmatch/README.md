# FedMatch

`fedmatch/`는 FedMatch 논문 method 의미를 소유한다.

- `original_spec.py`: 원본 repository/commit과 config.py 기반 hyperparameter snapshot
- `descriptor.py`: method identity, required views, runtime capability surface
- `local_objective.py`: sigma/psi loss routing metadata, confidence filter,
  agreement pseudo-label vote helper, tensor-level supervised/unsupervised loss core
- `server_policy.py`: labels-at-client / labels-at-server policy metadata
- `round_policy.py`: helper context policy metadata
- `helper_selection.py`: 원본 KDTree helper selection을 generic vector top-k로 보존한 core
- `parameter_routing.py`: 원본 full parameter sigma/psi를 LoRA-classifier trainable
  scope로 매핑하는 metadata

현재 상태는 `tensor_local_objective_core_v1`이다. 원본 FedMatch snapshot은
`https://github.com/wyjeong/FedMatch.git`
`4947aa255d59bd37915e25a719763aaaf5d7e067`로 고정한다.

완료된 것:

- 원본 labels-at-client/server 설정값 보존
- confidence filter와 agreement-based pseudo labeling의 framework 독립 helper
- `loss_fn_s`의 supervised CE를 `sigma` partition loss로 계산
- `loss_fn_u`의 confident row filter, helper KL, agreement CE, psi L1,
  sigma/psi L2 regularization을 PyTorch tensor core로 계산
- helper refresh/top-k selection helper
- full ResNet9 sigma/psi decomposition을 LoRA adapter + classifier head의
  logical sigma/psi partition으로 매핑

아직 runtime 실행은 열지 않는다. LoRA-classifier trainer에서 sigma/psi logical
partition을 실제 optimizer step으로 분리하는 adapter, helper prediction exchange,
sparse S2C/C2S delta sync, labels-at-server supervised server step은 다음 단계에서
구현한다.
