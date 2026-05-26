# LoRA-Classifier FL SSL Slice

이 폴더는 `lora_classifier` adapter family가 FL SSL method-local objective를 실제
PEFT-adapter/classifier 학습 loop와 shared update payload로 실행하는 slice다.
현재 concrete PEFT 구현은 LoRA지만, 이 폴더의 partitioned execution primitive는
DoRA 같은 다른 PEFT adapter로 교체될 수 있는 trainable-adapter mechanism으로
유지한다.

- `method_owned_training.py`: method descriptor의 `runtime_entrypoint`를 읽어
  LoRA-classifier method-owned local core를 호출한다.
- `server_update_policy.py`: FL SSL `server_update_policy`를 LoRA-classifier
  aggregation backend로 해석한다.
- `supervised_seed_step.py`: server bootstrap rows로 LoRA-classifier supervised
  seed delta를 계산하는 family execution primitive다.
- `helper_provider.py`: method-local runtime requirement가 helper weak probability
  provider를 요구할 때 LoRA-classifier helper snapshot을 provider로 materialize한다.
- `partitioned_objective_training.py`: method-owned partitioned local objective를
  LoRA-classifier model/loaders, delta materialization, update envelope에 연결한다.
- `partitioned_budget.py`: partitioned local trainer가 쓰는 original-method /
  max-step budget을 labeled/unlabeled exposure와 분리해 해석한다.
- `partitioned_trainable_model.py`: frozen feature extractor 위에서 physical
  trainable adapter/head partition을 보관하고 composed forward를 제공하는
  adapter-neutral primitive다.
- `partitioned_training_loop.py`: 기존 LoRA-classifier 단일 trainable tensor 위의
  logical partition step과, frozen backbone 위 physical trainable adapter/head
  partition step을 함께 제공한다. physical step은 `sigma/psi` 같은 method-owned
  partition name을 직접 해석하지 않고 caller가 지정한 supervised/unsupervised
  partition에 objective를 라우팅한다.
- `peer_predictions.py`: 이전 round client-local LoRA snapshot을 helper weak-view
  probability provider와 peer selection vector로 materialize한다.

FedMatch의 원본 의미, hyperparameter snapshot, agreement loss, helper policy,
server/round policy는 `methods/federated_ssl/fedmatch/`가 소유한다. `labels-at-server`
같은 labeled exposure/local supervision regime은 `methods/federated_ssl/`의 공통
capability 해석을 사용하고, 이 폴더는 그 의미를 특정 adapter family에서 실행하는
구현만 소유한다.

새 FL SSL method를 추가할 때 이 폴더에 `<method>_*.py`를 추가하는 것을 기본값으로
보지 않는다. 먼저 `methods/federated_ssl/<method>/`의 descriptor/local objective가
기존 partitioned training bridge로 표현되는지 확인하고, adapter-family tensor/update
shape가 실제로 달라질 때만 이 폴더의 실행 primitive를 확장한다.

새 full encoder family가 생기면 이 폴더에 파일을 추가하지 말고 해당 family package
아래의 FL SSL slice로 둔다. DoRA처럼 같은 LoRA-classifier scaffold 안의 PEFT 옵션이면
`config.py`와 PEFT builder seam을 확장하고 method 폴더를 늘리지 않는다.

## Partitioned Trainable Adapter Guard

FedMatch의 `sigma/psi`는 method-owned partition scheme이다. 이 폴더는 그 이름의
논문 의미를 소유하지 않고, adapter family가 제공하는 trainable partition mechanism만
소유한다.

- frozen backbone은 partition 대상이 아니다.
- physical partition 대상은 PEFT adapter tensor와 classifier head tensor다.
- `LoraTextClassifier` 기본 모델에 FedMatch 전용 `sigma/psi` 분기를 추가하지 않는다.
- 새 primitive는 `partition_name`과 `composition_policy`를 입력으로 받되, FedMatch
  이름을 직접 판단하지 않는다.
- LoRA tensor composition은 단순 tensor add로 가정하지 않는다. LoRA/DoRA 같은 PEFT별
  composition은 adapter-family mechanism이 맡고, FedMatch는 어떤 partition을
  조합할지만 선언한다.
- server/runtime 계층은 `fedmatch` 파일이나 조건문 대신 `partitioned_trainable_state`,
  `partitioned_update`, `composition_policy` 같은 capability 이름으로 연결한다.

이 guard의 목적은 FedMatch 원본의 parameter decomposition 의미를 보존하면서도,
현재 LoRA 구현을 DoRA 또는 다른 PEFT adapter로 교체할 때 method core와 runner를 다시
갈아엎지 않게 하는 것이다.

## Transitional Implementation Plan

당장은 기존 `lora_classifier` family 안에서 구현한다. 다만 새 코드 이름과 책임은
LoRA 전용이 아니라 나중에 `lora`, `dora` 같은 PEFT adapter 축으로 옮길 수 있게 둔다.

- `TrainableAdapterPartitionPlan`: partition 이름, 학습 대상 parameter scope,
  optimizer routing, upload policy를 표현한다. `sigma/psi`라는 이름은 FedMatch가
  주입하고, 이 plan type은 이름의 논문 의미를 해석하지 않는다.
- `PartitionedTrainableTextClassifier` protocol,
  `PartitionedTrainableAdapterClassifier` test primitive,
  `run_physical_partitioned_adapter_classifier_step`: frozen backbone 위에서
  partition별 adapter/head parameter를 학습시키는 execution primitive다. 현재
  concrete primitive는 feature-space adapter partition이므로 실제 PEFT LoRA/DoRA
  partition runtime에 바로 연결하지 않는다.
- `AdapterClassifierDeltaBundle`: 현재 payload의 `lora_parameter_deltas`와
  `classifier_head_*_deltas`를 감싸는 adapter-neutral 내부 표현이다. shared payload가
  아직 `lora_classifier` contract를 쓰더라도 내부 계산은 이 bundle을 거쳐 projection한다.
- `PartitionedAdapterStateProjector`: client partitioned delta를 global partitioned
  adapter/head state에 적용하고, evaluation용 composed state를 별도로 만든다.

이름은 adapter-neutral하게 두지만, 파일 위치는 가까운 owner인 이 폴더에서 시작한다.
두 개 이상 adapter family에서 의미가 안정되면 `methods/adaptation/<peft-axis>/` 같은
상위 축으로 승격한다.
