# Hydra Config Layout

`conf/`는 TraceMind 실험 조합과 method/runtime 파라미터의 루트 Hydra config
공간이다. 여러 entrypoint나 여러 preset이 공유하는 축만 config group으로
유지하고, 단일 entrypoint 전용 shape는 해당 job config 안에 둔다.

## 현재 분류

- Job config: `jobs/datasets/`, `jobs/prototypes/`, `jobs/experiments/`
- 공통 실행 축: `dataset/`, `embedding/`, `runtime/`, `paper_backbone/`
- Prototype 축: `prototype_builder/`
- Central SSL / LoRA 축: `lora/`, `lora_run_preset/`, `query_source/`,
  `pseudo_label_algorithm/`, `query_ssl_method/`, `query_ssl_augmenter/`,
  `query_adaptation_initial_checkpoint/`, `lora_experiment/`
- FL SSL 축: `federated_run_preset/`, `federated_shard_policy/`,
  `federated_ssl_method/`, `training_algorithm_profile/`
- FL simulation entrypoint-local section:
  `jobs/experiments/run_federated_simulation.yaml`의 `round_runtime`,
  `training_task`, `validation`, `report`

## 정리 기준

- 새 job은 `jobs/datasets/`, `jobs/prototypes/`, `jobs/experiments/` 중 하나에 둔다.
- 새 reusable group은 실행 rail을 이름에 드러낸다.
- preset이 1개뿐이고 해당 entrypoint에서만 쓰이면 group을 만들지 않는다.
- YAML은 실행 조합표와 파라미터를 담고, Python 구현이나 복잡한 계산 로직은 두지 않는다.
- namespace 이동은 Hydra config test와 관련 catalog/doc 갱신을 같이 닫는다.
