# Hydra Config Layout

`scripts/conf`는 실행 job config와 재사용 전략 group이 섞여 있기 때문에 탐색
비용이 생긴다. 현재 override 호환성을 깨지 않기 위해 기존 group 이름은 유지한다.

## 현재 분류

- Job config: `datasets/`, `prototypes/`, `experiments/`
- 공통 실행 축: `dataset/`, `embedding/`, `runtime/`, `paper_backbone/`
- Prototype 축: `prototype_builder/`
- Central SSL / LoRA 축: `lora/`, `lora_run_preset/`, `lora_train_source/`,
  `bootstrap_teacher_source/`, `pseudo_label_algorithm/`, `query_ssl_method/`,
  `query_ssl_train_source/`, `query_ssl_augmenter/`,
  `query_adaptation_initial_checkpoint/`, `lora_experiment/`
- FL SSL 축: `federated_run_preset/`, `federated_shard_policy/`,
  `federated_ssl_method/`, `federated_round_runtime/`,
  `federated_training_task/`, `federated_validation/`, `federated_report/`,
  `training_algorithm_profile/`

## 정리 기준

- 새 job은 `datasets/`, `prototypes/`, `experiments/` 중 하나에 둔다.
- 새 reusable group은 실행 rail을 이름에 드러낸다.
- 기존 CLI override가 많은 group은 한 번에 이동하지 않는다.
- namespace 이동은 compatibility alias와 Hydra config test를 같이 추가한 뒤 한다.
