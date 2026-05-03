# Hydra Config Layout

`scripts/conf`는 실행 job config와 재사용 전략 group이 섞여 있기 때문에 탐색
비용이 생긴다. 현재는 **새 namespace 디렉터리 + 기존 group alias wrapper** 방식으로
정리한다. 즉 source of truth는 namespace 아래에 두고, 기존 CLI override surface는
기존 group 이름으로 유지한다.

## 현재 분류

- Job config: `datasets/`, `prototypes/`, `experiments/`
- 공통 실행 축 source of truth: `common/paper_backbone/`
- Prototype 축 source of truth: `prototype/builder/`
- Central SSL / LoRA source of truth:
  `central_ssl/peft_adapter/`, `central_ssl/run_preset/`,
  `central_ssl/supervised_train_source/`,
  `central_ssl/bootstrap_teacher_source/`,
  `central_ssl/pseudo_label_algorithm/`, `central_ssl/method/`,
  `central_ssl/train_source/`, `central_ssl/augmenter/`,
  `central_ssl/initial_checkpoint/`, `central_ssl/experiment/`
- FL SSL source of truth:
  `fl_ssl/run_preset/`, `fl_ssl/shard_policy/`, `fl_ssl/method/`,
  `fl_ssl/round_runtime/`, `fl_ssl/training_task/`, `fl_ssl/validation/`,
  `fl_ssl/report/`, `fl_ssl/algorithm_profile/`
- Compatibility alias group:
  `paper_backbone/`, `prototype_builder/`, `lora/`, `lora_run_preset/`,
  `lora_train_source/`, `bootstrap_teacher_source/`,
  `pseudo_label_algorithm/`, `query_ssl_method/`, `query_ssl_train_source/`,
  `query_ssl_augmenter/`, `query_adaptation_initial_checkpoint/`,
  `lora_experiment/`, `federated_run_preset/`, `federated_shard_policy/`,
  `federated_ssl_method/`, `federated_round_runtime/`,
  `federated_training_task/`, `federated_validation/`, `federated_report/`,
  `training_algorithm_profile/`

## 정리 기준

- 새 job은 `datasets/`, `prototypes/`, `experiments/` 중 하나에 둔다.
- 새 reusable group은 가능하면 `common/`, `prototype/`, `central_ssl/`,
  `fl_ssl/` namespace 아래에 둔다.
- 기존 CLI override가 많은 group은 한 번에 이동하지 않는다.
- namespace 이동은 compatibility alias와 Hydra config test를 같이 추가한 뒤 한다.
