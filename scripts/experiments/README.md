# Experiments

이 디렉터리는 코어 runtime을 조합해서 연구/검증을 수행하는 entrypoint layer다.

원칙:

- 운영 후보 algorithm/method core는 `methods`에 둔다.
- 공통 contract/domain은 `shared`, runtime adapter는 `agent`와 `main_server`에 둔다.
- 이 디렉터리는 그 코어를 조합하는 실험 CLI와 실험 전용 helper만 둔다.
- Hydra entrypoint config는 `conf/entrypoints/**/*.yaml`이 source of truth다.

전략 축 전체와 현재 override 가능 여부는 `conf/README.md`와 관련
`conf/strategy_axes/**` leaf를 먼저 보는 편이 빠르다.

## 구조

- `central/`
  - 중앙 pooled/offline SSL control, PEFT/full text encoder 지도학습 control,
    fixed-feature 지도학습 baseline entrypoint를 둔다.
- `fl_ssl/`
  - client split, round loop, aggregation, per-client metric 같은 FL SSL orchestration을 둔다.

공용 text encoder SSL runner/helper는 `scripts/support/query_ssl_text_encoder/`,
dataset/result index 같은 작업형 CLI는 `scripts/workflows/`가 소유한다.

## 직접 실행하는 entrypoint

- `fl_ssl/run_federated_simulation.py`
  - agent/main_server 코어를 조합한 synthetic FL loop.
  - runtime/task/validation/report shape는
    `conf/entrypoints/fl_ssl/run_federated_simulation.yaml` 안의
    `round_runtime`, `training_task`, `validation`, `report` section이다.
  - `strategy_axes/ssl_objective/local_update_profile`은 local update backend,
    scoring/evidence, privacy 조합을 소유한다.
  - `strategy_axes/model_architecture/update_family`가 update family와 v1 payload adapter
    compatibility alias를 선언하고, `round_runtime.aggregation_backend_name`은
    aggregation backend를 직접 고른다.
  - `strategy_axes/fssl_method`는 method identity/report metadata를
    소유하고, 실제 runtime 구현이 완료된 method만 열어야 한다.
- `central/ssl_control/run_peft_supervised_control.py`
  - query-domain 적응 단계의 `frozen backbone + PEFT text encoder + linear head` canonical supervised baseline entrypoint.
- `central/ssl_control/run_full_text_encoder_supervised_control.py`
  - 중앙 실험용 `full mxbai encoder + linear head` supervised-only transfer baseline entrypoint.
- `central/ssl_control/run_query_ssl_control.py`
  - USB `FixMatch`, `PseudoLabel` 등 Query SSL method를 같은 PEFT text encoder scaffold에 얹는
    consistency family SSL entrypoint.
  - method/source/strong-view/initial checkpoint source of truth는
    `strategy_axes/ssl_objective/consistency_method`,
    `execution_context/query_data_source`,
    `query_ssl_strong_view_policy`,
    `strategy_axes/model_architecture/initial_checkpoint` selector다.
    현재 augmentation reader는 entrypoint의 `query_ssl_augmenter` 고정 설정으로
    precomputed USB candidates만 사용한다.
- `scripts/support/query_ssl_text_encoder/`
  - `runners/{supervised,full_text_encoder_supervised,consistency}.py`가 query-domain
    central supervised/SSL scaffold를 실행한다.
  - Query SSL family run context scaffolding은 `query_ssl/run_context.py`, strict USB NLP
    view preparation/cache는 `query_ssl/view_preparation.py`가 담당한다.
  - bootstrap teacher, pseudo-label replay, agent-local query adaptation export
    helper는 중앙 지도/중앙 SSL/FSSL canonical experiment surface가 아니므로
    제거했다.
- `fl_ssl/run_federated_simulation.py`
  - FSSL bootstrap은 `strategy_axes/model_architecture/initial_checkpoint` selector를
    읽을 수 있다. 중앙 PEFT supervised checkpoint manifest가 지정되면
    `scripts/runtime_adapters/federated_server/initial_checkpoint_artifacts.py`가
    PEFT adapter와 `classifier_head.safetensors`를 canonical tensor slot으로 만들고,
    `main_server`의 initial state artifact publication service가 server-owned ref로
    저장한 뒤 초기 shared state ref에 연결한다.

## 공통 Helper


## 먼저 읽을 파일

federated simulation:

1. `fl_ssl/run_federated_simulation.py`
2. `fl_ssl/federated_simulation/README.md`

central PEFT / SSL control:

1. `central/ssl_control/run_peft_supervised_control.py`
2. `central/ssl_control/run_query_ssl_control.py`
3. `../support/query_ssl_text_encoder/runners/supervised.py`
4. 필요하면 `../support/query_ssl_text_encoder/runners/consistency.py`

중앙 PEFT/SSL warm-start와 method별 실행 명령은
`central/ssl_control/README.md`와 각 entrypoint의 `--cfg job` preview를 기준으로 본다.

central fixed-feature supervised control:

1. `central/fixed_feature_control/README.md`
2. `central/fixed_feature_control/run_fixed_feature_baseline.py`
3. `central/fixed_feature_control/run_fixed_feature_self_training_baseline.py`

## 주의할 점

- 이 디렉터리의 helper는 실험용 canonical shape를 만들 수는 있지만,
  `shared` 계약의 source of truth를 대체하지는 않는다.
- 실험에서 잘 된 로직을 운영 경로로 올릴 때는 해당 소유 경계
  (`methods`, `shared`, `agent`, `main_server`, 필요 시 `conf`)로 옮겨야 한다.
- config에 field가 있다고 곧바로 runtime 구현이 있다는 뜻은 아니다.
  secure aggregation/encryption 계열은 현재 typed contract는 있지만
  실제 runtime은 아직 붙지 않았다.
