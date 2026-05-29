# Experiments

이 디렉터리는 코어 runtime을 조합해서 연구/검증을 수행하는 entrypoint layer다.

원칙:

- 운영 후보 algorithm/method core는 `methods`에 둔다.
- 공통 contract/domain은 `shared`, runtime adapter는 `agent`와 `main_server`에 둔다.
- 이 디렉터리는 그 코어를 조합하는 실험 CLI와 실험 전용 helper만 둔다.
- Hydra entrypoint config는 `conf/entrypoints/**/*.yaml`이 source of truth다.

전략 축 전체와 현재 override 가능 여부는
[docs/strategy_surface_map.md](../../docs/strategy_surface_map.md)를
먼저 보는 편이 빠르다.

## 구조

- `central/`
  - 중앙 pooled/offline SSL control entrypoint를 둔다.
- `fl_ssl/`
  - client split, round loop, aggregation, per-client metric 같은 FL SSL orchestration을 둔다.
- `prototype_analysis/`
  - prototype 전략 비교와 threshold sweep entrypoint를 둔다.

공용 PEFT SSL runner/helper는 `scripts/support/query_ssl_peft/`, dataset/prototype
pack/result index 같은 작업형 CLI는 `scripts/workflows/`가 소유한다.

## 직접 실행하는 entrypoint

- `prototype_analysis/prototype_strategy_experiment.py`
  - single/kmeans/dbscan prototype 전략 비교.
- `prototype_analysis/prototype_threshold_sweep.py`
  - 선택된 prototype 전략 위에서 threshold policy 비교.
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
- `central/ssl_control/run_peft_ssl_control.py`
  - USB `FixMatch`, `PseudoLabel` 등 Query SSL method를 같은 PEFT text encoder scaffold에 얹는
    consistency family SSL entrypoint.
  - method/source/strong-view/initial checkpoint source of truth는
    `strategy_axes/ssl_objective/consistency_method`,
    `execution_context/query_data_source`,
    `query_ssl_strong_view_policy`,
    `strategy_axes/model_architecture/initial_checkpoint` selector다.
    현재 augmentation reader는 entrypoint의 `query_ssl_augmenter` 고정 설정으로
    precomputed USB candidates만 사용한다.
- `scripts/support/query_ssl_peft/`
  - `runners/{supervised,consistency,pseudo_label,query_adaptation}.py`가 query-domain
    PEFT text encoder scaffold를 실행한다.
  - Query SSL family 공통 scaffolding은 `query_ssl/common.py`, strict USB NLP
    view preparation/cache는 `query_ssl/augmentation.py`가 담당한다.
  - bootstrap teacher와 pseudo-label replay helper는 public experiment entrypoint가
    아니라 내부 workflow/helper 표면으로만 남긴다.
  - agent-local query adaptation export는 `io/query_adaptation*.py`가 담당하고,
    `source_row.query_id`를 single source of truth로 쓴다.

## 공통 Helper

- `scripts/support/reporting/query_buffer_selection_diagnostics.py`: query-buffer selection summary/trace dump 저장 helper

## `scripts/workflows/prototype_pack`와의 차이

- `scripts/experiments/prototype_analysis/*`
  - prototype 전략이나 threshold 정책을 비교하는 연구형 실험 레일
- `scripts/workflows/prototype_pack/*`
  - prototype pack을 실제로 seed/evaluate/pull/activate/report 하는
    artifact workflow 레일

즉 이름은 비슷하지만, 전자는 `비교/탐색`, 후자는 `artifact lifecycle`이 핵심이다.

## 먼저 읽을 파일

prototype 전략 실험:

1. `prototype_analysis/prototype_strategy_experiment.py`
2. `prototype_analysis/prototype_strategy/README.md`

federated simulation:

1. `fl_ssl/run_federated_simulation.py`
2. `fl_ssl/federated_simulation/README.md`

central PEFT / SSL control:

1. `central/ssl_control/run_peft_supervised_control.py`
2. `central/ssl_control/run_peft_ssl_control.py`
3. `../support/query_ssl_peft/runners/supervised.py`
4. 필요하면 `../support/query_ssl_peft/runners/{consistency,query_adaptation,pseudo_label}.py`

중앙 PEFT/SSL warm-start와 method별 실행 명령은
`central/ssl_control/README.md`와 각 entrypoint의 `--cfg job` preview를 기준으로 본다.

## 주의할 점

- 이 디렉터리의 helper는 실험용 canonical shape를 만들 수는 있지만,
  `shared` 계약의 source of truth를 대체하지는 않는다.
- 실험에서 잘 된 로직을 운영 경로로 올릴 때는 해당 소유 경계
  (`methods`, `shared`, `agent`, `main_server`, 필요 시 `conf`)로 옮겨야 한다.
- config에 field가 있다고 곧바로 runtime 구현이 있다는 뜻은 아니다.
  secure aggregation/encryption 계열은 현재 typed contract는 있지만
  실제 runtime은 아직 붙지 않았다.
