# 전략 표면 맵

이 문서는 현재 열려 있는 전략 축을 빠르게 찾기 위한 지도다. 최종 구조 판단은
`docs/architecture/target-method-runtime-structure.md`가 우선하고, 실행 기본값은
각 `conf/` leaf와 code-adjacent README가 소유한다. 완료된 긴 표면 감사 기록은
`docs/notes/decisions/2026-05-28-archived-strategy-surface-map.md`에 보관했다.

## 읽는 법

- 새 algorithm/method/core 추가 위치는 먼저 `methods/`와 `conf/`에서 찾는다.
- `scripts`는 entrypoint, sweep, report/index wrapper만 소유한다.
- `agent`와 `main_server`는 method 이름 파일이 아니라 runtime capability adapter를
  소유한다.
- `lora_classifier`, `diagonal_scale`, `adapter_family_name`은 active 실행 축이 아니다.
  과거 artifact를 읽어야 할 때만 legacy ledger와 archive 문서를 확인한다.

## 주요 축

| 영역 | 선택 위치 | 구현/계약 owner |
|---|---|---|
| Dataset/source/view | `execution_context/*`, `query_data_selection.*` | dataset scripts, shared row contract |
| Central SSL method | `strategy_axes/ssl_objective/consistency_method` | `methods/ssl/algorithms/*` |
| SSL input mode | `strategy_axes/ssl_objective/input_mode` | config-declared runner callable |
| PEFT adapter mechanism | `strategy_axes/model_architecture/peft` | `methods/adaptation/peft_adapters/*` |
| Trainable update family | `strategy_axes/model_architecture/update_family`, `round_runtime.update_family_name` | `methods/classification/*`, `methods/adaptation/*`, `methods/prototype/*` |
| FL split/topology policy | `strategy_axes/fl_topology/shard_policy`, `labeled_exposure`, `materialized_split` | `methods/federated/*`, split materialization |
| FL round capability | `strategy_axes/fl_topology/server_step`, `server_update`, `peer_context`, `update_partition`, `aggregation_weight` | `methods/federated_ssl/*` + selected update family runtime |
| FL SSL method descriptor | `strategy_axes/fssl_method` | `methods/federated_ssl/<method>/` |
| FL local update profile | `strategy_axes/ssl_objective/local_update_profile` | `methods/federated_ssl/local_update_profile.py`, Query SSL/PEFT runtime |
| Prototype build/scoring | `strategy_axes/prototype/*`, prototype analysis config | `methods/prototype/*` |
| Report/evaluation metric | report builder config, evaluator selection | `methods/evaluation/*` |

## 현재 FL 기본 해석

- Manual baseline은 `fl_method.composition_mode=manual`이고 method descriptor를
  compose하지 않는다.
- 기본 lower axes는 `query_ssl_method.algorithm_name`,
  `round_runtime.update_family_name`, `round_runtime.aggregation_backend_name`에서
  파생한다.
- FedMatch 같은 method-owned 실행은 descriptor가 local objective와 server policy
  요구사항을 파생한다. `fedmatch_agreement`는 generic config leaf가 아니라
  FedMatch method-local objective다.
- `server_step_policy`와 `server_update_policy`는 다른 축이다. server-side seed step
  여부와 submitted update 해석 방식을 섞지 않는다.
- `fl_topology`는 이름보다 범위가 넓다. 현재는 FL data topology와 round capability
  leaf를 함께 담는 group이고, method identity는 `fssl_method`, local update recipe는
  `ssl_objective/local_update_profile`로 분리한다.
- `security_policy`는 method identity가 아니라 runtime capability 축이다.

## 확장 절차

- 새 strategy는 `docs/contracts/strategy_addition_playbook.md`를 따른다.
- 새 FL SSL method는 `methods/federated_ssl/NEW_METHOD.md`를 먼저 따른다.
- 새 Query SSL algorithm은 `methods/ssl/NEW_METHOD.md`를 따른다.
- historical compatibility 표면은 `docs/contracts/legacy_contract_ledger.md`에서만
  확인한다.
