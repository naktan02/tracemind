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
| Central replay/bootstrap workflow | explicit central SSL entrypoint | `scripts/support/query_ssl_peft/runners/*` |
| Central trainable surface | `strategy_axes/model_architecture/trainable_surface` | `methods/adaptation/*` trainer |
| PEFT adapter mechanism | `strategy_axes/model_architecture/peft` | `methods/adaptation/peft_adapters/*` |
| Trainable update family | `strategy_axes/model_architecture/update_family`, `round_runtime.update_family_name` | `methods/classification/*`, `methods/adaptation/*`, `methods/prototype/*` |
| FL split/topology policy | `strategy_axes/fl_topology/shard_policy`, `labeled_exposure`, `materialized_split` | `methods/federated/*`, split materialization |
| FL round capability | `strategy_axes/fl_topology/server_step`, `server_update`, `peer_context`, `update_partition`, `aggregation_weight` | `methods/federated_ssl/*` + selected update family runtime |
| FL SSL method descriptor | `strategy_axes/fssl_method` | `methods/federated_ssl/<method family>/` |
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
- canonical FedMatch main-comparison은
  `fssl_method=fedmatch_labels_at_client`처럼 variant 이름으로
  `peer_context`, `server_step`, `local_ssl_policy`, `server_update`,
  `update_partition`, `aggregation_weight`를 descriptor default로 닫는다.
- labels-at-server 경로도 `fssl_method=fedmatch_labels_at_server` variant로
  분리해 `server_only_seed + client_unlabeled_only + supervised_seed_step`
  의미를 lower-axis 밖으로 올린다.
- 두 FedMatch variant의 public 선택 leaf는 `conf/strategy_axes/fssl_method/`에
  있지만, implementation family와 policy source of truth는
  `methods/federated_ssl/fedmatch/`다. generic `fedmatch` leaf는
  compatibility/ablation 입력으로만 남긴다.
- `server_step_policy`와 `server_update_policy`는 다른 축이다. server-side seed step
  여부와 submitted update 해석 방식을 섞지 않는다.
- `fl_topology`는 이름보다 범위가 넓다. 현재는 FL data topology와 round capability
  leaf를 함께 담는 group이고, method identity는 `fssl_method`, local update recipe는
  `ssl_objective/local_update_profile`로 분리한다.
- `security_policy`는 method identity가 아니라 runtime capability 축이다.

## method 내부로 들어가야 하는 축 점검

중앙 SSL/FSSL 실행 표면은 method 선택을 우선한다. method를 골랐는데 다시 teacher,
peer, server step 같은 method recipe 조각을 고르는 구조는 축이 새는 신호로 본다.

Central SSL:

- `strategy_axes/ssl_objective/consistency_method`는 중앙 SSL method 선택 축이므로 유지한다.
- replay/bootstrap은 중앙 SSL method 축이 아니라 workflow surface다. 현재
  public experiment entrypoint가 아니라 내부 helper/workflow로만 두고, teacher 종류를
  별도 축으로 열지 않는다. 장기적으로는 `bootstrap_pseudolabel` 같은 method/recipe로
  승격하거나 사전 materialization workflow로 유지한다.
- `pseudo_label_selection`은 teacher bootstrap artifact를 만들 때 쓰는 selection
  policy다. 중앙 method 비교의 독립 축으로 쓰지 않고, method recipe 기본값이나
  explicit ablation metadata로만 남긴다.
- `augmentation_source`와 `query_ssl_strong_view_policy`는 USB 계열 input view
  materialization/runtime 입력으로 유지할 수 있다. 특정 method만 의미를 소유하면
  method config로 내린다.
- teacher source, checkpoint type, prototype/PEFT/EMA teacher 여부는 user-facing
  strategy axis로 만들지 않는다. 필요하면 method/recipe 내부 요구사항과 artifact
  provenance로 기록한다.

FL SSL:

- `strategy_axes/fssl_method`가 method-owned 실행의 주 선택 축이다.
- `server_step`, `server_update`, `peer_context`, `update_partition`,
  `local_ssl_policy`, `aggregation_weight`는 capability mechanism leaf로만 남긴다.
  `composition_mode=method_owned`에서는 descriptor가 요구/기본값을 파생하고, 사용자는
  method recipe 조각을 직접 고르지 않는다.
- 위 capability leaf를 사람이 직접 고르는 것은 `composition_mode=manual` baseline이나
  명시적 ablation에서만 허용한다. method 이름이 들어간 leaf는 만들지 않는다.
- `shard_policy`, `materialized_split`, `labeled_exposure`, `supervision_regime`,
  `client_participation`은 데이터/실험 조건 축이다. 단, 특정 method가 요구하는 regime은
  descriptor compatibility validator가 제한한다.
- `update_family`, PEFT mechanism, backbone은 trainable state/scaffold 축이다. method가
  요구할 수는 있지만 method identity 자체로 합치지 않는다.

## 확장 절차

- 새 strategy는 `docs/contracts/strategy_addition_playbook.md`를 따른다.
- 새 FL SSL method는 `methods/federated_ssl/NEW_METHOD.md`를 먼저 따른다.
- 새 Query SSL algorithm은 `methods/ssl/NEW_METHOD.md`를 따른다.
- historical compatibility 표면은 `docs/contracts/legacy_contract_ledger.md`에서만
  확인한다.
