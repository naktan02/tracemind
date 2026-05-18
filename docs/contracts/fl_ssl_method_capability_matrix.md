# FL SSL Method Capability Matrix

이 문서는 FedMatch/FedLGMatch/(FL)^2 중 다음에 구현할 method를 고르기 위한
capability matrix다. 구현 source of truth는 선택된 뒤
`methods/federated_ssl/<method>/`의 descriptor, local objective, server policy,
round policy가 된다. 이 문서는 선택 전 의사결정 보조 자료다.

## 현재 Runtime Surface

이미 열린 capability:

- `methods/federated_ssl.FederatedSslMethodDescriptor`
  - method identity, required views, local step, server step, recipe, runtime support를
    표현한다.
- `FederatedSslRoundStateExchangeSpec`
  - `none`
  - `client_metric_summary`
  - custom exchange는 descriptor에서 선언할 수 있지만 default live runtime에서는
    bootstrap/finalize 전에 실패한다.
- `DefaultServerPolicyExecutor`
  - 현재는 `round_runtime_aggregation_backend` + `round_active_pair_only`만 지원한다.
  - custom server policy는 method-specific 분기가 아니라 capability executor 추가로
    열어야 한다.
- `manual` composition
  - FixMatch/FlexMatch/FreeMatch/PseudoLabel + FedAvg + LoRA-classifier 같은 lower-axis
    baseline/ablation용이다.
- `method_owned` composition
  - FedMatch/FedLGMatch/(FL)^2처럼 client objective와 server/round policy를 함께
    소유하는 논문 method용이다.

현재 구현하지 않을 것:

- `agent/src/**/fedmatch*.py`, `main_server/src/**/fedmatch*.py` 같은 method-name runtime
  파일.
- method descriptor YAML만 먼저 추가하는 placeholder config.
- 새 full-budget 실행. 총 예정 communication round가 49를 넘으면 runner guard가
  막으며, 현재 FL SSL 트랙에서는 새 `50-round`/full-budget 실행을 하지 않는다.

## Candidate Matrix

| 후보 | 논문 setting과 핵심 아이디어 | 현재 TraceMind fit | 필요한 capability | 구현 난도 | 권장 순서 |
|---|---|---|---|---|---|
| FedMatch | labels-at-clients FSSL. inter-client consistency와 labeled/unlabeled parameter decomposition 중심. | 현재 `materialized_client_split`가 client별 labeled/unlabeled pool을 모두 가지므로 가장 가깝다. | method-owned descriptor, local objective, maybe `client_metric_summary`, FedAvg server policy. LoRA/classifier parameter split을 실제로 어떻게 표현할지 결정 필요. | 중간 | 1순위 |
| FedLGMatch | local/global pseudo-label을 함께 쓰는 FSSL. global pseudo-label state를 round마다 활용할 가능성이 높다. | 현재 global model/prototype은 있으나 global pseudo-label cache/state는 별도 policy로 고정되지 않았다. | method-owned descriptor, local objective, `round_state_exchange`로 global/local pseudo-label statistics, custom server/round policy 가능성. | 높음 | 2순위 |
| (FL)^2 | labels-at-server setting. server에 소량 labeled data, client는 unlabeled data 중심. | 현재 main split은 client에 labeled source도 분배한다. 논문 setting을 맞추려면 dataset/split policy부터 바꿔야 한다. | server-labeled seed regime, client unlabeled-only local objective, server-owned threshold/calibration state, custom round policy 가능성. | 높음 | 3순위 |

## First Method Recommendation

첫 구현 후보는 FedMatch가 가장 안전하다.

이유:

- 현재 TraceMind main comparison은 `client별 labeled + unlabeled` split이다.
- 현재 기본 조합인 `FixMatch + FedAvg + LoRA-classifier`에서 local objective만 더
  깊게 method-owned로 바꾸는 경로가 가장 짧다.
- server policy는 처음에는 FedAvg를 유지하고, 필요한 client metric summary만 추가해도
  descriptor seam을 검증할 수 있다.
- FedLGMatch와 (FL)^2는 global pseudo-label cache 또는 labels-at-server regime 때문에
  dataset/split/report 의미까지 같이 바뀔 가능성이 크다.

단, FedMatch도 다음 결정을 먼저 내려야 한다.

- parameter decomposition을 LoRA adapter/head 안에서 어떻게 표현할지
  - option A: LoRA adapter와 classifier head를 기존처럼 하나의 `lora_classifier`
    update family로 유지하고, method-local objective만 분해한다.
  - option B: shared/private parameter split을 adapter family payload에 드러낸다.
    이 경우 shared contract와 materialization test가 필요하다.
- inter-client consistency가 요구하는 state surface
  - option A: global model prediction/threshold만 사용하면 custom exchange 없이 시작한다.
  - option B: client pseudo-label statistics가 필요하면 `client_metric_summary`로 시작한다.
  - option C: logits/prototype/cache를 round state로 주고받아야 하면 custom
    `round_state_exchange` capability를 먼저 추가한다.

권장 시작점은 option A + option A다. 즉 payload family를 바꾸지 않고
`methods/federated_ssl/fedmatch/` method-owned local objective를 먼저 구현해,
실제 objective 변경과 report metadata 변경이 동시에 남는지 확인한다.

## Open Selection Gate

구현을 시작하려면 아래 결정을 먼저 확정한다.

- `first_fed_ssl_method`: `fedmatch`, `fedlgmatch`, `(fl)^2` 중 하나.
- FedMatch를 선택하면 기본 시작점은 `lora_classifier` payload family 유지,
  custom round-state exchange 없음, FedAvg server policy 유지다.
- FedLGMatch를 선택하면 global/local pseudo-label state를 어떤 artifact나
  `round_state_exchange`로 주고받을지 먼저 정한다.
- `(FL)^2`를 선택하면 labels-at-server regime을 맞추기 위해 split/source policy부터
  새로 정한다.
- 어떤 method를 선택해도 새 검증은 `1-round` smoke와 필요 시 `5-round` reduced
  run까지만 수행한다. 새 `50-round`/full-budget run은 현재 실행 대상이 아니다.

선택 전에는 `methods/federated_ssl/<method>/` 구현 파일과
`conf/strategy_axes/fl/method_descriptor/<method>.yaml` placeholder를 만들지 않는다.

## Implementation Gate

선택된 method는 아래 순서로만 추가한다.

1. `methods/federated_ssl/<method>/README.md`
2. `methods/federated_ssl/<method>/descriptor.py`
3. `methods/federated_ssl/<method>/<method>.py`
4. `methods/federated_ssl/<method>/local_objective.py`
5. `methods/federated_ssl/<method>/server_policy.py`
6. `methods/federated_ssl/<method>/round_policy.py`
7. 필요할 때만 `recipe.py` 또는 `aggregation.py`
8. descriptor가 실제로 resolve된 뒤에만
   `conf/strategy_axes/fl/method_descriptor/<method>.yaml`
9. capability가 부족하면 `agent`/`main_server`에 method 이름이 아니라 capability 이름의
   adapter를 추가한다.

## Verification Gate

새 method의 최소 검증:

- descriptor registry resolve.
- Hydra method descriptor compose.
- compatibility validator가 unsupported local profile/runtime pair를 bootstrap 전에 실패.
- `fl_method.composition_mode=method_owned` report metadata.
- 1-round smoke에서 method 이름 변경과 실제 local objective 변경이 동시에 기록.
- artifact-ref delta path 유지.
- long-run guard 유지.

full-budget comparison은 현재 실행 대상이 아니다. 향후 사용자가 새 결정을 내리기
전까지 새 method 검증은 `1-round` smoke와 필요 시 `5-round` reduced run으로 제한한다.

## Paper Anchors

- FedMatch: `Federated Semi-Supervised Learning with Inter-Client Consistency &
  Disjoint Learning`, arXiv `2006.12097`
  (`https://arxiv.org/abs/2006.12097`).
- FedLGMatch: `Federated semi-supervised learning via joint local and global pseudo
  labeling`, Knowledge-Based Systems, 2025
  (`https://www.sciencedirect.com/science/article/pii/S0950705125006884`).
- (FL)^2: `(FL)^2: Overcoming Few Labels in Federated Semi-Supervised Learning`,
  NeurIPS 2024
  (`https://papers.nips.cc/paper_files/paper/2024/hash/4d2aa4c034745f558bfea34643c8d6a6-Abstract-Conference.html`).
