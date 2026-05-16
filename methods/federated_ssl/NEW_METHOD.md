# Federated SSL New Method Guide

이 문서는 `methods/federated_ssl/`에 새 FL SSL method를 추가할 때의 최소 경계를
고정한다. Method는 알고리즘 identity이고, `fl_profile` 또는
`experiment_profile`은 Hydra 실행 조합 preset이다.

## 추가할 파일

새 method는 먼저 method-local module 하나로 시작한다.

```text
methods/federated_ssl/<method_name>/
  __init__.py
  descriptor.py
  <method_name>.py
  local_objective.py
  server_policy.py
  round_policy.py
  recipe.py          # optional: 조합표가 descriptor.py를 흐리게 할 때만 분리
  aggregation.py       # method-only aggregation 변형일 때만 추가
  README.md
```

`descriptor.py`는 method identity, required views, runtime capability,
local/server hint를 소유한다. `local_objective.py`, `server_policy.py`,
`round_policy.py`는 FedMatch/FedLGMatch류에서 local objective나 server-side
state exchange가 달라질 때 method-local 의미를 담는 seam이다. method recipe는 사람이
method 폴더를 열었을 때 "이 논문 방법론이 어떤 adapter family, aggregation backend,
round policy 조합으로 구성되는지"를 보는 조립표다. 작은 method는
`descriptor.py`에 recipe metadata를 함께 둘 수 있고, 조합표가 커지거나 별도
테스트/문서화가 필요할 때만 `recipe.py`로 분리한다.

round별 pseudo-label statistics, client metric summary, calibration state가 필요하면
`FederatedSslRoundStateExchangeSpec`에 `exchange_name`과
`required_client_metric_keys`를 먼저 선언한다. main_server에는 method 이름이 아니라
`round_state_exchange` capability executor만 추가한다.

`aggregation.py`는 특정 method에만 종속된 client weighting, state exchange,
pseudo-label 통계 결합 같은 변형이 있을 때만 둔다. 두 개 이상 method에서 공유되거나
adapter payload 해석/평균 규칙으로 안정화된 계산은 `methods/federated/aggregation/*`
또는 `methods/adaptation/<family>/*`로 승격한다. 즉 method 폴더는 사람이 읽는
조립점이고, 재사용 축은 별도 패키지에 남긴다.

## 추가할 설정

- `conf/strategy_axes/fl/method_descriptor/<method>.yaml`
- 필요하면 `conf/strategy_axes/fl/local_update_profile/<profile>.yaml`
- 필요하면 `conf/strategy_axes/fl/round_runtime_profile/<profile>.yaml`
- 사용자-facing 시작점이 필요할 때만
  `conf/strategy_axes/fl/experiment_profile/<profile>.yaml`

`experiment_profile`은 compose preset이다. threshold, LoRA rank, round 수 같은
실행값의 source of truth는 Hydra leaf config에 남긴다.

## Registry

`methods/federated_ssl/registry.py`는 `<method_name>/<method_name>.py` convention을
import해서 decorator 등록을 실행한다. 같은 convention을 따르면 새 method를 추가할 때
registry 목록을 수정하지 않는다. 새 method metadata는 중앙 registry가 아니라
method-local module이 소유한다.

## 건드리지 않을 계층

- `scripts`: entrypoint, sweep, report, artifact 저장 orchestration만 맡긴다.
- `agent`: local runtime adapter와 private/local state만 맡긴다.
- `main_server`: round lifecycle, aggregation publication, runtime adapter만 맡긴다.
- `shared`: cross-boundary contract와 canonical payload 해석만 맡긴다.

새 method를 추가하기 위해 `scripts`, `agent`, `main_server` core를 넓게 수정해야
하면 method seam이 충분히 깊지 않은 신호로 본다.

특히 `agent`와 `main_server`에는 method 이름을 가진 파일을 추가하지 않는다.
예외는 새 method가 아니라 새 runtime capability가 생긴 경우다. 이때도 파일명과
interface는 `raw_text_local_update`, `artifact_ref_materializer`,
`round_state_exchange`처럼 capability 이름을 사용하고, FedMatch/FedLGMatch 같은
method identity와 정책 의미는 method-local module에 남긴다.

## Hook과 helper

Tensor-level SSL objective 조각은 `methods/ssl/hooks/`의 공통 hook을 먼저 본다.
단, 한 method에서만 쓰는 helper는 처음부터 공통으로 올리지 않는다. 예를 들어
FreeMatch 전용 threshold state는 먼저
`methods/ssl/algorithms/freematch/thresholding.py`에 두고, 두 개 이상 method에서
안정적으로 공유될 때만 `methods/ssl/hooks/`로 승격한다.

method-only aggregation 변형도 같은 기준을 따른다. 처음에는 method 폴더 안에 둘 수
있지만, 다른 method가 같은 평균/투영/weighting 규칙을 쓰기 시작하면 축별 reusable
core로 올린다.

## 테스트 체크리스트

- descriptor resolve가 동작한다.
- Hydra `method_descriptor`와 필요한 profile compose가 동작한다.
- compatibility validator가 incompatible profile을 실행 전에 실패시킨다.
- 기존 smoke/report/manifest shape가 변하지 않는다.
- production `methods/federated_ssl/`에 dummy method를 남기지 않고
  `tests/fixtures/` 아래 test-only method fixture로 registry와 descriptor 경계를
  고정한다.
- shared payload를 바꾸면 golden fixture와 producer/consumer test를 함께 갱신한다.
