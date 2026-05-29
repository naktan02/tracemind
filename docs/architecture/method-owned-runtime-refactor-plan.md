# Method-Owned Runtime Refactor Status

이 문서는 FL SSL method framework의 현재 경계와 guard만 남긴다. 과거 Batch 이력은
source of truth가 아니며, 새 작업 판단은 코드, code-adjacent README, 이 문서의 현재
규칙을 기준으로 한다.

## 현재 상태

주요 runtime refactor는 완료됐다. 다음 변경은 새 method, update/payload family,
runtime capability가 실제로 필요할 때만 연다.

최종 method/runtime vocabulary와 `adapter_family`에서 `update_family`로의 전환 계획은
`docs/architecture/target-method-runtime-structure.md`를 우선한다. 이 문서는 현행
method-owned runtime guard를 설명한다.

## 소유 경계

| 계층 | 소유 |
|---|---|
| `shared/` | contract, domain entity, canonical payload 해석 |
| `methods/` | method identity, algorithm core, SSL hook, local objective, adaptation update core, aggregation math, server/round policy 의미 |
| `conf/` | Hydra 실행 조합과 파라미터 source of truth |
| `agent/` | private/local state, raw text 접근, local artifact materialization, selected method core 실행 port, payload upload |
| `main_server/` | round lifecycle, validation, repository/materialization, selected method/server policy 실행 port, publication |
| `scripts/` | experiment entrypoint, sweep, report/artifact orchestration, runtime bridge |

논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다. 이
폴더는 descriptor, recipe metadata, local objective, server/round policy,
method-only aggregation 변형을 묶을 수 있다. 두 개 이상 방법론에서 공유되는 계산은
`methods/federated/aggregation/*`, `methods/adaptation/<family>/*`, `methods/ssl/*`
같은 축별 패키지로 승격한다.

## 닫힌 결정

- 일반 runtime/strategy registry는 lookup, duplicate guard, lazy import trigger만 맡는다.
- concrete builtin 목록을 중앙 `registry.py`나 `builtin_loader.py`에 누적하지 않는다.
- shared canonical payload family의 explicit loader만 contract source-of-truth 예외로 둔다.
- local update backend, scoring, privacy guard, update-family aggregation core는
  `methods/adaptation/<family>/` 또는 `methods/classification/<family>/`가 소유한다.
- runtime/API fallback은 `methods/federated_ssl/runtime_fallbacks.py` 하나로 제한한다.
  agent training fallback과 main_server round runtime fallback은 named profile로
  같은 파일에 두고, runtime adapter는 env/request parsing만 맡는다.
- runtime/objective compatibility는 `methods/adaptation/runtime_objective_compatibility.py`
  dispatcher와 family-local 구현이 맡는다.
- `main_server`는 generic payload adapter, aggregation executor, artifact ref capability,
  server policy executor, round-state exchange bridge만 소유한다.
- scripts는 runtime adapter, report builder, artifact writer를 thin orchestration으로 유지한다.
- old path compatibility facade와 단순 re-export wrapper는 재도입하지 않는다.

## 활성 Guard

- 새 FL SSL method 추가를 위해 `agent/`나 `main_server/`에 method 이름을 가진 파일을
  만들지 않는다.
- `shared/src/config`를 profile/default source of truth로 되살리지 않는다.
- Python mapping/default module에 실행 기본값을 중복 정의하지 않는다. 실행값은 Hydra
  leaf config와 typed contract가 소유한다.
- runtime adapter는 method threshold, weighting, pseudo-label objective 의미를 판단하지 않는다.
- runtime 계층에 추가할 수 있는 것은 method가 아니라 `raw_text_local_update`,
  `artifact_ref_materializer`, `round_state_exchange`, `server_policy_executor` 같은
  capability다.
- builder는 payload를 만들고 writer/exporter는 IO를 맡는다.
- artifact ref는 소유 계층 밖에서는 opaque identifier로 다룬다.

## Future Gate

FedMatch, FedLGMatch, `(FL)^2` 중 하나를 실제 구현할 때는 먼저
`methods/federated_ssl/<method>/`에 method-local descriptor, recipe, local objective,
server/round policy, tests를 둔다. `agent/`나 `main_server/` 변경이 필요하면 그것이
method-specific 구현인지 generic capability인지 먼저 검증한다.

새 update/payload family를 추가할 때는 shared payload family contract, methods-owned update
core, aggregation projection, preflight를 먼저 정의한다. runtime은 bridge/capability만
추가한다.

script runtime branch가 한 update family를 넘어 커지면 scripts 내부 분기를 늘리지 말고
methods-owned dispatcher나 family-owned helper로 이동한다.

## 검증

경계 변경 뒤 최소 검증:

```bash
uv run pytest tests/architecture/test_layer_dependencies.py
uv run pytest tests/unit/test_methods_federated_ssl.py tests/unit/test_scripts_hydra_configs.py
uv run ruff check main_server/src agent/src shared/src scripts tests methods
```
