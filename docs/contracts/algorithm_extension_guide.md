# 알고리즘 확장 가이드

이 문서는 TraceMind에서 새 SSL, FL, adaptation, prototype 전략을 추가할 때 보는
짧은 전환 가이드다. 상세 구현 cookbook이 아니라 경계와 순서를 고정한다.

## 핵심 원칙

- 계산 core는 `methods/`에 둔다.
- 실행 조합과 파라미터는 `conf/`에 둔다.
- scripts는 Hydra entrypoint와 실험 harness만 담당한다.
- agent/main_server는 production/runtime adapter로 선택된 core를 호출한다.
- shared는 contract, domain entity, canonical payload 해석만 담당한다.
- 새 method/algorithm 추가 때문에 agent/main_server에 method-specific 파일을 만들지
  않는다. runtime 파일은 capability 이름으로 확장한다.
- 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다.
  이 폴더는 descriptor, optional recipe, local objective, server/round policy, method-only
  aggregation 변형을 묶는다. 두 개 이상 방법론에서 공유되는 계산은 축별 methods
  패키지로 승격한다.
- FL SSL 논문 method는 선택 전에는 구현 폴더나
  `conf/strategy_axes/fssl_method/<method>.yaml` placeholder를 만들지 않는다.
  현재 선택된 첫 method는 FedMatch이며, capability surface와 tensor local objective
  core, PEFT text encoder partitioned step core까지 열린 상태다.
- 단일 사용처용 추상화와 compatibility layer는 만들지 않는다.

## 전략 표면

| 축 | core owner | runtime owner | config | 대표 테스트 |
|---|---|---|---|---|
| SSL objective | `methods/ssl/*` | `scripts/experiments/*`, agent runtime 필요 시 adapter | `conf/strategy_axes/ssl_objective/*` | `tests/unit`, `tests/integration` |
| Pseudo-label selection | `methods/ssl/hooks/*` | agent acceptance/selection adapter, scripts control runner | `conf/strategy_axes/ssl_objective/pseudo_label_selection/*` | selection unit, central control smoke |
| Query SSL adaptation | `methods/adaptation/query_text_views/*` | scripts central SSL runners | `conf/entrypoints/central_ssl_control/*` | trainer smoke, artifact metadata |
| PEFT adapter | `methods/adaptation/*` | scripts trainer, future agent adapter | `conf/strategy_axes/model_architecture/peft/*` | adapter builder unit |
| Prototype building | `methods/prototype/building/*` | scripts prototype CLI, main_server publication adapter | `conf/strategy_axes/prototype/*` | builder unit, publication integration |
| Prototype scoring/evidence | `methods/prototype/scoring/*`, `methods/prototype/evidence/*` | agent scoring/evidence backends, scripts analysis | training objective config | scoring/evidence unit |
| Prototype SSL method | `methods/ssl/*` + `methods/prototype/*` | central/FL SSL runner | SSL/prototype strategy axes | SSL comparison smoke |
| FL shard policy | `methods/federated/shard_policy/*` | scripts FL simulation | `conf/strategy_axes/fl_topology/shard_policy/*` | shard determinism unit |
| FL client participation | `methods/federated/participation.py` | scripts FL simulation round loop | `conf/strategy_axes/fl_topology/client_participation/*` | participation unit, round report |
| FL aggregation weighting | `methods/federated/aggregation_weighting.py` | family FedAvg cores | `conf/strategy_axes/fl_topology/aggregation_weight/*` | weighting unit, aggregation unit |
| FL SSL capability plan | `methods/federated_ssl/capability_plan.py`, `capability_axes.py` | scripts FL simulation compatibility adapters, future runtime adapters | `conf/strategy_axes/fl_topology/*`, `conf/strategy_axes/ssl_objective/local_ssl_policy/*` | capability compatibility unit |
| FL aggregation | method-only는 `methods/federated_ssl/<method>/aggregation.py`, 재사용 backend는 `methods/federated/aggregation/*` + update-family projection | main_server aggregation adapter | update family + aggregation backend leaf | aggregation unit, round integration |
| FL SSL method descriptor | `methods/federated_ssl/*` | scripts FL simulation, future runtime translation | `conf/strategy_axes/fssl_method/*`는 method name/scenario/overrides만 선택 | simulation smoke |
| Secure update codec | `shared/src/services/*` | agent/main_server privacy/update boundary | shared config or runtime config | contract/integration |

## 추가 순서

1. 논문/방법 이름, 고정 변수, 변경 변수, dataset split, seed, metric, output metadata를 먼저 적는다.
2. FL SSL 논문 방법론이면 `docs/contracts/fl_ssl_method_capability_matrix.md`에서
   기존 capability 축으로 표현 가능한지 먼저 확인한다.
3. 논문 방법론이면 `methods/federated_ssl/<method>/descriptor.py`에 recipe metadata로
   조합을 먼저 드러내고, 조합표가 커질 때만 `recipe.py`로 분리한다.
4. 재사용 계산이면 `methods/<axis>/<method_name>/` 또는 기존 축 파일에 core를 둔다.
5. cross-boundary payload가 필요하면 `shared/src/contracts/`와 contract README를 먼저 갱신한다.
6. production/runtime 연결이 필요할 때만 `agent/` 또는 `main_server` adapter를 추가한다.
7. 실험 실행은 `scripts/experiments/` entrypoint나 기존 runner에 얇게 연결한다.
8. Hydra 조합은 `conf/entrypoints/`, 파라미터 축은 `conf/strategy_axes/`, 실행 조건 묶음은 `conf/run_controls/`에 둔다.
9. core unit test와 경계 integration test를 추가한다.
10. architecture guard가 금지 import와 FL method descriptor placeholder를 잡는지 확인한다.

## 위치 판단

- 수식, loss, threshold 계산, assignment, aggregation 산술은 `methods/`.
- DTO, manifest, update payload, canonical enum은 `shared/`.
- raw text, local state, local model execution, privacy guard runtime은 `agent/`.
- round lifecycle, update acceptance, publication, repository는 `main_server/`.
- method identity, local objective, server/round policy 의미는 `methods/`.
- Hydra compose, sweep, report, artifact dump는 `scripts/`.
- UI type generation과 API consumer shell은 `apps/`.

## Config 판단

- `threshold=0.95`, `rank=8`, `alpha=0.3`, `rounds=1/5` 같은 값은 YAML 값이다.
- `run_controls/fl_ssl/budget=main` 같은 full-budget 값도 YAML 값이다. 긴 실행은
  smoke/reduced 검증 뒤 명시적으로 선택한다.
- 논문/방법 단위로 독립 테스트와 README가 필요하면 폴더로 둔다.
- 사람이 직접 실행하는 조합은 `conf/entrypoints/`.
- 비교 track의 반복 실행 조건은 `conf/run_controls/`.
- 교체 가능한 leaf 전략은 `conf/strategy_axes/`.

## 금지 패턴

- `scripts`에서 운영 후보 알고리즘 core를 소유한다.
- `agent`나 `main_server`에 여러 연구 후보 구현을 직접 쌓는다.
- `shared`에 실험 편의 helper를 승격한다.
- `methods`가 `agent`, `main_server`, `scripts`를 import한다.
- `scripts`가 `scripts/runtime_adapters/` 밖에서 `agent.src`, `main_server.src`를 import한다.
- package-level export를 만들려고 `__all__`를 추가한다.

## 완료 기준

- 새 method의 시작 위치가 파일명만 보고 보인다.
- config 선택 위치와 Python 구현 위치가 분리돼 있다.
- 동일 축에 다음 알고리즘을 추가할 수 있다.
- core test와 runtime/entrypoint smoke가 각각 맡은 것을 검증한다.
- docs는 active rule만 남기고 긴 실행 로그나 대화 전문을 담지 않는다.
