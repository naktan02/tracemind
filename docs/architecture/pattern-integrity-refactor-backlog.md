# Pattern Integrity Refactor Backlog

이 문서는 완료된 Batch transcript가 아니라 현재 guard와 남은 후보만 추적한다.
source of truth는 코드와 code-adjacent README이며, 이 문서는 다음 리팩터링이 필요한지
판단하는 보조 문서다.

## 현재 규칙

- source of truth는 가능한 한 코드와 가까운 contract, domain entity, README에 둔다.
- 단순 pass-through facade, old path wrapper, package-level re-export는 만들지 않는다.
- registry는 primitive로 유지하고 concrete import 목록을 중앙 파일에 누적하지 않는다.
- runtime adapter는 method policy를 소유하지 않고 capability bridge만 맡는다.
- 실행 기본값은 Hydra leaf config와 typed contract가 소유한다.
- builder/writer/exporter는 payload assembly와 IO 책임을 분리한다.
- artifact ref는 owner 밖에서 opaque identifier로만 다룬다.
- 새 feature/method의 1차 읽기 경로는 3-5개 주요 파일 안에서 보여야 한다.
- 이름/상수/한 줄 위임만 가진 작은 sibling 파일은 가까운 axis나 descriptor owner로 합친다.
- adapter-family FL SSL slice는 method 이름 파일로 증식시키지 않는다.
  method 의미는 `methods/federated_ssl/<method>/`, family 실행 primitive는
  `methods/adaptation/<family>/federated_ssl/`가 소유한다.

## 닫힌 영역

| 영역 | 현재 경계 | Guard |
|---|---|---|
| shared adapter contract facade | shared contract와 domain service를 직접 import | old path wrapper 재도입 금지 |
| agent local update backend | concrete backend는 `methods/adaptation/<family>/` 소유 | `agent/src/services/training/backends/training/` old path 재도입 금지 |
| SSL hooks | `methods/ssl/hooks/*`가 selection/acceptance 판단 소유 | agent acceptance policy가 판단 중복 금지 |
| main server round family | generic shared-adapter runtime과 methods aggregation core 조합 | adapter family별 round family 파일 추가 금지 |
| aggregation | strategy wiring은 methods, artifact ref materialization은 server capability | main_server에 adapter family별 aggregation 구현 금지 |
| partitioned delta average | adapter family가 partition payload materialization과 평균 backend를 소유 | registry convention만 맞추는 `methods/federated/aggregation/partitioned_*` 얇은 폴더 금지 |
| scripts report/artifact IO | builder와 writer/exporter 분리 | orchestration 파일의 JSON serialization/model save 직접 소유 금지 |
| runtime fallback | `methods/federated_ssl/runtime_fallbacks.py`로 제한 | `training_defaults.py`, `training_default_values.py` 재도입 금지 |
| runtime/objective compatibility | methods dispatcher와 family-local implementation | runtime 계층의 method/profile 의미 판단 금지 |
| prototype scoring fallback | prototype 실험 로컬 상수와 scorer policy 분리 | FL SSL runtime fallback에 prototype 실험 기본값 혼입 금지 |
| FL SSL capability axes | `capability_plan.py`와 `capability_axes.py`가 공통 capability 이름을 소유 | tiny `local_ssl_policy.py`/`server_update_policy.py` 재분리 금지 |
| FedMatch recipe metadata | `fedmatch/descriptor.py`에서 recipe metadata를 직접 노출 | pass-through `fedmatch/recipe.py` 재도입 금지 |
| adapter-family FL SSL slice | family 실행 primitive는 `methods/adaptation/<family>/federated_ssl/` 소유 | `<method>_*.py` 파일 증식 금지 |

## 2026-05-23 repo-wide 감사

자동 점검 범위는 tracked file 1041개, Python 741개, Markdown 148개, YAML 79개다.
`__init__.py` import/re-export와 `__all__` 사용은 발견되지 않았다. direct file 수가 넓은
디렉터리는 `tests/unit`, `agent/tests/unit`, `scripts/experiments/fl_ssl/federated_simulation/io`,
`docs/notes/sessions`였다. `tests`와 `docs/notes`는 성격상 허용하고, FL simulation IO는
새 파일 추가 때 builder/writer 책임을 먼저 재사용하는 감시 대상으로 둔다.

이번 작업에서 정리한 항목:

- `methods/federated_ssl/local_ssl_policy.py`, `methods/federated_ssl/server_update_policy.py`
  제거: 같은 capability plan 주변의 작은 policy name/normalizer라 `capability_axes.py`로
  합쳤다.
- `methods/federated_ssl/fedmatch/recipe.py` 제거: descriptor recipe pass-through라
  reader path만 늘렸다.
- `methods/federated/aggregation/partitioned_fedavg/` 제거: 실제 generic 산술을 소유하지
  않고 registry convention만 만족시키던 얇은 package라 제거했다. LoRA-classifier
  partitioned update 평균은
  `methods/adaptation/peft_text_encoder/aggregation/peft_encoder_partitioned_projection.py`가
  소유한다.

다음 변경 시 함께 볼 후보:

1. `scripts/experiments/query_peft_ssl/harness/common.py`: 외부 framework context와 내부
   실행값이 섞여 보인다.
2. `agent/src/services/wellbeing/child_support_service.py`: wellbeing orchestration 첫 화면이
   길다.
3. `methods/adaptation/peft_text_encoder/training_backend.py`: facade 이름을
   유지할 만큼 interface가 깊은지 삭제 테스트가 필요하다.
4. `agent/src/services/training/examples/service.py`: service 이름이 실제 boundary를
   소유하는지 다음 변경 때 확인한다.

## 활성 후보

1. `scripts/runtime_adapters/federated_server/initial_state_factory.py`의 family branch는
   현재 단일 bridge로 허용한다. 새 family/state bootstrap이 추가되어 분기가 커지면
   methods-owned dispatcher나 family-owned helper로 이동한다.
2. `docs/architecture/code-expression-guidelines.md`의 읽기 난이도 후보는 해당 파일을
   실제로 수정하는 작업에서만 정리한다. 별도 미관성 리팩터링으로 열지 않는다.
3. FedMatch, FedLGMatch, `(FL)^2` custom round policy는 method-local 구현을 먼저 만들고,
   main_server에는 generic capability가 필요할 때만 추가한다.

## 현재 작업 아님

- 논문 method 실제 학습 구현은 선택된 method와 실험 변수 고정 전까지 시작하지 않는다.
- alpha=0.1/full stress 실행은 candidate set과 metric/output metadata가 고정된 뒤 진행한다.
- `shared/src/config`나 Python default catalog를 새 source of truth로 만들지 않는다.
- `agent/`나 `main_server/`에 `fedmatch_*`, `freematch_*`, `<method>_server_policy` 같은
  method-specific runtime 파일을 추가하지 않는다.

## 새 리팩터링 종료 기준

- 책임 경계를 보존해야 하는 규칙이면 architecture guard나 unit/integration test를 추가한다.
- 관련 contract, Hydra config, code, test, active docs가 함께 맞아야 한다.
- 완료된 작업 이력을 active backlog에 길게 남기지 않는다.
