# Federated SSL Module Deepening Plan

이 문서는 `methods/federated_ssl/`와 method-local package를 더 깊은 Module로
정리하기 위한 active plan이다. 목표는 파일 수를 줄이는 것이 아니라, 새 FL SSL
method를 추가할 때 읽기 경로와 수정 위치가 예측 가능해지는 것이다.

## 목표 구조

새 method reader의 1차 경로는 아래 3-5개 파일 안에 들어와야 한다.

```text
methods/federated_ssl/README.md
methods/federated_ssl/<method>/README.md
methods/federated_ssl/<method>/descriptor.py
methods/federated_ssl/<method>/objective/... 또는 local objective entrypoint
methods/federated_ssl/<method>/runtime/... 필요 시
```

공통 FL SSL package는 method discovery, execution plan, capability vocabulary,
diagnostics, runtime fallback을 소유한다. 일반 SSL local objective 조각은
`methods/ssl/hooks`나 `methods/ssl/algorithms`가 소유하고, `federated_ssl`에는 FL
method가 추가로 요구하는 peer context, partition, server policy 의미만 남긴다.

## 정리 원칙

- `methods/ssl` hook을 재사용할 수 있는 confidence mask, pseudo-label target,
  consistency CE, teacher/selection 의미를 `federated_ssl`에 다시 구현하지 않는다.
- FedMatch의 agreement vote, inter-client KL, sigma/psi partition routing,
  sparse sync policy, labels-at-client/server policy는 method-local 의미로 남긴다.
- method-local module import convention은 한 helper가 소유한다. `original_spec`,
  `compatibility`, `runtime_requirements`, `server_step_parameters`,
  `client_diagnostics`별 import 코드를 각 파일에 복제하지 않는다.
- scenario default와 report surface metadata는 descriptor 첫 화면을 흐리지 않게
  별도 method surface Module로 낮춘다.
- 얇은 pass-through 파일은 만들지 않는다. 새 파일은 독립 변경 이유나 독립 테스트
  표면이 있을 때만 만든다.

## 단계

### 0단계: 기준 고정

- 현재 FedMatch smoke/report verifier가 통과한 상태를 기준으로 한다.
- active config leaf는 유지한다. 새 Hydra axis를 열지 않는다.
- `data/**`, `runs/**` historical artifacts는 정리 대상이 아니다.

### 1단계: method-local module import 통합

대상:

- `methods/federated_ssl/method_module_resolution.py`
- `methods/federated_ssl/method_parameters.py`
- `methods/federated_ssl/hooks/local_objective.py`
- `methods/federated_ssl/hooks/server_step.py`
- `methods/federated_ssl/compatibility.py`
- `methods/federated_ssl/diagnostics/client.py`

작업:

- `import_method_family_module(method_name, module_leaf, required=False)` 같은 공통
  helper를 둔다.
- 각 호출부는 module leaf와 required/optional 의미만 넘긴다.
- behavior와 error message는 기존 테스트가 검증하는 범위에서 보존한다.

종료 기준:

- FedMatch capability/unit tests 통과.
- architecture guard에서 concrete method import 확산이 없다.

### 2단계: method config surface default resolver 축소

대상:

- `methods/federated_ssl/method_config_surface.py`
- `methods/federated_ssl/fedmatch/descriptor.py`

작업:

- 반복된 default resolver를 table-driven helper로 줄인다.
- scenario별 default table은 method-local surface Module로 옮길지 결정한다.
- descriptor는 identity, required capability, runtime entrypoint가 먼저 보이게 둔다.

종료 기준:

- Hydra compose tests가 FedMatch scenario default를 그대로 확인한다.
- `fssl_method=fedmatch` public leaf 의미가 변하지 않는다.

### 3단계: FedMatch surface 깊게 만들기

후보 구조:

```text
methods/federated_ssl/fedmatch/
  descriptor.py
  method_surface.py
  original_spec.py
  local_objective.py
  partitioning.py
  local_runtime.py
```

작업:

- `server_policy.py`, `round_policy.py`, `runtime_requirements.py`,
  `server_step_parameters.py`처럼 얇은 policy/runtime surface 파일은
  `method_surface.py` owner로 합친다.
- `parameter_routing.py`와 `partitioned_runtime_plan.py`는 단일 `partitioning.py`
  owner로 합친다.

종료 기준:

- FedMatch README의 읽기 경로가 실제 파일 구조와 맞는다.
- old pass-through facade를 만들지 않는다.

### 4단계: FedMatch local objective를 SSL hook 재사용 기준으로 분리

대상:

- `methods/federated_ssl/fedmatch/local_objective.py`
- `methods/ssl/hooks/*`

작업:

- confidence mask와 CE subroutine은 `methods/ssl/hooks` 재사용 후보로 검토한다.
- FedMatch agreement vote와 sigma/psi partition loss는 method-local objective로
  유지한다.
- pure sequence helper와 tensor helper가 같은 agreement 의미를 중복 구현하는지
  테스트 기준으로 정리한다.

종료 기준:

- 원본 FedMatch tensor core tests 통과.
- 일반 SSL hook에 FedMatch-only state나 policy가 새지 않는다.

### 5단계: root package 읽기 경로 정리

후보 구조:

```text
methods/federated_ssl/
  base.py
  registry.py
  planning/
  capabilities/
  diagnostics/
  runtime_fallbacks.py
```

작업:

- 파일 이동은 import blast radius가 작아진 뒤 진행한다.
- compatibility shim은 만들지 않고 direct import를 갱신한다.
- code-adjacent README를 새 구조에 맞춘다.

종료 기준:

- 새 method 추가 절차가 README와 `NEW_METHOD.md`에서 3-5개 주요 파일로 설명된다.
- layer dependency guard와 focused FL SSL tests가 통과한다.

## 검증 세트

각 단계는 변경 범위에 맞게 아래에서 고른다.

```bash
uv run pytest tests/unit/test_methods_federated_ssl.py
uv run pytest tests/unit/test_methods_federated_capabilities.py -k fedmatch
uv run pytest tests/unit/test_methods_fedmatch_original_core.py
uv run pytest tests/unit/test_methods_fedmatch_peft_partitioned_training.py
uv run pytest tests/unit/test_scripts_hydra_configs.py -k fedmatch
uv run pytest tests/architecture/test_layer_dependencies.py -k fedmatch
uv run ruff check methods/federated_ssl tests/unit/test_methods_federated_ssl.py tests/unit/test_methods_federated_capabilities.py
```

## 현재 상태

- 0단계 완료.
- 1단계 완료. method-local `original_spec`, `compatibility`,
  `runtime_requirements`, `server_step_parameters`, `client_diagnostics` import는
  `method_module_resolution.import_method_family_module()`을 공통 convention으로
  사용한다.
- 2단계 완료. `method_config_surface`의 public default resolver 함수들은 유지하되,
  scenario default, method-local attribute default, 단일 capability fallback을
  `_default_capability_name()`으로 모았다. FedMatch scenario default table은 아직
  descriptor에 남겨 두고, 3단계에서 `method_surface.py`로 분리할지 함께 판단한다.
- 3단계 일부 완료. FedMatch `server_policy.py`, `round_policy.py`,
  `runtime_requirements.py`, `server_step_parameters.py`를 `method_surface.py`로
  합치고, `descriptor.py`는 method identity/runtime capability와 required
  capability 중심으로 낮췄다. 공통 method module resolver는 기존 leaf convention을
  유지하되 얇은 surface leaf를 `method_surface.py`로 fallback한다.
- 3단계 완료. `parameter_routing.py`와 `partitioned_runtime_plan.py`는 단일
  `partitioning.py`로 이동했고, 호출부는 compatibility shim 없이 direct import를
  사용한다.
- 4단계 일부 완료. FedMatch tensor core의 confidence mask와 agreement CE 계산은
  `methods/ssl/hooks/masking.py`, `methods/ssl/hooks/consistency.py` primitive를
  재사용한다. helper/local agreement vote, inter-client KL, sigma/psi L1/L2 routing은
  FedMatch-only 의미라 method-local objective에 남긴다.
- 5단계 일부 완료. Root `client_diagnostics.py`와 `diagnostic_sampling.py`는
  `diagnostics/client.py`, `diagnostics/sampling.py`로 이동했고, scripts/tests는
  compatibility shim 없이 새 direct import를 사용한다.
- 5단계 일부 완료. Root `capability_axes.py`와 `capability_plan.py`는
  `capabilities/axes.py`, `capabilities/plan.py`로 이동했다.
- 5단계 일부 완료. Root `local_objective.py`, `peer_context.py`, `server_step.py`,
  `update_partition.py`는 `hooks/local_objective.py`, `hooks/peer_context.py`,
  `hooks/server_step.py`, `hooks/partitioned_update.py`로 이동했다. `hooks/`는
  FedMatch 이름이나 `sigma/psi` 의미를 소유하지 않고, 여러 FL SSL method가 공유할
  hook surface와 common primitive만 소유한다.
- 5단계 일부 완료. `methods/federated_ssl/README.md`는 root 공통 언어, method-local
  논문 의미, adaptation-family 실행 primitive의 3분할 읽기 경로로 축약했다.
  Architecture guard는 flat moved module과 FedMatch의 얇은 policy/runtime leaf
  재도입을 막는다.
