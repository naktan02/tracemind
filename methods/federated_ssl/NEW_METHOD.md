# Federated SSL New Method Guide

이 문서는 `methods/federated_ssl/`에 새 FL SSL method를 추가할 때의 최소 변경
범위를 고정한다. 새 method 추가가 root package, scripts, agent, main_server 수정으로
번지면 method seam이 얕은 상태로 본다.

## 기본 추가 파일

새 method는 기본적으로 아래 3개 파일에서 시작한다.

```text
methods/federated_ssl/<method_name>/
  __init__.py
  descriptor.py
  local_objective.py
```

`descriptor.py`는 method identity, required views, local step, server step, runtime
capability, required capability를 소유한다. Registry는
`<method_name>/descriptor.py` convention으로 module-level `descriptor`를 발견하므로
중앙 registry 목록을 수정하지 않는다.

`local_objective.py`는 method-local objective 의미를 둔다. 일반 SSL algorithm 조각이
이미 `methods/ssl/algorithms/*`나 `methods/ssl/hooks/*`에 있으면 재사용하고, 한 method에만
필요한 agreement vote, helper consistency, partition loss routing은 method package에 둔다.
여러 FL SSL method가 공유할 수 있는 runtime 교체 지점은
`methods/federated_ssl/hooks/`의 hook surface를 먼저 재사용한다.

## 조건부 파일

아래 파일은 필요할 때만 추가한다.

```text
methods/federated_ssl/<method_name>/
  README.md             # method 읽기 경로나 원본 의미 설명이 필요할 때
  method_surface.py     # scenario default, report metadata, helper/server policy가 커질 때
  original_spec.py      # 원본 논문 parameter snapshot을 보존할 때
  compatibility.py      # 공통 capability guard로 표현 안 되는 method-only 금지 조합이 있을 때
  client_diagnostics.py # method-only client diagnostic metric이 있을 때
  partitioning.py       # sigma/psi 같은 method-local partition scheme이 있을 때
```

`use_original_parameters=false`로 실행하는 method는 `original_spec.py` 없이도 report surface가
동작해야 한다. 원본 논문 설정값을 보존하거나 `use_original_parameters=true`를 지원할 때만
`original_spec.py`를 추가한다.

`method_surface.py`는 descriptor 첫 화면을 흐리는 metadata가 생겼을 때만 둔다. 이름만
나눈 `server_policy.py`, `round_policy.py`, `runtime_requirements.py`,
`server_step_parameters.py` 같은 leaf는 만들지 않는다.

`recipe.py`처럼 `descriptor.recipe`를 다시 export하는 pass-through 파일은 만들지 않는다.
조합표가 커져 독립 테스트/문서화가 필요해질 때만 별도 파일을 검토한다.

`hooks.py` 같은 method-local hook bundle 파일은 기본으로 만들지 않는다. Method가 요구하는
hook instance는 `descriptor.py`, `method_surface.py`, `partitioning.py`처럼 해당 의미의
owner에서 드러낸다.

## Config 축 판단

새 method 추가 시 아래 config group을 먼저 늘리지 않는다.

- `local_ssl_policy`: manual baseline/ablation에서 local SSL objective source를 고르는
  축이다. Method-owned 실행에서는 descriptor required capability와 method surface default로
  파생한다. `fedmatch_agreement` 같은 method-local objective를 Hydra leaf로 추가하지 않는다.
- `local_update_profile`: local update backend/example/privacy recipe다. Manual
  baseline에는 필요하고, method-owned에서는 descriptor recipe가 지원 profile을 검증한다.
  새 method 이름별 profile을 기본으로 만들지 않는다.
- query multiview source는 현재 materialized row만 지원하므로 Hydra leaf로 직접
  고르지 않는다. live agent나 다른 view source가 실제 구현될 때만 config 축을 연다.

논문 method identity는 `conf/strategy_axes/fssl_method/<method>.yaml`과
`methods/federated_ssl/<method>/`가 소유한다. 일반 실행값, split, budget, update family,
aggregation backend는 기존 axis를 override한다.

## Runtime 경계

custom client runtime core가 필요한 method는 `descriptor.py`의
`FederatedSslLocalStepSpec.runtime_entrypoint`에 `module:function`을 명시한다. Generic
runtime resolver가 `<method>/<update-family>_training.py` 같은 파일명을 추측하게 만들지
않는다.

특정 update family에서 method를 실행하는 구현은
`methods/adaptation/<family>/federated_ssl/`에 둔다. 이때도 파일명은
`partitioned_objective_training.py`, `server_update_policy.py`처럼 execution primitive나
capability 이름을 사용하고, `<method>_training.py`처럼 method 이름으로 증식시키지 않는다.

`scripts`, `agent`, `main_server`에는 method 이름을 가진 파일을 추가하지 않는다. 새 method가
아니라 새 runtime capability가 생긴 경우에도 파일명과 interface는
`round_state_exchange`, `artifact_ref_materializer`, `server_update_policy`처럼 capability
이름을 사용한다.

`methods/federated_ssl/hooks/`에는 `partitioned_update`, `peer_context`, `server_step`,
`local_objective`처럼 여러 method가 공유할 수 있는 hook surface만 둔다. `fedmatch`,
`sigma`, `psi` 같은 method-local literal과 routing 의미는 hook package로 올리지 않는다.

## 테스트 체크리스트

- descriptor resolve가 registry 수정 없이 동작한다.
- `method_surface.py`, `original_spec.py`, `compatibility.py`, `client_diagnostics.py`가 없어도
  최소 method의 report surface와 discovery가 동작한다.
- `use_original_parameters=true`를 지원하는 method는 `original_spec.py`와 parameter override
  검증을 둔다.
- method-owned 실행은 descriptor/capability default에서 local/server policy를 파생한다.
- manual baseline은 lower-axis config 조합으로 남고, method descriptor를 참조하지 않는다.
- 새 method 이름 때문에 `scripts`, `agent`, `main_server`, root registry 목록을 수정하지 않는다.
- 필요한 smoke는 1-round 또는 5-round reduced run으로 제한하고 full-budget run은 별도 결정 뒤
  실행한다.
