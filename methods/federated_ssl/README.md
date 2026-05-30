# Federated SSL Methods

`methods/federated_ssl/`는 SSL local update와 federated aggregation을 조합하는
FL SSL method 의미를 소유한다. `scripts`, `agent`, `main_server`는 FedMatch 같은
논문 method 이름으로 분기하지 않고, 여기서 선언한 descriptor/capability를 읽는다.

## 읽기 순서

새 method나 기존 method 동작을 볼 때는 아래 순서로 시작한다.

1. `methods/federated_ssl/README.md`
2. `methods/federated_ssl/<method>/README.md`
3. `methods/federated_ssl/<method>/descriptor.py`
4. `methods/federated_ssl/<method>/local_objective.py`
5. 필요할 때만 `<method>/method_surface.py`, `<method>/partitioning.py`,
   `methods/adaptation/<family>/federated_ssl/*`

## 파일 지도

| 위치 | 역할 |
|---|---|
| `base.py` | FL SSL method descriptor 계약 |
| `registry.py` | `<method>/descriptor.py` convention discovery |
| `execution_plan.py` | `method_owned`와 `manual_baseline` 실행 구분 |
| `compatibility.py` | descriptor/profile/capability 조합 검증 |
| `method_config_surface.py` | method-owned default와 report metadata resolver |
| `method_parameters.py` | 원본 method parameter snapshot과 override 병합 |
| `capabilities/axes.py` | local SSL policy, server update policy 이름 |
| `capabilities/plan.py` | runtime이 읽는 FL SSL capability 조합 |
| `hooks/local_objective.py` | method-owned objective protocol/result 타입 |
| `hooks/peer_context.py` | helper/peer context policy와 selection hook |
| `hooks/server_step.py` | server-side supervised seed step hook |
| `hooks/partitioned_update.py` | partitioned update hook metadata |
| `diagnostics/client.py` | method-local client diagnostic discovery/summary |
| `diagnostics/sampling.py` | diagnostic/probe row sampling algorithm |
| `runtime_fallbacks.py` | API/runtime no-config 요청용 named fallback |
| `local_supervision.py` | client/server labeled row 노출 regime |

`<method>/descriptor.py`는 method identity, required views, runtime entrypoint,
required capability를 보여주는 첫 화면이다. Scenario default, report metadata,
helper/server policy metadata처럼 descriptor를 흐리는 값은 `<method>/method_surface.py`에
둔다. 원본 논문 parameter snapshot은 `<method>/original_spec.py`가 소유한다.

## 경계

- Hydra config loading, simulation loop, artifact/report 저장은 이 패키지의 책임이
  아니다. 실행 glue는 `scripts/experiments/fl_ssl/federated_simulation/`에 둔다.
- PEFT text encoder 같은 family-specific 실행 구현은
  `methods/adaptation/<family>/federated_ssl/`에 둔다.
- `hooks/`는 여러 FL SSL method가 공유할 수 있는 교체 지점의 interface와 공통
  primitive만 둔다. Method-local 이름과 parameter 의미는 `<method>/` 안에 둔다.
- FedMatch의 `sigma/psi`, agreement vote, helper policy처럼 method-local 의미는
  `methods/federated_ssl/<method>/` 밖으로 올리지 않는다.
- `server_step_policy`는 server-side 추가 학습 여부이고, `server_update_policy`는
  client delta를 server가 어떤 의미로 해석할지다. 둘은 같은 축이 아니다.
- `labeled_exposure_policy`와 `local_supervision_regime`은 FedMatch 조건문이 아니라
  모든 FL SSL method가 공유하는 labeled row 노출 계약이다.
- `recipe.py`처럼 descriptor 값을 다시 export하는 얇은 파일은 만들지 않는다.
- `server_policy.py`, `round_policy.py`, `runtime_requirements.py`,
  `server_step_parameters.py`처럼 policy 이름만 나누는 leaf는 만들지 않고
  `<method>/method_surface.py`에 모은다.

새 method 추가 절차는 `NEW_METHOD.md`를 먼저 따른다.
