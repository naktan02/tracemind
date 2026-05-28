# Federated SSL New Method Guide

이 문서는 `methods/federated_ssl/`에 새 FL SSL method를 추가할 때의 최소 경계를
고정한다. Method는 알고리즘 identity이고, `method_descriptor/`는 그 identity와
report metadata를 고르는 Hydra config group이다.

## 추가할 파일

새 method는 먼저 method-local module 하나로 시작한다.

```text
methods/federated_ssl/<method_name>/
  __init__.py
  descriptor.py
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
method 폴더를 열었을 때 "이 논문 방법론이 어떤 update family, aggregation backend,
round policy 조합으로 구성되는지"를 보는 조립표다. shared payload용
`payload_adapter_kind`는 local update와 round aggregation compatibility 표면이고,
recipe가 소유하는 trainable-state 선택 축은 `update_family_name`이다. 작은 method는
`descriptor.py`에 recipe metadata를 함께 둘 수 있고, 조합표가 커지거나 별도
테스트/문서화가 필요할 때만 `recipe.py`로 분리한다.

custom client runtime core가 필요한 method는 `descriptor.py`의
`FederatedSslLocalStepSpec.runtime_entrypoint`에 `module:function`을 명시한다.
generic runtime resolver가 `<method>/lora_classifier_training.py` 같은 파일명을
추측하게 두지 않는다. method 폴더를 읽는 사람이 descriptor에서 호출 위치를 바로
확인할 수 있어야 한다. 특정 update family에서 method를 실행하는 구현은
`methods/adaptation/<family>/federated_ssl/`에 둔다. method 폴더에는 원본 의미,
policy, parameter routing처럼 family를 넘어 유지되는 내용을 남긴다.
이때 update family 폴더에도 `<method>_training.py` 같은 파일을 기본값으로 만들지
않는다. family 쪽 파일명은 `partitioned_objective_training.py`,
`server_update_policy.py`처럼 실행 primitive나 capability 이름을 사용하고, method
이름은 descriptor entrypoint와 method package에서만 드러나게 한다.

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

method descriptor YAML은 implementation/recipe가 실제로 존재할 때만 추가한다.
threshold, LoRA rank, round 수 같은 일반 실행값의 source of truth는 Hydra leaf config에
남긴다. 단, 논문 method 원본 기본값은 `methods/federated_ssl/<method>/original_spec.py`에
두고, method descriptor YAML은 `scenario`, `use_original_parameters`,
`parameter_overrides` 같은 실행 표면만 둔다. Update family와 aggregation backend
조합은 별도 preset YAML을 만들지 않고 `round_runtime.update_family_name`과
`round_runtime.aggregation_backend_name`을 직접 override하거나
`strategy_axes/trainable_state/update_family` group을 고른다. v1 shared payload
compatibility가 필요하면 새 config는 `round_runtime.payload_adapter_kind`를 생산한다.
`round_runtime.adapter_family_name`은 제거된 v1 입력 이름이다. 새 method recipe,
runtime config, report/result reader의 조합 키로 다시 만들지 않는다.

FL simulation entrypoint의 `fl_method` section은 `FederatedSslExecutionPlan`으로
해석된다. 새 논문 method는 기본적으로 `composition_mode=method_owned`로 실행하고,
client objective/server policy/round state exchange 요구사항은 method descriptor와
method-local policy module이 소유한다. `composition_mode=manual`은 논문 method가 아니라
lower-axis mechanism 조합 baseline/ablation을 명시할 때만 쓴다. 이때 사용자는
`local_update_profile`과 최종 `round_runtime.*` leaf를 고르고, report용 lower axes는
실행 plan builder가 runtime config에서 파생한다.

`security_policy`는 method identity가 아니라 runtime capability 축이다. 현재는
`plaintext`만 지원한다. secure aggregation, DP, 암호화 artifact ref 같은 기능이 필요하면
method 이름 파일을 `agent`/`main_server`에 추가하지 말고 capability adapter와
compatibility validator를 추가한다.

`required_capabilities`에는 method가 요구하는 공통 FL SSL capability를 적는다.
예를 들어 FedMatch는 `update_partition_policy=partitioned`와
`aggregation_weight_policy=uniform`을 요구한다. `sigma/psi` 같은 partition scheme
이름과 loss routing 의미는 FedMatch method package에 둔다. client participation,
labeled exposure, server step, peer context, query multiview source도 method 전용
분기가 아니라 `FederatedSslCapabilityPlan`의 공통 축으로 검증한다.
단, `peer_context_policy=fixed_probe_output_knn` 같은 공통 capability는 mechanism만
표현한다. `num_helpers`, refresh interval처럼 논문 원본에서 온 값은
`methods/federated_ssl/<method>/original_spec.py`와 method-local policy module에 둔다.
method descriptor YAML에는 실행 override hook과 trace/report metadata만 둔다.

## Registry

`methods/federated_ssl/registry.py`는 `<method_name>/descriptor.py` convention을
import해서 module-level `descriptor` 변수를 등록한다. 같은 convention을 따르면 새
method를 추가할 때 registry 목록을 수정하지 않는다. 새 method metadata는 중앙
registry가 아니라 method-local descriptor module이 소유한다. 별도 pass-through
registry wiring shim 파일은 만들지 않는다.

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
- `1-round` smoke에서 method name, descriptor, 실제 local/server policy 변경이
  report metadata와 artifact로 함께 남는다.
- 필요한 경우에도 새 검증 실행은 `5-round` reduced run까지로 제한한다. 현재 FL SSL
  트랙에서는 새 `50-round`/full-budget run을 실행하지 않는다.
- 기존 smoke/report/manifest shape가 변하지 않는다.
- production `methods/federated_ssl/`에 dummy method를 남기지 않고
  `tests/fixtures/` 아래 test-only method fixture로 registry와 descriptor 경계를
  고정한다.
- shared payload를 바꾸면 golden fixture와 producer/consumer test를 함께 갱신한다.
