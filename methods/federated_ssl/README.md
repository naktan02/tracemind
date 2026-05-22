# Federated SSL Methods

`methods/federated_ssl/`는 SSL local update와 federated aggregation을 조합하는
FL SSL method spec을 둔다. 사람이 새 논문 방법론을 읽을 때는 먼저
`methods/federated_ssl/<method>/` 폴더를 본다. 기본 manual baseline은 이
패키지에 descriptor를 두지 않고 lower-axis 조합으로 실행한다.

## 책임

- `federated_ssl_method` 이름별 method descriptor/spec
- required views, client local step, server round/aggregation 요구사항 표현
- simulation/live runtime capability 선언
- 실제 runtime adapter가 읽을 canonical method 요구사항 제공
- method-local descriptor module의 `descriptor` 변수와 convention import trigger
- method recipe metadata와 method-only aggregation/server policy 변형
- `execution_plan.py`의 method-owned/manual composition과 security policy 검증
- local update profile support, runtime pair support, adapter family 일치 검증
- `capability_axes.py`의 client local SSL objective 이름과 server-side update/delta
  해석 이름
- 명시 training config가 없는 API/runtime 요청용 `runtime_fallbacks.py`

## 제외

- Hydra config loading
- simulation loop와 artifact/report 저장
- `LocalTrainingService`, `RoundOpenRequest` 같은 runtime 객체 생성
- 실험 실행 조합 source of truth. 일반 실행값은 `conf/strategy_axes/fl/*`가
  소유하지만, 논문 method 원본 기본값은 method package의 `original_spec.py`가
  소유한다.

위 실행 glue는 `scripts/experiments/fl_ssl/federated_simulation/`에 남긴다. 새
논문 method의 계산 core는 이 패키지나 `methods/ssl/*`, `methods/federated/*`
같은 `methods` 계층에 두고, `agent`/`main_server`에는 선택된 method의 runtime
adapter만 둔다.

method에만 종속된 aggregation 변형은 method 폴더에 둘 수 있다. 두 개 이상 method에서
공유되는 평균/투영/adapter payload 해석은 `methods/federated/aggregation/*` 또는
`methods/adaptation/<family>/*`로 승격한다.

## 읽기 경로

새 FL SSL method를 볼 때는 `methods/federated_ssl/README.md`,
`methods/federated_ssl/<method>/README.md`, `<method>/descriptor.py`,
`<method>/local_objective.py` 순서로 먼저 읽는다. server/round policy는 method가 실제로
custom state exchange나 server step을 요구할 때만 따라간다. 공통 capability 이름은
`capability_plan.py`와 `capability_axes.py`를 함께 읽으면 된다.

`recipe.py` 같은 별도 조립표 파일은 descriptor가 흐려질 만큼 recipe metadata가 커질 때만
분리한다. 단순히 `descriptor.recipe`를 다시 export하는 파일은 두지 않는다.

`FederatedSslExecutionPlan`은 entrypoint의 `fl_method`와 `security_policy`를 해석해
상위 method가 소유하는 실행인지, lower-axis를 직접 조합하는 manual baseline인지
명시한다. report에는 `execution_role=manual_baseline` 또는
`execution_role=method_owned`가 남는다. manual mode의 `descriptor_name`은 비워
두고 lower-axis report 값은 실제 `query_ssl_method`, `local_update_profile`과
최종 `round_runtime.*` leaf에서 파생하고 stale preset metadata는 compatibility
검증에서 무시한다. 현재 security policy는 `plaintext`만 지원하고, secure
aggregation/DP/암호화 artifact ref는 method가 아니라 runtime capability 축으로 추가한다.

`FederatedSslCapabilityPlan`은 `server_step_policy`와 `server_update_policy`를
분리한다. 전자는 server-side supervised seed step 같은 추가 학습 여부이고, 후자는
client가 제출한 merged/partitioned delta를 server가 어떤 의미로 해석할지다.
`local_ssl_policy`는 local pseudo-label/consistency objective 이름만 소유한다.
FixMatch/FlexMatch/FreeMatch의 threshold와 state parameter는 기존
`query_ssl_method`가 계속 소유하고, FedMatch agreement vote 같은 method-local 의미는
`methods/federated_ssl/<method>/`가 소유한다.

새 method의 descriptor는 `methods/federated_ssl/<method>/descriptor.py`가
소유한다. Registry는 `<method>/descriptor.py` module convention을 import해
`descriptor` 변수를 등록하므로, 같은 convention을 따르면 registry 목록을 수정하지
않는다. 별도 registry wiring shim 파일은 두지 않는다.

method-owned client runtime core는 descriptor의
`local_step.runtime_entrypoint`에 `module:function` 형식으로 명시한다. 새 method가
adapter-family custom local loop를 요구하면 사람이 `<method>/descriptor.py`에서
호출 위치를 확인할 수 있어야 한다. LoRA-classifier처럼 family-specific 실행 구현은
`methods/adaptation/<family>/federated_ssl/` 아래에 두고, generic runtime이
`<method>/lora_classifier_training.py` 같은 파일명을 추측하게 만들지 않는다.

새 method 추가 절차는 `NEW_METHOD.md`를 먼저 따른다.
