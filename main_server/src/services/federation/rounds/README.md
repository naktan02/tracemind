# Round Services

이 디렉터리는 main_server가 소유하는 FL round runtime의 핵심 경로다.

## 이 폴더가 하는 일

1. 현재 active manifest를 기준으로 training task를 만든다.
2. agent update를 수집하고 검증한다.
3. aggregation으로 다음 shared adapter state를 만든다.
4. 다음 `ModelManifest`와 필요한 optional auxiliary artifact 발행까지 연결한다.

## 먼저 읽을 파일

### end-to-end 흐름을 한 번에 보고 싶을 때

1. `round_lifecycle_service.py`
2. `round_manager_service.py`
3. `families/registry.py`
4. `aggregation/registry.py`

### API 경계와 canonical shape를 보고 싶을 때

1. `boundary/models.py`
2. `boundary/payloads.py`
3. `boundary/mappers.py`

### runtime wiring 축을 보고 싶을 때

1. `runtime/config.py`
2. `runtime/factory.py`
3. `runtime/compatibility.py`
4. `server_policy/executor.py`
5. `round_state_exchange/executor.py`

## 파일 역할 빠른 맵

- `boundary/models.py`
  - round domain/canonical request 모델
- `boundary/payloads.py`
  - API payload shape
- `boundary/mappers.py`
  - payload를 canonical request로 정규화
- `runtime/`
  - server-owned runtime config, compatibility, factory wiring
- `server_policy/`
  - methods-owned method descriptor의 server policy 요구사항을 main_server live runtime
    capability로 검증한다. policy 의미 자체는 `methods/federated_ssl/<method>/`에 둔다
- `round_state_exchange/`
  - methods-owned method descriptor가 요구하는 client metric/state summary를
    main_server live runtime capability로 검증하고 publication summary에 분리해 남긴다
- `round_manager_service.py`
  - training task 생성, aggregation 결과 publication
  - `artifact_ref`/`payload_ref`는 파일 경로가 아니라 server-owned ref로 다루고,
    실제 저장소 해석은 infrastructure repository에 위임
- `round_lifecycle_service.py`
  - open/update/finalize orchestration
- `active_manifest_service.py`
  - 서버 current `ModelManifest` 저장/활성화
- `families/`
  - shared adapter payload registry를 generic round runtime으로 연결
  - concrete family별 파일을 두지 않고 registered payload family와 aggregation backend
    조합으로 해석한다
- `aggregation/`
  - server-owned aggregation backend registry와 methods strategy executor
  - `fedavg.py`, `fedprox.py` 같은 aggregation method 파일은 두지 않는다
  - `classifier_head.py`, `peft_classifier.py` 같은 adapter family 단위 module도
    두지 않는다
  - registry는 explicit test/backend override와 methods strategy resolve만 맡는다
  - adapter payload projection은 `methods/adaptation/<family>/`가 소유하고,
    재사용 aggregation backend 의미는 `methods/federated/aggregation/`이 소유한다
  - method-only aggregation/server policy 변형은 `methods/federated_ssl/<method>/`가
    소유하고, server-owned artifact ref 생성만 runtime capability로 제공한다
  - aggregation 결과의 `aggregated_artifacts` 저장은 main_server가 수행하지만,
    artifact payload 의미와 next-state projection은 `methods/adaptation/<family>/`가
    소유한다
  - LoRA-classifier FedAvg methods strategy는 inline delta와 server-owned
    `aggregation_artifact::` JSON artifact-ref update를 집계한다. client update는
    base revision 기준 delta이고, 다음 state가 참조하는 server aggregate artifact는
    누적된 global LoRA/head parameter snapshot이다. `agent-local://` ref는 서버
    direct accept 단계에서 거부한다. agent/simulation runtime이 먼저 server-owned
    artifact ref로 upload/materialize해야 한다
  - update accept 단계는 envelope의 active manifest revision뿐 아니라 family별
    payload compatibility도 확인한다. LoRA-classifier는 payload의 model/base
    revision/scope, backbone, LoRA config, label schema가 active state와 같아야 한다
- `acceptance/`
  - 중복 제출, 신뢰 정책, 라운드 상태 검증

## 새 전략 추가 시 어디를 보는가

- aggregation backend 추가: method-only 변형은 `methods/federated_ssl/<method>/`에,
  재사용 backend/projection은 `methods/federated/aggregation/` 또는
  `methods/adaptation/<family>/`에 두고, server wiring은 기존 generic
  `aggregation/executor.py`와 registry를 재사용한다
- payload adapter kind 추가: `shared/src/contracts/adapter_contract_families/` +
  aggregation backend. `families/`에 family-specific 파일을 추가하지 않는다.
- server runtime 기본 축 변경: `runtime/config.py`

새 FL SSL method가 round별 state exchange, client weighting, pseudo-label statistics,
server-side calibration을 요구하면 먼저
`methods/federated_ssl/<method>/server_policy.py` 또는 `round_policy.py`에 의미를
둔다. `main_server`에는 그 policy를 실행하기 위한 generic runtime capability만
추가한다. method 이름을 가진 server 파일이 늘어나면 runtime adapter가 method
framework 역할을 흡수하고 있다는 신호다.

새 family나 backend를 추가할 때는
`docs/contracts/algorithm_extension_guide.md`와
`docs/contracts/strategy_addition_playbook.md`를 먼저 읽는 편이 빠르다.
