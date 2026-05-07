# Round Services

이 디렉터리는 main_server가 소유하는 FL round runtime의 핵심 경로다.

## 이 폴더가 하는 일

1. 현재 active manifest를 기준으로 training task를 만든다.
2. agent update를 수집하고 검증한다.
3. aggregation으로 다음 shared adapter state를 만든다.
4. 다음 `ModelManifest / PrototypePack` 쌍 발행까지 연결한다.

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

## 파일 역할 빠른 맵

- `boundary/models.py`
  - round domain/canonical request 모델
- `boundary/payloads.py`
  - API payload shape
- `boundary/mappers.py`
  - payload를 canonical request로 정규화
- `runtime/`
  - server-owned runtime config, compatibility, factory wiring
- `round_manager_service.py`
  - training task 생성, aggregation 결과 publication
  - `artifact_ref`/`payload_ref`는 파일 경로가 아니라 server-owned ref로 다루고,
    실제 저장소 해석은 infrastructure repository에 위임
- `round_lifecycle_service.py`
  - open/update/finalize orchestration
- `active_manifest_service.py`
  - 서버 current `ModelManifest` 저장/활성화
- `families/`
  - shared adapter family contract metadata를 generic round runtime으로 연결
  - concrete family별 파일을 두지 않고 `shared/src/contracts/adapter_family_metadata.py`
    와 aggregation backend 조합으로 해석한다
- `aggregation/`
  - server-owned aggregation backend registry와 methods core adapter
  - registry는 lookup/catalog만 맡고 backend factory 등록은 각 backend module 옆
    decorator가 소유한다
  - adapter family별 payload/state materialization만 맡고 FedMatch/FedLGMatch 같은
    method-specific server policy를 소유하지 않는다
  - `lora_classifier.fedavg`는 inline delta smoke 경로를 집계하고,
    artifact-ref-only update는 artifact materializer가 붙기 전까지 거부한다
- `acceptance/`
  - 중복 제출, 신뢰 정책, 라운드 상태 검증

## 새 전략 추가 시 어디를 보는가

- aggregation backend 추가: server adapter/wiring은 `aggregation/`, 순수 method
  계산은 `methods/federated/aggregation/`
- adapter family 추가: `shared/src/contracts/adapter_contract_families/` +
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
