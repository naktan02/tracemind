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
3. `adapter_family_service.py`
4. `aggregation_service.py`

### API 경계와 canonical shape를 보고 싶을 때

1. `models.py`
2. `payloads.py`
3. `mappers.py`

### runtime wiring 축을 보고 싶을 때

1. `runtime_config.py`
2. `runtime_factory.py`
3. `runtime_compatibility.py`

## 파일 역할 빠른 맵

- `models.py`
  - round domain/canonical request 모델
- `payloads.py`
  - API payload shape
- `mappers.py`
  - payload를 canonical request로 정규화
- `round_manager_service.py`
  - training task 생성, aggregation 결과 publication
- `round_lifecycle_service.py`
  - open/update/finalize orchestration
- `adapter_family_service.py`
  - adapter family별 state/update/payload 해석과 aggregation wiring
- `aggregation_service.py`
  - concrete aggregation backend registry와 구현
- `update_acceptance_policy.py`
  - 중복 제출, 신뢰 정책, 라운드 상태 검증

## 새 전략 추가 시 어디를 보는가

- aggregation backend 추가: `aggregation_service.py`
- adapter family 추가: `adapter_family_service.py` + `shared/src/contracts/adapter_contracts.py`
- server runtime 기본 축 변경: `runtime_config.py`

새 family나 backend를 추가할 때는
`docs/contracts/algorithm_extension_guide.md`와
`docs/contracts/strategy_addition_playbook.md`를 먼저 읽는 편이 빠르다.
