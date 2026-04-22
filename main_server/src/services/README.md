# Main Server Services

이 디렉터리는 main_server가 소유하는 중앙 orchestration 서비스를 둔다.

## 하위 패키지 역할

- `rounds/`
  - FL round open/update/finalize, aggregation, runtime wiring
- `prototypes/`
  - prototype pack/build-state publication과 rebuild runtime
- `experiments/`
  - 개발자용 read-only experiment catalog와 이후 workspace compiler 진입점

## newcomer용 읽기 순서

### 1. round lifecycle부터 보고 싶을 때

1. `rounds/README.md`
2. `rounds/round_lifecycle_service.py`
3. `rounds/round_manager_service.py`

### 2. prototype rebuild/publication부터 보고 싶을 때

1. `prototypes/prototype_rebuild_service.py`
2. `prototypes/stored_input_rebuild_service.py`
3. `prototypes/publication_strategies.py`

## 경계 원칙

- round orchestration은 `rounds/`가 소유한다.
- prototype pack/build-state 생성과 publication은 `prototypes/`가 소유한다.
- 공용 계약은 `shared/src/contracts/`를 기준으로 읽는다.
