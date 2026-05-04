# Main Server Services

이 디렉터리는 main_server가 소유하는 중앙 orchestration 서비스를 둔다.

## 하위 패키지 역할

- `federation/rounds/`
  - FL round open/update/finalize, aggregation, runtime wiring
- `federation/assets/prototypes/`
  - prototype pack/build-state rebuild, publication, activation 같은
    federation asset lifecycle
- `experiment_workspace/`
  - 개발자용 experiment workspace catalog/compile/run/save backend
  - `catalog/`, `compiler/`, `run_execution/`가 각각 선택지 노출, Hydra preview 번역,
    local run orchestration을 맡는다.

## newcomer용 읽기 순서

### 1. round lifecycle부터 보고 싶을 때

1. `federation/rounds/README.md`
2. `federation/rounds/round_lifecycle_service.py`
3. `federation/rounds/round_manager_service.py`

### 2. prototype rebuild/publication부터 보고 싶을 때

1. `federation/assets/prototypes/prototype_rebuild_service.py`
2. `federation/assets/prototypes/stored_input_rebuild_service.py`
3. `federation/assets/prototypes/publication_strategies.py`

## 경계 원칙

- federation round orchestration은 `federation/rounds/`가 소유한다.
- prototype pack/build-state 생성과 publication은
  `federation/assets/prototypes/`가 소유한다.
- 개발자 실험 웹용 catalog/compile/workspace/run surface는
  `experiment_workspace/`가 소유한다.
- 공용 계약은 `shared/src/contracts/`를 기준으로 읽는다.
