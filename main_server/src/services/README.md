# Main Server Services

이 디렉터리는 main_server가 소유하는 중앙 orchestration 서비스를 둔다.

## 하위 패키지 역할

- `federation/rounds/`
  - FL round open/update/finalize, aggregation, runtime wiring
- `federation/prototypes/`
  - prototype pack/build-state rebuild, publication, activation 같은
    server-owned prototype artifact lifecycle
## newcomer용 읽기 순서

### 1. round lifecycle부터 보고 싶을 때

1. `federation/rounds/README.md`
2. `federation/rounds/round_lifecycle_service.py`
3. `federation/rounds/round_manager_service.py`

### 2. prototype rebuild/publication부터 보고 싶을 때

1. `federation/prototypes/prototype_rebuild_service.py`
2. `federation/prototypes/stored_input_rebuild_service.py`
3. `federation/prototypes/models.py`
4. `infrastructure/repositories/prototype_rebuild_input_repository.py`
5. `federation/prototypes/publication_strategies.py`

## 경계 원칙

- federation round orchestration은 `federation/rounds/`가 소유한다.
- prototype pack/build-state 생성과 publication은
  `federation/prototypes/`가 소유한다.
- `federation/assets/` 같은 넓은 catch-all service package를 새 source로 되살리지
  않는다. artifact 종류가 필요하면 resource/capability 이름으로 좁힌 package를 둔다.
- prototype rebuild input row는 `ServerReferencePrototypeSourceRow`로 표현되는
  server-owned reference만 허용한다. agent raw/query text는 이 경로로 승격하지 않는다.
- 공용 계약은 `shared/src/contracts/`를 기준으로 읽는다.
- 새 adapter family나 FL SSL method 때문에 `federation/rounds/families/`에
  family/method-specific 파일을 추가하지 않는다. 이 폴더는 shared payload registry와
  methods-owned aggregation backend를 조합하는 generic runtime seam만 둔다.
