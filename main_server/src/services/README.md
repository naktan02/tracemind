# Main Server Services

이 디렉터리는 main_server가 소유하는 중앙 orchestration 서비스를 둔다.

## 하위 패키지 역할

- `federation/rounds/`
  - FL round open/update/finalize, aggregation, runtime wiring
## newcomer용 읽기 순서

### 1. round lifecycle부터 보고 싶을 때

1. `federation/rounds/README.md`
2. `federation/rounds/round_lifecycle_service.py`
3. `federation/rounds/round_manager_service.py`

## 경계 원칙

- federation round orchestration은 `federation/rounds/`가 소유한다.
- `federation/assets/` 같은 넓은 catch-all service package를 새 source로 되살리지
  않는다. artifact 종류가 필요하면 resource/capability 이름으로 좁힌 package를 둔다.
- 공용 계약은 `shared/src/contracts/`를 기준으로 읽는다.
- 새 payload adapter kind나 FL SSL method 때문에
  `federation/rounds/payload_adapters/`에 payload-adapter/method-specific 파일을
  추가하지 않는다. 이 폴더는 shared payload registry와 methods-owned aggregation
  backend를 조합하는 generic runtime seam만 둔다.
- `RoundManagerService`는 기본 payload adapter를 직접 고르지 않는다. no-config live
  server fallback은 `federation/rounds/runtime/config.py`의 named legacy profile이
  소유하고, service는 caller가 조립한 `payload_adapter`를 받는다.
