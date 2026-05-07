# Shared

로컬 agent와 중앙 서버가 함께 사용하는 공용 contract, domain entity,
canonical payload 해석 규칙의 source of truth다.

`shared`는 방법론 구현, 실행 profile, runtime catalog, training backend,
server policy를 소유하지 않는다. adapter family 이름을 가진 파일은 state/update
payload shape와 parse/serialize 계약일 때만 둔다.

처음 읽을 때는 아래 순서를 권장한다.

1. `shared/src/contracts/README.md`
2. `shared/src/contracts/*.py`
3. `shared/src/domain/entities/*`
