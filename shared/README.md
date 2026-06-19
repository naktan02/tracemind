# Shared

`shared/`는 agent, main_server, scripts가 함께 읽는 canonical contract와 domain entity의
source of truth다.

여기에는 여러 runtime이 같은 의미로 해석해야 하는 payload shape, domain value,
parse/serialize 규칙만 둔다. 방법론 구현, 실행 profile, runtime catalog, training
backend, server policy는 소유하지 않는다.

## What Belongs Here

- FL round, model manifest, training task/update envelope 같은 cross-runtime payload
- shared adapter state/update family의 parse/serialize contract
- 여러 계층이 함께 쓰는 domain entity와 value object
- payload compatibility와 canonical representation 규칙
- secure aggregation metadata처럼 agent/server가 함께 해석해야 하는 envelope metadata

## What Does Not Belong Here

- SSL/FSSL method 구현
- local objective, pseudo-label algorithm, scorer implementation
- Hydra 실행 기본 조합
- runtime registry/catalog 기본값
- FastAPI route, repository, artifact storage 구현
- family extension이나 agent-local UI payload

Agent-local captured text, child support, family access, wellbeing UI payload는
`agent/src/contracts/`가 소유한다. `shared`에는 main_server, agent, scripts가 함께
해석해야 하는 payload만 둔다.

## Read Path

처음 읽을 때는 아래 순서가 가장 짧다.

1. [src/contracts/README.md](src/contracts/README.md)
2. `shared/src/contracts/*.py`
3. `shared/src/contracts/adapter_contract_families/*.py`
4. `shared/src/domain/entities/*`
5. 필요한 경우만 [../docs/contracts](../docs/contracts)

## Layout

| Path | Responsibility |
|---|---|
| `src/contracts/` | cross-runtime payload contracts |
| `src/contracts/adapter_contract_families/` | shared adapter state/update family contracts |
| `src/domain/entities/` | shared domain entities |
| `src/domain/value_objects/` | shared value objects |
| `src/domain/services/` | side-effect-free shared domain helpers |
| `src/services/` | small shared services used by multiple runtime layers |

## Change Rule

`shared` 변경은 계약 파일 수정으로 끝나지 않는다. 같은 턴에서 producer, consumer,
tests, code-adjacent README를 함께 맞춘다.

새 FL SSL method나 local objective를 추가하기 위해 `shared`를 고치는 흐름은 대개
잘못된 경계다. state/update payload shape 자체가 바뀔 때만
`shared/src/contracts/adapter_contract_families/`를 확장한다.
