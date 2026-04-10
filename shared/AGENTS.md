# Shared AGENTS

## 역할

`shared/`는 agent, main_server, scripts가 함께 읽는 canonical contract와
domain entity의 source of truth다.

## 먼저 읽을 것

1. `shared/src/contracts/*.py`, `shared/src/domain/entities/*`
2. `shared/src/contracts/README.md`
3. `docs/contracts/algorithm_extension_guide.md`
4. 필요한 경우만 `docs/contracts/*`

## 변경 규칙

- 필드 의미와 해석 규칙은 계약 파일 가까이에 직접 드러나야 한다.
- 같은 의미의 payload가 경로마다 다른 shape로 흐르지 않게 canonical
  representation을 유지한다.
- legacy format이나 임시 변환이 필요하면 compatibility 계층으로 명시적으로
  격리한다.
- `shared` 변경은 계약 파일 수정으로 끝나지 않는다. producer, consumer,
  테스트, 문서까지 같은 턴에서 닫는다.
- 실험 편의를 위한 로직을 `shared`에 넣지 않는다. 안정적으로 공유되는 규칙만
  둔다.

## 테스트 규칙

- `shared` 전용 unit/contract 검증은 `shared/tests`에 둔다.
- 경계를 넘는 검증은 repo 루트 `tests/`에 둔다.
