# Methods AGENTS

## 역할

`methods/`는 교체 가능한 알고리즘, method descriptor, objective, hook, scorer,
aggregation 계산 core를 소유한다.

## 먼저 읽을 것

1. `methods/README.md`
2. `docs/architecture/target-method-runtime-structure.md`
3. 새 전략 축이면 `docs/strategy_surface_map.md`
4. 관련 `conf/strategy_axes/**` leaf
5. 변경 대상 family의 code-adjacent README

## 변경 규칙

- 새 SSL/FL 방법론, pseudo-label 선택, masking, teacher hook, local objective,
  server/round policy 의미는 기본적으로 `methods/`에 둔다.
- `scripts`, `agent`, `main_server`가 method 이름이나 teacher 구현 이름으로
  분기해야 하면 먼저 `methods`의 descriptor, hook, execution plan 경계가 얕은지
  점검한다.
- teacher source는 SSL objective나 input mode의 hook이다. 공통 계약은
  `methods/ssl/hooks/teacher.py`가 소유하고, 재사용 가능한 source 의미나 prediction
  core는 `methods/ssl` 또는 해당 method/family owner 아래에 둔다.
- `fixed_classifier_seed` 같은 특정 artifact 생성 stage를 method 축으로 되살리지
  않는다. 필요하면 bootstrap source kind, artifact kind, checkpoint reference,
  `initial_checkpoint` provenance로 표현한다.
- artifact 저장소, Hydra entrypoint, report writer, HTTP transport, raw text/private
  state 접근은 `methods`로 끌어오지 않는다.
- concrete 구현이 한 method에만 필요한 경우에는 공통 hook package로 먼저 올리지
  말고 method-local module에 둔다. 두 개 이상에서 안정적으로 공유될 때만 승격한다.

## 테스트 규칙

- pure calculation과 hook contract 검증은 root `tests/unit`에 둔다.
- producer/consumer 경계나 execution plan 조합 검증은 root `tests/integration` 또는
  architecture guard에 둔다.
