# Conf AGENTS

## 역할

`conf/`는 Hydra 실행 조합과 파라미터의 source of truth다. 계산 의미, runtime
구현, artifact format은 소유하지 않는다.

## 먼저 읽을 것

1. `conf/README.md`
2. 변경 대상 entrypoint YAML
3. 관련 `strategy_axes/**`, `execution_context/**`, `run_controls/**` leaf
4. 해당 축의 owner 문서나 code-adjacent README

## 변경 규칙

- `entrypoints/`는 실행 시작점 조합만 소유한다.
- `strategy_axes/`는 실제로 교체 가능한 method, objective, trainable surface,
  update family, topology, runtime capability 선택만 소유한다.
- `execution_context/`는 dataset, embedding adapter, runtime environment 같은 실행
  재료를 소유한다.
- `run_controls/`는 budget, sweep, long-run guard, output root 같은 실행 조건을
  소유한다.
- YAML leaf는 callable path나 owner metadata를 선언할 수 있지만, Python 계산 의미나
  기본값 해석을 복제하지 않는다.
- teacher source는 독립 `teacher_provider` strategy axis로 열지 않는다. 중앙 SSL은
  explicit workflow entrypoint 내부 설정으로, FL SSL은 method descriptor나 runtime
  capability 요구로 표현한다.
- 새 config group은 실제 변화 축이 있을 때만 만든다. entrypoint 하나에서만 쓰는
  값 묶음은 entrypoint YAML에 남긴다.

## 테스트 규칙

- namespace 이동, default 변경, 새 leaf 추가는 Hydra compose test를 같이 갱신한다.
- config가 새 callable이나 owner path를 선언하면 architecture guard나 smoke test로
  import 경계를 확인한다.
