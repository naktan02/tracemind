# Experiment Config

`conf/`는 최종적으로 TraceMind 실험 조합과 method 파라미터의 루트 Hydra config
공간이 될 위치다.

## 책임

- 사람이 직접 실행할 experiment recipe
- method 기본 파라미터
- dataset, model, training, runtime, evaluation config
- 실험 조합의 source of truth

## 진행 상태

이 디렉터리는 1단계 scaffold다. 현재 활성 Hydra config는 아직 `scripts/conf/`에
남아 있다. 2단계에서 config group을 이동하면서 entrypoint의 `config_path`,
Hydra defaults, config compose test를 함께 갱신한다.

## 원칙

YAML은 실행 조합표와 파라미터를 담는다. Python 구현, DB 접근, FastAPI route,
복잡한 계산 로직은 이 계층에 두지 않는다.
