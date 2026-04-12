# 2026-04-02 Federated Simulation Orchestration Split

## 배경

`scripts/experiments/federated_simulation/simulation.py`가
한 파일에서 아래 책임을 모두 갖고 있었다.

1. JSONL 로드와 시간 파싱
2. training example build와 validation 계산
3. prototype rebuild runtime wiring
4. training task/validation config 변환
5. round orchestration

이 상태에서는 simulation 흐름을 바꾸지 않아도
평가 규칙이나 runtime wiring 수정이 같은 큰 파일로 모이게 된다.

## 결정 내용

아래 helper 모듈로 책임을 분리했다.

1. `io_utils.py`
   - JSONL 로드
   - `created_at` 파싱
2. `evaluation.py`
   - training example build
   - validation score/acceptance 계산
3. `runtime.py`
   - prototype rebuild runtime wiring
   - active state load
   - simulation adapter seam
4. `task_config.py`
   - legacy leaf 인자 -> canonical training task config 변환
   - round open request 조립

`simulation.py`는 bootstrap, round loop, publication 전이 같은 orchestration만 남긴다.

## 이유

1. 실험층이어도 변화 축 분리가 필요하다.
2. 평가 규칙과 runtime wiring은 서로 다른 이유로 바뀐다.
3. orchestration 파일은 흐름을 읽기 쉬워야 한다.

## 보류 사항

1. client round 실행 루프 자체는 아직 `simulation.py`에 남아 있다.
2. 필요하면 이후 `round_execution.py` 같은 별도 helper로 더 분리할 수 있다.

## 다음 액션

1. simulation client loop를 더 자를 필요가 있는지 실제 변경 패턴을 보고 판단한다.
2. 필요하면 `scripts/experiments/prototype_strategy` 쪽도 비슷한 방식으로
   orchestration과 evaluation helper를 더 분리한다.
