# Agent AGENTS

## 역할

`agent/`는 로컬 추론, 로컬 학습, 서버 참여 runtime을 소유한다.

## 먼저 읽을 것

1. `agent/README.md`
2. `agent/src/services/README.md`
3. 작업 축에 맞는 feature/service 경로
   - 추론: `agent/src/features/inference/`
   - 학습: `agent/src/features/training_runtime/`
   - FL 참여: `agent/src/features/federation/`
4. 관련 shared contract와 domain entity

## 변경 규칙

- 원문 텍스트, 개인 상태, 개인 threshold, 개인 prototype 같은 로컬 상태는
  `agent` 소유 경계에 남긴다.
- 서버 orchestration 책임을 `agent`로 끌어오지 않는다.
- payload shape를 바꾸기 전에 `shared` contract를 먼저 고친다.
- 로컬 heuristic, scorer, example generation, training backend는 서로 다른
  변화 축으로 보고 섞지 않는다.
- 새 method/algorithm을 추가하기 위해 `agent`에 method-specific backend 파일을
  추가하지 않는다. `agent`의 backend/runtime adapter는 raw text 접근, private
  state, artifact materialization, payload upload 같은 로컬 실행 capability만
  소유한다.
- local objective, pseudo-label algorithm, SSL hook, adaptation 계산 core는
  `methods/`에 둔다. `agent`는 선택된 core를 실행 가능한 local task로 변환하는
  port/adapter 역할만 맡는다.
- 관측 가능성이 부족하면 파라미터 튜닝보다 dump/trace/summary를 먼저 만든다.

## 테스트 규칙

- agent 내부 단위 검증은 `agent/tests`에 둔다.
- 서버와의 상호작용 검증은 root `tests/` integration 시나리오로 올린다.
