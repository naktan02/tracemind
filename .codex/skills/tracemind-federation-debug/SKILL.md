---
name: tracemind-federation-debug
description: Use when debugging FL rounds, model revision mismatches, artifact rebuild drift, stale processes, or agent-server runtime inconsistencies.
---

# TraceMind Federation Debug

이 스킬은 federation runtime의 실패 지점을 찾고 재현 경로를 고정할 때 쓴다.

## 언제 쓸지

- `model_revision` / `base_model_revision` mismatch
- artifact rebuild drift
- multi-agent round failure
- stale process 때문에 테스트나 실행이 꼬일 때
- local update와 central aggregation이 다른 가정을 가질 때

## 작업 순서

1. 먼저 stale process를 확인한다.
   - `ps aux`
2. 실패 지점을 레일 기준으로 분리한다.
   - local inference
   - local training/update generation
   - transport/envelope
   - central aggregation/publication
3. 관련 contract와 runtime owner를 확인한다.
   - `shared`
   - `agent`
   - `main_server`
4. 튜닝 전에 dump, trace, summary를 먼저 만든다.
5. 최소 재현 경로를 테스트나 스크립트로 남긴다.

## 체크리스트

- 실패가 계약 mismatch인지 실행 순서 문제인지 분리했는가
- revision과 artifact pair가 일관적인가
- raw text나 private state가 서버로 새고 있지 않은가
- 재현 가능한 검증 흔적이 남았는가
