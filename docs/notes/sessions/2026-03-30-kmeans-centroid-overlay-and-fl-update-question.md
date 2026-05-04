# 2026-03-30 k-means centroid overlay와 FL update 질문 요약

이 파일은 archive-only 세션 요약이다. 현재 source of truth는 active docs와 코드
가까운 README/contract다.

## 논의 요지

- k-means prototype analysis에서 선택된 centroid를 같은 projection 좌표계에 overlay로 남기도록 실험 산출물을 확장했다.
- 당시 FL 경로는 full encoder를 직접 업데이트하는 구조가 아니라, 공유 adapter state를 집계하고 새 manifest/prototype artifact를 발행하는 구조로 설명했다.
- multi-prototype/k-means는 분석 실험에는 유효하지만 active FL runtime 기본 경로는 single centroid/runtime-compatible artifact 중심이었다.
- 현재 경계 기준으로 prototype builder core는 `methods/prototype/building/`, prototype scoring core는 `methods/prototype/scoring/`이 소유한다.
- FL shard/aggregation/method 계산 core는 `methods/federated/*`, `methods/federated_ssl/*`로 분리한다.

## 현재 반영 위치

- prototype building core: `methods/prototype/building/*`
- prototype scoring/evidence core: `methods/prototype/scoring/*`, `methods/prototype/evidence/*`
- FL aggregation core: `methods/federated/aggregation/fedavg/*`
- FL runtime: `agent/src/services/federation/*`, `main_server/src/services/federation/*`
- simulation harness: `scripts/experiments/federated_simulation/*`

## 남은 판단

- multi-prototype runtime을 v1 active path로 열지, 분석 artifact로만 둘지.
- LoRA/classifier-head family를 FL runtime translation에서 어떤 payload로 표현할지.
