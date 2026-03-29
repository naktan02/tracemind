  $tracemind-project

  현재 저장소는 `/home/jmgjmg102/tracemind_server`이고, 먼저 아래 문서를 읽고 현재 상태를 파악해줘.

  1. `plan.md`
  2. `docs/project_execution_plan.md`

  현재 합의된 구조는 `main-server / agent / shared` 모노레포 v2.1이다.
  `agent/chrome-extension`은 브라우저 입력 수집 어댑터이고, 핵심 분석 로직은 `agent/src/services`에 둔다.
  개발 순서는 `계약 우선 -> 모델/데이터셋 정리 -> agent 분석 파이프라인 -> main-server 집계 -> chrome extension 연결
  `이다.

  이번 턴 목표:
  1. `docs/contracts/window_summary_v1.md` 작성
  2. `docs/contracts/norm_pack_v1.md` 작성
  3. 필요하면 `docs/contracts/sync_contracts_v1.md`까지 초안 작성
  4. 문서 기준에 맞춰 `shared/src/domain/entities/window_summary.py`와 `shared/src/domain/entities/norm_pack.py` 초
  안도 함께 설계

  주의사항:
  - 항상 한국어로 답변
  - `docs/notes/`는 읽지 말고, 현재 대화는 전문으로 `docs/notes/`에 기록
  - 임시방편보다 미래지향적이고 확장 가능한 구조 우선
  - parameter FL, DP, secure aggregation, HE 확장 가능성을 해치지 않는 방향으로 설계
  - API나 DB보다 먼저 계약과 도메인 객체를 고정