# TraceMind Execution Index

## 목적

이 문서는 TraceMind 작업의 짧은 진입점이다.
현재 활성 경로는 `normative analytics`가 아니라
`personalized local inference + federated shared model improvement`다.

## 읽는 순서

1. [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
   - 왜 이 구조를 택했는지
2. [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
   - 지금 무엇을 구현하고 무엇을 미루는지
3. [`shared/src/contracts/README.md`](/home/jmgjmg102/tracemind_server/shared/src/contracts/README.md)
   - 현재 payload 계약 해석
4. 필요한 경우만 [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
   - phase 이름과 검증 포인트 빠른 확인

## 문서 역할

- [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
  - 연구 비전, 핵심 가설, global/local 분리 원칙
- [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
  - 활성 아키텍처, 현재 phase, 다음 액션, 검증 기준
- [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
  - 중복 설명 없이 phase map만 제공
- [`shared/src/contracts/README.md`](/home/jmgjmg102/tracemind_server/shared/src/contracts/README.md)
  - 코드 가까운 contract 해설
- [`docs/contracts/`](/home/jmgjmg102/tracemind_server/docs/contracts)
  - 배경 설명과 설계 메모

## 작업 시작 체크리스트

1. 이번 요청이 `global/shared`인지 `local/private`인지 먼저 구분한다.
2. 현재 활성 경로가 `NormPack`이 아니라 `shared adapter + personalization`인지 다시 확인한다.
3. 바뀔 축이 무엇인지 적는다.
   - 예: adapter family, training backend, aggregation backend, privacy layer
4. 관련 contract를 코드 가까운 문서에서 먼저 본다.
5. 사용자 판단이 필요한 항목인지 확인한다.

## 사용자 확인이 필요한 변경

1. full encoder FL 활성화
2. pseudo-label을 정식 학습 신호로 승격
3. 서버로 보내는 update 메타데이터 확대
4. private adapter/head를 언제 열지
5. secure aggregation, DP, HE 도입 시점
