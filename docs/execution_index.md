# TraceMind Execution Index

## 목적

이 문서는 TraceMind 작업의 짧은 진입점이다.
현재 활성 경로는 `personalized local inference + federated shared model improvement`다.

## 읽는 순서

1. `plan.md` (repo root)
   - 왜 이 구조를 택했는지
2. `docs/project_execution_plan.md`
   - 지금 무엇을 구현하고 무엇을 미루는지, Phase별 현황
3. `shared/src/contracts/README.md`
   - 현재 payload 계약 해석
4. `docs/fl_runtime_implementation_checklist.md`
   - 실제 구현 순서와 완료 기준
5. `docs/contracts/algorithm_extension_guide.md`
   - 알고리즘/전략 교체 시 어느 파일을 보고 어떤 Protocol을 구현할지
6. 필요한 경우만 `docs/staged_execution_roadmap.md`
   - Phase 이름과 검증 포인트 빠른 확인

## 문서 역할

| 파일 | 역할 |
|---|---|
| `plan.md` | 연구 비전, 핵심 가설, global/local 분리 원칙 |
| `docs/project_execution_plan.md` | 활성 아키텍처, 현재 Phase, 다음 액션, 검증 기준 |
| `docs/fl_runtime_implementation_checklist.md` | 실제 구현 작업 순서, 파일 후보, 완료 기준 |
| `docs/staged_execution_roadmap.md` | 중복 설명 없이 Phase map만 제공 |
| `shared/src/contracts/README.md` | 코드 가까운 contract 해설 |
| `docs/contracts/` | 계약 설계 배경, 알고리즘 확장 가이드 |
| `docs/contracts/algorithm_extension_guide.md` | 교체 가능한 모든 전략 지점과 교체 절차 |
| `docs/contracts/shared_adapter_contracts_v1.md` | adapter payload 구조와 수학적 의미 |
| `docs/contracts/training_update_envelope_v1.md` | envelope 필드 설계 이유 |

## 작업 시작 체크리스트

1. 이번 요청이 `global/shared`인지 `local/private`인지 먼저 구분한다.
2. 바뀔 축이 무엇인지 적는다.
   - 예: adapter family, training backend, aggregation backend, privacy layer, scoring policy
3. `docs/contracts/algorithm_extension_guide.md`에서 해당 전략 지점을 찾는다.
4. 운영 후보 로직이면 `scripts`가 아니라 `shared/agent/main_server` 소유 경계에 먼저 둔다.
5. 사용자 판단이 필요한 항목인지 확인한다.

## 사용자 확인이 필요한 변경

1. full encoder FL 활성화
2. pseudo-label을 정식 학습 신호로 승격
3. 서버로 보내는 update 메타데이터 확대
4. private adapter/head를 언제 열지
5. secure aggregation, DP, HE 도입 시점
