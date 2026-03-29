# TraceMind Execution Index

## 목적

이 문서는 TraceMind 문서의 진입점이다.
2026-03-29 전환 결정 이후, 이 프로젝트의 활성 경로는
`WindowSummary/NormPack` 기반 normative analytics가 아니라
`personalized local adaptation + FL runtime` 기준으로 해석한다.

---

## 읽는 순서

1. [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
   - 연구 비전, 문제 정의, 전환 결정, 새 핵심 메시지

2. [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
   - 구현 순서, 활성 아키텍처 결정, 유지/폐기 범위

3. [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
   - 단계별 상세 작업, 검증 기준, 사용자 확인 게이트

4. 활성 계약 문서
   - [`docs/contracts/prototype_pack_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_pack_v1.md)
   - [`docs/contracts/prototype_build_state_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_build_state_v1.md)
   - [`docs/contracts/model_manifest_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/model_manifest_v1.md)
   - [`docs/contracts/training_task_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/training_task_v1.md)
   - [`docs/contracts/training_update_envelope_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/training_update_envelope_v1.md)
   - [`docs/contracts/decision_feedback_signal_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/decision_feedback_signal_v1.md)
   - [`docs/contracts/personalization_state_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/personalization_state_v1.md)

5. 보관 문서
   - [`docs/contracts/window_summary_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/window_summary_v1.md)
   - 위 문서는 현재 활성 계획의 contract source of truth가 아니라, 이전 normative 경로의 보관 문서다.

---

## 문서 역할

### 1. 비전 문서

- [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
- 연구 목적, 구조 전환 이유, 장기 메시지

### 2. 실행 계획 문서

- [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
- 실제 구현 순서와 아키텍처 결정

### 3. 실행 로드맵 문서

- [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
- 단계별 작업과 검증, 사용자 확인 지점

### 4. 활성 계약 문서

- [`docs/contracts/prototype_pack_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_pack_v1.md)
- [`docs/contracts/prototype_build_state_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_build_state_v1.md)
- [`docs/contracts/model_manifest_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/model_manifest_v1.md)
- [`docs/contracts/training_task_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/training_task_v1.md)
- [`docs/contracts/training_update_envelope_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/training_update_envelope_v1.md)
- [`docs/contracts/decision_feedback_signal_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/decision_feedback_signal_v1.md)
- [`docs/contracts/personalization_state_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/personalization_state_v1.md)

### 5. 보관 계약 문서

- [`docs/contracts/window_summary_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/window_summary_v1.md)
- 이전 normative analytics 경로 참고용

### 6. 대화 기록

- [`docs/notes/README.md`](/home/jmgjmg102/tracemind_server/docs/notes/README.md)
- 세션 기록과 전환 결정 이력

---

## 작업 시작 체크리스트

작업 전 최소 확인 순서:

1. `plan.md`에서 현재 연구 방향 확인
2. `docs/project_execution_plan.md`에서 현재 phase와 폐기 범위 확인
3. `docs/staged_execution_roadmap.md`에서 지금 작업의 산출물과 검증 기준 확인
4. 관련 활성 contract 문서 확인
5. FL 범위, feedback 신호, privacy 보호 수준 중 사용자 판단이 필요한 항목인지 확인

---

## 사용자 확인이 필요한 변경

아래 변경은 임의로 확정하지 않고 사용자 확인 후 진행한다.

1. full encoder FL 활성화 여부
2. pseudo-label을 정식 학습 신호로 채택할지 여부
3. self-report / delayed outcome / support action 중 어떤 feedback을 수집할지
4. 서버로 보내는 update 메타데이터 범위
5. clipping, secure aggregation, DP 도입 시점
6. `WindowSummary/NormPack`을 완전히 코드에서 제거할 시점
