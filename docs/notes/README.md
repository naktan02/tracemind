# Notes Directory

이 디렉터리는 구현 중 나온 대화 요약, 아키텍처 결정, incident 기록을 보관한다.
여기 문서에는 현재 비활성 또는 대체된 구조 논의가 남아 있을 수 있으므로,
현재 source of truth로 읽지 말고 참고/아카이브 용도로만 사용한다.

문제를 바로 해결하려고 할 때 이 디렉터리를 먼저 뒤지지 않는다.

- 현재 행동 규칙: root `AGENTS.md`
- 반복 실행 절차: 관련 `README.md`
- 활성 구현 계획: `docs/project_execution_plan.md`
- notes는 왜 그런 규칙이 생겼는지, 과거 대화에서 무엇을 결정했는지 찾는 용도다.

## 하위 디렉터리

- `incidents/`
  - 실수, 장애, 환경 오판, 재발 방지 guardrail 기록
- `decisions/`
  - 구조/계약/실행 흐름 관련 결정 기록
  - `배경 / 결정 내용 / 이유 / 보류 사항 / 다음 액션`처럼 구조화된 문서만 둔다.
- `sessions/`
  - 대화 요약, 질의응답, 세션 로그 성격의 기록
  - 기본적으로 300-500 words 요약만 둔다.
  - `User/Assistant` 대화 전문 transcript는 repo 안에 새로 추가하지 않는다.
  - 기존 긴 transcript는 historical archive로만 취급하고 통째로 읽지 않는다.

## 권장 규칙

파일명은 날짜와 주제를 함께 쓴다.

- `incidents/2026-04-12-gpu-runtime-preflight-guardrail.md`
- `decisions/2026-04-02-federated-simulation-orchestration-split.md`
- `sessions/2026-03-23-session-transcript.md`

## 문서 종류

1. 대화 요약
   - Codex와 논의한 핵심 결론만 정리
2. 결정 기록
   - 왜 이 구조를 선택했는지, 대안은 무엇이었는지 기록
   - 대화 전문이 아니라 정리된 결론 문서일 때만 `decisions/`에 둔다.
3. incident 기록
   - 무엇이 있었고, 왜 잘못 판단했는지, 어떤 guardrail로 승격했는지 기록
4. 임시 실험 메모
   - 추후 폐기될 수 있는 아이디어나 TODO 기록

## 권장 포맷

각 문서는 아래 항목을 짧게 포함한다.

1. 배경
2. 결정 내용
3. 이유
4. 보류 사항
5. 다음 액션

긴 원문 대화가 필요한 경우에도 이 디렉터리에는 현재 작업에 필요한 결론만 요약한다.
현재 규칙으로 쓰일 내용은 `docs/notes/**`에 머물지 말고 active docs나 코드 가까운
문서로 승격한다.

`plan.md`는 연구/비전 문서로 유지하고,
`docs/project_execution_plan.md`는 구현 계획 문서로 유지하며,
이 디렉터리는 작업 과정에서 나오는 메모만 저장하는 용도로 분리한다.
