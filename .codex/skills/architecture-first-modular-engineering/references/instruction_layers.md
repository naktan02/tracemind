# Instruction Layer Patterns

여러 도구의 공식 문서를 보면 공통 패턴은 비슷하다.

## 공통 패턴

1. repo-wide 공통 지침
   - 저장소 전체에 적용되는 짧은 지침 1장
2. path-specific 지침
   - 특정 디렉터리/파일군에만 적용되는 세부 규칙
3. user-global 지침
   - 개인 취향이나 도구 습관은 repo 밖 전역 레이어

## 도구별 대응

- `AGENTS.md`
  - 저장소 공통 agent 지침
- `CLAUDE.md`
  - Claude Code 프로젝트 메모리
- `.github/copilot-instructions.md`
  - GitHub Copilot repo-wide 지침
- `.github/instructions/*.instructions.md`
  - GitHub Copilot path-specific 지침
- `.cursor/rules/*`
  - Cursor project rules
- `SKILL.md`
  - Codex 같은 에이전트에서 재사용 가능한 작업 스타일 패키지

## 이 스킬의 권장 적용 방식

1. 재사용 가능한 개인/팀 스타일은 `SKILL.md`로 만든다.
2. 현재 저장소에 적용할 공통 규칙은 `AGENTS.md` 또는 repo-wide instruction 파일에 둔다.
3. 계약, 도메인, 서비스, 문서처럼 성격이 다른 경로는 path-specific instruction으로 분리한다.
4. 개인 실험 설정이나 로컬 URL 같은 값은 repo에 넣지 않는다.
