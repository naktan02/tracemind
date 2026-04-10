---
name: tracemind-doc-sync
description: Use when architecture, contracts, ownership boundaries, or execution workflow changed and the active docs must be synchronized without promoting notes into source of truth.
---

# TraceMind Doc Sync

이 스킬은 구조 변경 뒤 active docs의 drift를 정리할 때 쓴다.

## 언제 쓸지

- architecture refactor 뒤 execution index를 갱신할 때
- contract/plan/README의 설명이 어긋났을 때
- 새 instruction layer나 skill을 추가했을 때

## 작업 순서

1. 아래 순서로 읽는다.
   - root `AGENTS.md`
   - `docs/AGENTS.md`
   - `docs/ai_context_manifest.yaml`
   - `docs/execution_index.md`
   - `docs/project_execution_plan.md`
2. active doc와 archive를 구분한다.
3. 같은 배경 설명이 여러 문서에 중복되면 역할별로 정리한다.
4. source of truth를 문서가 대신하지 않도록 유지한다.
5. 관련 README와 manifest를 같이 맞춘다.

## 체크리스트

- 실행 순서와 읽기 순서가 최신 구조와 일치하는가
- notes가 active guidance로 승격되지 않았는가
- 문서 역할이 짧고 분명한가
- 계약 의미는 코드 가까이에 남아 있는가
