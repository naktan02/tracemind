# Docs AGENTS

## 역할

`docs/`는 코드 밖 설명 계층이다. 하지만 source of truth를 대체하지 않는다.

## 문서 계층

- `docs/architecture/`
  - 현재 런타임, 코드 경계, 활성 레일의 canonical 개요
- `docs/api/`
  - FastAPI endpoint 표면과 route owner 지도
- `docs/operations/`
  - 로컬 실행, GPU preflight, smoke/runbook
- `docs/quality/`
  - 테스트 층, 보호 범위, quality gate
- `docs/governance/`
  - 문서 class와 source-of-truth 운영 규칙
- `docs/contracts/`
  - contract 설계 배경과 전략 확장 절차
- `docs/notes/`
  - 세션, incident, decision archive

## 문서 우선순위

1. `shared/src/contracts/*.py`, `shared/src/domain/entities/*`
2. `shared/src/contracts/README.md`
3. `docs/contracts/*`, active `docs/*`

`docs/notes/**`는 archive-only다. 현재 규칙이나 작업 절차로 쓰려면 active
docs나 code-adjacent 문서로 요약 승격한다.

## 변경 규칙

- 코드 가까이에 둘 수 있는 필드 의미와 계약 의미는 문서로 우회하지 않는다.
- `docs/ai_context_manifest.yaml`은 AI가 읽을 진입점과 문서 권한을 정리한
  machine-friendly map이다.
- `docs/execution_index.md`는 짧은 human/agent 진입점으로 유지한다.
- `docs/project_execution_plan.md`는 현재 활성 구현 계획만 유지한다.
- `docs/architecture/system-overview.md`는 현재 코드 경계와 런타임 rail의
  canonical overview로 유지한다.
- `docs/api/api-surface.md`는 route 목록과 owner만 요약하고 payload field
  정본은 코드와 shared contract에 둔다.
- `docs/operations/local-runbook.md`는 실제 존재하는 dependency source와
  명령만 담는다.
- `docs/quality/test-strategy.md`는 테스트 층과 보호 범위를 설명한다.
- `docs/governance/document-governance.md`는 문서 class와 갱신 기준을 소유한다.
- `docs/ai_harness_operating_model.md`와 `docs/ai_harness_eval_cases.yaml`은
  maintainer 전용 보조 문서로만 둔다.
- `docs/contracts/*`는 설계 배경과 확장 절차를 설명한다.
- `docs/contracts/central_peft_text_encoder_trainer_contract.md`는 논문 트랙의
  canonical PEFT text encoder scaffold와 산출물 경계를 설명한다.
- `docs/notes/**`는 참고/아카이브 용도이며 source of truth로 취급하지 않는다.
- 새 `docs/notes/sessions/**`는 300-500 words 요약으로 제한하고 대화 전문
  transcript를 추가하지 않는다.
- active markdown은 index와 링크를 우선하고 반복 cookbook을 피한다. 계약 reference를
  제외하면 새 문서는 보통 250 lines 안쪽으로 유지한다.
- 긴 archive 문서는 더 길게 고치지 말고, 현재 결정만 active docs로 승격한 뒤
  archive 요약으로 줄인다.
- 구조 변경 시 관련 README, execution index, manifest를 같은 턴에서 함께
  맞춘다.
