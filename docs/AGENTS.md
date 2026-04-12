# Docs AGENTS

## 역할

`docs/`는 코드 밖 설명 계층이다. 하지만 source of truth를 대체하지 않는다.

## 문서 우선순위

1. `shared/src/contracts/*.py`, `shared/src/domain/entities/*`
2. `shared/src/contracts/README.md`
3. `docs/contracts/*`, `docs/*`
4. `docs/notes/**`

## 변경 규칙

- 코드 가까이에 둘 수 있는 필드 의미와 계약 의미는 문서로 우회하지 않는다.
- `docs/ai_context_manifest.yaml`은 AI가 읽을 진입점과 문서 권한을 정리한
  machine-friendly map이다.
- `docs/execution_index.md`는 짧은 human/agent 진입점으로 유지한다.
- `docs/project_execution_plan.md`는 현재 활성 구현 계획만 유지한다.
- `docs/ai_harness_operating_model.md`와 `docs/ai_harness_eval_cases.yaml`은
  maintainer 전용 보조 문서로만 둔다.
- `docs/contracts/*`는 설계 배경과 확장 절차를 설명한다.
- `docs/contracts/central_lora_classifier_trainer_contract.md`는 논문 트랙의
  canonical LoRA scaffold와 산출물 경계를 설명한다.
- `docs/notes/**`는 참고/아카이브 용도이며 source of truth로 취급하지 않는다.
- 구조 변경 시 관련 README, execution index, manifest를 같은 턴에서 함께
  맞춘다.
