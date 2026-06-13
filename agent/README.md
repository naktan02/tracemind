# Agent

전처리, 번역, 임베딩, scorer-family별 local analysis 계산,
개인화 상태 기반 해석, 로컬 학습, 서버 라운드 참여를 담당하는
로컬 런타임이다. 현재 운영 판정은 agent-local 해석 계층이 맡고,
classifier/prototype scoring은 scorer backend와 local asset provider로 분리한다.

코드 읽기 시작점:

- `agent/src/contracts/`
- `agent/src/services/README.md`
- `agent/src/features/README.md` — feature module 전환 목표 경계와 migration 순서
- `agent/src/runtime/README.md`
- `agent/REFACTOR_ROADMAP.md` — agent 폴더 구조 리팩터링 phase gate
- 로컬 추론 rail: `agent/src/features/inference/`
- scorer asset runtime: `agent/src/services/assets/`
- 로컬 학습 rail: `agent/src/features/training_runtime/`
- 서버 참여 orchestration: `agent/src/services/federation/rounds/`
- language helper: `agent/src/services/language/`
- 가족/아이 wellbeing feature: `agent/src/features/wellbeing/`

모델 교체 시작점:

- 실험/스크립트 embedding preset: `conf/execution_context/embedding_adapter/*.yaml`
- agent embedding adapter factory: `agent/src/infrastructure/model_adapters/embedding/factory.py`
- backtranslation runtime: `agent/src/services/language/backtranslation_service.py`
- 실제 추론 파이프라인 조립: `agent/src/features/inference/pipeline_service.py`

운영 추론 pipeline의 scoring backend는 `TRACEMIND_AGENT_SCORING_BACKEND`로
명시한다. `classifier_head_logits` 같은 scorer를 묵시 기본값으로 붙이지 않는다.
