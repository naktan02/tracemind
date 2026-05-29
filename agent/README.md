# Agent

전처리, 번역, 임베딩, shared scoring state/prototype evidence 계산,
개인화 상태 기반 해석, 로컬 학습, 서버 라운드 참여를 담당하는
로컬 런타임이다. 현재 운영 판정은 agent-local 해석 계층이 맡고,
shared state/prototype scoring은 evidence producer와 comparison 경로로 유지한다.

코드 읽기 시작점:

- `agent/src/services/README.md`
- 로컬 추론 rail: `agent/src/services/inference/`
- 프로토타입 runtime: `agent/src/services/assets/prototypes/`
- 로컬 학습 rail: `agent/src/services/training/`
- 서버 참여 orchestration: `agent/src/services/federation/rounds/`
- language helper: `agent/src/services/language/`

모델 교체 시작점:

- 실험/스크립트 embedding preset: `conf/execution_context/embedding_adapter/*.yaml`
- agent embedding adapter factory: `agent/src/infrastructure/model_adapters/embedding/factory.py`
- backtranslation runtime: `agent/src/services/language/backtranslation_service.py`
- 실제 추론 파이프라인 조립: `agent/src/services/inference/pipeline_service.py`
