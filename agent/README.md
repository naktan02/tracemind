# Agent

전처리, 번역, 임베딩, 프로토타입 점수 계산,
개인화 상태 기반 해석, 로컬 학습, 서버 라운드 참여를 담당하는
로컬 런타임이다.

코드 읽기 시작점:

- `agent/src/services/README.md`
- 로컬 추론 rail: `agent/src/services/inference/`
- 프로토타입 runtime: `agent/src/services/prototype/`
- 로컬 학습 rail: `agent/src/services/training/`
- 서버 참여 orchestration: `agent/src/services/federation/`

모델 교체 시작점:

- embedding model: `agent/conf/embedding/*.yaml`
- translation model: `agent/conf/translation/*.yaml`
- 실제 추론 파이프라인 조립: `agent/src/services/inference/pipeline_service.py`
