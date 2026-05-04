# Experiment Workspace

`experiment_workspace/`는 개발자 실험 UI/API가 고른 workspace manifest를 실제
Hydra/script 실행 표면으로 번역하는 서버 계층이다. 알고리즘 의미나 config 기본값을
소유하지 않고, `conf/`, `scripts/`, `methods/`, `shared`의 source of truth를 읽어
catalog와 preview를 만든다.

## 하위 패키지

- `catalog/`: 현재 repo에서 사용 가능한 entrypoint, preset, strategy surface를 읽어 API payload로 노출한다.
- `compiler/`: workspace manifest selection을 Hydra selector/override와 실행 command preview로 번역한다.
- `run_execution/`: compile된 실험을 local process로 실행하고 stdout/stderr, run summary를 조회한다.

`payloads.py`는 experiment workspace API payload를, `workspace_service.py`는 saved
workspace 저장/조회 흐름을 소유한다.
