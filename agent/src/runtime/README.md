# Agent Runtime

`runtime/`은 agent 프로세스가 시작될 때 필요한 repository, local runtime service,
provider를 조립한다.

역할:

- `composition.py`: 기본 Adapter와 service graph를 만든다.
- `state.py`: `app.state`에 설치할 runtime object 목록과 설치 규칙을 소유한다.
- `env.py`: agent API shell이 읽는 runtime 환경값을 정규화한다.

`runtime/`은 FastAPI route handler, inference/training/wellbeing 알고리즘 의미,
repository 내부 SQL을 소유하지 않는다. 이 Module은 실행 조립의 seam일 뿐이다.
