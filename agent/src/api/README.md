# Agent API

FastAPI shell과 route module을 소유한다. route module은 request/response payload 변환,
HTTP status 매핑, feature service 호출 흐름만 보여준다.

- `main.py`: app 생성, CORS, runtime state 설치, router registration.
- `dependencies.py`: FastAPI dependency glue와 `app.state` runtime lookup/cache helper.
  business rule이나 feature storage 의미를 소유하지 않는다.
- `*_*.py` route modules: endpoint payload와 feature service 호출만 소유한다.

Feature runtime 객체의 생성과 기본 조립은 `agent/src/runtime/`가 맡는다.
