# 2026-03-31 pyproject packaging and strategy discussion

## User

1. pyproject에서 where에서 main-server를 뺼거야 그리고 폴더 이름을 main_server로 바꾸면 src.으로 하던걸 main_server.으로 시작하게 할 수 있지? 이것만  이러니까 통일성이 없어서 이후에 헷갈릴 가능성이 높을 것 같아
2.  제미나이 의견인데 이거 어떻게 생각해?
2. 현재 pyproject.toml 구조에 대한 문제점 점검
올려주신 pyproject.toml에는 의존성 관리 및 uv 관련 설정이 훌륭하게 들어가 있습니다. 하지만, 앞선 대화에서 권장해 드린 "프로젝트 전체를 패키지로 설치하여 sys.path.insert 꼼수를 없애는 방식(pip install -e . 등)"을 적용하기에는 중요한 한 가지가 누락되어 있습니다. 바로 [build-system] 섹션입니다.

현재 문제점: 패키지 빌드 백엔드 부재
Python 환경(pip, uv 등)이 이 디렉토리를 소스코드가 아닌 "설치 가능한 패키지"로 인식하려면 소스를 어떻게 묶을지 알려주는 빌드 시스템(Build System) 설정이 필수입니다. 또한, 현재 agent.src, shared.src 처럼 루트 디렉토리에 여러 모듈이 흩어져 있는 모노레포 형태이므로, 이를 명시적으로 패키지로 인식시켜주어야 합니다.

해결 방법 (수정 제안)
파일 최상단이나 하단에 아래와 같이 [build-system]과 패키지 탐색 규칙을 추가해야 합니다. (아래는 setuptools를 사용하는 가장 보편적인 예시입니다.)

Ini, TOML
# --- pyproject.toml 파일에 추가해야 할 내용 ---

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
# 프로젝트 루트(.)에서 패키지를 찾도록 설정
where = ["."]
# 가져올 최상위 디렉토리들을 명시 (기존 sys.path에 의존하던 모듈들)
include = ["agent*", "shared*", "scripts*", "main-server*"]
이렇게 추가하고 나면 얻게 되는 효과:

터미널(가상환경 활성화 상태)에서 프로젝트 루트로 이동한 뒤 uv pip install -e . (또는 pip install -e .) 명령어를 실행할 수 있습니다.

설치 후에는 어느 폴더에서 파이썬 스크립트를 실행하더라도 from shared.src.contracts... 같은 임포트가 정상적으로 동작합니다.

따라서 run_federated_simulation.py 상단에 있던 지저분한 sys.path.insert(0, str(PROJECT_ROOT)) 코드를 완전히 지워버릴 수 있습니다.
3. 이건 그냥 궁금한건데 전략을 유연하게 받는 곳에서 문자열로 대부분 받아서 처리하게 되어있잖아 yaml때문에 그런데 지금은 스크립트 실험때문에 그런건데 완료되고 실서비스에서는 해당 곳에 의존성으로 집어넣나? 아니면 기본값을 서비스할것으로 해서 넣고 변경 가능하게? 또는 서버에서 내려준 고정 파일 또는 db에 값 넣어서 업데이트 없다면 그 값으로 넣어서 실행?

## Assistant

요청하신 세 가지는 `pyproject` 패키징 방향, Gemini 제안의 타당성, 그리고 전략 설정을 실서비스에서 어떻게 다룰지에 대한 구조 판단으로 보입니다. 먼저 저장소 규칙대로 관련 스킬과 현재 `pyproject.toml`, 패키지 구조, 계획 문서를 확인한 뒤 지금 구조에 맞는 판단으로 정리하겠습니다.

## Assistant

현재 `pyproject.toml`을 보니 이미 `[build-system]`은 들어가 있어서 Gemini가 지적한 핵심 문제는 현재 파일 기준으로는 맞지 않습니다. 추가로 실행 인덱스와 구현 계획 문서를 확인해서, 이 패키징 변경이 지금 구조 방향과도 맞는지 이어서 보겠습니다.

## Assistant

이제 실제 코드가 `src.`를 어떻게 쓰는지와 `main-server` 패키징이 얼마나 섞여 있는지 보고 판단하겠습니다. 이름만 바꾸는 수준인지, import 경로와 테스트/런처까지 같이 정리해야 하는지 확인이 필요합니다.

## Assistant

`src`가 실제 import path로 굳어 있는지, 아니면 디렉터리만 바꾸면 되는지 구분하려면 패키지 탐색 결과를 한 번 확인하는 게 정확합니다. 현재 setuptools가 어떤 이름의 패키지를 잡는지 바로 검증해 보겠습니다.

## Assistant

패키지 탐색 검증은 `python` 심볼이 없어서 실패했고, 현재 환경에서는 `python3`로 다시 보면 됩니다. 이건 구조 판단에 직접 필요한 확인이라 바로 재시도하겠습니다.

## Assistant

Gemini 제안 자체보다 더 중요한 건 현재 `setuptools`가 실제로 `main-server` 아래에서 `src`뿐 아니라 `tests`, `state`까지 패키지처럼 잡고 있다는 점입니다. 즉 지금은 “설치 가능하냐”보다 “어떤 이름으로 무엇을 설치하고 있느냐”가 더 큰 문제입니다.

## Assistant

전략 관련 질문은 지금 코드에서 문자열이 단순 실험용인지, 장기적으로도 wire format인지 구분해야 답이 정확합니다. 현재 어떤 지점이 문자열 discriminator를 쓰는지 몇 군데만 확인해서 서비스 단계 권장안을 묶어드리겠습니다.

## Assistant

전략 문자열은 현재 두 종류가 섞여 있습니다. 하나는 실험/실행 설정에서 클래스 wiring용 문자열이고, 다른 하나는 계약 payload에서 wire-format discriminator 역할의 문자열입니다. 이 둘은 실서비스에서 취급 방식이 달라야 해서 그 구분을 기준으로 정리하겠습니다.

## Assistant

답변 정리는 거의 끝났고, 저장소 작업 규칙에 따라 이번 대화 전문을 `docs/notes/`에 남기겠습니다. 코드 변경은 없고 기록 파일만 추가합니다.

## User

1.  uvicorn entrypoint은 뭘하는거야?
2. 실무적으로 갈 때에도 문제는 현재 구조 문제없는건가?  아니면 구조 수정해야하나?

## Assistant

`uvicorn`이 실제로 어디서 쓰이는지와 현재 구조 영향 범위를 짧게 확인한 뒤 답하겠습니다. 서비스 진입점은 실행 커맨드와 import 경로가 맞물려 있어서, 로컬 규칙 문서 말고 실제 저장소 사용처까지 보는 편이 정확합니다.

## Assistant

`uvicorn` entrypoint은 서버를 띄울 때 `어느 모듈에서 어떤 ASGI app 객체를 가져올지` 지정하는 문자열입니다. 형식은 `uvicorn <module_path>:<object_name>`이고, 지금 문서 기준 커맨드 `uvicorn src.api.main:app --reload`는 `main-server/src/api/main.py` 를 import해서 그 안의 `app` 객체를 실행하라는 뜻입니다.

그래서 import 경로를 `src...`에서 `main_server.src...`로 바꾸면 entrypoint도 같이 `uvicorn main_server.src.api.main:app --reload`로 바뀌어야 합니다. 이 문자열이 틀리면 서버는 바로 `ModuleNotFoundError`로 뜹니다. 현재 repo에서는 이 값이 실제 배포 파일에 박혀 있지는 않고, `AGENTS.md` 문서에만 적혀 있습니다.

실무적으로 보면 “지금 당장 안 돌아가는 구조”는 아닙니다. 하지만 “장기적으로 문제없는 구조”도 아닙니다. 수정하는 게 맞습니다.

핵심 문제는 두 가지입니다.

- import root가 `agent.src`, `shared.src`, `src`로 섞여 있습니다. 이건 새 사람이 보면 거의 반드시 헷갈립니다.
- package discovery가 거칠어서 `pyproject.toml` 기준으로는 `main-server` 아래 `src`뿐 아니라 `tests`, `state` 같은 것도 namespace package 후보로 잡힙니다.

그래서 제 판단은 이렇습니다.

- 지금 단계에서 최소한의 구조 수정은 하는 게 좋습니다.
- 최소 수정안은 `main-server -> main_server` 리네임, `src.` import를 `main_server.src.`로 통일, `pyproject`에서 `include/exclude`를 더 명시적으로 잡는 것입니다.
- 더 깔끔한 최종안은 아예 `src`를 import namespace에서 숨기는 정식 `src-layout`입니다. 예를 들면 `main_server/src/main_server/...` 형태로 바꿔서 import를 `main_server.api...`처럼 쓰는 방식입니다. 이게 제일 표준적이지만 변경 범위는 더 큽니다.

즉, 답을 짧게 하면:
- 현재 구조는 “돌아는 가는 구조”
- 실무 장기 운영 기준으로는 “지금 손보는 게 맞는 구조”

## User

main_server.src. 으로 해도 장기적으로 문제없잖아 어차피 main_server.api...해도 내부 폴더 위치 바뀌면 깨지는건 똑같잖아

## Assistant

맞습니다. “폴더 위치가 바뀌면 깨질 수 있다”는 점만 보면 둘 다 절대 안전하지는 않습니다.

다만 차이는 있습니다.

`main_server.src.api...`는 `src`라는 저장소 내부 구현 디렉터리까지 import 계약에 노출합니다. 즉 나중에 “이제 `src`는 layout일 뿐이고 import path에서는 빼자”라고 바꾸면 전부 깨집니다. 반면 `main_server.api...`는 `src`를 숨기기 때문에, 파일 시스템에서 `src/` 아래에 두든 다른 build root로 옮기든, 패키지로는 계속 `main_server.api`만 유지하면 소비자 코드는 안 깨질 수 있습니다.

즉 차이는 “안 깨지느냐”가 아니라 “무엇을 public contract로 고정하느냐”입니다.

실무적으로는 이렇게 보면 됩니다.

- `main_server.src...`
  - 내부 전용 repo라면 충분히 쓸 수 있습니다.
  - 특히 지금처럼 `agent.src`, `shared.src`가 이미 있으면 일관성 면에서는 오히려 낫습니다.
  - 대신 `src`가 장기 계약이 됩니다.

- `main_server.api...`
  - 더 표준적입니다.
  - `src`를 단순 layout으로 숨길 수 있어서 패키지 경계가 더 깔끔합니다.
  - 나중에 재배치할 때 선택지가 더 많습니다.

그래서 제 답은 이겁니다.

- 당신 말대로 `main_server.src...`도 장기 운영이 불가능한 구조는 아닙니다.
- 다만 “더 깔끔한 public namespace”는 아닙니다.
- 현재 repo 맥락에서는 일단 `main_server.src...`로 통일하는 선택도 충분히 합리적입니다.
- 대신 나중에 repo 전체에서 `.src`를 제거하는 대정리를 할 가능성은 남겨두는 게 좋습니다.

## User

main_server.src... 구조로 해서 진행하자

## Assistant

`main-server -> main_server` 리네임과 import 경로 통일을 진행하겠습니다. 먼저 현재 변경 사항 충돌 가능성과 실제 수정 범위를 확인한 뒤, 디렉터리 리네임, 패키지 설정 정리, 실행/테스트 경로 갱신 순서로 적용하겠습니다.

## Assistant

수정 범위를 확인했습니다. 실행에 직접 영향 있는 부분은 `main-server` 디렉터리, `src.` import, `pyproject`의 package discovery/testpaths, 그리고 개발 실행 문서의 entrypoint입니다. 개념적 서비스 명칭인 `main-server` 문자열은 경로와 구분해서, path/import에만 우선 반영하겠습니다.

## Assistant

이제 실제 리네임을 적용하고, 그 다음 import와 설정 파일을 한 번에 맞추겠습니다. 디렉터리 이동 후 바로 잔여 참조를 다시 검색해 누락 없이 정리하겠습니다.

## Assistant

리네임은 적용됐고, 이제 남은 건 import와 설정 파일 정리입니다. `pyproject`는 `main_server` 패키지를 루트에서 찾도록 바꾸고, 코드/스크립트/테스트의 `from src...`를 전부 `from main_server.src...`로 맞추겠습니다.

## Assistant

`pyproject`, 실행 문서, 코드 import를 함께 수정합니다. 이번 패치로 `main_server.src...`가 실제 import contract가 되도록 맞추고, 테스트 경로도 새 디렉터리 기준으로 갱신하겠습니다.

## Assistant

핵심 import와 설정은 반영됐습니다. 이제 잔여 참조를 재검색해서 누락을 잡고, `pytest`와 import 검증으로 실제로 새 namespace가 동작하는지 확인하겠습니다.

## Assistant

시스템 `python3`에는 프로젝트 의존성이 없어서 import 검증과 `pytest`가 실패했습니다. 저장소 방식대로 `uv run`으로 다시 검증하겠습니다.

## Assistant

`uv run` 검증은 sandbox가 `~/.cache/uv`에 쓰지 못해서 막혔습니다. 실제 검증에 필요한 작업이라 권한을 올려 다시 실행하겠습니다.

## Assistant

실행 검증은 통과했습니다. 마지막으로 새 `pyproject` 패턴이 `main_server.src`만 잡고 `main_server.tests`나 `main_server.state`를 끌어오지 않는지까지 확인한 뒤 마무리하겠습니다.

## Assistant

작업은 끝났고, 마지막으로 이번 추가 대화와 적용 결과를 노트에 이어 적겠습니다. 코드 변경은 이미 검증까지 끝난 상태입니다.
