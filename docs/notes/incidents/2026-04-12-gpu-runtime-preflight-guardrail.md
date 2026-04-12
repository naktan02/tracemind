# GPU Runtime Preflight Guardrail

## 배경

query adaptation supervised LoRA smoke를 검증하는 중, sandbox 안에서
`torch.cuda.is_available()`가 `False`로 보여 CPU fallback으로 실행했다.
같은 `.venv`를 sandbox 밖에서 확인하니 실제 머신에는 `RTX 4080 SUPER`가
정상적으로 보였다.

## 결정 내용

- GPU 의존 실행 전에는 실제 실행 환경에서 `nvidia-smi`와
  `.venv`의 `torch.cuda.is_available()`를 먼저 확인한다.
- sandbox 안에서 GPU가 보이지 않으면 GPU 부재로 단정하지 않는다.
- 공용 규칙은 root `AGENTS.md`에 짧게 올리고,
  반복 실행 절차는 `scripts/README.md`의 preflight checklist로 둔다.

## 이유

- 이번 문제는 코드 로직보다 실행 컨텍스트 오판에 가까웠다.
- 같은 코드와 같은 `.venv`도 sandbox 안/밖에서 GPU 가시성이 달랐다.
- 이런 종류의 실수는 `mistake.md` 누적보다 guardrail + checklist가 재발 방지에
  더 직접적이다.

## 보류 사항

- GPU smoke를 자주 반복한다면 `tests/integration`의 opt-in smoke나
  별도 script entrypoint로 승격할지 추후 판단한다.

## 다음 액션

1. 실제 GPU smoke/train run은 GPU 가시성을 확인한 환경에서 우선 실행한다.
2. 필요 시 `runtime=auto_local`을 기본 smoke runtime으로 사용한다.
