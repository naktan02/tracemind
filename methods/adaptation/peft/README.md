# PEFT Adaptation Seam

`methods/adaptation/peft/`는 frozen backbone 위에 PEFT adapter를 적용하는
builder protocol과 registry를 소유한다.

이 패키지는 transformer/peft optional dependency를 직접 import하지 않는다.
실행 rail이 `PeftAdapterBuildContext`로 dependency class/function을 넘기고,
builder는 adapter config 생성과 backbone 적용만 담당한다.

## 책임

- `PeftAdapterBuilder` protocol
- Hydra config에서 PEFT adapter 이름 해석
- adapter 이름을 builder factory로 연결하는 registry

## 제외

- Hydra entrypoint
- model checkpoint 저장
- classifier head 학습 loop
- agent-local runtime state
