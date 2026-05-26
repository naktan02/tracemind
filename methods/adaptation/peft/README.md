# PEFT Adaptation Seam

`methods/adaptation/peft/`는 legacy compatibility surface다. PEFT adapter
protocol과 registry의 source of truth는 `methods/adaptation/peft_adapters/`로
이동했다.

이 패키지의 `base.py`, `registry.py`는 새 경로의 named symbol만 가져오는 shim이다.
새 내부 코드는 `methods/adaptation/peft_adapters/`를 direct-file import한다.

새 PEFT adapter 계층은 transformer/peft optional dependency를 직접 import하지 않는다.
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
