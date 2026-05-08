# Adaptation 공통 scaffold

`methods/adaptation/common/`은 fixed embedding classifier와 LoRA classifier가
공유하는 classification scaffold만 둔다.

여기에는 model input 방식이나 runner orchestration을 넣지 않는다. feature tensor
모델과 token batch 모델은 각 adapter family가 계속 소유하고, 이 패키지는 평가
report 계산, best checkpoint 선택, selection-set epoch history record처럼
안정적으로 같은 부분만 담당한다.
