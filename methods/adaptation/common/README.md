# Adaptation 공통 scaffold

`methods/adaptation/common/`은 linear head, PEFT text encoder처럼 여러
adaptation/update family가 공유하는 scaffold만 둔다.

여기에는 model input 방식이나 runner orchestration을 넣지 않는다. feature tensor
모델과 token batch 모델은 각 update family가 계속 소유하고, 이 패키지는 best
checkpoint 선택, selection-set epoch history record처럼 adaptation 내부에서
안정적으로 같은 부분만 담당한다. 중앙/FL이 함께 쓰는 평가 metric 계산은
`methods/evaluation/`을 source of truth로 둔다.

- `checkpointing.py`: selection metric 기준 best model state 복원
- `training_history.py`: epoch history record와 logging summary
- `selection_training_loop.py`: caller가 소유한 epoch 학습 결과를 selection 평가,
  history, best checkpoint 흐름으로 닫는 얇은 scaffold
