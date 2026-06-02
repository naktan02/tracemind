# PiModel

`pimodel.py`는 Microsoft USB commit
`1ef4cbebcc0b368158315aeb425053858cf6c845`의
`semilearn/algorithms/pimodel/pimodel.py` 핵심 train step을 TraceMind Query SSL
surface로 옮긴 구현이다.

보존한 핵심 흐름:

1. labeled CE loss를 계산한다.
2. unlabeled weak/strong view logits를 계산한다.
3. weak probability를 detach해서 consistency target으로 쓴다.
4. strong probability와 weak probability 사이 MSE consistency loss를 계산한다.
5. `unsup_warm_up` 기반 linear ramp-up으로 unsupervised loss weight를 키운다.
6. `total_loss = sup_loss + lambda_u * unsup_loss * unsup_warmup`을 계산한다.

TraceMind 적용 차이:

- USB 원본의 BN freeze/unfreeze는 텍스트 encoder에 BN running stats가 없어서 적용하지
  않는다.
- EMA teacher, threshold mask, pseudo-label hardening은 PiModel 핵심이 아니므로
  추가하지 않는다.
