# MeanTeacher

이 폴더는 Microsoft USB commit
`1ef4cbebcc0b368158315aeb425053858cf6c845`의
`semilearn/algorithms/meanteacher/meanteacher.py`와 EMA hook 흐름을
TraceMind Query SSL trainer로 옮긴다.

보존한 핵심:

- labeled CE loss
- EMA teacher weak-view prediction
- student strong-view prediction
- teacher weak probability target과 student strong logits의 MSE consistency loss
- `lambda_u * unsup_warmup` weighting
- optimizer step 이후 EMA shadow update

TraceMind 적용 차이:

- frozen text backbone까지 full teacher model로 복제하지 않고 trainable parameter만
  EMA shadow로 둔다.
- USB의 BN freeze/unfreeze는 PEFT text encoder 경로에 BN running stats가 없어서
  생략한다.
- 원본 파일의 strong branch typo로 보이는 `outs_x_ulb_w` 재사용은 복제하지 않는다.
