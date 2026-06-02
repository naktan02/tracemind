# SimMatch

이 패키지는 USB `semilearn/algorithms/simmatch/simmatch.py`의 핵심 train-step을
TraceMind Query SSL algorithm seam에 맞춘 구현이다.

보존한 USB core:

- projection head로 만든 normalized feature
- labeled feature memory bank와 labels bank
- queue distribution alignment
- weak feature와 bank similarity에서 teacher probability 생성
- strong feature와 bank similarity에서 student probability 생성
- SimMatch feature-level `in_loss`
- similarity 기반 weak probability smoothing
- fixed confidence threshold + CE consistency loss

TraceMind 차이:

- USB 텍스트 dataset 경로처럼 EMA teacher를 기본으로 쓰지 않는다.
- USB `idx_lb`는 labeled dataloader가 제공하는 stable `row_indices`로 대체한다.
- `K=args.lb_dest_len`은 labeled dataset row count lifecycle로 설정한다.
