# ReMixMatch

USB ReMixMatch를 중앙 Query SSL control용 method core로 옮긴 구현이다.

원본 기준:

- repo: `microsoft/Semi-supervised-learning`
- commit: `1ef4cbebcc0b368158315aeb425053858cf6c845`
- path: `semilearn/algorithms/remixmatch/remixmatch.py`

TraceMind는 USB NLP config와 같이 `mixup_manifold=True`, `rot_loss_ratio=0.0` 경로만
지원한다. 문장 문자열이나 token id를 직접 섞지 않고, PEFT text encoder가 만든
classifier 직전 feature를 섞은 뒤 classifier head만 다시 실행한다.

보존한 핵심 흐름:

1. unlabeled weak view probability에 labeled class distribution 기준 EMA DA를 적용한다.
2. aligned probability를 `T`로 sharpening한다.
3. labeled one-hot label과 unlabeled soft pseudo-label 세 벌을 concat한다.
4. labeled, strong0, strong1, weak feature를 MixUp 한다.
5. mixed labeled chunk는 soft CE, mixed unlabeled chunks는 probability MSE로 학습한다.
6. 첫 strong view logits에는 sharpened weak target CE인 `u1_loss`를 추가한다.
7. `unsup_warm_up` 기반 linear ramp-up으로 unsupervised weight를 키운다.
