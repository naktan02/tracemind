# MixMatch

USB MixMatch를 중앙 Query SSL control용 method core로 옮긴 구현이다.

원본 기준:

- repo: `microsoft/Semi-supervised-learning`
- commit: `1ef4cbebcc0b368158315aeb425053858cf6c845`
- path: `semilearn/algorithms/mixmatch/mixmatch.py`

TraceMind는 USB NLP config와 같이 `mixup_manifold=True` 경로만 지원한다. 문장 문자열이나
token id를 직접 섞지 않고, PEFT text encoder가 만든 classifier 직전 feature를 섞은 뒤
classifier head만 다시 실행한다.

보존한 핵심 흐름:

1. unlabeled weak/strong 확률 평균으로 pseudo-label을 만든다.
2. `T`로 sharpening한다.
3. labeled one-hot label과 unlabeled soft pseudo-label 두 벌을 concat한다.
4. feature와 target을 같은 lambda로 MixUp 한다.
5. mixed labeled chunk는 soft CE, mixed unlabeled chunk는 probability MSE로 학습한다.
6. `unsup_warm_up` 기반 linear ramp-up으로 unsupervised weight를 키운다.
