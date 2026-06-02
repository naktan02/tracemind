# ReFixMatch

USB ReFixMatch를 중앙 Query SSL control용 method core로 옮긴 구현이다.

원본 기준:

- repo: `microsoft/Semi-supervised-learning`
- commit: `1ef4cbebcc0b368158315aeb425053858cf6c845`
- path: `semilearn/algorithms/refixmatch/refixmatch.py`

보존한 핵심 흐름:

1. labeled supervised CE를 계산한다.
2. unlabeled weak view probability로 fixed threshold mask와 pseudo-label을 만든다.
3. strong view logits에 masked CE consistency loss를 적용한다.
4. 같은 strong view logits에 weak probability target 기반 KL loss를 추가한다.
5. KL loss는 USB 원본처럼 `1 - mask` 영역에 적용한다.
6. `sup + lambda_u * unsup + lambda_u * refix_loss`로 합산한다.
