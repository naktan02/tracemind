# SSL Methods

`methods/ssl/`는 FixMatch, FlexMatch, UDA, pseudo-label self-training 같은
semi-supervised learning objective framework surface를 둔다.

`methods/ssl/algorithms/`에는 method-specific step 계산과 algorithm descriptor를
둔다.
`methods/ssl/hooks/`는 pseudo-label target 생성, confidence mask, threshold 기반
selection rule처럼 중앙 SSL과 FL SSL client가 공유하는 subroutine을 소유한다.
`methods/ssl/base.py`와 `methods/ssl/registry.py`는 공통 framework surface다.
Dataset loading, Hydra 조립, artifact 저장은 `scripts/`가 맡는다.
