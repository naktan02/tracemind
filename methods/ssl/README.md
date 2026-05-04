# SSL Methods

`methods/ssl/`는 FixMatch, FlexMatch, UDA, pseudo-label self-training 같은
semi-supervised learning objective core를 둔다.

여기에는 loss 계산, method-specific step 계산, SSL hook 조합을 둔다.
`methods/ssl/hooks/`는 pseudo-label target 생성, confidence mask, threshold 기반
selection rule처럼 중앙 SSL과 FL SSL client가 공유하는 subroutine을 소유한다.
Dataset loading, Hydra 조립, artifact 저장은 `scripts/`가 맡는다.
