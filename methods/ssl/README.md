# SSL Methods

`methods/ssl/`는 FixMatch, FlexMatch, UDA, pseudo-label self-training 같은
semi-supervised learning objective core를 둔다.

여기에는 loss 계산, pseudo-label 생성, confidence mask, method-specific step
계산을 둔다. Dataset loading, Hydra 조립, artifact 저장은 `scripts/`가 맡는다.
