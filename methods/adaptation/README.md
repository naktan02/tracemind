# Adaptation Methods

`methods/adaptation/`은 LoRA, RsLoRA, DoRA, linear probe, full fine-tune 같은
학습 adapter 적용 core를 둔다.

rank, alpha, target module 같은 실행 파라미터는 code folder가 아니라 config에서
선택한다.
