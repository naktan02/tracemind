# SSL Primitives

`methods/ssl/primitives/`는 여러 SSL algorithm이 같은 의미로 공유하는 순수
tensor/module 수식을 둔다.

- `probability.py`: softmax, temperature sharpening처럼 logits/probability 변환만
  수행하는 helper
- `losses.py`: soft-target CE, probability MSE처럼 method 간 의미가 같은 loss helper
- `mixup.py`: MixMatch/ReMixMatch류 feature/target MixUp primitive
- `projection.py`: CoMatch/SimMatch류 similarity objective가 공유하는 projection head

이 폴더는 method identity, Hydra config, dataset loading, artifact IO, trainer
lifecycle을 소유하지 않는다. 교체 가능한 pseudo-label, masking, thresholding,
distribution alignment role은 `methods/ssl/hooks/`에 둔다. 특정 method에서만 쓰는
세부 helper는 먼저 `methods/ssl/algorithms/<method>/`에 두고, 두 개 이상 algorithm에서
의미가 안정될 때만 이 폴더로 승격한다.
