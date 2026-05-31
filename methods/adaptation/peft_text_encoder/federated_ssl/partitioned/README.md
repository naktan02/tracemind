# Partitioned PEFT Text Encoder Runtime

`partitioned/`는 PEFT text encoder trainable state를 여러 partition으로 나누어
학습/동기화하는 method-neutral runtime primitive를 소유한다.

## 책임

- partitioned model builder
- sparse sync primitive
- partitioned trainable model wrapper
- partitioned training loop와 budget helper

## 파일

- `budget.py`: partitioned local trainer의 labeled/unlabeled step budget 해석
- `model_builder.py`: partition 이름별 PEFT text encoder module build
- `sparse_sync.py`: partitioned C2S/S2C sparse sync projection
- `trainable_model.py`: physical trainable partition wrapper와 composed forward
- `training_loop.py`: supervised/unsupervised partition objective routing

## 금지

- method-local partition 이름의 source of truth 소유
- method-owned supervised/unsupervised objective 직접 import
- method-specific file name 추가

partition routing과 objective 의미는 `methods/federated_ssl/<method>/`에서
runtime plan으로 주입한다.
