# Partitioned PEFT Text Classifier Runtime

`partitioned/`는 PEFT text classifier trainable state를 여러 partition으로 나누어
학습/동기화하는 method-neutral runtime primitive를 소유한다.

## 책임

- partitioned model builder
- sparse sync primitive
- partitioned trainable model wrapper
- partitioned training loop와 budget helper

## 파일

- `budget.py`: partitioned local trainer의 labeled/unlabeled step budget 해석
- `model_builder.py`: partition 이름별 PEFT text classifier module build
- `sparse_sync.py`: partitioned C2S/S2C sparse sync projection
- `trainable_model.py`: physical trainable partition wrapper와 composed forward
- `training_loop.py`: supervised/unsupervised partition objective routing

## 금지

- `sigma`, `psi` 같은 FedMatch partition 이름의 source of truth 소유
- FedMatch supervised/unsupervised objective 직접 import
- method-specific file name 추가

FedMatch partition routing과 objective 의미는 `methods/federated_ssl/fedmatch/`에서
주입한다.
