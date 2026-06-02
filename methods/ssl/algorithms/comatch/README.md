# CoMatch Query SSL Method

이 패키지는 CoMatch acceptance path의 tensor core, adapter, projection/memory 계열
primitive를 소유한다.

- `comatch.py`: Query SSL registry entrypoint, tensor core, algorithm adapter.
- `memory_bank.py`: weak feature/probability queue와 memory smoothing.
- `original_spec.py`: USB 원본 commit/path와 TraceMind v1 의도적 차이.

`queue_batch`는 USB처럼 batch 수 의미로 해석한다. 실제 memory bank row capacity는
학습 loop가 전달한 labeled/unlabeled batch 크기로 계산하고, memory smoothing은 초기
`queue_batch` optimizer step 이후부터 적용한다.

SimMatch 등 다른 method가 같은 의미를 반복해서 쓰는 시점에만 공통 hook/helper로
승격한다.
