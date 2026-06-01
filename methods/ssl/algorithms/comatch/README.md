# CoMatch Method-Local Primitives

이 패키지는 CoMatch acceptance path에서 먼저 필요한 projection/memory 계열 primitive를
소유한다. 아직 Query SSL registry에 `comatch` algorithm을 등록하지 않는다.

- `memory_bank.py`: weak feature/probability queue와 memory smoothing.

SimMatch 등 다른 method가 같은 의미를 반복해서 쓰는 시점에만 공통 hook/helper로
승격한다.
