# Prototype Building

`methods/prototype/building/`은 prototype pack을 만드는 알고리즘 축을 소유한다.

현재 포함:

- `pack_builder.py`: single-centroid pack과 exact incremental build-state 계산
- `base.py`: build request/artifact/protocol
- `vector_ops.py`: centroid 정규화와 deterministic sampling helper
- `single.py`: `single` build strategy
- `kmeans.py`: `kmeans` build strategy
- `dbscan.py`: `dbscan` build strategy

경계:

- `shared`는 prototype payload contract, domain entity, serialization을 소유한다.
- `methods`는 어떤 builder 알고리즘으로 prototype pack을 만들지 소유한다.
- `main_server`와 `scripts`는 builder를 선택하고 실행/발행하는 runtime adapter다.
- compatibility wrapper를 남기지 않고 direct-file import를 사용한다.
