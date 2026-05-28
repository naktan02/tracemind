# Federated Methods

`methods/federated/`는 federated learning에서 재사용되는 순수 method 계산을 둔다.

현재 `aggregation/fedavg/`는 example_count 기반 FedAvg의 공통 가중 평균 산술과
strategy wiring을 소유한다. payload adapter별 delta 해석과 next-state 계산은
`methods/adaptation/<family>/aggregation/fedavg.py`가 소유한다.

`shard_policy/`는 FL SSL non-IID 비교에서 재사용되는 label-dominant와
Dirichlet label-skew client assignment 계산을 소유한다.

`client_split.py`는 labeled pool selection, client-local labeled/unlabeled split,
labeled exposure policy의 client/bootstrap row 노출 규칙, storage group과
report/path용 compact slug를 소유한다.

`participation.py`는 round별 client subset 선택 정책을 소유한다. 기본값은
`all_clients`이고, FedMatch류 partial participation 실험은 `fraction_random` 또는
`fixed_count_random`을 config에서 고른다.

`aggregation_weighting.py`는 payload adapter와 분리된 aggregation weight 기준을
소유한다. 기본값은 기존 FedAvg와 같은 `example_count`이고, method가 필요하면
`uniform` 또는 `accepted_count`를 capability plan으로 요구한다.

반면 round 생성, client update acceptance, model revision 발행,
artifact publication은 `main_server` 책임으로 유지한다.
