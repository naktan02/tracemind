# Federated Methods

`methods/federated/`는 federated learning에서 재사용되는 순수 method 계산을 둔다.

예를 들어 FedAvg의 가중평균 계산은 이 계층으로 이동할 수 있다. 반면 round 생성,
client update acceptance, model revision 발행, artifact publication은
`main_server` 책임으로 유지한다.
