# Federation Strategy

main_server가 다음 round open에 적용할 운영 strategy 포인터를 소유한다.

- `models.py`: 파일로 저장되는 active strategy 상태 모델
- `active_strategy_service.py`: strategy 전환, 유효성 검증, 이력 조회

이 경계는 live 운영 선택만 관리한다. `ssl_method`는 composed SSL objective 전환이고,
`fssl_method`는 server가 다음 task에 method identity와 context를 실을 수 있는
method-owned FL SSL 전환이다. `fssl_method`가 설정된 active strategy에서는
`ssl_method`를 저장하지 않는다. method-owned runtime에 필요한 query SSL view/batch
scaffold는 task 생성 시 runtime fallback/config surface에서 파생되며, 사용자 선택값으로
노출하지 않는다.

SSL/FSSL method 의미와 기본 파라미터는 `methods/`와 `conf/strategy_axes/`가 소유하고,
agent에 전달되는 실행 계약은 `shared/src/contracts/training_contracts.py`의
`TrainingTaskPayload`가 소유한다.
