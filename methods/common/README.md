# Method Common

`methods/common/`은 여러 method family에서 공유하는 작은 helper를 둔다.

공유 유틸리티는 실제 두 개 이상의 method에서 필요할 때만 추가한다. 단일 사용처용
추상화는 만들지 않는다.

현재 공통 helper:

- `config_reading.py`: method/profile config mapping에서 primitive 값을 읽고
  공통 validation을 적용하는 helper
- `registry.py`: method family별 registry instance가 같은 normalize/register/list
  규칙을 쓰게 하는 얇은 helper. registry instance 자체는 domain별 package가
  소유한다.
