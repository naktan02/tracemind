# Fixed Embedding Classifier Bootstrap Source

이 패키지는 중앙 SSL bootstrap에서 사용할 수 있는 fixed embedding classifier
artifact source 구현이다. 실행 가능한 별도 seed entrypoint가 아니라
`scripts.support.query_ssl_peft.runners.teacher_source`가 선택적으로 호출하는
bootstrap workflow 내부 구현이다.

공유되는 hook 계약은 `methods/ssl/hooks/teacher.py`가 소유한다. 이 패키지는 Hydra
cfg, embedding runtime, artifact 저장/로드처럼 중앙 실험 실행에 붙은 glue만 맡는다.
