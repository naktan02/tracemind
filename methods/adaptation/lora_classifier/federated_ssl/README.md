# LoRA-Classifier FL SSL Slice

이 폴더는 `lora_classifier` adapter family가 FL SSL method-local objective를 실제
LoRA/classifier 학습 loop와 shared update payload로 실행하는 slice다.

- `method_owned_training.py`: method descriptor의 `runtime_entrypoint`를 읽어
  LoRA-classifier method-owned local core를 호출한다.
- `server_update_policy.py`: FL SSL `server_update_policy`를 LoRA-classifier
  aggregation backend로 해석한다.
- `fedmatch_training.py`: FedMatch local objective를 LoRA-classifier model/loaders,
  delta materialization, update envelope에 연결한다.
- `fedmatch_partitioned_loop.py`: LoRA-classifier trainable tensor 위에서 FedMatch
  `sigma`/`psi` logical partition step을 실행한다.

FedMatch의 원본 의미, hyperparameter snapshot, agreement loss, helper policy,
server/round policy는 `methods/federated_ssl/fedmatch/`가 소유한다. 이 폴더는 그
의미를 특정 adapter family에서 실행하는 구현만 소유한다.

새 full encoder family가 생기면 이 폴더에 파일을 추가하지 말고 해당 family package
아래의 FL SSL slice로 둔다. DoRA처럼 같은 LoRA-classifier scaffold 안의 PEFT 옵션이면
`config.py`와 PEFT builder seam을 확장하고 method 폴더를 늘리지 않는다.
