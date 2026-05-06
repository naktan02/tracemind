# LoRA Adaptation

`methods/adaptation/lora/`는 LoRA 계열 PEFT adapter builder 구현을 소유한다.

제공 builder는 `lora`와 `rslora`다. rank, alpha, dropout, target_modules,
bias, use_rslora 같은 값은 `conf/strategy_axes/adaptation/peft_adapter/`
YAML에서 선택한다.

DoRA, IA3 같은 새 PEFT method는 실제 config 생성과 manifest summary가 준비될
때 별도 파일로 추가한다. 이름만 미리 열어 두지 않는다.
