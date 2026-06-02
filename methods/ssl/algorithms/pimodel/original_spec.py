"""USB PiModel 원본 비교 기준."""

USB_PIMODEL_ORIGINAL_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
USB_PIMODEL_ORIGINAL_PATH = "semilearn/algorithms/pimodel/pimodel.py"

USB_PIMODEL_PRESERVED_FLOW = (
    "labeled logits로 supervised CE loss를 계산한다.",
    "unlabeled weak/strong logits를 각각 계산한다.",
    "weak probability를 detach한 consistency target으로 사용한다.",
    "strong probability와 weak probability 사이의 MSE consistency loss를 계산한다.",
    "unsup_warm_up 비율로 전체 step 초반의 unsupervised loss weight를 ramp-up한다.",
    "total_loss = sup_loss + lambda_u * unsup_loss * unsup_warmup 구조를 유지한다.",
)

USB_PIMODEL_TRACE_MIND_ADAPTATIONS = (
    "USB AlgorithmBase/AMP/distributed glue는 중앙 Query SSL trainer가 소유한다.",
    "USB BN freeze/unfreeze는 텍스트 encoder에 BN running stats가 없어서 적용하지 "
    "않는다.",
    "resume을 위해 USB self.it에 해당하는 iteration counter를 algorithm state로 "
    "저장한다.",
)
