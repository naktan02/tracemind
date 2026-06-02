"""USB MeanTeacher 원본 비교 기준."""

USB_MEANTEACHER_ORIGINAL_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
USB_MEANTEACHER_ORIGINAL_PATH = "semilearn/algorithms/meanteacher/meanteacher.py"
USB_MEANTEACHER_ORIGINAL_EMA_HOOK_PATH = "semilearn/core/hooks/ema.py"
USB_MEANTEACHER_ORIGINAL_EMA_UTILS_PATH = "semilearn/core/utils/misc.py::EMA"

USB_MEANTEACHER_PRESERVED_FLOW = (
    "labeled logits로 supervised CE loss를 계산한다.",
    "EMA teacher shadow weight를 적용한 model로 unlabeled weak logits를 계산한다.",
    "student model로 unlabeled strong logits를 계산한다.",
    "teacher weak probability를 detach한 consistency target으로 사용한다.",
    "student strong logits와 teacher weak probability 사이의 MSE consistency loss를 "
    "계산한다.",
    "unsup_warm_up 비율로 전체 step 초반의 unsupervised loss weight를 ramp-up한다.",
    "optimizer step 이후 EMA shadow를 갱신한다.",
    "total_loss = sup_loss + lambda_u * unsup_loss * unsup_warmup 구조를 유지한다.",
)

USB_MEANTEACHER_TRACE_MIND_ADAPTATIONS = (
    "USB AlgorithmBase/AMP/distributed glue는 중앙 Query SSL trainer가 소유한다.",
    "USB EMA full-model copy 대신 frozen backbone을 제외한 trainable parameter "
    "shadow를 사용한다.",
    "USB BN freeze/unfreeze는 텍스트 encoder에 BN running stats가 없어서 적용하지 "
    "않는다.",
    "USB 원본의 strong-branch logits typo로 보이는 대입은 복제하지 않고 student "
    "strong logits를 사용한다.",
    "resume을 위해 USB self.it와 EMA shadow를 algorithm state로 저장한다.",
)
