"""USB UDA 원본 비교 기준."""

USB_UDA_ORIGINAL_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
USB_UDA_ORIGINAL_PATH = "semilearn/algorithms/uda/uda.py"

USB_UDA_PRESERVED_FLOW = (
    "labeled logits와 unlabeled weak/strong logits를 계산한다.",
    "TSA schedule로 labeled CE loss를 sample별 mask 처리한다.",
    "weak probability의 max confidence로 fixed threshold mask를 만든다.",
    "weak probability를 soft pseudo-label target으로 사용한다.",
    "strong logits에 masked soft CE consistency loss를 적용한다.",
    "total_loss = sup_loss + lambda_u * unsup_loss 구조를 유지한다.",
)

USB_UDA_TRACE_MIND_ADAPTATIONS = (
    "USB AlgorithmBase/use_cat/AMP/distributed glue는 중앙 Query SSL trainer가 "
    "소유한다.",
    "텍스트 weak/strong view는 materialized Query SSL dataloader surface를 사용한다.",
    "USB 원본의 optional DistAlignHook 확장 경로는 기본 UDA preset에서 등록하지 "
    "않는다.",
    "resume을 위해 USB self.it에 해당하는 iteration counter를 algorithm state로 "
    "저장한다.",
)
