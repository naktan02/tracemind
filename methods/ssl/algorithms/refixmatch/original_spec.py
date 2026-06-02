"""USB ReFixMatch provenance와 TraceMind 이식 범위."""

USB_REFIXMATCH_SOURCE = {
    "repo": "microsoft/Semi-supervised-learning",
    "commit": "1ef4cbebcc0b368158315aeb425053858cf6c845",
    "path": "semilearn/algorithms/refixmatch/refixmatch.py",
}

PRESERVED_CORE = (
    "labeled batch supervised CE를 계산한다.",
    "unlabeled weak view에서 pseudo-label과 fixed-threshold mask를 만든다.",
    "strong view logits에 FixMatch-style masked CE consistency를 적용한다.",
    "같은 strong logits와 weak probability target에 ReFixMatch KL loss를 추가한다.",
    "KL loss는 USB ConsistencyLoss(name='kl')처럼 mask complement에 적용한다.",
    "total_loss = sup + lambda_u * unsup + lambda_u * refix_loss 구조를 보존한다.",
)

TRACEMIND_ADAPTATIONS = (
    "USB AlgorithmBase의 AMP, distributed, process_out_dict glue는 TraceMind 공통 "
    "Query SSL trainer에 위임한다.",
    "원본의 optional DistAlignHook path는 CReST 같은 imbalanced extension용이므로 "
    "ReFixMatch core preset에는 포함하지 않는다.",
)
