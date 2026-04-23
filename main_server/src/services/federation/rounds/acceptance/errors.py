"""Round acceptance error types."""


class RoundConflictError(ValueError):
    """현재 round 상태와 충돌하는 요청."""


class RoundValidationError(ValueError):
    """round 문맥과 맞지 않는 입력."""
