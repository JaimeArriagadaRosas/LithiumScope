class LithiumScopeError(RuntimeError):
    """Base application error."""


class DatasetError(LithiumScopeError):
    """Dataset acquisition or validation failed."""


class ModelNotReadyError(LithiumScopeError):
    """A requested model has not been trained or configured yet."""


class InputValidationError(LithiumScopeError):
    """Prediction input does not satisfy the model contract."""
