from enum import StrEnum


class DatasetState(StrEnum):
    MISSING = "missing"
    DOWNLOADING = "downloading"
    PARTIAL = "partial"
    READY = "ready"
    STALE = "stale"
    CORRUPTED = "corrupted"
    FAILED = "failed"


class ModelState(StrEnum):
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    READY = "ready"
    OUTDATED = "outdated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"
