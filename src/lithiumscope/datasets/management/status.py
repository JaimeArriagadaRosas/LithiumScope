from __future__ import annotations

from dataclasses import dataclass

from lithiumscope.core.states import DatasetState


@dataclass(frozen=True)
class DatasetStatus:
    key: str
    ready: bool
    path: str | None
    detail: str = ""
    state: DatasetState | None = None

    def __post_init__(self):
        if self.state is not None and self.ready != (self.state == DatasetState.READY):
            raise ValueError("Dataset readiness conflicts with explicit state")
        if self.ready and not self.path:
            raise ValueError("Ready dataset requires a path")

    @property
    def effective_state(self) -> DatasetState:
        if self.state is not None:
            return self.state
        return DatasetState.READY if self.ready else DatasetState.FAILED
