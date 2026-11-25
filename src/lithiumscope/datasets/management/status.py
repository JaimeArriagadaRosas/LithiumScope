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

    @property
    def effective_state(self) -> DatasetState:
        if self.state is not None:
            return self.state
        return DatasetState.READY if self.ready else DatasetState.FAILED
