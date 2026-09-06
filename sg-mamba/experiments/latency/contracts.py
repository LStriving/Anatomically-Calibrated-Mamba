"""Shared contracts for independently replaceable latency stages."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol


class StructuralRunError(RuntimeError):
    """A missing adapter, schema mismatch, or other non-recoverable error."""


class RecoverableVideoError(RuntimeError):
    """A failure limited to one input video."""


class ExecutionMode(str, Enum):
    VALIDATE = "validate"
    SMOKE = "smoke"
    BENCHMARK = "benchmark"


class WeightsMode(str, Enum):
    SKIP = "skip"
    REQUIRED = "required"


class ResultScope(str, Enum):
    VALIDATION = "validation"
    PARTIAL_PIPELINE = "partial_pipeline"
    FULL_PIPELINE = "full_pipeline"


@dataclass(frozen=True)
class Payload:
    values: Mapping[str, Any] = field(default_factory=dict)

    def require(self, name: str) -> Any:
        if name not in self.values:
            raise StructuralRunError("Payload is missing required value: {}".format(name))
        return self.values[name]

    def with_value(self, name: str, value: Any) -> "Payload":
        updated = dict(self.values)
        updated[name] = value
        return Payload(updated)


class StageAdapter(Protocol):
    """A stage may allocate resources outside per-video timing in prepare."""

    name: str

    def prepare(self, context: Mapping[str, Any]) -> None:
        ...

    def run(self, payload: Payload, context: Mapping[str, Any]) -> Payload:
        ...

    def close(self) -> None:
        ...


@dataclass
class VideoRecord:
    video_id: str
    status: str
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    failure_stage: Optional[str] = None
    failure_message: Optional[str] = None


@dataclass
class RunRecord:
    mode: ExecutionMode
    weights_mode: WeightsMode
    scope: ResultScope
    successes: list[VideoRecord] = field(default_factory=list)
    failures: list[VideoRecord] = field(default_factory=list)
