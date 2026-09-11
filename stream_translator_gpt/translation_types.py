"""Immutable contracts between translation scheduling and provider backends.

Only the pipeline/scheduler know about mutable ``TranslationTask`` objects. A
backend sees one immutable request and returns one immutable result; it neither
owns queue state nor decides when a request should run or be retried.
"""

import math
from dataclasses import dataclass
from typing import ContextManager, Protocol


@dataclass(frozen=True)
class TranslationRequest:
    task_id: int
    transcript: str
    references: tuple[str, ...] = ()
    context: str = ''
    # Absolute monotonic deadline for this attempt, including all provider calls.
    deadline: float | None = None


@dataclass(frozen=True)
class TranslationResult:
    translation: str
    context: str | None = None


@dataclass(frozen=True)
class TranslationSettings:
    workers: int = 3
    history_size: int = 3
    use_translation_context: bool = False
    timeout: float = 10
    retry_if_translation_fails: bool = False

    def __post_init__(self):
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError('Translation workers must be a positive integer')
        if (isinstance(self.history_size, bool) or not isinstance(self.history_size, int)
                or self.history_size < 0):
            raise ValueError('Translation history size must be a nonnegative integer')
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or self.timeout < 0):
            raise ValueError('Translation timeout must be finite and nonnegative')


class TranslationSession(Protocol):
    """Provider session owned and used by exactly one request worker."""

    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Execute one attempt, raising an exception on provider/protocol failure."""
        ...


class TranslationBackend(Protocol):
    """Factory for independently owned, reusable provider sessions."""

    def open_session(self) -> ContextManager[TranslationSession]:
        ...
