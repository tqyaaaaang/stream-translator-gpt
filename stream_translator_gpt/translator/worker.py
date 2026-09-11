"""One request worker: own a provider session and execute ready attempts.

This loop never waits for context, output order, or retry backoff. Those decisions
belong to the scheduler. Only immutable requests/results cross the boundary, so
an HTTP response arriving after cancellation cannot mutate a published task.
"""

import logging
from dataclasses import dataclass
from typing import Callable

from ..utils.common import LogTime
from .types import TranslationBackend, TranslationRequest, TranslationResult


logger = logging.getLogger('main')


@dataclass(frozen=True)
class TranslationAttempt:
    sequence: int
    generation: int
    request: TranslationRequest


@dataclass(frozen=True)
class TranslationCompletion:
    worker: int
    sequence: int
    generation: int
    completed_at: float
    result: TranslationResult | None
    error: Exception | None = None


@dataclass(frozen=True)
class TranslationWorkerEvent:
    worker: int
    error: BaseException | None = None


class TranslationWorker:
    """Reusable worker with a mailbox and dispatcher-owned publication hooks."""

    def __init__(self, index: int, backend: TranslationBackend, inbox, stop_event,
                 publish_event: Callable[[TranslationWorkerEvent], None],
                 publish_completion: Callable[[int, TranslationAttempt,
                                               TranslationResult | None, Exception | None], None]):
        self.index = index
        self.backend = backend
        self.inbox = inbox
        self.stop_event = stop_event
        self._publish_event = publish_event
        self._publish_completion = publish_completion

    def run(self):
        try:
            with self.backend.open_session() as session:
                self._publish_event(TranslationWorkerEvent(self.index))
                while True:
                    attempt = self.inbox.get()
                    if attempt is None or self.stop_event.is_set():
                        return
                    result = None
                    error = None
                    try:
                        with LogTime('Translated task %d', attempt.request.task_id):
                            result = session.translate(attempt.request)
                        if not isinstance(result, TranslationResult):
                            raise TypeError('Translation session must return TranslationResult')
                    except Exception as exc:
                        error = exc
                    self._publish_completion(self.index, attempt, result, error)
        except BaseException as exc:
            # Initialization/finalization failures must not silently lose a slot.
            if self.stop_event.is_set():
                logger.error('Translation worker %d failed during shutdown (%s)',
                             self.index, type(exc).__name__)
            else:
                self._publish_event(TranslationWorkerEvent(self.index, exc))
