"""Bounded, context-aware translation scheduling with ordered publication.

Only the dispatcher owns logical tasks/history/context. Workers receive private
attempts and return immutable snapshots; even a timed-out HTTP call cannot mutate
an exported task. Waiting for context, output capacity, or retry backoff never
occupies a request worker.
"""

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from ..utils.common import TranslationTask
from .types import TranslationBackend, TranslationRequest, TranslationResult, TranslationSettings
from .worker import (TranslationAttempt, TranslationCompletion, TranslationWorker,
                                 TranslationWorkerEvent)


logger = logging.getLogger('main')


@dataclass
class _Task:
    sequence: int
    original: TranslationTask
    references: tuple[str, ...]
    context: str = ''
    status: str = 'new'
    generation: int = 0
    deadline: float | None = None
    retry_at: float = 0.0


class ParallelTranslationScheduler:
    """Run at most ``workers`` physical requests, publishing in source order.

    ``clock`` is injectable for deterministic deadline/backoff tests. An input
    queue has no notification hook, so idle input and output are polled finitely;
    completions and stop wake the dispatcher immediately.
    """

    POLL_SECONDS = 0.05
    REQUEST_TIMEOUT_SECONDS = 60.0
    RETRY_BASE_SECONDS = 1.0
    RETRY_MAX_SECONDS = 30.0
    SHUTDOWN_SECONDS = 1.0

    def __init__(self, backend: TranslationBackend, settings: TranslationSettings, *,
                 clock=time.monotonic, output_finished: threading.Event | None = None):
        self.backend = backend
        self.settings = settings
        self.workers = settings.workers
        self.output_finished = output_finished if output_finished is not None else threading.Event()
        self.max_pending = max(16, 4 * settings.workers)
        self._clock = clock
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        # Linearize stop() against publication, without holding a lock during IO.
        # Workers briefly hold this same lock while timestamping/enqueuing a
        # completion. Reentrancy lets _send_event share that atomic publication.
        self._publication_lock = threading.RLock()
        self._events = queue.SimpleQueue()
        self._inboxes = [queue.Queue(maxsize=1) for _ in range(settings.workers)]
        self._threads = []
        self._tasks: dict[int, _Task] = {}
        self._new_tasks: deque[int] = deque()
        self._idle: deque[int] = deque()
        # Physical slots remain occupied after a logical timeout until return.
        self._running: dict[int, tuple[int, int]] = {}
        self._history = deque(maxlen=settings.history_size)
        self._next_sequence = 0
        self._next_output = 0
        self._context_sequence = -1
        self._context = ''
        self._eof = False
        self._eof_sent = False
        self._started = False

    def stop(self):
        with self._publication_lock:
            self._stop_event.set()
        self._wake.set()

    def _send_event(self, event: TranslationCompletion | TranslationWorkerEvent):
        with self._publication_lock:
            self._events.put(event)
        self._wake.set()

    def _record_completion(self, worker: int, attempt: TranslationAttempt,
                           result: TranslationResult | None, error: Exception | None):
        with self._publication_lock:
            # Timestamp and enqueue are atomic relative to expiration and stop.
            # An already-published timely result cannot lose to a later check.
            self._send_event(TranslationCompletion(
                worker, attempt.sequence, attempt.generation, self._clock(), result, error,
            ))

    @property
    def latest_translation_context(self):
        """Most recent accepted context by source order, for diagnostics only."""
        return self._context

    def _finish(self, task, *, translation=None, context=None):
        original = task.original
        with original._translation_lock:
            original.translation = translation
            original.translation_context = context
            original.translation_failed = not bool(translation)
            original._translation_finalized = True
            original.phase = TranslationTask.Phase.TRANSLATED
        task.status = 'done'
        if (translation and context and self.settings.use_translation_context
                and task.sequence > self._context_sequence):
            self._context_sequence = task.sequence
            self._context = context

    def _complete(self, completion: TranslationCompletion):
        running = self._running.pop(completion.worker, None)
        if running != (completion.sequence, completion.generation):
            raise RuntimeError('Translation worker returned an unexpected attempt')
        self._idle.append(completion.worker)
        task = self._tasks.get(completion.sequence)
        if task is None or task.status != 'running' or task.generation != completion.generation:
            # Logical timeout may already have finalized and exported the task.
            return
        if task.deadline is not None and completion.completed_at >= task.deadline:
            logger.error('Translation task %d timed out after %ss', task.original.id, self.settings.timeout)
            self._finish(task)
            return
        result = completion.result
        success = (completion.error is None and result is not None and bool(result.translation)
                   and (not self.settings.use_translation_context or bool(result.context)))
        if success:
            self._finish(task, translation=result.translation, context=result.context)
        elif self.settings.retry_if_translation_fails:
            task.status = 'retry'
            backoff = min(self.RETRY_BASE_SECONDS * 2 ** min(task.generation - 1, 10),
                          self.RETRY_MAX_SECONDS)
            task.retry_at = completion.completed_at + backoff
            logger.info('Translation task %d failed; retrying after %.2fs', task.original.id, backoff)
        else:
            # Provider exceptions may include URLs, credentials, or input text.
            logger.error('Translation task %d failed (%s)', task.original.id,
                         type(completion.error).__name__ if completion.error else 'invalid response')
            self._finish(task)

    def _drain_events(self):
        progress = False
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return progress
            progress = True
            with self._publication_lock:
                if self._stop_event.is_set():
                    continue
                if isinstance(event, TranslationCompletion):
                    self._complete(event)
                elif event.error is not None:
                    raise RuntimeError(f'Translation worker {event.worker} failed') from event.error
                else:
                    self._idle.append(event.worker)

    def _expire(self):
        progress = False
        now = self._clock()
        with self._publication_lock:
            if self._stop_event.is_set():
                return False
            if not self._events.empty():
                # A worker may have published after _drain_events returned. Give
                # those events their timestamp-based decision before expiring
                # anything; the common lock makes this check atomic with publish.
                return False
            for task in self._tasks.values():
                if task.status in ('running', 'retry') and task.deadline is not None and now >= task.deadline:
                    logger.error('Translation task %d timed out after %ss', task.original.id, self.settings.timeout)
                    self._finish(task)
                    progress = True
        return progress

    def _publish(self, output_queue):
        progress = False
        with self._publication_lock:
            while not self._stop_event.is_set():
                task = self._tasks.get(self._next_output)
                if task is None or task.status != 'done':
                    break
                try:
                    output_queue.put_nowait(task.original)
                except queue.Full:
                    break
                del self._tasks[self._next_output]
                self._next_output += 1
                progress = True
        return progress

    def _admit(self, input_queue):
        progress = False
        while not self._eof and not self._stop_event.is_set() and len(self._tasks) < self.max_pending:
            try:
                original = input_queue.get_nowait()
            except queue.Empty:
                break
            progress = True
            if original is None:
                self._eof = True
                break
            sequence = self._next_sequence
            self._next_sequence += 1
            self._tasks[sequence] = _Task(sequence, original, tuple(self._history))
            self._history.append(original.transcript)
            self._new_tasks.append(sequence)
        return progress

    def _context_ready(self, task):
        if not self.settings.use_translation_context:
            return True
        required_sequence = task.sequence - self.settings.history_size - 1
        if self._context_sequence >= required_sequence:
            return True
        can_update = any(candidate.sequence >= required_sequence
                         and candidate.status in ('running', 'retry')
                         for candidate in self._tasks.values())
        if can_update:
            return False
        logger.info('Translation task %d has a context gap; continuing with the latest available context',
                    task.original.id)
        return True

    def _dispatch(self):
        progress = False
        while self._idle:
            with self._publication_lock:
                if self._stop_event.is_set():
                    break
                now = self._clock()
                task = next((candidate for candidate in self._tasks.values()
                             if candidate.status == 'retry' and candidate.retry_at <= now), None)
                if task is None:
                    if not self._new_tasks:
                        break
                    task = self._tasks[self._new_tasks[0]]
                    if not self._context_ready(task):
                        break
                    self._new_tasks.popleft()
                    task.context = self._context if self.settings.use_translation_context else ''
                    task.original.context_transcripts = list(task.references)
                    task.original.input_translation_context = task.context
                    task.original.start_time = datetime.now(timezone.utc)
                    task.deadline = now + self.settings.timeout if self.settings.timeout > 0 else None
                # No retry may reset its deadline, input snapshot, or capacity slot.
                if task.deadline is not None and now >= task.deadline:
                    self._finish(task)
                    progress = True
                    continue
                request_deadline = (min(now + self.REQUEST_TIMEOUT_SECONDS, task.deadline)
                                    if task.deadline is not None else now + self.REQUEST_TIMEOUT_SECONDS)
                request = TranslationRequest(task.original.id, task.original.transcript,
                                             task.references, task.context, request_deadline)
                task.generation += 1
                task.status = 'running'
                worker = self._idle.popleft()
                self._running[worker] = (task.sequence, task.generation)
                self._inboxes[worker].put_nowait(TranslationAttempt(task.sequence, task.generation, request))
                progress = True
        return progress

    def _wait_seconds(self):
        now = self._clock()
        upcoming = [self.POLL_SECONDS]
        for task in self._tasks.values():
            if task.status in ('running', 'retry') and task.deadline is not None:
                upcoming.append(max(0.0, task.deadline - now))
            if task.status == 'retry' and task.retry_at > now:
                upcoming.append(task.retry_at - now)
        return min(upcoming)

    def _send_eof(self, output_queue, *, force=False):
        if self._eof_sent:
            return True
        self.output_finished.set()
        while True:
            try:
                output_queue.put_nowait(None)
                break
            except queue.Full:
                if not force:
                    self.output_finished.clear()
                    return False
                # On cancellation/fatal error it is preferable to discard one
                # buffered result than leave a consumer waiting forever for EOF.
                # Normal EOF never discards output and waits for consumer space.
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    pass
        self._eof_sent = True
        return True

    def _shutdown(self):
        self.stop()
        for inbox in self._inboxes:
            try:
                inbox.put_nowait(None)
            except queue.Full:
                # A not-yet-picked assignment will observe stop before starting.
                pass
        # One bounded budget for the entire pool, not one full budget per worker.
        deadline = time.monotonic() + self.SHUTDOWN_SECONDS
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        still_running = sum(thread.is_alive() for thread in self._threads)
        if still_running:
            logger.info('%d translation request worker(s) still finishing; their clients close on return', still_running)

    def loop(self, input_queue, output_queue):
        with self._publication_lock:
            if self._started:
                raise RuntimeError('A translation scheduler can only be run once')
            self._started = True
        try:
            if not self._stop_event.is_set():
                for index in range(self.workers):
                    worker = TranslationWorker(index, self.backend, self._inboxes[index], self._stop_event,
                                               self._send_event, self._record_completion)
                    thread = threading.Thread(target=worker.run,
                                              name=f'translation-request-{index}', daemon=True)
                    thread.start()
                    self._threads.append(thread)
            while not self._stop_event.is_set():
                # Clear BEFORE reading event queues: a producer cannot lose a
                # wakeup between our final queue read and Event.wait().
                self._wake.clear()
                progress = self._drain_events()
                progress = self._expire() or progress
                progress = self._publish(output_queue) or progress
                progress = self._admit(input_queue) or progress
                progress = self._dispatch() or progress
                if self._eof and not self._tasks and self._send_eof(output_queue):
                    logger.debug('All pending translation tasks completed')
                    return
                if not progress:
                    self._wake.wait(self._wait_seconds())
            # Active shutdown discards pending work rather than waiting for a
            # blocked consumer or uncooperative provider. Normal EOF always drains.
            with self._publication_lock:
                for task in self._tasks.values():
                    if task.status != 'done':
                        self._finish(task)
            self._send_eof(output_queue, force=True)
        except BaseException:
            # The scheduler owns the EOF signal; the outer pipeline wrapper must
            # not append another sentinel when propagating this worker failure.
            self._send_eof(output_queue, force=True)
            raise
        finally:
            self._shutdown()
