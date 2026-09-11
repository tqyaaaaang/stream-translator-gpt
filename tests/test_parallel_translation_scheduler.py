import queue
import random
import threading
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from unittest import mock

from stream_translator_gpt.utils.common import TranslationTask
from stream_translator_gpt.translator.scheduler import ParallelTranslationScheduler
from stream_translator_gpt.translator.types import TranslationRequest, TranslationResult, TranslationSettings


class _Clock:
    def __init__(self):
        self.value = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def advance(self, seconds, scheduler):
        with self.lock:
            self.value += seconds
        scheduler._wake.set()


class _Call:
    def __init__(self, request):
        self.request = request
        self.worker = threading.get_ident()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.error = None
        self.result = TranslationResult('')

    def succeed(self, context=None):
        self.result = TranslationResult(f'translation-{self.request.transcript}',
                                        context or f'context-{self.request.transcript}')
        self.release.set()

    def fail(self, exception=None):
        self.error = exception
        self.release.set()


class _Backend:
    def __init__(self, history=2, context=True, timeout=0, retry=False, automatic=False):
        self.test_settings = TranslationSettings(history_size=history, use_translation_context=context,
                                                  timeout=timeout, retry_if_translation_fails=retry)
        self.calls = queue.Queue()
        self.all_calls = []
        self.automatic = automatic
        self.lock = threading.Lock()
        self.active = set()
        self.max_active = 0
        self.overlap = False
        self.session_threads = []
        self.closed_threads = []

    @contextmanager
    def open_session(self):
        owner = threading.get_ident()
        backend = self

        class Session:
            def translate(self, request):
                if threading.get_ident() != owner:
                    raise AssertionError('Session used outside its owning worker')
                return backend._execute(request)

        with self.lock:
            self.session_threads.append(owner)
        try:
            yield Session()
        finally:
            with self.lock:
                self.closed_threads.append(threading.get_ident())

    def _execute(self, request):
        call = _Call(request)
        with self.lock:
            self.overlap |= request.transcript in self.active
            self.active.add(request.transcript)
            self.max_active = max(self.max_active, len(self.active))
            self.all_calls.append(call)
        self.calls.put(call)
        if self.automatic:
            call.succeed()
        try:
            if not call.release.wait(5):
                raise RuntimeError('Test did not release a controlled request')
            if call.error:
                raise call.error
            return call.result
        finally:
            with self.lock:
                self.active.remove(request.transcript)
            call.returned.set()


class _Run:
    def __init__(self, backend, tasks=(), workers=3, clock=None, eof=True, output_size=0, capacity=None):
        self.backend = backend
        self.input = queue.Queue()
        self.output = queue.Queue(maxsize=output_size)
        self.scheduler = ParallelTranslationScheduler(backend, replace(backend.test_settings, workers=workers),
                                                       **({'clock': clock} if clock else {}))
        self.scheduler.SHUTDOWN_SECONDS = 0.05
        self.completions = []
        self.completed_condition = threading.Condition()
        send_event = self.scheduler._send_event

        def record_event(event):
            if hasattr(event, 'completed_at'):
                with self.completed_condition:
                    self.completions.append((event.sequence, event.generation))
                    self.completed_condition.notify_all()
            send_event(event)

        self.scheduler._send_event = record_event
        if capacity is not None:
            self.scheduler.max_pending = capacity
        self.tasks = []
        for index, line_id in enumerate(tasks):
            task = TranslationTask(line_id, None, (index, index + 1))
            task.transcript = f'source-{index}'
            self.tasks.append(task)
            self.input.put(task)
        if eof:
            self.input.put(None)
        self.error = None
        self.finished = threading.Event()

        def run():
            try:
                self.scheduler.loop(self.input, self.output)
            except BaseException as exc:
                self.error = exc
            finally:
                self.finished.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def call(self):
        return self.backend.calls.get(timeout=2)

    def no_call(self):
        try:
            call = self.backend.calls.get(timeout=0.03)
        except queue.Empty:
            return
        raise AssertionError(f'Unexpected request: {call.request.transcript}')

    def result(self):
        return self.output.get(timeout=2)

    def completed(self, sequence, generation=1):
        with self.completed_condition:
            if not self.completed_condition.wait_for(
                    lambda: (sequence, generation) in self.completions, timeout=1):
                raise AssertionError('Worker did not capture its completion')

    def finish(self):
        if not self.finished.wait(2):
            raise AssertionError('Scheduler did not terminate (possible deadlock)')
        if self.error:
            raise self.error
        if not self.output.empty():
            raise AssertionError('Unexpected duplicate/unconsumed output')

    def cleanup(self):
        self.scheduler.stop()
        # Catch already assigned attempts that started concurrently with stop.
        for call in list(self.backend.all_calls):
            call.release.set()
        self.finished.wait(2)
        for thread in self.scheduler._threads:
            thread.join(1)


class ParallelTranslationSchedulerTest(unittest.TestCase):
    def start(self, backend=None, tasks=(0,), **kwargs):
        run = _Run(backend or _Backend(), tasks, **kwargs)
        self.addCleanup(run.cleanup)
        return run

    def assert_outputs(self, run, ids):
        self.assertEqual([run.result().id for _ in ids], list(ids))
        self.assertIsNone(run.result())
        run.finish()

    def test_ordered_output_uses_latest_source_context_not_latest_completion(self):
        run = self.start(tasks=(90, 3, 700, -2, 4), workers=3)
        first = {call.request.transcript: call for call in (run.call(), run.call(), run.call())}
        self.assertEqual(first['source-2'].request.references, ('source-0', 'source-1'))
        self.assertTrue(all(call.request.context == '' for call in first.values()))
        first['source-2'].succeed('newest')
        fourth = run.call()
        self.assertEqual(fourth.request.context, 'newest')
        self.assertEqual(fourth.request.references, ('source-1', 'source-2'))
        first['source-1'].succeed('older')
        fifth = run.call()
        self.assertEqual(fifth.request.context, 'newest')
        fifth.succeed('newest-of-all')
        fourth.succeed('not-newest')
        self.assertTrue(run.output.empty())
        first['source-0'].succeed('oldest')
        self.assert_outputs(run, (90, 3, 700, -2, 4))
        self.assertEqual(run.scheduler.latest_translation_context, 'newest-of-all')
        self.assertEqual(run.backend.max_active, 3)

    def test_zero_history_serializes_only_context_mode(self):
        for context in (False, True):
            with self.subTest(context=context):
                run = self.start(_Backend(history=0, context=context), tasks=range(3), workers=3)
                first = run.call()
                if context:
                    run.no_call()
                    first.succeed()
                    second = run.call()
                    self.assertEqual(second.request.context, 'context-source-0')
                    second.succeed()
                    run.call().succeed()
                else:
                    others = [run.call(), run.call()]
                    for call in [first] + others:
                        self.assertEqual(call.request.references, ())
                        call.succeed()
                self.assert_outputs(run, range(3))
                self.assertEqual(run.backend.max_active, 1 if context else 3)

    def test_gate_exact_boundary_for_multiple_windows_and_worker_counts(self):
        for history, workers in ((0, 1), (0, 4), (1, 3), (2, 5)):
            with self.subTest(history=history, workers=workers):
                run = self.start(_Backend(history=history), tasks=range(history + 2), workers=workers)
                first = [run.call() for _ in range(min(workers, history + 1))]
                run.no_call()
                newest = max(first, key=lambda call: int(call.request.transcript.rsplit('-', 1)[1]))
                newest.succeed()
                following = run.call()
                self.assertEqual(following.request.context,
                                 f'context-{newest.request.transcript}')
                following.succeed()
                for call in first:
                    if call is not newest:
                        call.succeed()
                self.assert_outputs(run, range(history + 2))

    def test_retry_zero_history_single_worker_cannot_deadlock_after_eof(self):
        clock = _Clock()
        run = self.start(_Backend(history=0, retry=True, timeout=20), tasks=(10, 20), workers=1, clock=clock)
        first = run.call()
        first.fail(ConnectionError('temporary'))
        self.assertTrue(first.returned.wait(1))
        run.no_call()
        clock.advance(2, run.scheduler)
        retry = run.call()
        self.assertEqual(retry.request.task_id, 10)
        self.assertEqual(retry.request.references, first.request.references)
        self.assertEqual(retry.request.context, first.request.context)
        # An absolute deadline makes the retry's remaining budget smaller
        # without changing the logical deadline or request input snapshot.
        self.assertEqual(retry.request.deadline, first.request.deadline)
        retry.succeed()
        following = run.call()
        self.assertEqual(following.request.task_id, 20)
        following.succeed()
        self.assert_outputs(run, (10, 20))
        self.assertFalse(run.backend.overlap)

    def test_full_buffer_does_not_prevent_retry_or_latest_context_update(self):
        clock = _Clock()
        run = self.start(_Backend(history=1, retry=True), tasks=range(5), workers=3,
                         clock=clock, capacity=3)
        calls = sorted((run.call(), run.call()), key=lambda call: call.request.task_id)
        calls[0].fail()
        calls[1].succeed()
        third = run.call()
        self.assertEqual(third.request.task_id, 2)
        third.succeed()
        run.no_call()
        self.assertEqual(run.scheduler._next_sequence, 3)
        self.assertTrue(run.output.empty())
        clock.advance(2, run.scheduler)
        retry = run.call()
        self.assertEqual(retry.request.task_id, 0)
        self.assertEqual(retry.request.context, '')
        retry.succeed()
        tail = [run.call(), run.call()]
        self.assertTrue(all(call.request.context == 'context-source-2' for call in tail))
        for call in tail:
            call.succeed()
        self.assert_outputs(run, range(5))

    def test_matured_retries_prioritized_in_source_order(self):
        clock = _Clock()
        run = self.start(_Backend(context=False, retry=True), tasks=range(3), workers=2,
                         clock=clock, eof=False)
        first = sorted((run.call(), run.call()), key=lambda call: call.request.task_id)
        first[1].fail()
        slow = run.call()
        first[0].fail()
        run.no_call()
        clock.advance(2, run.scheduler)
        retry0 = run.call()
        self.assertEqual(retry0.request.task_id, 0)
        retry0.succeed()
        retry1 = run.call()
        self.assertEqual(retry1.request.task_id, 1)
        retry1.succeed()
        slow.succeed()
        run.input.put(None)
        self.assert_outputs(run, range(3))

    def test_retry_can_move_to_another_idle_worker(self):
        clock = _Clock()
        run = self.start(_Backend(context=False, retry=True), tasks=range(3), workers=2, clock=clock)
        first = sorted((run.call(), run.call()), key=lambda call: call.request.task_id)
        first[0].fail()
        replacement = run.call()
        self.assertEqual(replacement.worker, first[0].worker)
        first[1].succeed()
        clock.advance(2, run.scheduler)
        retry = run.call()
        self.assertEqual(retry.request.task_id, 0)
        self.assertNotEqual(retry.worker, first[0].worker)
        retry.succeed()
        replacement.succeed()
        self.assert_outputs(run, range(3))

    def test_timeout_releases_logical_slot_but_not_physical_worker(self):
        clock = _Clock()
        run = self.start(_Backend(history=0, timeout=5, retry=True), tasks=(0, 1), workers=1, clock=clock)
        late = run.call()
        self.assertEqual(late.request.deadline, 105)
        self.assertFalse(hasattr(late.request, 'start_time'))
        self.assertFalse(hasattr(late.request, 'translation'))
        self.assertIsNot(late.request, run.tasks[0])
        clock.advance(5, run.scheduler)
        failed = run.result()
        self.assertTrue(failed.translation_failed)
        run.no_call()
        with self.assertLogs('main', level='INFO') as logs:
            late.succeed('must-not-be-published')
            following = run.call()
        self.assertTrue(any('context gap' in line for line in logs.output))
        self.assertEqual(following.request.context, '')
        self.assertIsNone(failed.translation)
        following.succeed()
        self.assertEqual(run.result().id, 1)
        self.assertIsNone(run.result())
        run.finish()
        self.assertEqual(run.backend.max_active, 1)
        self.assertFalse(run.backend.overlap)

    def test_retry_wait_expires_without_starting_another_attempt(self):
        clock = _Clock()
        run = self.start(_Backend(retry=True, timeout=0.5), tasks=(0,), workers=1, clock=clock)
        run.call().fail()
        run.no_call()
        clock.advance(0.5, run.scheduler)
        result = run.result()
        self.assertTrue(result.translation_failed)
        self.assertIsNone(run.result())
        run.finish()
        self.assertEqual(len(run.backend.all_calls), 1)

    def test_completion_before_deadline_remains_valid_if_dispatcher_delayed(self):
        clock = _Clock()
        run = self.start(_Backend(timeout=5), clock=clock)
        call = run.call()
        reached_expiration = threading.Event()
        resume_dispatcher = threading.Event()
        expire = run.scheduler._expire

        def paused_expire():
            reached_expiration.set()
            if not resume_dispatcher.wait(2):
                raise RuntimeError('Test did not resume the dispatcher')
            return expire()

        run.scheduler._expire = paused_expire
        run.scheduler._wake.set()
        self.assertTrue(reached_expiration.wait(1))
        # The dispatcher has drained its events but has not checked timeouts.
        # Publish now, advance beyond deadline, and resume precisely in that gap.
        call.succeed('valid')
        run.completed(0)
        clock.advance(10, run.scheduler)
        resume_dispatcher.set()
        self.assertFalse(run.result().translation_failed)
        self.assertIsNone(run.result())
        run.finish()

    def test_late_context_cannot_overwrite_valid_newer_context(self):
        clock = _Clock()
        run = self.start(_Backend(history=1, timeout=5), tasks=(0, 1, 2), workers=2, clock=clock)
        first = sorted((run.call(), run.call()), key=lambda call: call.request.task_id)
        first[1].succeed('current')
        third = run.call()
        self.assertEqual(third.request.context, 'current')
        third.succeed('newest')
        run.completed(2)
        clock.advance(5, run.scheduler)
        self.assertTrue(run.result().translation_failed)
        first[0].succeed('expired')
        self.assertEqual(run.result().id, 1)
        self.assertEqual(run.result().id, 2)
        self.assertIsNone(run.result())
        run.finish()
        self.assertEqual(run.scheduler.latest_translation_context, 'newest')

    def test_failure_gap_fallback_is_info_and_preserves_reference(self):
        run = self.start(_Backend(history=0), tasks=(0, 1), workers=3)
        with self.assertLogs('main', level='INFO') as logs:
            run.call().fail()
            second = run.call()
        gap = [record for record in logs.records if 'context gap' in record.message]
        self.assertEqual(len(gap), 1)
        self.assertEqual(gap[0].levelname, 'INFO')
        self.assertEqual(second.request.context, '')
        second.succeed()
        self.assert_outputs(run, (0, 1))

    def test_fallback_uses_existing_context_without_advancing_its_source(self):
        run = self.start(_Backend(history=0), tasks=(0, 1, 2), workers=3)
        run.call().succeed('old-but-usable')
        run.call().fail()
        third = run.call()
        self.assertEqual(third.request.context, 'old-but-usable')
        self.assertEqual(run.scheduler._context_sequence, 0)
        third.succeed()
        self.assert_outputs(run, (0, 1, 2))

    def test_failed_source_stays_in_future_reference(self):
        run = self.start(_Backend(history=1), tasks=(0, 1, 2), workers=1)
        run.call().fail()
        second = run.call()
        self.assertEqual(second.request.references, ('source-0',))
        second.succeed()
        third = run.call()
        self.assertEqual(third.request.references, ('source-1',))
        third.succeed()
        self.assert_outputs(run, (0, 1, 2))

    def test_output_backpressure_is_bounded_and_does_not_block_stop(self):
        run = self.start(_Backend(context=False, automatic=True), tasks=range(50), workers=3,
                         output_size=1, capacity=4)
        for _ in range(5):
            run.call()
        run.no_call()
        self.assertLessEqual(len(run.scheduler._tasks), 4)
        run.scheduler.stop()
        self.assertTrue(run.finished.wait(1))
        self.assertIsNone(run.error)
        self.assertIsNone(run.result())
        self.assertTrue(run.scheduler.output_finished.is_set())
        self.assertTrue(run.output.empty())

    def test_bounded_output_eventually_drains_with_one_eof(self):
        run = self.start(_Backend(automatic=True), tasks=range(30), workers=3, output_size=1, capacity=4)
        self.assert_outputs(run, range(30))

    def test_stop_rejects_late_results_and_closes_client_in_owning_thread(self):
        run = self.start(tasks=(0, 1), workers=1)
        pending = run.call()
        run.scheduler.stop()
        self.assertIsNone(run.result())
        self.assertTrue(run.finished.wait(1))
        pending.succeed('must-not-commit')
        for thread in run.scheduler._threads:
            thread.join(1)
        self.assertIsNone(run.tasks[0].translation)
        self.assertEqual(run.scheduler.latest_translation_context, '')
        self.assertCountEqual(run.backend.session_threads, run.backend.closed_threads)
        self.assertTrue(run.output.empty())

    def test_stop_rejects_an_already_queued_unpublished_completion(self):
        run = self.start()
        call = run.call()
        reached_expiration = threading.Event()
        resume_dispatcher = threading.Event()
        expire = run.scheduler._expire

        def paused_expire():
            reached_expiration.set()
            if not resume_dispatcher.wait(2):
                raise RuntimeError('Test did not resume the dispatcher')
            return expire()

        run.scheduler._expire = paused_expire
        run.scheduler._wake.set()
        self.assertTrue(reached_expiration.wait(1))
        call.succeed('not-yet-published')
        run.completed(0)
        run.scheduler.stop()
        resume_dispatcher.set()
        self.assertIsNone(run.result())
        run.finish()
        self.assertIsNone(run.tasks[0].translation)
        self.assertEqual(run.scheduler.latest_translation_context, '')

    def test_stop_while_waiting_for_retry_does_not_start_it(self):
        clock = _Clock()
        run = self.start(_Backend(history=0, retry=True), tasks=(0, 1), workers=1, clock=clock)
        run.call().fail()
        run.no_call()
        run.scheduler.stop()
        clock.advance(20, run.scheduler)
        self.assertIsNone(run.result())
        run.finish()
        self.assertEqual(len(run.backend.all_calls), 1)

    def test_stop_before_loop_starts_does_not_construct_clients(self):
        backend = _Backend()
        scheduler = ParallelTranslationScheduler(backend, backend.test_settings)
        scheduler.stop()
        output = queue.SimpleQueue()
        scheduler.loop(queue.SimpleQueue(), output)
        self.assertIsNone(output.get_nowait())
        self.assertEqual(backend.session_threads, [])

    def test_worker_initialization_failure_is_reported_and_emits_eof(self):
        backend = _Backend()

        @contextmanager
        def broken_session():
            raise RuntimeError('client initialization failed')
            yield

        backend.open_session = broken_session
        run = self.start(backend, workers=1)
        self.assertTrue(run.finished.wait(2))
        self.assertIsInstance(run.error, RuntimeError)
        self.assertIn('client initialization failed', str(run.error.__cause__))
        self.assertIsNone(run.result())
        self.assertTrue(run.output.empty())

    def test_thread_start_failure_preserves_error_and_closes_started_workers(self):
        backend = _Backend()
        scheduler = ParallelTranslationScheduler(backend, replace(backend.test_settings, workers=3))
        output = queue.Queue(maxsize=1)
        output.put('pending-result')
        original_start = threading.Thread.start

        def fail_second_start(thread):
            if thread.name == 'translation-request-1':
                raise OSError('Cannot start second thread')
            return original_start(thread)

        with mock.patch.object(threading.Thread, 'start', fail_second_start):
            with self.assertRaisesRegex(OSError, 'Cannot start second thread'):
                scheduler.loop(queue.SimpleQueue(), output)
        self.assertEqual(len(scheduler._threads), 1)
        self.assertFalse(scheduler._threads[0].is_alive())
        self.assertCountEqual(backend.session_threads, backend.closed_threads)
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())
        self.assertTrue(scheduler.output_finished.is_set())

    def test_worker_close_failure_is_logged_without_hanging_shutdown(self):
        backend = _Backend(automatic=True)

        @contextmanager
        def broken_close():
            try:
                with _Backend.open_session(backend) as session:
                    yield session
            finally:
                raise RuntimeError('close failed')

        backend.open_session = broken_close
        with self.assertLogs('main', level='ERROR') as logs:
            run = self.start(backend, workers=1)
            self.assert_outputs(run, (0,))
        self.assertTrue(any('shutdown' in line for line in logs.output))

    def test_exception_from_translate_is_task_failure_not_lost_worker(self):
        run = self.start(_Backend(context=False), tasks=(0, 1), workers=1)
        run.call().fail(RuntimeError('network failed'))
        run.call().succeed()
        self.assertTrue(run.result().translation_failed)
        self.assertFalse(run.result().translation_failed)
        self.assertIsNone(run.result())
        run.finish()

    def test_invalid_session_return_is_task_failure_not_lost_worker(self):
        run = self.start(_Backend(context=False), tasks=(0, 1), workers=1)
        first = run.call()
        first.result = 'not-a-TranslationResult'
        first.release.set()
        run.call().succeed()
        self.assertTrue(run.result().translation_failed)
        self.assertFalse(run.result().translation_failed)
        self.assertIsNone(run.result())
        run.finish()

    def test_unexpected_worker_exit_propagates_once_and_does_not_hang(self):
        run = self.start(workers=1)
        run.call().fail(SystemExit('unexpected provider exit'))
        self.assertTrue(run.finished.wait(2))
        self.assertIsInstance(run.error, RuntimeError)
        self.assertIsInstance(run.error.__cause__, SystemExit)
        self.assertIsNone(run.result())
        self.assertTrue(run.output.empty())

    def test_failure_logs_do_not_include_provider_exception_payload(self):
        run = self.start(workers=1)
        with self.assertLogs('main', level='ERROR') as logs:
            run.call().fail(RuntimeError('https://host/?key=SECRET&prompt=PRIVATE'))
            self.assert_outputs(run, (0,))
        output = '\n'.join(logs.output)
        self.assertIn('RuntimeError', output)
        self.assertNotIn('SECRET', output)
        self.assertNotIn('PRIVATE', output)

    def test_retry_requests_are_immutable_and_do_not_share_result_fields(self):
        clock = _Clock()
        run = self.start(_Backend(history=1, retry=True), tasks=(0, 1), workers=1, clock=clock)
        run.call().succeed()
        first = run.call()
        with self.assertRaises(AttributeError):
            first.request.references.append('must-not-leak')
        with self.assertRaises(FrozenInstanceError):
            first.request.context = 'must-not-leak'
        first.fail()
        run.no_call()
        clock.advance(2, run.scheduler)
        retry = run.call()
        self.assertIsNot(retry.request, first.request)
        self.assertEqual(retry.request.references, ('source-0',))
        self.assertEqual(retry.request.context, 'context-source-0')
        self.assertEqual(run.tasks[1].context_transcripts, ['source-0'])
        self.assertIsNone(run.tasks[1].translation)
        retry.succeed()
        self.assert_outputs(run, (0, 1))

    def test_eof_is_signaled_before_consumer_can_read_sentinel(self):
        backend = _Backend(automatic=True)

        class CheckingQueue(queue.SimpleQueue):
            def put_nowait(self, item):
                if item is None and not scheduler.output_finished.is_set():
                    raise AssertionError('EOF state must be visible before EOF is consumable')
                super().put_nowait(item)

        scheduler = ParallelTranslationScheduler(backend, replace(backend.test_settings, workers=1))
        source = queue.SimpleQueue()
        source.put(None)
        output = CheckingQueue()
        scheduler.loop(source, output)
        self.assertIsNone(output.get_nowait())
        self.assertTrue(scheduler.output_finished.is_set())

    def test_deadline_eof_does_not_wait_indefinitely_for_hung_last_request(self):
        clock = _Clock()
        run = self.start(_Backend(timeout=5), clock=clock, workers=1)
        call = run.call()
        clock.advance(5, run.scheduler)
        self.assertTrue(run.result().translation_failed)
        self.assertIsNone(run.result())
        run.finish()
        self.assertTrue(run.scheduler._threads[0].is_alive())
        call.succeed('too-late')
        run.scheduler._threads[0].join(1)
        self.assertFalse(run.scheduler._threads[0].is_alive())
        self.assertIsNone(run.tasks[0].translation)

    def test_missing_context_is_failure_and_cannot_advance_context(self):
        run = self.start()
        call = run.call()
        call.result = TranslationResult('translation-without-context')
        call.release.set()
        self.assertTrue(run.result().translation_failed)
        self.assertEqual(run.scheduler._context_sequence, -1)
        self.assertIsNone(run.result())
        run.finish()

    def test_backend_requires_only_open_session_not_scheduling_or_translate_methods(self):
        backend = _Backend(automatic=True)
        self.assertFalse(hasattr(backend, 'translate'))
        self.assertFalse(hasattr(backend, 'loop'))
        self.assertFalse(hasattr(backend, 'stop'))
        self.assertFalse(hasattr(backend, 'history_size'))
        run = self.start(backend, workers=1)
        self.assert_outputs(run, (0,))

    def test_empty_input_finishes_without_translation_and_closes_all_clients(self):
        run = self.start(tasks=(), workers=3)
        self.assertIsNone(run.result())
        run.finish()
        self.assertEqual(run.backend.all_calls, [])
        self.assertCountEqual(run.backend.session_threads, run.backend.closed_threads)

    def test_dynamic_work_assignment_keeps_fast_worker_busy(self):
        run = self.start(_Backend(context=False), tasks=range(7), workers=3)
        first = [run.call(), run.call(), run.call()]
        fast = first[-1]
        fast_worker = fast.worker
        for _ in range(4):
            fast.succeed()
            fast = run.call()
            self.assertEqual(fast.worker, fast_worker)
        fast.succeed()
        first[0].succeed()
        first[1].succeed()
        self.assert_outputs(run, range(7))
        self.assertEqual(run.backend.max_active, 3)

    def test_randomized_completion_stress_preserves_order_and_capacity(self):
        rng = random.Random(87231)
        for iteration in range(10):
            with self.subTest(iteration=iteration):
                run = self.start(_Backend(context=False), tasks=range(40), workers=4, capacity=8)
                active = [run.call() for _ in range(4)]
                for index in range(40):
                    chosen = active.pop(rng.randrange(len(active)))
                    chosen.succeed()
                    # A slow head can fill the buffer: release another current
                    # request instead of blocking this test on a new dispatch.
                    if index < 36:
                        try:
                            active.append(run.backend.calls.get(timeout=0.01))
                        except queue.Empty:
                            pass
                    if not active and index < 39:
                        active.append(run.call())
                    while len(active) < 4:
                        try:
                            active.append(run.backend.calls.get_nowait())
                        except queue.Empty:
                            break
                self.assert_outputs(run, range(40))
                self.assertLessEqual(run.backend.max_active, 4)
                self.assertFalse(run.backend.overlap)

    def test_randomized_context_retries_keep_snapshots_and_cannot_deadlock(self):
        rng = random.Random(57113)
        for iteration in range(5):
            with self.subTest(iteration=iteration):
                clock = _Clock()
                run = self.start(_Backend(history=2, retry=True), tasks=range(40),
                                 workers=6, capacity=8, clock=clock)
                originals = {}
                finished = set()
                active = []
                while len(finished) < 40:
                    if not active:
                        active.append(run.call())
                    while True:
                        try:
                            active.append(run.backend.calls.get_nowait())
                        except queue.Empty:
                            break
                    call = active.pop(rng.randrange(len(active)))
                    request = call.request
                    self.assertEqual(request.references,
                                     tuple(f'source-{index}' for index in
                                           range(max(0, request.task_id - 2), request.task_id)))
                    if request.task_id % 5 == 0 and request.task_id not in originals:
                        originals[request.task_id] = request
                        call.fail(ConnectionError('retry once'))
                        run.completed(request.task_id)
                        clock.advance(2, run.scheduler)
                    else:
                        if request.task_id in originals:
                            first = originals[request.task_id]
                            self.assertEqual(request.references, first.references)
                            self.assertEqual(request.context, first.context)
                        call.succeed()
                        finished.add(request.task_id)
                self.assert_outputs(run, range(40))
                self.assertLessEqual(run.backend.max_active, 6)
                self.assertFalse(run.backend.overlap)

    def test_invalid_worker_counts_rejected(self):
        for value in (0, -1, 1.2, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ParallelTranslationScheduler(_Backend(), TranslationSettings(workers=value))


if __name__ == '__main__':
    unittest.main()
