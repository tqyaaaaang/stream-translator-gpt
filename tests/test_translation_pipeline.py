import queue
import threading
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from unittest import mock

from stream_translator_gpt.utils.common import LoopWorkerBase, TranslationTask
from stream_translator_gpt.translator.pipeline import TranslationPipeline
from stream_translator_gpt.translator.types import TranslationRequest, TranslationResult, TranslationSettings


class _Backend:
    """Deliberately no scheduler properties or direct translate method."""

    def __init__(self):
        self.requests = queue.Queue()
        self.sessions = queue.Queue()
        self.closed = queue.Queue()

    @contextmanager
    def open_session(self):
        backend = self
        owner = threading.get_ident()

        class Session:
            def translate(self, request):
                if threading.get_ident() != owner:
                    raise AssertionError('Session crossed request worker threads')
                if not isinstance(request, TranslationRequest):
                    raise AssertionError('Mutable pipeline task crossed backend boundary')
                backend.requests.put(request)
                return TranslationResult(f'translated-{request.transcript}', f'context-{request.task_id}')

        session = Session()
        self.sessions.put((owner, session))
        try:
            yield session
        finally:
            self.closed.put((threading.get_ident(), session))


class TranslationPipelineTest(unittest.TestCase):
    def test_pipeline_composes_backend_and_scheduler_without_translation_method(self):
        backend = _Backend()
        settings = TranslationSettings(workers=3, use_translation_context=True)
        pipeline = TranslationPipeline(backend, settings)
        self.assertIsInstance(pipeline, LoopWorkerBase)
        self.assertIs(pipeline.scheduler.backend, backend)
        self.assertIs(pipeline.scheduler.settings, settings)
        self.assertIs(pipeline.scheduler.output_finished, pipeline.output_finished)
        for name in ('translate', 'open_session', 'worker_session', 'use_old_translator'):
            self.assertFalse(hasattr(pipeline, name), name)
        self.assertFalse(hasattr(pipeline.scheduler, 'translator'))

        source, output = queue.SimpleQueue(), queue.SimpleQueue()
        original = []
        for index in (10, 3, 81, 7, 900, 6):
            task = TranslationTask(index, b'audio-not-for-backend', (index, index + 1))
            task.transcript = f'source-{index}'
            original.append(task)
            source.put(task)
        source.put(None)
        owner = threading.get_ident()
        pipeline.loop(source, output)

        self.assertTrue(pipeline.output_finished.is_set())
        for task in original:
            self.assertIs(output.get_nowait(), task)
            self.assertEqual(task.translation, f'translated-{task.transcript}')
            self.assertFalse(task.translation_failed)
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())
        sessions = [backend.sessions.get_nowait() for _ in range(3)]
        closed = [backend.closed.get_nowait() for _ in range(3)]
        self.assertCountEqual(sessions, closed)
        self.assertEqual(len({id(session) for _, session in sessions}), 3)
        self.assertTrue(all(thread != owner for thread, _ in sessions))
        requests = sorted([backend.requests.get_nowait() for _ in original],
                          key=lambda request: [task.id for task in original].index(request.task_id))
        for index, request in enumerate(requests):
            self.assertEqual(request.references, tuple(task.transcript for task in original[max(0, index - 3):index]))
            self.assertFalse(hasattr(request, 'audio'))
            self.assertFalse(hasattr(request, '_translation_lock'))
            self.assertFalse(hasattr(request, 'translation'))

    def test_stop_before_run_is_idempotent_and_creates_no_sessions(self):
        backend = _Backend()
        pipeline = TranslationPipeline(backend, TranslationSettings())
        pipeline.stop()
        pipeline.stop()
        output = queue.SimpleQueue()
        pipeline.loop(queue.SimpleQueue(), output)
        self.assertTrue(pipeline.output_finished.is_set())
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())
        self.assertTrue(backend.sessions.empty())

    def test_pipeline_cannot_run_twice(self):
        pipeline = TranslationPipeline(_Backend(), TranslationSettings())
        source, output = queue.SimpleQueue(), queue.SimpleQueue()
        source.put(None)
        pipeline.loop(source, output)
        with self.assertRaisesRegex(RuntimeError, 'only be run once'):
            pipeline.loop(source, output)
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())

    def test_simultaneous_loop_calls_have_exactly_one_dispatcher(self):
        for _ in range(20):
            pipeline = TranslationPipeline(_Backend(), TranslationSettings())
            source, output = queue.SimpleQueue(), queue.SimpleQueue()
            source.put(None)
            start = threading.Barrier(3)
            results = queue.SimpleQueue()

            def run():
                start.wait(timeout=2)
                try:
                    pipeline.loop(source, output)
                except RuntimeError as error:
                    results.put(error)
                else:
                    results.put(None)

            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads:
                thread.start()
            start.wait(timeout=2)
            for thread in threads:
                thread.join(2)
                self.assertFalse(thread.is_alive(), 'Competing dispatcher loops did not finish')
            outcomes = [results.get_nowait(), results.get_nowait()]
            self.assertEqual(sum(value is None for value in outcomes), 1)
            self.assertEqual(sum(isinstance(value, RuntimeError) for value in outcomes), 1)
            self.assertIsNone(output.get_nowait())
            self.assertTrue(output.empty())

    def test_only_context_thread_limit_emits_one_warning(self):
        for context, history, workers, expected in (
                (False, 0, 10, False), (True, 2, 3, False), (True, 2, 4, True),
                (True, 0, 1, False), (True, 0, 3, True)):
            with self.subTest(context=context, history=history, workers=workers):
                with mock.patch('stream_translator_gpt.translator.pipeline.logger.warning') as warning:
                    pipeline = TranslationPipeline(_Backend(), TranslationSettings(
                        workers=workers, history_size=history, use_translation_context=context))
                self.assertEqual(warning.call_count, int(expected))
                self.assertEqual(pipeline.scheduler.workers, workers)

    def test_pipeline_construction_and_stop_are_safe_concurrently(self):
        for _ in range(20):
            backend = _Backend()
            pipeline = TranslationPipeline(backend, TranslationSettings())
            source, output = queue.SimpleQueue(), queue.SimpleQueue()
            start = threading.Barrier(5)

            def run():
                start.wait(timeout=2)
                pipeline.loop(source, output)

            def stop():
                start.wait(timeout=2)
                pipeline.stop()

            threads = [threading.Thread(target=run)] + [threading.Thread(target=stop) for _ in range(3)]
            for thread in threads:
                thread.start()
            start.wait(timeout=2)
            for thread in threads:
                thread.join(2)
                self.assertFalse(thread.is_alive(), 'Concurrent stop deadlocked')
            self.assertIsNone(output.get_nowait())
            self.assertTrue(output.empty())
            self.assertTrue(pipeline.output_finished.is_set())
            self.assertTrue(all(not thread.is_alive() for thread in pipeline.scheduler._threads))


class TranslationContractsTest(unittest.TestCase):
    def test_request_result_and_settings_are_frozen(self):
        for instance, field, value in (
                (TranslationRequest(1, 'source', ('previous',), 'context'), 'context', 'changed'),
                (TranslationResult('translation', 'context'), 'translation', 'changed'),
                (TranslationSettings(), 'workers', 4)):
            with self.subTest(type=type(instance).__name__), self.assertRaises(FrozenInstanceError):
                setattr(instance, field, value)

    def test_invalid_settings_are_rejected(self):
        for name, invalid in (
                ('workers', (0, -1, 1.1, True, '3', None)),
                ('history_size', (-1, 1.2, True, '2', None)),
                ('timeout', (-1, float('inf'), float('-inf'), float('nan'), True, '10', None))):
            for value in invalid:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TranslationSettings(**{name: value})

    def test_zero_history_and_zero_timeout_are_supported(self):
        settings = TranslationSettings(workers=1, history_size=0, timeout=0)
        self.assertEqual(settings.history_size, 0)
        self.assertEqual(settings.timeout, 0)


if __name__ == '__main__':
    unittest.main()
