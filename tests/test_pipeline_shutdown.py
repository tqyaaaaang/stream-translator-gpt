import contextlib
import queue
import threading
import unittest
from unittest import mock

from stream_translator_gpt import main
from stream_translator_gpt.main import Config


class PipelineShutdownTest(unittest.TestCase):

    @contextlib.contextmanager
    def _pipeline(self, getter, transcriber, translator, exporter):
        config = Config(url='input', use_modulate_transcription_api=True, modulate_api_key='test-key',
                        translation_prompt='Translate.', openai_api_key='test-key')
        with mock.patch.object(main, 'ClientPool'), \
                mock.patch.object(main, '_create_audio_getter', return_value=getter), \
                mock.patch.object(main, 'ModulateStreamingTranscriber', return_value=transcriber), \
                mock.patch.object(main, '_create_translator', return_value=translator), \
                mock.patch.object(main, 'ResultExporter', return_value=exporter):
            yield config

    def test_normal_eof_drains_translation_without_cancelling_it(self):
        class Getter:
            stop = mock.Mock()

            def loop(self, output_queue):
                output_queue.put('source')
                output_queue.put(None)

        class Transcriber:
            def loop(self, input_queue, output_queue):
                while (item := input_queue.get(timeout=2)) is not None:
                    output_queue.put(item)
                output_queue.put(None)

        class Translator:
            stop = mock.Mock()

            def __init__(self):
                self.output_finished = threading.Event()

            def loop(self, input_queue, output_queue):
                while (item := input_queue.get(timeout=2)) is not None:
                    output_queue.put(f'translated {item}')
                self.output_finished.set()
                output_queue.put(None)

        class Exporter:
            def __init__(self):
                self.items = []

            def loop(self, input_queue):
                while (item := input_queue.get(timeout=2)) is not None:
                    self.items.append(item)

        getter, translator, exporter = Getter(), Translator(), Exporter()
        with self._pipeline(getter, Transcriber(), translator, exporter) as config:
            main.run(config)
        self.assertEqual(exporter.items, ['translated source'])
        translator.stop.assert_not_called()

    def test_early_exporter_exit_cancels_translator_even_after_input_eof(self):
        class Getter:
            def loop(self, output_queue):
                output_queue.put(None)

        class Transcriber:
            def loop(self, input_queue, output_queue):
                self.received = input_queue.get(timeout=2)
                output_queue.put(None)

        class WaitingTranslator:
            def __init__(self):
                self.started = threading.Event()
                self.stop_requested = threading.Event()
                self.finished = threading.Event()

            def loop(self, input_queue, output_queue):
                self.started.set()
                try:
                    if not self.stop_requested.wait(timeout=2):
                        raise AssertionError('translator was not cancelled after exporter exit')
                finally:
                    output_queue.put(None)
                    self.finished.set()

            def stop(self):
                self.stop_requested.set()

        translator = WaitingTranslator()

        class EarlyExporter:
            def loop(self, input_queue):
                if not translator.started.wait(timeout=2):
                    raise AssertionError('translator did not start')

        with self._pipeline(Getter(), Transcriber(), translator, EarlyExporter()) as config:
            main.run(config)
        self.assertTrue(translator.stop_requested.is_set())
        self.assertTrue(translator.finished.is_set())

    def test_published_translation_eof_does_not_cancel_client_cleanup(self):
        from types import SimpleNamespace

        output_finished = threading.Event()
        output_finished.set()
        translator = SimpleNamespace(loop=mock.Mock(), stop=mock.Mock(), output_finished=output_finished)
        getter = SimpleNamespace(loop=mock.Mock(), stop=mock.Mock())
        workers = {}

        def start(func, *args, **kwargs):
            worker = mock.Mock(name=kwargs['name'])
            worker.name = kwargs['name']
            # The translator has published EOF, but is still releasing its clients
            # when the exporter exits. Joining it lets that cleanup finish normally.
            worker.is_alive.return_value = kwargs['name'] == 'translator'
            worker.join.side_effect = lambda timeout: setattr(worker.is_alive, 'return_value', False)
            workers[kwargs['name']] = worker
            return worker

        with self._pipeline(getter, mock.Mock(), translator, mock.Mock()) as config, \
                mock.patch.object(main, 'start_daemon_thread', side_effect=start):
            main.run(config)
        translator.stop.assert_not_called()
        getter.stop.assert_not_called()
        workers['translator'].join.assert_called_once()

    def test_main_interrupt_cancels_waiting_translator_and_joins_workers(self):
        for interruption in (KeyboardInterrupt(), SystemExit(130)):
            with self.subTest(interruption=type(interruption).__name__):
                class Getter:
                    def __init__(self):
                        self.stop_requested = threading.Event()
                        self.finished = threading.Event()

                    def loop(self, output_queue):
                        try:
                            if not self.stop_requested.wait(timeout=2):
                                raise AssertionError('getter was not cancelled')
                        finally:
                            output_queue.put(None)
                            self.finished.set()

                    def stop(self):
                        self.stop_requested.set()

                class Transcriber:
                    def loop(self, input_queue, output_queue):
                        input_queue.get(timeout=2)
                        output_queue.put(None)

                class Translator:
                    def __init__(self):
                        self.stop_requested = threading.Event()
                        self.finished = threading.Event()

                    def loop(self, input_queue, output_queue):
                        try:
                            if not self.stop_requested.wait(timeout=2):
                                raise AssertionError('translator was not cancelled')
                        finally:
                            output_queue.put(None)
                            self.finished.set()

                    def stop(self):
                        self.stop_requested.set()

                class Exporter:
                    def loop(self, input_queue):
                        input_queue.get(timeout=2)

                getter, translator = Getter(), Translator()
                with self._pipeline(getter, Transcriber(), translator, Exporter()) as config, \
                        mock.patch.object(main.time, 'sleep', side_effect=interruption), \
                        self.assertRaises(type(interruption)) as raised:
                    main.run(config)
                self.assertIs(raised.exception, interruption)
                self.assertTrue(getter.stop_requested.is_set())
                self.assertTrue(getter.finished.is_set())
                self.assertTrue(translator.stop_requested.is_set())
                self.assertTrue(translator.finished.is_set())

    def test_transcriber_failure_cancels_context_wait_and_translator_owns_one_eof(self):
        class Getter:
            def loop(self, output_queue):
                output_queue.put(None)

        class FailingTranscriber:
            def loop(self, input_queue, output_queue):
                raise RuntimeError('transcriber failed')

        class Translator:
            def __init__(self):
                self.stop_requested = threading.Event()

            def stop(self):
                self.stop_requested.set()

            def loop(self, input_queue, output_queue):
                try:
                    if not self.stop_requested.wait(timeout=2):
                        raise AssertionError('translator did not receive worker-failure cancellation')
                finally:
                    output_queue.put(None)

        class Exporter:
            def loop(self, input_queue):
                self.queue = input_queue
                self.received = input_queue.get(timeout=2)

        translator, exporter = Translator(), Exporter()
        with self._pipeline(Getter(), FailingTranscriber(), translator, exporter) as config, \
                self.assertRaisesRegex(RuntimeError, 'Pipeline stopped because a worker failed'):
            main.run(config)
        self.assertTrue(translator.stop_requested.is_set())
        self.assertIsNone(exporter.received)
        with self.assertRaises(queue.Empty):
            exporter.queue.get_nowait()

    def test_worker_start_failure_stops_and_joins_already_started_input(self):
        class Getter:
            def __init__(self):
                self.stop_requested = threading.Event()
                self.finished = threading.Event()

            def loop(self, output_queue):
                try:
                    if not self.stop_requested.wait(timeout=2):
                        raise AssertionError('getter did not stop after another worker failed to start')
                finally:
                    output_queue.put(None)
                    self.finished.set()

            def stop(self):
                self.stop_requested.set()

        getter = Getter()
        translator = mock.Mock()
        actual_start = main.start_daemon_thread

        def start(func, *args, **kwargs):
            if kwargs['name'] == 'getter':
                return actual_start(func, *args, **kwargs)
            raise RuntimeError('could not start worker')

        with self._pipeline(getter, mock.Mock(), translator, mock.Mock()) as config, \
                mock.patch.object(main, 'start_daemon_thread', side_effect=start), \
                self.assertRaisesRegex(RuntimeError, 'could not start worker'):
            main.run(config)
        self.assertTrue(getter.stop_requested.is_set())
        self.assertTrue(getter.finished.is_set())
        translator.stop.assert_called_once()

    def test_backend_session_failure_unblocks_exporter_with_exactly_one_eof(self):
        from stream_translator_gpt.utils.common import TranslationTask
        from stream_translator_gpt.translator.pipeline import TranslationPipeline
        from stream_translator_gpt.translator.types import TranslationSettings

        class BrokenBackend:
            @contextlib.contextmanager
            def open_session(self):
                raise RuntimeError('backend session construction failed')
                yield  # Make this a context manager without constructing any API client.

        class Getter:
            def loop(self, output_queue):
                output_queue.put(None)

        class Transcriber:
            def loop(self, input_queue, output_queue):
                input_queue.get(timeout=2)
                task = TranslationTask(1, None, (0.0, 1.0))
                task.transcript = '原文'
                output_queue.put(task)
                output_queue.put(None)

        class Exporter:
            def __init__(self):
                self.received = object()
                self.finished = threading.Event()

            def loop(self, input_queue):
                self.queue = input_queue
                try:
                    # Bound the test itself: missing EOF must fail rather than
                    # leave the suite hanging in the main exporter-wait loop.
                    self.received = input_queue.get(timeout=2)
                finally:
                    self.finished.set()

        translator = TranslationPipeline(BrokenBackend(), TranslationSettings(workers=1))
        exporter = Exporter()
        with self._pipeline(Getter(), Transcriber(), translator, exporter) as config, \
                self.assertRaisesRegex(RuntimeError, 'Pipeline stopped because a worker failed'):
            main.run(config)
        self.assertTrue(exporter.finished.is_set())
        self.assertIsNone(exporter.received)
        self.assertTrue(translator.output_finished.is_set())
        with self.assertRaises(queue.Empty):
            exporter.queue.get_nowait()

    def test_downstream_worker_failure_stops_and_joins_audio_getter(self):
        class BlockingGetter:
            def __init__(self):
                self.started = threading.Event()
                self.stop_requested = threading.Event()
                self.exited = threading.Event()

            def loop(self, output_queue):
                self.started.set()
                self.stop_requested.wait(timeout=3)
                self.exited.set()

            def stop(self):
                self.stop_requested.set()

        class FailingTranscriber:
            def __init__(self, getter):
                self.getter = getter

            def loop(self, input_queue, output_queue):
                if not self.getter.started.wait(timeout=1):
                    raise AssertionError('audio getter did not start')
                raise RuntimeError('transcriber failed')

        class SentinelExporter:
            def __init__(self):
                self.received = object()

            def loop(self, input_queue):
                self.received = input_queue.get()

        getter = BlockingGetter()
        transcriber = FailingTranscriber(getter)
        exporter = SentinelExporter()
        config = Config(url='input', use_modulate_transcription_api=True, modulate_api_key='key')

        with mock.patch.object(main, 'ClientPool') as client_pool, \
                mock.patch.object(main, '_create_audio_getter', return_value=getter), \
                mock.patch.object(main, 'ModulateStreamingTranscriber', return_value=transcriber), \
                mock.patch.object(main, 'ResultExporter', return_value=exporter):
            with self.assertRaisesRegex(RuntimeError, 'Pipeline stopped because a worker failed'):
                main.run(config)

        client_pool.init.assert_called_once()
        self.assertIsNone(exporter.received)
        self.assertTrue(getter.stop_requested.is_set())
        self.assertTrue(getter.exited.is_set())

    def test_translator_failure_stops_getter_and_unblocks_transcriber_priming(self):
        class PrimingGetter:
            def __init__(self):
                self.started = threading.Event()
                self.stop_requested = threading.Event()
                self.exited = threading.Event()

            def loop(self, output_queue):
                self.started.set()
                self.stop_requested.wait(timeout=3)
                output_queue.put(None)
                self.exited.set()

            def stop(self):
                self.stop_requested.set()

        class PrimingTranscriber:
            def __init__(self):
                self.waiting = threading.Event()
                self.unblocked = threading.Event()
                self.received = object()

            def loop(self, input_queue, output_queue):
                self.waiting.set()
                self.received = input_queue.get()
                self.unblocked.set()
                output_queue.put(None)

        class FailingTranslator:
            def __init__(self, getter, transcriber):
                self.getter = getter
                self.transcriber = transcriber
                self.stop_requested = threading.Event()

            def stop(self):
                self.stop_requested.set()

            def loop(self, input_queue, output_queue):
                try:
                    if not self.getter.started.wait(timeout=1):
                        raise AssertionError('audio getter did not start')
                    if not self.transcriber.waiting.wait(timeout=1):
                        raise AssertionError('transcriber did not begin priming')
                    raise RuntimeError('translator failed')
                finally:
                    # Translators own EOF on exceptional as well as normal exits.
                    output_queue.put(None)

        class SentinelExporter:
            def __init__(self):
                self.received = object()

            def loop(self, input_queue):
                self.received = input_queue.get()

        getter = PrimingGetter()
        transcriber = PrimingTranscriber()
        translator = FailingTranslator(getter, transcriber)
        exporter = SentinelExporter()
        config = Config(url='input',
                        use_modulate_transcription_api=True,
                        modulate_api_key='key',
                        openai_api_key='key',
                        translation_prompt='translate')

        with mock.patch.object(main, 'ClientPool') as client_pool, \
                mock.patch.object(main, '_create_audio_getter', return_value=getter), \
                mock.patch.object(main, 'ModulateStreamingTranscriber', return_value=transcriber), \
                mock.patch.object(main, '_create_translator', return_value=translator), \
                mock.patch.object(main, 'ResultExporter', return_value=exporter):
            with self.assertRaisesRegex(RuntimeError, 'Pipeline stopped because a worker failed'):
                main.run(config)

        client_pool.init.assert_called_once()
        self.assertIsNone(exporter.received)
        self.assertTrue(getter.stop_requested.is_set())
        self.assertTrue(getter.exited.is_set())
        self.assertTrue(transcriber.unblocked.is_set())
        self.assertIsNone(transcriber.received)
        self.assertTrue(translator.stop_requested.is_set())


if __name__ == '__main__':
    unittest.main()
