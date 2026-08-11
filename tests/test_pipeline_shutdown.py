import threading
import unittest
from unittest import mock

from stream_translator_gpt import main
from stream_translator_gpt.main import Config


class PipelineShutdownTest(unittest.TestCase):

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

            def loop(self, input_queue, output_queue):
                if not self.getter.started.wait(timeout=1):
                    raise AssertionError('audio getter did not start')
                if not self.transcriber.waiting.wait(timeout=1):
                    raise AssertionError('transcriber did not begin priming')
                raise RuntimeError('translator failed')

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
                mock.patch.object(main, 'GPTTranslator', return_value=translator), \
                mock.patch.object(main, 'ResultExporter', return_value=exporter):
            with self.assertRaisesRegex(RuntimeError, 'Pipeline stopped because a worker failed'):
                main.run(config)

        client_pool.init.assert_called_once()
        self.assertIsNone(exporter.received)
        self.assertTrue(getter.stop_requested.is_set())
        self.assertTrue(getter.exited.is_set())
        self.assertTrue(transcriber.unblocked.is_set())
        self.assertIsNone(transcriber.received)


if __name__ == '__main__':
    unittest.main()
