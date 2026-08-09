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


if __name__ == '__main__':
    unittest.main()
