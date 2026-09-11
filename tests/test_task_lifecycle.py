import queue
import unittest
from collections import deque
from unittest import mock

import numpy as np

from stream_translator_gpt.slicer.audio_slicer import AudioSlicer
from stream_translator_gpt.transcriber.audio_transcriber import AudioTranscriber
from stream_translator_gpt.utils.common import TranslationTask
from stream_translator_gpt.translator.legacy.legacy_translator import LLMTranslator


class _FakeTranscriber(AudioTranscriber):

    def transcribe(self, audio, history_context=None):
        return 'hello', [1, 2]


class _FakeTranslator(LLMTranslator):

    def translate(self, translation_task):
        pass


class TaskLifecycleTest(unittest.TestCase):

    def test_task_starts_in_sliced_phase(self):
        task = TranslationTask(7, np.zeros(512, dtype=np.float32), (1.0, 2.0))

        self.assertEqual(task.id, 7)
        self.assertIs(task.phase, TranslationTask.Phase.SLICED)

    def test_slicer_assigns_monotonically_increasing_ids(self):
        slicer = AudioSlicer.__new__(AudioSlicer)
        slicer.counter = 0
        slicer.slice_count = 0
        slicer.vad = None
        slicer.put = lambda audio: setattr(slicer, 'counter', slicer.counter + 1)
        slicer.should_slice = lambda: True
        slicer.slice = lambda: (np.zeros(512, dtype=np.float32), (slicer.counter - 1.0, float(slicer.counter)))
        input_queue = queue.SimpleQueue()
        output_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(None)

        slicer.loop(input_queue, output_queue)

        first = output_queue.get_nowait()
        second = output_queue.get_nowait()
        self.assertEqual((first.id, second.id), (1, 2))
        self.assertIs(first.phase, TranslationTask.Phase.SLICED)
        self.assertIsNone(output_queue.get_nowait())

    def test_transcriber_advances_task_phase(self):
        transcriber = _FakeTranscriber(language='en',
                                       transcription_filters='',
                                       language_filters=None,
                                       print_result=False,
                                       output_timestamps=False,
                                       use_history_context=False,
                                       transcription_keywords=None)
        task = TranslationTask(1, np.zeros(512, dtype=np.float32), (0.0, 1.0))
        input_queue = queue.SimpleQueue()
        output_queue = queue.SimpleQueue()
        input_queue.put(task)
        input_queue.put(None)

        transcriber.loop(input_queue, output_queue)

        self.assertIs(output_queue.get_nowait().phase, TranslationTask.Phase.TRANSCRIBED)
        self.assertIsNone(output_queue.get_nowait())

    def test_legacy_translator_advances_success_and_failure_to_translated_phase(self):
        translator = _FakeTranslator(model='test', prompt='Translate', history_size=0,
                                     use_json_result=False, timeout=0, retry_if_translation_fails=False)
        successful = TranslationTask(1, np.empty(0), (0.0, 1.0))
        successful.translation = 'translated'
        failed = TranslationTask(2, np.empty(0), (1.0, 2.0))
        failed.translation_failed = True
        translator.processing_queue = deque([successful, failed])

        with mock.patch('stream_translator_gpt.translator.legacy.legacy_translator.logger.error'):
            results = translator._get_results()

        self.assertEqual(results, [successful, failed])
        self.assertTrue(all(task.phase is TranslationTask.Phase.TRANSLATED for task in results))


if __name__ == '__main__':
    unittest.main()
