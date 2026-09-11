import io
import logging
import unittest
from unittest import mock

import numpy as np

from stream_translator_gpt.slicer.audio_slicer import AudioSlicer
from stream_translator_gpt.utils.common import LogTime, TranslationTask
from stream_translator_gpt.utils.logging_utils import DDEBUG, configure_logging, shutdown_logging


class TimingTest(unittest.TestCase):

    def tearDown(self):
        shutdown_logging()

    def test_log_time_uses_perf_counter_and_lazy_log_arguments(self):
        output = io.StringIO()
        configure_logging(1, stream=output)

        with mock.patch('stream_translator_gpt.utils.common.time.perf_counter', side_effect=[10.0, 10.125]):
            with LogTime('Processed task %d', 4):
                pass
        shutdown_logging()

        self.assertIn('Processed task 4, run in time 125.00ms', output.getvalue())

    def test_log_time_skips_clock_when_level_is_disabled(self):
        configure_logging(0, stream=io.StringIO())

        with mock.patch('stream_translator_gpt.utils.common.time.perf_counter') as perf_counter:
            with LogTime('hidden', level=logging.INFO):
                pass

        perf_counter.assert_not_called()

    def test_vad_timing_uses_ddebug(self):
        slicer = AudioSlicer.__new__(AudioSlicer)
        slicer.counter = 0
        slicer.vad = mock.Mock()
        slicer.vad.get_speech_prob.return_value = 0.9
        slicer.vad_threshold = 0.35
        slicer.vad_neg_threshold = 0.15
        slicer.speech_count = 0
        slicer.no_speech_count = 0
        slicer.continuous_no_speech_count = 0
        slicer.audio_buffer = []
        slicer.dynamic_vad_threshold = False
        timer = mock.MagicMock()
        timer.__enter__.return_value = timer

        with mock.patch('stream_translator_gpt.slicer.audio_slicer.LogTime', return_value=timer) as log_time:
            slicer.put(np.zeros(512, dtype=np.float32))

        log_time.assert_called_once_with('', level=DDEBUG)
        timer.set_log.assert_called_once_with('VAD evaluated block %d with speech probability %.4f', 1, 0.9)

    def test_translation_wrapper_logs_task_id(self):
        from stream_translator_gpt.translator.legacy.legacy_translator import LLMTranslator

        class FakeTranslator(LLMTranslator):

            def translate(self, translation_task):
                translation_task.translation = 'done'

        translator = FakeTranslator.__new__(FakeTranslator)
        task = TranslationTask(9, np.empty(0), (0.0, 1.0))
        timer = mock.MagicMock()
        timer.__enter__.return_value = timer

        with mock.patch('stream_translator_gpt.translator.legacy.legacy_translator.LogTime', return_value=timer) as log_time:
            translator._translate_with_timing(task)

        log_time.assert_called_once_with('Translated task %d', 9)
        self.assertEqual(task.translation, 'done')


if __name__ == '__main__':
    unittest.main()
