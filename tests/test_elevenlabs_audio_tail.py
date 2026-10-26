import io
import queue
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from stream_translator_gpt.getter.audio_getter import LocalFileAudioGetter, StreamAudioGetter, _read_ffmpeg_loop
from stream_translator_gpt.utils.common import SAMPLES_PER_FRAME


GETTER_MODULE = 'stream_translator_gpt.getter.audio_getter'


class PreservedFinalAudioBlockTest(unittest.TestCase):

    def setUp(self):
        self.full = np.arange(SAMPLES_PER_FRAME, dtype=np.float32)
        self.tail = np.arange(17, dtype=np.float32) + 1000

    def _process(self, payload=None):
        process = mock.Mock(stdout=io.BytesIO(payload if payload is not None
                                            else self.full.tobytes() + self.tail.tobytes()))
        process.wait.return_value = process.poll.return_value = 0
        return process

    def _assert_output(self, output, expected_tail):
        np.testing.assert_array_equal(output.get_nowait(), self.full)
        np.testing.assert_array_equal(output.get_nowait(), expected_tail)
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())

    def test_preserved_tail_takes_precedence_over_padding_and_paces_valid_samples(self):
        for padding in (False, True):
            with self.subTest(padding=padding):
                output = queue.SimpleQueue()
                pacer = mock.Mock()
                pacer.wait_for_block.return_value = True
                count = _read_ffmpeg_loop(self._process(), self.full.nbytes, output,
                                         pad_final_block=padding, preserve_final_block=True, pacer=pacer)
                self.assertEqual(count, 2)
                np.testing.assert_array_equal(output.get_nowait(), self.full)
                np.testing.assert_array_equal(output.get_nowait(), self.tail)
                self.assertTrue(output.empty())
                self.assertEqual(pacer.wait_for_block.call_args_list,
                                 [mock.call(SAMPLES_PER_FRAME), mock.call(len(self.tail))])

    def test_disabled_preservation_keeps_legacy_padding_and_discard_behavior(self):
        for padding in (False, True):
            with self.subTest(padding=padding):
                output = queue.SimpleQueue()
                with mock.patch(f'{GETTER_MODULE}.logger.warning'):
                    count = _read_ffmpeg_loop(self._process(), self.full.nbytes, output,
                                             pad_final_block=padding, preserve_final_block=False)
                np.testing.assert_array_equal(output.get_nowait(), self.full)
                self.assertEqual(count, 2 if padding else 1)
                if padding:
                    np.testing.assert_array_equal(output.get_nowait(),
                                                  np.pad(self.tail, (0, SAMPLES_PER_FRAME - len(self.tail))))
                self.assertTrue(output.empty())

    def test_malformed_partial_float_is_still_discarded(self):
        output = queue.SimpleQueue()
        process = self._process(self.full.tobytes() + self.tail.tobytes() + b'\x01')
        with self.assertLogs('main', level='WARNING'):
            count = _read_ffmpeg_loop(process, self.full.nbytes, output, preserve_final_block=True)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(output.get_nowait(), self.full)
        self.assertTrue(output.empty())

    def test_local_getter_preserves_tail_in_unlimited_realtime_and_ranged_modes(self):
        for options in ({}, {'realtime': True}, {'ss': 0, 'to': 1}):
            with self.subTest(options=options):
                getter = LocalFileAudioGetter('clip.wav', preserve_final_block=True, **options)
                builder = mock.Mock()
                builder.output.return_value.run_async.return_value = self._process()
                output = queue.SimpleQueue()
                with mock.patch(f'{GETTER_MODULE}.ffmpeg.input', return_value=builder), \
                        mock.patch(f'{GETTER_MODULE}._RealtimeAudioPacer') as pacer:
                    pacer.return_value.wait_for_block.return_value = True
                    getter.loop(output)
                self._assert_output(output, self.tail)

    def test_stream_getter_preserves_tail_in_unlimited_realtime_and_ranged_modes(self):
        for options in ({}, {'realtime': True}, {'ss': 0, 'to': 1}):
            with self.subTest(options=options):
                getter = StreamAudioGetter('https://example.test/stream', format='bestaudio', cookies=None,
                                           proxy=None, preserve_final_block=True, **options)
                process = self._process()
                transport = mock.Mock()
                transport.is_alive.return_value = False
                stats = SimpleNamespace(started_at=time.monotonic(), ytdlp_return_code=0,
                                        transport_error=None, media_bytes=1)
                output = queue.SimpleQueue()
                with mock.patch(f'{GETTER_MODULE}._open_stream',
                                return_value=(process, mock.Mock(), transport, stats)), \
                        mock.patch(f'{GETTER_MODULE}._RealtimeAudioPacer') as pacer:
                    pacer.return_value.wait_for_block.return_value = True
                    getter.loop(output)
                self._assert_output(output, self.tail)


if __name__ == '__main__':
    unittest.main()
