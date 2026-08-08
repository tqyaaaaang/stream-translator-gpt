import io
import queue
import subprocess
import unittest
from unittest import mock

import numpy as np

from stream_translator_gpt.audio_getter import _open_stream, _read_ffmpeg_loop, _transport, _wait_for_process
from stream_translator_gpt.logging_utils import configure_logging, shutdown_logging


class _SequenceReader:

    def __init__(self, values):
        self.values = iter(values)

    def read(self, size):
        return next(self.values)


class StreamDiagnosticsTest(unittest.TestCase):

    def tearDown(self):
        shutdown_logging()

    def test_ffmpeg_reader_reports_short_block_without_logging_audio_bytes(self):
        output = io.StringIO()
        configure_logging(0, stream=output)
        full_block = np.arange(4, dtype=np.float32).tobytes()
        short_block = b'secret-audio'
        process = mock.Mock(stdout=_SequenceReader([full_block, short_block, b'']))
        output_queue = queue.SimpleQueue()

        block_count = _read_ffmpeg_loop(process, len(full_block), output_queue, source_name='test ffmpeg')
        shutdown_logging()

        self.assertEqual(block_count, 1)
        np.testing.assert_array_equal(output_queue.get_nowait(), np.arange(4, dtype=np.float32))
        self.assertIn('expected 16 bytes, received 12 bytes', output.getvalue())
        self.assertNotIn('secret-audio', output.getvalue())

    def test_transport_closes_ffmpeg_input_after_ytdlp_eof(self):
        ytdlp = mock.Mock()
        ytdlp.stdout = _SequenceReader([b'audio', b''])
        ytdlp.wait.return_value = 0
        ytdlp.poll.return_value = 0
        ffmpeg = mock.Mock()

        _transport(ytdlp, ffmpeg)

        ffmpeg.stdin.write.assert_called_once_with(b'audio')
        ffmpeg.stdin.close.assert_called_once_with()
        ytdlp.terminate.assert_not_called()
        ytdlp.kill.assert_not_called()

    def test_transport_terminates_ytdlp_when_ffmpeg_pipe_breaks(self):
        output = io.StringIO()
        configure_logging(0, stream=output)
        ytdlp = mock.Mock()
        ytdlp.stdout = _SequenceReader([b'audio'])
        ytdlp.wait.return_value = 0
        ytdlp.poll.return_value = None
        ffmpeg = mock.Mock()
        ffmpeg.stdin.write.side_effect = BrokenPipeError

        _transport(ytdlp, ffmpeg)
        shutdown_logging()

        ytdlp.terminate.assert_called_once_with()
        ffmpeg.stdin.close.assert_called_once_with()
        self.assertIn('ffmpeg closed its input pipe before yt-dlp finished', output.getvalue())

    def test_nonzero_process_exit_is_reported(self):
        output = io.StringIO()
        configure_logging(0, stream=output)
        process = mock.Mock()
        process.wait.return_value = 23

        return_code = _wait_for_process(process, 'yt-dlp')
        shutdown_logging()

        self.assertEqual(return_code, 23)
        self.assertIn('yt-dlp exited with status 23', output.getvalue())

    def test_hung_process_is_killed_and_reaped(self):
        output = io.StringIO()
        configure_logging(0, stream=output)
        process = mock.Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired('ffmpeg', 0.1), -9]

        return_code = _wait_for_process(process, 'ffmpeg', timeout=0.1)
        shutdown_logging()

        self.assertEqual(return_code, -9)
        process.kill.assert_called_once_with()
        self.assertIn('did not exit within 0.1s', output.getvalue())

    def test_ffmpeg_start_failure_reaps_started_ytdlp(self):
        ytdlp = mock.Mock()
        ytdlp.poll.return_value = None
        ytdlp.wait.return_value = -15

        with mock.patch('stream_translator_gpt.audio_getter.subprocess.Popen', return_value=ytdlp):
            with mock.patch('stream_translator_gpt.audio_getter.ffmpeg.input',
                            side_effect=RuntimeError('start failed')):
                with self.assertRaisesRegex(RuntimeError, 'start failed'):
                    _open_stream('https://example.com/stream', 'ba', None, None, 'workdir')

        ytdlp.terminate.assert_called_once_with()
        ytdlp.wait.assert_called_once_with(timeout=5.0)


if __name__ == '__main__':
    unittest.main()
