import io
import os
import queue
import signal
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stream_translator_gpt.audio_getter import (
    StreamAudioGetter,
    _log_stream_tool_versions,
    _log_stream_format_diagnostic,
    _kill_process,
    _open_stream,
    _read_ffmpeg_loop,
    _StreamTransportStats,
    _transport,
    _wait_for_process,
)
from stream_translator_gpt.logging_utils import configure_logging, shutdown_logging


class _SequenceReader:

    def __init__(self, values):
        self.values = iter(values)
        self.closed = False

    def read(self, size):
        return next(self.values)

    def close(self):
        self.closed = True


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
        ffmpeg.stdin.flush.assert_called_once_with()
        ffmpeg.stdin.close.assert_called_once_with()
        ytdlp.terminate.assert_not_called()
        ytdlp.kill.assert_not_called()
        self.assertTrue(ytdlp.stdout.closed)

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

        ytdlp.kill.assert_called_once_with()
        ffmpeg.stdin.close.assert_called_once_with()
        self.assertTrue(ytdlp.stdout.closed)
        self.assertIn('ffmpeg closed its input pipe before yt-dlp finished', output.getvalue())

    def test_transport_logs_first_media_timing_and_totals_without_media_contents(self):
        output = io.StringIO()
        configure_logging(2, stream=output)
        ytdlp = mock.Mock(stdout=_SequenceReader([b'secret-media', b'']))
        ytdlp.wait.return_value = 0
        ytdlp.poll.return_value = 0
        ffmpeg = mock.Mock()
        stats = _StreamTransportStats(started_at=10.0)

        with mock.patch('stream_translator_gpt.audio_getter.time.monotonic', side_effect=[12.5, 14.0]):
            result = _transport(ytdlp, ffmpeg, stats)
        shutdown_logging()

        self.assertIs(result, stats)
        self.assertEqual(stats.first_media_elapsed, 2.5)
        self.assertEqual(stats.media_bytes, len(b'secret-media'))
        self.assertEqual(stats.media_chunks, 1)
        self.assertEqual(stats.ytdlp_return_code, 0)
        self.assertIn('first media bytes after 2.500s', output.getvalue())
        self.assertIn(f'bytes={len(b"secret-media")}', output.getvalue())
        self.assertNotIn('secret-media', output.getvalue())

    def test_ffmpeg_reader_logs_bounded_first_pcm_statistics(self):
        output = io.StringIO()
        configure_logging(2, stream=output)
        audio = np.array([-0.5, 0.25, np.nan, np.inf], dtype=np.float32)
        process = mock.Mock(stdout=_SequenceReader([audio.tobytes(), b'']))
        output_queue = queue.SimpleQueue()

        with mock.patch('stream_translator_gpt.audio_getter.time.monotonic', return_value=12.0):
            block_count = _read_ffmpeg_loop(process,
                                            audio.nbytes,
                                            output_queue,
                                            source_name='test ffmpeg',
                                            started_at=10.0)
        shutdown_logging()

        self.assertEqual(block_count, 1)
        self.assertIn('first PCM block after 2.000s', output.getvalue())
        self.assertIn('samples=4, finite=2, min=-0.500000, max=0.250000', output.getvalue())
        self.assertNotIn(str(audio), output.getvalue())

    def test_ffmpeg_reader_skips_first_pcm_statistics_below_debug(self):
        output = io.StringIO()
        configure_logging(0, stream=output)
        audio = np.array([-0.5, 0.25], dtype=np.float32)
        process = mock.Mock(stdout=_SequenceReader([audio.tobytes(), b'']))

        with mock.patch('stream_translator_gpt.audio_getter.np.isfinite') as isfinite:
            block_count = _read_ffmpeg_loop(process,
                                            audio.nbytes,
                                            queue.SimpleQueue(),
                                            source_name='test ffmpeg',
                                            started_at=10.0)
        shutdown_logging()

        self.assertEqual(block_count, 1)
        isfinite.assert_not_called()
        self.assertNotIn('first PCM block', output.getvalue())

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

        ytdlp.kill.assert_called_once_with()
        ytdlp.wait.assert_called_once_with(timeout=5.0)

    def test_posix_process_tree_kill_uses_private_process_group(self):
        process = mock.Mock(pid=1234)

        with mock.patch('stream_translator_gpt.audio_getter.os.name', 'posix'), \
                mock.patch('stream_translator_gpt.audio_getter.os.killpg', create=True) as killpg, \
                mock.patch('stream_translator_gpt.audio_getter.signal.SIGKILL', 9, create=True):
            _kill_process(process, 'yt-dlp', process_tree=True)

        killpg.assert_called_once_with(1234, 9)
        process.kill.assert_not_called()

    def test_windows_process_tree_kill_uses_parent_process(self):
        process = mock.Mock(pid=1234)

        with mock.patch('stream_translator_gpt.audio_getter.os.name', 'nt'), \
                mock.patch('stream_translator_gpt.audio_getter.os.killpg', create=True) as killpg:
            _kill_process(process, 'yt-dlp', process_tree=True)

        process.kill.assert_called_once_with()
        killpg.assert_not_called()

    def test_stream_ffmpeg_uses_error_logging_without_a_stderr_pipe(self):
        ytdlp = mock.Mock(pid=11)
        ffmpeg_process = mock.Mock(pid=12)
        output_builder = mock.Mock()
        output_builder.run_async.return_value = ffmpeg_process
        input_builder = mock.Mock()
        input_builder.output.return_value = output_builder
        transport_thread = mock.Mock()

        with mock.patch('stream_translator_gpt.audio_getter.subprocess.Popen', return_value=ytdlp), \
                mock.patch('stream_translator_gpt.audio_getter.ffmpeg.input', return_value=input_builder) as ffinput, \
                mock.patch('stream_translator_gpt.audio_getter.threading.Thread', return_value=transport_thread):
            _open_stream('https://example.test/live', 'ba', None, None, 'workdir')

        ffinput.assert_called_once_with('pipe:0', loglevel='error')
        input_builder.output.assert_called_once_with('pipe:1',
                                                     format='f32le',
                                                     acodec='pcm_f32le',
                                                     ac=1,
                                                     ar=16000)
        output_builder.run_async.assert_called_once_with(pipe_stdin=True, pipe_stdout=True)

    def test_tool_versions_are_logged_once_without_exposing_commands(self):
        import stream_translator_gpt.audio_getter as audio_getter

        output = io.StringIO()
        configure_logging(2, stream=output)
        completed = subprocess.CompletedProcess(['ffmpeg', '-version'], 0, 'ffmpeg version test-build\nmore', '')
        previous = audio_getter._stream_versions_logged
        audio_getter._stream_versions_logged = False
        try:
            with mock.patch('stream_translator_gpt.audio_getter.version', return_value='test-ytdlp'), \
                    mock.patch('stream_translator_gpt.audio_getter.subprocess.run', return_value=completed) as run:
                _log_stream_tool_versions()
                _log_stream_tool_versions()
            shutdown_logging()
        finally:
            audio_getter._stream_versions_logged = previous

        run.assert_called_once_with(['ffmpeg', '-version'],
                                    capture_output=True,
                                    text=True,
                                    encoding='utf-8',
                                    errors='replace',
                                    timeout=5,
                                    check=False)
        self.assertIn('yt-dlp test-ytdlp; ffmpeg version test-build', output.getvalue())

    def test_selected_format_diagnostic_contains_only_allowlisted_metadata(self):
        output = io.StringIO()
        configure_logging(2, stream=output)
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostic_path = Path(temp_dir) / 'format.txt'
            diagnostic_path.write_text(
                'format_id=251 protocol=https ext=webm acodec=opus vcodec=none live_status=is_live\n',
                encoding='utf-8')
            stats = _StreamTransportStats(started_at=0, format_diagnostic_path=diagnostic_path)
            _log_stream_format_diagnostic(stats)
            _log_stream_format_diagnostic(stats)
        shutdown_logging()

        self.assertEqual(output.getvalue().count('yt-dlp selected stream format:'), 1)
        self.assertIn('format_id=251 protocol=https ext=webm', output.getvalue())
        self.assertNotIn('watch?', output.getvalue())

    def test_stream_loop_rejects_zero_pcm_even_when_children_exit_cleanly(self):
        getter = StreamAudioGetter('https://example.test/live', 'ba', '', '')
        ffmpeg_process = mock.Mock()
        ytdlp_process = mock.Mock()
        transport_thread = mock.Mock()
        transport_thread.is_alive.return_value = False
        stats = _StreamTransportStats(started_at=10.0, media_bytes=1024, ytdlp_return_code=0)
        output_queue = queue.SimpleQueue()

        with mock.patch('stream_translator_gpt.audio_getter._open_stream',
                        return_value=(ffmpeg_process, ytdlp_process, transport_thread, stats)), \
                mock.patch('stream_translator_gpt.audio_getter._read_ffmpeg_loop', return_value=0), \
                mock.patch('stream_translator_gpt.audio_getter._wait_for_process', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'no decoded PCM audio was produced'):
                getter.loop(output_queue)

        self.assertIsNone(output_queue.get_nowait())
        self.assertTrue(output_queue.empty())

    def test_stream_loop_propagates_both_child_failures(self):
        getter = StreamAudioGetter('https://example.test/live', 'ba', '', '')
        ffmpeg_process = mock.Mock()
        ytdlp_process = mock.Mock()
        transport_thread = mock.Mock()
        transport_thread.is_alive.return_value = False
        stats = _StreamTransportStats(started_at=10.0, media_bytes=1024, ytdlp_return_code=3)
        output_queue = queue.SimpleQueue()

        with mock.patch('stream_translator_gpt.audio_getter._open_stream',
                        return_value=(ffmpeg_process, ytdlp_process, transport_thread, stats)), \
                mock.patch('stream_translator_gpt.audio_getter._read_ffmpeg_loop', return_value=1), \
                mock.patch('stream_translator_gpt.audio_getter._wait_for_process', return_value=7):
            with self.assertRaisesRegex(RuntimeError, 'yt-dlp exited with status 3; ffmpeg exited with status 7'):
                getter.loop(output_queue)

        self.assertIsNone(output_queue.get_nowait())
        self.assertTrue(output_queue.empty())

    def test_stop_requested_at_natural_completion_discards_a_full_output_queue(self):
        getter = StreamAudioGetter('https://example.test/live', 'ba', '', '')
        ffmpeg_process = mock.Mock()
        ytdlp_process = mock.Mock()
        transport_thread = mock.Mock()
        transport_thread.is_alive.return_value = False
        stats = _StreamTransportStats(started_at=10.0, media_bytes=1024, ytdlp_return_code=0)
        output_queue = queue.Queue(maxsize=1)
        output_queue.put('buffered-audio')

        def request_stop(message, *args):
            if message == 'Stream audio input finished':
                getter.stop_event.set()

        with mock.patch('stream_translator_gpt.audio_getter._open_stream',
                        return_value=(ffmpeg_process, ytdlp_process, transport_thread, stats)), \
                mock.patch('stream_translator_gpt.audio_getter._read_ffmpeg_loop', return_value=1), \
                mock.patch('stream_translator_gpt.audio_getter._wait_for_process', return_value=0), \
                mock.patch('stream_translator_gpt.audio_getter.logger.info', side_effect=request_stop):
            getter.loop(output_queue)

        self.assertIsNone(output_queue.get_nowait())
        self.assertTrue(output_queue.empty())

    def test_stream_read_failure_kills_reaps_and_joins_both_children(self):
        getter = StreamAudioGetter('https://example.test/live', 'ba', '', '')
        ffmpeg_process = mock.Mock()
        ffmpeg_process.poll.return_value = None
        ytdlp_process = mock.Mock()
        ytdlp_process.poll.return_value = None
        transport_thread = mock.Mock()
        transport_thread.is_alive.side_effect = [True, False]
        stats = _StreamTransportStats(started_at=10.0)
        output_queue = queue.SimpleQueue()

        with mock.patch('stream_translator_gpt.audio_getter._open_stream',
                        return_value=(ffmpeg_process, ytdlp_process, transport_thread, stats)), \
                mock.patch('stream_translator_gpt.audio_getter._read_ffmpeg_loop', side_effect=OSError('read failed')):
            with self.assertRaisesRegex(OSError, 'read failed'):
                getter.loop(output_queue)

        ffmpeg_process.kill.assert_called_once_with()
        ytdlp_process.kill.assert_called_once_with()
        ffmpeg_process.wait.assert_called_once_with(timeout=5)
        ytdlp_process.wait.assert_called_once_with(timeout=5)
        transport_thread.join.assert_called_once_with(timeout=5)
        self.assertIsNone(output_queue.get_nowait())
        self.assertTrue(output_queue.empty())

    def test_external_stop_kills_reaps_and_joins_stream_children(self):
        getter = StreamAudioGetter('https://example.test/live', 'ba', '', '')
        ffmpeg_process = mock.Mock(pid=1233)
        ytdlp_process = mock.Mock(pid=1234)
        ffmpeg_process.poll.side_effect = lambda: -9 if ffmpeg_process.kill.called else None
        ffmpeg_process.wait.return_value = -9
        ytdlp_process.wait.return_value = -9
        transport_running = [True]
        transport_thread = mock.Mock()
        transport_thread.is_alive.side_effect = lambda: transport_running[0]
        transport_thread.join.side_effect = lambda timeout: transport_running.__setitem__(0, False)
        stats = _StreamTransportStats(started_at=10.0)
        read_started = threading.Event()
        stop_completed = threading.Event()
        output_queue = queue.Queue(maxsize=1)
        output_queue.put('buffered-audio')

        def wait_until_stopped(*_args, **_kwargs):
            read_started.set()
            self.assertTrue(getter.stop_event.wait(timeout=1))
            # stop_event is set before stop() signals the children. Let that
            # finish before cleanup so the mock process state is deterministic.
            self.assertTrue(stop_completed.wait(timeout=2))
            return 0

        errors = []

        def run_getter():
            try:
                getter.loop(output_queue)
            except Exception as exc:  # pragma: no cover - assertion below reports the failure
                errors.append(exc)

        with mock.patch('stream_translator_gpt.audio_getter._open_stream',
                        return_value=(ffmpeg_process, ytdlp_process, transport_thread, stats)), \
                mock.patch('stream_translator_gpt.audio_getter._read_ffmpeg_loop', side_effect=wait_until_stopped), \
                mock.patch('stream_translator_gpt.audio_getter.os.killpg', create=True) as killpg:
            # Always mock the OS call: these are fake PIDs, never real children.
            ytdlp_process.poll.side_effect = lambda: (
                -9 if ytdlp_process.kill.called or killpg.called else None)
            worker = threading.Thread(target=run_getter)
            worker.start()
            try:
                self.assertTrue(read_started.wait(timeout=1))
                getter.stop()
                stop_calls = (list(ffmpeg_process.kill.mock_calls),
                              list(ytdlp_process.kill.mock_calls), list(killpg.mock_calls))
                getter.stop()
                self.assertEqual((ffmpeg_process.kill.mock_calls,
                                  ytdlp_process.kill.mock_calls, killpg.mock_calls), stop_calls)
            finally:
                getter.stop()
                stop_completed.set()
                worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(getter.stop_event.is_set())
        self.assertTrue(getter.finished.is_set())
        ffmpeg_process.kill.assert_called_once_with()
        if os.name == 'nt':
            ytdlp_process.kill.assert_called_once_with()
            killpg.assert_not_called()
        else:
            # macOS/Linux also clean up the private group after the parent
            # exits, since an HLS child may still hold the stream pipe open.
            self.assertEqual(killpg.call_args_list, [
                mock.call(ytdlp_process.pid, signal.SIGKILL),
                mock.call(ytdlp_process.pid, signal.SIGKILL),
            ])
            ytdlp_process.kill.assert_not_called()
        ffmpeg_process.wait.assert_called_once_with(timeout=5)
        ytdlp_process.wait.assert_called_once_with(timeout=5)
        transport_thread.join.assert_called_once_with(timeout=5)
        self.assertIsNone(output_queue.get_nowait())
        self.assertTrue(output_queue.empty())

    def test_stop_event_unblocks_full_stream_output_queue(self):
        full_block = np.arange(512, dtype=np.float32).tobytes()
        process = mock.Mock(stdout=io.BytesIO(full_block))
        stop_event = threading.Event()

        class ObservedQueue(queue.Queue):
            def __init__(self):
                super().__init__(maxsize=1)
                self.audio_put_attempted = threading.Event()

            def put(self, item, *args, **kwargs):
                if isinstance(item, np.ndarray):
                    self.audio_put_attempted.set()
                return super().put(item, *args, **kwargs)

        output_queue = ObservedQueue()
        output_queue.put('occupied')
        result = []
        worker = threading.Thread(target=lambda: result.append(
            _read_ffmpeg_loop(process, len(full_block), output_queue, stop_event=stop_event)))
        worker.start()
        self.assertTrue(output_queue.audio_put_attempted.wait(timeout=1))

        stop_event.set()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [0])

    def test_transport_thread_start_failure_kills_and_reaps_started_children(self):
        ytdlp_process = mock.Mock(pid=11)
        ytdlp_process.poll.return_value = None
        ffmpeg_process = mock.Mock(pid=12)
        ffmpeg_process.poll.return_value = None
        output_builder = mock.Mock()
        output_builder.run_async.return_value = ffmpeg_process
        input_builder = mock.Mock()
        input_builder.output.return_value = output_builder
        transport_thread = mock.Mock()
        transport_thread.start.side_effect = RuntimeError('thread failed')

        def safe_kill(process, _process_name, process_tree=False):
            process.kill()

        with mock.patch('stream_translator_gpt.audio_getter.subprocess.Popen', return_value=ytdlp_process), \
                mock.patch('stream_translator_gpt.audio_getter.ffmpeg.input', return_value=input_builder), \
                mock.patch('stream_translator_gpt.audio_getter.threading.Thread', return_value=transport_thread), \
                mock.patch('stream_translator_gpt.audio_getter._kill_process', side_effect=safe_kill) as kill_process:
            with self.assertRaisesRegex(RuntimeError, 'thread failed'):
                _open_stream('https://example.test/live', 'ba', None, None, 'workdir')

        kill_process.assert_has_calls([
            mock.call(ffmpeg_process, 'ffmpeg', process_tree=False),
            mock.call(ytdlp_process, 'yt-dlp', process_tree=True),
        ])
        ffmpeg_process.kill.assert_called_once_with()
        ytdlp_process.kill.assert_called_once_with()
        ffmpeg_process.wait.assert_called_once_with(timeout=5)
        ytdlp_process.wait.assert_called_once_with(timeout=5)


if __name__ == '__main__':
    unittest.main()
