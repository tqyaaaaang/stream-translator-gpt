import io
import json
import math
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from stream_translator_gpt.getter.audio_getter import (
    LocalFileAudioGetter,
    StreamAudioGetter,
    _open_stream,
    _signal_end_of_stream,
    _StreamTransportStats,
)
from stream_translator_gpt.utils.common import SAMPLE_RATE, SAMPLES_PER_FRAME
from stream_translator_gpt.getter.time_range import is_local_hls


GETTER_MODULE = 'stream_translator_gpt.getter.audio_getter'


class LocalFileTimeRangeTest(unittest.TestCase):

    @staticmethod
    def _run_mocked(ss=None, to=None, realtime=False):
        process = mock.Mock(stdout=io.BytesIO(np.zeros(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()))
        process.wait.return_value = 0
        process.poll.return_value = 0
        builder = mock.Mock()
        builder.output.return_value.run_async.return_value = process
        getter = LocalFileAudioGetter('clip.wav', realtime=realtime, ss=ss, to=to)
        output = queue.SimpleQueue()
        with mock.patch(f'{GETTER_MODULE}.ffmpeg.input', return_value=builder) as ffmpeg_input:
            getter.loop(output)
        return ffmpeg_input, builder, output, getter

    def test_range_is_source_seek_and_remaining_output_duration(self):
        cases = [
            (None, None, None, {}),
            ('12.5', None, '2.5', {'af': 'atrim=start=12.5,asetpts=PTS-STARTPTS'}),
            (None, '1:02.5', '0', {'af': 'atrim=start=0,asetpts=PTS-STARTPTS', 't': '62.5'}),
            ('1:00', '1:02.5', '50', {'af': 'atrim=start=60,asetpts=PTS-STARTPTS', 't': '2.5'}),
            (0, 1, '0', {'af': 'atrim=start=0,asetpts=PTS-STARTPTS', 't': '1'}),
        ]
        for ss, to, coarse_start, expected_output in cases:
            with self.subTest(ss=ss, to=to):
                ffmpeg_input, builder, output, getter = self._run_mocked(ss, to)
                expected_input = {'loglevel': 'panic'}
                if coarse_start is not None:
                    expected_input.update(copyts=None, start_at_zero=None, ss=coarse_start)
                ffmpeg_input.assert_called_once_with('clip.wav', **expected_input)
                builder.output.assert_called_once_with('pipe:', format='f32le', acodec='pcm_f32le',
                                                       ac=1, ar=SAMPLE_RATE, **expected_output)
                self.assertEqual(output.get_nowait().shape, (SAMPLES_PER_FRAME,))
                self.assertIsNone(output.get_nowait())
                self.assertTrue(output.empty())
                self.assertTrue(getter.finished.is_set())

    def test_invalid_direct_constructor_ranges_fail_before_starting_decoder(self):
        for ss, to in [(-1, None), (None, 0), (5, 4), (5, 5), ('bad', None), (None, math.inf)]:
            with self.subTest(ss=ss, to=to), mock.patch(f'{GETTER_MODULE}.ffmpeg.input') as ffmpeg_input:
                with self.assertRaises(ValueError):
                    LocalFileAudioGetter('clip.wav', ss=ss, to=to)
                ffmpeg_input.assert_not_called()

    def test_hls_header_detection_does_not_open_fifos_or_devices(self):
        with mock.patch('stream_translator_gpt.getter.time_range.Path.is_file', return_value=False), \
                mock.patch('stream_translator_gpt.getter.time_range.Path.open') as open_file:
            self.assertFalse(is_local_hls('audio-pipe'))
        open_file.assert_not_called()

    def test_local_transport_streams_use_linear_output_seeking(self):
        for extension in ('.ts', '.mpegts', '.M3U8'):
            with self.subTest(extension=extension):
                process = mock.Mock(stdout=io.BytesIO(np.zeros(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()))
                process.wait.return_value = process.poll.return_value = 0
                builder = mock.Mock()
                builder.output.return_value.run_async.return_value = process
                getter = LocalFileAudioGetter('clip' + extension, ss=220, to=222)
                with mock.patch(f'{GETTER_MODULE}.ffmpeg.input', return_value=builder) as ffmpeg_input:
                    getter.loop(queue.SimpleQueue())
                ffmpeg_input.assert_called_once_with('clip' + extension, loglevel='panic')
                self.assertEqual(builder.output.call_args.kwargs['ss'], '220')
                self.assertEqual(builder.output.call_args.kwargs['t'], '2')
                self.assertNotIn('af', builder.output.call_args.kwargs)

    def test_ffmpeg_command_places_seek_before_input_for_efficient_access(self):
        process = mock.Mock(stdout=io.BytesIO(np.zeros(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()))
        process.wait.return_value = 0
        process.poll.return_value = 0
        getter = LocalFileAudioGetter('clip.wav', ss='1:00', to='1:02.5')
        with mock.patch(f'{GETTER_MODULE}.subprocess.Popen', return_value=process) as popen:
            getter.loop(queue.SimpleQueue())
        command = popen.call_args.args[0]
        self.assertLess(command.index('-ss'), command.index('-i'))
        self.assertGreater(command.index('-t'), command.index('-i'))
        self.assertEqual(command[command.index('-ss') + 1], '50')
        self.assertEqual(command[command.index('-t') + 1], '2.5')
        self.assertEqual(command[command.index('-af') + 1], 'atrim=start=60,asetpts=PTS-STARTPTS')
        self.assertIn('-copyts', command)
        self.assertIn('-start_at_zero', command)
        self.assertNotIn('-to', command)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for offline audio range integration')
    def test_actual_ffmpeg_extracts_the_requested_source_samples(self):
        # All audio is synthesized locally: no URLs, devices or paid APIs.
        source = (np.arange(SAMPLE_RATE, dtype=np.int32) - SAMPLE_RATE // 2).astype('<i2')
        ranges = [(None, 0.064), (0.032, 0.096), (0.968, None), (0.125, 0.126), (0.75, 9.0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'range source.wav'
            with wave.open(str(path), 'wb') as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(SAMPLE_RATE)
                audio_file.writeframes(source.tobytes())
            for ss, to in ranges:
                with self.subTest(ss=ss, to=to):
                    getter = LocalFileAudioGetter(str(path), ss=ss, to=to)
                    output = queue.SimpleQueue()
                    try:
                        getter.loop(output)
                        blocks = []
                        while (block := output.get_nowait()) is not None:
                            blocks.append(block)
                        self.assertTrue(output.empty())
                        self.assertTrue(getter.finished.is_set())
                        actual = np.concatenate(blocks)
                        first = round((ss or 0) * SAMPLE_RATE)
                        last = min(len(source), round(to * SAMPLE_RATE)) if to is not None else len(source)
                        expected = source[first:last].astype(np.float32) / 32768
                        self.assertEqual(len(actual), math.ceil(len(expected) / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME)
                        np.testing.assert_array_equal(actual[:len(expected)], expected)
                        np.testing.assert_array_equal(actual[len(expected):], 0)
                    finally:
                        getter.ffmpeg_process.stdout.close()
                        getter.ffmpeg_process.stdin.close()


class StreamTimeRangeTest(unittest.TestCase):

    @staticmethod
    def _open_mocked(ss=None, to=None, live_from_start=False):
        downloader = mock.Mock(pid=111)
        decoder = mock.Mock(pid=112)
        builder = mock.Mock()
        builder.output.return_value.run_async.return_value = decoder
        with mock.patch(f'{GETTER_MODULE}._log_stream_tool_versions'), \
                mock.patch(f'{GETTER_MODULE}.subprocess.Popen', return_value=downloader) as popen, \
                mock.patch(f'{GETTER_MODULE}.ffmpeg.input', return_value=builder) as ffmpeg_input, \
                mock.patch(f'{GETTER_MODULE}.threading.Thread') as transport:
            _open_stream('https://example.test/video', 'bestaudio', '', '', 'workdir',
                         ss=ss, to=to, live_from_start=live_from_start)
        transport.return_value.start.assert_called_once_with()
        return popen.call_args.args[0], ffmpeg_input, builder

    def test_downloader_uses_coarse_seek_exact_trim_and_output_duration(self):
        from stream_translator_gpt.getter.range_downloader import configure_range_downloader, parse_range_options

        for ss, to, coarse, start, duration in [
            (None, None, None, None, None),
            ('12.5', None, '2.5', '12.5', None),
            (None, '1:02.5', '0', '0', '62.5'),
            ('1:00', '1:02.5', '50', '60', '2.5'),
            ('03:40.1', None, '210.1', '220.1', None),
            (0, 1, '0', '0', '1'),
            (0, None, '0', '0', None),
        ]:
            with self.subTest(ss=ss, to=to):
                command, ffmpeg_input, builder = self._open_mocked(ss, to)
                self.assertNotIn('--download-sections', command)
                if coarse is None:
                    self.assertEqual(command[2], 'yt_dlp')
                    self.assertNotIn('--downloader-args', command)
                    self.assertNotIn('--downloader', command)
                else:
                    self.assertEqual(command[2], 'stream_translator_gpt.getter.range_downloader')
                    # Use yt-dlp's actual parser so quoting/tokenization is
                    # checked without invoking a shell or making any request.
                    parsed_start, parsed_end, parsed = parse_range_options(command[3:])
                    self.assertEqual(parsed_start, float(start) if ss is not None else None)
                    self.assertEqual(parsed_end, float(start) + float(duration) if duration is not None else None)
                    options = parsed.ydl_opts
                    configure_range_downloader(options, {'protocol': 'https', 'url': 'https://example.test/media'},
                                               parsed_start, parsed_end)
                    args = options['external_downloader_args']
                    self.assertEqual(args['ffmpeg_i'], ['-copyts', '-start_at_zero', '-ss', coarse])
                    expected = ['-vn', '-c:a', 'pcm_f32le', '-ar', str(SAMPLE_RATE), '-ac', '1',
                                '-af', f'atrim=start={start},asetpts=PTS-STARTPTS']
                    if duration is not None:
                        expected += ['-t', duration]
                    expected += ['-f', 'matroska']
                    self.assertEqual(args['ffmpeg_o'], expected)
                # The downstream decoder must not apply the same seek or close
                # its input early; yt-dlp owns the finite section's natural EOF.
                ffmpeg_input.assert_called_once_with('pipe:0', loglevel='error')
                builder.output.assert_called_once_with('pipe:1', format='f32le', acodec='pcm_f32le',
                                                       ac=1, ar=SAMPLE_RATE)

    def test_direct_opener_ignores_even_invalid_range_when_live_from_start_is_enabled(self):
        with self.assertLogs('main', level='WARNING') as logs:
            command, _, _ = self._open_mocked('invalid', -1, live_from_start=True)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('--live-from-start', logs.output[0])
        self.assertIn('--live-from-start', command)
        self.assertNotIn('--download-sections', command)

    def test_direct_constructor_ignores_range_before_validation_for_live_start(self):
        with self.assertLogs('main', level='WARNING') as logs:
            getter = StreamAudioGetter('https://example.test/live', 'bestaudio', '', '',
                                       live_from_start=True, ss='invalid', to=-1)
        try:
            self.assertEqual(len(logs.output), 1)
            self.assertIsNone(getter.ss)
            self.assertIsNone(getter.to)
        finally:
            getter.__del__()

    def test_direct_opener_rejects_invalid_range_before_starting_children(self):
        with mock.patch(f'{GETTER_MODULE}.subprocess.Popen') as popen:
            with self.assertRaises(ValueError):
                _open_stream('https://example.test/video', 'bestaudio', '', '', 'workdir', ss=2, to=1)
        popen.assert_not_called()

    def test_finite_section_has_normal_eof_even_for_less_than_one_audio_frame(self):
        getter = StreamAudioGetter('https://example.test/video', 'bestaudio', '', '', ss='3', to='3.001')
        samples = np.arange(16, dtype=np.float32)
        decoder = mock.Mock(stdout=io.BytesIO(samples.tobytes()))
        decoder.wait.return_value = 0
        downloader = mock.Mock()
        transport = mock.Mock()
        transport.is_alive.return_value = False
        stats = _StreamTransportStats(started_at=0, media_bytes=256, ytdlp_return_code=0)
        output = queue.SimpleQueue()
        with mock.patch(f'{GETTER_MODULE}._open_stream',
                        return_value=(decoder, downloader, transport, stats)) as open_stream, \
                mock.patch(f'{GETTER_MODULE}._cleanup_stream_children') as cleanup:
            getter.loop(output)
        self.assertEqual(open_stream.call_args.kwargs['ss'], 3)
        self.assertEqual(open_stream.call_args.kwargs['to'], 3.001)
        actual = output.get_nowait()
        np.testing.assert_array_equal(actual[:len(samples)], samples)
        np.testing.assert_array_equal(actual[len(samples):], 0)
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())
        self.assertTrue(getter.finished.is_set())
        cleanup.assert_not_called()
        downloader.kill.assert_not_called()
        decoder.kill.assert_not_called()

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for offline section container integration')
    def test_section_downloader_matches_native_ffmpeg_for_aac_opus_and_hls(self):
        # Exercise the installed yt-dlp FFmpeg downloader on generated local
        # media, bypassing extraction and any HTTP requests entirely.
        downloader_script = '''
import sys
import json
from yt_dlp import YoutubeDL
from yt_dlp.downloader.external import FFmpegFD
from stream_translator_gpt.getter.range_downloader import configure_range_downloader, parse_range_options
start, end, parsed = parse_range_options(json.loads(sys.argv[4]))
params = parsed.ydl_opts
params['quiet'] = False
info = {
    'url': sys.argv[1], 'ext': sys.argv[2], 'protocol': sys.argv[3],
}
configure_range_downloader(params, info, start, end)
with YoutubeDL(params) as ydl:
    sys.exit(FFmpegFD(ydl, params)._call_downloader('-', info))
'''
        command, _, _ = self._open_mocked('0.2', '0.5')
        with tempfile.TemporaryDirectory() as directory:
            for extension, encoder, protocol in [
                ('m4a', 'aac', 'http'),
                ('webm', 'libopus', 'http'),
                ('m3u8', 'aac', 'm3u8_native'),
            ]:
                with self.subTest(extension=extension):
                    source = Path(directory) / f'local source.{extension}'
                    encode_command = [
                        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                        'sine=frequency=1000:sample_rate=48000:duration=1', '-c:a', encoder,
                    ]
                    if extension == 'm3u8':
                        encode_command.extend(['-f', 'hls', '-hls_time', '0.2'])
                    encoded = subprocess.run(
                        [*encode_command, str(source)],
                        capture_output=True, timeout=15, check=False)
                    if encoded.returncode and b'Unknown encoder' in encoded.stderr:
                        # Some minimal FFmpeg builds omit the optional Opus
                        # encoder. The AAC case still exercises m4a pipe safety.
                        continue
                    self.assertEqual(encoded.returncode, 0, encoded.stderr.decode(errors='replace'))
                    section = subprocess.run(
                        [sys.executable, '-B', '-c', downloader_script, str(source),
                         'mp4' if extension == 'm3u8' else extension, protocol, json.dumps(command[3:])],
                        capture_output=True, timeout=15, check=False)
                    self.assertEqual(section.returncode, 0, section.stderr.decode(errors='replace'))
                    self.assertTrue(section.stdout)
                    decoded = subprocess.run(
                        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0', '-f', 'f32le',
                         '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1'],
                        input=section.stdout, capture_output=True, timeout=15, check=False)
                    self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors='replace'))
                    reference = subprocess.run(
                        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
                         '-ss', '0.2', '-t', '0.3', '-vn', '-c:a', 'pcm_f32le', '-ar', str(SAMPLE_RATE), '-ac', '1',
                         '-f', 'f32le', 'pipe:1'],
                        capture_output=True, timeout=15, check=False)
                    self.assertEqual(reference.returncode, 0, reference.stderr.decode(errors='replace'))
                    # The reference decodes from the beginning, so it cannot
                    # reproduce the input-side HLS seek bug under test.
                    self.assertTrue(reference.stdout)
                    self.assertEqual(len(decoded.stdout), len(reference.stdout))
                    # atrim before resampling and output-side -ss after it
                    # initialize the resampler differently at the boundary.
                    # Require matching waveform/timing, not identical floats.
                    prefix_samples = SAMPLE_RATE // 20
                    actual = np.frombuffer(decoded.stdout, np.float32)[:prefix_samples]
                    expected = np.frombuffer(reference.stdout, np.float32)[:prefix_samples]
                    self.assertGreater(float(np.corrcoef(actual, expected)[0, 1]), 0.999)


class EndOfStreamCancellationTest(unittest.TestCase):

    class ObservedQueue(queue.Queue):
        def __init__(self):
            super().__init__(maxsize=1)
            self.waiting_for_eof = threading.Event()

        def put(self, item, block=True, timeout=None):
            if item is None:
                self.waiting_for_eof.set()
            return super().put(item, block=block, timeout=timeout)

    def test_cancel_after_natural_eof_started_waiting_for_full_queue(self):
        output = self.ObservedQueue()
        output.put('buffered audio')
        stopped = threading.Event()
        worker = threading.Thread(target=_signal_end_of_stream, args=(output,), kwargs={'stop_event': stopped},
                                  daemon=True)
        worker.start()
        try:
            self.assertTrue(output.waiting_for_eof.wait(2))
            self.assertTrue(worker.is_alive())
            stopped.set()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive(), 'EOF must not deadlock after its consumer stops')
            self.assertIsNone(output.get_nowait())
            self.assertTrue(output.empty())
        finally:
            stopped.set()
            if worker.is_alive():
                output.get_nowait()
                worker.join(timeout=2)

    def test_natural_eof_waits_without_discarding_unconsumed_audio(self):
        output = self.ObservedQueue()
        output.put('buffered audio')
        stopped = threading.Event()
        worker = threading.Thread(target=_signal_end_of_stream, args=(output,), kwargs={'stop_event': stopped},
                                  daemon=True)
        worker.start()
        try:
            self.assertTrue(output.waiting_for_eof.wait(2))
            self.assertEqual(output.get_nowait(), 'buffered audio')
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertIsNone(output.get_nowait())
        finally:
            stopped.set()
            worker.join(timeout=2)

    def test_local_getter_stop_interrupts_already_blocked_eof(self):
        output = self.ObservedQueue()
        process = mock.Mock(stdout=io.BytesIO(np.ones(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()))
        process.wait.return_value = process.poll.return_value = 0
        builder = mock.Mock()
        builder.output.return_value.run_async.return_value = process
        getter = LocalFileAudioGetter('clip.wav')
        with mock.patch(f'{GETTER_MODULE}.ffmpeg.input', return_value=builder):
            worker = threading.Thread(target=getter.loop, args=(output,), daemon=True)
            worker.start()
            try:
                self.assertTrue(output.waiting_for_eof.wait(2))
                getter.stop()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(getter.finished.is_set())
                self.assertIsNone(output.get_nowait())
                self.assertTrue(output.empty())
            finally:
                getter.stop()
                if worker.is_alive():
                    output.get_nowait()
                    worker.join(timeout=2)

    def test_stream_getter_stop_interrupts_already_blocked_eof(self):
        output = self.ObservedQueue()
        process = mock.Mock(stdout=io.BytesIO(np.ones(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()))
        process.wait.return_value = process.poll.return_value = 0
        downloader = mock.Mock()
        transport = mock.Mock()
        transport.is_alive.return_value = False
        stats = _StreamTransportStats(started_at=0, media_bytes=4096, ytdlp_return_code=0)
        getter = StreamAudioGetter('https://example.test/video', 'bestaudio', '', '')
        with mock.patch(f'{GETTER_MODULE}._open_stream', return_value=(process, downloader, transport, stats)):
            worker = threading.Thread(target=getter.loop, args=(output,), daemon=True)
            worker.start()
            try:
                self.assertTrue(output.waiting_for_eof.wait(2))
                getter.stop()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(getter.finished.is_set())
                self.assertIsNone(output.get_nowait())
                self.assertTrue(output.empty())
            finally:
                getter.stop()
                if worker.is_alive():
                    output.get_nowait()
                    worker.join(timeout=2)


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for offline long-seek regression tests')
class LongSeekRegressionTest(unittest.TestCase):
    """Nonconstant audio catches a seek to the wrong but valid HLS segment."""

    DOWNLOADER = '''
import json
import sys
from yt_dlp import YoutubeDL
from yt_dlp.downloader.external import FFmpegFD
from stream_translator_gpt.getter.range_downloader import configure_range_downloader, parse_range_options
start, end, parsed = parse_range_options(json.loads(sys.argv[1]))
params = parsed.ydl_opts
params['quiet'] = True
info = json.loads(sys.argv[2])
if info.pop('__capture_segment_reads', False):
    params['quiet'] = False
    params['verbose'] = True
configure_range_downloader(params, info, start, end)
with YoutubeDL(params) as ydl:
    sys.exit(FFmpegFD(ydl, params)._call_downloader('-', info))
'''

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        wav = cls.root / 'position coded source.wav'
        with wave.open(str(wav), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48000)
            second = np.arange(48000) / 48000
            for position in range(225):
                frequency = 200 + position * 10
                samples = (np.sin(2 * np.pi * frequency * second) * 8000).astype('<i2')
                output.writeframes(samples.tobytes())
        cls.sources = {}
        variants = [
            ('aac', 'm4a', 'aac', [], 'http'),
            ('opus', 'webm', 'libopus', [], 'http'),
            ('hls', 'm3u8', 'aac', ['-f', 'hls', '-hls_time', '5', '-hls_list_size', '0'], 'm3u8_native'),
            ('offset', 'm4a', 'aac', ['-output_ts_offset', '1200'], 'http'),
        ]
        for name, extension, codec, extra, protocol in variants:
            path = cls.root / f'{name}.{extension}'
            result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(wav),
                                     '-c:a', codec, *extra, str(path)], capture_output=True, timeout=30)
            if result.returncode and codec == 'libopus' and b'Unknown encoder' in result.stderr:
                continue
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors='replace'))
            cls.sources[name] = (path, {'url': str(path), 'ext': 'mp4' if extension == 'm3u8' else extension,
                                       'protocol': protocol})

    @staticmethod
    def _reference(path, ss, to):
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(path), '-ss', str(ss or 0)]
        if to is not None:
            command += ['-t', str(to - (ss or 0))]
        command += ['-vn', '-f', 'f32le', '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1']
        result = subprocess.run(command, capture_output=True, timeout=20)
        if result.returncode:
            raise AssertionError(result.stderr.decode(errors='replace'))
        return np.frombuffer(result.stdout, np.float32)

    def _get_audio(self, path, info, ss, to, local=False, debug_log=None):
        getter = (LocalFileAudioGetter(str(path), ss=ss, to=to) if local else
                  StreamAudioGetter(path.as_uri(), 'bestaudio/best', '', '', ss=ss, to=to))
        output = queue.SimpleQueue()
        real_popen = subprocess.Popen
        diagnostic_file = tempfile.TemporaryFile() if debug_log is not None else None
        info = dict(info)
        if diagnostic_file is not None:
            info['__capture_segment_reads'] = True

        def spawn(command, *args, **kwargs):
            if isinstance(command, list) and command[1:3] == ['-m', 'stream_translator_gpt.getter.range_downloader']:
                # Keep the complete real getter, transport thread and decoder;
                # replace only extraction/network access with local source info.
                command = [sys.executable, '-B', '-c', self.DOWNLOADER,
                           json.dumps(command[3:]), json.dumps(info)]
                if diagnostic_file is not None:
                    kwargs['stderr'] = diagnostic_file
            return real_popen(command, *args, **kwargs)

        watchdog = threading.Timer(30, getter.stop)
        watchdog.daemon = True
        watchdog.start()
        try:
            with mock.patch('subprocess.Popen', side_effect=spawn):
                getter.loop(output)
            self.assertFalse(getter.stop_event.is_set(), 'Getter exceeded its bounded offline test timeout')
            blocks = []
            while (block := output.get_nowait()) is not None:
                blocks.append(block)
            self.assertTrue(output.empty())
            self.assertTrue(getter.finished.is_set())
            return np.concatenate(blocks) if blocks else np.array([], np.float32)
        finally:
            watchdog.cancel()
            if getter.ffmpeg_process:
                for pipe in (getter.ffmpeg_process.stdout, getter.ffmpeg_process.stdin):
                    if pipe is not None and not pipe.closed:
                        pipe.close()
            if diagnostic_file is not None:
                diagnostic_file.seek(0)
                debug_log.append(diagnostic_file.read().decode(errors='replace'))
                diagnostic_file.close()

    def _assert_position_matches(self, actual, reference):
        # Only the getter's existing final-frame padding may extend duration.
        self.assertEqual(len(actual), math.ceil(len(reference) / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(len(reference), SAMPLE_RATE // 10)
        self.assertGreater(float(np.sqrt(np.mean(actual[:len(reference)] ** 2))), 0.05)
        # Every source second has a distinct frequency. Compare several short
        # windows, including transitions, rather than merely duration or RMS.
        window = SAMPLE_RATE // 4
        for offset in range(0, len(reference) - window + 1, window):
            expected = np.abs(np.fft.rfft(reference[offset:offset+window] * np.hanning(window)))
            observed = np.abs(np.fft.rfft(actual[offset:offset+window] * np.hanning(window)))
            self.assertLessEqual(abs(int(np.argmax(observed)) - int(np.argmax(expected))), 1,
                                 f'Wrong source position near output sample {offset}')

    def test_long_integer_and_fractional_seek_preserve_positions_for_all_formats(self):
        for name, (path, info) in self.sources.items():
            for ss, to in [(220, None), (220, 222), (220.1, 223.1)]:
                reference = self._reference(path, ss, to)
                for local in (False, True):
                    with self.subTest(format=name, ss=ss, to=to, local=local):
                        actual = self._get_audio(path, info, ss, to, local=local)
                        self._assert_position_matches(actual, reference)

    def test_near_zero_to_only_and_past_eof_ranges(self):
        for name in ('aac', 'opus', 'offset', 'hls'):
            if name not in self.sources:
                continue
            path, info = self.sources[name]
            for ss, to in [(None, 0.75), (0, 0.75), (0.1, 0.85), (224.5, 230)]:
                reference = self._reference(path, ss, to)
                with self.subTest(format=name, ss=ss, to=to):
                    self._assert_position_matches(self._get_audio(path, info, ss, to), reference)

    def test_zero_start_without_endpoint_is_not_treated_as_disabled(self):
        path, info = self.sources.get('opus', self.sources['aac'])
        actual = self._get_audio(path, info, 0, None)
        reference = self._reference(path, 0, None)
        self._assert_position_matches(actual, reference)

    def test_http_media_retains_bounded_input_seek(self):
        path, info = self.sources['aac']
        diagnostics = []
        actual = self._get_audio(path, info, 220.1, 223.1, debug_log=diagnostics)
        self._assert_position_matches(actual, self._reference(path, 220.1, 223.1))
        command = next(line for line in '\n'.join(diagnostics).splitlines() if 'ffmpeg command line:' in line)
        self.assertIn('-ss 210.1', command)
        self.assertLess(command.index('-ss 210.1'), command.index(' -i '))
        self.assertIn('atrim=start=220.1,asetpts=PTS-STARTPTS', command)

    def test_hls_discontinuity_resets_do_not_drop_or_select_the_wrong_audio(self):
        parts = []
        for name, frequency in [('first', 440), ('second', 880)]:
            path = self.root / f'{name}.m3u8'
            encoded = subprocess.run(
                ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                 f'sine=frequency={frequency}:sample_rate=48000:duration=10', '-c:a', 'aac',
                 '-f', 'hls', '-hls_time', '2', '-hls_list_size', '0', str(path)],
                capture_output=True, timeout=20)
            self.assertEqual(encoded.returncode, 0, encoded.stderr.decode(errors='replace'))
            parts.append('\n'.join(line for line in path.read_text().splitlines()
                                   if line.startswith('#EXTINF') or line.endswith('.ts')))
        playlist = ('#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n'
                    + parts[0] + '\n#EXT-X-DISCONTINUITY\n' + parts[1] + '\n#EXT-X-ENDLIST\n')
        path = self.root / 'discontinuous.m3u8'
        path.write_text(playlist, encoding='utf-8')
        extensionless = self.root / 'discontinuous playlist.txt'
        extensionless.write_text(playlist, encoding='utf-8')
        info = {'url': str(path), 'ext': 'mp4', 'protocol': 'm3u8_native'}
        for ss, to in [(15, 18), (8, 13)]:
            reference = self._reference(path, ss, to)
            for local_path, local in [(path, False), (path, True), (extensionless, True)]:
                with self.subTest(ss=ss, to=to, local=local, path=local_path.name):
                    actual = self._get_audio(local_path, info, ss, to, local=local)
                    self._assert_position_matches(actual, reference)

    def test_split_video_and_audio_inputs_keep_audio_from_second_input(self):
        audio_path, audio_info = self.sources['aac']
        video = self.root / 'video only.mp4'
        encoded = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
             'color=size=16x16:rate=1:duration=225', '-an', '-c:v', 'mpeg4', str(video)],
            capture_output=True, timeout=20)
        self.assertEqual(encoded.returncode, 0, encoded.stderr.decode(errors='replace'))
        info = {'url': str(video), 'ext': 'mkv', 'protocol': 'http', 'requested_formats': [
            {'url': str(video), 'ext': 'mp4', 'protocol': 'http', 'acodec': 'none'},
            audio_info,
        ]}
        actual = self._get_audio(video, info, 220.1, 223.1)
        self._assert_position_matches(actual, self._reference(audio_path, 220.1, 223.1))

    def test_multiple_audio_tracks_preserve_ffmpeg_default_selection(self):
        audio_path, _ = self.sources['aac']
        source = self.root / 'multiple tracks.mka'
        encoded = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(audio_path), '-f', 'lavfi', '-i',
             'sine=frequency=700:sample_rate=16000:duration=225', '-map', '0:a', '-map', '1:a',
             '-c:a', 'aac', '-disposition:a:0', '0', '-disposition:a:1', 'default', str(source)],
            capture_output=True, timeout=30)
        self.assertEqual(encoded.returncode, 0, encoded.stderr.decode(errors='replace'))
        info = {'url': str(source), 'ext': 'mkv', 'protocol': 'http'}
        actual = self._get_audio(source, info, 220, 222)
        reference = self._reference(source, 220, 222)
        self._assert_position_matches(actual, reference)
        first_second = actual[:SAMPLE_RATE]
        self.assertEqual(int(np.argmax(np.abs(np.fft.rfft(first_second)))), 700)


if __name__ == '__main__':
    unittest.main()
