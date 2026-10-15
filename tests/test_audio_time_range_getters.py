import io
import math
import queue
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from stream_translator_gpt.getter.audio_getter import (
    LocalFileAudioGetter,
    StreamAudioGetter,
    _open_stream,
    _StreamTransportStats,
)
from stream_translator_gpt.utils.common import SAMPLE_RATE, SAMPLES_PER_FRAME


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
            (None, None, {'loglevel': 'panic'}, {}),
            ('12.5', None, {'loglevel': 'panic', 'ss': '12.5'}, {}),
            (None, '1:02.5', {'loglevel': 'panic'}, {'t': '62.5'}),
            ('1:00', '1:02.5', {'loglevel': 'panic', 'ss': '60'}, {'t': '2.5'}),
            (0, 1, {'loglevel': 'panic', 'ss': '0'}, {'t': '1'}),
        ]
        for ss, to, expected_input, expected_output in cases:
            with self.subTest(ss=ss, to=to):
                ffmpeg_input, builder, output, getter = self._run_mocked(ss, to)
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
        self.assertEqual(command[command.index('-ss') + 1], '60')
        self.assertEqual(command[command.index('-t') + 1], '2.5')
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

    def test_download_sections_encodes_absolute_boundaries_only_once(self):
        for ss, to, section in [
            (None, None, None),
            ('12.5', None, '*12.5-inf'),
            (None, '1:02.5', '*0-62.5'),
            ('1:00', '1:02.5', '*60-62.5'),
            (0, 1, '*0-1'),
        ]:
            with self.subTest(ss=ss, to=to):
                command, ffmpeg_input, builder = self._open_mocked(ss, to)
                if section is None:
                    self.assertNotIn('--download-sections', command)
                    self.assertNotIn('--downloader-args', command)
                else:
                    self.assertEqual(command.count('--download-sections'), 1)
                    self.assertEqual(command[command.index('--download-sections') + 1], section)
                    self.assertEqual(command[command.index('--downloader-args') + 1],
                                     f'ffmpeg_o:-vn -c:a pcm_f32le -ar {SAMPLE_RATE} -ac 1 -f matroska')
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
from yt_dlp import YoutubeDL
from yt_dlp.downloader.external import FFmpegFD
params = {
    'quiet': False,
    'enable_file_urls': True,
    'external_downloader_args': {
        'ffmpeg_o': ['-vn', '-c:a', 'pcm_f32le', '-ar', sys.argv[4], '-ac', '1', '-f', 'matroska'],
    },
}
info = {
    'url': sys.argv[1], 'ext': sys.argv[2], 'protocol': sys.argv[3],
    'section_start': 0.2, 'section_end': 0.5,
}
with YoutubeDL(params) as ydl:
    sys.exit(FFmpegFD(ydl, params)._call_downloader('-', info))
'''
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
                         'mp4' if extension == 'm3u8' else extension, protocol, str(SAMPLE_RATE)],
                        capture_output=True, timeout=15, check=False)
                    self.assertEqual(section.returncode, 0, section.stderr.decode(errors='replace'))
                    self.assertTrue(section.stdout)
                    decoded = subprocess.run(
                        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0', '-f', 'f32le',
                         '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1'],
                        input=section.stdout, capture_output=True, timeout=15, check=False)
                    self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors='replace'))
                    reference = subprocess.run(
                        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', '0.2', '-i', str(source),
                         '-t', '0.3', '-vn', '-c:a', 'pcm_f32le', '-ar', str(SAMPLE_RATE), '-ac', '1',
                         '-f', 'f32le', 'pipe:1'],
                        capture_output=True, timeout=15, check=False)
                    self.assertEqual(reference.returncode, 0, reference.stderr.decode(errors='replace'))
                    # Native FFmpeg can output fewer samples for AAC HLS near
                    # codec priming/segment timestamp boundaries. Preserve its
                    # semantics rather than inventing missing samples or using
                    # a loose duration tolerance in this regression test.
                    self.assertTrue(reference.stdout)
                    self.assertEqual(len(decoded.stdout), len(reference.stdout))
                    # Input-side and output-side duration limits flush the
                    # resampler differently at EOF. Compare the first 50ms
                    # exactly enough to catch an incorrect seek as well.
                    prefix_samples = SAMPLE_RATE // 20
                    np.testing.assert_allclose(np.frombuffer(decoded.stdout, np.float32)[:prefix_samples],
                                               np.frombuffer(reference.stdout, np.float32)[:prefix_samples],
                                               atol=1e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
