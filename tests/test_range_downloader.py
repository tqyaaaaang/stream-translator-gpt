import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

import numpy as np
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from stream_translator_gpt.getter.range_downloader import (
    RangeDownloadPP,
    configure_range_downloader,
    parse_range_options,
)


class RangeDownloaderTest(unittest.TestCase):
    def test_protocol_selection_and_no_arguments_leak_between_playlist_entries(self):
        params = {'external_downloader_args': {'curl': ['--retry', '2'],
                                               'ffmpeg_i1': ['-ss', '999'], 'ffmpeg_o1': ['-t', '999']}}
        cases = [
            ({'protocol': 'https'}, False, 'indexed'),
            ({'protocol': 'm3u8_native'}, False, 'fast'),
            ({'protocol': 'm3u8_native'}, True, 'linear'),
            ({'protocol': 'm3u8_native'}, False, 'fast'),
            ({'protocol': 'http_dash_segments'}, True, 'indexed'),
            ({'url': 'https://example.test/stream.M3U8?token=secret'}, False, 'fast'),
            ({'url': 'https://example.test/stream.MPEGTS?token=secret'}, True, 'linear'),
            ({'ext': 'ts'}, False, 'fast'),
            ({'protocol': 'https+https', 'requested_formats': [
                {'protocol': 'https'}, {'protocol': 'm3u8'}]}, True, 'linear'),
            ({'protocol': 'https'}, False, 'indexed'),
        ]
        for info, strict, seek_mode in cases:
            with self.subTest(info=info, strict=strict):
                self.assertEqual(configure_range_downloader(params, info, 220, 223, strict), seek_mode)
                args = params['external_downloader_args']
                self.assertEqual(args['curl'], ['--retry', '2'])
                self.assertNotIn('ffmpeg_i1', args)
                self.assertNotIn('ffmpeg_o1', args)
                expected_input = {'linear': [], 'fast': ['-ss', '220'],
                                  'indexed': ['-copyts', '-start_at_zero', '-ss', '210']}
                self.assertEqual(args['ffmpeg_i'], expected_input[seek_mode])
                output_range = {'linear': ['-ss', '220'], 'fast': [],
                                'indexed': ['-af', 'atrim=start=220,asetpts=PTS-STARTPTS']}[seek_mode]
                self.assertEqual(args['ffmpeg_o'], ['-vn', '-c:a', 'pcm_f32le', '-ar', '16000', '-ac', '1',
                                                    *output_range, '-t', '3', '-f', 'matroska'])

    def test_parser_preserves_transport_options_and_replaces_inherited_section_ranges(self):
        ss, to, strict, parsed = parse_range_options([
            '220', '', 'strict', '--', '--ignore-config', 'https://example.test/watch?v=1', '-o', '-', '--quiet',
            '--cookies', 'cookies.txt', '--proxy', 'http://127.0.0.1:8080', '--wait-for-video', '60',
            '--download-sections', '*2-3', '--print-to-file', 'before_dl:%(protocol)s', 'diagnostic.txt',
        ])
        self.assertEqual((ss, to), (220, None))
        self.assertTrue(strict)
        self.assertEqual(parsed.urls, ['https://example.test/watch?v=1'])
        params = parsed.ydl_opts
        self.assertEqual(params['cookiefile'], 'cookies.txt')
        self.assertEqual(params['proxy'], 'http://127.0.0.1:8080')
        self.assertIsNotNone(params['wait_for_video'])
        self.assertNotIn('download_ranges', params)
        self.assertEqual(params['outtmpl']['default'], '-')
        self.assertTrue(params['quiet'])
        self.assertTrue(params['forceprint'] == {})
        self.assertIn('before_dl', params['print_to_file'])

    def test_invalid_boundaries_fail_before_yt_dlp_or_network_access(self):
        for argv in [[], ['', '', 'fast', '--'], ['-1', '', 'fast', '--'],
                     ['2', '1', 'strict', '--'], ['1', '2', 'fast', '-'],
                     ['1', '2', '--'], ['1', '2', 'unknown', '--']]:
            with self.subTest(argv=argv), mock.patch(
                    'stream_translator_gpt.getter.range_downloader.parse_options') as parse:
                with self.assertRaises(ValueError):
                    parse_range_options(argv)
                parse.assert_not_called()

    def test_unsupported_native_fallback_fails_instead_of_returning_uncut_media(self):
        with YoutubeDL({'quiet': True}) as downloader:
            hook = RangeDownloadPP(downloader, 220, 222)
            with mock.patch('stream_translator_gpt.getter.range_downloader.get_suitable_downloader',
                            return_value=object), self.assertRaises(DownloadError):
                # Exercise the actual before_dl error path, not only run(): a
                # deferred error here would permit an uncut native download.
                downloader.add_post_processor(hook, when='before_dl')
                downloader.pre_process({'protocol': 'unsupported', 'url': 'https://example.test/audio'}, 'before_dl')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for the offline child-process test')
    def test_real_child_entrypoint_from_temp_cwd_outputs_only_cropped_media(self):
        # Use saved extraction metadata and a local media path: exercise the real
        # before_dl hook and FFmpeg downloader without any external request.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / '音声 source.wav'
            samples = (np.arange(16000, dtype=np.int32) - 8000).astype('<i2')
            with wave.open(str(source), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(samples.tobytes())
            info = root / 'info.json'
            info.write_text(json.dumps({
                'id': 'offline', 'title': 'offline', 'extractor': 'generic', 'extractor_key': 'Generic',
                # FFmpeg's file: URL parser doesn't percent-decode paths.
                'url': unquote(source.as_uri()),
                'protocol': 'https', 'ext': 'wav', 'format_id': 'test', 'acodec': 'pcm_s16le',
                'vcodec': 'none', 'duration': 1,
            }), encoding='utf-8')
            diagnostic = root / 'format.txt'
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
            result = subprocess.run([
                sys.executable, '-B', '-m', 'stream_translator_gpt.getter.range_downloader',
                '0.2', '0.5', 'fast', '--', '--ignore-config', '--enable-file-urls',
                '--load-info-json', str(info), '-o', '-', '--quiet', '-f', 'b',
                '--print-to-file', 'before_dl:%(protocol)s', str(diagnostic),
            ], cwd=directory, env=environment, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            self.assertTrue(result.stdout.startswith(bytes.fromhex('1a45dfa3')), 'stdout must start with Matroska')
            decoded = subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
                '-f', 'f32le', '-ar', '16000', '-ac', '1', 'pipe:1',
            ], input=result.stdout, capture_output=True, timeout=10)
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors='replace'))
            np.testing.assert_array_equal(np.frombuffer(decoded.stdout, dtype='<f4'),
                                          samples[3200:8000].astype(np.float32) / 32768)
            self.assertEqual(diagnostic.read_text(encoding='utf-8').strip(), 'https')


if __name__ == '__main__':
    unittest.main()
