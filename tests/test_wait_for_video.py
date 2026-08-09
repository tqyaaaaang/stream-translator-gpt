import os
import signal
import subprocess
import unittest
from unittest import mock

import tyro

from stream_translator_gpt.audio_getter import StreamAudioGetter, _open_stream
from stream_translator_gpt.main import (
    DEFAULT_WAIT_FOR_VIDEO_SECONDS,
    Config,
    _create_audio_getter,
    _preprocess_deprecated_flags,
    _validate_and_normalize,
)


class WaitForVideoCliTest(unittest.TestCase):

    @staticmethod
    def _parse(*args):
        return tyro.cli(Config, args=_preprocess_deprecated_flags(args))

    def test_omitted_option_stays_disabled(self):
        self.assertIsNone(self._parse('https://example.test/live').wait_for_video)

    def test_bare_option_uses_sixty_seconds(self):
        after_url = self._parse('https://example.test/live', '--wait-for-video')
        before_url = self._parse('--wait-for-video', 'https://example.test/live')

        self.assertEqual(after_url.wait_for_video, DEFAULT_WAIT_FOR_VIDEO_SECONDS)
        self.assertEqual(before_url.wait_for_video, DEFAULT_WAIT_FOR_VIDEO_SECONDS)

    def test_explicit_and_equals_values_are_preserved(self):
        separated = self._parse('https://example.test/live', '--wait-for-video', '30')
        equals = self._parse('https://example.test/live', '--wait-for-video=45')
        underscore = self._parse('https://example.test/live', '--wait_for_video', '90')

        self.assertEqual(separated.wait_for_video, 30)
        self.assertEqual(equals.wait_for_video, 45)
        self.assertEqual(underscore.wait_for_video, 90)

    def test_bare_option_before_another_option_does_not_consume_it(self):
        config = self._parse(
            'https://example.test/live',
            '--wait-for-video',
            '--format',
            '251',
        )

        self.assertEqual(config.wait_for_video, DEFAULT_WAIT_FOR_VIDEO_SECONDS)
        self.assertEqual(config.format, '251')

    def test_non_positive_interval_is_rejected(self):
        for interval in ('0', '-1'):
            with self.subTest(interval=interval):
                config = self._parse('https://example.test/live', '--wait-for-video', interval)
                with self.assertRaises(SystemExit):
                    _validate_and_normalize(config)


class LiveFromStartCliTest(unittest.TestCase):

    @staticmethod
    def _parse(*args):
        return tyro.cli(Config, args=_preprocess_deprecated_flags(args))

    def test_option_is_a_disabled_by_default_bare_flag(self):
        disabled = self._parse('https://example.test/live')
        after_url = self._parse('https://example.test/live', '--live-from-start')
        before_url = self._parse('--live-from-start', 'https://example.test/live')
        before_option = self._parse('https://example.test/live', '--live-from-start', '--format', '251')

        self.assertFalse(disabled.live_from_start)
        self.assertTrue(after_url.live_from_start)
        self.assertTrue(before_url.live_from_start)
        self.assertTrue(before_option.live_from_start)
        self.assertEqual(before_option.format, '251')

    def test_option_does_not_accept_a_value(self):
        invalid_options = [
            ('--live-from-start', 'true'),
            ('--live-from-start=true',),
        ]
        for option in invalid_options:
            with self.subTest(option=option), self.assertRaises(SystemExit):
                self._parse('https://example.test/live', *option)


class WaitForVideoWiringTest(unittest.TestCase):

    def test_stream_getter_receives_wait_and_live_start_options(self):
        sentinel = object()
        config = Config(url='https://example.test/live', wait_for_video=30, live_from_start=True)

        with mock.patch('stream_translator_gpt.main.StreamAudioGetter', return_value=sentinel) as getter_cls:
            result = _create_audio_getter(config)

        self.assertIs(result, sentinel)
        getter_cls.assert_called_once_with(
            url=config.url,
            format=config.format,
            cookies=config.cookies,
            proxy=config.input_proxy,
            realtime=False,
            wait_for_video=30,
            live_from_start=True,
        )

    @staticmethod
    def _open_and_get_popen(wait_for_video, live_from_start=False):
        ytdlp_process = mock.Mock(pid=11)
        ffmpeg_process = mock.Mock(pid=12)
        ffmpeg_builder = mock.Mock()
        ffmpeg_builder.output.return_value.run_async.return_value = ffmpeg_process
        transport_thread = mock.Mock()

        with mock.patch('stream_translator_gpt.audio_getter.subprocess.Popen',
                        return_value=ytdlp_process) as popen, \
                mock.patch('stream_translator_gpt.audio_getter.ffmpeg.input',
                           return_value=ffmpeg_builder), \
                mock.patch('stream_translator_gpt.audio_getter.threading.Thread',
                           return_value=transport_thread):
            _open_stream(
                'https://example.test/live',
                'bestaudio',
                'cookies.txt',
                'http://proxy.test',
                'workdir',
                wait_for_video=wait_for_video,
                live_from_start=live_from_start,
            )

        transport_thread.start.assert_called_once_with()
        return popen.call_args

    def test_disabled_option_is_not_passed_to_ytdlp(self):
        popen_call = self._open_and_get_popen(None)
        command = popen_call.args[0]

        self.assertNotIn('--wait-for-video', command)
        self.assertNotIn('--live-from-start', command)
        self.assertIn('--quiet', command)
        diagnostic_index = command.index('--print-to-file')
        diagnostic_template = command[diagnostic_index + 1]
        self.assertTrue(diagnostic_template.startswith('before_dl:format_id='))
        self.assertNotIn('%(url)', diagnostic_template)
        self.assertEqual(command[diagnostic_index + 2], os.path.join('workdir', 'yt-dlp-stream-format.txt'))
        self.assertEqual(popen_call.kwargs['stdout'], subprocess.PIPE)
        self.assertIsNone(popen_call.kwargs['stderr'])
        self.assertEqual(popen_call.kwargs['start_new_session'], os.name != 'nt')

    def test_live_from_start_is_passed_once_and_coexists_with_waiting(self):
        popen_call = self._open_and_get_popen(30, live_from_start=True)
        command = popen_call.args[0]

        self.assertEqual(command.count('--live-from-start'), 1)
        wait_index = command.index('--wait-for-video')
        self.assertEqual(command[wait_index + 1], '30')
        self.assertIn('--quiet', command)
        self.assertEqual(popen_call.kwargs['stdout'], subprocess.PIPE)
        self.assertIsNone(popen_call.kwargs['stderr'])

    def test_enabled_option_is_passed_without_mixing_diagnostics_into_audio(self):
        popen_call = self._open_and_get_popen(30)
        command = popen_call.args[0]
        option_index = command.index('--wait-for-video')

        self.assertEqual(command[option_index + 1], '30')
        self.assertEqual(command.count('--wait-for-video'), 1)
        self.assertIn('--quiet', command)
        self.assertEqual(popen_call.kwargs['stdout'], subprocess.PIPE)
        self.assertIsNone(popen_call.kwargs['stderr'])

    def test_signal_stops_children_while_waiting_for_stream(self):
        getter = StreamAudioGetter(
            url='https://example.test/live',
            format='bestaudio',
            cookies='',
            proxy='',
            wait_for_video=60,
        )
        temp_dir = getter.temp_dir
        getter.ffmpeg_process = mock.Mock()
        getter.ytdlp_process = mock.Mock()

        with mock.patch('stream_translator_gpt.audio_getter.sys.exit') as exit_process:
            getter._exit_handler(signal.SIGINT, None)

        self.assertTrue(getter.stop_event.is_set())
        getter.ffmpeg_process.kill.assert_called_once_with()
        getter.ytdlp_process.kill.assert_called_once_with()
        self.assertFalse(os.path.exists(temp_dir))
        exit_process.assert_called_once_with(0)


if __name__ == '__main__':
    unittest.main()
