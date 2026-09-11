import os
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

import tyro

from stream_translator_gpt.getter.audio_getter import StreamAudioGetter, _open_stream
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


class StreamFormatDiagnosticCliTest(unittest.TestCase):

    @staticmethod
    def _parse(*args):
        return tyro.cli(Config, args=_preprocess_deprecated_flags(args))

    def test_option_is_disabled_by_default_and_accepts_a_file(self):
        disabled = self._parse('https://example.test/live')
        enabled = self._parse(
            'https://example.test/live',
            '--stream-format-diagnostic-file',
            'selected-format.txt',
        )

        self.assertIsNone(disabled.stream_format_diagnostic_file)
        self.assertEqual(enabled.stream_format_diagnostic_file, 'selected-format.txt')


class WaitForVideoWiringTest(unittest.TestCase):

    def test_stream_getter_receives_wait_and_live_start_options(self):
        sentinel = object()
        config = Config(url='https://example.test/live',
                        wait_for_video=30,
                        live_from_start=True,
                        stream_format_diagnostic_file='selected-format.txt')

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
            stream_format_diagnostic_file='selected-format.txt',
        )

    @staticmethod
    def _open_and_get_popen(wait_for_video, live_from_start=False, stream_format_diagnostic_file=None):
        ytdlp_process = mock.Mock(pid=11)
        ffmpeg_process = mock.Mock(pid=12)
        ffmpeg_builder = mock.Mock()
        ffmpeg_builder.output.return_value.run_async.return_value = ffmpeg_process
        transport_thread = mock.Mock()

        with mock.patch('stream_translator_gpt.getter.audio_getter.subprocess.Popen',
                        return_value=ytdlp_process) as popen, \
                mock.patch('stream_translator_gpt.getter.audio_getter.ffmpeg.input',
                           return_value=ffmpeg_builder), \
                mock.patch('stream_translator_gpt.getter.audio_getter.threading.Thread',
                           return_value=transport_thread):
            _open_stream(
                'https://example.test/live',
                'bestaudio',
                'cookies.txt',
                'http://proxy.test',
                'workdir',
                wait_for_video=wait_for_video,
                live_from_start=live_from_start,
                stream_format_diagnostic_file=stream_format_diagnostic_file,
            )

        transport_thread.start.assert_called_once_with()
        return popen.call_args

    def test_disabled_option_is_not_passed_to_ytdlp(self):
        popen_call = self._open_and_get_popen(None)
        command = popen_call.args[0]

        self.assertNotIn('--wait-for-video', command)
        self.assertNotIn('--live-from-start', command)
        self.assertIn('--quiet', command)
        self.assertNotIn('--print-to-file', command)
        self.assertEqual(popen_call.kwargs['stdout'], subprocess.PIPE)
        self.assertIsNone(popen_call.kwargs['stderr'])
        self.assertEqual(popen_call.kwargs['start_new_session'], os.name != 'nt')

    def test_empty_format_diagnostic_file_is_not_passed_to_ytdlp(self):
        for diagnostic_file in ('', '   '):
            with self.subTest(diagnostic_file=diagnostic_file):
                popen_call = self._open_and_get_popen(
                    None,
                    stream_format_diagnostic_file=diagnostic_file,
                )
                self.assertNotIn('--print-to-file', popen_call.args[0])

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

    def test_explicit_format_diagnostic_file_is_passed_to_ytdlp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostic_file = os.path.join(temp_dir, 'selected-format.txt')
            with open(diagnostic_file, 'w', encoding='utf-8') as existing_file:
                existing_file.write('stale format metadata')
            popen_call = self._open_and_get_popen(
                None,
                stream_format_diagnostic_file=diagnostic_file,
            )
            with open(diagnostic_file, encoding='utf-8') as reset_file:
                self.assertEqual(reset_file.read(), '')
            command = popen_call.args[0]

            diagnostic_index = command.index('--print-to-file')
            diagnostic_template = command[diagnostic_index + 1]
            self.assertTrue(diagnostic_template.startswith('before_dl:format_id='))
            self.assertNotIn('%(url)', diagnostic_template)
            # The launcher resolves symlinks as well as making the path
            # absolute (e.g. macOS /var -> /private/var).
            self.assertEqual(command[diagnostic_index + 2], os.path.realpath(diagnostic_file))
            self.assertTrue(os.path.samefile(command[diagnostic_index + 2], diagnostic_file))

    def test_format_diagnostic_file_normalizes_parent_components_and_native_symlinks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostic_dir = os.path.join(temp_dir, 'diagnostic outputs')
            nested_dir = os.path.join(diagnostic_dir, 'nested')
            os.makedirs(nested_dir)
            diagnostic_file = os.path.join(diagnostic_dir, 'selected-format.txt')
            paths = [os.path.join(nested_dir, '..', 'selected-format.txt')]
            if os.name == 'posix':
                # POSIX supports directory symlinks without the extra
                # privileges/developer mode that Windows may require.
                linked_dir = os.path.join(temp_dir, 'diagnostic link')
                os.symlink(diagnostic_dir, linked_dir, target_is_directory=True)
                paths.append(os.path.join(linked_dir, 'selected-format.txt'))

            for supplied_path in paths:
                with self.subTest(path=supplied_path):
                    with open(diagnostic_file, 'w', encoding='utf-8') as existing_file:
                        existing_file.write('stale format metadata')
                    popen_call = self._open_and_get_popen(
                        None,
                        stream_format_diagnostic_file=supplied_path,
                    )
                    command = popen_call.args[0]
                    actual_path = command[command.index('--print-to-file') + 2]

                    self.assertTrue(os.path.isabs(actual_path))
                    self.assertEqual(actual_path, os.path.realpath(diagnostic_file))
                    self.assertTrue(os.path.samefile(actual_path, supplied_path))
                    with open(diagnostic_file, encoding='utf-8') as reset_file:
                        self.assertEqual(reset_file.read(), '')

    def test_sigterm_uses_a_nonzero_exit_status(self):
        getter = StreamAudioGetter(
            url='https://example.test/live',
            format='bestaudio',
            cookies='',
            proxy='',
        )
        getter.started.set()

        with mock.patch.object(getter.finished, 'wait', return_value=True) as wait_for_cleanup, \
                mock.patch('stream_translator_gpt.getter.audio_getter.sys.exit') as exit_process:
            getter._exit_handler(signal.SIGTERM, None)

        wait_for_cleanup.assert_called_once_with(timeout=10.0)
        exit_process.assert_called_once_with(128 + signal.SIGTERM)

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

        with mock.patch('stream_translator_gpt.getter.audio_getter.sys.exit') as exit_process:
            getter._exit_handler(signal.SIGINT, None)

        self.assertTrue(getter.stop_event.is_set())
        getter.ffmpeg_process.kill.assert_called_once_with()
        getter.ytdlp_process.kill.assert_called_once_with()
        self.assertTrue(os.path.exists(temp_dir))
        exit_process.assert_called_once_with(0)
        getter.__del__()
        self.assertFalse(os.path.exists(temp_dir))


if __name__ == '__main__':
    unittest.main()
