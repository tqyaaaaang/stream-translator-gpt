import contextlib
import dataclasses
import math
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import tyro

from stream_translator_gpt import main
from stream_translator_gpt.main import Config


STREAMING_BACKENDS = (
    ({'use_modulate_transcription_api': True, 'modulate_api_key': 'modulate-test-key'},
     'ModulateStreamingTranscriber'),
    ({'use_elevenlabs_scribe_api': True, 'elevenlabs_api_key': 'scribe-test-key'},
     'ElevenLabsScribeTranscriber'),
)

LEGACY_BACKENDS = (
    ({}, 'OpenaiWhisper'),
    ({'use_faster_whisper': True}, 'FasterWhisper'),
    ({'use_simul_streaming': True}, 'SimulStreaming'),
    ({'use_openai_transcription_api': True, 'whisper_api_key': 'whisper-test-key'},
     'RemoteOpenaiTranscriber'),
    ({'use_hf_asr': True}, 'HFTranscriber'),
)


class TranscribeReconnectConfigTest(unittest.TestCase):

    def test_canonical_defaults_are_shared_without_duplicate_legacy_fields(self):
        config = tyro.cli(Config, args=['clip.mp3'])

        self.assertEqual(config.transcribe_reconnect_attempts, 3)
        self.assertEqual(config.transcribe_reconnect_backoff_seconds, 1.0)
        field_names = {field.name for field in dataclasses.fields(config)}
        self.assertNotIn('modulate_reconnect_attempts', field_names)
        self.assertNotIn('modulate_reconnect_backoff_seconds', field_names)

    def test_parser_accepts_canonical_options_and_deprecated_aliases_directly(self):
        for prefix in ('transcribe', 'modulate'):
            for separator in ('-', '_'):
                for inline in (False, True):
                    attempts = '--' + f'{prefix}-reconnect-attempts'.replace('-', separator)
                    backoff = '--' + f'{prefix}-reconnect-backoff-seconds'.replace('-', separator)
                    args = ([f'{attempts}=7', f'{backoff}=0.25'] if inline
                            else [attempts, '7', backoff, '0.25'])
                    with self.subTest(prefix=prefix, separator=separator, inline=inline):
                        config = tyro.cli(Config, args=['clip.mp3', *args])
                        self.assertEqual(config.transcribe_reconnect_attempts, 7)
                        self.assertEqual(config.transcribe_reconnect_backoff_seconds, 0.25)

    def test_preprocessor_warns_and_rewrites_each_deprecated_option(self):
        for separator in ('-', '_'):
            for inline in (False, True):
                attempts = '--' + 'modulate-reconnect-attempts'.replace('-', separator)
                backoff = '--' + 'modulate-reconnect-backoff-seconds'.replace('-', separator)
                args = ([f'{attempts}=4', f'{backoff}=0.5'] if inline
                        else [attempts, '4', backoff, '0.5'])
                expected = (['--transcribe-reconnect-attempts=4', '--transcribe-reconnect-backoff-seconds=0.5']
                            if inline else ['--transcribe-reconnect-attempts', '4',
                                            '--transcribe-reconnect-backoff-seconds', '0.5'])
                with self.subTest(separator=separator, inline=inline), \
                        mock.patch.object(main.logger, 'warning') as warning:
                    processed = main._preprocess_deprecated_flags(['clip.mp3', *args])
                self.assertEqual(processed, ['clip.mp3', *expected])
                self.assertEqual(warning.call_count, 2)
                messages = '\n'.join(str(call) for call in warning.call_args_list)
                self.assertIn(attempts, messages)
                self.assertIn(backoff, messages)
                for suffix in ('attempts', 'backoff-seconds'):
                    self.assertIn(f'--transcribe-reconnect-{suffix}', messages)
                self.assertIn('deprecated', messages.lower())

    def test_canonical_options_do_not_produce_deprecation_warnings(self):
        args = ['clip.mp3', '--transcribe-reconnect-attempts=0',
                '--transcribe-reconnect-backoff-seconds', '0']
        with mock.patch.object(main.logger, 'warning') as warning:
            processed = main._preprocess_deprecated_flags(args)
        self.assertEqual(processed, args)
        warning.assert_not_called()

    def test_last_occurrence_wins_when_new_and_old_names_are_mixed(self):
        for first, second in (('modulate', 'transcribe'), ('transcribe', 'modulate')):
            for preprocess in (False, True):
                args = ['clip.mp3', f'--{first}-reconnect-attempts', '8',
                        f'--{first}-reconnect-backoff-seconds=2',
                        '--' + f'{second}_reconnect_attempts=0',
                        '--' + f'{second}_reconnect_backoff_seconds', '0.125']
                with self.subTest(first=first, preprocess=preprocess), \
                        mock.patch.object(main.logger, 'warning'):
                    if preprocess:
                        args = main._preprocess_deprecated_flags(args)
                    config = tyro.cli(Config, args=args)
                self.assertEqual(config.transcribe_reconnect_attempts, 0)
                self.assertEqual(config.transcribe_reconnect_backoff_seconds, 0.125)

    def test_cli_entrypoint_applies_legacy_options_and_warnings_for_scribe(self):
        argv = ['stream-translator-gpt', 'clip.mp3', '--use-elevenlabs-scribe-api',
                '--elevenlabs-api-key', 'scribe-test-key',
                '--modulate_reconnect_attempts=0', '--modulate-reconnect-backoff-seconds', '0']
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(main, 'configure_logging'), \
                mock.patch.object(main, 'shutdown_logging'), \
                mock.patch.object(main, '_check_ffmpeg'), \
                mock.patch.object(main.logger, 'warning') as warning, \
                mock.patch.object(main, 'run') as run:
            main.cli()

        run.assert_called_once()
        config = run.call_args.args[0]
        self.assertTrue(config.use_elevenlabs_scribe_api)
        self.assertEqual(config.transcribe_reconnect_attempts, 0)
        self.assertEqual(config.transcribe_reconnect_backoff_seconds, 0.0)
        messages = '\n'.join(str(call) for call in warning.call_args_list)
        self.assertIn('--modulate_reconnect_attempts', messages)
        self.assertIn('--modulate-reconnect-backoff-seconds', messages)
        self.assertIn('deprecated', messages.lower())

    def test_both_streaming_backends_accept_zero_and_positive_values(self):
        for options, backend in STREAMING_BACKENDS:
            for attempts, backoff in ((0, 0.0), (6, 0.25)):
                config = Config(url='clip.mp3', pacer='realtime',
                                transcribe_reconnect_attempts=attempts,
                                transcribe_reconnect_backoff_seconds=backoff, **options)
                with self.subTest(backend=backend, attempts=attempts, backoff=backoff):
                    main._validate_and_normalize(config)
                    self.assertEqual(config.transcribe_reconnect_attempts, attempts)
                    self.assertEqual(config.transcribe_reconnect_backoff_seconds, backoff)

    def test_both_streaming_backends_reject_negative_attempts_and_invalid_backoff(self):
        invalid_values = [(-1, 1.0), (3, -0.1), (3, math.nan), (3, math.inf), (3, -math.inf)]
        for options, backend in STREAMING_BACKENDS:
            for attempts, backoff in invalid_values:
                config = Config(url='clip.mp3', pacer='realtime',
                                transcribe_reconnect_attempts=attempts,
                                transcribe_reconnect_backoff_seconds=backoff, **options)
                with self.subTest(backend=backend, attempts=attempts, backoff=backoff), \
                        mock.patch.object(main.logger, 'error'), self.assertRaises(SystemExit) as raised:
                    main._validate_and_normalize(config)
                self.assertEqual(raised.exception.code, 1)

    def test_unrelated_backends_do_not_validate_unused_reconnect_options(self):
        for options, backend in LEGACY_BACKENDS:
            config = Config(url='clip.mp3', transcribe_reconnect_attempts=-1,
                            transcribe_reconnect_backoff_seconds=math.nan, **options)
            with self.subTest(backend=backend):
                main._validate_and_normalize(config)


class TranscribeReconnectPipelineTest(unittest.TestCase):

    def _constructor_calls(self, config):
        thread = mock.Mock()
        thread.is_alive.return_value = False
        constructors = {}
        with contextlib.ExitStack() as stack:
            for name in ('ClientPool', '_create_audio_getter', '_create_translator', 'ResultExporter'):
                stack.enter_context(mock.patch.object(main, name))
            stack.enter_context(mock.patch.object(main.signal, 'signal'))
            stack.enter_context(mock.patch.object(main, 'start_daemon_thread', return_value=thread))
            stack.enter_context(mock.patch.dict(sys.modules, {
                'stream_translator_gpt.slicer.audio_slicer': SimpleNamespace(AudioSlicer=mock.Mock()),
            }))
            for _, name in (*STREAMING_BACKENDS, *LEGACY_BACKENDS):
                constructors[name] = stack.enter_context(mock.patch.object(main, name))
            main.run(config)
        return constructors

    def test_canonical_values_are_passed_to_both_streaming_backend_constructors(self):
        for options, backend in STREAMING_BACKENDS:
            for attempts, backoff in ((0, 0.0), (7, 0.375)):
                config = Config(url='clip.mp3', pacer='realtime',
                                transcribe_reconnect_attempts=attempts,
                                transcribe_reconnect_backoff_seconds=backoff, **options)
                with self.subTest(backend=backend, attempts=attempts, backoff=backoff):
                    constructors = self._constructor_calls(config)
                    constructors[backend].assert_called_once()
                    kwargs = constructors[backend].call_args.kwargs
                    self.assertEqual(kwargs['reconnect_attempts'], attempts)
                    self.assertEqual(kwargs['reconnect_backoff_seconds'], backoff)
                    self.assertNotIn('transcribe_reconnect_attempts', kwargs)
                    self.assertNotIn('transcribe_reconnect_backoff_seconds', kwargs)
                    for name, constructor in constructors.items():
                        if name != backend:
                            constructor.assert_not_called()

    def test_other_backend_constructor_arguments_are_unchanged(self):
        for options, backend in LEGACY_BACKENDS:
            baseline = Config(url='clip.mp3', **options)
            configured = dataclasses.replace(baseline, transcribe_reconnect_attempts=9,
                                             transcribe_reconnect_backoff_seconds=0.125)
            with self.subTest(backend=backend):
                default_calls = self._constructor_calls(baseline)
                configured_calls = self._constructor_calls(configured)
                default_calls[backend].assert_called_once()
                configured_calls[backend].assert_called_once()
                self.assertEqual(default_calls[backend].call_args, configured_calls[backend].call_args)
                kwargs = configured_calls[backend].call_args.kwargs
                self.assertNotIn('reconnect_attempts', kwargs)
                self.assertNotIn('reconnect_backoff_seconds', kwargs)


if __name__ == '__main__':
    unittest.main()
