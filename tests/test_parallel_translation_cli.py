import unittest
from types import SimpleNamespace
from unittest import mock

import tyro

from stream_translator_gpt import main
from stream_translator_gpt.main import Config, _validate_and_normalize


class ParallelTranslationCliTest(unittest.TestCase):

    def test_default_and_explicit_scheduler_options(self):
        default = tyro.cli(Config, args=['input'])
        explicit = tyro.cli(Config, args=['input', '--translation-threads', '7', '--use-old-translator'])
        self.assertEqual(default.translation_threads, 3)
        self.assertFalse(default.use_old_translator)
        self.assertEqual(explicit.translation_threads, 7)
        self.assertTrue(explicit.use_old_translator)

    def test_invalid_scheduler_values_are_rejected_before_initialization(self):
        for field, value, option in (
                ('translation_threads', 0, '--translation-threads'),
                ('translation_threads', -3, '--translation-threads'),
                ('translation_history_size', -1, '--translation-history-size'),
                ('translation_timeout', -1, '--translation-timeout')):
            with self.subTest(field=field, value=value):
                config = Config(url='input', **{field: value})
                with mock.patch.object(main.logger, 'error') as error, self.assertRaises(SystemExit) as raised:
                    _validate_and_normalize(config)
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(error.call_args.args[1], option)

    def test_zero_history_and_unlimited_total_timeout_are_valid(self):
        config = Config(url='input', translation_threads=1, translation_history_size=0,
                        translation_timeout=0, retry_if_translation_fails=True)
        _validate_and_normalize(config)
        self.assertEqual(config.translation_history_size, 0)
        self.assertEqual(config.translation_timeout, 0)

    def test_warning_policy_is_left_to_implementation_selection(self):
        config = Config(url='input', translation_prompt='translate', openai_api_key='test-key',
                        translation_threads=20, translation_history_size=0, use_translation_context=True,
                        use_old_translator=True)
        with mock.patch.object(main.logger, 'warning') as warning:
            _validate_and_normalize(config)
        warning.assert_not_called()

    def test_new_factory_composes_separate_components_for_both_providers(self):
        from stream_translator_gpt.translator import backends as translation_backends
        from stream_translator_gpt.translator import clients as translation_clients
        from stream_translator_gpt.translator import pipeline as translation_pipeline
        from stream_translator_gpt.translator import protocol as translation_protocol
        from stream_translator_gpt.translator import types as translation_types

        for google_key in (None, 'google-test-key'):
            with self.subTest(google=bool(google_key)):
                config = Config(url='input', openai_api_key='openai-test-key', google_api_key=google_key,
                                translation_prompt='Translate faithfully.', translation_threads=4,
                                translation_history_size=5, use_translation_context=True,
                                translation_timeout=15, retry_if_translation_fails=True,
                                openai_base_url='https://openai.example/v1',
                                google_base_url='https://gemini.example', processing_proxy='http://proxy.example')
                with mock.patch.object(translation_clients, 'TranslationClientPool') as pool, \
                        mock.patch.object(translation_protocol, 'TranslationProtocol') as protocol, \
                        mock.patch.object(translation_backends, 'OpenAITranslationBackend') as gpt, \
                        mock.patch.object(translation_backends, 'GeminiTranslationBackend') as gemini, \
                        mock.patch.object(translation_pipeline, 'TranslationPipeline') as pipeline, \
                        mock.patch.object(translation_types, 'TranslationSettings') as settings:
                    result = main._create_translator(config)

                backend, unused = (gemini, gpt) if google_key else (gpt, gemini)
                backend.assert_called_once()
                unused.assert_not_called()
                pool.assert_called_once_with(
                    provider='google' if google_key else 'openai', api_keys=google_key or 'openai-test-key',
                    base_url=config.google_base_url if google_key else config.openai_base_url,
                    proxy=config.processing_proxy, verify_ssl=True)
                protocol.assert_called_once_with(prompt=config.translation_prompt, history_size=5,
                                                 use_json_result=False, use_translation_context=True)
                self.assertIs(backend.call_args.kwargs['client_pool'], pool.return_value)
                self.assertIs(backend.call_args.kwargs['protocol'], protocol.return_value)
                self.assertNotIn('translation_threads', backend.call_args.kwargs)
                self.assertNotIn('timeout', backend.call_args.kwargs)
                settings.assert_called_once_with(workers=4, history_size=5, use_translation_context=True,
                                                 timeout=15, retry_if_translation_fails=True)
                pipeline.assert_called_once_with(backend.return_value, settings.return_value)
                self.assertIs(result, pipeline.return_value)

    def test_main_initializes_only_required_global_clients_and_correct_eof_owner(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                config = Config(url='input', use_modulate_transcription_api=True,
                                modulate_api_key='modulate-test-key', openai_api_key='openai-test-key',
                                google_api_key='google-test-key', translation_prompt='Translate.',
                                openai_base_url='https://openai.example/v1', use_old_translator=legacy)
                getter = SimpleNamespace(loop=mock.Mock())
                translator = SimpleNamespace(loop=mock.Mock(), stop=mock.Mock())
                thread = mock.Mock()
                thread.is_alive.return_value = False
                with mock.patch.object(main, 'ClientPool') as client_pool, \
                        mock.patch.object(main, '_create_audio_getter', return_value=getter), \
                        mock.patch.object(main, 'ModulateStreamingTranscriber'), \
                        mock.patch.object(main, 'ResultExporter'), \
                        mock.patch.object(main, '_create_translator', return_value=translator) as factory, \
                        mock.patch.object(main, 'start_daemon_thread', return_value=thread) as start:
                    main.run(config)

                factory.assert_called_once_with(config)
                kwargs = client_pool.init.call_args.kwargs
                self.assertEqual(kwargs['openai_api_key'], 'openai-test-key' if legacy else None)
                self.assertEqual(kwargs['google_api_key'], 'google-test-key' if legacy else None)
                self.assertEqual(kwargs['whisper_api_key'], None if legacy else 'openai-test-key')
                self.assertEqual(kwargs['whisper_base_url'], None if legacy else config.openai_base_url)
                self.assertNotIn('legacy_translation_clients', kwargs)
                translator.stop.assert_not_called()
                worker = next(call for call in start.call_args_list if call.kwargs['name'] == 'translator')
                self.assertEqual(worker.kwargs['output_eof_on_error'], legacy)

    def test_transcription_only_does_not_construct_translation_components(self):
        config = Config(url='input', translation_threads=8, use_old_translator=True)
        self.assertIsNone(main._create_translator(config))


if __name__ == '__main__':
    unittest.main()
