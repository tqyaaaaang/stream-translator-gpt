import unittest
from types import SimpleNamespace
from unittest import mock

from stream_translator_gpt import main
from stream_translator_gpt.main import Config, _transcription_engine_label


class TranscriptionEngineLoggingTest(unittest.TestCase):

    def test_transcription_engine_labels_cover_every_backend(self):
        cases = [
            ({}, 'Whisper'),
            ({'use_faster_whisper': True}, 'Faster Whisper'),
            ({'use_simul_streaming': True}, 'SimulStreaming'),
            ({'use_simul_streaming': True, 'use_faster_whisper': True}, 'SimulStreaming'),
            ({'use_openai_transcription_api': True}, 'OpenAI'),
            ({'use_hf_asr': True}, 'Hugging Face ASR'),
            ({'use_modulate_transcription_api': True}, 'Modulate'),
        ]

        for options, expected in cases:
            with self.subTest(options=options):
                self.assertEqual(_transcription_engine_label(Config(url='input', **options)), expected)

    def test_transcription_engine_is_logged_before_translation_engine(self):
        config = Config(url='input',
                        use_modulate_transcription_api=True,
                        modulate_api_key='modulate-key',
                        translation_prompt='translate')
        fake_getter = SimpleNamespace(loop=mock.Mock())
        fake_transcriber = SimpleNamespace(loop=mock.Mock())
        fake_translator = SimpleNamespace(loop=mock.Mock())
        fake_exporter = SimpleNamespace(loop=mock.Mock())
        worker_thread = mock.Mock()
        worker_thread.is_alive.return_value = False
        messages = []

        def record_warning(message, *args):
            messages.append(message % args if args else message)

        def make_translator(**kwargs):
            main.logger.warning('Using %s API as translation engine.', kwargs['model'])
            return fake_translator

        with mock.patch.object(main, 'ClientPool'), \
                mock.patch.object(main, '_create_audio_getter', return_value=fake_getter), \
                mock.patch.object(main, 'ModulateStreamingTranscriber', return_value=fake_transcriber), \
                mock.patch.object(main, 'GPTTranslator', side_effect=make_translator), \
                mock.patch.object(main, 'ResultExporter', return_value=fake_exporter), \
                mock.patch.object(main, 'start_daemon_thread', return_value=worker_thread), \
                mock.patch.object(main.logger, 'warning', side_effect=record_warning):
            main.run(config)

        transcription_message = 'Using Modulate as transcription engine.'
        translation_message = f'Using {config.gpt_model} API as translation engine.'
        self.assertEqual(messages.count(transcription_message), 1)
        self.assertEqual(messages.count(translation_message), 1)
        self.assertLess(messages.index(transcription_message), messages.index(translation_message))

    def test_transcription_engine_is_logged_without_translation(self):
        for translation_prompt in (None, ''):
            with self.subTest(translation_prompt=translation_prompt):
                config = Config(url='input',
                                use_modulate_transcription_api=True,
                                modulate_api_key='modulate-key',
                                translation_prompt=translation_prompt)
                fake_getter = SimpleNamespace(loop=mock.Mock())
                fake_transcriber = SimpleNamespace(loop=mock.Mock())
                fake_exporter = SimpleNamespace(loop=mock.Mock())
                worker_thread = mock.Mock()
                worker_thread.is_alive.return_value = False
                messages = []

                def record_warning(message, *args):
                    messages.append(message % args if args else message)

                with mock.patch.object(main, 'ClientPool'), \
                        mock.patch.object(main, '_create_audio_getter', return_value=fake_getter), \
                        mock.patch.object(main, 'ModulateStreamingTranscriber', return_value=fake_transcriber), \
                        mock.patch.object(main, 'GPTTranslator') as gpt_translator, \
                        mock.patch.object(main, 'GeminiTranslator') as gemini_translator, \
                        mock.patch.object(main, 'ResultExporter', return_value=fake_exporter), \
                        mock.patch.object(main, 'start_daemon_thread', return_value=worker_thread), \
                        mock.patch.object(main.logger, 'warning', side_effect=record_warning):
                    main.run(config)

                self.assertEqual(messages.count('Using Modulate as transcription engine.'), 1)
                self.assertFalse(any(message.endswith('as translation engine.') for message in messages))
                gpt_translator.assert_not_called()
                gemini_translator.assert_not_called()


if __name__ == '__main__':
    unittest.main()
