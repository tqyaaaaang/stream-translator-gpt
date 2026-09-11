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
        # Construct the real selected implementation, but never start its workers
        # or create SDK clients. This catches logging changes in either factory.
        for legacy in (False, True):
            for google_key in (None, 'google-test-key'):
                with self.subTest(legacy=legacy, google=bool(google_key)):
                    config = Config(url='input', use_modulate_transcription_api=True,
                                    modulate_api_key='modulate-test-key', openai_api_key='openai-test-key',
                                    google_api_key=google_key, use_old_translator=legacy,
                                    translation_prompt='Translate.')
                    worker_thread = mock.Mock()
                    worker_thread.is_alive.return_value = False
                    messages = []

                    def record_warning(message, *args):
                        messages.append(message % args if args else message)

                    with mock.patch.object(main, 'ClientPool'), \
                            mock.patch.object(main, '_create_audio_getter',
                                              return_value=SimpleNamespace(loop=mock.Mock())), \
                            mock.patch.object(main, 'ModulateStreamingTranscriber'), \
                            mock.patch.object(main, 'ResultExporter'), \
                            mock.patch.object(main, 'start_daemon_thread', return_value=worker_thread), \
                            mock.patch.object(main.logger, 'warning', side_effect=record_warning):
                        main.run(config)

                    transcription_message = 'Using Modulate as transcription engine.'
                    model = config.gemini_model if google_key else config.gpt_model
                    translation_message = f'Using {model} API as translation engine.'
                    self.assertEqual(messages.count(transcription_message), 1)
                    self.assertEqual(messages.count(translation_message), 1)
                    self.assertLess(messages.index(transcription_message), messages.index(translation_message))
                    self.assertEqual(sum('--use-old-translator is deprecated' in text for text in messages),
                                     int(legacy))

    def test_transcription_engine_is_logged_without_translation(self):
        for translation_prompt in (None, ''):
            with self.subTest(translation_prompt=translation_prompt):
                config = Config(url='input', use_modulate_transcription_api=True,
                                modulate_api_key='modulate-key', translation_prompt=translation_prompt)
                worker_thread = mock.Mock()
                worker_thread.is_alive.return_value = False
                messages = []

                def record_warning(message, *args):
                    messages.append(message % args if args else message)

                with mock.patch.object(main, 'ClientPool'), \
                        mock.patch.object(main, '_create_audio_getter',
                                          return_value=SimpleNamespace(loop=mock.Mock())), \
                        mock.patch.object(main, 'ModulateStreamingTranscriber'), \
                        mock.patch.object(main, '_create_translator', wraps=main._create_translator) as factory, \
                        mock.patch.object(main, 'ResultExporter'), \
                        mock.patch.object(main, 'start_daemon_thread', return_value=worker_thread) as start, \
                        mock.patch.object(main.logger, 'warning', side_effect=record_warning):
                    main.run(config)

                self.assertEqual(messages.count('Using Modulate as transcription engine.'), 1)
                self.assertFalse(any(message.endswith('as translation engine.') for message in messages))
                factory.assert_called_once_with(config)
                self.assertFalse(any(call.kwargs['name'] == 'translator' for call in start.call_args_list))


if __name__ == '__main__':
    unittest.main()
