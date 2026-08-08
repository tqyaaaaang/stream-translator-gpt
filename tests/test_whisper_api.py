import sys
import types
import unittest
from unittest import mock

import numpy as np
import tyro

from stream_translator_gpt.audio_transcriber import RemoteOpenaiTranscriber
from stream_translator_gpt.common import ClientPool
from stream_translator_gpt.main import Config, _validate_and_normalize


class WhisperApiConfigTest(unittest.TestCase):

    def setUp(self):
        self.client_pool_state = {
            name: getattr(ClientPool, name)
            for name in ('_openai_clients', '_openai_index', '_whisper_clients', '_whisper_index')
        }

    def tearDown(self):
        for name, value in self.client_pool_state.items():
            setattr(ClientPool, name, value)

    def test_cli_accepts_independent_whisper_credentials(self):
        config = tyro.cli(Config,
                          args=[
                              'input', '--use-openai-transcription-api', '--whisper-api-key', 'asr-key',
                              '--whisper-base-url', 'https://asr.example/v1', '--openai-transcription-model',
                              'custom-whisper'
                          ])

        self.assertEqual(config.whisper_api_key, 'asr-key')
        self.assertEqual(config.whisper_base_url, 'https://asr.example/v1')
        self.assertEqual(config.openai_transcription_model, 'custom-whisper')

    def test_remote_transcription_validation_accepts_whisper_key_without_openai_key(self):
        config = Config(url='input', use_openai_transcription_api=True, whisper_api_key='asr-key')

        _validate_and_normalize(config)

        self.assertIsNone(config.openai_api_key)

    def test_client_pool_keeps_translation_and_transcription_credentials_separate(self):

        class FakeOpenAI:

            def __init__(self, **kwargs):
                self.api_key = kwargs['api_key']
                self.base_url = kwargs['base_url']

        fake_openai = types.ModuleType('openai')
        fake_openai.OpenAI = FakeOpenAI
        fake_httpx = types.ModuleType('httpx')
        fake_httpx.Client = lambda **kwargs: kwargs

        with mock.patch.dict(sys.modules, {'openai': fake_openai, 'httpx': fake_httpx}):
            ClientPool.init(openai_api_key='gpt-key',
                            google_api_key=None,
                            openai_base_url='https://gpt.example/v1',
                            whisper_api_key='asr-key',
                            whisper_base_url='https://asr.example/v1')

        gpt_client = ClientPool.get_openai_client()
        whisper_client = ClientPool.get_whisper_client()
        self.assertEqual((gpt_client.api_key, gpt_client.base_url), ('gpt-key', 'https://gpt.example/v1'))
        self.assertEqual((whisper_client.api_key, whisper_client.base_url), ('asr-key', 'https://asr.example/v1'))

    def test_remote_transcriber_uses_whisper_client_pool(self):
        captured = {}

        class FakeTranscriptions:

            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(text='transcribed')

        fake_client = types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=FakeTranscriptions()))
        transcriber = RemoteOpenaiTranscriber.__new__(RemoteOpenaiTranscriber)
        transcriber.model = 'whisper-1'
        transcriber.language = 'en'
        transcriber.transcription_keywords = ''

        with mock.patch.object(ClientPool, 'get_whisper_client', return_value=fake_client) as get_client:
            text, tokens = transcriber.transcribe(np.zeros(512, dtype=np.float32))

        get_client.assert_called_once_with()
        self.assertEqual((text, tokens), ('transcribed', None))
        self.assertEqual(captured['model'], 'whisper-1')
        self.assertEqual(captured['language'], 'en')


if __name__ == '__main__':
    unittest.main()
