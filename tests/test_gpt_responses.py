import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import tyro

from stream_translator_gpt.common import ClientPool, TranslationTask
from stream_translator_gpt.legacy_translator import GPTTranslator, _parse_json_completion
from stream_translator_gpt.logging_utils import configure_logging, shutdown_logging
from stream_translator_gpt.main import Config, _validate_and_normalize


class _FakeResponses:

    def __init__(self, outputs=None, usage=None, input_tokens=2048):
        self.calls = []
        self.outputs = list(outputs or ['translated'])
        self.usage = usage
        self.input_tokens = SimpleNamespace(count=mock.Mock(return_value=SimpleNamespace(input_tokens=input_tokens)))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.pop(0), usage=self.usage)


class _FakeChatCompletions:

    def __init__(self, output='translated'):
        self.calls = []
        self.output = output

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.output)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class GPTResponsesTranslatorTest(unittest.TestCase):

    def test_legacy_json_parser_preserves_compatible_model_fallbacks(self):
        self.assertEqual(_parse_json_completion('plain translation'), 'plain translation')
        self.assertEqual(
            _parse_json_completion('{"translation":"译文","metadata":"ignored"}'),
            '译文',
        )
        self.assertEqual(
            _parse_json_completion('{"translation":"他说"你好""}'),
            '他说"你好"',
        )

    @staticmethod
    def _translator(**overrides):
        kwargs = {
            'model': 'gpt-5.6-luna',
            'prompt': 'Translate Japanese into Chinese.',
            'history_size': 0,
            'use_json_result': False,
            'timeout': 10,
            'retry_if_translation_fails': False,
            'debug_mode': False,
        }
        kwargs.update(overrides)
        return GPTTranslator(**kwargs)

    @staticmethod
    def _task(line_id=1, transcript='こんにちは', context=None):
        task = TranslationTask(line_id, np.empty(0, dtype=np.float32), (0.0, 1.0))
        task.transcript = transcript
        task.context_transcripts = context
        return task

    def test_responses_api_is_default_and_reads_output_text(self):
        responses = _FakeResponses(outputs=['你好'])
        client = SimpleNamespace(responses=responses)
        translator = self._translator()
        task = self._task()

        with mock.patch.object(ClientPool, 'get_openai_client', return_value=client):
            translator.translate(task)

        self.assertEqual(task.translation, '你好')
        self.assertFalse(task.translation_failed)
        request = responses.calls[0]
        self.assertEqual(request['model'], 'gpt-5.6-luna')
        self.assertFalse(request['store'])
        self.assertNotIn('messages', request)
        self.assertNotIn('response_format', request)
        self.assertEqual(request['reasoning'], {'effort': 'none'})
        self.assertEqual(request['input'][0]['role'], 'system')
        self.assertIn('Translate Japanese into Chinese.', request['input'][0]['content'][0]['text'])
        self.assertEqual(request['input'][1]['content'][0]['text'], 'こんにちは')
        self.assertNotIn('prompt_cache_key', request)
        self.assertNotIn('prompt_cache_options', request)
        self.assertNotIn('prompt_cache_breakpoint', request['input'][0]['content'][0])
        json.dumps(request)

    def test_full_prompt_content_is_only_logged_at_ddebug(self):
        translator = self._translator(debug_mode=True)
        debug_output = io.StringIO()
        ddebug_output = io.StringIO()

        try:
            configure_logging(2, stream=debug_output)
            translator._log_prompt_messages('SYSTEM_PROMPT_SENTINEL', 'USER_PROMPT_SENTINEL')
            shutdown_logging()

            configure_logging(3, stream=ddebug_output)
            translator._log_prompt_messages('SYSTEM_PROMPT_SENTINEL', 'USER_PROMPT_SENTINEL')
            shutdown_logging()
        finally:
            shutdown_logging()

        self.assertNotIn('PROMPT_SENTINEL', debug_output.getvalue())
        self.assertIn(':DDEBUG:main: [System] SYSTEM_PROMPT_SENTINEL', ddebug_output.getvalue())
        self.assertIn(':DDEBUG:main: [User] USER_PROMPT_SENTINEL', ddebug_output.getvalue())

    def test_explicit_cache_marks_only_the_complete_stable_prefix(self):
        responses = _FakeResponses(outputs=['一', '二'])
        client = SimpleNamespace(responses=responses)
        translator = self._translator(history_size=2, prompt_cache_enabled=True)
        first = self._task(1, '最初の文', ['以前の文'])
        second = self._task(2, '次の文', ['別の以前の文'])

        with mock.patch.object(ClientPool, 'get_openai_client', return_value=client):
            translator.translate(first)
            translator.translate(second)

        first_request, second_request = responses.calls
        self.assertEqual(first_request['input'][0], second_request['input'][0])
        self.assertNotEqual(first_request['input'][1], second_request['input'][1])
        self.assertEqual(first_request['prompt_cache_key'], second_request['prompt_cache_key'])
        self.assertTrue(first_request['prompt_cache_key'].startswith('stream-translator-gpt:'))
        self.assertEqual(first_request['prompt_cache_options'], {'mode': 'explicit', 'ttl': '30m'})

        stable_block = first_request['input'][0]['content'][0]
        self.assertEqual(stable_block['prompt_cache_breakpoint'], {'mode': 'explicit'})
        self.assertNotIn('以前の文', stable_block['text'])
        self.assertNotIn('最初の文', stable_block['text'])
        self.assertNotIn('prompt_cache_breakpoint', first_request['input'][1]['content'][0])

    def test_explicit_cache_accepts_a_caller_supplied_key(self):
        responses = _FakeResponses()
        translator = self._translator(prompt_cache_enabled=True, prompt_cache_key='channel:translation:v1')

        with mock.patch.object(ClientPool,
                               'get_openai_client',
                               return_value=SimpleNamespace(responses=responses)):
            translator.translate(self._task())

        self.assertEqual(responses.calls[0]['prompt_cache_key'], 'channel:translation:v1')

    def test_manual_cache_key_keeps_implicit_routing_without_explicit_cache(self):
        responses = _FakeResponses()
        translator = self._translator(prompt_cache_enabled=False, prompt_cache_key='unused-key')

        with mock.patch.object(ClientPool,
                               'get_openai_client',
                               return_value=SimpleNamespace(responses=responses)):
            translator.translate(self._task())

        request = responses.calls[0]
        self.assertEqual(request['prompt_cache_key'], 'unused-key')
        self.assertNotIn('prompt_cache_options', request)
        self.assertNotIn('prompt_cache_breakpoint', request['input'][0]['content'][0])

    def test_json_reasoning_verbosity_and_sampling_use_responses_shapes(self):
        responses = _FakeResponses(outputs=['{"translation":"译文"}'])
        translator = self._translator(use_json_result=True,
                                      reasoning_effort='high',
                                      verbosity='low',
                                      temperature=0.2,
                                      top_p=0.9,
                                      service_tier='priority')

        with mock.patch.object(ClientPool,
                               'get_openai_client',
                               return_value=SimpleNamespace(responses=responses)):
            task = self._task()
            translator.translate(task)

        request = responses.calls[0]
        self.assertEqual(task.translation, '译文')
        self.assertEqual(request['reasoning'], {'effort': 'high'})
        self.assertEqual(request['text'], {'format': {'type': 'json_object'}, 'verbosity': 'low'})
        self.assertEqual(request['temperature'], 0.2)
        self.assertEqual(request['top_p'], 0.9)
        self.assertEqual(request['service_tier'], 'priority')
        self.assertNotIn('reasoning_effort', request)
        self.assertNotIn('verbosity', request)

    def test_empty_response_marks_the_task_failed(self):
        responses = _FakeResponses(outputs=[''])
        translator = self._translator()
        task = self._task()

        with mock.patch.object(ClientPool,
                               'get_openai_client',
                               return_value=SimpleNamespace(responses=responses)):
            translator.translate(task)

        self.assertTrue(task.translation_failed)
        self.assertEqual(task.translation, '')

    def test_cache_usage_reads_responses_input_token_details(self):
        usage = SimpleNamespace(input_tokens_details=SimpleNamespace(cached_tokens=1024, cache_write_tokens=1280))
        responses = _FakeResponses(usage=usage)
        translator = self._translator(prompt_cache_enabled=True)

        with mock.patch.object(ClientPool,
                               'get_openai_client',
                               return_value=SimpleNamespace(responses=responses)), mock.patch(
                                   'stream_translator_gpt.legacy_translator.logger.debug') as debug:
            translator.translate(self._task())

        self.assertTrue(any(call.args[:3] ==
                            ('OpenAI prompt cache usage: read=%s token(s), written=%s token(s)', 1024, 1280)
                            for call in debug.call_args_list))

    def test_short_explicit_cache_prefix_warns_only_once(self):
        client = SimpleNamespace(responses=_FakeResponses(input_tokens=1023))
        translator = self._translator(prompt_cache_enabled=True)

        with mock.patch('stream_translator_gpt.legacy_translator.logger.warning') as warning:
            translator._warn_if_prompt_cache_prefix_is_too_short(client, 'x' * 1200)
            translator._warn_if_prompt_cache_prefix_is_too_short(client, 'x' * 1200)

        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], 1023)

    def test_short_prefix_uses_the_rendered_input_token_count(self):
        responses = _FakeResponses(input_tokens=1023)
        translator = self._translator(prompt_cache_enabled=True)

        with mock.patch('stream_translator_gpt.legacy_translator.logger.warning') as warning:
            translator._warn_if_prompt_cache_prefix_is_too_short(SimpleNamespace(responses=responses), 'short')

        warning.assert_called_once()
        responses.input_tokens.count.assert_called_once()

    def test_cache_prefix_at_the_minimum_does_not_warn(self):
        client = SimpleNamespace(responses=_FakeResponses(input_tokens=1024))
        translator = self._translator(prompt_cache_enabled=True)

        with mock.patch('stream_translator_gpt.legacy_translator.logger.warning') as warning:
            translator._warn_if_prompt_cache_prefix_is_too_short(client, 'short')

        warning.assert_not_called()
        client.responses.input_tokens.count.assert_called_once()

    def test_chat_completions_remains_an_explicit_compatibility_mode(self):
        completions = _FakeChatCompletions(output='兼容译文')
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        translator = self._translator(api_mode='chat-completions')
        task = self._task()

        with mock.patch.object(ClientPool, 'get_openai_client', return_value=client):
            translator.translate(task)

        self.assertEqual(task.translation, '兼容译文')
        request = completions.calls[0]
        self.assertIn('messages', request)
        self.assertNotIn('input', request)
        self.assertEqual(request['reasoning_effort'], 'none')

    def test_manual_cache_key_is_preserved_in_chat_compatibility_mode(self):
        completions = _FakeChatCompletions()
        translator = self._translator(api_mode='chat-completions', prompt_cache_key='legacy-routing-key')

        with mock.patch.object(
                ClientPool,
                'get_openai_client',
                return_value=SimpleNamespace(chat=SimpleNamespace(completions=completions))):
            translator.translate(self._task())

        self.assertEqual(completions.calls[0]['prompt_cache_key'], 'legacy-routing-key')

    def test_cli_requires_explicit_cache_opt_in(self):
        default = tyro.cli(Config, args=['input'])
        enabled = tyro.cli(Config, args=['input', '--openai-prompt-cache'])
        compatibility = tyro.cli(Config, args=['input', '--openai-translation-api', 'chat-completions'])

        self.assertFalse(default.openai_prompt_cache)
        self.assertTrue(enabled.openai_prompt_cache)
        self.assertEqual(default.openai_translation_api, 'auto')
        self.assertEqual(compatibility.openai_translation_api, 'chat-completions')

    def test_auto_api_mode_uses_responses_for_openai_and_chat_for_custom_endpoints(self):
        openai_config = Config(url='input')
        custom_config = Config(url='input', openai_base_url='https://compatible.example/v1')

        _validate_and_normalize(openai_config)
        _validate_and_normalize(custom_config)

        self.assertEqual(openai_config.openai_translation_api, 'responses')
        self.assertEqual(custom_config.openai_translation_api, 'chat-completions')

    def test_explicit_cache_is_rejected_in_chat_compatibility_mode(self):
        config = Config(url='input',
                        openai_api_key='key',
                        translation_prompt='translate',
                        openai_prompt_cache=True,
                        openai_translation_api='chat-completions')

        with mock.patch('stream_translator_gpt.main.logger.error'), self.assertRaises(SystemExit) as raised:
            _validate_and_normalize(config)

        self.assertEqual(raised.exception.code, 1)

    def test_explicit_cache_rejects_known_models_older_than_gpt_5_6(self):
        config = Config(url='input',
                        openai_api_key='key',
                        translation_prompt='translate',
                        gpt_model='gpt-5.4-mini',
                        openai_prompt_cache=True,
                        openai_translation_api='responses')

        with mock.patch('stream_translator_gpt.main.logger.error'), self.assertRaises(SystemExit) as raised:
            _validate_and_normalize(config)

        self.assertEqual(raised.exception.code, 1)

    def test_openai_sdk_serializes_the_responses_wire_request(self):
        import httpx
        from openai import OpenAI

        captured = {}

        def handle(request):
            if request.url.path == '/v1/responses/input_tokens':
                return httpx.Response(200, json={'object': 'response.input_tokens', 'input_tokens': 2048})
            captured['path'] = request.url.path
            captured['body'] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    'id': 'resp_test',
                    'object': 'response',
                    'created_at': 0,
                    'model': 'gpt-5.6-luna',
                    'status': 'completed',
                    'output': [{
                        'id': 'msg_test',
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{
                            'type': 'output_text',
                            'text': 'wire-ok',
                            'annotations': [],
                        }],
                    }],
                    'usage': {
                        'input_tokens': 12,
                        'input_tokens_details': {
                            'cached_tokens': 0,
                            'cache_write_tokens': 0,
                        },
                        'output_tokens': 2,
                        'output_tokens_details': {
                            'reasoning_tokens': 0,
                        },
                        'total_tokens': 14,
                    },
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = OpenAI(api_key='test-key', base_url='https://example.test/v1', http_client=http_client)
        translator = self._translator(prompt_cache_enabled=True)
        task = self._task()
        try:
            with mock.patch.object(ClientPool, 'get_openai_client', return_value=client):
                translator.translate(task)
        finally:
            client.close()

        self.assertEqual(task.translation, 'wire-ok')
        self.assertEqual(captured['path'], '/v1/responses')
        self.assertEqual(captured['body']['prompt_cache_options'], {'mode': 'explicit', 'ttl': '30m'})
        self.assertEqual(captured['body']['input'][0]['content'][0]['prompt_cache_breakpoint'],
                         {'mode': 'explicit'})


if __name__ == '__main__':
    unittest.main()
