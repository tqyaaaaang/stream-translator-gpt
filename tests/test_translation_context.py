import json
import queue
import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import tyro

from stream_translator_gpt.common import ClientPool, TranslationTask
from stream_translator_gpt.llm_translator import (
    GeminiTranslator,
    GPTTranslator,
    LLMTranslator,
    TRANSLATION_CONTEXT_MAX_CHARS,
    TRANSLATION_CONTEXT_MIN_CHARS,
    _parse_translation_context_completion,
    _translation_context_target_min_chars,
)
from stream_translator_gpt.main import Config, _validate_and_normalize
from stream_translator_gpt.result_exporter import ResultExporter, _format_translation_result


class _ContextTranslator(LLMTranslator):

    def translate(self, translation_task):
        pass


class _RecordingContextTranslator(_ContextTranslator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_contexts = []

    def translate(self, translation_task):
        self.input_contexts.append(translation_task.input_translation_context)
        self._apply_completion(
            translation_task,
            json.dumps({
                'translation': f'translation-{translation_task.id}',
                'context': f'context-{translation_task.id}',
            }),
        )


class TranslationContextTest(unittest.TestCase):

    @staticmethod
    def _translator(**overrides):
        kwargs = {
            'model': 'test-model',
            'prompt': 'Translate faithfully.',
            'history_size': 2,
            'use_json_result': False,
            'timeout': 0,
            'retry_if_translation_fails': False,
            'use_translation_context': True,
            'debug_mode': False,
        }
        kwargs.update(overrides)
        return _ContextTranslator(**kwargs)

    @staticmethod
    def _task(line_id=1, transcript='current'):
        task = TranslationTask(line_id, np.empty(0, dtype=np.float32), (0.0, 1.0))
        task.transcript = transcript
        return task

    def test_cli_flag_is_opt_in_and_requires_translation(self):
        default = tyro.cli(Config, args=['input'])
        enabled = tyro.cli(Config, args=['input', '--use-translation-context'])

        self.assertFalse(default.use_translation_context)
        self.assertTrue(enabled.use_translation_context)

        with mock.patch('stream_translator_gpt.main.logger.error'), self.assertRaises(SystemExit) as raised:
            _validate_and_normalize(enabled)
        self.assertEqual(raised.exception.code, 1)

    def test_messages_define_and_include_context_reference_and_translate(self):
        translator = self._translator()
        task = self._task(transcript='今の文')
        task.input_translation_context = '配信者はゲームについて話している。'
        task.context_transcripts = ['前の文1', '前の文2']

        system_prompt, user_content = translator._build_messages(task)

        self.assertIn('Application output protocol (mandatory)', system_prompt)
        self.assertIn('"Context" is long-horizon memory', system_prompt)
        self.assertIn('"Reference" contains the recent source-language transcripts', system_prompt)
        self.assertIn('"Translate" is the current source-language utterance', system_prompt)
        self.assertIn('{"translation":', system_prompt)
        self.assertIn('Preserve high-level topics', system_prompt)
        self.assertIn('ongoing activities or goals', system_prompt)
        self.assertIn('key entities and relationships', system_prompt)
        self.assertIn('unresolved referents or decisions', system_prompt)
        self.assertIn('Do not restate transient local details', system_prompt)
        self.assertIn('when the topic has not genuinely changed', system_prompt)
        self.assertIn(
            f'Aim for {TRANSLATION_CONTEXT_MIN_CHARS}-{TRANSLATION_CONTEXT_MAX_CHARS} Unicode characters',
            system_prompt,
        )
        self.assertIn(f'hard maximum is {TRANSLATION_CONTEXT_MAX_CHARS}', system_prompt)
        self.assertIn('Never invent unsupported facts', system_prompt)
        self.assertEqual(
            user_content,
            '**Context:**\n配信者はゲームについて話している。\n\n'
            '**Reference:**\n> 前の文1\n> 前の文2\n\n'
            '**Translate:**\n今の文',
        )

    def test_context_target_minimum_is_half_of_maximum_rounded_up(self):
        self.assertEqual(TRANSLATION_CONTEXT_MIN_CHARS,
                         (TRANSLATION_CONTEXT_MAX_CHARS + 1) // 2)
        self.assertEqual(_translation_context_target_min_chars(50), 25)
        self.assertEqual(_translation_context_target_min_chars(100), 50)
        self.assertEqual(_translation_context_target_min_chars(101), 51)
        with self.assertRaises(ValueError):
            _translation_context_target_min_chars(0)

    def test_same_runtime_prompt_uses_hardcoded_output_protocol_for_every_mode(self):
        shared_prompt = 'Translate faithfully using the supplied terminology.'
        task = self._task()
        translators = {
            'plain': self._translator(prompt=shared_prompt, history_size=0,
                                      use_translation_context=False, use_json_result=False),
            'json': self._translator(prompt=shared_prompt, history_size=0,
                                     use_translation_context=False, use_json_result=True),
            'context': self._translator(prompt=shared_prompt, history_size=0,
                                        use_translation_context=True, use_json_result=False),
            'context_with_json_flag': self._translator(prompt=shared_prompt, history_size=0,
                                                       use_translation_context=True, use_json_result=True),
        }

        prompts = {name: translator._build_messages(task)[0] for name, translator in translators.items()}
        for system_prompt in prompts.values():
            self.assertEqual(system_prompt.count(shared_prompt), 1)
            self.assertGreater(system_prompt.index('Application output protocol'),
                               system_prompt.index(shared_prompt))

        self.assertIn('Return only the translation text as one physical line', prompts['plain'])
        self.assertNotIn('Return exactly one JSON object', prompts['plain'])
        self.assertIn('{"translation":"the target-language translation of the current utterance"}',
                      prompts['json'])
        self.assertNotIn('"context":"the updated rolling Context"', prompts['json'])
        self.assertIn('"context":"the updated rolling Context"', prompts['context'])
        self.assertEqual(prompts['context'], prompts['context_with_json_flag'])

    def test_messages_keep_all_three_fields_when_context_and_history_are_empty(self):
        translator = self._translator(history_size=0)
        task = self._task()

        translator._prepare_context(task)
        _, user_content = translator._build_messages(task)

        self.assertEqual(
            user_content,
            '**Context:**\n(empty)\n\n**Reference:**\n(empty)\n\n**Translate:**\ncurrent',
        )

    def test_strict_result_parser_normalizes_and_limits_context(self):
        context = ('猫\n  の話 ' + 'あ' * (TRANSLATION_CONTEXT_MAX_CHARS + 10))
        parsed = _parse_translation_context_completion(json.dumps({
            'translation': '  译文  ',
            'context': context,
        }, ensure_ascii=False))

        self.assertIsNotNone(parsed)
        translation, normalized_context = parsed
        self.assertEqual(translation, '译文')
        self.assertNotIn('\n', normalized_context)
        self.assertEqual(len(normalized_context), TRANSLATION_CONTEXT_MAX_CHARS)

    def test_strict_result_parser_rejects_missing_empty_wrong_and_extra_fields(self):
        invalid = [
            '',
            'not json',
            '{"translation":"译文"}',
            '{"translation":"","context":"摘要"}',
            '{"translation":"译文","context":[]}',
            '{"translation":"译文","context":"摘要","extra":true}',
        ]

        for response in invalid:
            with self.subTest(response=response):
                self.assertIsNone(_parse_translation_context_completion(response))

    def test_completion_publishes_context_before_translation_and_rejects_partial_output(self):
        translator = self._translator()
        task = self._task()

        self.assertFalse(translator._apply_completion(task, '{"translation":"译文"}'))
        self.assertIsNone(task.translation)
        self.assertIsNone(task.translation_context)

        self.assertTrue(translator._apply_completion(
            task,
            '{"translation":"译文","context":"正在讨论游戏。"}',
        ))
        self.assertEqual(task.translation, '译文')
        self.assertEqual(task.translation_context, '正在讨论游戏。')

    def test_context_translation_preserves_all_lines_as_normalized_text(self):
        translator = self._translator()
        task = self._task()

        self.assertTrue(translator._apply_completion(
            task,
            '{"translation":"第一行\\n  第二行","context":"摘要"}',
        ))
        self.assertEqual(task.translation, '第一行 第二行')

    def test_context_mode_is_serial_and_commits_only_successful_ordered_results(self):
        translator = self._translator()
        self.assertEqual(translator.parallel_max_number, 1)
        self.assertEqual(self._translator(use_translation_context=False).parallel_max_number,
                         LLMTranslator.PARALLEL_MAX_NUMBER)

        first = self._task(1, 'first transcript')
        translator._prepare_context(first)
        self.assertEqual(first.input_translation_context, '')
        translator._apply_completion(first, '{"translation":"一","context":"新摘要"}')
        translator.processing_queue.append(first)

        self.assertEqual(translator._get_results(), [first])
        self.assertEqual(translator.latest_translation_context, '新摘要')

        failed = self._task(2, 'failed transcript')
        translator._prepare_context(failed)
        failed.translation_failed = True
        translator.processing_queue.append(failed)
        self.assertEqual(translator._get_results(), [failed])
        self.assertEqual(translator.latest_translation_context, '新摘要')

        following = self._task(3, 'following transcript')
        translator._prepare_context(following)
        self.assertEqual(following.input_translation_context, '新摘要')
        self.assertEqual(following.context_transcripts, ['first transcript', 'failed transcript'])

    def test_loop_supplies_each_successful_context_to_the_next_request(self):
        translator = _RecordingContextTranslator(
            model='test-model',
            prompt='Translate faithfully.',
            history_size=2,
            use_json_result=False,
            timeout=0,
            retry_if_translation_fails=False,
            use_translation_context=True,
        )
        input_queue = queue.SimpleQueue()
        output_queue = queue.SimpleQueue()
        input_queue.put(self._task(1, 'first'))
        input_queue.put(self._task(2, 'second'))
        input_queue.put(None)

        translator.loop(input_queue, output_queue)

        self.assertEqual(translator.input_contexts, ['', 'context-1'])
        self.assertEqual(output_queue.get_nowait().translation, 'translation-1')
        self.assertEqual(output_queue.get_nowait().translation, 'translation-2')
        self.assertIsNone(output_queue.get_nowait())

    def test_retry_keeps_the_original_context_snapshot(self):
        translator = self._translator()
        task = self._task()
        translator.latest_translation_context = 'original summary'
        translator._prepare_context(task)
        translator.latest_translation_context = 'newer summary'

        self.assertEqual(task.input_translation_context, 'original summary')

    def test_late_response_is_discarded_without_publishing_context(self):
        translator = self._translator(timeout=1)
        task = self._task()
        task.start_time = datetime.now(timezone.utc) - timedelta(seconds=2)

        accepted = translator._apply_completion(
            task,
            '{"translation":"late","context":"must not be committed"}',
        )

        self.assertFalse(accepted)
        self.assertIsNone(task.translation)
        self.assertIsNone(task.translation_context)

    def test_timeout_finalization_atomically_wins_against_late_publication(self):
        translator = self._translator(timeout=1)
        task = self._task()
        task.start_time = datetime.now(timezone.utc) - timedelta(seconds=2)
        translator.processing_queue.append(task)
        parser_entered = threading.Event()
        release_parser = threading.Event()
        accepted = []

        def blocked_parse(_completion):
            parser_entered.set()
            release_parser.wait(timeout=2)
            return 'late translation', 'late context'

        def publish():
            accepted.append(translator._apply_completion(task, 'response'))

        with mock.patch('stream_translator_gpt.llm_translator._parse_translation_context_completion',
                        side_effect=blocked_parse):
            worker = threading.Thread(target=publish)
            worker.start()
            self.assertTrue(parser_entered.wait(timeout=1))
            try:
                self.assertEqual(translator._get_results(), [task])
                self.assertTrue(task._translation_finalized)
            finally:
                release_parser.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(accepted, [False])
        self.assertIsNone(task.translation)
        self.assertIsNone(task.translation_context)
        self.assertFalse(translator._mark_translation_failed(task))
        self.assertFalse(task.translation_failed)

    def test_openai_responses_requests_json_and_extracts_both_fields(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text='{"translation":"译文","context":"摘要"}',
                usage=None,
            )

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        translator = GPTTranslator(
            model='gpt-5.6-luna',
            prompt='Translate faithfully.',
            history_size=0,
            use_json_result=False,
            timeout=10,
            retry_if_translation_fails=False,
            use_translation_context=True,
            api_mode='responses',
        )
        task = self._task()

        with mock.patch.object(ClientPool, 'get_openai_client', return_value=client):
            translator.translate(task)

        response_format = calls[0]['text']['format']
        self.assertEqual(response_format['type'], 'json_schema')
        self.assertTrue(response_format['strict'])
        self.assertEqual(response_format['schema']['required'], ['translation', 'context'])
        self.assertFalse(response_format['schema']['additionalProperties'])
        self.assertEqual(task.translation, '译文')
        self.assertEqual(task.translation_context, '摘要')

    def test_explicit_cache_prefix_excludes_dynamic_context_and_reference(self):
        translator = GPTTranslator(
            model='gpt-5.6-luna',
            prompt='Stable translation instructions.',
            history_size=2,
            use_json_result=False,
            timeout=10,
            retry_if_translation_fails=False,
            use_translation_context=True,
            api_mode='responses',
            prompt_cache_enabled=True,
        )
        first = self._task(1, 'current one')
        first.input_translation_context = 'dynamic summary one'
        first.context_transcripts = ['dynamic reference one']
        second = self._task(2, 'current two')
        second.input_translation_context = 'dynamic summary two'
        second.context_transcripts = ['dynamic reference two']

        first_system, first_user = translator._build_messages(first)
        second_system, second_user = translator._build_messages(second)
        first_input = translator._build_responses_input(first_system, first_user)
        second_input = translator._build_responses_input(second_system, second_user)

        self.assertEqual(first_input[0], second_input[0])
        self.assertNotEqual(first_input[1], second_input[1])
        self.assertEqual(first_input[0]['content'][0]['prompt_cache_breakpoint'], {'mode': 'explicit'})
        stable_text = first_input[0]['content'][0]['text']
        self.assertNotIn('dynamic summary one', stable_text)
        self.assertNotIn('dynamic reference one', stable_text)
        self.assertIn('dynamic summary one', first_input[1]['content'][0]['text'])

    def test_openai_chat_compatibility_requests_json(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='{"translation":"译文","context":"摘要"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        translator = GPTTranslator(
            model='compatible-model',
            prompt='Translate faithfully.',
            history_size=0,
            use_json_result=False,
            timeout=10,
            retry_if_translation_fails=False,
            use_translation_context=True,
            api_mode='chat-completions',
        )

        with mock.patch.object(ClientPool, 'get_openai_client', return_value=client):
            translator.translate(self._task())

        self.assertEqual(calls[0]['response_format'], {'type': 'json_object'})

    def test_openai_sdk_serializes_strict_context_schema(self):
        import httpx
        from openai import OpenAI

        captured = {}

        def handle(request):
            captured['body'] = json.loads(request.content)
            return httpx.Response(200, json={
                'id': 'resp_context_test',
                'object': 'response',
                'created_at': 0,
                'model': 'gpt-5.6-luna',
                'status': 'completed',
                'output': [{
                    'id': 'msg_context_test',
                    'type': 'message',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{
                        'type': 'output_text',
                        'text': '{"translation":"译文","context":"摘要"}',
                        'annotations': [],
                    }],
                }],
                'usage': {
                    'input_tokens': 10,
                    'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                    'output_tokens': 5,
                    'output_tokens_details': {'reasoning_tokens': 0},
                    'total_tokens': 15,
                },
            })

        client = OpenAI(
            api_key='test-key',
            base_url='https://example.test/v1',
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        translator = GPTTranslator(
            model='gpt-5.6-luna',
            prompt='Translate faithfully.',
            history_size=0,
            use_json_result=False,
            timeout=10,
            retry_if_translation_fails=False,
            use_translation_context=True,
            api_mode='responses',
        )
        try:
            with mock.patch.object(ClientPool, 'get_openai_client', return_value=client):
                task = self._task()
                translator.translate(task)
        finally:
            client.close()

        response_format = captured['body']['text']['format']
        self.assertEqual(response_format['type'], 'json_schema')
        self.assertTrue(response_format['strict'])
        self.assertEqual(response_format['schema']['required'], ['translation', 'context'])
        self.assertFalse(response_format['schema']['additionalProperties'])
        self.assertEqual(task.translation_context, '摘要')

    def test_gemini_requests_json_and_extracts_both_fields(self):
        calls = []

        class Models:

            @staticmethod
            def generate_content(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(text='{"translation":"译文","context":"摘要"}')

        fake_types = SimpleNamespace(
            SafetySetting=lambda **kwargs: SimpleNamespace(**kwargs),
            GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        fake_genai = ModuleType('google.genai')
        fake_genai.types = fake_types
        translator = GeminiTranslator(
            model='compatible-model',
            prompt='Translate faithfully.',
            history_size=0,
            use_json_result=False,
            timeout=10,
            retry_if_translation_fails=False,
            use_translation_context=True,
        )

        with mock.patch.dict(sys.modules, {'google.genai': fake_genai}), mock.patch.object(
                ClientPool, 'get_google_client', return_value=SimpleNamespace(models=Models())):
            task = self._task()
            translator.translate(task)

        self.assertEqual(calls[0]['config'].response_mime_type, 'application/json')
        self.assertEqual(task.translation, '译文')
        self.assertEqual(task.translation_context, '摘要')

    def test_export_format_keeps_translation_context_internal(self):
        task = self._task()
        task.translation = '译文'
        task.translation_context = '摘要'

        self.assertEqual(_format_translation_result(task), '译文')

        task.translation_context = None
        self.assertEqual(_format_translation_result(task), '译文')

    def test_all_exporter_payloads_omit_internal_translation_context(self):
        for include_transcript in (False, True):
            for include_timestamp in (False, True):
                with self.subTest(include_transcript=include_transcript, include_timestamp=include_timestamp):
                    exporter = ResultExporter('', '', '', '', 0, '', '',
                                              include_transcript, include_timestamp)
                    exporter.cqhttp_queue = queue.SimpleQueue()
                    exporter.discord_queue = queue.SimpleQueue()
                    exporter.telegram_queue = queue.SimpleQueue()
                    exporter.file_queue = queue.SimpleQueue()

                    task = self._task()
                    task.transcript = '原文'
                    task.translation = '译文'
                    task.translation_context = 'INTERNAL_CONTEXT_SENTINEL'
                    input_queue = queue.SimpleQueue()
                    input_queue.put(task)
                    input_queue.put(None)

                    with mock.patch('stream_translator_gpt.result_exporter.write_console_utf8') as console_write:
                        exporter.loop(input_queue)

                    console_payload = console_write.call_args.args[0]
                    self.assertIn('译文', console_payload)
                    self.assertNotIn('INTERNAL_CONTEXT_SENTINEL', console_payload)
                    self.assertNotIn('[Context]', console_payload)

                    for output_queue in (exporter.cqhttp_queue, exporter.discord_queue,
                                         exporter.telegram_queue, exporter.file_queue):
                        payload = output_queue.get_nowait()
                        self.assertIn('译文', payload)
                        self.assertNotIn('INTERNAL_CONTEXT_SENTINEL', payload)
                        self.assertNotIn('[Context]', payload)
                        self.assertIsNone(output_queue.get_nowait())


if __name__ == '__main__':
    unittest.main()
