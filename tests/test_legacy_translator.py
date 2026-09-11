"""The emergency fallback is an archived implementation, not a new scheduler adapter."""

import asyncio
import hashlib
import json
import queue
import socket
import subprocess
import sys
import threading
import unittest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import httpx

from stream_translator_gpt import main
from stream_translator_gpt import legacy_translator
from stream_translator_gpt.common import ClientPool, TranslationTask
from stream_translator_gpt.main import Config


class LegacyTranslatorTest(unittest.TestCase):

    def test_source_is_byte_identical_to_pre_parallel_history(self):
        # git show ab314cf:stream_translator_gpt/llm_translator.py | shasum -a 256
        expected = '8a07828cab42ff171c78d94bb3aeaec518bab3a9c9f9587ebbc6413ace9df223'
        # Git can check text files out with CRLF on Windows; compare canonical
        # source bytes rather than treating that checkout policy as a code edit.
        source = Path(legacy_translator.__file__).read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(source).hexdigest(), expected)

    def test_factory_calls_historical_constructors_without_scheduler_options(self):
        for google_key in (None, 'google-test-key'):
            with self.subTest(google=bool(google_key)):
                config = Config(url='input', openai_api_key='openai-test-key', google_api_key=google_key,
                                translation_prompt='Translate faithfully.', translation_threads=99,
                                use_old_translator=True, translation_history_size=5,
                                use_translation_context=True, translation_timeout=15,
                                retry_if_translation_fails=True)
                with mock.patch.object(legacy_translator, 'GPTTranslator') as gpt, \
                        mock.patch.object(legacy_translator, 'GeminiTranslator') as gemini, \
                        self.assertLogs('main', level='WARNING') as logs:
                    translator = main._create_translator(config)
                constructor, unused = (gemini, gpt) if google_key else (gpt, gemini)
                constructor.assert_called_once()
                unused.assert_not_called()
                self.assertIs(translator, constructor.return_value)
                kwargs = constructor.call_args.kwargs
                self.assertEqual(kwargs['history_size'], 5)
                self.assertTrue(kwargs['use_translation_context'])
                self.assertEqual(kwargs['timeout'], 15)
                self.assertTrue(kwargs['retry_if_translation_fails'])
                for name in ('translation_threads', 'use_old_translator', 'client_pool', 'protocol', 'settings'):
                    self.assertNotIn(name, kwargs)
                self.assertEqual(len(logs.records), 1)
                self.assertIn('deprecated', logs.output[0])

    def test_legacy_selection_never_imports_new_translation_stack(self):
        script = '''
import sys
from stream_translator_gpt.main import Config, _create_translator, _validate_and_normalize
for google_key in (None, 'fake-google-key'):
    config = Config(url='input', translation_prompt='Translate.', openai_api_key='fake-key',
                    google_api_key=google_key, use_old_translator=True, openai_prompt_cache=True)
    _validate_and_normalize(config)
    translator = _create_translator(config)
    assert translator.__class__.__module__ == 'stream_translator_gpt.legacy_translator'
for name in sys.modules:
    assert not name.startswith('stream_translator_gpt.translation_'), name
    assert name != 'stream_translator_gpt.llm_translator', name
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @contextmanager
    def _historical_client_pool(self, config):
        names = ('verify_ssl', '_openai_clients', '_openai_index', '_google_clients', '_google_index',
                 '_whisper_clients', '_whisper_index')
        state = {name: getattr(ClientPool, name) for name in names}
        previous_ids = {id(client) for name in ('_openai_clients', '_google_clients', '_whisper_clients')
                        for client in state[name]}
        try:
            ClientPool.init(openai_api_key=config.openai_api_key, google_api_key=config.google_api_key)
            yield
        finally:
            # Legacy intentionally retains historical global-pool ownership.
            # Tests release their own SDK clients, including unused Whisper and
            # Gemini's independently allocated async transport, then restore globals.
            created = [client for name in ('_openai_clients', '_google_clients', '_whisper_clients')
                       for client in getattr(ClientPool, name) if id(client) not in previous_ids]
            try:
                for client in created:
                    client.close()
                    if hasattr(client, 'aio'):
                        asyncio.run(client.aio.aclose())
            finally:
                for name, value in state.items():
                    setattr(ClientPool, name, value)

    def _check_historical_pipeline(self, provider, mode, retry=False):
        config = Config(url='input', translation_prompt='Translate into Chinese.',
                        openai_api_key='fake-key-a,fake-key-b',
                        google_api_key='fake-key-a,fake-key-b' if provider == 'gemini' else None,
                        openai_translation_api=provider if provider != 'gemini' else 'responses',
                        use_old_translator=True, use_json_result=mode == 'json',
                        use_translation_context=mode == 'context',
                        translation_history_size=2, translation_timeout=4,
                        retry_if_translation_fails=retry, translation_threads=99)
        translator = main._create_translator(config)
        self.assertFalse(hasattr(translator, 'scheduler'))
        self.assertFalse(hasattr(translator, 'stop'))
        self.assertEqual(translator.parallel_max_number, 1 if mode == 'context' else 10)
        recorded, transports, errors = [], {}, []
        record_lock = threading.Lock()

        def send(client, request, **kwargs):
            try:
                body = json.loads(request.content)
                if provider == 'gemini':
                    text = body['contents'][0]['parts'][0]['text']
                    key = request.headers['x-goog-api-key']
                    self.assertTrue(request.url.path.endswith(':generateContent'))
                elif provider == 'responses':
                    text = body['input'][1]['content'][0]['text']
                    key = request.headers['authorization'].removeprefix('Bearer ')
                    self.assertEqual(request.url.path, '/v1/responses')
                else:
                    text = body['messages'][1]['content']
                    key = request.headers['authorization'].removeprefix('Bearer ')
                    self.assertEqual(request.url.path, '/v1/chat/completions')
                with record_lock:
                    recorded.append((text, key))
                    transports[id(client)] = client
                    attempt = len(recorded)
                if retry and attempt == 1:
                    # A non-retryable SDK status exercises the program-level
                    # historical retry path, instead of only SDK internal retries.
                    return httpx.Response(400, request=request, json={'error': {
                        'code': 400, 'message': 'synthetic failure', 'status': 'INVALID_ARGUMENT',
                        'type': 'invalid_request_error'}})
                completion = '这是翻译'
                if mode != 'plain':
                    result = {'translation': completion}
                    if mode == 'context':
                        result['context'] = '正在谈论游戏和猫'
                    completion = json.dumps(result, ensure_ascii=False)
                if provider == 'gemini':
                    payload = {'candidates': [{'index': 0, 'content': {'role': 'model',
                               'parts': [{'text': completion}]}, 'finishReason': 'STOP'}]}
                elif provider == 'responses':
                    payload = {'id': 'resp_test', 'object': 'response', 'created_at': 0,
                               'model': config.gpt_model, 'status': 'completed',
                               'output': [{'type': 'message', 'id': 'msg_test', 'status': 'completed',
                                           'role': 'assistant', 'content': [{'type': 'output_text',
                                           'text': completion, 'annotations': []}]}]}
                else:
                    payload = {'id': 'chat_test', 'object': 'chat.completion', 'created': 0,
                               'model': config.gpt_model, 'choices': [{'index': 0,
                               'message': {'role': 'assistant', 'content': completion}, 'finish_reason': 'stop'}]}
                return httpx.Response(200, request=request, json=payload)
            except BaseException as exc:
                errors.append(exc)
                raise

        incoming, outgoing = queue.SimpleQueue(), queue.SimpleQueue()
        tasks = []
        for task_id in ((10,) if retry else (10, 30, 50)):
            task = TranslationTask(task_id, None, (float(task_id), float(task_id + 1)))
            task.transcript = f'原文{task_id}'
            tasks.append(task)
            incoming.put(task)
        incoming.put(None)

        def run():
            try:
                translator.loop(incoming, outgoing)
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(httpx.Client, 'send', new=send), \
                mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \
                self._historical_client_pool(config):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=7)
            self.assertFalse(worker.is_alive(), 'historical dispatcher failed to drain')
        self.assertFalse(errors, errors)
        self.assertTrue(all(client.is_closed for client in transports.values()))
        results = [outgoing.get_nowait() for _ in tasks]
        self.assertEqual(results, tasks)
        self.assertIsNone(outgoing.get_nowait())
        with self.assertRaises(queue.Empty):
            outgoing.get_nowait()
        self.assertTrue(all(task.translation == '这是翻译' for task in tasks))
        if retry:
            self.assertEqual(len(recorded), 2)
            self.assertEqual(tasks[0].retry_count, 1)
            self.assertEqual([key for _, key in recorded], ['fake-key-a', 'fake-key-b'])
            self.assertEqual(recorded[0][0], recorded[1][0])
        else:
            self.assertEqual(len(recorded), len(tasks))
            self.assertEqual(Counter(key for _, key in recorded), {'fake-key-a': 2, 'fake-key-b': 1})
            self.assertEqual(tasks[-1].context_transcripts, ['原文10', '原文30'])
            if mode == 'context':
                self.assertEqual(tasks[0].input_translation_context, '')
                self.assertEqual(tasks[1].input_translation_context, '正在谈论游戏和猫')
                self.assertEqual(translator.latest_translation_context, '正在谈论游戏和猫')

    def test_historical_pipeline_calls_all_providers_and_output_modes(self):
        # Real installed SDK serializers/parsers and the original global pool;
        # only HTTP transport is replaced, and real sockets are prohibited.
        for provider in ('responses', 'chat-completions', 'gemini'):
            for mode in ('plain', 'json', 'context'):
                with self.subTest(provider=provider, mode=mode):
                    self._check_historical_pipeline(provider, mode)

    def test_historical_program_retry_retains_snapshot_and_rotates_key(self):
        for provider in ('responses', 'chat-completions', 'gemini'):
            with self.subTest(provider=provider):
                self._check_historical_pipeline(provider, 'context', retry=True)

    def test_legacy_loop_exception_uses_original_worker_wrapper_eof(self):
        class Getter:
            def loop(self, output_queue):
                output_queue.put(None)

        class Transcriber:
            def loop(self, input_queue, output_queue):
                input_queue.get(timeout=2)
                output_queue.put(None)

        class Exporter:
            def loop(self, input_queue):
                self.queue = input_queue
                self.received = input_queue.get(timeout=2)

        config = Config(url='input', use_modulate_transcription_api=True, modulate_api_key='test-key',
                        translation_prompt='Translate.', openai_api_key='test-key', use_old_translator=True)
        exporter = Exporter()
        with mock.patch.object(main, 'ClientPool'), \
                mock.patch.object(main, '_create_audio_getter', return_value=Getter()), \
                mock.patch.object(main, 'ModulateStreamingTranscriber', return_value=Transcriber()), \
                mock.patch.object(main, 'ResultExporter', return_value=exporter), \
                mock.patch.object(legacy_translator.GPTTranslator, 'loop',
                                  side_effect=RuntimeError('historical loop failed')) as loop, \
                self.assertRaisesRegex(RuntimeError, 'Pipeline stopped because a worker failed'):
            main.run(config)
        loop.assert_called_once()
        self.assertIsNone(exporter.received)
        with self.assertRaises(queue.Empty):
            exporter.queue.get_nowait()


if __name__ == '__main__':
    unittest.main()
