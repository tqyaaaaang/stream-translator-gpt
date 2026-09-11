"""Exercise the actual composition/SDKs; only HTTP transport is simulated."""
import hashlib
import json
import queue
import socket
import threading
import time
import unittest
from collections import Counter
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest import mock

import httpx

from stream_translator_gpt.utils.common import TranslationTask
from stream_translator_gpt.translator.backends import OpenAITranslationBackend, GeminiTranslationBackend
from stream_translator_gpt.translator.clients import TranslationClientPool
from stream_translator_gpt.translator.pipeline import TranslationPipeline
from stream_translator_gpt.translator.protocol import TranslationProtocol
from stream_translator_gpt.translator.types import TranslationRequest, TranslationResult, TranslationSettings


class _Pool:
    def __init__(self, client):
        self.client, self.opened, self.closed = client, [], []

    @contextmanager
    def worker_session(self):
        owner = threading.get_ident()
        self.opened.append(owner)

        def get_client():
            assert threading.get_ident() == owner
            return self.client

        try:
            yield get_client
        finally:
            self.closed.append(threading.get_ident())


def backend(client, context=True, **options):
    return OpenAITranslationBackend(client_pool=_Pool(client), model='gpt-5.6-luna',
                                   protocol=TranslationProtocol('Translate faithfully.', 2,
                                                                use_translation_context=context), **options)


class ProtocolAndBackendTest(unittest.TestCase):
    def test_messages_match_pre_refactor_golden_hashes(self):
        # Captured from the parallel implementation before responsibility split.
        golden = {
            (0, False, False): '585794d469bd1efe435e919c44d6802c3f1cf0fb51b63b715fd5b1791275e7aa',
            (0, False, True): 'e3b38c5d1a63c6f6278d060f49f3dd6777ded1c48213de902e052adb9fe7e109',
            (0, True, False): '018cf56332991cca93b0b02a27ac3ea5127eb6f4d3587978155e2e202b1ebc61',
            (2, False, False): '92bbffe34ad3d10c795a3fb112b50c6132a67649b433d361130255b91618d179',
            (2, False, True): '727e4ba42746d50383b9f1ecb0b5d88af251dea77a74825e3918c7e8e6a9e352',
            (2, True, False): 'e2b84632ebc4f85c2b6143a10e7a158b22e1815aa8867a22ea090386a33b9fa3',
        }
        for (history, context, use_json), expected in golden.items():
            with self.subTest(history=history, context=context, use_json=use_json):
                protocol = TranslationProtocol('日文直播 → 简体中文；忠实翻译。', history, use_json, context)
                request = TranslationRequest(70, 'むーちゃん、ゲームの続きだよ。',
                                             ('猫が寝ていた。', '昨日のボス戦を再開する。') if history else (),
                                             '猫の話からゲームに戻った。')
                actual = hashlib.sha256(json.dumps(protocol.build_messages(request), ensure_ascii=False).encode())
                self.assertEqual(actual.hexdigest(), expected)

    def test_parser_modes_and_invalid_context(self):
        self.assertEqual(TranslationProtocol('').parse('first\nlast'), TranslationResult('last'))
        self.assertEqual(TranslationProtocol('', use_json_result=True).parse('{"translation":"译文"}'),
                         TranslationResult('译文'))
        protocol = TranslationProtocol('', use_translation_context=True)
        self.assertEqual(protocol.parse('{"translation":"一\\n二","context":"摘要"}'),
                         TranslationResult('一 二', '摘要'))
        for value in ('', None, '{}', '{"translation":"译文"}', '{"translation":"译文","context":""}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.parse(value)

    def test_values_are_frozen_and_backend_has_no_pipeline_responsibilities(self):
        for value, field in ((TranslationRequest(1, 'text'), 'context'),
                             (TranslationResult('text'), 'translation'), (TranslationProtocol('text'), 'prompt')):
            with self.assertRaises(FrozenInstanceError):
                setattr(value, field, 'changed')
        current = backend(None)
        for attr in ('loop', 'stop', '_get_scheduler', '_worker_local', 'use_old_translator',
                     'latest_translation_context', 'processing_queue'):
            self.assertFalse(hasattr(current, attr), attr)

    def test_session_returns_result_and_closes_without_mutating_request(self):
        create = mock.Mock(return_value=SimpleNamespace(output_text='{"translation":"译文","context":"摘要"}'))
        current = backend(SimpleNamespace(responses=SimpleNamespace(create=create)))
        request = TranslationRequest(1, 'source', ('prior',), 'memory', time.monotonic() + 2)
        with current.open_session() as session:
            self.assertEqual(session.translate(request), TranslationResult('译文', '摘要'))
        self.assertEqual(request.context, 'memory')
        self.assertFalse(hasattr(request, 'translation'))
        self.assertEqual(current.client_pool.opened, current.client_pool.closed)

    def test_error_propagates_and_closes_session(self):
        current = backend(SimpleNamespace(responses=SimpleNamespace(create=mock.Mock(side_effect=TimeoutError))))
        with self.assertRaises(TimeoutError), current.open_session() as session:
            session.translate(TranslationRequest(1, 'source'))
        self.assertEqual(current.client_pool.opened, current.client_pool.closed)

    def test_prefix_count_and_generation_share_deadline(self):
        for remaining in (2, 0):
            with self.subTest(remaining=remaining):
                now = [10.0]

                def count(**kwargs):
                    self.assertEqual(kwargs['timeout'], 5)
                    now[0] = 15.0 - remaining
                    return SimpleNamespace(input_tokens=1100)

                create = mock.Mock(return_value=SimpleNamespace(output_text='{"translation":"译文","context":"摘要"}'))
                current = backend(SimpleNamespace(responses=SimpleNamespace(
                    input_tokens=SimpleNamespace(count=count), create=create)), prompt_cache_enabled=True)
                with mock.patch('stream_translator_gpt.translator.backends.time.monotonic', side_effect=lambda: now[0]):
                    with current.open_session() as session:
                        request = TranslationRequest(1, 'source', deadline=15)
                        if remaining:
                            session.translate(request)
                            self.assertEqual(create.call_args.kwargs['timeout'], remaining)
                        else:
                            with self.assertRaises(TimeoutError):
                                session.translate(request)
                            create.assert_not_called()

    def test_slow_prefix_count_does_not_block_other_workers(self):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        failures = []

        def count(**kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError('count not released')
            return SimpleNamespace(input_tokens=1100)

        current = backend(SimpleNamespace(responses=SimpleNamespace(input_tokens=SimpleNamespace(count=count),
                          create=lambda **kwargs: SimpleNamespace(output_text='ok'))), context=False,
                          prompt_cache_enabled=True)

        def run(completed=None):
            try:
                with current.open_session() as session:
                    session.translate(TranslationRequest(1, 'source'))
            except BaseException as exc:
                failures.append(exc)
            finally:
                if completed:
                    completed.set()

        first, second = threading.Thread(target=run), threading.Thread(target=run, args=(done,))
        first.start()
        try:
            self.assertTrue(entered.wait(1))
            second.start()
            self.assertTrue(done.wait(1))
        finally:
            release.set()
            first.join(2)
            if second.ident is not None:
                second.join(2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(failures, [])


class RealSDKPipelineTest(unittest.TestCase):
    def check_pipeline(self, provider, mode, retry=False):
        context = mode == 'context'
        protocol = TranslationProtocol('Translate faithfully.', 2, mode == 'json', context)
        pool = TranslationClientPool('google' if provider == 'gemini' else 'openai', 'fake-key-a,fake-key-b')
        if provider == 'gemini':
            current = GeminiTranslationBackend(client_pool=pool, protocol=protocol, model='gemini-3.5-flash-lite')
        else:
            current = OpenAITranslationBackend(client_pool=pool, protocol=protocol, model='gpt-5.6-luna',
                                              api_mode='responses' if provider == 'responses' else 'chat-completions')
        pipeline = TranslationPipeline(current, TranslationSettings(
            workers=3, history_size=2, use_translation_context=context, timeout=5, retry_if_translation_fails=retry))
        incoming, outgoing = queue.SimpleQueue(), queue.SimpleQueue()
        count = 1 if retry else 6
        for i in range(count):
            task = TranslationTask(i * 10, None, (i, i + 1))
            task.transcript = f'source-{i}'
            incoming.put(task)
        incoming.put(None)
        calls, completed, clients = [], [], {}
        lock = threading.Lock()
        barrier = threading.Barrier(3, timeout=2)
        zero, one = threading.Event(), threading.Event()

        def send(client, request, **kwargs):
            body = json.loads(request.content)
            if provider == 'gemini':
                text, key = body['contents'][0]['parts'][0]['text'], request.headers['x-goog-api-key']
                self.assertTrue(request.url.path.endswith(':generateContent'))
            elif provider == 'responses':
                text = body['input'][1]['content'][0]['text']
                key = request.headers['authorization'].removeprefix('Bearer ')
                self.assertEqual(request.url.path, '/v1/responses')
                self.assertFalse(body['store'])
            else:
                text = body['messages'][1]['content']
                key = request.headers['authorization'].removeprefix('Bearer ')
                self.assertEqual(request.url.path, '/v1/chat/completions')
            index = int(text.rsplit('source-', 1)[1])
            for timeout in request.extensions['timeout'].values():
                self.assertGreater(timeout, 0)
                self.assertLessEqual(timeout, 5)
            with lock:
                owner = threading.get_ident()
                if id(client) in clients:
                    self.assertEqual(clients[id(client)][1], owner)
                clients[id(client)] = (client, owner)
                calls.append((index, key, text))
                attempt = len(calls)
            if retry and attempt == 1:
                return httpx.Response(400, request=request, json={'error': {
                    'code': 400, 'message': 'synthetic error', 'status': 'INVALID_ARGUMENT', 'type': 'invalid_request_error'}})
            if not retry:
                if index < 3:
                    barrier.wait()
                if index == 0:
                    self.assertTrue(zero.wait(2))
                elif index == 1:
                    self.assertTrue(one.wait(2))
                elif index == 3:
                    if context:
                        self.assertIn('**Context:**\ncontext-2\n\n', text)
                    one.set()
                elif index == 4:
                    zero.set()
            content = f'translation-{index}'
            if context:
                content = json.dumps({'translation': content, 'context': f'context-{index}'})
            elif mode == 'json':
                content = json.dumps({'translation': content})
            with lock:
                completed.append(index)
            if provider == 'gemini':
                payload = {'candidates': [{'index': 0, 'content': {'role': 'model', 'parts': [{'text': content}]},
                                           'finishReason': 'STOP'}]}
            elif provider == 'responses':
                payload = {'id': 'resp_test', 'object': 'response', 'created_at': 0, 'model': current.model,
                           'status': 'completed', 'output': [{'type': 'message', 'id': 'msg_test',
                           'status': 'completed', 'role': 'assistant',
                           'content': [{'type': 'output_text', 'text': content, 'annotations': []}]}]}
            else:
                payload = {'id': 'chat_test', 'object': 'chat.completion', 'created': 0, 'model': current.model,
                           'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                                        'finish_reason': 'stop'}]}
            return httpx.Response(200, request=request, json=payload)

        errors = []

        def run():
            try:
                pipeline.loop(incoming, outgoing)
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(httpx.Client, 'send', new=send), \
                mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
            runner = threading.Thread(target=run, daemon=True)
            runner.start()
            runner.join(8)
            if runner.is_alive():
                pipeline.stop()
                zero.set()
                one.set()
                runner.join(2)
                self.fail('Pipeline did not finish')
        self.assertFalse(errors, errors)
        results = [outgoing.get_nowait() for _ in range(count)]
        self.assertEqual([result.id for result in results], [i * 10 for i in range(count)])
        self.assertEqual([result.translation for result in results], [f'translation-{i}' for i in range(count)])
        self.assertIsNone(outgoing.get_nowait())
        self.assertTrue(outgoing.empty())
        self.assertTrue(pipeline.output_finished.is_set())
        self.assertTrue(all(client.is_closed for client, _ in clients.values()))
        if retry:
            self.assertEqual(len(calls), 2)
            self.assertEqual([key for _, key, _ in calls], ['fake-key-a', 'fake-key-b'])
            self.assertEqual(calls[0][2], calls[1][2])
        else:
            self.assertEqual(completed[0], 2)
            self.assertNotEqual(completed, list(range(count)))
            self.assertEqual(len({owner for _, owner in clients.values()}), 3)
            self.assertEqual(Counter(key for _, key, _ in calls), {'fake-key-a': 3, 'fake-key-b': 3})

    def test_providers_and_output_modes(self):
        for provider in ('responses', 'chat', 'gemini'):
            for mode in ('plain', 'json', 'context'):
                with self.subTest(provider=provider, mode=mode):
                    self.check_pipeline(provider, mode)

    def test_program_retry_and_key_rotation(self):
        for provider in ('responses', 'chat', 'gemini'):
            with self.subTest(provider=provider):
                self.check_pipeline(provider, 'context', retry=True)


if __name__ == '__main__':
    unittest.main()
