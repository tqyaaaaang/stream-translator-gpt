import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from unittest import mock

from stream_translator_gpt import __version__
from stream_translator_gpt.common import ClientPool
from stream_translator_gpt.translation_clients import TranslationClientPool


class TranslationWorkerClientsTest(unittest.TestCase):
    """All SDKs are fakes: synchronization checks cannot issue network requests."""

    def setUp(self):
        names = ('verify_ssl', '_openai_clients', '_openai_index',
                 '_google_clients', '_google_index', '_whisper_clients', '_whisper_index')
        self.pool_state = {name: getattr(ClientPool, name) for name in names}

        def restore_pool_state():
            for name, value in self.pool_state.items():
                setattr(ClientPool, name, value)

        self.addCleanup(restore_pool_state)
        self.clients = []
        self.transports = []
        self.record_lock = threading.Lock()
        test = self

        class FakeTransport:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.owner = threading.get_ident()
                self.close_threads = []
                with test.record_lock:
                    test.transports.append(self)

            def close(self):
                self.close_threads.append(threading.get_ident())

        class FakeClient:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.api_key = kwargs['api_key']
                self.owner = threading.get_ident()
                self.close_threads = []
                self.async_close_threads = []
                self.transport = kwargs.get('http_client')
                self.close_error = False
                self.aio = mock.Mock(aclose=self.aclose)
                with test.record_lock:
                    test.clients.append(self)

            def close(self):
                self.close_threads.append(threading.get_ident())
                if self.transport is not None:
                    self.transport.close()
                if self.close_error:
                    raise RuntimeError('sensitive SDK details must not be logged')

            async def aclose(self):
                self.async_close_threads.append(threading.get_ident())

        self.FakeClient = FakeClient
        self.FakeTransport = FakeTransport
        import httpx
        import openai
        from google import genai
        self.real_httpx_client = httpx.Client
        self.real_openai_client = openai.OpenAI
        self.real_google_client = genai.Client
        self.openai_mock = self.start_patch('openai.OpenAI', side_effect=FakeClient)
        self.google_mock = self.start_patch('google.genai.Client', side_effect=FakeClient)
        self.httpx_mock = self.start_patch('httpx.Client', side_effect=FakeTransport)

    def start_patch(self, target, **kwargs):
        patcher = mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def initialize(self, openai='a,b', google='g,h', **kwargs):
        self.pools = {
            'openai': TranslationClientPool('openai', openai, base_url=kwargs.get('openai_base_url'),
                                            proxy=kwargs.get('proxy'), verify_ssl=kwargs.get('verify_ssl', True)),
            'google': TranslationClientPool('google', google, base_url=kwargs.get('google_base_url'),
                                            proxy=kwargs.get('proxy'), verify_ssl=kwargs.get('verify_ssl', True)),
        }
        self.clients.clear()
        self.transports.clear()
        self.openai_mock.reset_mock()
        self.google_mock.reset_mock()
        self.httpx_mock.reset_mock()

    def test_same_worker_reuses_one_client_per_key_and_closes_on_exit(self):
        for provider in ('openai', 'google'):
            with self.subTest(provider=provider):
                self.initialize()
                with self.pools[provider].worker_session() as get_client:
                    first, second, third = get_client(), get_client(), get_client()
                    self.assertIs(first, third)
                    self.assertIsNot(first, second)
                    self.assertFalse(first.close_threads)
                    self.assertFalse(second.close_threads)
                    self.assertEqual(len(self.clients), 2)
                for client in self.clients:
                    self.assertEqual(client.close_threads, [client.owner])
                    if provider == 'google':
                        self.assertEqual(client.async_close_threads, [client.owner])

    def test_no_clients_are_created_for_an_idle_worker(self):
        self.initialize()
        for provider in ('openai', 'google'):
            with self.pools[provider].worker_session():
                self.assertEqual(self.clients, [])
        self.assertEqual(self.clients, [])

    def test_unused_keys_do_not_create_extra_worker_clients(self):
        self.initialize(openai='a,b,c,d')
        with self.pools['openai'].worker_session() as get_client:
            self.assertEqual(get_client().api_key, 'a')
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(len(self.transports), 1)

    def test_new_pools_configure_keys_without_prebuilding_any_clients(self):
        with self.assertLogs('main', level='INFO') as logs:
            self.initialize()
        self.assertEqual(self.clients, [])
        self.assertEqual(self.transports, [])
        self.assertIn('Configured OpenAI translation key pool with 2 key(s)', '\n'.join(logs.output))
        self.assertIn('Configured Google translation key pool with 2 key(s)', '\n'.join(logs.output))
        self.assertNotIn('Initialized OpenAI client pool', '\n'.join(logs.output))
        self.assertNotIn('Initialized Google client pool', '\n'.join(logs.output))
        with self.pools['openai'].worker_session() as get_client:
            self.assertEqual(get_client().api_key, 'a')
        with self.pools['google'].worker_session() as get_client:
            self.assertEqual(get_client().api_key, 'g')
        self.assertEqual([client.api_key for client in self.clients], ['a', 'g'])

    def test_round_robin_is_global_across_sessions_and_retry_attempts(self):
        self.initialize(openai='a,b,c')
        with self.pools['openai'].worker_session() as first_worker, \
                self.pools['openai'].worker_session() as second_worker:
            first = first_worker()
            second = second_worker()
            retry = first_worker()
            fourth = second_worker()
            self.assertEqual([client.api_key for client in (first, second, retry, fourth)], ['a', 'b', 'c', 'a'])
            self.assertIsNot(first, fourth)

    def test_provider_round_robins_are_independent(self):
        self.initialize()
        with self.pools['openai'].worker_session() as openai, \
                self.pools['google'].worker_session() as google:
            self.assertEqual([openai().api_key, google().api_key, openai().api_key, google().api_key],
                             ['a', 'g', 'b', 'h'])

    def test_same_provider_pools_do_not_share_selection_clients_or_connection_config(self):
        first = TranslationClientPool('openai', 'a,b', base_url='https://first.example/v1', verify_ssl=False)
        second = TranslationClientPool('openai', 'a,b', base_url='https://second.example/v1', verify_ssl=True)
        with first.worker_session() as get_first, second.worker_session() as get_second:
            first_a, second_a = get_first(), get_second()
            self.assertEqual([first_a.api_key, second_a.api_key, get_first().api_key, get_second().api_key],
                             ['a', 'a', 'b', 'b'])
            self.assertIsNot(first_a, second_a)
            self.assertIsNot(first_a.transport, second_a.transport)
            self.assertEqual(first_a.options['base_url'], 'https://first.example/v1')
            self.assertEqual(second_a.options['base_url'], 'https://second.example/v1')
            self.assertFalse(first_a.transport.options['verify'])
            self.assertTrue(second_a.transport.options['verify'])

    def test_concurrent_independent_pools_preserve_separate_round_robins_and_client_bounds(self):
        first = TranslationClientPool('openai', 'a,b', base_url='https://first.example/v1')
        second = TranslationClientPool('openai', 'a,b,c', base_url='https://second.example/v1')
        barrier = threading.Barrier(8)

        def run_worker(pool):
            with pool.worker_session() as get_client:
                barrier.wait(timeout=3)
                return [get_client().api_key for _ in range(120)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            selected = list(executor.map(run_worker, [first] * 4 + [second] * 4))
        self.assertEqual(Counter(key for keys in selected[:4] for key in keys), {'a': 240, 'b': 240})
        self.assertEqual(Counter(key for keys in selected[4:] for key in keys), {'a': 160, 'b': 160, 'c': 160})
        first_clients = [client for client in self.clients if client.options['base_url'] == 'https://first.example/v1']
        second_clients = [client for client in self.clients if client.options['base_url'] == 'https://second.example/v1']
        self.assertLessEqual(len(first_clients), 4 * 2)
        self.assertLessEqual(len(second_clients), 4 * 3)
        for client in self.clients:
            self.assertEqual(client.close_threads, [client.owner])

    def test_connection_config_is_immutable_and_repr_and_logs_do_not_expose_credentials(self):
        secret = 'sensitive-key-value'
        with self.assertLogs('main', level='INFO') as logs:
            pool = TranslationClientPool('openai', secret,
                                         base_url='https://user:sensitive-url-password@gateway.example/v1',
                                         proxy='http://user:sensitive-proxy-password@proxy.example:3128')
        visible = repr(pool._config) + '\n'.join(logs.output)
        self.assertNotIn(secret, visible)
        self.assertNotIn('sensitive-url-password', visible)
        self.assertNotIn('sensitive-proxy-password', visible)
        with self.assertRaises(FrozenInstanceError):
            pool._config.keys = ('replacement',)
        with self.assertRaises(AttributeError):
            pool.provider = 'google'

    def test_duplicate_key_reuses_client_without_changing_round_robin_weight(self):
        self.initialize(openai='a,b,a')
        with self.pools['openai'].worker_session() as get_client:
            clients = [get_client() for _ in range(6)]
            self.assertEqual([client.api_key for client in clients], ['a', 'b', 'a', 'a', 'b', 'a'])
            self.assertIs(clients[0], clients[2])
        self.assertEqual(len(self.clients), 2)

    def test_same_key_is_not_shared_between_concurrent_workers(self):
        for provider in ('openai', 'google'):
            with self.subTest(provider=provider):
                self.initialize(openai='a', google='g')
                barrier = threading.Barrier(4)

                def run_worker(_):
                    with self.pools[provider].worker_session() as get_client:
                        first = get_client()
                        barrier.wait(timeout=3)
                        self.assertIs(first, get_client())
                        self.assertFalse(first.close_threads)
                        return first

                with ThreadPoolExecutor(max_workers=4) as executor:
                    clients = list(executor.map(run_worker, range(4)))
                self.assertEqual(len({id(client) for client in clients}), 4)
                self.assertEqual(len({client.owner for client in clients}), 4)
                for client in clients:
                    self.assertEqual(client.close_threads, [client.owner])
                if provider == 'openai':
                    self.assertEqual(len({id(client.transport) for client in clients}), 4)

    def test_concurrent_round_robin_is_balanced_and_cache_stays_bounded(self):
        self.initialize(openai='a,b,c')
        barrier = threading.Barrier(8)

        def run_worker(_):
            with self.pools['openai'].worker_session() as get_client:
                barrier.wait(timeout=3)
                return [get_client().api_key for _ in range(120)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            selected = [key for result in executor.map(run_worker, range(8)) for key in result]
        self.assertEqual(Counter(selected), {'a': 320, 'b': 320, 'c': 320})
        self.assertLessEqual(len(self.clients), 8 * 3)
        per_worker_key = Counter((client.owner, client.api_key) for client in self.clients)
        self.assertEqual(set(per_worker_key.values()), {1})
        for client in self.clients:
            self.assertEqual(client.close_threads, [client.owner])

    def test_slow_client_creation_does_not_hold_key_selection_lock(self):
        self.initialize(openai='a,b')
        construction_started = threading.Event()
        release_construction = threading.Event()
        other_finished = threading.Event()
        original = self.pools['openai']._new_client

        def make_client(key):
            if key == 'a':
                construction_started.set()
                self.assertTrue(release_construction.wait(timeout=3))
            return original(key)

        def first_worker():
            with self.pools['openai'].worker_session() as get_client:
                return get_client()

        def second_worker():
            with self.pools['openai'].worker_session() as get_client:
                result = get_client()
                other_finished.set()
                return result

        with mock.patch.object(self.pools['openai'], '_new_client', side_effect=make_client), \
                ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(first_worker)
            try:
                self.assertTrue(construction_started.wait(timeout=3))
                second = executor.submit(second_worker)
                self.assertTrue(other_finished.wait(timeout=3))
                self.assertEqual(second.result(timeout=3).api_key, 'b')
            finally:
                release_construction.set()
            self.assertEqual(first.result(timeout=3).api_key, 'a')

    def test_slow_request_keeps_its_client_open_without_blocking_other_workers(self):
        self.initialize(openai='a')
        request_started = threading.Event()
        request_returned = threading.Event()
        first_client = []

        def slow_worker():
            with self.pools['openai'].worker_session() as get_client:
                client = get_client()
                first_client.append(client)
                request_started.set()
                self.assertTrue(request_returned.wait(timeout=3))
                self.assertFalse(client.close_threads)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_worker)
            try:
                self.assertTrue(request_started.wait(timeout=3))
                with self.pools['openai'].worker_session() as get_client:
                    self.assertIsNot(get_client(), first_client[0])
                self.assertFalse(first_client[0].close_threads)
            finally:
                request_returned.set()
            future.result(timeout=3)
        self.assertEqual(first_client[0].close_threads, [first_client[0].owner])

    def test_all_clients_close_when_worker_raises(self):
        self.initialize()
        for provider in ('openai', 'google'):
            with self.subTest(provider=provider):
                with self.assertRaisesRegex(ValueError, 'request failed'):
                    with self.pools[provider].worker_session() as get_client:
                        first, second = get_client(), get_client()
                        raise ValueError('request failed')
                self.assertEqual(first.close_threads, [first.owner])
                self.assertEqual(second.close_threads, [second.owner])

    def test_failed_request_does_not_close_or_rebuild_a_usable_client(self):
        self.initialize(openai='a')
        with self.pools['openai'].worker_session() as get_client:
            first = get_client()
            with self.assertRaisesRegex(ConnectionError, 'connection lost'):
                raise ConnectionError('connection lost')
            self.assertIs(get_client(), first)
            self.assertFalse(first.close_threads)
            self.assertEqual(len(self.clients), 1)

    def test_cleanup_failure_does_not_skip_remaining_clients_or_hide_request_error(self):
        self.initialize()
        with self.assertLogs('main', level='ERROR') as logs, self.assertRaisesRegex(ValueError, 'request failed'):
            with self.pools['google'].worker_session() as get_client:
                first, second = get_client(), get_client()
                first.close_error = True
                raise ValueError('request failed')
        self.assertEqual(first.close_threads, [first.owner])
        self.assertEqual(second.close_threads, [second.owner])
        self.assertEqual(first.async_close_threads, [first.owner])
        self.assertEqual(second.async_close_threads, [second.owner])
        self.assertNotIn('sensitive', '\n'.join(logs.output))

    def test_openai_constructor_failure_closes_unowned_transport(self):
        self.initialize(openai='a')
        self.openai_mock.side_effect = RuntimeError('constructor failed')
        with self.pools['openai'].worker_session() as get_client:
            with self.assertRaisesRegex(RuntimeError, 'constructor failed'):
                get_client()
        self.assertEqual(len(self.transports), 1)
        self.assertEqual(self.transports[0].close_threads, [threading.get_ident()])

    def test_openai_options_preserve_config_and_disable_nested_retries(self):
        self.initialize(openai=' a ', proxy='http://proxy.example:3128',
                        openai_base_url='https://gateway.example/v1', verify_ssl=False)
        with self.pools['openai'].worker_session() as get_client:
            client = get_client()
        self.assertEqual(client.api_key, 'a')
        self.assertEqual(client.options['base_url'], 'https://gateway.example/v1')
        self.assertEqual(client.options['default_headers'], {'User-Agent': f'stream-translator-gpt/{__version__}'})
        self.assertEqual(client.options['max_retries'], 0)
        self.assertEqual(client.options['timeout'], 60.0)
        self.assertEqual(client.transport.options,
                         {'proxy': 'http://proxy.example:3128', 'verify': False, 'timeout': 60.0})

    def test_google_options_preserve_config_and_disable_nested_retries(self):
        self.initialize(google=' g ', proxy='http://proxy.example:3128',
                        google_base_url='https://gateway.example', verify_ssl=False)
        with self.pools['google'].worker_session() as get_client:
            client = get_client()
        options = client.options['http_options']
        self.assertEqual(client.api_key, 'g')
        self.assertEqual(options['base_url'], 'https://gateway.example')
        self.assertEqual(options['client_args'], {'verify': False, 'proxy': 'http://proxy.example:3128'})
        self.assertEqual(options['headers'], {'User-Agent': f'stream-translator-gpt/{__version__}'})
        self.assertEqual(options['timeout'], 60000)
        self.assertEqual(options['retry_options'], {'attempts': 1})
        from google.genai import types
        validated = types.HttpOptions.model_validate(options)
        self.assertEqual(validated.retry_options.attempts, 1)

    def test_installed_sdks_accept_options_and_close_their_real_transports_without_requests(self):
        self.initialize(openai='test-key', google='test-key')
        with mock.patch('openai.OpenAI', self.real_openai_client), \
                mock.patch('google.genai.Client', self.real_google_client), \
                mock.patch('httpx.Client', self.real_httpx_client), \
                mock.patch('httpx.Client.send', side_effect=AssertionError('network must not be used')):
            with self.pools['openai'].worker_session() as get_client:
                openai_client = get_client()
                self.assertEqual(openai_client.max_retries, 0)
                self.assertEqual(openai_client.timeout, 60.0)
                self.assertFalse(openai_client.is_closed())
            self.assertTrue(openai_client.is_closed())
            with self.pools['google'].worker_session() as get_client:
                google_client = get_client()
                self.assertEqual(google_client._api_client._http_options.timeout, 60000)
                self.assertEqual(google_client._api_client._http_options.retry_options.attempts, 1)
                self.assertFalse(google_client._api_client._httpx_client.is_closed)
            self.assertTrue(google_client._api_client._httpx_client.is_closed)
            self.assertTrue(google_client._api_client._async_httpx_client.is_closed)

    def test_bare_ip_uses_same_ssl_policy_as_legacy_client(self):
        with self.assertLogs('main', level='WARNING'):
            self.initialize(openai_base_url='https://127.0.0.1/v1', google_base_url='https://127.0.0.1')
        with self.pools['openai'].worker_session() as get_client:
            self.assertFalse(get_client().transport.options['verify'])
        with self.pools['google'].worker_session() as get_client:
            self.assertFalse(get_client().options['http_options']['client_args']['verify'])

    def test_active_session_keeps_immutable_configuration_when_another_pool_is_created(self):
        self.initialize(openai='a,b', openai_base_url='https://old.example/v1', verify_ssl=False)
        with self.pools['openai'].worker_session() as old_getter:
            self.assertEqual(old_getter().api_key, 'a')
            self.initialize(openai='new', openai_base_url='https://new.example/v1', verify_ssl=True)
            old_client = old_getter()
            self.assertEqual(old_client.api_key, 'b')
            self.assertEqual(old_client.options['base_url'], 'https://old.example/v1')
            self.assertFalse(old_client.transport.options['verify'])
            with self.pools['openai'].worker_session() as new_getter:
                new_client = new_getter()
                self.assertEqual(new_client.api_key, 'new')
                self.assertTrue(new_client.transport.options['verify'])

    def test_legacy_clients_are_not_reused_closed_or_advanced_by_workers(self):
        ClientPool.init(openai_api_key='a,b', google_api_key='g,h')
        self.initialize()
        openai_legacy = tuple(ClientPool._openai_clients)
        google_legacy = tuple(ClientPool._google_clients)
        whisper_legacy = tuple(ClientPool._whisper_clients)
        with self.pools['openai'].worker_session() as get_client:
            self.assertNotIn(get_client(), openai_legacy)
        with self.pools['google'].worker_session() as get_client:
            self.assertNotIn(get_client(), google_legacy)
        self.assertIs(ClientPool.get_openai_client(), openai_legacy[0])
        self.assertIs(ClientPool.get_google_client(), google_legacy[0])
        self.assertIs(ClientPool.get_whisper_client(), whisper_legacy[0])
        for client in openai_legacy + google_legacy + whisper_legacy:
            self.assertFalse(client.close_threads)

    def test_getter_cannot_escape_its_worker_session_or_owner_thread(self):
        self.initialize()
        with self.pools['openai'].worker_session() as get_client:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_client)
                with self.assertRaisesRegex(RuntimeError, 'owning worker session'):
                    future.result(timeout=3)
            self.assertEqual(get_client().api_key, 'a')
        with self.assertRaisesRegex(RuntimeError, 'owning worker session'):
            get_client()

    def test_invalid_or_missing_provider_fails_without_constructing_client(self):
        self.initialize(openai=None, google=None)
        for provider in ('openai', 'google'):
            with self.assertRaisesRegex(RuntimeError, 'key pool is empty'):
                with self.pools[provider].worker_session():
                    self.fail('must not enter missing-key session')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            TranslationClientPool('whisper', 'key')
        self.assertFalse(self.clients)


if __name__ == '__main__':
    unittest.main()
