import io
import logging
import queue
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from stream_translator_gpt import logging_utils
from stream_translator_gpt.common import ClientPool, start_daemon_thread
from stream_translator_gpt.logging_utils import (DDEBUG, configure_logging, normalize_verbosity_args,
                                                 redact_command, redact_url, shutdown_logging, verbosity_from_argv)


class LoggingUtilsTest(unittest.TestCase):

    def tearDown(self):
        shutdown_logging()

    def test_default_logging_is_warning_with_plain_format(self):
        output = io.StringIO()
        logger = configure_logging(0, stream=output)

        logger.info('hidden')
        logger.warning('visible')
        shutdown_logging()

        self.assertEqual(output.getvalue(), 'visible\n')

    def test_verbose_logging_includes_level_and_thread(self):
        output = io.StringIO()
        logger = configure_logging(1, stream=output)

        logger.info('ready')
        shutdown_logging()

        self.assertRegex(output.getvalue(), r'^\d\d:\d\d:\d\d\.\d{3}:INFO:main: ready\n$')

    def test_ddebug_has_a_registered_name_and_uses_level_three(self):
        output = io.StringIO()
        logger = configure_logging(3, stream=output)

        logger.log(DDEBUG, 'details')
        shutdown_logging()

        self.assertEqual(logging.getLevelName(DDEBUG), 'DDEBUG')
        self.assertIn(':DDEBUG:main: details', output.getvalue())

    def test_reconfiguration_does_not_duplicate_records(self):
        first_output = io.StringIO()
        second_output = io.StringIO()
        logger = configure_logging(1, stream=first_output)
        logger.info('first')

        logger = configure_logging(1, stream=second_output)
        logger.info('second')
        shutdown_logging()

        self.assertEqual(first_output.getvalue().count('first'), 1)
        self.assertNotIn('second', first_output.getvalue())
        self.assertEqual(second_output.getvalue().count('second'), 1)

    def test_records_are_retained_while_the_listener_is_reconfigured(self):

        class GateStream(io.StringIO):

            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()

            def write(self, value):
                if 'before reconfigure' in value:
                    self.entered.set()
                    self.release.wait(timeout=1)
                return super().write(value)

        output = GateStream()
        logger = configure_logging(0, stream=output)
        logger.warning('before reconfigure')
        self.assertTrue(output.entered.wait(timeout=1))

        reconfigured = threading.Event()

        def reconfigure():
            configure_logging(0, stream=output)
            reconfigured.set()

        thread = threading.Thread(target=reconfigure)
        thread.start()

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            log_queue = logging_utils._queue_handler.queue
            with log_queue.mutex:
                if any(item is None for item in log_queue.queue):
                    break
            time.sleep(0.001)
        else:
            self.fail('listener did not begin reconfiguration')

        logger.warning('during reconfigure')
        output.release.set()
        thread.join(timeout=1)
        self.assertTrue(reconfigured.is_set())
        shutdown_logging()

        self.assertEqual(output.getvalue().count('before reconfigure'), 1)
        self.assertEqual(output.getvalue().count('during reconfigure'), 1)

    def test_listener_formats_exception_records(self):
        output = io.StringIO()
        logger = configure_logging(0, stream=output)

        try:
            raise RuntimeError('boom')
        except RuntimeError:
            logger.exception('worker failed')
        shutdown_logging()

        self.assertIn('worker failed', output.getvalue())
        self.assertIn('RuntimeError: boom', output.getvalue())

    def test_worker_exception_is_logged_and_propagates_end_of_stream(self):
        output = io.StringIO()
        output_queue = queue.SimpleQueue()
        configure_logging(2, stream=output)

        def fail(output_queue):
            raise RuntimeError('worker boom')

        thread = start_daemon_thread(fail, name='test-worker', output_queue=output_queue)
        thread.join(timeout=1)
        shutdown_logging()

        self.assertFalse(thread.is_alive())
        self.assertIsNone(output_queue.get_nowait())
        self.assertEqual(output.getvalue().count('Unhandled exception in worker: test-worker'), 1)
        self.assertEqual(output.getvalue().count('RuntimeError: worker boom'), 1)

    def test_verbosity_argument_compatibility(self):
        self.assertEqual(verbosity_from_argv(['-v']), 1)
        self.assertEqual(verbosity_from_argv(['-vv']), 2)
        self.assertEqual(verbosity_from_argv(['-vvv']), 3)
        self.assertEqual(verbosity_from_argv(['--verbose', '2']), 2)
        self.assertEqual(verbosity_from_argv(['--verbose=3']), 3)
        self.assertEqual(verbosity_from_argv(['--verbosity', 'high']), 0)

        self.assertEqual(normalize_verbosity_args(['-v', '2', '--verbosity', 'high']),
                         ['--verbose', '--verbose', '--verbosity', 'high'])

    def test_urls_and_command_diagnostics_are_redacted(self):
        self.assertEqual(redact_url('https://user:password@example.com/live?id=signed#token'),
                         'https://<redacted>@example.com/live?<redacted>#<redacted>')
        self.assertEqual(redact_url('data:text/plain,a-secret'), 'data:<redacted>')
        command = [
            'python', '-m', 'stream_translator_gpt', 'https://example.com/live?signature=secret',
            '--openai-api-key', 'sk-secret', '--telegram-token=bot-secret', '--verbosity', 'high'
        ]
        rendered = ' '.join(redact_command(command))
        self.assertNotIn('signature=secret', rendered)
        self.assertNotIn('sk-secret', rendered)
        self.assertNotIn('bot-secret', rendered)
        self.assertIn('--verbosity high', rendered)

    def test_client_pool_rotation_remains_atomic_with_debug_logging(self):
        output = io.StringIO()
        configure_logging(2, stream=output)
        original_clients = ClientPool._openai_clients
        original_index = ClientPool._openai_index
        clients = [object() for _ in range(4)]
        try:
            ClientPool._openai_clients = clients
            ClientPool._openai_index = 0
            with ThreadPoolExecutor(max_workers=16) as executor:
                selected = list(executor.map(lambda _: ClientPool.get_openai_client(), range(400)))
        finally:
            ClientPool._openai_clients = original_clients
            ClientPool._openai_index = original_index

        self.assertEqual(Counter(map(id, selected)), Counter({id(client): 100 for client in clients}))


if __name__ == '__main__':
    unittest.main()
