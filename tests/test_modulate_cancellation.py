"""Offline cancellation tests for every blocking stage of Modulate streaming."""

import json
import queue
import socket
import threading
import unittest
from unittest import mock

import numpy as np

from stream_translator_gpt.transcriber.modulate_transcriber import (
    ModulateStreamingTranscriber, _AudioSendProgress, _AudioStreamState,
)
from stream_translator_gpt.utils.common import start_daemon_thread


def _transcriber(**kwargs):
    return ModulateStreamingTranscriber(
        api_key='test-key', partial_cutoff_seconds=10, proxy=None, language=None,
        transcription_filters='', language_filters=None, print_result=False,
        output_timestamps=False, use_history_context=False, transcription_keywords=None,
        **kwargs,
    )


def _audio_queue(eof=True):
    audio = queue.SimpleQueue()
    audio.put(np.zeros(512, dtype=np.float32))
    if eof:
        audio.put(None)
    return audio


class _Socket:
    def __init__(self):
        self.interrupted = threading.Event()
        self.shutdown_calls = 0

    def shutdown(self, how):
        self.shutdown_calls += 1
        self.interrupted.set()

    def close(self):
        self.interrupted.set()


class _WebSocket:
    def __init__(self, block_send=False):
        self.socket = _Socket()
        self.block_send = block_send
        self.messages = queue.Queue()
        self.sent = []
        self.receiving = threading.Event()
        self.sending_audio = threading.Event()
        self.eof_sent = threading.Event()
        self.exited = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited.set()

    def send(self, value):
        if isinstance(value, bytes):
            self.sending_audio.set()
            if self.block_send:
                if not self.socket.interrupted.wait(3):
                    raise AssertionError('Blocked send was not interrupted')
                raise ConnectionError('Cancelled send')
        self.sent.append(value)
        if value == '':
            self.eof_sent.set()

    def recv(self):
        self.receiving.set()
        while not self.socket.interrupted.is_set():
            try:
                return json.dumps(self.messages.get(timeout=0.02))
            except queue.Empty:
                continue
        raise ConnectionError('Cancelled recv')

    def close(self):
        raise AssertionError('Cancellation must not acquire the WebSocket send mutex')


class ModulateCancellationTest(unittest.TestCase):
    def _run(self, transcriber, input_queue):
        output_queue = queue.SimpleQueue()
        errors = []

        def run():
            try:
                transcriber.loop(input_queue, output_queue)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.addCleanup(transcriber.stop)
        self.addCleanup(worker.join, 2)
        return worker, output_queue, errors

    def _assert_stopped(self, worker, output_queue, errors):
        worker.join(1)
        self.assertFalse(worker.is_alive(), 'Cancellation did not stop the transcriber promptly')
        self.assertEqual(errors, [])
        self.assertIsNone(output_queue.get_nowait())
        self.assertTrue(output_queue.empty(), 'EOF was published more than once')

    def test_stop_before_loop_never_reads_audio_or_connects(self):
        connector = mock.Mock()
        transcriber = _transcriber(connector=connector)
        transcriber.stop()
        transcriber.stop()
        worker, output, errors = self._run(transcriber, queue.SimpleQueue())
        self._assert_stopped(worker, output, errors)
        connector.assert_not_called()

    def test_stop_interrupts_initial_audio_wait(self):
        started = threading.Event()

        class InputQueue(queue.Queue):
            def get(self, *args, **kwargs):
                started.set()
                return super().get(*args, **kwargs)

        connector = mock.Mock()
        transcriber = _transcriber(connector=connector)
        worker, output, errors = self._run(transcriber, InputQueue())
        self.assertTrue(started.wait(1))
        transcriber.stop()
        self._assert_stopped(worker, output, errors)
        connector.assert_not_called()

    def test_stop_interrupts_silent_receive_and_does_not_reconnect(self):
        websocket = _WebSocket()
        connector = mock.Mock(return_value=websocket)
        transcriber = _transcriber(connector=connector)
        worker, output, errors = self._run(transcriber, _audio_queue())
        self.assertTrue(websocket.eof_sent.wait(1))
        self.assertTrue(websocket.receiving.wait(1))
        transcriber.stop()
        transcriber.stop()
        self._assert_stopped(worker, output, errors)
        connector.assert_called_once()
        self.assertTrue(websocket.socket.interrupted.is_set())
        self.assertTrue(websocket.exited.is_set())

    def test_stop_unblocks_real_websockets_receive_without_a_server_reply(self):
        # A local socket pair exercises the actual SDK receive thread without
        # DNS, an HTTP handshake, listening ports, or any external service.
        from websockets.protocol import CLIENT, OPEN, Protocol
        from websockets.sync.connection import Connection

        client, server = socket.socketpair()
        self.addCleanup(server.close)
        self.addCleanup(client.close)
        websocket = Connection(client, Protocol(CLIENT, state=OPEN), ping_interval=None,
                               close_timeout=0.2)
        self.addCleanup(websocket.close)
        receiving = threading.Event()
        original_recv = websocket.recv

        def recv():
            receiving.set()
            return original_recv()

        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket)
        with mock.patch.object(websocket, 'recv', side_effect=recv):
            worker, output, errors = self._run(transcriber, _audio_queue())
            self.assertTrue(receiving.wait(1))
            transcriber.stop()
            self._assert_stopped(worker, output, errors)
        websocket.recv_events_thread.join(1)
        self.assertFalse(websocket.recv_events_thread.is_alive())

    def test_stop_interrupts_blocked_sender_without_waiting_for_send_mutex(self):
        websocket = _WebSocket(block_send=True)
        connector = mock.Mock(return_value=websocket)
        transcriber = _transcriber(connector=connector)
        worker, output, errors = self._run(transcriber, _audio_queue())
        self.assertTrue(websocket.sending_audio.wait(1))
        self.assertTrue(websocket.receiving.wait(1))
        transcriber.stop()
        self._assert_stopped(worker, output, errors)
        connector.assert_called_once()
        self.assertNotIn('', websocket.sent)

    def test_stop_interrupts_initial_config_send_before_sender_thread_exists(self):
        config_started = threading.Event()

        class ConfigBlockedWebSocket(_WebSocket):
            def send(self, value):
                config_started.set()
                if not self.socket.interrupted.wait(3):
                    raise AssertionError('Config send was not interrupted')
                raise ConnectionError('Cancelled config send')

        websocket = ConfigBlockedWebSocket()
        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket)
        worker, output, errors = self._run(transcriber, _audio_queue())
        self.assertTrue(config_started.wait(1))
        transcriber.stop()
        self._assert_stopped(worker, output, errors)
        self.assertFalse(websocket.sending_audio.is_set())

    def test_stop_interrupts_context_manager_close_after_done(self):
        closing = threading.Event()

        class CloseBlockedWebSocket(_WebSocket):
            def __exit__(self, *args):
                closing.set()
                if not self.socket.interrupted.wait(3):
                    raise AssertionError('WebSocket close was not interrupted')
                super().__exit__(*args)

        websocket = CloseBlockedWebSocket()
        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket)
        worker, output, errors = self._run(transcriber, _audio_queue())
        self.assertTrue(websocket.eof_sent.wait(1))
        websocket.messages.put({'type': 'done', 'duration_ms': 32})
        self.assertTrue(closing.wait(1))
        transcriber.stop()
        self._assert_stopped(worker, output, errors)
        self.assertTrue(websocket.exited.is_set())

    def test_stop_interrupts_sender_waiting_for_more_live_audio(self):
        websocket = _WebSocket()
        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket)
        worker, output, errors = self._run(transcriber, _audio_queue(eof=False))
        self.assertTrue(websocket.sending_audio.wait(1))
        transcriber.stop()
        self._assert_stopped(worker, output, errors)
        self.assertNotIn('', websocket.sent)

    def test_stop_interrupts_pacer_wait(self):
        websocket = _WebSocket()
        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket, max_audio_send_rate=0.001)
        audio = _audio_queue(eof=False)
        audio.put(np.zeros(512, dtype=np.float32))
        worker, output, errors = self._run(transcriber, audio)
        self.assertTrue(websocket.sending_audio.wait(1))
        transcriber.stop()
        self._assert_stopped(worker, output, errors)
        self.assertEqual(len([value for value in websocket.sent if isinstance(value, bytes)]), 1)

    def test_reconnect_stop_during_queue_get_preserves_dequeued_audio_or_eof(self):
        for audio in (np.ones(512, dtype=np.float32), None):
            with self.subTest(eof=audio is None):
                connection_stop = threading.Event()
                input_queue = mock.Mock()

                def get(timeout=None):
                    connection_stop.set()
                    return audio

                input_queue.get.side_effect = get
                state = _AudioStreamState()
                websocket = _WebSocket()
                transcriber = _transcriber()
                transcriber._send_audio(websocket, input_queue, state, [], False,
                                        connection_stop, queue.SimpleQueue(), _AudioSendProgress())
                _, payloads, eof = state.replay_snapshot()
                self.assertEqual(eof, audio is None)
                self.assertEqual(payloads, [] if audio is None else [audio.astype('<f4').tobytes()])
                self.assertEqual(websocket.sent, [])

    def test_stop_interrupts_thirty_second_retry_backoff(self):
        backoff = threading.Event()
        connector = mock.Mock(side_effect=ConnectionError('test failure'))
        transcriber = _transcriber(connector=connector, reconnect_backoff_seconds=30)
        original_wait = transcriber._stop_event.wait

        def wait(timeout=None):
            if timeout == 30:
                backoff.set()
            return original_wait(timeout)

        with mock.patch.object(transcriber._stop_event, 'wait', side_effect=wait):
            worker, output, errors = self._run(transcriber, _audio_queue())
            self.assertTrue(backoff.wait(1))
            transcriber.stop()
            self._assert_stopped(worker, output, errors)
        connector.assert_called_once()

    def test_stop_during_handshake_closes_late_connection_without_sending(self):
        started = threading.Event()
        release = threading.Event()
        websocket = _WebSocket()

        def connector(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise AssertionError('Test did not release the handshake')
            return websocket

        transcriber = _transcriber(connector=connector)
        worker, output, errors = self._run(transcriber, _audio_queue())
        self.assertTrue(started.wait(1))
        try:
            transcriber.stop()
            self._assert_stopped(worker, output, errors)
        finally:
            release.set()
        self.assertTrue(websocket.socket.interrupted.wait(1))
        self.assertEqual(websocket.sent, [])

    def test_stop_between_socket_registration_and_handshake_result_consumption(self):
        registered = threading.Event()
        release = threading.Event()
        websocket = _WebSocket()
        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket)
        register = transcriber._register_websocket

        def delay_registration_result(connection):
            result = register(connection)
            registered.set()
            if not release.wait(3):
                raise AssertionError('Test did not release registration')
            return result

        with mock.patch.object(transcriber, '_register_websocket', side_effect=delay_registration_result):
            worker, output, errors = self._run(transcriber, _audio_queue())
            self.assertTrue(registered.wait(1))
            try:
                transcriber.stop()
                self._assert_stopped(worker, output, errors)
                self.assertTrue(websocket.socket.interrupted.is_set())
                self.assertEqual(websocket.sent, [])
            finally:
                release.set()

    def test_stop_does_not_hold_registry_lock_during_socket_shutdown(self):
        websocket = _WebSocket()
        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket)
        transcriber._register_websocket(websocket)
        acquired = []

        def shutdown(_how):
            locked = transcriber._connection_lock.acquire(blocking=False)
            acquired.append(locked)
            if locked:
                transcriber._connection_lock.release()
            transcriber.stop()

        with mock.patch.object(websocket.socket, 'shutdown', side_effect=shutdown):
            transcriber.stop()
        self.assertEqual(acquired, [True])

    def test_simultaneous_connect_completion_and_stop_never_leaks_or_reconnects(self):
        for attempt in range(30):
            with self.subTest(attempt=attempt):
                ready = threading.Event()
                barrier = threading.Barrier(2)
                websocket = _WebSocket()

                def connector(*args, **kwargs):
                    ready.set()
                    barrier.wait(timeout=3)
                    return websocket

                connector_mock = mock.Mock(side_effect=connector)
                transcriber = _transcriber(connector=connector_mock)
                worker, output, errors = self._run(transcriber, _audio_queue())
                self.assertTrue(ready.wait(1))
                barrier.wait(timeout=3)
                transcriber.stop()
                self._assert_stopped(worker, output, errors)
                self.assertTrue(websocket.socket.interrupted.wait(1))
                connector_mock.assert_called_once()

    def test_normal_eof_waits_for_final_and_done_instead_of_cancelling(self):
        websocket = _WebSocket()
        transcriber = _transcriber(connector=lambda *args, **kwargs: websocket)
        worker, output, errors = self._run(transcriber, _audio_queue())
        self.assertTrue(websocket.eof_sent.wait(1))
        self.assertTrue(websocket.receiving.wait(1))
        self.assertTrue(worker.is_alive())
        self.assertFalse(transcriber._stop_event.is_set())
        self.assertFalse(websocket.socket.interrupted.is_set())
        websocket.messages.put({'type': 'utterance', 'utterance': {
            'text': 'final result', 'start_ms': 0, 'duration_ms': 32,
        }})
        websocket.messages.put({'type': 'done', 'duration_ms': 32})
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(output.get_nowait().transcript, 'final result')
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())
        transcriber.stop()
        self.assertTrue(output.empty())

    def test_real_failure_leaves_error_eof_to_worker_wrapper(self):
        connector = mock.Mock(side_effect=ValueError('test failure'))
        transcriber = _transcriber(connector=connector)
        output = queue.SimpleQueue()
        on_error = mock.Mock()
        with mock.patch('stream_translator_gpt.utils.common.logger.exception'):
            worker = start_daemon_thread(transcriber.loop, input_queue=_audio_queue(),
                                         output_queue=output, on_error=on_error)
            worker.join(1)
        self.assertFalse(worker.is_alive())
        on_error.assert_called_once()
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())


if __name__ == '__main__':
    unittest.main()
