import json
import queue
import ssl
import threading
import unittest
from unittest import mock
from urllib.parse import parse_qs, quote, quote_plus, urlsplit

import numpy as np
import tyro
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from stream_translator_gpt.main import Config, _validate_and_normalize
from stream_translator_gpt.modulate_transcriber import (ModulateStreamingError, ModulateStreamingTranscriber,
                                                        _AudioStreamState)


def _transcriber(**overrides):
    kwargs = {
        'api_key': 'key with +?/ characters',
        'partial_cutoff_seconds': 10.0,
        'reconnect_attempts': 0,
        'reconnect_backoff_seconds': 0.0,
        'proxy': None,
        'language': None,
        'transcription_filters': '',
        'language_based_filter': False,
        'print_result': False,
        'output_timestamps': False,
        'use_history_context': False,
        'transcription_keywords': None,
    }
    kwargs.update(overrides)
    return ModulateStreamingTranscriber(**kwargs)


def _websocket_close(code: int) -> ConnectionClosedError:
    if code == 1006:
        # websockets represents an abnormal closure as the absence of both
        # close frames; ConnectionClosed.code resolves this case to 1006.
        return ConnectionClosedError(None, None)
    return ConnectionClosedError(Close(code, f'test close {code}'), None)


def _handshake_status(status_code: int) -> InvalidStatus:
    return InvalidStatus(Response(status_code, 'test response', Headers()))


class _FakeWebSocket:

    def __init__(self, messages=None, receive_error=None, wait_for_eof=True):
        self.messages = list(messages or [])
        self.receive_error = receive_error
        self.wait_for_eof = wait_for_eof
        self.sent = []
        self.eof_sent = threading.Event()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def send(self, value):
        self.sent.append(value)
        if value == '':
            self.eof_sent.set()

    def recv(self):
        if self.wait_for_eof and not self.eof_sent.wait(timeout=2):
            raise TimeoutError('test sender never sent EOF')
        if self.receive_error is not None:
            raise self.receive_error
        if not self.messages:
            raise RuntimeError('test message list exhausted')
        return json.dumps(self.messages.pop(0))

    def close(self):
        self.closed = True
        self.eof_sent.set()


class _SenderFailureWebSocket(_FakeWebSocket):

    def __init__(self, send_error):
        super().__init__(wait_for_eof=False)
        self.send_error = send_error
        self.audio_attempted = threading.Event()

    def send(self, value):
        if isinstance(value, bytes):
            self.audio_attempted.set()
            raise self.send_error
        super().send(value)

    def recv(self):
        if not self.audio_attempted.wait(timeout=2):
            raise TimeoutError('test sender never attempted audio')
        return json.dumps({'type': 'done', 'duration_ms': 0})


class _InterruptibleSocket:

    def __init__(self):
        self.interrupted = threading.Event()
        self.shutdown_calls = 0

    def shutdown(self, _how):
        self.shutdown_calls += 1
        self.interrupted.set()

    def close(self):
        self.interrupted.set()


class _BlockingSendWebSocket(_FakeWebSocket):

    def __init__(self):
        super().__init__(wait_for_eof=False)
        self.socket = _InterruptibleSocket()
        self.audio_attempted = threading.Event()

    def send(self, value):
        if isinstance(value, bytes):
            self.audio_attempted.set()
            if not self.socket.interrupted.wait(timeout=2):
                raise TimeoutError('test socket was not interrupted')
            raise ConnectionError('blocked send interrupted')
        super().send(value)

    def recv(self):
        if not self.audio_attempted.wait(timeout=2):
            raise TimeoutError('test sender never attempted audio')
        raise ConnectionError('network dropped while sender was blocked')


class _EofFailureWebSocket(_FakeWebSocket):

    def __init__(self):
        super().__init__(wait_for_eof=False)
        self.eof_attempted = threading.Event()

    def send(self, value):
        self.sent.append(value)
        if value == '':
            self.eof_attempted.set()
            raise ConnectionError('network dropped while sending EOF')

    def recv(self):
        if not self.eof_attempted.wait(timeout=2):
            raise TimeoutError('test sender never attempted EOF')
        raise ConnectionError('network dropped after EOF send failed')


class _DropAfterFirstAudioWebSocket(_FakeWebSocket):

    def __init__(self):
        super().__init__(wait_for_eof=False)
        self.audio_sent = threading.Event()
        self.interrupted = threading.Event()

    def send(self, value):
        super().send(value)
        if isinstance(value, bytes):
            self.audio_sent.set()

    def recv(self):
        if not self.audio_sent.wait(timeout=2):
            raise TimeoutError('test sender never sent the first audio chunk')
        raise ConnectionError('network dropped while input was still open')

    def close(self):
        super().close()
        self.interrupted.set()


class _GatedDisconnectInputQueue:
    """Keep later chunks unavailable until the first sender has been stopped."""

    def __init__(self, first, remaining, interrupted: threading.Event):
        self.first = first
        self.remaining = list(remaining)
        self.interrupted = interrupted
        self.first_returned = False
        self.first_sender_released = False
        self.lock = threading.Lock()

    def get(self, timeout=None):
        with self.lock:
            if not self.first_returned:
                self.first_returned = True
                return self.first

        if not self.interrupted.wait(timeout=timeout):
            raise queue.Empty

        with self.lock:
            # The first connection's sender is already inside get() when the
            # receiver interrupts it. Make that call empty so it observes its
            # stop event before the reconnect may consume the remaining input.
            if not self.first_sender_released:
                self.first_sender_released = True
                raise queue.Empty
            if not self.remaining:
                raise queue.Empty
            return self.remaining.pop(0)


class ModulateProtocolTest(unittest.TestCase):

    def test_transcript_print_uses_shared_utf8_writer(self):
        with mock.patch('stream_translator_gpt.modulate_transcriber.write_console_utf8') as writer:
            ModulateStreamingTranscriber._print_transcript('テスト。')
        writer.assert_called_once_with('テスト。')

    def test_url_encodes_key_and_declares_raw_float_audio(self):
        transcriber = _transcriber()
        parsed = urlsplit(transcriber._connection_url())
        query = parse_qs(parsed.query)

        self.assertEqual(query['api_key'], ['key with +?/ characters'])
        self.assertEqual(query['partial_results'], ['true'])
        self.assertEqual(query['audio_format'], ['f32le'])
        self.assertEqual(query['sample_rate'], ['16000'])
        self.assertEqual(query['num_channels'], ['1'])

    def test_configuration_frame_contains_language_and_custom_terms(self):
        transcriber = _transcriber(language='ja', transcription_keywords='Modulate, Velma')
        self.assertEqual(transcriber._connection_config(), {
            'partial_results': True,
            'language': 'ja',
            'custom_terms': ['Modulate', 'Velma'],
        })

    def test_auto_language_is_omitted_for_server_side_detection(self):
        transcriber = _transcriber(language='auto')
        self.assertEqual(transcriber._connection_config(), {'partial_results': True})

    def test_stream_sends_config_float32_audio_text_eof_and_waits_for_done(self):
        websocket = _FakeWebSocket([
            {
                'type': 'utterance',
                'utterance': {
                    'utterance_uuid': 'u1',
                    'text': 'hello.',
                    'start_ms': 0,
                    'duration_ms': 64,
                },
            },
            {'type': 'done', 'duration_ms': 64},
        ])
        transcriber = _transcriber(connector=lambda *_args, **_kwargs: websocket)
        input_queue = queue.SimpleQueue()
        first = np.array([0.25, -0.5], dtype=np.float64)
        second = np.array([1.0, 0.0], dtype=np.float32)
        input_queue.put(first)
        input_queue.put(second)
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        config = json.loads(websocket.sent[0])
        self.assertTrue(config['partial_results'])
        binary_messages = [message for message in websocket.sent if isinstance(message, bytes)]
        self.assertEqual(len(binary_messages), 2)
        np.testing.assert_array_equal(np.frombuffer(binary_messages[0], dtype='<f4'), first.astype(np.float32))
        self.assertEqual(websocket.sent[-1], '')
        task = output_queue.get_nowait()
        self.assertEqual(task.transcript, 'hello.')
        self.assertEqual(task.time_range, (0.0, 0.064))
        self.assertIsNone(output_queue.get_nowait())

    def test_duplicate_final_uuid_is_ignored(self):
        websocket = _FakeWebSocket([
            {
                'type': 'utterance',
                'utterance': {
                    'utterance_uuid': 'same',
                    'text': 'once.',
                    'start_ms': 0,
                    'duration_ms': 32,
                },
            },
            {
                'type': 'utterance',
                'utterance': {
                    'utterance_uuid': 'same',
                    'text': 'once.',
                    'start_ms': 0,
                    'duration_ms': 32,
                },
            },
            {'type': 'done', 'duration_ms': 32},
        ])
        transcriber = _transcriber(connector=lambda *_args, **_kwargs: websocket)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        self.assertEqual(output_queue.get_nowait().transcript, 'once.')
        self.assertIsNone(output_queue.get_nowait())
        with self.assertRaises(queue.Empty):
            output_queue.get_nowait()

    def test_done_flushes_pending_partial_exactly_once_without_a_final(self):
        websocket = _FakeWebSocket([
            {
                'type': 'partial_utterance',
                'partial_utterance': {
                    'text': 'pending preview.',
                    'start_ms': 0,
                },
            },
            {'type': 'done', 'duration_ms': 32},
        ])
        transcriber = _transcriber(connector=lambda *_args, **_kwargs: websocket)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        task = output_queue.get_nowait()
        self.assertEqual(task.transcript, 'pending preview.')
        self.assertEqual(task.time_range, (0.0, 0.032))
        self.assertIsNone(output_queue.get_nowait())
        with self.assertRaises(queue.Empty):
            output_queue.get_nowait()

    def test_server_error_is_not_retried_and_redacts_key(self):
        key = 'top-secret-key'
        websocket = _FakeWebSocket([{'type': 'error', 'error': f'bad credential {key}'}], wait_for_eof=False)
        calls = []

        def connector(*args, **kwargs):
            calls.append((args, kwargs))
            return websocket

        transcriber = _transcriber(api_key=key, reconnect_attempts=3, connector=connector, sleeper=lambda _: None)
        with self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(queue.SimpleQueue(), queue.SimpleQueue())

        self.assertEqual(len(calls), 1)
        self.assertNotIn(key, str(raised.exception))
        self.assertIn('<redacted>', str(raised.exception))

    def test_server_error_redacts_url_encoded_and_double_encoded_key(self):
        key = 'key with +?/ characters'
        encoded_key = quote_plus(key, safe='')
        double_encoded_key = quote(encoded_key, safe='')
        websocket = _FakeWebSocket([{
            'type': 'error',
            'error': f'failed URL ?api_key={encoded_key}; nested={double_encoded_key}',
        }], wait_for_eof=False)
        transcriber = _transcriber(api_key=key, connector=lambda *_args, **_kwargs: websocket)

        with self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(queue.SimpleQueue(), queue.SimpleQueue())

        error_text = str(raised.exception)
        self.assertNotIn(key, error_text)
        self.assertNotIn(encoded_key, error_text)
        self.assertNotIn(double_encoded_key, error_text)
        self.assertIn('<redacted>', error_text)

    def test_server_error_remains_nonretryable_when_sender_also_observes_close(self):
        class ServerAndSenderFailure(_FakeWebSocket):

            def __init__(self):
                super().__init__()
                self.audio_attempted = threading.Event()

            def send(self, value):
                if isinstance(value, bytes):
                    self.audio_attempted.set()
                    raise ConnectionError('socket already closed')
                super().send(value)

            def recv(self):
                if not self.audio_attempted.wait(timeout=2):
                    raise TimeoutError('sender did not attempt audio')
                return json.dumps({'type': 'error', 'error': 'invalid credentials'})

        websocket = ServerAndSenderFailure()
        calls = 0

        def connector(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return websocket

        transcriber = _transcriber(reconnect_attempts=3, connector=connector, sleeper=lambda _: None)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        with self.assertRaisesRegex(ModulateStreamingError, 'Modulate API error'):
            transcriber.loop(input_queue, queue.SimpleQueue())
        self.assertEqual(calls, 1)

    def test_local_result_processing_error_is_not_hidden_by_connection_cleanup(self):
        websocket = _FakeWebSocket([{
            'type': 'partial_utterance',
            'partial_utterance': {'text': 'text', 'start_ms': 0},
        }])
        calls = 0

        def connector(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return websocket

        transcriber = _transcriber(reconnect_attempts=3, connector=connector, sleeper=lambda _: None)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(None)
        with mock.patch.object(transcriber, '_handle_partial', side_effect=UnicodeEncodeError(
                'ascii', 'テスト', 0, 1, 'test console')):
            with self.assertRaisesRegex(ModulateStreamingError, 'UnicodeEncodeError'):
                transcriber.loop(input_queue, queue.SimpleQueue())
        self.assertEqual(calls, 1)


class ModulateReconnectTest(unittest.TestCase):

    def test_failed_eof_send_reconnects_replays_audio_and_resends_eof(self):
        first = _EofFailureWebSocket()
        second = _FakeWebSocket([{'type': 'done', 'duration_ms': 32}])
        connections = iter([first, second])
        transcriber = _transcriber(reconnect_attempts=1,
                                   connector=lambda *_args, **_kwargs: next(connections),
                                   sleeper=lambda _: None)
        audio = np.arange(512, dtype=np.float32)
        input_queue = queue.SimpleQueue()
        input_queue.put(audio)
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        first_audio = [value for value in first.sent if isinstance(value, bytes)]
        second_audio = [value for value in second.sent if isinstance(value, bytes)]
        self.assertEqual(first_audio, [audio.astype('<f4').tobytes()])
        self.assertEqual(second_audio, first_audio)
        self.assertEqual(first.sent[-1], '')
        self.assertEqual(second.sent[-1], '')
        self.assertIsNone(output_queue.get_nowait())

    def test_midstream_disconnect_replays_sent_audio_then_consumes_remaining_input(self):
        first = _DropAfterFirstAudioWebSocket()
        second = _FakeWebSocket([
            {
                'type': 'utterance',
                'utterance': {
                    'utterance_uuid': 'complete-after-reconnect',
                    'text': 'all chunks arrived.',
                    'start_ms': 0,
                    'duration_ms': 96,
                },
            },
            {'type': 'done', 'duration_ms': 96},
        ])
        connections = iter([first, second])
        chunks = [
            np.arange(512, dtype=np.float32),
            np.arange(512, dtype=np.float32) + 1000,
            np.arange(512, dtype=np.float32) + 2000,
        ]
        input_queue = _GatedDisconnectInputQueue(chunks[0], [chunks[1], chunks[2], None], first.interrupted)
        output_queue = queue.SimpleQueue()
        transcriber = _transcriber(reconnect_attempts=1,
                                   connector=lambda *_args, **_kwargs: next(connections),
                                   sleeper=lambda _: None)

        transcriber.loop(input_queue, output_queue)

        first_audio = [value for value in first.sent if isinstance(value, bytes)]
        second_audio = [value for value in second.sent if isinstance(value, bytes)]
        self.assertEqual(first_audio, [chunks[0].astype('<f4').tobytes()])
        self.assertEqual(second_audio, [chunk.astype('<f4').tobytes() for chunk in chunks])
        self.assertEqual(second.sent[-1], '')
        self.assertEqual(output_queue.get_nowait().transcript, 'all chunks arrived.')
        self.assertIsNone(output_queue.get_nowait())

    def test_duplicate_final_uuid_across_reconnect_emits_only_one_task(self):
        utterance = {
            'type': 'utterance',
            'utterance': {
                'utterance_uuid': 'same-across-connections',
                'text': 'emit once.',
                'start_ms': 0,
                'duration_ms': 32,
            },
        }

        class FinalThenDisconnect(_FakeWebSocket):

            def recv(self):
                if not self.eof_sent.wait(timeout=2):
                    raise TimeoutError('test sender never sent EOF')
                if self.messages:
                    return json.dumps(self.messages.pop(0))
                raise ConnectionError('dropped after final acknowledgement')

        first = FinalThenDisconnect([utterance])
        second = _FakeWebSocket([utterance, {'type': 'done', 'duration_ms': 32}])
        connections = iter([first, second])
        transcriber = _transcriber(reconnect_attempts=1,
                                   connector=lambda *_args, **_kwargs: next(connections),
                                   sleeper=lambda _: None)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        self.assertEqual(output_queue.get_nowait().transcript, 'emit once.')
        self.assertIsNone(output_queue.get_nowait())
        with self.assertRaises(queue.Empty):
            output_queue.get_nowait()

    def test_nonretryable_websocket_close_codes_stop_immediately(self):
        for close_code in (4001, 4002, 4003, 4029, 1003):
            with self.subTest(close_code=close_code):
                attempts = 0

                def connector(*_args, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    return _FakeWebSocket(receive_error=_websocket_close(close_code), wait_for_eof=False)

                transcriber = _transcriber(reconnect_attempts=2, connector=connector, sleeper=lambda _: None)
                with self.assertRaises(ModulateStreamingError) as raised:
                    transcriber.loop(queue.SimpleQueue(), queue.SimpleQueue())

                self.assertEqual((attempts, raised.exception.retryable), (1, False))

    def test_retryable_websocket_close_codes_reconnect(self):
        for close_code in (1006, 1011):
            with self.subTest(close_code=close_code):
                attempts = 0
                recovered = _FakeWebSocket([{'type': 'done', 'duration_ms': 0}])

                def connector(*_args, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        return _FakeWebSocket(receive_error=_websocket_close(close_code), wait_for_eof=False)
                    return recovered

                delays = []
                transcriber = _transcriber(reconnect_attempts=1,
                                           connector=connector,
                                           sleeper=delays.append)
                input_queue = queue.SimpleQueue()
                input_queue.put(None)
                output_queue = queue.SimpleQueue()

                transcriber.loop(input_queue, output_queue)

                self.assertEqual(attempts, 2)
                self.assertEqual(delays, [0.0])
                self.assertIsNone(output_queue.get_nowait())

    def test_nonretryable_websocket_close_codes_from_sender_stop_immediately(self):
        for close_code in (4001, 4002, 4003, 4029, 1003, 1008):
            with self.subTest(close_code=close_code):
                attempts = 0

                def connector(*_args, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    return _SenderFailureWebSocket(_websocket_close(close_code))

                transcriber = _transcriber(reconnect_attempts=2, connector=connector, sleeper=lambda _: None)
                input_queue = queue.SimpleQueue()
                input_queue.put(np.zeros(512, dtype=np.float32))
                with self.assertRaises(ModulateStreamingError) as raised:
                    transcriber.loop(input_queue, queue.SimpleQueue())

                self.assertEqual((attempts, raised.exception.retryable), (1, False))

    def test_retryable_websocket_close_codes_from_sender_reconnect(self):
        for close_code in (1006, 1011, 1013):
            with self.subTest(close_code=close_code):
                attempts = 0
                recovered = _FakeWebSocket([{'type': 'done', 'duration_ms': 0}])

                def connector(*_args, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        return _SenderFailureWebSocket(_websocket_close(close_code))
                    return recovered

                delays = []
                transcriber = _transcriber(reconnect_attempts=1,
                                           connector=connector,
                                           sleeper=delays.append)
                input_queue = queue.SimpleQueue()
                input_queue.put(np.zeros(512, dtype=np.float32))
                input_queue.put(None)
                output_queue = queue.SimpleQueue()

                transcriber.loop(input_queue, output_queue)

                self.assertEqual(attempts, 2)
                self.assertEqual(delays, [0.0])
                self.assertIsNone(output_queue.get_nowait())

    def test_network_errors_reconnect(self):
        error_factories = (
            lambda: ConnectionError('connection reset'),
            lambda: TimeoutError('socket timed out'),
            lambda: OSError('network unavailable'),
        )
        for error_factory in error_factories:
            with self.subTest(error_type=type(error_factory()).__name__):
                attempts = 0
                recovered = _FakeWebSocket([{'type': 'done', 'duration_ms': 0}])

                def connector(*_args, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        return _FakeWebSocket(receive_error=error_factory(), wait_for_eof=False)
                    return recovered

                delays = []
                transcriber = _transcriber(reconnect_attempts=1,
                                           connector=connector,
                                           sleeper=delays.append)
                input_queue = queue.SimpleQueue()
                input_queue.put(None)
                output_queue = queue.SimpleQueue()

                transcriber.loop(input_queue, output_queue)

                self.assertEqual(attempts, 2)
                self.assertEqual(delays, [0.0])
                self.assertIsNone(output_queue.get_nowait())

    def test_receiver_interrupts_blocked_sender_before_reconnecting(self):
        blocked = _BlockingSendWebSocket()
        recovered = _FakeWebSocket([{'type': 'done', 'duration_ms': 32}])
        connections = iter([blocked, recovered])
        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            return next(connections)

        transcriber = _transcriber(reconnect_attempts=1,
                                   connector=connector,
                                   sleeper=lambda _: None)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        self.assertEqual(attempts, 2)
        self.assertGreaterEqual(blocked.socket.shutdown_calls, 1)
        self.assertTrue(blocked.socket.interrupted.is_set())
        self.assertIsNone(output_queue.get_nowait())

    def test_http_auth_handshake_failures_stop_immediately(self):
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                attempts = 0

                def connector(*_args, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    raise _handshake_status(status_code)

                transcriber = _transcriber(reconnect_attempts=2, connector=connector, sleeper=lambda _: None)
                with self.assertRaises(ModulateStreamingError) as raised:
                    transcriber.loop(queue.SimpleQueue(), queue.SimpleQueue())

                self.assertEqual((attempts, raised.exception.retryable), (1, False))

    def test_certificate_verification_failure_stops_immediately(self):
        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise ssl.SSLCertVerificationError(1, 'certificate verification failed')

        transcriber = _transcriber(reconnect_attempts=2, connector=connector, sleeper=lambda _: None)
        with self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(queue.SimpleQueue(), queue.SimpleQueue())

        self.assertEqual((attempts, raised.exception.retryable), (1, False))

    def test_retryable_http_handshake_failures_reconnect(self):
        for status_code in (429, 500, 503, 599):
            with self.subTest(status_code=status_code):
                attempts = 0
                recovered = _FakeWebSocket([{'type': 'done', 'duration_ms': 0}])

                def connector(*_args, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise _handshake_status(status_code)
                    return recovered

                delays = []
                transcriber = _transcriber(reconnect_attempts=1,
                                           connector=connector,
                                           sleeper=delays.append)
                input_queue = queue.SimpleQueue()
                input_queue.put(None)
                output_queue = queue.SimpleQueue()

                transcriber.loop(input_queue, output_queue)

                self.assertEqual(attempts, 2)
                self.assertEqual(delays, [0.0])
                self.assertIsNone(output_queue.get_nowait())

    def test_valid_partial_resets_consecutive_reconnect_budget(self):
        class PartialThenDisconnect(_FakeWebSocket):

            def __init__(self, text):
                super().__init__([{
                    'type': 'partial_utterance',
                    'partial_utterance': {'text': text, 'start_ms': 0},
                }], wait_for_eof=False)

            def recv(self):
                if self.messages:
                    return json.dumps(self.messages.pop(0))
                raise ConnectionError('connection dropped after a valid partial')

        connections = iter([
            PartialThenDisconnect('first'),
            PartialThenDisconnect('first second'),
            _FakeWebSocket([{'type': 'done', 'duration_ms': 0}]),
        ])
        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            return next(connections)

        delays = []
        transcriber = _transcriber(reconnect_attempts=1,
                                   connector=connector,
                                   sleeper=delays.append)
        input_queue = queue.SimpleQueue()
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [0.0, 0.0])
        self.assertEqual(output_queue.get_nowait().transcript, 'first second')
        self.assertIsNone(output_queue.get_nowait())

    def test_replayed_identical_partial_does_not_reset_reconnect_budget(self):
        class SamePartialThenDisconnect(_FakeWebSocket):

            def __init__(self):
                super().__init__([{
                    'type': 'partial_utterance',
                    'partial_utterance': {'text': 'unchanged preview', 'start_ms': 0},
                }], wait_for_eof=False)

            def recv(self):
                if self.messages:
                    return json.dumps(self.messages.pop(0))
                raise ConnectionError('connection dropped after replaying the same partial')

        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts > 3:
                raise ValueError('reconnect budget was reset by an unchanged partial')
            return SamePartialThenDisconnect()

        delays = []
        transcriber = _transcriber(reconnect_attempts=2,
                                   connector=connector,
                                   sleeper=delays.append)
        input_queue = queue.SimpleQueue()
        input_queue.put(None)

        with self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(input_queue, queue.SimpleQueue())

        self.assertTrue(raised.exception.retryable)
        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [0.0, 0.0])

    def test_replayed_partial_sequence_does_not_reset_reconnect_budget(self):
        class ReplayedSequenceThenDisconnect(_FakeWebSocket):

            def __init__(self):
                super().__init__([
                    {
                        'type': 'partial_utterance',
                        'partial_utterance': {'text': 'a', 'start_ms': 0},
                    },
                    {
                        'type': 'partial_utterance',
                        'partial_utterance': {'text': 'ab', 'start_ms': 0},
                    },
                ], wait_for_eof=False)

            def recv(self):
                if self.messages:
                    return json.dumps(self.messages.pop(0))
                raise ConnectionError('connection dropped after replaying the same sequence')

        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts > 3:
                raise ValueError('replayed rollback sequence reset the reconnect budget')
            return ReplayedSequenceThenDisconnect()

        delays = []
        transcriber = _transcriber(reconnect_attempts=2,
                                   connector=connector,
                                   sleeper=delays.append)
        input_queue = queue.SimpleQueue()
        input_queue.put(None)

        with self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(input_queue, queue.SimpleQueue())

        self.assertTrue(raised.exception.retryable)
        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [0.0, 0.0])

    def test_unexpected_disconnect_reconnects_and_replays_unfinalized_audio(self):
        first = _FakeWebSocket(receive_error=ConnectionError('network dropped'))
        second = _FakeWebSocket([
            {
                'type': 'utterance',
                'utterance': {
                    'utterance_uuid': 'after-reconnect',
                    'text': 'recovered.',
                    'start_ms': 0,
                    'duration_ms': 64,
                },
            },
            {'type': 'done', 'duration_ms': 64},
        ])
        connections = iter([first, second])
        delays = []
        transcriber = _transcriber(reconnect_attempts=2,
                                   connector=lambda *_args, **_kwargs: next(connections),
                                   sleeper=delays.append)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.arange(512, dtype=np.float32))
        input_queue.put(np.arange(512, dtype=np.float32) + 1000)
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        first_audio = [value for value in first.sent if isinstance(value, bytes)]
        replayed_audio = [value for value in second.sent if isinstance(value, bytes)]
        self.assertEqual(replayed_audio, first_audio)
        self.assertEqual(len(replayed_audio), 2)
        self.assertEqual(first.sent[-1], '')
        self.assertEqual(second.sent[-1], '')
        self.assertEqual(delays, [0.0])
        self.assertEqual(output_queue.get_nowait().transcript, 'recovered.')
        self.assertIsNone(output_queue.get_nowait())

    def test_handshake_failure_retries_with_exponential_backoff(self):
        websocket = _FakeWebSocket([{'type': 'done', 'duration_ms': 0}])
        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise OSError('temporary handshake failure')
            return websocket

        delays = []
        transcriber = _transcriber(reconnect_attempts=3,
                                   reconnect_backoff_seconds=0.5,
                                   connector=connector,
                                   sleeper=delays.append)
        input_queue = queue.SimpleQueue()
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [0.5, 1.0])
        self.assertIsNone(output_queue.get_nowait())

    def test_reconnect_replays_only_audio_after_last_finalized_utterance(self):
        class FinalThenDisconnect(_FakeWebSocket):

            def recv(self):
                if not self.eof_sent.wait(timeout=2):
                    raise TimeoutError('test sender never sent EOF')
                if self.messages:
                    return json.dumps(self.messages.pop(0))
                raise ConnectionError('dropped after first final')

        first = FinalThenDisconnect([{
            'type': 'utterance',
            'utterance': {
                'utterance_uuid': 'before-drop',
                'text': 'first.',
                'start_ms': 0,
                'duration_ms': 32,
            },
        }])
        second = _FakeWebSocket([
            {
                'type': 'utterance',
                'utterance': {
                    'utterance_uuid': 'after-drop',
                    'text': 'second.',
                    'start_ms': 0,
                    'duration_ms': 32,
                },
            },
            {'type': 'done', 'duration_ms': 32},
        ])
        connections = iter([first, second])
        transcriber = _transcriber(reconnect_attempts=1,
                                   connector=lambda *_args, **_kwargs: next(connections),
                                   sleeper=lambda _: None)
        input_queue = queue.SimpleQueue()
        input_queue.put(np.zeros(512, dtype=np.float32))
        input_queue.put(np.ones(512, dtype=np.float32))
        input_queue.put(None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(input_queue, output_queue)

        replayed_audio = [value for value in second.sent if isinstance(value, bytes)]
        self.assertEqual(len(replayed_audio), 1)
        np.testing.assert_array_equal(np.frombuffer(replayed_audio[0], dtype='<f4'), np.ones(512, dtype=np.float32))
        first_task = output_queue.get_nowait()
        second_task = output_queue.get_nowait()
        self.assertEqual(first_task.time_range, (0.0, 0.032))
        self.assertEqual(second_task.time_range, (0.032, 0.064))
        self.assertIsNone(output_queue.get_nowait())

    def test_retries_exhaust_without_leaking_key(self):
        key = 'never-log-this-key'
        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise OSError(f'failed URI containing {key}')

        transcriber = _transcriber(api_key=key,
                                   reconnect_attempts=2,
                                   reconnect_backoff_seconds=0,
                                   connector=connector,
                                   sleeper=lambda _: None)
        with self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(queue.SimpleQueue(), queue.SimpleQueue())
        self.assertEqual(attempts, 3)
        self.assertNotIn(key, str(raised.exception))


class ModulateConfigTest(unittest.TestCase):

    def test_cli_defaults_and_modulate_options(self):
        default = tyro.cli(Config, args=['input'])
        enabled = tyro.cli(Config, args=[
            'input', '--use-modulate-transcription-api', '--modulate-api-key', 'mod-key',
            '--modulate-partial-cutoff-seconds', '7.5', '--modulate-reconnect-attempts', '5',
            '--modulate-reconnect-backoff-seconds', '0.25',
        ])

        self.assertFalse(default.use_modulate_transcription_api)
        self.assertEqual(default.modulate_reconnect_attempts, 3)
        self.assertTrue(enabled.use_modulate_transcription_api)
        self.assertEqual(enabled.modulate_api_key, 'mod-key')
        self.assertEqual(enabled.modulate_partial_cutoff_seconds, 7.5)
        self.assertEqual(enabled.modulate_reconnect_attempts, 5)
        self.assertEqual(enabled.modulate_reconnect_backoff_seconds, 0.25)

    def test_validation_requires_key_and_valid_retry_values(self):
        invalid_configs = [
            Config(url='input', use_modulate_transcription_api=True),
            Config(url='input', use_modulate_transcription_api=True, modulate_api_key='key',
                   modulate_partial_cutoff_seconds=0),
            Config(url='input', use_modulate_transcription_api=True, modulate_api_key='key',
                   modulate_reconnect_attempts=-1),
            Config(url='input', use_modulate_transcription_api=True, modulate_api_key='key',
                   modulate_reconnect_backoff_seconds=float('nan')),
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(SystemExit):
                _validate_and_normalize(config)

    def test_modulate_is_mutually_exclusive_with_other_transcribers(self):
        config = Config(url='input',
                        use_modulate_transcription_api=True,
                        modulate_api_key='key',
                        use_openai_transcription_api=True,
                        whisper_api_key='whisper-key')
        with self.assertRaises(SystemExit):
            _validate_and_normalize(config)

    def test_run_bypasses_slicer_and_connects_getter_queue_directly(self):
        from stream_translator_gpt import main

        config = Config(url='input', use_modulate_transcription_api=True, modulate_api_key='key')
        fake_getter = mock.Mock()
        fake_transcriber = mock.Mock()
        fake_exporter = mock.Mock()
        thread = mock.Mock()
        thread.is_alive.return_value = False
        worker_calls = []

        def fake_start(func, *args, **kwargs):
            worker_calls.append((func, kwargs))
            return thread

        with mock.patch.object(main, 'ClientPool') as client_pool, \
                mock.patch.object(main, '_create_audio_getter', return_value=fake_getter), \
                mock.patch.object(main, 'ModulateStreamingTranscriber', return_value=fake_transcriber) as constructor, \
                mock.patch.object(main, 'ResultExporter', return_value=fake_exporter), \
                mock.patch.object(main, 'start_daemon_thread', side_effect=fake_start):
            main.run(config)

        client_pool.init.assert_called_once()
        constructor.assert_called_once()
        names = [kwargs.get('name') for _, kwargs in worker_calls]
        self.assertEqual(names, ['getter', 'transcriber', 'exporter'])
        getter_queue = worker_calls[0][1]['output_queue']
        transcriber_queue = worker_calls[1][1]['input_queue']
        self.assertIs(getter_queue, transcriber_queue)
        self.assertEqual(getter_queue.maxsize, 3750)


class AudioReplayStateTest(unittest.TestCase):

    def test_acknowledgement_trims_replay_at_exact_sample(self):
        state = _AudioStreamState(max_replay_seconds=10)
        state.add_audio(np.arange(10, dtype=np.float32))
        state.add_audio(np.arange(10, dtype=np.float32) + 10)
        state.acknowledge(12 / 16000)

        start, chunks, eof = state.replay_snapshot()

        self.assertEqual(start, 12)
        np.testing.assert_array_equal(np.frombuffer(chunks[0], dtype='<f4'), np.arange(12, 20, dtype=np.float32))
        self.assertFalse(eof)

    def test_replay_overflow_warning_is_rate_limited_until_acknowledged(self):
        state = _AudioStreamState(max_replay_seconds=2 / 16000)

        with self.assertLogs('main', level='WARNING') as captured:
            state.add_audio(np.arange(3, dtype=np.float32))
            state.add_audio(np.arange(3, 6, dtype=np.float32))
        self.assertEqual(len(captured.output), 1)

        state.acknowledge(state.seconds())
        with self.assertLogs('main', level='WARNING') as captured:
            state.add_audio(np.arange(6, 9, dtype=np.float32))
        self.assertEqual(len(captured.output), 1)


if __name__ == '__main__':
    unittest.main()
