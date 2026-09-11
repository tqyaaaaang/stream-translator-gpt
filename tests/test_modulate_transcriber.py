import json
import logging
import math
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

from stream_translator_gpt.utils.common import SAMPLE_RATE, SAMPLES_PER_FRAME
from stream_translator_gpt.main import Config, _validate_and_normalize
from stream_translator_gpt.transcriber.modulate_transcriber import (MODULATE_INPUT_QUEUE_SECONDS, MODULATE_REPLAY_SECONDS,
                                                        MODULATE_STREAM_MAX_SEND_RATE, ModulateStreamingError,
                                                        ModulateStreamingTranscriber, _AudioSendPacer,
                                                        _AudioSendProgress, _AudioStreamState)


def _transcriber(**overrides):
    kwargs = {
        'api_key': 'key with +?/ characters',
        'partial_cutoff_seconds': 10.0,
        'reconnect_attempts': 0,
        'reconnect_backoff_seconds': 0.0,
        'proxy': None,
        'websocket_compression': False,
        'language': None,
        'transcription_filters': '',
        'language_filters': None,
        'print_result': False,
        'output_timestamps': False,
        'use_history_context': False,
        'transcription_keywords': None,
    }
    kwargs.update(overrides)
    return ModulateStreamingTranscriber(**kwargs)


def _audio_input_queue(*chunks, eof=True):
    input_queue = queue.SimpleQueue()
    if not chunks:
        chunks = (np.zeros(512, dtype=np.float32), )
    for chunk in chunks:
        input_queue.put(chunk)
    if eof:
        input_queue.put(None)
    return input_queue


def _websocket_close(code: int) -> ConnectionClosedError:
    if code == 1006:
        # websockets represents an abnormal closure as the absence of both
        # close frames; ConnectionClosed.code resolves this case to 1006.
        return ConnectionClosedError(None, None)
    return ConnectionClosedError(Close(code, f'test close {code}'), None)


def _handshake_status(status_code: int) -> InvalidStatus:
    return InvalidStatus(Response(status_code, 'test response', Headers()))


class _FakeClock:

    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _FakeStopEvent:

    def __init__(self, clock: _FakeClock):
        self.clock = clock
        self.wait_calls = []
        self.stopped = False
        self.interrupt_next_wait = False

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.interrupt_next_wait:
            self.stopped = True
            return True
        if timeout is not None:
            self.clock.advance(timeout)
        return self.stopped

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True


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


class AudioSendPacerTest(unittest.TestCase):

    def test_fake_clock_limits_consecutive_audio_to_one_point_five_times_realtime(self):
        clock = _FakeClock()
        stop_event = _FakeStopEvent(clock)
        pacer = _AudioSendPacer(MODULATE_STREAM_MAX_SEND_RATE, stop_event, clock=clock)

        self.assertTrue(pacer.wait_until_ready())
        pacer.record_sent(SAMPLE_RATE)
        self.assertTrue(pacer.wait_until_ready())

        self.assertAlmostEqual(clock.now, 1 / MODULATE_STREAM_MAX_SEND_RATE)
        self.assertEqual(len(stop_event.wait_calls), 1)
        self.assertAlmostEqual(stop_event.wait_calls[0], 1 / MODULATE_STREAM_MAX_SEND_RATE)

    def test_idle_time_does_not_accumulate_send_credit(self):
        clock = _FakeClock()
        stop_event = _FakeStopEvent(clock)
        pacer = _AudioSendPacer(MODULATE_STREAM_MAX_SEND_RATE, stop_event, clock=clock)

        pacer.record_sent(SAMPLE_RATE)
        clock.advance(10.0)
        self.assertTrue(pacer.wait_until_ready())
        self.assertEqual(stop_event.wait_calls, [])

        pacer.record_sent(SAMPLE_RATE)
        self.assertTrue(pacer.wait_until_ready())
        self.assertEqual(len(stop_event.wait_calls), 1)
        self.assertAlmostEqual(stop_event.wait_calls[0], 1 / MODULATE_STREAM_MAX_SEND_RATE)

    def test_stop_event_interrupts_pacing_wait(self):
        clock = _FakeClock()
        stop_event = _FakeStopEvent(clock)
        pacer = _AudioSendPacer(MODULATE_STREAM_MAX_SEND_RATE, stop_event, clock=clock)
        pacer.record_sent(SAMPLE_RATE)
        stop_event.interrupt_next_wait = True

        self.assertFalse(pacer.wait_until_ready())
        self.assertEqual(len(stop_event.wait_calls), 1)
        self.assertAlmostEqual(stop_event.wait_calls[0], 1 / MODULATE_STREAM_MAX_SEND_RATE)
        self.assertEqual(clock.now, 0.0)

    def test_replay_and_queued_live_audio_share_one_pacer(self):
        clock = _FakeClock()
        stop_event = _FakeStopEvent(clock)
        replay = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        live = np.ones(SAMPLES_PER_FRAME, dtype=np.float32)
        stream_state = _AudioStreamState()
        replay_payload, _ = stream_state.add_audio(replay)
        input_queue = queue.SimpleQueue()
        input_queue.put(live)
        input_queue.put(None)
        sent_at = []

        class TimestampingWebSocket:

            def send(self, value):
                sent_at.append((value, clock()))

        transcriber = _transcriber(max_audio_send_rate=MODULATE_STREAM_MAX_SEND_RATE, clock=clock)
        errors = queue.SimpleQueue()
        replay_payloads = [replay_payload]
        transcriber._send_audio(TimestampingWebSocket(), input_queue, stream_state, replay_payloads, False,
                                stop_event, errors, _AudioSendProgress())

        binary_sends = [(payload, timestamp) for payload, timestamp in sent_at if isinstance(payload, bytes)]
        self.assertEqual([payload for payload, _ in binary_sends],
                         [replay.astype('<f4').tobytes(), live.astype('<f4').tobytes()])
        self.assertAlmostEqual(binary_sends[0][1], 0.0)
        self.assertAlmostEqual(binary_sends[1][1],
                               SAMPLES_PER_FRAME / SAMPLE_RATE / MODULATE_STREAM_MAX_SEND_RATE)
        self.assertEqual(sent_at[-1][0], '')
        self.assertEqual(replay_payloads, [])
        with self.assertRaises(queue.Empty):
            errors.get_nowait()

    def test_live_audio_dequeued_before_interrupted_wait_remains_replayable(self):
        clock = _FakeClock()
        stop_event = _FakeStopEvent(clock)
        stop_event.interrupt_next_wait = True
        first = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        queued_live = np.ones(SAMPLES_PER_FRAME, dtype=np.float32)
        stream_state = _AudioStreamState()
        first_payload, _ = stream_state.add_audio(first)
        input_queue = queue.SimpleQueue()
        input_queue.put(queued_live)
        sent = []

        class RecordingWebSocket:

            def send(self, value):
                sent.append(value)

        transcriber = _transcriber(max_audio_send_rate=MODULATE_STREAM_MAX_SEND_RATE, clock=clock)
        transcriber._send_audio(RecordingWebSocket(), input_queue, stream_state, [first_payload], False,
                                stop_event, queue.SimpleQueue(), _AudioSendProgress())

        _, replay_payloads, _ = stream_state.replay_snapshot()
        self.assertEqual([value for value in sent if isinstance(value, bytes)], [first_payload])
        self.assertEqual(replay_payloads,
                         [first.astype('<f4').tobytes(), queued_live.astype('<f4').tobytes()])


class ModulateProtocolTest(unittest.TestCase):

    def test_transcript_print_uses_shared_utf8_writer(self):
        with mock.patch('stream_translator_gpt.transcriber.modulate_transcriber.write_console_utf8') as writer:
            ModulateStreamingTranscriber._print_transcript('テスト。')
        writer.assert_called_once_with('テスト。')

    def test_filtered_empty_segment_does_not_emit_empty_transcription_warning(self):
        transcriber = _transcriber(transcription_filters='emoji_filter')
        output_queue = queue.SimpleQueue()

        with self.assertLogs('main', level='DEBUG') as captured:
            transcriber._emit_task('😀', 0.0, 0.032, output_queue)

        self.assertIn('DEBUG:main:Skipping Modulate transcription segment removed by filters', captured.output)
        self.assertNotIn('WARNING:main:Skipping empty Modulate transcription segment', captured.output)
        with self.assertRaises(queue.Empty):
            output_queue.get_nowait()

    def test_source_empty_segment_keeps_empty_transcription_warning(self):
        transcriber = _transcriber()

        with self.assertLogs('main', level='WARNING') as captured:
            transcriber._emit_task('   ', 0.0, 0.032, queue.SimpleQueue())

        self.assertEqual(captured.output, ['WARNING:main:Skipping empty Modulate transcription segment'])

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
            'speaker_diarization': False,
            'language': 'ja',
            'custom_terms': ['Modulate', 'Velma'],
        })

    def test_custom_terms_debug_log_matches_sent_japanese_array_without_credentials(self):
        websocket = _FakeWebSocket([{'type': 'done', 'duration_ms': 32}])
        transcriber = _transcriber(
            connector=lambda *_args, **_kwargs: websocket,
            transcription_keywords=' 結城さくな, 神楽めあ, クリオネ , ,',
        )

        with self.assertLogs('main', level='DEBUG') as captured:
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

        records = [record for record in captured.records
                   if record.getMessage().startswith('Modulate custom terms (')]
        self.assertEqual(len(records), 1)
        record = records[0]
        terms = json.loads(websocket.sent[0])['custom_terms']
        self.assertEqual(terms, ['結城さくな', '神楽めあ', 'クリオネ'])
        self.assertEqual(record.levelno, logging.DEBUG)
        self.assertEqual(record.args[0], len(terms))
        self.assertEqual(json.loads(record.args[1]), terms)
        self.assertIn('結城さくな', record.getMessage())
        self.assertNotIn('\\u', record.getMessage())
        self.assertNotIn(transcriber.api_key, '\n'.join(captured.output))
        self.assertNotIn('api_key', record.getMessage())
        self.assertNotIn('speaker_diarization', record.getMessage())

    def test_custom_terms_debug_log_is_emitted_for_each_reconnection(self):
        first = _FakeWebSocket(receive_error=_websocket_close(1006))
        second = _FakeWebSocket([{'type': 'done', 'duration_ms': 32}])
        connector = mock.Mock(side_effect=[first, second])
        transcriber = _transcriber(
            connector=connector, reconnect_attempts=1, sleeper=lambda _: None,
            transcription_keywords='結城さくな,神楽めあ',
        )

        with self.assertLogs('main', level='DEBUG') as captured:
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

        records = [record for record in captured.records
                   if record.getMessage().startswith('Modulate custom terms (')]
        self.assertEqual(connector.call_count, 2)
        self.assertEqual(len(records), 2)
        for record, websocket in zip(records, (first, second)):
            terms = json.loads(websocket.sent[0])['custom_terms']
            self.assertEqual(record.args[0], len(terms))
            self.assertEqual(json.loads(record.args[1]), terms)

    def test_empty_custom_terms_debug_log_reports_zero_and_empty_array(self):
        websocket = _FakeWebSocket([{'type': 'done', 'duration_ms': 32}])
        transcriber = _transcriber(connector=lambda *_args, **_kwargs: websocket)

        with self.assertLogs('main', level='DEBUG') as captured:
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

        messages = [record.getMessage() for record in captured.records
                    if record.getMessage().startswith('Modulate custom terms (')]
        self.assertEqual(messages, ['Modulate custom terms (0): []'])
        self.assertNotIn('custom_terms', json.loads(websocket.sent[0]))

    def test_custom_terms_are_not_logged_at_info_level(self):
        websocket = _FakeWebSocket([{'type': 'done', 'duration_ms': 32}])
        transcriber = _transcriber(
            connector=lambda *_args, **_kwargs: websocket,
            transcription_keywords='結城さくな,神楽めあ',
        )

        with self.assertLogs('main', level='INFO') as captured:
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

        self.assertEqual(json.loads(websocket.sent[0])['custom_terms'], ['結城さくな', '神楽めあ'])
        logs = '\n'.join(captured.output)
        self.assertNotIn('Modulate custom terms', logs)
        self.assertNotIn('結城さくな', logs)
        self.assertNotIn(transcriber.api_key, logs)

    def test_auto_language_is_omitted_for_server_side_detection(self):
        transcriber = _transcriber(language='auto')
        self.assertEqual(transcriber._connection_config(), {
            'partial_results': True,
            'speaker_diarization': False,
        })

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

        with self.assertLogs('main', level='DEBUG') as captured:
            transcriber.loop(input_queue, output_queue)

        config = json.loads(websocket.sent[0])
        self.assertTrue(config['partial_results'])
        self.assertIsInstance(websocket.sent[1], bytes)
        binary_messages = [message for message in websocket.sent if isinstance(message, bytes)]
        self.assertEqual(len(binary_messages), 2)
        np.testing.assert_array_equal(np.frombuffer(binary_messages[0], dtype='<f4'), first.astype(np.float32))
        self.assertEqual(websocket.sent[-1], '')
        task = output_queue.get_nowait()
        self.assertEqual(task.transcript, 'hello.')
        self.assertEqual(task.time_range, (0.0, 0.064))
        self.assertIsNone(output_queue.get_nowait())
        logs = '\n'.join(captured.output)
        self.assertIn('Sent first Modulate audio payload; source=replay', logs)
        self.assertIn('Received first Modulate server message; type=utterance', logs)
        self.assertIn('sent_audio_seconds=', logs)

    def test_connection_waits_for_initial_audio_then_sends_config_before_buffered_payload(self):
        first_audio = np.arange(512, dtype=np.float32)
        get_started = threading.Event()
        release_audio = threading.Event()
        connector_called = threading.Event()

        class DelayedInitialAudioQueue:

            def __init__(self):
                self.calls = 0
                self.lock = threading.Lock()

            def get(self, timeout=None):
                with self.lock:
                    call = self.calls
                    self.calls += 1
                if call == 0:
                    get_started.set()
                    if not release_audio.wait(timeout=2):
                        raise TimeoutError('test did not release initial audio')
                    return first_audio
                return None

        websocket = _FakeWebSocket([{'type': 'done', 'duration_ms': 32}])

        def connector(*_args, **_kwargs):
            connector_called.set()
            return websocket

        transcriber = _transcriber(connector=connector)
        output_queue = queue.SimpleQueue()
        errors = []

        def run_transcriber():
            try:
                transcriber.loop(DelayedInitialAudioQueue(), output_queue)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_transcriber)
        worker.start()
        self.assertTrue(get_started.wait(timeout=1))
        self.assertFalse(connector_called.is_set())

        release_audio.set()
        worker.join(timeout=3)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(connector_called.is_set())
        self.assertIsInstance(websocket.sent[0], str)
        self.assertEqual(json.loads(websocket.sent[0])['partial_results'], True)
        self.assertEqual(websocket.sent[1], first_audio.astype('<f4').tobytes())
        self.assertEqual(websocket.sent[2], '')
        self.assertIsNone(output_queue.get_nowait())

    def test_empty_upstream_fails_before_connecting(self):
        for upstream_values in ([None], [np.empty(0, dtype=np.float32), np.empty((0, 2)), None]):
            with self.subTest(upstream_values=len(upstream_values)):
                connector = mock.Mock()
                transcriber = _transcriber(connector=connector)
                input_queue = queue.SimpleQueue()
                for value in upstream_values:
                    input_queue.put(value)

                with self.assertRaisesRegex(ModulateStreamingError,
                                            'Audio source ended before producing any samples'):
                    transcriber.loop(input_queue, queue.SimpleQueue())

                connector.assert_not_called()

    def test_server_error_log_reports_sent_audio_without_counting_as_progress(self):
        websocket = _FakeWebSocket([{'type': 'error', 'error': 'Invalid input audio'}])
        transcriber = _transcriber(connector=lambda *_args, **_kwargs: websocket)

        with self.assertLogs('main', level='DEBUG') as captured, \
                self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

        self.assertFalse(raised.exception.made_progress)
        logs = '\n'.join(captured.output)
        self.assertIn('Received first Modulate server message; type=error', logs)
        self.assertIn('Modulate server reported an error; sent_audio_seconds=0.032', logs)

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
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

        self.assertEqual(len(calls), 1)
        self.assertFalse(raised.exception.made_progress)
        self.assertNotIn(key, str(raised.exception))
        self.assertIn('<redacted>', str(raised.exception))

    def test_connection_disables_websocket_compression_by_default(self):
        kwargs = _transcriber()._connection_kwargs()

        self.assertIn('compression', kwargs)
        self.assertIsNone(kwargs['compression'])

    def test_connection_enables_deflate_when_requested(self):
        kwargs = _transcriber(websocket_compression=True)._connection_kwargs()

        self.assertEqual(kwargs['compression'], 'deflate')

    def test_transport_diagnostics_expose_only_numeric_status_and_close_codes(self):
        transcriber = _transcriber()
        secret = 'must-not-appear'
        handshake = InvalidStatus(Response(429, secret, Headers({'X-Secret': secret})))
        closed = ConnectionClosedError(Close(4029, secret), None)

        self.assertEqual(transcriber._transport_error_label(handshake), 'InvalidStatus; status_code=429')
        self.assertEqual(transcriber._transport_error_label(closed), 'ConnectionClosedError; close_code=4029')
        self.assertEqual(transcriber._transport_error_label(OSError(secret)), 'OSError')
        for label in (transcriber._transport_error_label(handshake),
                      transcriber._transport_error_label(closed),
                      transcriber._transport_error_label(OSError(secret))):
            self.assertNotIn(secret, label)

    def test_transport_diagnostics_ignore_non_integer_values_without_stringifying_them(self):
        class DoNotStringify:

            def __str__(self):
                raise AssertionError('diagnostic value must not be stringified')

        exc = OSError('safe')
        exc.response = mock.Mock(status_code=DoNotStringify())
        exc.rcvd = mock.Mock(code=DoNotStringify())

        self.assertEqual(_transcriber()._transport_error_label(exc), 'OSError')

    def test_invalid_audio_server_error_is_not_retried(self):
        for chunks in ([np.zeros(512, dtype=np.float32)],
                       [np.zeros(512, dtype=np.float32) for _ in range(40)]):
            with self.subTest(audio_chunks=len(chunks)):
                websocket = _FakeWebSocket([{'type': 'error', 'error': 'Invalid input audio'}])
                connector = mock.Mock(return_value=websocket)
                transcriber = _transcriber(reconnect_attempts=3, connector=connector, sleeper=lambda _delay: None)

                with self.assertRaises(ModulateStreamingError) as raised:
                    transcriber.loop(_audio_input_queue(*chunks), queue.SimpleQueue())

                self.assertFalse(raised.exception.retryable)
                connector.assert_called_once()

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
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

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

    def test_disconnect_after_eof_and_full_ack_does_not_open_empty_retry(self):
        class FullyAcknowledgedThenDisconnect(_FakeWebSocket):

            def __init__(self):
                super().__init__([{
                    'type': 'utterance',
                    'utterance': {
                        'utterance_uuid': 'fully-acknowledged',
                        'text': 'complete.',
                        'start_ms': 0,
                        'duration_ms': 32,
                    },
                }])

            def recv(self):
                if not self.eof_sent.wait(timeout=2):
                    raise TimeoutError('test sender never sent EOF')
                if self.messages:
                    return json.dumps(self.messages.pop(0))
                raise ConnectionError('connection dropped after acknowledging all audio')

        websocket = FullyAcknowledgedThenDisconnect()
        connector_calls = 0

        def connector(*_args, **_kwargs):
            nonlocal connector_calls
            connector_calls += 1
            if connector_calls > 1:
                raise AssertionError('must not open a config-only retry after all audio was acknowledged')
            return websocket

        transcriber = _transcriber(reconnect_attempts=1, connector=connector, sleeper=lambda _: None)
        output_queue = queue.SimpleQueue()

        transcriber.loop(_audio_input_queue(), output_queue)

        self.assertEqual(connector_calls, 1)
        self.assertEqual(output_queue.get_nowait().transcript, 'complete.')
        self.assertIsNone(output_queue.get_nowait())

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
                    transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

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
                input_queue = _audio_input_queue()
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
                input_queue = _audio_input_queue()
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
                    transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

                self.assertEqual((attempts, raised.exception.retryable), (1, False))

    def test_certificate_verification_failure_stops_immediately(self):
        attempts = 0

        def connector(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise ssl.SSLCertVerificationError(1, 'certificate verification failed')

        transcriber = _transcriber(reconnect_attempts=2, connector=connector, sleeper=lambda _: None)
        with self.assertRaises(ModulateStreamingError) as raised:
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())

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
                input_queue = _audio_input_queue()
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
        input_queue = _audio_input_queue()
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
        input_queue = _audio_input_queue()

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
        input_queue = _audio_input_queue()

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
        input_queue = _audio_input_queue()
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
            transcriber.loop(_audio_input_queue(), queue.SimpleQueue())
        self.assertEqual(attempts, 3)
        self.assertNotIn(key, str(raised.exception))


class ModulateConfigTest(unittest.TestCase):

    def test_cli_defaults_and_modulate_options(self):
        default = tyro.cli(Config, args=['input'])
        enabled = tyro.cli(Config, args=[
            'input', '--use-modulate-transcription-api', '--modulate-api-key', 'mod-key',
            '--modulate-partial-cutoff-seconds', '7.5', '--modulate-reconnect-attempts', '5',
            '--modulate-reconnect-backoff-seconds', '0.25', '--modulate-websocket-compression',
        ])

        self.assertFalse(default.use_modulate_transcription_api)
        self.assertEqual(default.modulate_reconnect_attempts, 3)
        self.assertFalse(default.modulate_websocket_compression)
        self.assertTrue(enabled.use_modulate_transcription_api)
        self.assertEqual(enabled.modulate_api_key, 'mod-key')
        self.assertEqual(enabled.modulate_partial_cutoff_seconds, 7.5)
        self.assertEqual(enabled.modulate_reconnect_attempts, 5)
        self.assertEqual(enabled.modulate_reconnect_backoff_seconds, 0.25)
        self.assertTrue(enabled.modulate_websocket_compression)

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

    def test_local_file_guidance_uses_unified_pacer_option(self):
        from stream_translator_gpt import main

        common = {
            'url': 'input.mp3',
            'use_modulate_transcription_api': True,
            'modulate_api_key': 'key',
        }
        with mock.patch.object(main.logger, 'warning') as warning:
            _validate_and_normalize(Config(**common, pacer='unlimited'))
        warning.assert_called_once()
        self.assertIn('--pacer realtime', warning.call_args.args[0])

        with mock.patch.object(main.logger, 'warning') as warning:
            _validate_and_normalize(Config(**common, pacer='realtime'))
        warning.assert_not_called()

    def test_run_bypasses_slicer_and_connects_getter_queue_directly(self):
        from stream_translator_gpt import main

        config = Config(url='https://example.test/live', use_modulate_transcription_api=True, modulate_api_key='key',
                        modulate_websocket_compression=True)
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
                mock.patch.object(main.signal, 'signal') as register_signal, \
                mock.patch.object(main, 'start_daemon_thread', side_effect=fake_start):
            main.run(config)

        client_pool.init.assert_called_once()
        constructor.assert_called_once()
        self.assertTrue(constructor.call_args.kwargs['websocket_compression'])
        self.assertEqual(constructor.call_args.kwargs['max_audio_send_rate'], MODULATE_STREAM_MAX_SEND_RATE)
        names = [kwargs.get('name') for _, kwargs in worker_calls]
        self.assertEqual(names, ['getter', 'transcriber', 'exporter'])
        getter_queue = worker_calls[0][1]['output_queue']
        transcriber_queue = worker_calls[1][1]['input_queue']
        self.assertIs(getter_queue, transcriber_queue)
        expected_chunks = math.ceil(MODULATE_INPUT_QUEUE_SECONDS * SAMPLE_RATE / SAMPLES_PER_FRAME)
        self.assertEqual(expected_chunks, 18750)
        self.assertEqual(getter_queue.maxsize, expected_chunks)
        expected_signals = [mock.call(main.signal.SIGINT, fake_getter._exit_handler)]
        if hasattr(main.signal, 'SIGTERM'):
            expected_signals.append(mock.call(main.signal.SIGTERM, fake_getter._exit_handler))
        register_signal.assert_has_calls(expected_signals)


class AudioReplayStateTest(unittest.TestCase):

    def test_default_replay_capacity_is_ten_minutes(self):
        state = _AudioStreamState()

        self.assertEqual(MODULATE_REPLAY_SECONDS, 600)
        self.assertEqual(state._max_replay_samples, 600 * SAMPLE_RATE)

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
