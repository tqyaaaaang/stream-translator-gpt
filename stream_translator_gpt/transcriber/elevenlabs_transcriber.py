"""Scribe v2 Realtime with local VAD commits, isolated from the legacy backends.

Only one manual commit is outstanding at a time. Scribe has no commit IDs or
session-level EOF acknowledgement; this makes the two final event types
unambiguous, including empty transcripts and consecutive identical sentences.
Audio is journalled before sending and released only after a stable final.
"""

import base64
import json
import logging
import math
import queue
import socket
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import quote, quote_plus, urlencode

import numpy as np

from .audio_transcriber import AudioTranscriber
from ..utils.common import SAMPLE_RATE, SAMPLES_PER_FRAME, TranslationTask, sec2str, write_console_utf8


logger = logging.getLogger('main')
SCRIBE_ENDPOINT = 'wss://api.elevenlabs.io/v1/speech-to-text/realtime'
SCRIBE_STREAM_MAX_SEND_RATE = 5.0
CHUNK_SAMPLES = SAMPLES_PER_FRAME * 4
# The service reports this lower bound in commit_throttled responses. Allow a
# small margin for its audio-duration rounding. This isn't a network chunk size.
MIN_COMMIT_SAMPLES = round(0.32 * SAMPLE_RATE)
ERROR_EVENTS = frozenset({
    'error', 'auth_error', 'quota_exceeded', 'transcriber_error', 'input_error',
    'invalid_request', 'commit_throttled', 'unaccepted_terms', 'rate_limited',
    'queue_overflow', 'resource_exhausted', 'session_time_limit_exceeded',
    'chunk_size_exceeded', 'insufficient_audio_activity', 'throttled',
})


class ElevenLabsScribeError(RuntimeError):
    def __init__(self, message, *, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def normalize_keyterms(value: str | None) -> list[str]:
    """Never construct the paid query parameter for an empty keyword list."""
    terms = [term.strip() for term in (value or '').split(',') if term.strip()]
    if len(terms) > 50:
        logger.warning('ElevenLabs Scribe received %d keyterms; using the first 50. Full supplied list: %s',
                       len(terms), terms)
        terms = terms[:50]
    too_long = [term for term in terms if len(term) > 20]
    if too_long:
        raise ValueError(f'ElevenLabs Scribe keyterms must contain at most 20 characters each: {too_long!r}')
    return terms


class _SendPacer:
    """Use absolute deadlines, compensating at most one packet of lateness."""

    def __init__(self, rate, stop_event, clock):
        self.rate = rate
        self.stop_event = stop_event
        self.clock = clock
        self.next_send = clock()
        self.last_interval = 0.0

    def wait(self):
        while not self.stop_event.is_set():
            now = self.clock()
            # Small scheduler delays must not accumulate on every packet.
            # A long idle or blocked send starts a fresh schedule instead.
            if now - self.next_send > self.last_interval:
                self.next_send = now
            delay = self.next_send - now
            if delay <= 0:
                return True
            if self.stop_event.wait(delay):
                break
        return False

    def record(self, samples):
        self.last_interval = samples / SAMPLE_RATE / self.rate
        self.next_send += self.last_interval


@dataclass
class _Packet:
    start: int
    pcm: bytes
    commit: bool

    @property
    def end(self):
        return self.start + len(self.pcm) // 2


@dataclass
class _Commit:
    start: int
    end: int
    sent_at: float
    text: str | None = None
    timestamps: dict | None = None
    acknowledged: bool = False


class ElevenLabsScribeTranscriber(AudioTranscriber):
    def __init__(self, api_key, proxy=None, verify_ssl=True, max_audio_send_rate: float | None = None,
                 min_audio_length=0.5, max_audio_length=30.0, target_audio_length=5.0,
                 continuous_no_speech_threshold=1.0, dynamic_no_speech_threshold=True,
                 prefix_retention_length=0.5, vad_threshold=0.35, dynamic_vad_threshold=True,
                 reconnect_attempts=3, reconnect_backoff_seconds=1.0,
                 connector=None, boundary_detector=None, clock=None,
                 final_timeout=30.0, idle_timeout=10.0, **kwargs):
        self.keyterms = normalize_keyterms(kwargs.get('transcription_keywords'))
        super().__init__(**kwargs)
        self.api_key = (api_key or '').strip()
        if not self.api_key:
            raise ValueError('ElevenLabs API key is required')
        if (max_audio_send_rate is not None
                and (not math.isfinite(max_audio_send_rate) or max_audio_send_rate <= 0)):
            raise ValueError('ElevenLabs maximum audio send rate must be finite and greater than zero')
        for name, value in (('final timeout', final_timeout), ('idle timeout', idle_timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'ElevenLabs {name} must be finite and greater than zero')
        if reconnect_attempts < 0 or not math.isfinite(reconnect_backoff_seconds) or reconnect_backoff_seconds < 0:
            raise ValueError('ElevenLabs reconnect settings must be finite and non-negative')
        self.proxy = proxy
        self.verify_ssl = verify_ssl
        self.max_audio_send_rate = max_audio_send_rate
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.final_timeout = final_timeout
        self.idle_timeout = idle_timeout
        self.connector = connector
        self.clock = clock or time.monotonic
        if boundary_detector is None:
            # Loading this module must not introduce a VAD dependency into the
            # existing Modulate or CLI-help paths.
            from ..slicer.manual_commit_slicer import ManualCommitBoundaryDetector
            boundary_detector = ManualCommitBoundaryDetector(
                min_audio_length, max_audio_length, target_audio_length,
                continuous_no_speech_threshold, dynamic_no_speech_threshold,
                prefix_retention_length, vad_threshold, dynamic_vad_threshold)
        self.detector = boundary_detector
        self._stop_event = threading.Event()
        self._connection_lock = threading.Lock()
        self._websocket = None
        self._journal = deque()
        self._buffer = bytearray()
        self._frames = deque()
        self._source_eof = False
        self._next_sample = 0
        self._ack_sample = 0
        self._last_boundary = 0
        self._boundary_requested = False
        self._task_count = 0
        self._last_task_end = 0.0
        self._pacer = (_SendPacer(max_audio_send_rate, self._stop_event, self.clock)
                       if max_audio_send_rate is not None else None)
        if prefix_retention_length:
            logger.info('ElevenLabs sends continuous audio; --prefix-retention-length does not replay audio')
        if self.use_history_context:
            logger.info('ElevenLabs maintains context within its session; --transcription-context '
                        'does not inject additional text prompts')

    def transcribe(self, audio, history_context=None):
        raise NotImplementedError('ElevenLabs Scribe operates on a continuous stream')

    def _connection_url(self):
        query = [('model_id', 'scribe_v2_realtime'), ('audio_format', 'pcm_16000'),
                 ('commit_strategy', 'manual'), ('include_timestamps', 'true')]
        if self.language and self.language.lower() != 'auto':
            query.append(('language_code', self.language))
        query.extend(('keyterms', term) for term in self.keyterms)
        return SCRIBE_ENDPOINT + '?' + urlencode(query)

    def _connection_kwargs(self):
        kwargs = dict(additional_headers={'xi-api-key': self.api_key}, compression=None,
                      open_timeout=15, close_timeout=5, ping_interval=20, ping_timeout=20,
                      max_size=4 * 1024 * 1024)
        if self.proxy:
            kwargs['proxy'] = self.proxy
        if not self.verify_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            kwargs['ssl'] = context
        return kwargs

    def _safe_error(self, value):
        message = str(value or 'unknown server error')
        secrets = {self.api_key, quote(self.api_key, safe=''), quote_plus(self.api_key)}
        secrets |= {quote(value, safe='') for value in secrets}
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                message = message.replace(secret, '<redacted>')
        return message[:500]

    @staticmethod
    def _interrupt(websocket):
        raw_socket = getattr(websocket, 'socket', None)
        if raw_socket is not None:
            try:
                raw_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                raw_socket.close()
            except OSError:
                pass
        else:
            try:
                websocket.close()
            except Exception:
                pass

    def stop(self):
        with self._connection_lock:
            self._stop_event.set()
            websocket = self._websocket
            self._websocket = None
        if websocket is not None:
            self._interrupt(websocket)

    def _connect(self):
        """A cancelled handshake may finish later, but can never send audio."""
        connector = self.connector
        if connector is None:
            from websockets.sync.client import connect
            connector = connect
        results = queue.SimpleQueue()

        def open_connection():
            try:
                websocket = connector(self._connection_url(), **self._connection_kwargs())
                with self._connection_lock:
                    cancelled = self._stop_event.is_set()
                    if not cancelled:
                        self._websocket = websocket
                if cancelled:
                    self._interrupt(websocket)
                results.put((websocket, None))
            except Exception as exc:
                results.put((None, exc))

        threading.Thread(target=open_connection, name='scribe-connect', daemon=True).start()
        while not self._stop_event.is_set():
            try:
                websocket, error = results.get(timeout=0.05)
            except queue.Empty:
                continue
            if error is not None:
                raise error
            return websocket
        return None

    @staticmethod
    def _transport_retryable(exc):
        from websockets.exceptions import ConnectionClosed, InvalidStatus
        if isinstance(exc, ssl.SSLCertVerificationError):
            return False
        if isinstance(exc, InvalidStatus):
            status = getattr(exc.response, 'status_code', 0)
            return status >= 500 or status in (408, 425, 429)
        if isinstance(exc, ConnectionClosed):
            code = getattr(exc.rcvd or exc.sent, 'code', None)
            return code not in (1002, 1003, 1007, 1008, 1009)
        return isinstance(exc, (ConnectionError, TimeoutError, OSError))

    def _receive(self, websocket, timeout):
        try:
            raw = websocket.recv(timeout=timeout)
        except TimeoutError:
            return None
        if not isinstance(raw, str):
            raise ElevenLabsScribeError('ElevenLabs returned a non-text protocol message')
        try:
            event = json.loads(raw)
        except (ValueError, TypeError):
            raise ElevenLabsScribeError('ElevenLabs returned malformed JSON') from None
        if not isinstance(event, dict):
            raise ElevenLabsScribeError('ElevenLabs returned an invalid event')
        kind = event.get('message_type', '')
        if kind in ERROR_EVENTS or kind.endswith('_error'):
            message = self._safe_error(event.get('error') or event.get('message') or kind)
            raise ElevenLabsScribeError(f'ElevenLabs {kind}: {message}',
                                       retryable=kind in ('session_time_limit_exceeded', 'transcriber_error',
                                                          'resource_exhausted'))
        return event

    def _await_session(self, websocket):
        deadline = self.clock() + 15
        while not self._stop_event.is_set():
            if self.clock() >= deadline:
                raise ElevenLabsScribeError('Timed out waiting for ElevenLabs session_started', retryable=True)
            event = self._receive(websocket, 0.05)
            if event and event.get('message_type') == 'session_started':
                return

    def _packet(self, commit=False):
        packet = _Packet(self._next_sample, bytes(self._buffer), commit)
        self._buffer.clear()
        self._next_sample = packet.end
        if commit:
            self._last_boundary = packet.end
            self._boundary_requested = False
            self.detector.mark_boundary()
        self._journal.append(packet)
        return packet

    def _next_packet(self, input_queue):
        """Keep decoded but unsent audio across reconnects; VAD runs only once."""
        while not self._stop_event.is_set():
            if not self._frames:
                if self._source_eof:
                    if self._buffer:
                        return self._packet(commit=True)
                    if self._next_sample > self._last_boundary:
                        return self._packet(commit=True)
                    return None
                try:
                    audio = input_queue.get(timeout=0.05)
                except queue.Empty:
                    return self._packet() if self._buffer else None
                if audio is None:
                    self._source_eof = True
                    continue
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if not np.all(np.isfinite(audio)):
                    raise ElevenLabsScribeError('Non-finite input audio for ElevenLabs')
                self._frames.extend(audio[i:i + SAMPLES_PER_FRAME]
                                    for i in range(0, len(audio), SAMPLES_PER_FRAME))
                if not self._frames:
                    continue
            frame = self._frames.popleft()
            self._boundary_requested |= self.detector.feed(frame)
            pcm = (np.clip(frame, -1, 32767 / 32768) * 32768).astype('<i2').tobytes()
            self._buffer.extend(pcm)
            end = self._next_sample + len(self._buffer) // 2
            commit = self._boundary_requested and end - self._last_boundary >= MIN_COMMIT_SAMPLES
            if commit or len(self._buffer) >= CHUNK_SAMPLES * 2:
                return self._packet(commit=commit)
        return None

    def _send(self, websocket, pcm, commit):
        if self._stop_event.is_set():
            return False
        if self._pacer is not None and not self._pacer.wait():
            return False
        websocket.send(json.dumps(dict(message_type='input_audio_chunk',
                                       audio_base_64=base64.b64encode(pcm).decode('ascii'),
                                       sample_rate=SAMPLE_RATE, commit=commit)))
        if self._pacer is not None:
            self._pacer.record(len(pcm) // 2)
        return True

    def _emit_commit(self, pending, connection_start, output_queue):
        text = self.filter_text(pending.text or '').strip()
        start, end = pending.start / SAMPLE_RATE, pending.end / SAMPLE_RATE
        words = (pending.timestamps or {}).get('words') or []
        times = [(word.get('start'), word.get('end')) for word in words if isinstance(word, dict)]
        times = [(a, b) for a, b in times if isinstance(a, (int, float)) and isinstance(b, (int, float))
                 and math.isfinite(a) and math.isfinite(b) and 0 <= a <= b]
        if times:
            start = max(start, min(a for a, _ in times) + connection_start / SAMPLE_RATE)
            end = min(end, max(b for _, b in times) + connection_start / SAMPLE_RATE)
        elif text:
            logger.info('ElevenLabs word timestamps unavailable; using local commit audio range %.3f–%.3f',
                        start, end)
        start = max(min(start, pending.end / SAMPLE_RATE), self._last_task_end)
        end = max(end, start)
        # A stable result acknowledges source audio even if filters remove text.
        self._ack_sample = pending.end
        pending.acknowledged = True
        while self._journal and self._journal[0].end <= self._ack_sample:
            self._journal.popleft()
        if not text:
            return
        self._task_count += 1
        task = TranslationTask(self._task_count, np.empty(0, dtype=np.float32), (start, end))
        task.transcript = text
        task.phase = TranslationTask.Phase.TRANSCRIBED
        self._last_task_end = end
        if self.print_result:
            prefix = f'{sec2str(start)} --> {sec2str(end)} ' if self.output_timestamps else ''
            write_console_utf8(prefix + text)
        output_queue.put(task)

    @staticmethod
    def _accept_final(pending, event):
        """Both event orders are valid; words:null is a completed timestamp event."""
        kind = event.get('message_type')
        if kind == 'committed_transcript':
            text = event.get('text')
            if not isinstance(text, str):
                raise ElevenLabsScribeError('ElevenLabs committed transcript has invalid text')
            pending.text = text
        elif kind == 'committed_transcript_with_timestamps':
            if not isinstance(event.get('text'), str) or not isinstance(event.get('words'), (list, type(None))):
                raise ElevenLabsScribeError('ElevenLabs timestamp event has invalid text or words')
            pending.timestamps = event
        if pending.text is not None and pending.timestamps is not None:
            if pending.timestamps.get('text') != pending.text:
                raise ElevenLabsScribeError('ElevenLabs final text and timestamp events disagree')
            return True
        return False

    def _preserve_final_on_disconnect(self, exc, pending, connection_start, output_queue):
        # Don't replay a stable final when only its companion event was lost.
        # Protocol errors and local output failures must never publish fallback.
        interrupted = (exc.retryable if isinstance(exc, ElevenLabsScribeError)
                       else self._transport_retryable(exc))
        if (interrupted and pending is not None and not pending.acknowledged
                and not self._stop_event.is_set()
                and (pending.text is not None or pending.timestamps is not None)):
            if pending.text is None:
                pending.text = pending.timestamps['text']
            self._emit_commit(pending, connection_start, output_queue)

    def _finish_commit(self, websocket, pending, connection_start, output_queue):
        try:
            while not self._stop_event.is_set():
                if self.clock() - pending.sent_at > self.final_timeout:
                    raise ElevenLabsScribeError('Timed out waiting for ElevenLabs committed transcript', retryable=True)
                event = self._receive(websocket, 0.05)
                if event is not None and self._accept_final(pending, event):
                    break
        except Exception as exc:
            self._preserve_final_on_disconnect(exc, pending, connection_start, output_queue)
            raise
        if not self._stop_event.is_set():
            self._emit_commit(pending, connection_start, output_queue)

    def _run_connection(self, websocket, input_queue, output_queue):
        self._await_session(websocket)
        connection_start = self._ack_sample
        replay = deque(self._journal)
        sent_end = connection_start
        committed_end = connection_start
        last_input_at = self.clock()
        pending = None
        try:
            while not self._stop_event.is_set():
                # Stream the next sentence while its predecessor is finalized.
                # Only the next commit, not every audio packet, must wait for
                # both finals. This avoids accumulating final latency at 1x
                # while keeping one unambiguous commit outstanding.
                for _ in range(32):
                    event = self._receive(websocket, 0)
                    if event is None:
                        break
                    kind = event.get('message_type')
                    if kind in ('committed_transcript', 'committed_transcript_with_timestamps'):
                        if pending is None:
                            raise ElevenLabsScribeError('Unexpected ElevenLabs commit without a local audio boundary')
                        if self._accept_final(pending, event):
                            self._emit_commit(pending, connection_start, output_queue)
                            pending = None
                if pending is not None and self.clock() - pending.sent_at > self.final_timeout:
                    raise ElevenLabsScribeError('Timed out waiting for ElevenLabs committed transcript', retryable=True)
                idle_commit = False
                packet = replay.popleft() if replay else self._next_packet(input_queue)
                if packet is None:
                    if self._source_eof:
                        if pending is not None:
                            self._finish_commit(websocket, pending, connection_start, output_queue)
                        return
                    if self.clock() - last_input_at < self.idle_timeout:
                        continue
                    # Don't manufacture endless silence for a stalled source.
                    # Flush before the service's 15s no-client-message timeout.
                    if sent_end > committed_end:
                        packet = self._packet(commit=True)
                        idle_commit = True
                    else:
                        if pending is not None:
                            self._finish_commit(websocket, pending, connection_start, output_queue)
                        return
                last_input_at = self.clock()
                pcm = packet.pcm
                if packet.commit:
                    if pending is not None:
                        self._finish_commit(websocket, pending, connection_start, output_queue)
                        pending = None
                    # An EOF/idle short tail needs padding. End this connection
                    # afterwards so the next session's source mapping is exact.
                    missing = max(MIN_COMMIT_SAMPLES - (packet.end - committed_end), 0)
                    if missing:
                        pcm += bytes(missing * 2)
                        logger.debug('Padding final Scribe commit with %.3fs of silence', missing / SAMPLE_RATE)
                if not self._send(websocket, pcm, packet.commit):
                    return
                sent_end = packet.end
                if packet.commit:
                    logger.info('Scribe local commit %s to %s', sec2str(committed_end / SAMPLE_RATE),
                                sec2str(sent_end / SAMPLE_RATE))
                    pending = _Commit(committed_end, sent_end, self.clock())
                    committed_end = sent_end
                    if (self._source_eof and not replay) or missing or idle_commit:
                        self._finish_commit(websocket, pending, connection_start, output_queue)
                        return
        except Exception as exc:
            self._preserve_final_on_disconnect(exc, pending, connection_start, output_queue)
            raise

    def loop(self, input_queue, output_queue):
        failures = 0
        try:
            while not self._stop_event.is_set():
                # Don't open/bill an empty session, or reconnect repeatedly while
                # a live source is paused. All consumed audio is journalled.
                while not self._journal and not self._source_eof and not self._stop_event.is_set():
                    self._next_packet(input_queue)
                if self._stop_event.is_set() or (self._source_eof and not self._journal):
                    break
                websocket = None
                acknowledged_before = self._ack_sample
                try:
                    websocket = self._connect()
                    if websocket is None or self._stop_event.is_set():
                        break
                    self._run_connection(websocket, input_queue, output_queue)
                    failures = 0
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    retryable = exc.retryable if isinstance(exc, ElevenLabsScribeError) else self._transport_retryable(exc)
                    if not retryable:
                        if isinstance(exc, ElevenLabsScribeError):
                            raise
                        raise ElevenLabsScribeError(f'ElevenLabs connection failed ({type(exc).__name__})') from None
                    if self._ack_sample > acknowledged_before:
                        failures = 0
                    if failures >= self.reconnect_attempts:
                        raise ElevenLabsScribeError('ElevenLabs reconnection attempts exhausted') from None
                    delay = min(self.reconnect_backoff_seconds * 2 ** failures, 30.0)
                    failures += 1
                    logger.warning('ElevenLabs transport interrupted (%s); reconnecting in %.1fs (%d/%d)',
                                   type(exc).__name__, delay, failures, self.reconnect_attempts)
                    if self._stop_event.wait(delay):
                        break
                finally:
                    if websocket is not None:
                        try:
                            websocket.close()
                        except Exception:
                            self._interrupt(websocket)
                        finally:
                            with self._connection_lock:
                                if self._websocket is websocket:
                                    self._websocket = None
        except Exception:
            # The pipeline worker wrapper owns error EOF, as for Modulate.
            raise
        else:
            output_queue.put(None)
