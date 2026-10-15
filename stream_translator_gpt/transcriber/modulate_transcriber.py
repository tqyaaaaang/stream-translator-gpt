import difflib
import json
import logging
import math
import queue
import socket
import ssl
import threading
import time
from collections import deque
from urllib.parse import quote, quote_plus, urlencode

import numpy as np

from .audio_transcriber import AudioTranscriber
from ..utils.common import SAMPLE_RATE, TranslationTask, sec2str, write_console_utf8


logger = logging.getLogger('main')

MODULATE_STREAMING_ENDPOINT = 'wss://platform.modulate.ai/api/velma-2-stt-streaming'
MODULATE_REPLAY_SECONDS = 600
MODULATE_INPUT_QUEUE_SECONDS = 600
MODULATE_STREAM_MAX_SEND_RATE = 1.5
NON_RETRYABLE_CLOSE_CODES = frozenset({
    1002,  # protocol error
    1003,  # unsupported / invalid configuration
    1007,  # invalid payload
    1008,  # policy violation
    1009,  # message too large
    4001,  # invalid API key
    4002,  # audio decoding / format error
    4003,  # authentication or permission failure
    4029,  # credits or concurrency limit
})

# Clause and sentence delimiters in ASCII, full-width CJK, and half-width Japanese forms.
# Apostrophes and hyphens are intentionally excluded because they commonly occur inside words.
SPLIT_PUNCTUATION = frozenset('.,!?;:．，！？；：。、｡､…⋯؟،؛।॥‼⁇⁈⁉')
TRAILING_PUNCTUATION = SPLIT_PUNCTUATION | frozenset('"\'”’」』】）》〕〉)]}＂＇')
NUMERIC_SEPARATORS = frozenset('.,:．，：')


class ModulateStreamingError(RuntimeError):
    """A display-safe failure raised by the Modulate streaming backend."""

    def __init__(self, message: str, retryable: bool = False, made_progress: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.made_progress = made_progress


def _map_text_offset(old_text: str, new_text: str, old_offset: int) -> int:
    """Map a boundary through a replacement-style partial transcript update.

    Replacements that overlap an already committed boundary consume no more
    replacement characters than the corresponding old span. This preserves a
    newly added suffix for the authoritative final result while still avoiding
    an exact replay of the published prefix.
    """
    old_offset = max(0, min(old_offset, len(old_text)))
    if old_offset == 0 or not old_text:
        return 0
    if old_text == new_text or new_text.startswith(old_text):
        return old_offset

    matcher = difflib.SequenceMatcher(a=old_text, b=new_text, autojunk=False)
    mapped = 0
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if old_offset <= old_start:
            return new_start
        if tag == 'insert':
            mapped = new_end
            continue
        if old_offset <= old_end:
            if tag == 'equal':
                return new_start + (old_offset - old_start)
            if tag == 'delete':
                return new_start
            # The old boundary intersects revised text. Preserve any newly
            # added suffix instead of consuming the whole replacement: text
            # beyond the corresponding old character count hasn't previously
            # been emitted and must remain available to the final result.
            return min(new_start + (old_offset - old_start), new_end)
        mapped = new_end
    return min(mapped, len(new_text))


class PartialUtteranceSplitter:
    """Turn replacement-style partial utterances into bounded committed pieces."""

    def __init__(self, cutoff_seconds: float, clock=None):
        if not math.isfinite(cutoff_seconds) or cutoff_seconds <= 0:
            raise ValueError('Modulate partial cutoff seconds must be a finite value greater than zero')
        self.cutoff_seconds = float(cutoff_seconds)
        self.clock = clock or time.monotonic
        self.reset()

    def reset(self):
        self.preview_text = ''
        self.committed_offset = 0
        self.committed_source_text = ''
        self.started_at = None
        self.punctuation_search_offset = None
        self.prefix_conflicted = False

    def _replace_preview(self, new_text: str):
        old_text = self.preview_text
        old_committed_offset = self.committed_offset
        old_committed_text = old_text[:old_committed_offset]

        if self.committed_source_text and new_text.startswith(self.committed_source_text):
            self.committed_offset = len(self.committed_source_text)
            if self.prefix_conflicted:
                logger.info('Modulate partial transcript realigned with its committed prefix')
            self.prefix_conflicted = False
        else:
            self.committed_offset = _map_text_offset(old_text, new_text, old_committed_offset)

        if (self.committed_source_text and not new_text.startswith(self.committed_source_text)
                and old_committed_text and new_text[:self.committed_offset] != old_committed_text):
            if not self.prefix_conflicted:
                logger.info('Modulate revised an already committed partial-transcript prefix; '
                               'keeping the revised prefix consumed to avoid duplicate output')
            self.prefix_conflicted = True
        self.preview_text = new_text

    @staticmethod
    def _boundary_end(text: str, start: int, stop: int | None = None) -> int | None:
        for index in range(max(start, 0), min(stop if stop is not None else len(text), len(text))):
            if text[index] not in SPLIT_PUNCTUATION:
                continue
            if (text[index] in NUMERIC_SEPARATORS and index > 0 and index + 1 < len(text)
                    and text[index - 1].isdigit() and text[index + 1].isdigit()):
                continue
            end = index + 1
            while end < len(text) and text[end] in TRAILING_PUNCTUATION:
                end += 1
            return end
        return None

    @classmethod
    def _new_boundary_end(cls, old_text: str, new_text: str, minimum: int) -> int | None:
        """Find punctuation introduced by this replacement-style update."""
        matcher = difflib.SequenceMatcher(a=old_text, b=new_text, autojunk=False)
        for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
            if tag in {'equal', 'delete'} or new_end <= minimum:
                continue
            boundary_end = cls._boundary_end(new_text, max(new_start, minimum), stop=new_end)
            if boundary_end is not None:
                return boundary_end
        return None

    def update_partial(self, text: str | None) -> str | None:
        """Replace the preview and return at most one newly forced segment."""
        text = text or ''
        previous_text = self.preview_text
        self._replace_preview(text)

        uncommitted_text = text[self.committed_offset:]
        if not uncommitted_text.strip():
            self.started_at = None
            self.punctuation_search_offset = None
            return None

        if self.prefix_conflicted:
            self.started_at = None
            self.punctuation_search_offset = None
            return None

        now = self.clock()
        if self.started_at is None:
            self.started_at = now
            return None

        if self.punctuation_search_offset is None:
            # The cutoff is strict: equality isn't overdue yet. Once overdue,
            # only punctuation introduced by this or a later replacement-style
            # update is eligible; existing punctuation isn't used retroactively.
            if now - self.started_at <= self.cutoff_seconds:
                return None
            self.punctuation_search_offset = self.committed_offset

        boundary_end = self._new_boundary_end(previous_text, text, self.committed_offset)
        if boundary_end is None:
            return None

        segment = text[self.committed_offset:boundary_end].strip()
        self.committed_offset = boundary_end
        self.committed_source_text = text[:boundary_end]
        self.started_at = now if text[boundary_end:].strip() else None
        self.punctuation_search_offset = None
        return segment or None

    def finalize(self, text: str | None) -> str | None:
        """Return only the uncommitted final remainder, then reset state."""
        self._replace_preview(text or '')
        remainder = self.preview_text[self.committed_offset:].strip()
        self.reset()
        return remainder or None


class _AudioStreamState:

    def __init__(self, max_replay_seconds: float = MODULATE_REPLAY_SECONDS):
        self._chunks = deque()
        self._next_sample = 0
        self._eof = False
        self._max_replay_samples = max(round(max_replay_seconds * SAMPLE_RATE), 1)
        self._overflow_warned = False
        self._lock = threading.Lock()

    def add_audio(self, audio) -> tuple[bytes, int]:
        samples = np.asarray(audio, dtype='<f4').reshape(-1)
        payload = samples.tobytes()
        sample_count = samples.size
        if sample_count == 0:
            return payload, 0

        with self._lock:
            start = self._next_sample
            self._chunks.append((start, payload, sample_count))
            self._next_sample += sample_count
            replay_floor = self._next_sample - self._max_replay_samples
            trimmed = False
            while self._chunks and self._chunks[0][0] + self._chunks[0][2] <= replay_floor:
                self._chunks.popleft()
                trimmed = True
            if self._chunks and self._chunks[0][0] < replay_floor:
                chunk_start, chunk_payload, chunk_count = self._chunks.popleft()
                offset = replay_floor - chunk_start
                self._chunks.appendleft((replay_floor, chunk_payload[offset * 4:], chunk_count - offset))
                trimmed = True
            should_warn = trimmed and not self._overflow_warned
            if trimmed:
                self._overflow_warned = True
        if should_warn:
            logger.warning('Modulate reconnect replay buffer reached %d seconds; dropping its oldest audio',
                           round(self._max_replay_samples / SAMPLE_RATE))
        return payload, sample_count

    def acknowledge(self, absolute_seconds: float):
        target = max(round(absolute_seconds * SAMPLE_RATE), 0)
        with self._lock:
            target = min(target, self._next_sample)
            while self._chunks and self._chunks[0][0] + self._chunks[0][2] <= target:
                self._chunks.popleft()
            if self._chunks and self._chunks[0][0] < target:
                chunk_start, chunk_payload, chunk_count = self._chunks.popleft()
                offset = target - chunk_start
                self._chunks.appendleft((target, chunk_payload[offset * 4:], chunk_count - offset))
            retained_start = self._chunks[0][0] if self._chunks else self._next_sample
            if self._next_sample - retained_start < self._max_replay_samples:
                self._overflow_warned = False

    def replay_snapshot(self) -> tuple[int, list[bytes], bool]:
        with self._lock:
            start = self._chunks[0][0] if self._chunks else self._next_sample
            return start, [payload for _, payload, _ in self._chunks], self._eof

    def mark_eof(self):
        with self._lock:
            self._eof = True

    def seconds(self) -> float:
        with self._lock:
            return self._next_sample / SAMPLE_RATE


class _AudioSendProgress:
    """Track successfully queued WebSocket audio without retaining its contents."""

    def __init__(self):
        self._samples_sent = 0
        self._lock = threading.Lock()

    def record_payload(self, payload: bytes) -> tuple[bool, int, float]:
        sample_count = len(payload) // np.dtype(np.float32).itemsize
        with self._lock:
            is_first = self._samples_sent == 0
            self._samples_sent += sample_count
            seconds_sent = self._samples_sent / SAMPLE_RATE
        return is_first, sample_count, seconds_sent

    def seconds(self) -> float:
        with self._lock:
            return self._samples_sent / SAMPLE_RATE


class _AudioSendPacer:
    """Limit ready-to-send audio without accumulating burst credit while idle."""

    def __init__(self, max_rate: float, stop_event: threading.Event, clock=None):
        if not math.isfinite(max_rate) or max_rate <= 0:
            raise ValueError('Audio send rate must be a finite value greater than zero')
        self.max_rate = float(max_rate)
        self.stop_event = stop_event
        self.clock = clock or time.monotonic
        self.next_allowed_at = None

    def wait_until_ready(self) -> bool:
        while self.next_allowed_at is not None:
            delay = self.next_allowed_at - self.clock()
            if delay <= 0:
                break
            if self.stop_event.wait(delay):
                return False
        return not self.stop_event.is_set()

    def record_sent(self, sample_count: int) -> None:
        now = self.clock()
        self.next_allowed_at = now + sample_count / SAMPLE_RATE / self.max_rate


class ModulateStreamingTranscriber(AudioTranscriber):
    """Stream raw getter audio directly to Modulate without VAD slicing."""

    def __init__(self,
                 api_key: str,
                 partial_cutoff_seconds: float,
                 proxy: str | None,
                 verify_ssl: bool = True,
                 reconnect_attempts: int = 3,
                 reconnect_backoff_seconds: float = 1.0,
                 connector=None,
                 clock=None,
                 sleeper=None,
                 websocket_compression: bool = False,
                 max_audio_send_rate: float | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        api_key = (api_key or '').strip()
        if not api_key:
            raise ValueError('Modulate API key is required')
        self.api_key = api_key
        self.proxy = proxy
        self.verify_ssl = verify_ssl
        self.websocket_compression = bool(websocket_compression)
        if (max_audio_send_rate is not None
                and (not math.isfinite(max_audio_send_rate) or max_audio_send_rate <= 0)):
            raise ValueError('Modulate maximum audio send rate must be a finite value greater than zero')
        self.max_audio_send_rate = (float(max_audio_send_rate) if max_audio_send_rate is not None else None)
        self.clock = clock or time.monotonic
        if reconnect_attempts < 0:
            raise ValueError('Modulate reconnect attempts must be zero or greater')
        if not math.isfinite(reconnect_backoff_seconds) or reconnect_backoff_seconds < 0:
            raise ValueError('Modulate reconnect backoff seconds must be a finite value of zero or greater')
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_backoff_seconds = float(reconnect_backoff_seconds)
        self.partial_splitter = PartialUtteranceSplitter(partial_cutoff_seconds, clock=clock)
        self.connector = connector
        # The default backoff must wake immediately when the pipeline stops.
        # Keep the injected sleeper for deterministic retry tests.
        self.sleeper = sleeper
        self._stop_event = threading.Event()
        self._connection_lock = threading.Lock()
        self._active_websocket = None
        self._sender_stop_event = None
        self.task_count = 0
        self.last_task_end = 0.0
        self.active_segment_start = None
        self.active_partial_start_ms = None
        self.active_was_forced = False
        self._seen_final_uuids = set()
        self._seen_final_uuid_order = deque(maxlen=1024)
        self._partial_progress_seen = set()
        self._partial_progress_order = deque(maxlen=4096)

    def stop(self):
        """Cancel this transcriber's I/O without waiting for a server response.

        State publication and cancellation share a lock, but socket shutdown
        never holds it (nor the WebSocket send mutex). A connection that finishes
        opening after cancellation is interrupted by its registering thread.
        """
        with self._connection_lock:
            if self._stop_event.is_set():
                return
            self._stop_event.set()
            websocket = self._active_websocket
            self._active_websocket = None
            sender_stop_event = self._sender_stop_event
        if sender_stop_event is not None:
            sender_stop_event.set()
        if websocket is not None:
            self._interrupt_websocket(websocket)

    def _register_websocket(self, websocket) -> bool:
        with self._connection_lock:
            cancelled = self._stop_event.is_set()
            if not cancelled:
                self._active_websocket = websocket
        if cancelled:
            self._interrupt_websocket(websocket)
        return not cancelled

    def transcribe(self, audio: np.ndarray, history_context: str | None = None) -> tuple[str, list | None]:
        raise NotImplementedError('Modulate transcription operates on a continuous stream')

    def _connection_url(self) -> str:
        query = urlencode({
            'api_key': self.api_key,
            'partial_results': 'true',
            'audio_format': 'f32le',
            'sample_rate': str(SAMPLE_RATE),
            'num_channels': '1',
        })
        return f'{MODULATE_STREAMING_ENDPOINT}?{query}'

    def _connection_config(self) -> dict:
        # The application doesn't consume speaker labels, so avoid requesting
        # diarization work whose output would otherwise be discarded.
        config = {'partial_results': True, 'speaker_diarization': False}
        if self.language and self.language.lower() != 'auto':
            config['language'] = self.language
        custom_terms = [term.strip() for term in self.transcription_keywords.split(',') if term.strip()]
        if custom_terms:
            config['custom_terms'] = custom_terms
        return config

    def _connection_kwargs(self) -> dict:
        kwargs = {
            'open_timeout': 15,
            'close_timeout': 10,
            'ping_interval': 20,
            'ping_timeout': 20,
            'max_size': 1024 * 1024,
            'compression': 'deflate' if self.websocket_compression else None,
        }
        if self.proxy:
            kwargs['proxy'] = self.proxy
        if not self.verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            kwargs['ssl'] = ssl_context
        return kwargs

    def _safe_server_error(self, value) -> str:
        message = str(value or 'unknown server error')
        encoded_keys = {
            self.api_key,
            quote(self.api_key, safe=''),
            quote_plus(self.api_key, safe=''),
        }
        encoded_keys |= {quote(key, safe='') for key in encoded_keys}
        for key in sorted(encoded_keys, key=len, reverse=True):
            if key:
                message = message.replace(key, '<redacted>')
        return message[:500]

    @staticmethod
    def _print_transcript(text: str):
        write_console_utf8(text)

    def _emit_task(self, text: str, start: float, end: float, output_queue):
        had_source_text = bool(text.strip())
        text = self.filter_text(text).strip()
        if not text:
            if had_source_text:
                logger.debug('Skipping Modulate transcription segment removed by filters')
            else:
                logger.warning('Skipping empty Modulate transcription segment')
            return

        start = max(float(start), self.last_task_end)
        end = max(float(end), start)
        self.task_count += 1
        task = TranslationTask(self.task_count, np.empty(0, dtype=np.float32), (start, end))
        task.transcript = text
        task.phase = TranslationTask.Phase.TRANSCRIBED
        self.last_task_end = end

        if self.print_result:
            if self.output_timestamps:
                self._print_transcript(f'{sec2str(start)} --> {sec2str(end)} {text}')
            else:
                self._print_transcript(text)
        output_queue.put(task)

    def _handle_partial(self, partial: dict, stream_state: _AudioStreamState, connection_offset: float,
                        output_queue):
        text = partial.get('text') or ''
        start_ms = partial.get('start_ms')
        if text.strip() and self.active_segment_start is None:
            self.active_partial_start_ms = start_ms
            self.active_segment_start = ((connection_offset + float(start_ms) / 1000)
                                         if start_ms is not None else self.last_task_end)
        elif (text.strip() and not self.active_was_forced and start_ms is not None
              and self.active_partial_start_ms is None):
            self.active_partial_start_ms = start_ms
            self.active_segment_start = connection_offset + float(start_ms) / 1000

        segment = self.partial_splitter.update_partial(text)
        if segment is None:
            return

        start = self.active_segment_start if self.active_segment_start is not None else self.last_task_end
        end = max(stream_state.seconds(), start)
        logger.info('Force-splitting overdue Modulate partial result at punctuation')
        self._emit_task(segment, start, end, output_queue)
        self.active_segment_start = end
        self.active_was_forced = True

    def _record_partial_progress(self, partial: dict) -> bool:
        """Record a hypothesis once so reconnect replay cannot refresh the retry budget."""
        text = partial.get('text') or ''
        if not text.strip():
            return False
        fingerprint = text
        if fingerprint in self._partial_progress_seen:
            return False
        if len(self._partial_progress_order) == self._partial_progress_order.maxlen:
            self._partial_progress_seen.discard(self._partial_progress_order[0])
        self._partial_progress_order.append(fingerprint)
        self._partial_progress_seen.add(fingerprint)
        return True

    def _reset_partial_progress(self):
        self._partial_progress_seen.clear()
        self._partial_progress_order.clear()

    def _handle_final(self, utterance: dict, stream_state: _AudioStreamState, connection_offset: float,
                      output_queue) -> bool:
        utterance_uuid = utterance.get('utterance_uuid')
        if utterance_uuid and utterance_uuid in self._seen_final_uuids:
            logger.debug('Ignoring duplicate Modulate final utterance UUID')
            return False
        if utterance_uuid:
            if len(self._seen_final_uuid_order) == self._seen_final_uuid_order.maxlen:
                self._seen_final_uuids.discard(self._seen_final_uuid_order[0])
            self._seen_final_uuid_order.append(utterance_uuid)
            self._seen_final_uuids.add(utterance_uuid)

        official_start = max(connection_offset + float(utterance.get('start_ms') or 0) / 1000, 0.0)
        duration = max(float(utterance.get('duration_ms') or 0) / 1000, 0.0)
        official_end = official_start + duration
        stream_seconds = stream_state.seconds()
        logger.debug('Received Modulate final utterance metadata; official_end_seconds=%.3f, '
                     'stream_audio_seconds=%.3f, stream_minus_final_seconds=%.3f', official_end, stream_seconds,
                     stream_seconds - official_end)
        stream_state.acknowledge(official_end)
        remainder = self.partial_splitter.finalize(utterance.get('text') or '')

        if remainder:
            start = self.active_segment_start if self.active_was_forced else official_start
            if start is None:
                start = official_start
            self._emit_task(remainder, start, max(official_end, start), output_queue)

        self.active_segment_start = None
        self.active_partial_start_ms = None
        self.active_was_forced = False
        return True

    def _receive_results(self, websocket, stream_state: _AudioStreamState, connection_offset: float,
                         output_queue, progress_event: threading.Event,
                         send_progress: _AudioSendProgress, response_started_at: float) -> bool:
        first_message = True
        while not self._stop_event.is_set():
            message = websocket.recv()
            if self._stop_event.is_set():
                return False
            if not isinstance(message, str):
                raise ModulateStreamingError('Modulate returned an unexpected binary message')
            try:
                payload = json.loads(message)
            except (TypeError, json.JSONDecodeError):
                raise ModulateStreamingError('Modulate returned malformed JSON') from None

            message_type = payload.get('type')
            if first_message:
                logger.debug('Received first Modulate server message; type=%s, sent_audio_seconds=%.3f, '
                             'stream_audio_seconds=%.3f, response_elapsed_seconds=%.3f', message_type,
                             send_progress.seconds(), stream_state.seconds(), time.monotonic() - response_started_at)
                first_message = False
            if message_type == 'partial_utterance':
                partial = payload.get('partial_utterance') or {}
                self._handle_partial(partial, stream_state, connection_offset, output_queue)
                if self._record_partial_progress(partial):
                    progress_event.set()
            elif message_type == 'utterance':
                if self._handle_final(payload.get('utterance') or {}, stream_state, connection_offset, output_queue):
                    self._reset_partial_progress()
                    progress_event.set()
            elif message_type == 'done':
                progress_event.set()
                pending = self.partial_splitter.finalize(self.partial_splitter.preview_text)
                if pending:
                    start = self.active_segment_start if self.active_segment_start is not None else self.last_task_end
                    self._emit_task(pending, start, max(stream_state.seconds(), start), output_queue)
                self.active_segment_start = None
                self.active_partial_start_ms = None
                self.active_was_forced = False
                self._reset_partial_progress()
                logger.info('Modulate finished %.3f seconds of streamed audio',
                            float(payload.get('duration_ms') or 0) / 1000)
                return True
            elif message_type == 'error':
                sent_audio_seconds = send_progress.seconds()
                safe_error = self._safe_server_error(payload.get('error'))
                logger.debug('Modulate server reported an error; sent_audio_seconds=%.3f, '
                             'stream_audio_seconds=%.3f', sent_audio_seconds, stream_state.seconds())
                raise ModulateStreamingError(f'Modulate API error: {safe_error}')
            else:
                logger.debug('Ignoring unknown Modulate message type: %s', message_type)
        return False

    @staticmethod
    def _transport_diagnostic(exc: BaseException) -> str:
        """Return numeric-only WebSocket diagnostics safe for logs/errors."""
        details = []
        status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
        if isinstance(status_code, int) and not isinstance(status_code, bool):
            details.append(f'status_code={status_code}')
        received_close = getattr(exc, 'rcvd', None)
        sent_close = getattr(exc, 'sent', None)
        close_code = getattr(received_close or sent_close, 'code', None)
        if isinstance(close_code, int) and not isinstance(close_code, bool):
            details.append(f'close_code={close_code}')
        return ', '.join(details)

    @classmethod
    def _transport_error_label(cls, exc: BaseException) -> str:
        diagnostic = cls._transport_diagnostic(exc)
        return type(exc).__name__ + (f'; {diagnostic}' if diagnostic else '')

    @classmethod
    def _queue_error(cls, error_queue: queue.SimpleQueue, exc: BaseException, retryable: bool):
        error_queue.put((cls._transport_error_label(exc), retryable))

    @staticmethod
    def _interrupt_websocket(websocket):
        """Abort socket I/O without waiting for the WebSocket send mutex."""
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
            return
        try:
            websocket.close()
        except Exception:
            pass

    @staticmethod
    def _transport_retryability(exc: BaseException) -> bool | None:
        """Return whether a transport failure is retryable, or None if it isn't transport-related."""
        try:
            from websockets.exceptions import ConnectionClosed, InvalidProxyStatus, InvalidStatus
            websocket_exceptions = (ConnectionClosed, )
            status_exceptions = (InvalidStatus, InvalidProxyStatus)
        except ImportError:
            websocket_exceptions = ()
            status_exceptions = ()

        if isinstance(exc, status_exceptions):
            status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status_code is None:
                return False
            # Retry server failures, timeouts, early-data rejection, and rate
            # limiting. Other 4xx responses indicate a key/configuration error.
            return status_code >= 500 or status_code in {408, 425, 429}
        if isinstance(exc, websocket_exceptions):
            received_close = getattr(exc, 'rcvd', None)
            sent_close = getattr(exc, 'sent', None)
            close_code = getattr(received_close or sent_close, 'code', None)
            return close_code not in NON_RETRYABLE_CLOSE_CODES
        if isinstance(exc, ssl.SSLCertVerificationError):
            return False
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True
        return None

    @staticmethod
    def _record_sent_payload(payload: bytes, source: str, stream_state: _AudioStreamState,
                             send_progress: _AudioSendProgress):
        is_first, sample_count, seconds_sent = send_progress.record_payload(payload)
        if is_first:
            logger.debug('Sent first Modulate audio payload; source=%s, bytes=%d, samples=%d, '
                         'sent_audio_seconds=%.3f, stream_audio_seconds=%.3f', source, len(payload), sample_count,
                         seconds_sent, stream_state.seconds())

    def _send_audio(self, websocket, input_queue, stream_state: _AudioStreamState, replay_payloads: list[bytes],
                    replay_reaches_eof: bool, stop_event: threading.Event, error_queue: queue.SimpleQueue,
                    send_progress: _AudioSendProgress):
        pacer = (_AudioSendPacer(self.max_audio_send_rate, stop_event, clock=self.clock)
                 if self.max_audio_send_rate is not None else None)

        def send_payload(payload: bytes, source: str) -> bool:
            if stop_event.is_set():
                return False
            if pacer is not None and not pacer.wait_until_ready():
                return False
            websocket.send(payload)
            self._record_sent_payload(payload, source, stream_state, send_progress)
            if pacer is not None:
                pacer.record_sent(len(payload) // np.dtype('<f4').itemsize)
            return True

        try:
            try:
                for payload in replay_payloads:
                    if stop_event.is_set():
                        return
                    try:
                        if not send_payload(payload, 'replay'):
                            return
                    except Exception as exc:
                        self._queue_error(error_queue, exc, retryable=self._transport_retryability(exc) is True)
                        stop_event.set()
                        self._interrupt_websocket(websocket)
                        return
            finally:
                # A large reconnect snapshot can otherwise keep already
                # acknowledged payloads alive for the rest of the connection.
                replay_payloads.clear()

            if stop_event.is_set():
                return
            if replay_reaches_eof:
                websocket.send('')
                return

            while not stop_event.is_set():
                try:
                    audio = input_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._stop_event.is_set():
                    return
                if audio is None:
                    stream_state.mark_eof()
                    if stop_event.is_set():
                        return
                    websocket.send('')
                    return

                try:
                    payload, sample_count = stream_state.add_audio(audio)
                except Exception as exc:
                    self._queue_error(error_queue, exc, retryable=False)
                    stop_event.set()
                    self._interrupt_websocket(websocket)
                    return
                if sample_count == 0:
                    continue
                if not send_payload(payload, 'live'):
                    return
        except Exception as exc:
            self._queue_error(error_queue, exc, retryable=self._transport_retryability(exc) is True)
            stop_event.set()
            self._interrupt_websocket(websocket)

    def _connect(self):
        connector = self.connector
        if connector is None:
            from websockets.sync.client import connect
            connector = connect
        return connector(self._connection_url(), **self._connection_kwargs())

    def _connect_interruptibly(self):
        """Let cancellation finish even during a blocking DNS/TLS handshake.

        The SDK bounds a handshake with open_timeout, but it cannot expose the
        socket before connect() returns. The daemon owns that pending attempt;
        registering the result guarantees even a late connection gets closed.
        """
        if self._stop_event.is_set():
            return None
        result = queue.SimpleQueue()

        def establish():
            if self._stop_event.is_set():
                return
            try:
                connection = self._connect()
            except Exception as exc:
                result.put((None, exc))
                return
            if self._register_websocket(connection):
                result.put((connection, None))

        threading.Thread(target=establish, name='modulate-connect', daemon=True).start()
        while not self._stop_event.is_set():
            try:
                connection, error = result.get(timeout=0.1)
            except queue.Empty:
                continue
            if error is not None:
                raise error
            return connection
        return None

    def _run_connection(self, input_queue, output_queue, stream_state: _AudioStreamState):
        stop_event = threading.Event()
        sender_errors = queue.SimpleQueue()
        sender_thread = None
        receive_error = None
        done_received = False
        progress_event = threading.Event()
        send_progress = _AudioSendProgress()
        connection = None

        replay_start, replay_payloads, replay_reaches_eof = stream_state.replay_snapshot()
        connection_offset = replay_start / SAMPLE_RATE

        # A final utterance can acknowledge the last retained sample just
        # before the socket disappears, leaving only the server's ``done``
        # message outstanding. Reopening a connection in that state would send
        # config followed by an empty EOF frame, which the service can reject
        # as "Invalid input audio". The acknowledged final is already
        # authoritative, so there is no audio left to recover.
        if replay_reaches_eof and not replay_payloads:
            logger.debug('All Modulate audio was acknowledged before disconnect; skipping an empty reconnect')
            return

        with self._connection_lock:
            if self._stop_event.is_set():
                return
            self._sender_stop_event = stop_event
        try:
            try:
                connection = self._connect_interruptibly()
                if connection is None:
                    return
            except Exception as exc:
                raise ModulateStreamingError(
                    f'Failed to connect to Modulate streaming API ({self._transport_error_label(exc)})',
                    retryable=self._transport_retryability(exc) is True) from None

            with connection as websocket:
                if not self._register_websocket(websocket):
                    return
                response = getattr(websocket, 'response', None)
                response_headers = getattr(response, 'headers', None)
                negotiated_extensions = (response_headers.get('Sec-WebSocket-Extensions')
                                         if response_headers is not None else None)
                logger.debug('Modulate WebSocket connected; remote_address=%s, negotiated_extensions=%s',
                             getattr(websocket, 'remote_address', None), negotiated_extensions or 'none')
                connection_config = self._connection_config()
                if logger.isEnabledFor(logging.DEBUG):
                    custom_terms = connection_config.get('custom_terms', [])
                    logger.debug('Modulate custom terms (%d): %s', len(custom_terms),
                                 json.dumps(custom_terms, ensure_ascii=False))
                websocket.send(json.dumps(connection_config, ensure_ascii=False))
                response_started_at = time.monotonic()
                sender_thread = threading.Thread(
                    target=self._send_audio,
                    args=(websocket, input_queue, stream_state, replay_payloads, replay_reaches_eof, stop_event,
                          sender_errors, send_progress),
                    name='modulate-audio-sender',
                    daemon=True,
                )
                sender_thread.start()
                try:
                    done_received = self._receive_results(websocket, stream_state, connection_offset, output_queue,
                                                          progress_event, send_progress, response_started_at)
                except Exception as exc:
                    receive_error = exc
                finally:
                    stop_event.set()
                    if sender_thread.is_alive():
                        self._interrupt_websocket(websocket)
                    sender_thread.join(timeout=5)

            if self._stop_event.is_set():
                return
            sender_stuck = sender_thread is not None and sender_thread.is_alive()
            try:
                sender_error_label, sender_error_retryable = sender_errors.get_nowait()
            except queue.Empty:
                sender_error_label = None
                sender_error_retryable = False
            if isinstance(receive_error, ModulateStreamingError) and not receive_error.retryable:
                raise receive_error
            receive_retryability = self._transport_retryability(receive_error) if receive_error else None
            if receive_error and not isinstance(receive_error, ModulateStreamingError):
                if receive_retryability is None:
                    raise ModulateStreamingError(
                        f'Failed to process a Modulate streaming result ({type(receive_error).__name__})') from None
                if not receive_retryability:
                    raise ModulateStreamingError(
                        'Modulate streaming request was rejected '
                        f'({self._transport_error_label(receive_error)})') from None
            if sender_error_label:
                raise ModulateStreamingError(f'Failed to send audio to Modulate ({sender_error_label})',
                                             retryable=sender_error_retryable) from None
            if receive_error:
                if isinstance(receive_error, ModulateStreamingError):
                    raise receive_error
                raise ModulateStreamingError(
                    'Modulate streaming connection closed unexpectedly '
                    f'({self._transport_error_label(receive_error)})',
                    retryable=receive_retryability is True) from None
            if sender_stuck:
                raise ModulateStreamingError('Modulate audio sender did not stop cleanly', retryable=True)
            if not done_received:
                raise ModulateStreamingError('Modulate streaming connection ended before the done message',
                                             retryable=True)
        except ModulateStreamingError as exc:
            if self._stop_event.is_set():
                return
            exc.made_progress = exc.made_progress or progress_event.is_set()
            raise
        except Exception as exc:
            if self._stop_event.is_set():
                return
            raise ModulateStreamingError(
                f'Modulate streaming connection failed ({self._transport_error_label(exc)})',
                retryable=self._transport_retryability(exc) is True,
                made_progress=progress_event.is_set()) from None
        finally:
            with self._connection_lock:
                if self._sender_stop_event is stop_event:
                    self._active_websocket = None
                    self._sender_stop_event = None

    def _prime_stream(self, input_queue, stream_state: _AudioStreamState) -> bool:
        """Buffer the first usable block so a WebSocket never opens without audio ready."""
        logger.debug('Waiting for initial Modulate audio before opening the streaming connection')
        empty_blocks = 0
        while not self._stop_event.is_set():
            try:
                audio = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._stop_event.is_set():
                return False
            if audio is None:
                stream_state.mark_eof()
                raise ModulateStreamingError(
                    'Audio source ended before producing any samples for Modulate')
            try:
                payload, sample_count = stream_state.add_audio(audio)
            except Exception as exc:
                raise ModulateStreamingError(
                    f'Invalid initial audio from upstream ({type(exc).__name__})') from None
            if sample_count == 0:
                empty_blocks += 1
                continue
            if not np.isfinite(np.frombuffer(payload, dtype='<f4')).all():
                raise ModulateStreamingError('Invalid initial audio from upstream (non-finite samples)')
            logger.debug('Buffered initial Modulate audio; bytes=%d, samples=%d, skipped_empty_blocks=%d',
                         len(payload), sample_count, empty_blocks)
            return True
        return False

    def _run_stream(self, input_queue, output_queue):
        stream_state = _AudioStreamState()
        if not self._prime_stream(input_queue, stream_state):
            return
        logger.info('Connecting to Modulate streaming endpoint; proxy=%s, verify_ssl=%s', bool(self.proxy),
                    self.verify_ssl)
        retry_count = 0
        while not self._stop_event.is_set():
            try:
                self._run_connection(input_queue, output_queue, stream_state)
                break
            except ModulateStreamingError as exc:
                if self._stop_event.is_set():
                    break
                if exc.made_progress:
                    retry_count = 0
                if not exc.retryable or retry_count >= self.reconnect_attempts:
                    raise
                retry_count += 1
                delay = min(self.reconnect_backoff_seconds * (2**(retry_count - 1)), 30.0)
                logger.warning('Modulate connection interrupted; reconnecting (%d/%d) in %.1f seconds', retry_count,
                               self.reconnect_attempts, delay)
                if self.sleeper is not None:
                    self.sleeper(delay)
                else:
                    self._stop_event.wait(delay)

    def loop(self, input_queue, output_queue):
        try:
            self._run_stream(input_queue, output_queue)
        except Exception:
            if not self._stop_event.is_set():
                # The worker wrapper reports a real failure and owns its EOF.
                raise
        # Natural completion and explicit cancellation each publish one EOF.
        output_queue.put(None)
