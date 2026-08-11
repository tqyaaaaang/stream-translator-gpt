import ipaddress
import logging
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from urllib.parse import urlparse

import numpy as np

from . import __version__

SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 512  # Requested by silero-vad >= v5
FRAME_DURATION = SAMPLES_PER_FRAME / SAMPLE_RATE

BOLD = '\033[1m'
ENDC = '\033[0m'

logger = logging.getLogger('main')


def write_console_utf8(text: str, *, stream=None) -> None:
    """Write one line of business output without depending on the console's text encoding."""
    stream = sys.stdout if stream is None else stream
    line = text + '\n'
    try:
        stream.write(line)
        stream.flush()
    except UnicodeEncodeError:
        binary_stream = getattr(stream, 'buffer', None)
        if binary_stream is not None:
            binary_stream.write(line.encode('utf-8'))
            binary_stream.flush()
            return

        encoding = getattr(stream, 'encoding', None) or 'ascii'
        safe_line = line.encode(encoding, errors='backslashreplace').decode(encoding)
        stream.write(safe_line)
        stream.flush()


class TranslationTask:

    class Phase(Enum):
        SLICED = 1
        TRANSCRIBED = 2
        TRANSLATED = 3

    def __init__(self, line_id: int, audio: np.array, time_range: tuple[float, float]):
        self.id = line_id
        self.audio = audio
        self.transcript = None
        self.context_transcripts = None
        # Translation-context mode snapshots the rolling input state separately from
        # the state generated for this task. Keeping both on the task makes retries
        # deterministic and prevents a late response from reading newer state.
        self.input_translation_context = None
        self.translation_context = None
        self.translation = None
        self.time_range = time_range
        self.start_time = None
        self.translation_failed = False
        # Translation API workers and the ordered queue worker use this lock to
        # linearize result publication against timeout/failure finalization.
        self._translation_lock = threading.Lock()
        self._translation_finalized = False
        self.phase = self.Phase.SLICED


class LoopWorkerBase(ABC):

    @abstractmethod
    def loop(self):
        pass


class LogTime:
    """Log elapsed wall-clock duration without work when the level is disabled."""

    def __init__(self, text: str, *args, level=logging.INFO):
        self.text = text
        self.args = args
        self.level = level
        self.start_time = None

    def set_log(self, text: str, *args) -> None:
        self.text = text
        self.args = args

    def __enter__(self):
        if logger.isEnabledFor(self.level):
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.start_time is not None and self.text:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            logger.log(self.level, self.text + ', run in time %.2fms', *self.args, duration_ms)
        return False


def start_daemon_thread(func, *args, name=None, on_error=None, output_eof_on_error=True, **kwargs):

    def wrapper():
        logger.debug('Worker started: %s', threading.current_thread().name)
        try:
            func(*args, **kwargs)
        except Exception:
            logger.exception('Unhandled exception in worker: %s', threading.current_thread().name)
            if on_error is not None:
                try:
                    on_error()
                except Exception:
                    logger.exception('Worker error callback failed: %s', threading.current_thread().name)
            output_queue = kwargs.get('output_queue', None)
            if output_queue is not None and output_eof_on_error:
                output_queue.put(None)
        finally:
            logger.debug('Worker stopped: %s', threading.current_thread().name)

    thread = threading.Thread(target=wrapper, name=name or getattr(func, '__name__', 'worker'))
    thread.daemon = True
    thread.start()
    return thread


def sec2str(second: float):
    dt = datetime.fromtimestamp(second, tz=timezone.utc)
    result = dt.strftime('%H:%M:%S')
    result += ',' + str(int(second * 10 % 10))
    return result


def is_ip_host(url: str):
    if not url:
        return False
    try:
        ipaddress.ip_address(urlparse(url).hostname)
        return True
    except (ValueError, TypeError):
        return False


class ClientPool:
    verify_ssl = True
    _openai_clients = []
    _openai_index = 0
    _openai_lock = threading.Lock()
    _whisper_clients = []
    _whisper_index = 0
    _whisper_lock = threading.Lock()
    _google_clients = []
    _google_index = 0
    _google_lock = threading.Lock()

    @classmethod
    def _should_verify_ssl(cls, base_url):
        if not cls.verify_ssl:
            return False
        if is_ip_host(base_url):
            logger.warning('Base URL host "%s" is a bare IP; disabling TLS certificate verification for it.',
                           urlparse(base_url).hostname)
            return False
        return True

    @classmethod
    def init(cls,
             openai_api_key,
             google_api_key,
             proxy=None,
             openai_base_url=None,
             whisper_api_key=None,
             whisper_base_url=None,
             google_base_url=None,
             verify_ssl=True):
        cls.verify_ssl = verify_ssl

        cls._openai_clients = []
        cls._openai_index = 0
        if openai_api_key:
            from openai import OpenAI
            import httpx
            verify = cls._should_verify_ssl(openai_base_url)
            for key in openai_api_key.split(','):
                key = key.strip()
                client = OpenAI(
                    api_key=key,
                    base_url=openai_base_url,
                    default_headers={'User-Agent': f'stream-translator-gpt/{__version__}'},
                    http_client=httpx.Client(proxy=proxy, verify=verify),
                )
                cls._openai_clients.append(client)
            logger.info('Initialized OpenAI client pool with %d client(s); custom endpoint=%s, proxy=%s, verify_ssl=%s',
                        len(cls._openai_clients), bool(openai_base_url), bool(proxy), verify)

        cls._whisper_clients = []
        cls._whisper_index = 0
        effective_whisper_api_key = whisper_api_key or openai_api_key
        effective_whisper_base_url = whisper_base_url or openai_base_url
        if effective_whisper_api_key:
            from openai import OpenAI
            import httpx
            verify = cls._should_verify_ssl(effective_whisper_base_url)
            for key in effective_whisper_api_key.split(','):
                key = key.strip()
                client = OpenAI(
                    api_key=key,
                    base_url=effective_whisper_base_url,
                    default_headers={'User-Agent': f'stream-translator-gpt/{__version__}'},
                    http_client=httpx.Client(proxy=proxy, verify=verify),
                )
                cls._whisper_clients.append(client)
            logger.info(
                'Initialized Whisper API client pool with %d client(s); custom endpoint=%s, proxy=%s, verify_ssl=%s',
                len(cls._whisper_clients), bool(effective_whisper_base_url), bool(proxy), verify)

        cls._google_clients = []
        cls._google_index = 0
        if google_api_key:
            from google import genai
            http_options = {'client_args': {'verify': cls._should_verify_ssl(google_base_url)}}
            if proxy:
                http_options['client_args']['proxy'] = proxy
            if google_base_url:
                http_options['base_url'] = google_base_url
            for key in google_api_key.split(','):
                key = key.strip()
                client = genai.Client(api_key=key, http_options=http_options)
                cls._google_clients.append(client)
            logger.info('Initialized Google client pool with %d client(s); custom endpoint=%s, proxy=%s, verify_ssl=%s',
                        len(cls._google_clients), bool(google_base_url), bool(proxy),
                        http_options['client_args']['verify'])

    @classmethod
    def get_openai_client(cls):
        with cls._openai_lock:
            if not cls._openai_clients:
                client = None
                index = None
                client_count = 0
            else:
                index = cls._openai_index
                client_count = len(cls._openai_clients)
                client = cls._openai_clients[index]
                cls._openai_index = (index + 1) % client_count
        if client is None:
            logger.error('OpenAI client pool is empty')
            return None
        logger.debug('Using OpenAI client %d of %d', index + 1, client_count)
        return client

    @classmethod
    def get_google_client(cls):
        with cls._google_lock:
            if not cls._google_clients:
                client = None
                index = None
                client_count = 0
            else:
                index = cls._google_index
                client_count = len(cls._google_clients)
                client = cls._google_clients[index]
                cls._google_index = (index + 1) % client_count
        if client is None:
            logger.error('Google client pool is empty')
            return None
        logger.debug('Using Google client %d of %d', index + 1, client_count)
        return client

    @classmethod
    def get_whisper_client(cls):
        with cls._whisper_lock:
            if not cls._whisper_clients:
                client = None
                index = None
                client_count = 0
            else:
                index = cls._whisper_index
                client_count = len(cls._whisper_clients)
                client = cls._whisper_clients[index]
                cls._whisper_index = (index + 1) % client_count
        if client is None:
            logger.error('Whisper API client pool is empty')
            return None
        logger.debug('Using Whisper API client %d of %d', index + 1, client_count)
        return client


def is_url(address):
    parsed_url = urlparse(address)

    if parsed_url.scheme and parsed_url.scheme != 'file':
        if parsed_url.netloc or (parsed_url.scheme in ['mailto', 'tel', 'data']):
            return True

    if parsed_url.scheme == 'file':
        return False

    if parsed_url.netloc:
        return True

    if os.name == 'nt':
        if re.match(r'^[a-zA-Z]:[\\/]', address):
            return False
        if address.startswith('\\\\') or address.startswith('//'):
            return False
        if '\\' in address and '/' not in address:
            return False

    if address.startswith('/') or address.startswith('./') or address.startswith('../'):
        return False

    if '/' in address or (os.name == 'nt' and '\\' in address):
        if not parsed_url.scheme and not parsed_url.netloc:
            return False

    return False
