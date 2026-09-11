"""Independent, worker-owned API clients for the parallel translation pipeline.

One pool belongs to one translation backend. Workers share only immutable
connection settings and the round-robin selector; neither transports nor their
lifetimes are shared between workers. The historical common.ClientPool remains
separate for legacy translation and Whisper transcription.
"""

import asyncio
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from . import __version__
from .common import is_ip_host

logger = logging.getLogger('main')


@dataclass(frozen=True)
class _ClientConfig:
    provider: Literal['openai', 'google']
    keys: tuple[str, ...] = field(repr=False)
    base_url: str | None = field(repr=False)
    proxy: str | None = field(repr=False)
    verify_ssl: bool
    user_agent: str


class TranslationClientPool:
    """Lazy per-worker client caches with pool-wide round-robin key selection.

Use ``with pool.worker_session() as get_client`` inside each worker. Every
attempt calls ``get_client()`` once to select its key, including retries. With
N workers and K distinct keys, at most N * K clients can be owned concurrently.
Constructing another pool never changes this pool's settings or selection state.
"""

    REQUEST_TIMEOUT = 60.0

    def __init__(self,
                 provider: Literal['openai', 'google'],
                 api_keys: str | None,
                 base_url: str | None = None,
                 proxy: str | None = None,
                 verify_ssl: bool = True):
        if provider not in ('openai', 'google'):
            raise ValueError(f'Unsupported translation client provider: {provider}')
        verify = verify_ssl
        if verify and is_ip_host(base_url):
            logger.warning('Base URL host "%s" is a bare IP; disabling TLS certificate verification for it.',
                           urlparse(base_url).hostname)
            verify = False
        self._config = _ClientConfig(
            provider=provider,
            keys=tuple(key.strip() for key in api_keys.split(',')) if api_keys else (),
            base_url=base_url,
            proxy=proxy,
            verify_ssl=verify,
            user_agent=f'stream-translator-gpt/{__version__}',
        )
        self._key_index = 0
        self._key_lock = threading.Lock()
        logger.info(
            'Configured %s translation key pool with %d key(s); custom endpoint=%s, proxy=%s, verify_ssl=%s',
            'OpenAI' if provider == 'openai' else 'Google', len(self._config.keys),
            bool(base_url), bool(proxy), verify)

    @property
    def provider(self) -> Literal['openai', 'google']:
        return self._config.provider

    def _select_key(self) -> str:
        # Client construction and all network I/O happen outside this short lock.
        with self._key_lock:
            index = self._key_index
            self._key_index = (index + 1) % len(self._config.keys)
        return self._config.keys[index]

    def _new_client(self, key):
        """Create one owned client; retries are delegated to the scheduler."""
        config = self._config
        if self.provider == 'openai':
            import httpx
            from openai import OpenAI

            transport = httpx.Client(proxy=config.proxy, verify=config.verify_ssl,
                                     timeout=self.REQUEST_TIMEOUT)
            try:
                return OpenAI(
                    api_key=key,
                    base_url=config.base_url,
                    default_headers={'User-Agent': config.user_agent},
                    http_client=transport,
                    max_retries=0,
                    timeout=self.REQUEST_TIMEOUT,
                )
            except BaseException:
                # SDK construction failed before ownership could transfer.
                try:
                    transport.close()
                except Exception:
                    logger.error('Failed to close an unowned OpenAI worker transport')
                raise

        from google import genai

        http_options = {
            'client_args': {'verify': config.verify_ssl},
            'timeout': round(self.REQUEST_TIMEOUT * 1000),
            'retry_options': {'attempts': 1},
            'headers': {'User-Agent': config.user_agent},
        }
        if config.proxy:
            http_options['client_args']['proxy'] = config.proxy
        if config.base_url:
            http_options['base_url'] = config.base_url
        return genai.Client(api_key=key, http_options=http_options)

    def _close_client(self, client):
        try:
            client.close()
        except Exception:
            # Never leak exception text containing credential-bearing URLs.
            # Cleanup must continue for the rest of this worker's clients.
            logger.error('Failed to close a %s worker client', self.provider)
        if self.provider == 'google':
            # genai constructs both transports; close() closes only sync HTTP.
            # Synchronous translation workers do not run an asyncio event loop.
            try:
                asyncio.run(client.aio.aclose())
            except Exception:
                logger.error('Failed to close a Google worker async transport')

    @contextmanager
    def worker_session(self):
        """Yield an owner-thread-only getter and close its clients upon exit.

        The worker enters and exits the session itself, so an uncancellable HTTP
        request retains its client until it returns, even after logical timeout.
        An idle worker does not construct any clients.
        """
        if not self._config.keys:
            raise RuntimeError(f'{self.provider.capitalize()} translation key pool is empty')
        owner = threading.get_ident()
        clients = {}
        active = True

        def get_client():
            if not active or threading.get_ident() != owner:
                raise RuntimeError('Translation clients must be used inside their owning worker session')
            key = self._select_key()
            if key not in clients:
                clients[key] = self._new_client(key)
            return clients[key]

        try:
            yield get_client
        finally:
            active = False
            for client in clients.values():
                self._close_client(client)
