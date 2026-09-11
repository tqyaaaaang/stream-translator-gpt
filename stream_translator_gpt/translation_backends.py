"""Provider-specific translation only: no task queues, scheduling or legacy dispatch.

A backend is reusable configuration. Each request worker opens its own session;
that session binds an explicitly owned client getter, avoiding thread-local state
or callbacks into the pipeline. Requests/results are immutable value objects.
"""

import hashlib
import logging
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Any

from .logging_utils import DDEBUG
from .translation_clients import TranslationClientPool
from .translation_protocol import TranslationProtocol, TranslationOutputMode, _TRANSLATION_CONTEXT_RESPONSE_FORMAT
from .translation_types import TranslationRequest, TranslationResult

logger = logging.getLogger('main')

def _gpt_model_version(model: str) -> tuple[int, int] | None:
    match = re.match(r'^gpt-(\d+)(?:\.(\d+))?', model.lower())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)



def _request_timeout_options(deadline: float | None) -> dict:
    if deadline is None:
        return {}
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Translation request deadline expired before the API call')
    return {'timeout': remaining}


@dataclass(frozen=True)
class _BoundTranslationSession:
    backend: '_LLMBackend'
    get_client: Callable[[], Any]

    def translate(self, request: TranslationRequest) -> TranslationResult:
        return self.backend.translate(request, self.get_client())


class _LLMBackend(ABC):
    """Shared provider plumbing, without pipeline or mutable task responsibilities."""

    def __init__(self, *, client_pool: TranslationClientPool, protocol: TranslationProtocol,
                 model: str, debug_mode: bool = False):
        self.client_pool = client_pool
        self.protocol = protocol
        self.model = model
        self.debug_mode = debug_mode
        logger.warning('Using %s API as translation engine.', model)

    @contextmanager
    def open_session(self):
        with self.client_pool.worker_session() as get_client:
            yield _BoundTranslationSession(self, get_client)

    def _log_prompt_messages(self, system_prompt: str, user_content: str) -> None:
        if self.debug_mode:
            logger.log(DDEBUG, '[System] %s', system_prompt)
            logger.log(DDEBUG, '[User] %s', user_content)

    def translate(self, request: TranslationRequest, client) -> TranslationResult:
        system_prompt, user_content = self.protocol.build_messages(request)
        self._log_prompt_messages(system_prompt, user_content)
        try:
            completion, usage = self._request(client, system_prompt, user_content, request.deadline)
            result = self.protocol.parse(completion)
            if usage and self.debug_mode:
                logger.debug('[Usage] %s', usage)
            return result
        except Exception as exc:
            logger.error('%s translation request failed (%s)', self.provider_name, type(exc).__name__)
            raise

    @abstractmethod
    def _request(self, client, system_prompt: str, user_content: str, deadline: float | None):
        """Return completion text and usage for one attempt; do not retry."""
        raise NotImplementedError


class OpenAITranslationBackend(_LLMBackend):
    provider_name = 'OpenAI'

    def __init__(self,
                 prompt_cache_key: str = None,
                 prompt_cache_enabled: bool = False,
                 api_mode: str = 'responses',
                 temperature: float = None,
                 top_p: float = None,
                 reasoning_effort: str = None,
                 verbosity: str = None,
                 service_tier: str = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_enabled = prompt_cache_enabled
        self.api_mode = api_mode
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.service_tier = service_tier
        self._prompt_cache_size_checked = False
        self._prompt_cache_size_lock = threading.Lock()

    def _uses_explicit_prompt_cache(self) -> bool:
        return self.api_mode == 'responses' and self.prompt_cache_enabled

    def _effective_prompt_cache_key(self, system_prompt: str) -> str | None:
        if self.prompt_cache_key:
            return self.prompt_cache_key
        if not self.prompt_cache_enabled:
            return None
        stable_prefix = f'{self.model}\0{system_prompt}'.encode('utf-8')
        digest = hashlib.sha256(stable_prefix).hexdigest()[:32]
        return f'stream-translator-gpt:{digest}'

    def _build_responses_input(self, system_prompt: str, user_content: str):
        system_content = {
            'type': 'input_text',
            'text': system_prompt,
        }
        if self._uses_explicit_prompt_cache():
            system_content['prompt_cache_breakpoint'] = {'mode': 'explicit'}

        return [
            {
                'role': 'system',
                'content': [system_content],
            },
            {
                'role': 'user',
                'content': [{
                    'type': 'input_text',
                    'text': user_content,
                }],
            },
        ]

    def _warn_if_prompt_cache_prefix_is_too_short(self, client, system_prompt: str,
                                                request_deadline: float | None = None) -> None:
        if not self._uses_explicit_prompt_cache() or self._prompt_cache_size_checked:
            return
        with self._prompt_cache_size_lock:
            if self._prompt_cache_size_checked:
                return
            self._prompt_cache_size_checked = True
        # Reserve this one-time check under the lock, but never hold a global
        # mutex during network I/O: other workers may translate immediately.
        try:
            stable_input = [self._build_responses_input(system_prompt, '')[0]]
            count = client.responses.input_tokens.count(model=self.model, input=stable_input,
                                                       **_request_timeout_options(request_deadline))
            token_count = count.input_tokens
        except Exception as exc:
            logger.warning('Unable to determine the explicit prompt cache prefix size (%s)', type(exc).__name__)
            return

        if token_count < 1024:
            logger.warning('OpenAI explicit prompt cache is enabled, but the stable system prompt is only '
                           '%d token(s); OpenAI requires at least 1024 prefix tokens, so this prompt is not '
                           'cache-eligible', token_count)
        else:
            logger.debug('OpenAI explicit prompt cache prefix contains %d token(s)', token_count)

    @staticmethod
    def _log_prompt_cache_usage(usage) -> None:
        details = getattr(usage, 'input_tokens_details', None)
        if details is None:
            details = getattr(usage, 'prompt_tokens_details', None)
        if details is None:
            return
        cached_tokens = getattr(details, 'cached_tokens', None)
        cache_write_tokens = getattr(details, 'cache_write_tokens', None)
        if cached_tokens is not None or cache_write_tokens is not None:
            logger.debug('OpenAI prompt cache usage: read=%s token(s), written=%s token(s)',
                         cached_tokens if cached_tokens is not None else 'unknown',
                         cache_write_tokens if cache_write_tokens is not None else 'unknown')

    def _translate_with_responses(self, client, system_prompt: str, user_content: str,
                                  request_deadline: float | None = None):
        self._warn_if_prompt_cache_prefix_is_too_short(client, system_prompt, request_deadline)
        response_input = self._build_responses_input(system_prompt, user_content)
        prompt_cache_key = self._effective_prompt_cache_key(system_prompt)
        kwargs = {
            "model": self.model,
            "input": response_input,
            # Translation requests are independent and don't need server-side conversation state.
            "store": False,
        }

        text_options = {}
        if self.protocol.output_mode is TranslationOutputMode.CONTEXT_JSON:
            text_options["format"] = _TRANSLATION_CONTEXT_RESPONSE_FORMAT
        elif self.protocol.output_mode is TranslationOutputMode.JSON:
            text_options["format"] = {"type": "json_object"}

        version = _gpt_model_version(self.model)
        if version is not None and self.reasoning_effort is None and version >= (5, 0):
            # Use the lowest supported reasoning level for latency-sensitive translation.
            kwargs["reasoning"] = {"effort": "none" if version >= (5, 1) else "minimal"}

        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key
        if self._uses_explicit_prompt_cache():
            kwargs["prompt_cache_options"] = {
                "mode": "explicit",
                "ttl": "30m",
            }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.verbosity is not None:
            text_options["verbosity"] = self.verbosity
        if text_options:
            kwargs["text"] = text_options
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier

        logger.debug('Sending translation request to OpenAI Responses API')
        kwargs.update(_request_timeout_options(request_deadline))
        response = client.responses.create(**kwargs)
        logger.debug('Received translation response from OpenAI Responses API')
        return response.output_text, getattr(response, 'usage', None)

    def _translate_with_chat_completions(self, client, system_prompt: str, user_content: str,
                                        request_deadline: float | None = None):
        kwargs = {
            "model": self.model,
            "messages": [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ],
        }
        if self.protocol.expects_json_result:
            kwargs["response_format"] = {"type": "json_object"}

        version = _gpt_model_version(self.model)
        if version is not None and self.reasoning_effort is None and version >= (5, 0):
            kwargs["reasoning_effort"] = "none" if version >= (5, 1) else "minimal"
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            kwargs["verbosity"] = self.verbosity
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier
        if self.prompt_cache_key:
            kwargs["prompt_cache_key"] = self.prompt_cache_key

        logger.debug('Sending translation request to OpenAI Chat Completions compatibility API')
        kwargs.update(_request_timeout_options(request_deadline))
        completion = client.chat.completions.create(**kwargs)
        logger.debug('Received translation response from OpenAI Chat Completions compatibility API')
        return completion.choices[0].message.content, getattr(completion, 'usage', None)


    def _request(self, client, system_prompt: str, user_content: str, deadline: float | None):
        if self.api_mode == 'chat-completions':
            completion, usage = self._translate_with_chat_completions(client, system_prompt, user_content, deadline)
        else:
            completion, usage = self._translate_with_responses(client, system_prompt, user_content, deadline)
        if usage:
            self._log_prompt_cache_usage(usage)
        return completion, usage


class GeminiTranslationBackend(_LLMBackend):
    provider_name = 'Gemini'

    def __init__(self, temperature: float = None, top_p: float = None, top_k: int = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def _get_thinking_config(self, types):
        """Return the lowest supported thinking setting for the selected Gemini model."""
        model = self.model.lower().removeprefix('models/')

        if model.startswith('gemini-2.5-'):
            if '-pro' in model:
                return types.ThinkingConfig(thinking_budget=128)
            return types.ThinkingConfig(thinking_budget=0)

        major_version_match = re.match(r'gemini-(\d+)', model)
        if major_version_match and int(major_version_match.group(1)) >= 3:
            thinking_level = 'low' if '-pro' in model else 'minimal'
            return types.ThinkingConfig(thinking_level=thinking_level)

        return None


    def _request(self, client, system_prompt: str, user_content: str, deadline: float | None):
        from google.genai import types

        messages = [{'role': 'user', 'parts': [{'text': user_content}]}]
        config_kwargs = {
            'system_instruction': system_prompt,
            'response_mime_type': 'application/json' if self.protocol.expects_json_result else 'text/plain',
            'safety_settings': [
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
            ],
        }
        if self.temperature is not None:
            config_kwargs['temperature'] = self.temperature
        if self.top_p is not None:
            config_kwargs['top_p'] = self.top_p
        if self.top_k is not None:
            config_kwargs['top_k'] = self.top_k
        thinking_config = self._get_thinking_config(types)
        if thinking_config is not None:
            config_kwargs['thinking_config'] = thinking_config
        request_options = _request_timeout_options(deadline)
        if request_options:
            config_kwargs['http_options'] = types.HttpOptions(
                timeout=max(1, math.ceil(request_options['timeout'] * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1),
            )
        config = types.GenerateContentConfig(**config_kwargs)
        logger.debug('Sending translation request to Gemini API')
        response = client.models.generate_content(model=self.model, contents=messages, config=config)
        logger.debug('Received translation response from Gemini API')
        return response.text, getattr(response, 'usage_metadata', None)
