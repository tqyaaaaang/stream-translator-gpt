import hashlib
import json
import logging
import queue
import re
import threading
import time
from abc import abstractmethod
from collections import deque
from datetime import datetime, timedelta, timezone

from .common import TranslationTask, LoopWorkerBase, ClientPool, LogTime


logger = logging.getLogger('main')


def _gpt_model_version(model: str) -> tuple[int, int] | None:
    match = re.match(r'^gpt-(\d+)(?:\.(\d+))?', model.lower())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


# The double quotes in the values of JSON have not been escaped, so manual escaping is necessary.
def _escape_specific_quotes(input_string):
    quote_positions = [i for i, char in enumerate(input_string) if char == '"']

    if len(quote_positions) <= 4:
        return input_string

    for i in range(3, len(quote_positions) - 1):
        position = quote_positions[i]
        input_string = input_string[:position] + '\\"' + input_string[position + 1:]
        quote_positions = [pos + 1 if pos > position else pos for pos in quote_positions]

    return input_string


def _parse_json_completion(completion):
    pattern = re.compile(r'\{.*}', re.DOTALL)
    json_match = pattern.search(completion)

    if not json_match:
        return completion

    json_str = json_match.group(0)
    json_str = _escape_specific_quotes(json_str)

    try:
        json_obj = json.loads(json_str)
        translate_text = json_obj.get('translation', None)
        if not translate_text:
            return completion
        return translate_text
    except json.JSONDecodeError:
        logger.debug('Failed to parse the translation response as JSON', exc_info=True)
        return completion


def _is_task_timeout(task: TranslationTask, timeout: float) -> bool:
    if timeout == 0.0:
        return False
    return datetime.now(timezone.utc) - task.start_time > timedelta(seconds=timeout)


class LLMTranslator(LoopWorkerBase):
    PARALLEL_MAX_NUMBER = 10

    def __init__(self,
                 model: str,
                 prompt: str,
                 history_size: int,
                 use_json_result: bool,
                 timeout: int,
                 retry_if_translation_fails: bool,
                 debug_mode: bool = False) -> None:
        logger.warning('Using %s API as translation engine.', model)
        self.model = model
        self.prompt = prompt
        self.history_size = history_size
        self.use_json_result = use_json_result
        self.timeout = timeout
        self.retry_if_translation_fails = retry_if_translation_fails
        self.debug_mode = debug_mode
        self.processing_queue = deque()
        self.recent_transcripts = deque(maxlen=history_size) if history_size else None

    def _build_messages(self, translation_task: TranslationTask):
        system_prompt = 'You are a professional translator.'
        system_prompt += f'\n{self.prompt}'
        if self.use_json_result:
            system_prompt += '\nOutput the answer in json format, key is translation.'

        if self.history_size and translation_task.context_transcripts is not None:
            system_prompt += '\nThe text under "Reference" is prior context, do NOT translate it.'
            system_prompt += ' Translate ONLY the text under "Translate".'

            user_content = ''
            if translation_task.context_transcripts:
                user_content += '**Reference:**'
                for t in translation_task.context_transcripts:
                    user_content += f'\n> {t}'
                user_content += '\n'
            user_content += f'\n**Translate:**\n{translation_task.transcript}'
        else:
            user_content = translation_task.transcript

        system_prompt += '\nOutput only the translation in one line, nothing else.'
        return system_prompt, user_content

    def _validate_translation(self, translation: str) -> str:
        if not translation:
            return translation
        lines = [l.strip() for l in translation.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            logger.debug('Translation response contained multiple lines; using the final non-empty line')
            return lines[-1]
        return translation

    @abstractmethod
    def translate(self, translation_task: TranslationTask):
        pass

    def _prepare_context(self, task: TranslationTask):
        if self.recent_transcripts is not None:
            task.context_transcripts = list(self.recent_transcripts)
            self.recent_transcripts.append(task.transcript)
            logger.debug('Prepared translation context with %d prior transcript(s)', len(task.context_transcripts))

    def _trigger(self, translation_task: TranslationTask):
        if not translation_task.start_time:
            translation_task.start_time = datetime.now(timezone.utc)
        translation_task.translation_failed = False
        thread = threading.Thread(target=self._translate_with_timing,
                                  name=f'translator-instance-{translation_task.id}',
                                  args=(translation_task,))
        thread.daemon = True
        thread.start()

    def _translate_with_timing(self, translation_task: TranslationTask):
        with LogTime('Translated task %d', translation_task.id):
            self.translate(translation_task)

    def _retrigger_failed_tasks(self):
        now = datetime.now(timezone.utc)
        for task in self.processing_queue:
            if task.translation_failed and not _is_task_timeout(task, self.timeout):
                next_retry_time = getattr(task, 'next_retry_time', None)
                if next_retry_time is not None and now < next_retry_time:
                    continue
                task.retry_count = getattr(task, 'retry_count', 0) + 1
                backoff = min(2**task.retry_count, 30)
                task.next_retry_time = now + timedelta(seconds=backoff)
                self._trigger(task)
                logger.info('Translation failed, retrying (backoff %ds)', backoff)

    def _get_results(self):
        results = []
        while self.processing_queue and (
                self.processing_queue[0].translation or _is_task_timeout(self.processing_queue[0], self.timeout) or
            (self.processing_queue[0].translation_failed and not self.retry_if_translation_fails)):
            task = self.processing_queue.popleft()
            task.phase = TranslationTask.Phase.TRANSLATED
            if not task.translation:
                if _is_task_timeout(task, self.timeout):
                    logger.error('Translation timed out after %ss', self.timeout)
                else:
                    logger.error('Translation failed')
            results.append(task)
        return results

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask]):
        while True:
            if not input_queue.empty() and len(self.processing_queue) < self.PARALLEL_MAX_NUMBER:
                task = input_queue.get()
                if task is None:
                    while len(self.processing_queue) > 0:
                        finished_tasks = self._get_results()
                        for task in finished_tasks:
                            output_queue.put(task)
                        if self.retry_if_translation_fails:
                            self._retrigger_failed_tasks()
                        time.sleep(0.1)
                    logger.debug('All pending translation tasks completed')
                    output_queue.put(None)
                    break
                self._prepare_context(task)
                self.processing_queue.append(task)
                self._trigger(task)
            finished_tasks = self._get_results()
            for task in finished_tasks:
                output_queue.put(task)
            if self.retry_if_translation_fails:
                self._retrigger_failed_tasks()
            time.sleep(0.1)


class GPTTranslator(LLMTranslator):

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

    def _warn_if_prompt_cache_prefix_is_too_short(self, client, system_prompt: str) -> None:
        if not self._uses_explicit_prompt_cache() or self._prompt_cache_size_checked:
            return
        with self._prompt_cache_size_lock:
            if self._prompt_cache_size_checked:
                return
            self._prompt_cache_size_checked = True
            try:
                # Count the same rendered system message used as the stable cache prefix. Message framing also
                # contributes tokens, so string length or a local tokenizer would be inaccurate near the boundary.
                stable_input = [self._build_responses_input(system_prompt, '')[0]]
                count = client.responses.input_tokens.count(model=self.model, input=stable_input)
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

    def _translate_with_responses(self, client, system_prompt: str, user_content: str):
        self._warn_if_prompt_cache_prefix_is_too_short(client, system_prompt)
        response_input = self._build_responses_input(system_prompt, user_content)
        prompt_cache_key = self._effective_prompt_cache_key(system_prompt)
        kwargs = {
            "model": self.model,
            "input": response_input,
            # Translation requests are independent and don't need server-side conversation state.
            "store": False,
        }

        text_options = {}
        if self.use_json_result:
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
        response = client.responses.create(**kwargs)
        logger.debug('Received translation response from OpenAI Responses API')
        return response.output_text, getattr(response, 'usage', None)

    def _translate_with_chat_completions(self, client, system_prompt: str, user_content: str):
        kwargs = {
            "model": self.model,
            "messages": [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ],
        }
        if self.use_json_result:
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
        completion = client.chat.completions.create(**kwargs)
        logger.debug('Received translation response from OpenAI Chat Completions compatibility API')
        return completion.choices[0].message.content, getattr(completion, 'usage', None)

    def translate(self, translation_task: TranslationTask):
        logger.debug('Preparing OpenAI translation request')
        client = ClientPool.get_openai_client()

        system_prompt, user_content = self._build_messages(translation_task)
        if self.debug_mode:
            logger.debug('[System] %s', system_prompt)
            logger.debug('[User] %s', user_content)

        try:
            if self.api_mode == 'chat-completions':
                translation, usage = self._translate_with_chat_completions(client, system_prompt, user_content)
            else:
                translation, usage = self._translate_with_responses(client, system_prompt, user_content)

            translation_task.translation = translation
            if not translation_task.translation:
                translation_task.translation_failed = True
                logger.error('OpenAI translation API returned no translation text')
                return
            if usage:
                self._log_prompt_cache_usage(usage)
                if self.debug_mode:
                    logger.debug('[Usage] %s', usage)
            if self.use_json_result:
                translation_task.translation = _parse_json_completion(translation_task.translation)
            translation_task.translation = self._validate_translation(translation_task.translation)
        except Exception as e:
            translation_task.translation_failed = True
            logger.error('OpenAI translation request failed (%s)', type(e).__name__)
            return


class GeminiTranslator(LLMTranslator):

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

    def translate(self, translation_task: TranslationTask):
        from google.genai import types

        logger.debug('Preparing Gemini translation request')
        client = ClientPool.get_google_client()

        system_prompt, user_content = self._build_messages(translation_task)
        if self.debug_mode:
            logger.debug('[System] %s', system_prompt)
            logger.debug('[User] %s', user_content)
        messages = [{'role': 'user', 'parts': [{'text': user_content}]}]

        config_kwargs = {
            'system_instruction':
                system_prompt,
            'response_mime_type':
                'application/json' if self.use_json_result else 'text/plain',
            'safety_settings': [
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE')
            ]
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
        config = types.GenerateContentConfig(**config_kwargs)

        try:
            logger.debug('Sending translation request to Gemini API')
            response = client.models.generate_content(model=self.model, contents=messages, config=config)
            logger.debug('Received translation response from Gemini API')
            translation_task.translation = response.text
            if self.debug_mode and hasattr(response, 'usage_metadata') and response.usage_metadata:
                logger.debug('[Usage] %s', response.usage_metadata)
            if self.use_json_result:
                translation_task.translation = _parse_json_completion(translation_task.translation)
            translation_task.translation = self._validate_translation(translation_task.translation)
        except Exception as e:
            translation_task.translation_failed = True
            logger.error('Gemini translation request failed (%s)', type(e).__name__)
            return
