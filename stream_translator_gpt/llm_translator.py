import json
import queue
import threading
import time
import re
import logging
from collections import deque
from datetime import datetime, timedelta

import google.generativeai as genai
from google.api_core.exceptions import InternalServerError, ResourceExhausted, TooManyRequests
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI, DefaultHttpxClient, APITimeoutError, APIConnectionError

from .common import TranslationTask, LoopWorkerBase, LogTime


logger = logging.getLogger('main')


# The double quotes in the values of JSON have not been escaped, so manual escaping is necessary.
def escape_specific_quotes(input_string):
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
    json_str = escape_specific_quotes(json_str)

    try:
        json_obj = json.loads(json_str)
        translate_text = json_obj.get('translation', None)
        if not translate_text:
            return completion
        return translate_text
    except json.JSONDecodeError:
        return completion


class LLMClint():

    class LLM_TYPE:
        GPT = 'GPT'
        GEMINI = 'Gemini'

    def __init__(self, llm_type: str, model: str, prompt: str, history_size: int, proxy: str,
                 use_json_result: bool) -> None:
        if llm_type not in (self.LLM_TYPE.GPT, self.LLM_TYPE.GEMINI):
            raise ValueError('Unknow LLM type: {}'.format(llm_type))
        logger.warning('Using %s API as translation engine.', model)
        self.llm_type = llm_type
        self.model = model
        self.prompt = prompt
        self.history_size = history_size
        self.history_messages = []
        self.proxy = proxy
        self.use_json_result = use_json_result

    def _append_history_message(self, user_content: str, assistant_content: str):
        if not user_content or not assistant_content:
            return
        self.history_messages.extend([{
            'role': 'user',
            'content': user_content
        }, {
            'role': 'assistant',
            'content': assistant_content
        }])
        while (len(self.history_messages) > self.history_size * 2):
            self.history_messages.pop(0)

    def _translate_by_gpt(self, translation_task: TranslationTask):
        # https://platform.openai.com/docs/api-reference/chat/create?lang=python
        logger.debug('initiating OpenAI API client')
        client = OpenAI(http_client=DefaultHttpxClient(proxy=self.proxy))
        logger.debug('preparing request')
        system_prompt = 'You are a translation engine.'
        if self.use_json_result:
            system_prompt += " Output the answer in json format, key is translation."
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(self.history_messages)
        user_content = '{}: \n{}'.format(self.prompt, translation_task.transcribed_text)
        messages.append({'role': 'user', 'content': user_content})
        try:
            logger.debug('sending translation task to OpenAI API')
            completion = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=1,
                presence_penalty=1,
                messages=messages,
            )

            logger.debug('response received')
            translation_task.translated_text = completion.choices[0].message.content
            if self.use_json_result:
                translation_task.translated_text = _parse_json_completion(translation_task.translated_text)
        except (APITimeoutError, APIConnectionError) as e:
            logger.error(str(e))
            return
        if self.history_size:
            self._append_history_message(user_content, translation_task.translated_text)

    @staticmethod
    def _gpt_to_gemini(gpt_messages: list):
        gemini_messages = []
        for gpt_message in gpt_messages:
            gemini_message = {}
            gemini_message['role'] = gpt_message['role']
            if gemini_message['role'] == 'assistant':
                gemini_message['role'] = 'model'
            gemini_message['parts'] = [gpt_message['content']]
            gemini_messages.append(gemini_message)
        return gemini_messages

    def _translate_by_gemini(self, translation_task: TranslationTask):
        # https://ai.google.dev/tutorials/python_quickstart
        logger.debug('initiating Gemini API client')
        client = genai.GenerativeModel(self.model)
        logger.debug('preparing request')
        messages = self._gpt_to_gemini(self.history_messages)
        user_content = '{}: \n{}'.format(self.prompt, translation_task.transcribed_text)
        messages.append({'role': 'user', 'parts': [user_content]})
        config = genai.types.GenerationConfig(candidate_count=1, temperature=0)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        try:
            logger.debug('sending translation task to OpenAI API')
            response = client.generate_content(messages, generation_config=config, safety_settings=safety_settings)
            logger.debug('response received')
            translation_task.translated_text = response.text
            if self.use_json_result:
                translation_task.translated_text = _parse_json_completion(translation_task.translated_text)
        except (ValueError, InternalServerError, ResourceExhausted, TooManyRequests) as e:
            logger.error(str(e))
            return
        if self.history_size:
            self._append_history_message(user_content, translation_task.translated_text)

    def translate(self, translation_task: TranslationTask):
        with LogTime('translated task %d', translation_task.id):
            if self.llm_type == self.LLM_TYPE.GPT:
                self._translate_by_gpt(translation_task)
            elif self.llm_type == self.LLM_TYPE.GEMINI:
                self._translate_by_gemini(translation_task)
            else:
                raise ValueError('Unknow LLM type: {}'.format(self.llm_type))


class ParallelTranslator(LoopWorkerBase):
    PARALLEL_MAX_NUMBER = 10

    def __init__(self, llm_client: LLMClint, timeout: int, retry_if_translation_fails: bool):
        self.llm_client = llm_client
        self.timeout = timeout
        self.retry_if_translation_fails = retry_if_translation_fails
        self.processing_queue = deque()

    def trigger(self, translation_task: TranslationTask):
        self.processing_queue.append(translation_task)
        translation_task.start_time = datetime.utcnow()
        thread = threading.Thread(target=self.llm_client.translate, name=f'translator-instance-{translation_task.id}', args=(translation_task,))
        thread.daemon = True
        thread.start()

    def get_results(self):
        results = []
        while self.processing_queue and (self.processing_queue[0].translated_text or datetime.utcnow() -
                                         self.processing_queue[0].start_time > timedelta(seconds=self.timeout)):
            task = self.processing_queue.popleft()
            if task.translated_text:
                results.append(task)
            else:
                if self.retry_if_translation_fails:
                    self.trigger(task)
                else:
                    results.append(task)
                    logger.error('Translation timeout or failed: %s', task.transcribed_text)
        return results

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask]):
        while True:
            if not input_queue.empty() and len(self.processing_queue) < self.PARALLEL_MAX_NUMBER:
                task = input_queue.get()
                self.trigger(task)
            finished_tasks = self.get_results()
            for task in finished_tasks:
                task.phase = TranslationTask.Phase.TRANSLATED
                output_queue.put(task)
            time.sleep(0.1)


class SerialTranslator(LoopWorkerBase):

    def __init__(self, llm_client: LLMClint, timeout: int, retry_if_translation_fails: bool):
        self.llm_client = llm_client
        self.timeout = timeout
        self.retry_if_translation_fails = retry_if_translation_fails

    def trigger(self, translation_task: TranslationTask):
        translation_task.start_time = datetime.utcnow()
        thread = threading.Thread(target=self.llm_client.translate, name=f'translator-instance-{translation_task.id}', args=(translation_task,))
        thread.daemon = True
        thread.start()

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask], output_queue: queue.SimpleQueue[TranslationTask]):
        current_task = None
        while True:
            if current_task:
                if (current_task.translated_text or
                        datetime.utcnow() - current_task.start_time > timedelta(seconds=self.timeout)):
                    if not current_task.translated_text:
                        if self.retry_if_translation_fails:
                            logger.info('Translation timeout detected, retrying: %s', current_task.transcribed_text)
                            self.trigger(current_task)
                            continue
                        logger.error('Translation timeout or failed: %s', current_task.transcribed_text)
                    current_task.phase = TranslationTask.Phase.TRANSLATED
                    output_queue.put(current_task)
                    current_task = None

            if current_task is None and not input_queue.empty():
                current_task = input_queue.get()
                self.trigger(current_task)
            time.sleep(0.1)
