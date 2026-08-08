import logging
import os
import queue
from urllib.parse import urlencode

import requests

from .common import TranslationTask, LoopWorkerBase, sec2str, start_daemon_thread, BOLD, ENDC, write_console_utf8


logger = logging.getLogger('main')


def _log_http_status(channel: str, status_code: int):
    level = logging.WARNING if status_code >= 400 else logging.DEBUG
    logger.log(level, '%s export returned HTTP status %s', channel, status_code)


def _format_translation_result(task: TranslationTask) -> str:
    """Keep rolling translation context internal and export only the current translation."""
    return task.translation


class ResultExporter(LoopWorkerBase):

    def __init__(self, cqhttp_url: str, cqhttp_token: str, discord_webhook_url: str, telegram_token: str,
                 telegram_chat_id: int, output_file_path: str, proxy: str, output_whisper_result: bool,
                 output_timestamps: bool) -> None:
        self.proxies = {"http": proxy, "https": proxy} if proxy else None
        self.cqhttp_queue = None
        self.discord_queue = None
        self.telegram_queue = None
        self.file_queue = None
        self.output_whisper_result = output_whisper_result
        self.output_timestamps = output_timestamps

        if cqhttp_url:
            logger.info('CQHTTP export channel enabled')
            self.cqhttp_queue = queue.SimpleQueue()
            start_daemon_thread(self._send_message_to_cqhttp, url=cqhttp_url, token=cqhttp_token)
        if discord_webhook_url:
            logger.info('Discord export channel enabled')
            self.discord_queue = queue.SimpleQueue()
            start_daemon_thread(self._send_message_to_discord, webhook_url=discord_webhook_url)
        if telegram_token and telegram_chat_id:
            logger.info('Telegram export channel enabled')
            self.telegram_queue = queue.SimpleQueue()
            start_daemon_thread(self._send_message_to_telegram, token=telegram_token, chat_id=telegram_chat_id)
        if output_file_path:
            logger.info('File export channel enabled')
            self.file_queue = queue.SimpleQueue()
            start_daemon_thread(self._write_message_to_file, file_path=output_file_path)

    def _send_message_to_cqhttp(self, url: str, token: str):
        headers = {'Authorization': f'Bearer {token}'} if token else None
        while True:
            text = self.cqhttp_queue.get()
            if text is None:
                logger.debug('CQHTTP export channel stopped')
                break
            data = {'message': text}
            try:
                response = requests.post(url, headers=headers, data=data, timeout=10, proxies=self.proxies)
                _log_http_status('CQHTTP', response.status_code)
            except Exception as e:
                logger.error('CQHTTP export failed (%s)', type(e).__name__)

    def _send_message_to_discord(self, webhook_url: str):
        while True:
            text = self.discord_queue.get()
            if text is None:
                logger.debug('Discord export channel stopped')
                break
            for sub_text in text.split('\n') + ['\u200b']:
                data = {'content': sub_text}
                try:
                    response = requests.post(webhook_url, json=data, timeout=10, proxies=self.proxies)
                    _log_http_status('Discord', response.status_code)
                except Exception as e:
                    logger.error('Discord export failed (%s)', type(e).__name__)

    def _send_message_to_telegram(self, token: str, chat_id: int):
        while True:
            text = self.telegram_queue.get()
            if text is None:
                logger.debug('Telegram export channel stopped')
                break
            url = f'https://api.telegram.org/bot{token}/sendMessage'
            params = urlencode({'chat_id': chat_id, 'text': text})
            try:
                response = requests.post(f'{url}?{params}', timeout=10, proxies=self.proxies)
                _log_http_status('Telegram', response.status_code)
            except Exception as e:
                logger.error('Telegram export failed (%s)', type(e).__name__)

    def _write_message_to_file(self, file_path: str):
        if file_path:
            if os.path.exists(file_path):
                logger.debug('Replacing the existing output file')
                os.remove(file_path)
        while True:
            text = self.file_queue.get()
            if text is None:
                logger.debug('File export channel stopped')
                break
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n\n')
            logger.debug('Appended one result to the output file')

    def loop(self, input_queue: queue.SimpleQueue[TranslationTask]):
        while True:
            task = input_queue.get()
            if task is None:
                if self.cqhttp_queue:
                    self.cqhttp_queue.put(None)
                if self.discord_queue:
                    self.discord_queue.put(None)
                if self.telegram_queue:
                    self.telegram_queue.put(None)
                if self.file_queue:
                    self.file_queue.put(None)
                logger.debug('All export channels were signalled to stop')
                break
            timestamp_text = f'{sec2str(task.time_range[0])} --> {sec2str(task.time_range[1])}'
            text_to_send = (task.transcript + '\n') if self.output_whisper_result else ''
            if self.output_timestamps:
                text_to_send = timestamp_text + '\n' + text_to_send
            if task.translation:
                formatted_translation = _format_translation_result(task)
                text_to_print = formatted_translation
                if self.output_timestamps:
                    text_to_print = timestamp_text + ' ' + text_to_print
                text_to_print = text_to_print.strip()
                write_console_utf8(f'{BOLD}{text_to_print}{ENDC}')
                text_to_send += formatted_translation
            text_to_send = text_to_send.strip()
            if self.cqhttp_queue:
                self.cqhttp_queue.put(text_to_send)
            if self.discord_queue:
                self.discord_queue.put(text_to_send)
            if self.telegram_queue:
                self.telegram_queue.put(text_to_send)
            if self.file_queue:
                self.file_queue.put(text_to_send)
