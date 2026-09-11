import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tyro

from stream_translator_gpt.utils.common import BOLD, ENDC, TranslationTask
from stream_translator_gpt.main import Config
from stream_translator_gpt.exporter.result_exporter import ResultExporter


class ResultExporterTimestampTest(unittest.TestCase):

    @staticmethod
    def _exporter(*, console_timestamps=False, file_timestamps=False, include_transcript=False):
        exporter = ResultExporter(
            cqhttp_url='',
            cqhttp_token='',
            discord_webhook_url='',
            telegram_token='',
            telegram_chat_id=0,
            output_file_path='',
            proxy='',
            output_whisper_result=include_transcript,
            output_timestamps=console_timestamps,
            output_file_timestamps=file_timestamps,
        )
        exporter.file_queue = queue.SimpleQueue()
        return exporter

    @staticmethod
    def _task(*, transcript='source text', translation='translated text'):
        task = TranslationTask(1, np.empty(0, dtype=np.float32), (3723.4, 3728.9))
        task.transcript = transcript
        task.translation = translation
        return task

    def _run_exporter(self, exporter, task):
        input_queue = queue.SimpleQueue()
        input_queue.put(task)
        input_queue.put(None)

        with mock.patch('stream_translator_gpt.exporter.result_exporter.write_console_utf8') as console_write:
            exporter.loop(input_queue)

        file_payload = exporter.file_queue.get_nowait()
        self.assertIsNone(exporter.file_queue.get_nowait())
        return console_write, file_payload

    def test_console_and_file_timestamp_switches_are_independent(self):
        cases = (
            (False, False, 'translated text', 'source text\ntranslated text'),
            (True, False, '01:02:03,4 --> 01:02:08,9 translated text',
             'source text\ntranslated text'),
            (False, True, 'translated text',
             '[01:02:03,4] source text\n[01:02:03,4] translated text'),
            (True, True, '01:02:03,4 --> 01:02:08,9 translated text',
             '[01:02:03,4] source text\n[01:02:03,4] translated text'),
        )
        for console_timestamps, file_timestamps, expected_console, expected_file in cases:
            with self.subTest(console_timestamps=console_timestamps, file_timestamps=file_timestamps):
                exporter = self._exporter(
                    console_timestamps=console_timestamps,
                    file_timestamps=file_timestamps,
                    include_transcript=True,
                )

                console_write, file_payload = self._run_exporter(exporter, self._task())

                console_write.assert_called_once_with(f'{BOLD}{expected_console}{ENDC}')
                self.assertEqual(file_payload, expected_file)

    def test_file_timestamp_uses_only_the_start_time_on_the_content_line(self):
        exporter = self._exporter(file_timestamps=True, include_transcript=True)

        _, file_payload = self._run_exporter(exporter, self._task())

        self.assertTrue(file_payload.startswith('[01:02:03,4] source text'))
        self.assertNotIn('[01:02:03,4]\n', file_payload)
        self.assertNotIn('01:02:08,9', file_payload)
        self.assertNotIn('-->', file_payload)
        self.assertEqual(file_payload.count('[01:02:03,4] '), 2)

    def test_file_timestamp_formats_transcript_and_translation_combinations(self):
        cases = (
            (False, None, 'translated text', '[01:02:03,4] translated text'),
            (True, 'source text', None, '[01:02:03,4] source text'),
            (True, 'source text', 'translated text',
             '[01:02:03,4] source text\n[01:02:03,4] translated text'),
            (False, None, 'translated line one\ntranslated line two',
             '[01:02:03,4] translated line one\n[01:02:03,4] translated line two'),
            (True, 'source text', 'translated line one\ntranslated line two',
             '[01:02:03,4] source text\n'
             '[01:02:03,4] translated line one\n'
             '[01:02:03,4] translated line two'),
        )
        for include_transcript, transcript, translation, expected in cases:
            with self.subTest(include_transcript=include_transcript, translation=translation):
                exporter = self._exporter(
                    file_timestamps=True,
                    include_transcript=include_transcript,
                )

                _, file_payload = self._run_exporter(
                    exporter,
                    self._task(transcript=transcript, translation=translation),
                )

                self.assertEqual(file_payload, expected)

    def test_timestamped_transcript_and_multiline_translation_are_written_to_file(self):
        exporter = self._exporter(file_timestamps=True, include_transcript=True)
        input_queue = queue.SimpleQueue()
        input_queue.put(self._task(translation='translated line one\ntranslated line two'))
        input_queue.put(None)

        with mock.patch('stream_translator_gpt.exporter.result_exporter.write_console_utf8'):
            exporter.loop(input_queue)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, 'results.txt')
            exporter._write_message_to_file(str(output_path))

            self.assertEqual(
                output_path.read_text(encoding='utf-8'),
                '[01:02:03,4] source text\n'
                '[01:02:03,4] translated line one\n'
                '[01:02:03,4] translated line two\n',
            )

    def test_file_writer_appends_to_an_existing_file_by_default(self):
        exporter = self._exporter()
        exporter.file_queue.put('new result')
        exporter.file_queue.put(None)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, 'results.txt')
            output_path.write_text('existing result\n\n', encoding='utf-8')

            exporter._write_message_to_file(str(output_path))

            self.assertEqual(
                output_path.read_text(encoding='utf-8'),
                'existing result\n\nnew result\n\n',
            )

    def test_file_writer_separates_appended_output_when_existing_file_has_no_trailing_newline(self):
        exporter = self._exporter()
        exporter.file_queue.put('new result')
        exporter.file_queue.put(None)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, 'results.txt')
            output_path.write_text('existing result', encoding='utf-8')

            exporter._write_message_to_file(str(output_path))

            self.assertEqual(
                output_path.read_text(encoding='utf-8'),
                'existing result\n\nnew result\n\n',
            )

    def test_timestamped_file_records_are_written_one_per_line(self):
        exporter = self._exporter(file_timestamps=True)
        exporter.file_queue.put('[00:00:01,2] first result')
        exporter.file_queue.put('[00:00:03,4] second result')
        exporter.file_queue.put(None)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, 'results.txt')

            exporter._write_message_to_file(str(output_path))

            self.assertEqual(
                output_path.read_text(encoding='utf-8'),
                '[00:00:01,2] first result\n[00:00:03,4] second result\n',
            )

    def test_remote_timestamp_format_still_uses_console_switch_and_is_independent_from_file(self):
        cases = (
            (
                False,
                True,
                'source text\ntranslated text',
                '[01:02:03,4] source text\n[01:02:03,4] translated text',
            ),
            (
                True,
                False,
                '01:02:03,4 --> 01:02:08,9\nsource text\ntranslated text',
                'source text\ntranslated text',
            ),
        )
        for console_timestamps, file_timestamps, expected_remote, expected_file in cases:
            with self.subTest(console_timestamps=console_timestamps, file_timestamps=file_timestamps):
                exporter = self._exporter(
                    console_timestamps=console_timestamps,
                    file_timestamps=file_timestamps,
                    include_transcript=True,
                )
                exporter.cqhttp_queue = queue.SimpleQueue()
                exporter.discord_queue = queue.SimpleQueue()
                exporter.telegram_queue = queue.SimpleQueue()

                _, file_payload = self._run_exporter(exporter, self._task())

                self.assertEqual(file_payload, expected_file)
                for remote_queue in (
                    exporter.cqhttp_queue,
                    exporter.discord_queue,
                    exporter.telegram_queue,
                ):
                    self.assertEqual(remote_queue.get_nowait(), expected_remote)
                    self.assertIsNone(remote_queue.get_nowait())


class ResultExporterTimestampCliTest(unittest.TestCase):

    def test_timestamp_options_are_disabled_by_default(self):
        config = tyro.cli(Config, args=['input'])

        self.assertFalse(config.output_timestamps)
        self.assertFalse(config.output_file_timestamps)

    def test_file_timestamp_option_does_not_enable_console_timestamps(self):
        config = tyro.cli(Config, args=['input', '--output-file-timestamps'])

        self.assertFalse(config.output_timestamps)
        self.assertTrue(config.output_file_timestamps)

    def test_console_timestamp_option_does_not_enable_file_timestamps(self):
        config = tyro.cli(Config, args=['input', '--output-timestamps'])

        self.assertTrue(config.output_timestamps)
        self.assertFalse(config.output_file_timestamps)

    def test_both_timestamp_options_can_be_enabled(self):
        config = tyro.cli(Config, args=[
            'input',
            '--output-timestamps',
            '--output-file-timestamps',
        ])

        self.assertTrue(config.output_timestamps)
        self.assertTrue(config.output_file_timestamps)


if __name__ == '__main__':
    unittest.main()
