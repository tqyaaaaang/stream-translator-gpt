import unittest

import tyro

from stream_translator_gpt import filters
from stream_translator_gpt.audio_transcriber import AudioTranscriber
from stream_translator_gpt.main import Config, _preprocess_deprecated_flags


class _TestTranscriber(AudioTranscriber):

    def transcribe(self, audio, history_context=None):
        return '', None


def _transcriber(*, language='auto', transcription_filters='', language_filters=None):
    return _TestTranscriber(
        language=language,
        transcription_filters=transcription_filters,
        language_filters=language_filters,
        print_result=False,
        output_timestamps=False,
        use_history_context=False,
        transcription_keywords=None,
    )


class LanguageFiltersCliTest(unittest.TestCase):

    @staticmethod
    def _parse(*args):
        return tyro.cli(Config, args=_preprocess_deprecated_flags(args))

    def test_language_filters_are_disabled_by_default(self):
        config = self._parse('https://example.test/live')

        self.assertIsNone(config.language_filters)

    def test_language_filters_accept_a_comma_separated_list(self):
        config = self._parse(
            'https://example.test/live',
            '--language-filters',
            'en,ja,zh-tw',
        )

        self.assertEqual(config.language_filters, 'en,ja,zh-tw')

    def test_legacy_disable_flag_is_a_deprecated_noop(self):
        with self.assertLogs('main', level='WARNING') as logs:
            processed = _preprocess_deprecated_flags((
                'https://example.test/live',
                '--no-language-based-filter',
                '--format',
                '251',
            ))

        self.assertEqual(processed, ['https://example.test/live', '--format', '251'])
        self.assertTrue(any('deprecated' in message.lower() for message in logs.output))
        self.assertIsNone(tyro.cli(Config, args=processed).language_filters)

    def test_legacy_enable_flag_uses_the_final_asr_language(self):
        cases = (
            (('https://example.test/live', '--language-based-filter'), 'en,ja,zh'),
            (('--language_based_filter', 'https://example.test/live', '--language', 'Japanese'), 'ja'),
            (('https://example.test/live', '--language-based-filter', '--language=zh-tw'), 'zh'),
            (('https://example.test/live', '--language', 'ru', '--language-based-filter'), None),
        )
        for args, expected in cases:
            with self.subTest(args=args), self.assertLogs('main', level='WARNING'):
                processed = _preprocess_deprecated_flags(args)
            self.assertEqual(tyro.cli(Config, args=processed).language_filters, expected)


class LanguageFilterResolutionTest(unittest.TestCase):

    def test_missing_or_empty_list_selects_no_filters(self):
        self.assertEqual(filters.get_language_filters(None), [])
        self.assertEqual(filters.get_language_filters(''), [])
        self.assertEqual(filters.get_language_filters('  ,  '), [])

    def test_aliases_preserve_first_seen_order_and_are_deduplicated(self):
        selected = filters.get_language_filters(
            'Japanese, en, zh-TW, english, JA, chinese, zh-cn'
        )

        self.assertEqual(selected, [
            filters.japanese_filter,
            filters.english_filter,
            filters.chinese_filter,
        ])

    def test_unknown_languages_warn_and_do_not_hide_valid_languages(self):
        with self.assertLogs('main', level='WARNING') as logs:
            selected = filters.get_language_filters('xx, English, klingon')

        self.assertEqual(selected, [filters.english_filter])
        messages = '\n'.join(logs.output).lower()
        self.assertIn('xx', messages)
        self.assertIn('klingon', messages)


class LanguageFilterChainTest(unittest.TestCase):

    def test_asr_language_does_not_enable_a_language_filter_by_default(self):
        transcriber = _transcriber(language='ja')

        self.assertEqual(transcriber.filter_chain, [])

    def test_explicit_filters_are_independent_of_asr_language(self):
        transcriber = _transcriber(language='en', language_filters='ja,zh')

        self.assertEqual(transcriber.filter_chain, [
            filters.japanese_filter,
            filters.chinese_filter,
        ])

    def test_language_and_transcription_filters_share_ordered_deduplication(self):
        transcriber = _transcriber(
            transcription_filters='emoji_filter,japanese_filter',
            language_filters='ja,en,ja',
        )

        self.assertEqual(transcriber.filter_chain, [
            filters.emoji_filter,
            filters.japanese_filter,
            filters.english_filter,
        ])


if __name__ == '__main__':
    unittest.main()
