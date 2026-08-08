import subprocess
import sys
import unittest

from utils.term_utils import (
    build_modulate_custom_terms,
    extract_stream_keyword_terms,
    extract_translation_source_terms,
    split_term_list,
)


class TermUtilsTest(unittest.TestCase):

    def test_term_list_supports_common_multilingual_separators(self):
        terms = split_term_list(
            'alpha, beta，gamma、delta; epsilon；zeta|eta｜theta\n'
            'iota\r\nkappa\tlambda•mu‣nu⁃xi◦omicron\u2028pi\u2029rho, alpha'
        )

        self.assertEqual(terms, [
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
            'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron',
            'pi', 'rho',
        ])

    def test_term_list_preserves_spaces_middle_dots_slashes_and_bracket_content(self):
        terms = split_term_list(
            'ソードアート・オンライン, ACME Live, A/B, '
            'メン限（説明, with comma）、次'
        )

        self.assertEqual(terms, [
            'ソードアート・オンライン', 'ACME Live', 'A/B',
            'メン限（説明, with comma）', '次',
        ])

    def test_translation_keys_support_common_mapping_and_item_separators(self):
        mappings = (
            'クリオネ -> 海天使，タマオネ：蛋天使; A=>甲|B⇒乙\n'
            'C＝丙、D ➜ 丁；E:戊, Re:ゼロ -> Re:Zero, malformed, empty -> '
        )

        self.assertEqual(
            extract_translation_source_terms(mappings),
            ['クリオネ', 'タマオネ', 'A', 'B', 'C', 'D', 'E', 'Re:ゼロ'],
        )

    def test_stream_keywords_drop_balanced_nested_annotations_only(self):
        keywords = (
            'さくたん, メン限（メンバー,限定）, '
            'さくもに【朝活（早朝）】、むーちゃん[飼い猫]; '
            '作品《説明、補足》|作品（2024）版|【推しの子】|未閉じ（注釈,次'
        )

        self.assertEqual(extract_stream_keyword_terms(keywords), [
            'さくたん', 'メン限', 'さくもに', 'むーちゃん', '作品《説明、補足》',
            '作品（2024）版', '【推しの子】', '未閉じ（注釈,次',
        ])

    def test_combiner_keeps_distinct_spellings_and_deduplicates_across_sources(self):
        combined = build_modulate_custom_terms(
            '結城さくな,さくたん,ＡＣＭＥ',
            'クリオネ -> 海天使; タマオネ -> 蛋天使; 結城さくな -> 保留',
            'さくたん, メン限（メンバーシップ限定）, ACME（別表記）',
            'メン限、クリオネ、神楽めあ',
        )

        self.assertEqual(
            combined,
            '結城さくな,さくたん,ＡＣＭＥ,クリオネ,タマオネ,メン限,ACME,神楽めあ',
        )

    def test_default_script_inputs_keep_existing_order_and_append_mapping_keys(self):
        combined = build_modulate_custom_terms(
            '結城さくな,さくたん,こんさくな,おつさくな,神楽めあ,犬山たまき,海月雲ろあ,'
            'むーちゃん,むーたん,むーちゃお,湊あくあ,憂姫はぐれ,吉田音,'
            'ソードアート・オンライン,結城明日奈,メン限,さくもに',
            'クリオネ -> 海天使; タマオネ -> 蛋天使',
            'さくたん, メン限（メンバーシップ限定）, さくもに（朝活配信）, むーちゃん（飼い猫）',
            '結城さくな、神楽めあ、犬山たまき、海月雲ろあ、湊あくあ、むーちゃん、むーたん、'
            'むーちゃお、こんさくな、おつさくな',
        )

        self.assertEqual(
            combined,
            '結城さくな,さくたん,こんさくな,おつさくな,神楽めあ,犬山たまき,海月雲ろあ,'
            'むーちゃん,むーたん,むーちゃお,湊あくあ,憂姫はぐれ,吉田音,'
            'ソードアート・オンライン,結城明日奈,メン限,さくもに,クリオネ,タマオネ',
        )
        self.assertNotIn('海天使', combined)
        self.assertNotIn('蛋天使', combined)
        self.assertNotIn('（', combined)

    def test_module_cli_matches_library_output(self):
        args = [
            'base,duplicate',
            'クリオネ -> 海天使; duplicate -> ignored',
            'keyword（annotation）',
            'preserve、base',
        ]
        expected = build_modulate_custom_terms(*args)

        result = subprocess.run(
            [sys.executable, '-m', 'utils.term_utils', *args],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
        )

        self.assertEqual(result.stdout, expected)
        self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
