import math
import unittest

from stream_translator_gpt.transcriber.modulate_transcriber import PartialUtteranceSplitter


class _FakeClock:

    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now


class PartialUtteranceSplitterTest(unittest.TestCase):

    CUTOFF = 5.0

    def _splitter(self):
        clock = _FakeClock()
        return PartialUtteranceSplitter(self.CUTOFF, clock=clock), clock

    def test_cutoff_is_strict_and_old_punctuation_is_not_used_retroactively(self):
        splitter, clock = self._splitter()

        self.assertIsNone(splitter.update_partial('alpha'))
        clock.now = 4.0
        self.assertIsNone(splitter.update_partial('alpha,'))

        # Equality is not overdue. The comma consequently becomes part of the
        # old preview and cannot be selected when the next update is overdue.
        clock.now = 5.0
        self.assertIsNone(splitter.update_partial('alpha, beta'))
        clock.now = 5.001
        self.assertIsNone(splitter.update_partial('alpha, beta grows'))

        # Only the first punctuation introduced after becoming overdue may cut.
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('alpha, beta grows.'), 'alpha, beta grows.')

    def test_new_internal_punctuation_after_cutoff_is_not_missed_by_replacement(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('hello world'))

        # Partials replace the full preview. A punctuation mark introduced only
        # after the cutoff is new even when a revision inserts it before the old
        # preview's end rather than appending it.
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('hello, world'), 'hello,')

    def test_each_partial_cuts_only_the_first_new_boundary_and_resets_the_clock(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('first'))

        clock.now = 6.0
        text = 'first, second; third.'
        self.assertEqual(splitter.update_partial(text), 'first,')
        self.assertEqual(splitter.started_at, 6.0)

        # The later punctuation was already present when the first cut reset
        # the clock, so neither it nor a newly appended mark can cut early.
        clock.now = 7.0
        self.assertIsNone(splitter.update_partial(text + ' fourth!'))
        clock.now = 11.0
        self.assertIsNone(splitter.update_partial(text + ' fourth! waiting'))

        # At equality the clock still isn't overdue. After it becomes overdue,
        # a newly introduced question mark commits the whole current remainder.
        clock.now = 11.001
        self.assertEqual(splitter.update_partial(text + ' fourth! waiting?'),
                         'second; third. fourth! waiting?')

    def test_full_width_half_width_and_multilingual_punctuation(self):
        punctuation_cases = [
            ',',
            '.',
            '!',
            '?',
            ';',
            ':',
            '\uff0e',  # full-width full stop
            '\uff0c',  # full-width comma
            '\uff01',
            '\uff1f',
            '\uff1b',
            '\uff1a',
            '\u3002',  # ideographic full stop
            '\u3001',  # ideographic comma
            '\uff61',  # half-width ideographic full stop
            '\uff64',  # half-width ideographic comma
            '\u2026',  # horizontal ellipsis
            '\u22ef',  # midline ellipsis
            '\u061f',  # Arabic question mark
            '\u060c',  # Arabic comma
            '\u061b',  # Arabic semicolon
            '\u0964',  # Devanagari danda
            '\u0965',  # Devanagari double danda
            '\u203c',
            '\u2047',
            '\u2048',
            '\u2049',
            '...',
        ]

        for punctuation in punctuation_cases:
            with self.subTest(punctuation=punctuation.encode('unicode_escape').decode('ascii')):
                splitter, clock = self._splitter()
                self.assertIsNone(splitter.update_partial('word'))
                clock.now = 6.0
                self.assertEqual(splitter.update_partial(f'word{punctuation}tail'), f'word{punctuation}')

    def test_consecutive_punctuation_and_closing_quotes_belong_to_previous_segment(self):
        cases = [
            ('What?!\u201d next', 'What?!\u201d'),
            ('Really\uff01\uff1f\u300d\u300f next', 'Really\uff01\uff1f\u300d\u300f'),
            ('Wait......\u2019 next', 'Wait......\u2019'),
            ('What\u061f!\u201d next', 'What\u061f!\u201d'),
        ]

        for text, expected in cases:
            with self.subTest(text=text.encode('unicode_escape').decode('ascii')):
                splitter, clock = self._splitter()
                self.assertIsNone(splitter.update_partial('prefix'))
                clock.now = 6.0
                candidate = 'prefix ' + text
                expected_candidate = 'prefix ' + expected
                self.assertEqual(splitter.update_partial(candidate), expected_candidate)

    def test_numeric_separators_do_not_split_numbers(self):
        numeric_cases = [
            '3.14',
            '1,000',
            '12:30',
            '\uff13\uff0e\uff11\uff14',
            '\uff11\uff0c\uff10\uff10\uff10',
            '\uff11\uff12\uff1a\uff13\uff10',
        ]

        for number in numeric_cases:
            with self.subTest(number=number.encode('unicode_escape').decode('ascii')):
                splitter, clock = self._splitter()
                self.assertIsNone(splitter.update_partial('value '))
                clock.now = 6.0
                text = 'value ' + number
                self.assertIsNone(splitter.update_partial(text))
                clock.now = 7.0
                self.assertEqual(splitter.update_partial(text + '!'), text + '!')

    def test_empty_or_whitespace_partial_resets_timeout_baseline(self):
        for empty in ('', '   ', '\t\n'):
            with self.subTest(empty=repr(empty)):
                splitter, clock = self._splitter()
                self.assertIsNone(splitter.update_partial('old text'))
                clock.now = 6.0
                self.assertIsNone(splitter.update_partial(empty))
                self.assertIsNone(splitter.started_at)
                self.assertIsNone(splitter.punctuation_search_offset)

                # A new non-empty replacement starts a fresh interval even if it
                # already contains punctuation.
                clock.now = 100.0
                self.assertIsNone(splitter.update_partial('new,'))
                self.assertEqual(splitter.started_at, 100.0)
                clock.now = 105.0
                self.assertIsNone(splitter.update_partial('new, text'))
                clock.now = 105.001
                self.assertEqual(splitter.update_partial('new, text!'), 'new, text!')

    def test_silence_after_a_final_does_not_consume_the_next_sentence_timeout(self):
        clock = _FakeClock()
        splitter = PartialUtteranceSplitter(10.0, clock=clock)

        self.assertIsNone(splitter.update_partial('previous sentence.'))
        self.assertEqual(splitter.finalize('previous sentence.'), 'previous sentence.')
        self.assertIsNone(splitter.started_at)

        # Nine seconds of silence pass after the prior final. The next
        # sentence receives a fresh ten-second budget from its first text.
        clock.now = 9.0
        self.assertIsNone(splitter.update_partial('next sentence begins'))
        self.assertEqual(splitter.started_at, 9.0)
        clock.now = 19.0
        self.assertIsNone(splitter.update_partial('next sentence begins and grows'))

        clock.now = 19.001
        self.assertEqual(splitter.update_partial('next sentence begins and grows,'),
                         'next sentence begins and grows,')

    def test_silence_after_a_forced_cut_waits_for_the_next_uncommitted_text(self):
        clock = _FakeClock()
        splitter = PartialUtteranceSplitter(10.0, clock=clock)

        self.assertIsNone(splitter.update_partial('previous sentence'))
        clock.now = 10.001
        self.assertEqual(splitter.update_partial('previous sentence.'), 'previous sentence.')
        self.assertIsNone(splitter.started_at)

        # The full replacement still includes the committed prefix. Only the
        # first new, uncommitted characters after nine seconds of silence start
        # the next sentence's timer.
        clock.now = 19.001
        self.assertIsNone(splitter.update_partial('previous sentence. next sentence begins'))
        self.assertEqual(splitter.started_at, 19.001)
        clock.now = 29.001
        self.assertIsNone(splitter.update_partial('previous sentence. next sentence begins and grows'))

        clock.now = 29.002
        self.assertEqual(splitter.update_partial('previous sentence. next sentence begins and grows,'),
                         'next sentence begins and grows,')

    def test_partial_updates_replace_instead_of_append(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('I lie about ca'))
        clock.now = 2.0
        self.assertIsNone(splitter.update_partial('I like ca'))
        clock.now = 4.0
        self.assertIsNone(splitter.update_partial('I love ca'))
        clock.now = 6.0

        self.assertEqual(splitter.update_partial('I love cats,'), 'I love cats,')
        self.assertNotIn('lie', splitter.committed_source_text)
        self.assertNotIn('like', splitter.committed_source_text)

    def test_uncommitted_rollback_keeps_the_original_timeout_baseline(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('abcdef'))
        clock.now = 2.0
        self.assertIsNone(splitter.update_partial('abc'))
        self.assertEqual(splitter.started_at, 0.0)

        clock.now = 6.0
        self.assertEqual(splitter.update_partial('abc,'), 'abc,')

    def test_suffix_can_be_rewritten_after_a_forced_prefix(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('Hello'))
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('Hello, cat'), 'Hello,')

        clock.now = 7.0
        self.assertIsNone(splitter.update_partial('Hello, cats'))
        clock.now = 8.0
        self.assertIsNone(splitter.update_partial('Hello, dog'))
        clock.now = 12.0
        self.assertEqual(splitter.update_partial('Hello, dog.'), 'dog.')

    def test_finalize_returns_only_uncommitted_authoritative_remainder(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('Hello'))
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('Hello, provisional'), 'Hello,')

        self.assertEqual(splitter.finalize('Hello, final world.'), 'final world.')
        self.assertEqual(splitter.preview_text, '')
        self.assertEqual(splitter.committed_offset, 0)
        self.assertIsNone(splitter.started_at)

    def test_finalize_does_not_repeat_an_exact_committed_prefix(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('Done'))
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('Done.'), 'Done.')

        self.assertIsNone(splitter.finalize('Done.'))

    def test_finalize_without_a_forced_segment_uses_the_final_replacement(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('I like cats'))
        clock.now = 2.0
        self.assertIsNone(splitter.update_partial('I love dogs'))

        self.assertEqual(splitter.finalize('I love dogs.'), 'I love dogs.')

    def test_prefix_conflict_stops_force_splitting_and_final_is_conservative(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('I like'))
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('I like, tail'), 'I like,')

        # This revision crosses the already published boundary. Even though it
        # contains punctuation and is overdue, it must not be force-published.
        clock.now = 12.0
        self.assertIsNone(splitter.update_partial('I love cats.'))
        self.assertTrue(splitter.prefix_conflicted)
        clock.now = 20.0
        self.assertIsNone(splitter.update_partial('I love cats! More?'))

        # The portion conservatively mapped onto the already-published prefix is
        # consumed, while the uncommitted final remainder is still delivered.
        self.assertEqual(splitter.finalize('I love cats! More? Tail.'),
                         'cats! More? Tail.')

    def test_direct_final_revision_across_committed_prefix_is_not_repeated(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('I like'))
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('I like,'), 'I like,')

        # The revised lexical prefix cannot retract the already-published text,
        # but the authoritative final tail must not be swallowed with it.
        self.assertEqual(splitter.finalize('I love cats.'), 'cats.')

    def test_conflicted_prefix_can_realign_before_final(self):
        splitter, clock = self._splitter()
        self.assertIsNone(splitter.update_partial('I like'))
        clock.now = 6.0
        self.assertEqual(splitter.update_partial('I like, tail'), 'I like,')

        clock.now = 7.0
        self.assertIsNone(splitter.update_partial('I love'))
        self.assertTrue(splitter.prefix_conflicted)
        clock.now = 8.0
        self.assertIsNone(splitter.update_partial('I like, cats'))
        self.assertFalse(splitter.prefix_conflicted)

        self.assertEqual(splitter.finalize('I like, cats.'), 'cats.')

    def test_invalid_cutoffs_are_rejected(self):
        for cutoff in (0, -1, math.inf, -math.inf, math.nan):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    PartialUtteranceSplitter(cutoff)


if __name__ == '__main__':
    unittest.main()
