import json
import math
import ssl
import unittest
from unittest import mock
from urllib.parse import parse_qs, parse_qsl, quote, quote_plus, urlsplit

from stream_translator_gpt.transcriber.elevenlabs_transcriber import (
    ElevenLabsScribeTranscriber, _SendPacer, normalize_keyterms,
)
from stream_translator_gpt.utils.common import SAMPLE_RATE


class _FakeBoundaryDetector:

    def feed(self, frame):
        return False

    def mark_boundary(self):
        pass


def _transcriber(**overrides):
    common_args = {
        'api_key': 'test key with +?/ characters',
        'boundary_detector': _FakeBoundaryDetector(),
        'language': None,
        'transcription_filters': '',
        'language_filters': None,
        'print_result': False,
        'output_timestamps': False,
        'use_history_context': False,
        'transcription_keywords': None,
        'prefix_retention_length': 0,
    }
    common_args.update(overrides)
    return ElevenLabsScribeTranscriber(**common_args)


class _FakeClock:

    def __init__(self, now=100.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _FakeStopEvent:

    def __init__(self, clock):
        self.clock = clock
        self.wait_calls = []
        self.stopped = False
        self.cancel_on_wait = False
        self.oversleep = 0.0

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.wait_calls.append(seconds)
        if self.cancel_on_wait:
            self.stopped = True
        else:
            self.clock.advance(seconds + self.oversleep)
        return self.stopped


class ElevenLabsKeytermOptionsTest(unittest.TestCase):

    def test_empty_inputs_produce_no_keyterms(self):
        for value in (None, '', '   ', ',', ', , ,', ' \t,\n, \t'):
            with self.subTest(value=value):
                self.assertEqual(normalize_keyterms(value), [])

    def test_terms_are_trimmed_but_duplicates_and_order_are_preserved(self):
        self.assertEqual(normalize_keyterms(' さくな , , OpenAI,さくな, GPT-4o  ,'),
                         ['さくな', 'OpenAI', 'さくな', 'GPT-4o'])

    def test_exactly_fifty_terms_do_not_warn(self):
        terms = [f'term-{index}' for index in range(50)]
        with self.assertNoLogs('main', level='WARNING'):
            self.assertEqual(normalize_keyterms(','.join(terms)), terms)

    def test_fifty_one_terms_warn_with_full_list_and_keep_first_fifty_in_order(self):
        terms = ['repeated', 'repeated'] + [f'term-{index}' for index in range(49)]
        with self.assertLogs('main', level='WARNING') as logs:
            result = normalize_keyterms(','.join(terms))
        self.assertEqual(result, terms[:50])
        self.assertEqual(len(logs.output), 1)
        self.assertIn('51 keyterms', logs.output[0])
        self.assertIn('first 50', logs.output[0])
        self.assertIn(repr(terms), logs.output[0])
        self.assertEqual(result[:2], ['repeated', 'repeated'])

    def test_twenty_unicode_characters_are_allowed(self):
        term = 'あ' * 20
        self.assertEqual(normalize_keyterms(f' {term} '), [term])

    def test_long_selected_term_is_rejected(self):
        for terms in (['a' * 21], ['valid'] * 49 + ['あ' * 21]):
            with self.subTest(terms=terms), self.assertRaisesRegex(ValueError, 'at most 20 characters'):
                normalize_keyterms(','.join(terms))

    def test_long_discarded_term_is_logged_but_does_not_reject_first_fifty(self):
        terms = [f'term-{index}' for index in range(50)] + ['あ' * 21]
        with self.assertLogs('main', level='WARNING') as logs:
            result = normalize_keyterms(','.join(terms))
        self.assertEqual(result, terms[:50])
        self.assertIn(repr(terms), logs.output[0])


class ElevenLabsConnectionOptionsTest(unittest.TestCase):

    def test_default_url_has_fixed_model_manual_commits_and_timestamps(self):
        parsed = urlsplit(_transcriber()._connection_url())
        self.assertEqual(parsed.scheme, 'wss')
        self.assertEqual(parsed.netloc, 'api.elevenlabs.io')
        self.assertEqual(parsed.path, '/v1/speech-to-text/realtime')
        self.assertEqual(parse_qs(parsed.query), {
            'model_id': ['scribe_v2_realtime'],
            'audio_format': ['pcm_16000'],
            'commit_strategy': ['manual'],
            'include_timestamps': ['true'],
        })

    def test_empty_keyterms_are_completely_omitted_from_query(self):
        for keywords in (None, '', ' ', ', , ,'):
            with self.subTest(keywords=keywords):
                query = urlsplit(_transcriber(transcription_keywords=keywords)._connection_url()).query
                self.assertNotIn('keyterms', query)
                self.assertNotIn('keyterms', parse_qs(query, keep_blank_values=True))

    def test_nonempty_keyterms_are_repeated_unicode_encoded_query_parameters(self):
        terms = ['さくな', 'A+B & C', 'さくな']
        query = urlsplit(_transcriber(transcription_keywords=','.join(terms))._connection_url()).query
        self.assertEqual([value for key, value in parse_qsl(query) if key == 'keyterms'], terms)
        self.assertEqual(query.count('keyterms='), len(terms))
        for term in terms:
            self.assertIn('keyterms=' + quote_plus(term), query)
        self.assertNotIn('さくな', query)

    def test_auto_language_omits_language_code_but_explicit_language_is_sent(self):
        for language in (None, '', 'auto', 'AUTO'):
            with self.subTest(language=language):
                query = parse_qs(urlsplit(_transcriber(language=language)._connection_url()).query)
                self.assertNotIn('language_code', query)
        query = parse_qs(urlsplit(_transcriber(language='ja')._connection_url()).query)
        self.assertEqual(query['language_code'], ['ja'])

    def test_api_key_is_trimmed_and_passed_only_as_header(self):
        key = 'test key +?/保密'
        transcriber = _transcriber(api_key='  ' + key + '  ')
        kwargs = transcriber._connection_kwargs()
        self.assertEqual(kwargs['additional_headers'], {'xi-api-key': key})
        url = transcriber._connection_url()
        for representation in (key, quote(key, safe=''), quote_plus(key)):
            self.assertNotIn(representation, url)
        self.assertNotIn('xi-api-key', url)
        self.assertNotIn('api_key', url)

    def test_missing_key_is_rejected(self):
        for key in (None, '', ' \t\n'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'API key is required'):
                _transcriber(api_key=key)

    def test_server_error_redacts_raw_encoded_and_double_encoded_key(self):
        transcriber = _transcriber(api_key='test key +?/保密')
        variants = {transcriber.api_key, quote(transcriber.api_key, safe=''), quote_plus(transcriber.api_key)}
        variants.update(quote(value, safe='') for value in list(variants))
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(transcriber._safe_error(f'Bad credential: {value}; denied'),
                                 'Bad credential: <redacted>; denied')

    def test_server_error_is_bounded_and_handles_missing_message(self):
        transcriber = _transcriber()
        self.assertEqual(transcriber._safe_error(None), 'unknown server error')
        self.assertEqual(len(transcriber._safe_error('x' * 1000)), 500)

    def test_explicit_proxy_is_forwarded_and_missing_proxy_leaves_library_default(self):
        proxy = 'http://proxy.example.test:8080'
        self.assertEqual(_transcriber(proxy=proxy)._connection_kwargs()['proxy'], proxy)
        self.assertNotIn('proxy', _transcriber(proxy=None)._connection_kwargs())

    def test_tls_verification_is_default_and_can_be_explicitly_disabled(self):
        self.assertNotIn('ssl', _transcriber()._connection_kwargs())
        context = _transcriber(verify_ssl=False)._connection_kwargs()['ssl']
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertFalse(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_NONE)

    def test_history_flag_does_not_send_previous_text_or_change_connection_options(self):
        without_history = _transcriber(use_history_context=False)
        with self.assertLogs('main', level='INFO') as logs:
            with_history = _transcriber(use_history_context=True)
        self.assertTrue(any('does not inject additional text prompts' in text for text in logs.output))
        self.assertEqual(with_history._connection_url(), without_history._connection_url())
        self.assertEqual(with_history._connection_kwargs(), without_history._connection_kwargs())
        for transcriber in (without_history, with_history):
            websocket = mock.Mock()
            self.assertTrue(transcriber._send(websocket, b'', True))
            self.assertEqual(json.loads(websocket.send.call_args.args[0]), {
                'message_type': 'input_audio_chunk', 'audio_base_64': '',
                'sample_rate': SAMPLE_RATE, 'commit': True,
            })

    def test_continuous_backend_does_not_accept_legacy_chunk_transcribe_calls(self):
        with self.assertRaisesRegex(NotImplementedError, 'continuous stream'):
            _transcriber().transcribe([], history_context='previous sentence')

    def test_pacer_and_timeout_settings_reject_nonpositive_or_nonfinite_values(self):
        for option in ('max_audio_send_rate', 'final_timeout', 'idle_timeout'):
            for value in (0, -1, math.nan, math.inf):
                with self.subTest(option=option, value=value), self.assertRaises(ValueError):
                    _transcriber(**{option: value})


class ElevenLabsSendPacerTest(unittest.TestCase):

    def test_first_send_is_immediate_and_following_sends_follow_one_or_five_x_rate(self):
        for rate in (1.0, 5.0):
            with self.subTest(rate=rate):
                clock = _FakeClock()
                stop = _FakeStopEvent(clock)
                pacer = _SendPacer(rate, stop, clock)
                self.assertTrue(pacer.wait())
                self.assertEqual(stop.wait_calls, [])
                for index in range(3):
                    pacer.record(SAMPLE_RATE)
                    self.assertTrue(pacer.wait())
                    self.assertAlmostEqual(clock(), 100.0 + (index + 1) / rate)
                self.assertEqual(len(stop.wait_calls), 3)
                for delay in stop.wait_calls:
                    self.assertAlmostEqual(delay, 1.0 / rate)

    def test_short_scheduler_oversleep_does_not_accumulate_per_packet(self):
        for rate in (1.0, 5.0):
            with self.subTest(rate=rate):
                clock = _FakeClock()
                stop = _FakeStopEvent(clock)
                stop.oversleep = 0.005
                pacer = _SendPacer(rate, stop, clock)
                samples = 2048
                interval = samples / SAMPLE_RATE / rate
                self.assertTrue(pacer.wait())
                for index in range(20):
                    pacer.record(samples)
                    self.assertTrue(pacer.wait())
                    self.assertAlmostEqual(clock(), 100.0 + (index + 1) * interval + stop.oversleep)
                self.assertAlmostEqual(stop.wait_calls[0], interval)
                for delay in stop.wait_calls[1:]:
                    self.assertAlmostEqual(delay, interval - stop.oversleep)

    def test_long_scheduler_oversleep_starts_fresh_schedule(self):
        for rate in (1.0, 5.0):
            with self.subTest(rate=rate):
                clock = _FakeClock()
                stop = _FakeStopEvent(clock)
                stop.oversleep = 2.0
                pacer = _SendPacer(rate, stop, clock)
                pacer.record(SAMPLE_RATE)
                self.assertTrue(pacer.wait())
                resumed_at = clock()
                stop.oversleep = 0.0
                pacer.record(SAMPLE_RATE)
                self.assertTrue(pacer.wait())
                self.assertAlmostEqual(clock(), resumed_at + 1.0 / rate)

    def test_idle_before_first_audio_does_not_delay_or_accelerate_following_send(self):
        for rate in (1.0, 5.0):
            with self.subTest(rate=rate):
                clock = _FakeClock()
                stop = _FakeStopEvent(clock)
                pacer = _SendPacer(rate, stop, clock)
                clock.advance(60)
                self.assertTrue(pacer.wait())
                self.assertEqual(stop.wait_calls, [])
                pacer.record(SAMPLE_RATE)
                self.assertTrue(pacer.wait())
                self.assertAlmostEqual(clock(), 160.0 + 1.0 / rate)
                self.assertEqual(len(stop.wait_calls), 1)

    def test_idle_time_does_not_buy_burst_credit(self):
        for rate in (1.0, 5.0):
            with self.subTest(rate=rate):
                clock = _FakeClock()
                stop = _FakeStopEvent(clock)
                pacer = _SendPacer(rate, stop, clock)
                pacer.record(SAMPLE_RATE)
                clock.advance(60)
                self.assertTrue(pacer.wait())
                self.assertEqual(stop.wait_calls, [])
                pacer.record(SAMPLE_RATE)
                self.assertTrue(pacer.wait())
                self.assertAlmostEqual(clock(), 160.0 + 1.0 / rate)
                self.assertEqual(len(stop.wait_calls), 1)
                self.assertAlmostEqual(stop.wait_calls[0], 1.0 / rate)

    def test_existing_stop_cancels_even_an_immediately_available_send(self):
        clock = _FakeClock()
        stop = _FakeStopEvent(clock)
        stop.stopped = True
        pacer = _SendPacer(5.0, stop, clock)
        self.assertFalse(pacer.wait())
        self.assertEqual(stop.wait_calls, [])

    def test_stop_during_wait_cancels_without_consuming_remaining_delay(self):
        clock = _FakeClock()
        stop = _FakeStopEvent(clock)
        pacer = _SendPacer(1.0, stop, clock)
        pacer.record(SAMPLE_RATE)
        stop.cancel_on_wait = True
        self.assertFalse(pacer.wait())
        self.assertEqual(stop.wait_calls, [1.0])
        self.assertEqual(clock(), 100.0)

    def test_actual_send_charges_same_pacer_for_replayed_and_new_audio(self):
        for rate in (1.0, 5.0):
            with self.subTest(rate=rate):
                clock = _FakeClock()
                stop = _FakeStopEvent(clock)
                transcriber = _transcriber(max_audio_send_rate=rate, clock=clock)
                transcriber._pacer = _SendPacer(rate, stop, clock)
                sent_at = []
                websocket = mock.Mock()
                websocket.send.side_effect = lambda message: sent_at.append(clock())
                for pcm in (bytes(SAMPLE_RATE * 2), bytes(SAMPLE_RATE * 2), b'\x01\x00' * SAMPLE_RATE):
                    self.assertTrue(transcriber._send(websocket, pcm, False))
                self.assertEqual(len(sent_at), 3)
                for index, at in enumerate(sent_at):
                    self.assertAlmostEqual(at, 100.0 + index / rate)

    def test_long_blocked_send_does_not_accumulate_burst_credit(self):
        for rate in (1.0, 5.0):
            with self.subTest(rate=rate):
                clock = _FakeClock()
                stop = _FakeStopEvent(clock)
                transcriber = _transcriber(max_audio_send_rate=rate, clock=clock)
                transcriber._pacer = _SendPacer(rate, stop, clock)
                sent_at = []

                def send(message):
                    sent_at.append(clock())
                    if len(sent_at) == 1:
                        clock.advance(30)

                websocket = mock.Mock()
                websocket.send.side_effect = send
                for _ in range(3):
                    self.assertTrue(transcriber._send(websocket, bytes(SAMPLE_RATE * 2), False))
                self.assertEqual(sent_at[:2], [100.0, 130.0])
                self.assertAlmostEqual(sent_at[2], 130.0 + 1.0 / rate)
                self.assertEqual(len(stop.wait_calls), 1)

    def test_cancelled_send_does_not_write_websocket_or_charge_more_audio(self):
        clock = _FakeClock()
        stop = _FakeStopEvent(clock)
        transcriber = _transcriber(clock=clock)
        transcriber._pacer = _SendPacer(5.0, stop, clock)
        transcriber._pacer.record(SAMPLE_RATE)
        next_send = transcriber._pacer.next_send
        stop.cancel_on_wait = True
        websocket = mock.Mock()
        self.assertFalse(transcriber._send(websocket, bytes(SAMPLE_RATE * 2), True))
        websocket.send.assert_not_called()
        self.assertEqual(transcriber._pacer.next_send, next_send)


if __name__ == '__main__':
    unittest.main()
