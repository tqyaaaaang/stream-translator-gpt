import contextlib
import io
import math
import unittest
from unittest import mock

import tyro

from stream_translator_gpt.getter.time_range import format_seconds, normalize_time_range
from stream_translator_gpt.main import (
    Config,
    _create_audio_getter,
    _preprocess_deprecated_flags,
    _validate_and_normalize,
)


class AudioTimeRangeParsingTest(unittest.TestCase):

    def test_omitted_boundaries_remain_unset(self):
        self.assertEqual(normalize_time_range(None, None), (None, None))

    def test_seconds_clock_and_unit_formats(self):
        for text, seconds in (
            ('0', 0.0),
            ('12.5', 12.5),
            ('12.5s', 12.5),
            ('1.', 1.0),
            ('1.s', 1.0),
            ('+1', 1.0),
            (' 1', 1.0),
            ('\t+1.25', 1.25),
            ('+1250ms', 1.25),
            ('1250ms', 1.25),
            ('1250000us', 1.25),
            ('00:01.250', 1.25),
            ('02:03', 123.0),
            ('01:02.', 62.0),
            ('01:02:03.004', 3723.004),
            ('12:34:56.000001', 45296.000001),
            (0, 0.0),
            (1.25, 1.25),
        ):
            with self.subTest(value=text):
                start, stop = normalize_time_range(text, None)
                self.assertAlmostEqual(start, seconds)
                self.assertIsNone(stop)
                if seconds > 0:
                    start, stop = normalize_time_range(None, text)
                    self.assertIsNone(start)
                    self.assertAlmostEqual(stop, seconds)

    def test_end_is_absolute_not_duration(self):
        self.assertEqual(normalize_time_range('01:00', '01:01.5'), (60.0, 61.5))

    def test_non_finite_negative_and_malformed_boundaries_are_rejected(self):
        invalid_values = (
            '', ' ', '-1', '-0.01s', '-10ms', '-1us', '-00:01',
            'NaN', 'nan', 'inf', '-inf', 'Infinity', 'one', '1x',
            '1:two', '1:2:3:4', '01::02', ':01', '01:', '00:00:NaN',
            '1.2.3', '1s trailing', '1 ', '1 s', '+ 1', '01: 02',
            '90:00', '00:60', '01:60:00', '01:01:60',
            '000:01', '.5', '1e3', '1e-6', '１２', '١٢',
            -0.001, math.nan, math.inf, -math.inf, True, False, [], {},
        )
        for value in invalid_values:
            for position in ('ss', 'to'):
                with self.subTest(value=value, position=position), self.assertRaises(ValueError):
                    normalize_time_range(value if position == 'ss' else None,
                                         value if position == 'to' else None)

    def test_end_must_be_after_start(self):
        for start, stop in ((None, '0'), ('0', '0'), ('1', '1'), ('10', '9'),
                            ('01:00', '59.999'), ('1000ms', '1s')):
            with self.subTest(start=start, stop=stop), self.assertRaises(ValueError):
                normalize_time_range(start, stop)

    def test_large_finite_seconds_and_microsecond_boundaries(self):
        self.assertEqual(normalize_time_range('1us', '2us'), (0.000001, 0.000002))
        self.assertEqual(normalize_time_range('86400000', '86400001'), (86400000.0, 86400001.0))

    def test_boundaries_truncate_to_ffmpeg_microsecond_precision(self):
        for value, expected in (('1.9999999', 1.999999),
                                ('0.99999999999999999999999999999', 0.999999),
                                ('00:00.99999999999999999999999999999', 0.999999),
                                ('00:01.0000009', 1.0),
                                ('0.0000009', 0.0),
                                ('0.9us', 0.0),
                                ('1.9us', 0.000001),
                                ('1500.9999ms', 1.500999)):
            with self.subTest(value=value):
                start, stop = normalize_time_range(value, None)
                self.assertEqual(start, expected)
                self.assertIsNone(stop)
                self.assertEqual(float(format_seconds(start)), expected)

    def test_ranges_collapsing_after_microsecond_conversion_are_rejected(self):
        for start, stop in ((None, '0.0000009'), (None, '0.9us'),
                            ('0.0000001', '0.0000002'),
                            ('1.0000001', '1.0000009'),
                            ('1us', '1.9us')):
            with self.subTest(start=start, stop=stop), self.assertRaises(ValueError):
                normalize_time_range(start, stop)
        self.assertEqual(normalize_time_range('1.0000009', '1.0000011'), (1.0, 1.000001))

    def test_values_outside_ffmpeg_duration_capacity_are_rejected(self):
        for value in ('9223372036855', 9223372036855.0, '9' * 400, 10 ** 400):
            for position in ('ss', 'to'):
                with self.subTest(value_type=type(value).__name__, position=position), \
                        self.assertRaises(ValueError):
                    normalize_time_range(value if position == 'ss' else None,
                                         value if position == 'to' else None)

    def test_formatter_avoids_scientific_notation_and_preserves_precision(self):
        for seconds in (0.0, 0.000001, 0.000002, 0.032, 1.25, 3723.004, 86400000.0):
            with self.subTest(seconds=seconds):
                rendered = format_seconds(seconds)
                self.assertNotIn('e', rendered.lower())
                self.assertEqual(float(rendered), seconds)
                self.assertEqual(normalize_time_range(rendered, None), (seconds, None))


class AudioTimeRangeCliTest(unittest.TestCase):

    @staticmethod
    def _parse(*args):
        return tyro.cli(Config, args=_preprocess_deprecated_flags(args))

    def test_options_default_to_disabled(self):
        for source in ('clip.mp3', 'https://example.test/video', 'device'):
            with self.subTest(source=source):
                config = self._parse(source)
                self.assertIsNone(config.ss)
                self.assertIsNone(config.to)
                self.assertFalse(config.time_range_strict)

    def test_options_can_appear_before_or_after_input_and_use_equals(self):
        cases = (
            ('clip.mp3', '--ss', '00:01.5', '--to', '10'),
            ('--ss', '00:01.5', '--to', '10', 'clip.mp3'),
            ('--ss=00:01.5', 'clip.mp3', '--to=10'),
            ('https://example.test/video', '--ss=00:01.5', '--to=10'),
        )
        for args in cases:
            with self.subTest(args=args):
                config = self._parse(*args)
                self.assertEqual(config.ss, '00:01.5')
                self.assertEqual(config.to, '10')

    def test_each_option_can_be_given_independently(self):
        for option, value in (('--ss', '1.25'), ('--to', '00:10')):
            with self.subTest(option=option):
                config = self._parse('clip.mp3', option, value)
                self.assertEqual(getattr(config, option[2:]), value)
                self.assertIsNone(config.to if option == '--ss' else config.ss)

    def test_bare_wait_option_does_not_consume_range_options(self):
        config = self._parse('https://example.test/video', '--wait-for-video',
                             '--time-range-strict', '--ss', '1', '--to', '2', '--pacer', 'realtime')
        self.assertEqual(config.wait_for_video, 60)
        self.assertEqual(config.ss, '1')
        self.assertEqual(config.to, '2')
        self.assertEqual(config.pacer, 'realtime')
        self.assertTrue(config.time_range_strict)

    def test_strict_flag_accepts_either_boundary_including_zero_start(self):
        for source in ('clip.mp3', 'https://example.test/video'):
            for boundary in (('--ss', '0'), ('--ss=03:40',), ('--to', '12.5')):
                for flag_first in (False, True):
                    args = (('--time-range-strict', source, *boundary) if flag_first else
                            (source, *boundary, '--time-range-strict'))
                    with self.subTest(args=args):
                        config = self._parse(*args)
                        with mock.patch('stream_translator_gpt.main.logger.warning') as warning:
                            _validate_and_normalize(config)
                        self.assertTrue(config.time_range_strict)
                        warning.assert_not_called()

    def test_strict_flag_does_not_accept_an_argument(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._parse('clip.mp3', '--ss', '0', '--time-range-strict', 'true')

    def test_options_require_values(self):
        for args in (
            ('clip.mp3', '--ss'),
            ('clip.mp3', '--to'),
            ('clip.mp3', '--ss', '--to', '10'),
            ('clip.mp3', '--to', '--ss', '1'),
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self._parse(*args)

    def test_valid_ranges_are_accepted_for_both_input_types(self):
        for source in ('clip.mp3', 'https://example.test/video'):
            for boundaries in ({'ss': '01:02.5'}, {'to': '90'}, {'ss': '1us', 'to': '2us'}):
                with self.subTest(source=source, boundaries=boundaries):
                    config = Config(url=source, **boundaries)
                    with mock.patch('stream_translator_gpt.main.logger.warning') as warning:
                        _validate_and_normalize(config)
                    warning.assert_not_called()
                    for name, value in boundaries.items():
                        self.assertEqual(getattr(config, name), value)

    def test_invalid_ranges_exit_before_worker_initialization(self):
        for source in ('clip.mp3', 'https://example.test/video'):
            for boundaries in ({'ss': '-1'}, {'to': 'inf'}, {'ss': ''},
                               {'to': '0'}, {'ss': '10', 'to': '9'}):
                with self.subTest(source=source, boundaries=boundaries):
                    config = Config(url=source, **boundaries)
                    with mock.patch('stream_translator_gpt.main.logger.error') as error, \
                            self.assertRaises(SystemExit) as raised:
                        _validate_and_normalize(config)
                    self.assertEqual(raised.exception.code, 1)
                    error.assert_called()


class AudioTimeRangeFactoryTest(unittest.TestCase):

    def test_strict_is_forwarded_to_local_and_stream_only_when_enabled(self):
        for source, factory in (('clip.mp3', 'LocalFileAudioGetter'),
                                ('https://example.test/video', 'StreamAudioGetter')):
            for boundaries in ({'ss': '0'}, {'ss': '03:40'}, {'to': '90'}, {'ss': '2', 'to': '4'}):
                for strict in (False, True):
                    with self.subTest(source=source, boundaries=boundaries, strict=strict):
                        config = Config(url=source, pacer='realtime', time_range_strict=strict, **boundaries)
                        with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                                mock.patch('stream_translator_gpt.main.logger.warning') as warning:
                            _validate_and_normalize(config)
                            _create_audio_getter(config)
                        self.assertEqual(getter.call_args.kwargs.get('time_range_strict', False), strict)
                        self.assertTrue(getter.call_args.kwargs['realtime'])
                        warning.assert_not_called()

    def test_strict_without_boundaries_warns_once_and_is_ignored(self):
        cases = (
            ('clip.mp3', 'LocalFileAudioGetter', {}),
            ('https://example.test/video', 'StreamAudioGetter', {}),
            ('device', 'DeviceAudioGetter', {}),
            ('device', 'SoundDeviceAudioGetter', {'device_audio_getter': 'sounddevice'}),
            ('https://example.test/video', 'StreamAudioGetter', {'live_from_start': True}),
        )
        for source, factory, extra in cases:
            for validate_first in (False, True):
                with self.subTest(source=source, extra=extra, validate_first=validate_first):
                    config = Config(url=source, time_range_strict=True, **extra)
                    with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                            self.assertLogs('main', level='WARNING') as logs:
                        if validate_first:
                            _validate_and_normalize(config)
                        _create_audio_getter(config)
                        _validate_and_normalize(config)
                    self.assertEqual(len(logs.output), 1)
                    self.assertIn('--time-range-strict', logs.output[0])
                    self.assertIn('--ss', logs.output[0])
                    self.assertIn('--to', logs.output[0])
                    self.assertFalse(config.time_range_strict)
                    self.assertNotIn('time_range_strict', getter.call_args.kwargs)
                    self.assertNotIn('ss', getter.call_args.kwargs)
                    self.assertNotIn('to', getter.call_args.kwargs)

    def test_incompatible_inputs_ignore_strict_and_boundaries_together_once(self):
        cases = (
            ('device', 'DeviceAudioGetter', {}),
            ('DEVICE', 'SoundDeviceAudioGetter', {'device_audio_getter': 'sounddevice'}),
            ('device', 'DeviceAudioGetter', {'live_from_start': True}),
            ('https://example.test/video', 'StreamAudioGetter', {'live_from_start': True}),
            ('clip.mp3', 'LocalFileAudioGetter', {'live_from_start': True}),
        )
        for source, factory, extra in cases:
            for boundaries in ({'ss': '0'}, {'ss': 'invalid'}, {'to': '-1'}):
                with self.subTest(source=source, extra=extra, boundaries=boundaries):
                    config = Config(url=source, time_range_strict=True, **extra, **boundaries)
                    with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                            self.assertLogs('main', level='WARNING') as logs:
                        _validate_and_normalize(config)
                        _create_audio_getter(config)
                    self.assertEqual(len(logs.output), 1)
                    self.assertIn('--time-range-strict', logs.output[0])
                    self.assertFalse(config.time_range_strict)
                    self.assertIsNone(config.ss)
                    self.assertIsNone(config.to)
                    for name in ('ss', 'to', 'time_range_strict'):
                        self.assertNotIn(name, getter.call_args.kwargs)

    def test_factories_forward_present_boundaries_without_changing_pacing(self):
        for source, factory in (('clip.mp3', 'LocalFileAudioGetter'),
                                ('https://example.test/video', 'StreamAudioGetter')):
            for pacer in ('unlimited', 'realtime'):
                for boundaries in ({'ss': '01:02.5'}, {'to': '90'},
                                   {'ss': '1us', 'to': '2us'}):
                    with self.subTest(source=source, pacer=pacer, boundaries=boundaries):
                        config = Config(url=source, pacer=pacer, **boundaries)
                        with mock.patch(f'stream_translator_gpt.main.{factory}') as getter:
                            result = _create_audio_getter(config)
                        self.assertIs(result, getter.return_value)
                        self.assertEqual(getter.call_args.kwargs['realtime'], pacer == 'realtime')
                        for name in ('ss', 'to'):
                            if name in boundaries:
                                self.assertEqual(getter.call_args.kwargs[name], boundaries[name])
                            else:
                                self.assertNotIn(name, getter.call_args.kwargs)

    def test_omitted_boundaries_do_not_add_getter_arguments_or_warnings(self):
        for source, factory in (('clip.mp3', 'LocalFileAudioGetter'),
                                ('https://example.test/video', 'StreamAudioGetter'),
                                ('device', 'DeviceAudioGetter')):
            with self.subTest(source=source):
                config = Config(url=source)
                with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                        mock.patch('stream_translator_gpt.main.logger.warning') as warning:
                    _validate_and_normalize(config)
                    _create_audio_getter(config)
                warning.assert_not_called()
                self.assertNotIn('ss', getter.call_args.kwargs)
                self.assertNotIn('to', getter.call_args.kwargs)

    def test_factory_rejects_invalid_ranges_even_without_cli_validation(self):
        for source, factory in (('clip.mp3', 'LocalFileAudioGetter'),
                                ('https://example.test/video', 'StreamAudioGetter')):
            for boundaries in ({'ss': 'invalid'}, {'ss': '5', 'to': '4'}):
                with self.subTest(source=source, boundaries=boundaries):
                    with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                            mock.patch('stream_translator_gpt.main.logger.error'), \
                            self.assertRaises((ValueError, SystemExit)):
                        _create_audio_getter(Config(url=source, **boundaries))
                    getter.assert_not_called()

    def test_device_modes_warn_and_ignore_even_invalid_ranges(self):
        for device_backend, factory in (('pyaudio', 'DeviceAudioGetter'),
                                        ('sounddevice', 'SoundDeviceAudioGetter')):
            for source in ('device', 'DEVICE'):
                for validate_first in (False, True):
                    for boundaries in ({'ss': 'bad'}, {'to': '-1'},
                                       {'ss': '100', 'to': '10'}):
                        with self.subTest(backend=device_backend, source=source,
                                          validate_first=validate_first, boundaries=boundaries):
                            config = Config(url=source, device_audio_getter=device_backend, **boundaries)
                            with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                                    self.assertLogs('main', level='WARNING') as captured:
                                if validate_first:
                                    _validate_and_normalize(config)
                                _create_audio_getter(config)
                            self.assertEqual(len(captured.output), 1)
                            self.assertIn('device', captured.output[0].lower())
                            self.assertTrue('--ss' in captured.output[0] or '--to' in captured.output[0])
                            self.assertIsNone(config.ss)
                            self.assertIsNone(config.to)
                            self.assertNotIn('ss', getter.call_args.kwargs)
                            self.assertNotIn('to', getter.call_args.kwargs)

    def test_live_from_start_warns_and_ignores_ranges_including_local_input(self):
        for source, factory in (('clip.mp3', 'LocalFileAudioGetter'),
                                ('https://example.test/video', 'StreamAudioGetter')):
            for validate_first in (False, True):
                for boundaries in ({'ss': 'bad'}, {'to': '-1'},
                                   {'ss': '100', 'to': '10'}):
                    with self.subTest(source=source, validate_first=validate_first,
                                      boundaries=boundaries):
                        config = Config(url=source, live_from_start=True, **boundaries)
                        with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                                self.assertLogs('main', level='WARNING') as captured:
                            if validate_first:
                                _validate_and_normalize(config)
                            _create_audio_getter(config)
                        self.assertEqual(len(captured.output), 1)
                        self.assertIn('--live-from-start', captured.output[0])
                        self.assertIsNone(config.ss)
                        self.assertIsNone(config.to)
                        self.assertNotIn('ss', getter.call_args.kwargs)
                        self.assertNotIn('to', getter.call_args.kwargs)
                        if source.startswith('https://'):
                            self.assertTrue(getter.call_args.kwargs['live_from_start'])

    def test_live_from_start_without_range_does_not_warn(self):
        for source, factory in (('clip.mp3', 'LocalFileAudioGetter'),
                                ('https://example.test/video', 'StreamAudioGetter')):
            with self.subTest(source=source):
                config = Config(url=source, live_from_start=True)
                with mock.patch(f'stream_translator_gpt.main.{factory}') as getter, \
                        mock.patch('stream_translator_gpt.main.logger.warning') as warning:
                    _validate_and_normalize(config)
                    _create_audio_getter(config)
                warning.assert_not_called()
                self.assertNotIn('ss', getter.call_args.kwargs)
                self.assertNotIn('to', getter.call_args.kwargs)

    def test_device_and_live_start_together_report_both_reasons_once(self):
        config = Config(url='device', live_from_start=True, ss='invalid', to='0')
        with mock.patch('stream_translator_gpt.main.DeviceAudioGetter') as getter, \
                self.assertLogs('main', level='WARNING') as captured:
            _validate_and_normalize(config)
            _create_audio_getter(config)
        self.assertEqual(len(captured.output), 1)
        self.assertIn('device', captured.output[0].lower())
        self.assertIn('--live-from-start', captured.output[0])
        self.assertIsNone(config.ss)
        self.assertIsNone(config.to)
        self.assertNotIn('ss', getter.call_args.kwargs)
        self.assertNotIn('to', getter.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
