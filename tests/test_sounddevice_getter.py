import sys
import types
import unittest
from unittest import mock

import numpy as np
import tyro

from stream_translator_gpt.getter.audio_getter import SoundDeviceAudioGetter, _device_gain, _sounddevice_gain
from stream_translator_gpt.main import Config, _create_audio_getter, _validate_and_normalize


class SoundDeviceGetterTest(unittest.TestCase):

    def test_legacy_volume_curve_is_preserved(self):
        self.assertIs(_sounddevice_gain, _device_gain)
        self.assertAlmostEqual(_sounddevice_gain(0.5), 0.1586730901)
        self.assertEqual(_sounddevice_gain(1.0), 1.0)
        self.assertAlmostEqual(_sounddevice_gain(2.0), 7.4448740672)
        with self.assertRaises(ValueError):
            _sounddevice_gain(-1.0)

    def test_cli_defaults_to_pyaudio_and_accepts_sounddevice(self):
        default_config = tyro.cli(Config, args=['device'])
        sounddevice_config = tyro.cli(Config,
                                      args=[
                                          'device', '--device-audio-getter', 'sounddevice', '--device-multiplier', '0.5'
                                      ])

        self.assertEqual(default_config.device_audio_getter, 'pyaudio')
        self.assertEqual(default_config.device_multiplier, 1.0)
        self.assertEqual(sounddevice_config.device_audio_getter, 'sounddevice')
        self.assertEqual(sounddevice_config.device_multiplier, 0.5)

    def test_cli_accepts_bounded_debug_audio_options(self):
        config = tyro.cli(Config,
                          args=[
                              'device', '--device-audio-getter', 'sounddevice', '--device-audio-debug-dir', 'clips',
                              '--device-audio-debug-clip-duration', '2.5', '--device-audio-debug-max-clips', '3'
                          ])

        self.assertEqual(config.device_audio_debug_dir, 'clips')
        self.assertEqual(config.device_audio_debug_clip_duration, 2.5)
        self.assertEqual(config.device_audio_debug_max_clips, 3)

    def test_device_index_zero_is_preserved(self):
        fake_sounddevice = types.ModuleType('sounddevice')
        fake_sounddevice.default = types.SimpleNamespace(device=(3, 4))
        fake_sounddevice.query_devices = mock.Mock(return_value={'name': 'Zero', 'max_input_channels': 1})

        with mock.patch.dict(sys.modules, {'sounddevice': fake_sounddevice}):
            getter = SoundDeviceAudioGetter(device_index=0)

        self.assertEqual(getter.device_index, 0)
        fake_sounddevice.query_devices.assert_called_once_with(0)

    def test_factory_selects_sounddevice_without_constructing_pyaudio_getter(self):
        config = Config(url='device', device_audio_getter='sounddevice', device_index=4, device_multiplier=2.0)
        sentinel = object()

        with mock.patch('stream_translator_gpt.main.SoundDeviceAudioGetter', return_value=sentinel) as sounddevice_cls:
            with mock.patch('stream_translator_gpt.main.DeviceAudioGetter') as pyaudio_cls:
                result = _create_audio_getter(config)

        self.assertIs(result, sentinel)
        sounddevice_cls.assert_called_once_with(device_index=4,
                                                interval=0.5,
                                                multiplier=2.0,
                                                debug_audio_dir=None,
                                                debug_clip_duration=10.0,
                                                debug_max_clips=10)
        pyaudio_cls.assert_not_called()

    def test_factory_passes_gain_and_debug_options_to_pyaudio(self):
        config = Config(url='device',
                        device_index=0,
                        mic=True,
                        device_recording_interval=0.25,
                        device_multiplier=2.0,
                        device_audio_debug_dir='clips',
                        device_audio_debug_clip_duration=2.5,
                        device_audio_debug_max_clips=3)
        sentinel = object()

        with mock.patch('stream_translator_gpt.main.DeviceAudioGetter', return_value=sentinel) as pyaudio_cls:
            with mock.patch('stream_translator_gpt.main.SoundDeviceAudioGetter') as sounddevice_cls:
                result = _create_audio_getter(config)

        self.assertIs(result, sentinel)
        pyaudio_cls.assert_called_once_with(device_index=0,
                                            use_mic=True,
                                            interval=0.25,
                                            multiplier=2.0,
                                            debug_audio_dir='clips',
                                            debug_clip_duration=2.5,
                                            debug_max_clips=3)
        sounddevice_cls.assert_not_called()

    def test_gain_and_debug_audio_are_accepted_for_pyaudio(self):
        config = Config(url='device', device_multiplier=2.0, device_audio_debug_dir='clips')

        with mock.patch('stream_translator_gpt.main.logger.error'):
            _validate_and_normalize(config)

    def test_debug_audio_limits_must_be_positive(self):
        for field, value in [('device_audio_debug_clip_duration', 0), ('device_audio_debug_max_clips', 0)]:
            with self.subTest(field=field):
                config = Config(url='device', device_audio_getter='sounddevice', device_audio_debug_dir='clips')
                setattr(config, field, value)
                with mock.patch('stream_translator_gpt.main.logger.error'):
                    with self.assertRaises(SystemExit):
                        _validate_and_normalize(config)


if __name__ == '__main__':
    unittest.main()
