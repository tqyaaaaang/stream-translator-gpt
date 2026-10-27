import contextlib
import math
import queue
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import tyro

from stream_translator_gpt import main
from stream_translator_gpt.main import Config, _create_audio_getter, _validate_and_normalize
from stream_translator_gpt.transcriber.elevenlabs_transcriber import SCRIBE_STREAM_MAX_SEND_RATE
from stream_translator_gpt.transcriber.modulate_transcriber import MODULATE_STREAM_MAX_SEND_RATE
from stream_translator_gpt.utils.common import SAMPLE_RATE, SAMPLES_PER_FRAME


class ElevenLabsConfigTest(unittest.TestCase):

    def test_cli_flag_is_opt_in_and_uses_its_own_explicit_key(self):
        default = tyro.cli(Config, args=['clip.mp3'])
        enabled = tyro.cli(Config, args=[
            'clip.mp3', '--use-elevenlabs-scribe-api', '--elevenlabs-api-key', 'scribe-test-key',
        ])
        self.assertFalse(default.use_elevenlabs_scribe_api)
        self.assertIsNone(default.elevenlabs_api_key)
        self.assertEqual(main._transcription_engine_label(default), 'Whisper')
        self.assertTrue(enabled.use_elevenlabs_scribe_api)
        self.assertEqual(enabled.elevenlabs_api_key, 'scribe-test-key')
        self.assertEqual(main._transcription_engine_label(enabled), 'ElevenLabs Scribe v2 Realtime')

    def test_other_provider_keys_do_not_enable_scribe(self):
        for key in (None, '', '   '):
            config = Config(url='clip.mp3', use_elevenlabs_scribe_api=True, elevenlabs_api_key=key,
                            openai_api_key='openai-test-key', modulate_api_key='modulate-test-key')
            with self.subTest(key=key), self.assertRaises(SystemExit):
                _validate_and_normalize(config)

    def test_new_backend_is_mutually_exclusive_with_each_old_backend(self):
        for flag in ('use_faster_whisper', 'use_simul_streaming', 'use_openai_transcription_api',
                     'use_modulate_transcription_api', 'use_hf_asr'):
            config = Config(url='clip.mp3', use_elevenlabs_scribe_api=True, elevenlabs_api_key='test-key',
                            **{flag: True})
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                _validate_and_normalize(config)

    def test_maximum_commit_length_is_clamped_only_for_scribe(self):
        for options in ({}, {'use_modulate_transcription_api': True, 'modulate_api_key': 'mod-key'},
                        {'use_elevenlabs_scribe_api': True, 'elevenlabs_api_key': 'scribe-key'}):
            config = Config(url='clip.mp3', max_audio_length=45.0, pacer='realtime', **options)
            with self.subTest(options=options), mock.patch.object(main.logger, 'warning') as warning:
                _validate_and_normalize(config)
            if config.use_elevenlabs_scribe_api:
                self.assertEqual(config.max_audio_length, 30.0)
                warning.assert_called_once()
                self.assertIn('--max-audio-length', warning.call_args.args[0])
            else:
                self.assertEqual(config.max_audio_length, 45.0)
                warning.assert_not_called()

    def test_scribe_rejects_invalid_or_inverted_commit_lengths(self):
        for minimum, maximum in ((0, 30), (-1, 30), (math.nan, 30), (0.5, 0),
                                 (0.5, math.inf), (31, 60), (4, 3)):
            config = Config(url='clip.mp3', use_elevenlabs_scribe_api=True, elevenlabs_api_key='test-key',
                            min_audio_length=minimum, max_audio_length=maximum)
            with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(SystemExit):
                _validate_and_normalize(config)

    def test_scribe_does_not_apply_whisper_model_language_rules(self):
        config = Config(url='clip.mp3', use_elevenlabs_scribe_api=True, elevenlabs_api_key='test-key',
                        model='large.en', language='ja')
        _validate_and_normalize(config)
        self.assertEqual(config.language, 'ja')

    def test_factory_preserves_final_block_only_for_scribe_file_and_url_input(self):
        for url, constructor in (('clip.mp3', 'LocalFileAudioGetter'),
                                 ('https://example.test/stream', 'StreamAudioGetter')):
            for enabled in (False, True):
                config = Config(url=url, use_elevenlabs_scribe_api=enabled)
                with self.subTest(url=url, enabled=enabled), mock.patch.object(main, constructor) as getter:
                    _create_audio_getter(config)
                kwargs = getter.call_args.kwargs
                if enabled:
                    self.assertTrue(kwargs['preserve_final_block'])
                else:
                    self.assertNotIn('preserve_final_block', kwargs)

    def test_device_factory_receives_no_new_internal_option(self):
        for backend, constructor in (('pyaudio', 'DeviceAudioGetter'),
                                     ('sounddevice', 'SoundDeviceAudioGetter')):
            config = Config(url='device', device_audio_getter=backend, use_elevenlabs_scribe_api=True)
            with self.subTest(backend=backend), mock.patch.object(main, constructor) as getter:
                _create_audio_getter(config)
            self.assertNotIn('preserve_final_block', getter.call_args.kwargs)

    def test_scribe_getter_pacing_remains_controlled_by_pacer_option(self):
        for url, constructor in (('clip.mp3', 'LocalFileAudioGetter'),
                                 ('https://example.test/stream', 'StreamAudioGetter')):
            for pacer in ('unlimited', 'realtime'):
                config = Config(url=url, pacer=pacer, use_elevenlabs_scribe_api=True)
                with self.subTest(url=url, pacer=pacer), mock.patch.object(main, constructor) as getter:
                    _create_audio_getter(config)
                    self.assertEqual(getter.call_args.kwargs['realtime'], pacer == 'realtime')
                    self.assertTrue(getter.call_args.kwargs['preserve_final_block'])

    def test_scribe_device_getters_keep_natural_capture_pacing(self):
        for backend, constructor in (('pyaudio', 'DeviceAudioGetter'),
                                     ('sounddevice', 'SoundDeviceAudioGetter')):
            calls = []
            for pacer in ('unlimited', 'realtime'):
                config = Config(url='device', pacer=pacer, device_audio_getter=backend,
                                use_elevenlabs_scribe_api=True)
                with self.subTest(backend=backend, pacer=pacer), \
                        mock.patch.object(main, constructor) as getter, \
                        mock.patch.object(main.logger, 'warning') as warning:
                    _create_audio_getter(config)
                    calls.append(getter.call_args)
                    self.assertNotIn('realtime', getter.call_args.kwargs)
                    if pacer == 'realtime':
                        warning.assert_called_once()
                        self.assertIn('already real-time', warning.call_args.args[0])
                    else:
                        warning.assert_not_called()
            self.assertEqual(calls[0], calls[1])


class ElevenLabsPipelineRoutingTest(unittest.TestCase):

    def _run(self, config):
        constructors = {}
        workers = []
        thread = mock.Mock()
        thread.is_alive.return_value = False
        slicer_constructor = mock.Mock()

        def start(func, *args, **kwargs):
            workers.append((func, kwargs))
            return thread

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, 'ClientPool'))
            stack.enter_context(mock.patch.object(main, '_create_audio_getter'))
            stack.enter_context(mock.patch.object(main, 'ResultExporter'))
            stack.enter_context(mock.patch.object(main.signal, 'signal'))
            stack.enter_context(mock.patch.object(main, 'start_daemon_thread', side_effect=start))
            stack.enter_context(mock.patch.dict(sys.modules, {
                'stream_translator_gpt.slicer.audio_slicer': SimpleNamespace(AudioSlicer=slicer_constructor),
            }))
            for name in ('ElevenLabsScribeTranscriber', 'ModulateStreamingTranscriber', 'OpenaiWhisper',
                         'FasterWhisper', 'SimulStreaming', 'RemoteOpenaiTranscriber', 'HFTranscriber'):
                constructors[name] = stack.enter_context(mock.patch.object(main, name))
            main.run(config)
        return constructors, workers, slicer_constructor

    def test_scribe_uses_bounded_raw_queue_and_url_only_five_x_send_limit(self):
        self.assertEqual(SCRIBE_STREAM_MAX_SEND_RATE, 5.0)
        for url, expected_rate in (('clip.mp3', None), ('https://example.test/stream', 5.0), ('device', None)):
            for pacer in ('unlimited', 'realtime'):
                config = Config(url=url, pacer=pacer, use_elevenlabs_scribe_api=True,
                                elevenlabs_api_key='scribe-test-key', language='ja',
                                processing_proxy='http://proxy.test:8080', verify_ssl=False,
                                min_audio_length=0.75, max_audio_length=20, target_audio_length=6,
                                prefix_retention_length=0.25, transcription_keywords='さくな')
                with self.subTest(url=url, pacer=pacer):
                    constructors, workers, slicer = self._run(config)
                    constructors['ElevenLabsScribeTranscriber'].assert_called_once()
                    kwargs = constructors['ElevenLabsScribeTranscriber'].call_args.kwargs
                    self.assertEqual(kwargs['max_audio_send_rate'], expected_rate)
                    self.assertEqual(kwargs['api_key'], 'scribe-test-key')
                    self.assertEqual(kwargs['language'], 'ja')
                    self.assertEqual(kwargs['proxy'], config.processing_proxy)
                    self.assertFalse(kwargs['verify_ssl'])
                    self.assertNotIn('model', kwargs)
                    for name in ('min_audio_length', 'max_audio_length', 'target_audio_length',
                                 'continuous_no_speech_threshold', 'dynamic_no_speech_threshold',
                                 'prefix_retention_length', 'vad_threshold', 'dynamic_vad_threshold',
                                 'transcription_keywords', 'transcription_filters', 'language_filters'):
                        self.assertEqual(kwargs[name], getattr(config, name))
                    slicer.assert_not_called()
                    names = [worker['name'] for _, worker in workers]
                    self.assertEqual(names, ['getter', 'transcriber', 'exporter'])
                    getter_queue = workers[0][1]['output_queue']
                    self.assertIs(getter_queue, workers[1][1]['input_queue'])
                    self.assertIsInstance(getter_queue, queue.Queue)
                    self.assertEqual(getter_queue.maxsize,
                                     math.ceil(main.ELEVENLABS_INPUT_QUEUE_SECONDS * SAMPLE_RATE / SAMPLES_PER_FRAME))

    def test_modulate_keeps_its_queue_routing_and_url_only_send_limit(self):
        for url, expected_rate in (('clip.mp3', None), ('https://example.test/stream', MODULATE_STREAM_MAX_SEND_RATE)):
            for pacer in ('unlimited', 'realtime'):
                config = Config(url=url, pacer=pacer, use_modulate_transcription_api=True, modulate_api_key='key')
                with self.subTest(url=url, pacer=pacer):
                    constructors, workers, slicer = self._run(config)
                    kwargs = constructors['ModulateStreamingTranscriber'].call_args.kwargs
                    self.assertEqual(kwargs['max_audio_send_rate'], expected_rate)
                    constructors['ElevenLabsScribeTranscriber'].assert_not_called()
                    slicer.assert_not_called()
                    self.assertIs(workers[0][1]['output_queue'], workers[1][1]['input_queue'])
                    self.assertEqual(workers[0][1]['output_queue'].maxsize, 18750)

    def test_legacy_sliced_backends_keep_original_workers_and_slicer_parameters(self):
        cases = [({}, 'OpenaiWhisper'), ({'use_faster_whisper': True}, 'FasterWhisper'),
                 ({'use_simul_streaming': True}, 'SimulStreaming'),
                 ({'use_openai_transcription_api': True}, 'RemoteOpenaiTranscriber'),
                 ({'use_hf_asr': True}, 'HFTranscriber')]
        for options, constructor in cases:
            config = Config(url='clip.mp3', max_audio_length=45.0, prefix_retention_length=0.75, **options)
            with self.subTest(options=options):
                constructors, workers, slicer = self._run(config)
                constructors[constructor].assert_called_once()
                constructors['ElevenLabsScribeTranscriber'].assert_not_called()
                self.assertEqual([kwargs['name'] for _, kwargs in workers],
                                 ['getter', 'slicer', 'transcriber', 'exporter'])
                slicer.assert_called_once()
                self.assertEqual(slicer.call_args.kwargs['max_audio_length'], 45.0)
                self.assertEqual(slicer.call_args.kwargs['prefix_retention_length'], 0.75)
                self.assertIsInstance(workers[0][1]['output_queue'], queue.SimpleQueue)
                self.assertIs(workers[0][1]['output_queue'], workers[1][1]['input_queue'])
                self.assertIs(workers[1][1]['output_queue'], workers[2][1]['input_queue'])
                self.assertIsNot(workers[0][1]['output_queue'], workers[2][1]['input_queue'])


if __name__ == '__main__':
    unittest.main()
