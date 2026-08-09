import queue
import sys
import tempfile
import threading
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from stream_translator_gpt.audio_getter import DeviceAudioGetter, _device_gain
from stream_translator_gpt.common import SAMPLE_RATE, SAMPLES_PER_FRAME


class _FakeStream:

    def __init__(self, callback, blocks):
        self.callback = callback
        self.blocks = blocks
        self.active = False
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start_stream(self):
        self.started += 1
        self.active = True
        for audio, status in self.blocks:
            self.callback(audio, len(audio) // 4, object(), status)
        self.active = False

    def is_active(self):
        return self.active

    def stop_stream(self):
        self.stopped += 1
        self.active = False

    def close(self):
        self.closed += 1

    def read(self, *args, **kwargs):
        raise AssertionError('callback streams must not call read()')


class _FakePyAudio:

    def __init__(self, device_info, blocks):
        self.device_info = device_info
        self.blocks = blocks
        self.open_kwargs = None
        self.stream = None
        self.terminated = 0

    def get_device_info_by_index(self, index):
        return dict(self.device_info, index=index)

    def get_default_input_device_info(self):
        return dict(self.device_info, index=0)

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        self.stream = _FakeStream(kwargs['stream_callback'], self.blocks)
        return self.stream

    def terminate(self):
        self.terminated += 1


def _make_getter(*,
                 blocks=(),
                 native_rate=SAMPLE_RATE,
                 channels=1,
                 interval=0.5,
                 multiplier=1.0,
                 debug_audio_dir=None,
                 debug_clip_duration=10.0,
                 debug_max_clips=10):
    device_info = {
        'name': 'Fake PyAudio input',
        'defaultSampleRate': native_rate,
        'maxInputChannels': channels,
    }
    instance = _FakePyAudio(device_info, blocks)
    module = types.ModuleType('pyaudio')
    module.paFloat32 = 1
    module.paContinue = 2
    module.paComplete = 3
    module.paAbort = 4
    module.PyAudio = mock.Mock(return_value=instance)
    with mock.patch('stream_translator_gpt.audio_getter.platform.system', return_value='Linux'):
        with mock.patch.dict(sys.modules, {'pyaudio': module}):
            getter = DeviceAudioGetter(device_index=0,
                                       use_mic=True,
                                       interval=interval,
                                       multiplier=multiplier,
                                       debug_audio_dir=debug_audio_dir,
                                       debug_clip_duration=debug_clip_duration,
                                       debug_max_clips=debug_max_clips)
    return getter, instance, module


def _drain(audio_queue):
    values = []
    while not audio_queue.empty():
        values.append(audio_queue.get_nowait())
    return values


class PyAudioCallbackTest(unittest.TestCase):

    def test_windows_default_output_maps_to_matching_wasapi_loopback(self):

        class FakeWasapiPyAudio:

            def __init__(self):
                self.terminated = 0

            def get_host_api_info_by_type(self, api_type):
                self.api_type = api_type
                return {'defaultOutputDevice': 1}

            def get_device_info_by_index(self, index):
                devices = {
                    1: {
                        'index': 1,
                        'name': 'Fake Speakers',
                        'isLoopbackDevice': False,
                        'defaultSampleRate': 48000,
                        'maxInputChannels': 0,
                    },
                    7: {
                        'index': 7,
                        'name': 'Fake Speakers [Loopback]',
                        'isLoopbackDevice': True,
                        'defaultSampleRate': 48000,
                        'maxInputChannels': 2,
                    },
                }
                return devices[index]

            def get_loopback_device_info_generator(self):
                yield self.get_device_info_by_index(7)

            def terminate(self):
                self.terminated += 1

        instance = FakeWasapiPyAudio()
        module = types.ModuleType('pyaudiowpatch')
        module.paWASAPI = 13
        module.PyAudio = mock.Mock(return_value=instance)
        with mock.patch('stream_translator_gpt.audio_getter.platform.system', return_value='Windows'):
            with mock.patch.dict(sys.modules, {'pyaudiowpatch': module}):
                getter = DeviceAudioGetter(device_index=None, use_mic=False)
                module.PyAudio.assert_not_called()
                getter._initialize_pyaudio()

        self.assertEqual(getter.device_index, 7)
        self.assertEqual(getter.device_name, 'Fake Speakers [Loopback]')
        self.assertEqual(getter.channels, 2)
        self.assertEqual(instance.api_type, module.paWASAPI)
        getter._cleanup()
        self.assertEqual(instance.terminated, 1)

    def test_zero_channel_metadata_keeps_legacy_two_channel_fallback(self):
        with mock.patch('stream_translator_gpt.audio_getter.logger.warning') as warning:
            getter, instance, _ = _make_getter(channels=0)
            getter._initialize_pyaudio()

        self.assertEqual(getter.device_index, 0)
        self.assertEqual(getter.channels, 2)
        warning.assert_called_once_with('PyAudio device %s reported no input channels; trying two channels', 0)
        getter._cleanup()
        self.assertEqual(instance.terminated, 1)

    def test_portaudio_is_initialized_in_the_capture_thread(self):
        main_thread_id = threading.get_ident()
        instances = []
        device_info = {
            'name': 'Thread-bound WASAPI loopback',
            'defaultSampleRate': SAMPLE_RATE,
            'maxInputChannels': 1,
        }

        class ThreadBoundPyAudio(_FakePyAudio):

            def __init__(self):
                super().__init__(device_info, [(b'', 0)])
                self.initialized_thread_id = threading.get_ident()
                self.open_thread_id = None
                self.terminate_thread_id = None

            def open(self, **kwargs):
                self.open_thread_id = threading.get_ident()
                if self.open_thread_id != self.initialized_thread_id:
                    raise OSError(-9999, 'Unanticipated host error')
                return super().open(**kwargs)

            def terminate(self):
                self.terminate_thread_id = threading.get_ident()
                super().terminate()

        module = types.ModuleType('pyaudio')
        module.paFloat32 = 1
        module.paContinue = 2
        module.paComplete = 3
        module.paAbort = 4

        def create_pyaudio():
            instance = ThreadBoundPyAudio()
            instances.append(instance)
            return instance

        module.PyAudio = create_pyaudio
        with mock.patch('stream_translator_gpt.audio_getter.platform.system', return_value='Linux'):
            with mock.patch.dict(sys.modules, {'pyaudio': module}):
                getter = DeviceAudioGetter(device_index=0, use_mic=True)

        self.assertEqual(instances, [])
        output_queue = queue.SimpleQueue()
        loop_thread = threading.Thread(target=getter.loop, args=(output_queue,))
        loop_thread.start()
        loop_thread.join(timeout=2)

        self.assertFalse(loop_thread.is_alive())
        self.assertEqual(len(instances), 1)
        self.assertNotEqual(instances[0].initialized_thread_id, main_thread_id)
        self.assertEqual(instances[0].initialized_thread_id, loop_thread.ident)
        self.assertEqual(instances[0].open_thread_id, loop_thread.ident)
        self.assertEqual(instances[0].terminate_thread_id, loop_thread.ident)
        self.assertEqual(instances[0].terminated, 1)
        self.assertEqual(_drain(output_queue), [None])

    def test_callback_copies_input_reports_status_and_uses_callback_stream(self):
        getter, instance, module = _make_getter(interval=SAMPLES_PER_FRAME / SAMPLE_RATE)
        callback_bytes = bytearray(np.full(SAMPLES_PER_FRAME, 0.25, dtype=np.float32).tobytes())

        result = getter._recording_callback(callback_bytes, SAMPLES_PER_FRAME, object(), 17)
        callback_bytes[:] = np.full(SAMPLES_PER_FRAME, -0.75, dtype=np.float32).tobytes()
        output_queue = queue.SimpleQueue()

        # Supply the already-populated callback queue through the normal stream lifecycle.
        instance.blocks = [(b'', 0)]
        with mock.patch('stream_translator_gpt.audio_getter.logger.warning') as warning:
            getter.loop(output_queue)

        output = _drain(output_queue)
        self.assertEqual(result, (None, module.paContinue))
        np.testing.assert_array_equal(output[0], np.full(SAMPLES_PER_FRAME, 0.25, dtype=np.float32))
        self.assertIsNone(output[1])
        warning.assert_any_call('PyAudio recorder returned status=%s', 17)
        self.assertEqual(instance.open_kwargs['stream_callback'], getter._recording_callback)
        self.assertFalse(instance.open_kwargs['start'])
        self.assertEqual(instance.stream.started, 1)
        self.assertEqual(instance.stream.closed, 1)
        self.assertEqual(instance.terminated, 1)

    def test_queue_is_bounded_and_drops_the_oldest_block(self):
        getter, _, module = _make_getter()
        for value in range(9):
            audio = np.full(4, value, dtype=np.float32).tobytes()
            self.assertEqual(getter._recording_callback(audio, 4, object(), 0), (None, module.paContinue))

        queued = []
        while not getter.recording_queue.empty():
            block, _ = getter.recording_queue.get_nowait()
            queued.append(np.frombuffer(block, dtype=np.float32)[0])

        self.assertEqual(queued, list(range(1, 9)))
        self.assertEqual(getter.dropped_blocks, 1)
        with mock.patch('stream_translator_gpt.audio_getter.logger.warning') as warning:
            getter._log_dropped_blocks()
        warning.assert_called_once_with('PyAudio processing is behind; dropped %d queued audio block(s)', 1)

    def test_callback_failure_aborts_and_queues_end_of_stream(self):

        class BrokenBytes:

            def __bytes__(self):
                raise ValueError('broken callback data')

        getter, _, module = _make_getter()

        result = getter._recording_callback(BrokenBytes(), 1, object(), 0)

        self.assertEqual(result, (None, module.paAbort))
        self.assertTrue(getter.stopping.is_set())
        self.assertIsNotNone(getter.callback_error)
        self.assertIsNone(getter.recording_queue.get_nowait())

    def test_stereo_native_audio_is_downmixed_resampled_gained_and_framed(self):
        native_rate = 48000
        native_frames = 3072
        stereo = np.column_stack((
            np.full(native_frames, -0.25, dtype=np.float32),
            np.full(native_frames, 0.75, dtype=np.float32),
        )).reshape(-1)
        blocks = [(stereo.tobytes(), 0), (b'', 0)]
        getter, instance, _ = _make_getter(blocks=blocks,
                                           native_rate=native_rate,
                                           channels=2,
                                           interval=native_frames / native_rate,
                                           multiplier=2.0)
        output_queue = queue.SimpleQueue()

        getter.loop(output_queue)

        output = _drain(output_queue)
        self.assertIsNone(output[-1])
        chunks = output[:-1]
        self.assertEqual([chunk.shape for chunk in chunks], [(SAMPLES_PER_FRAME,), (SAMPLES_PER_FRAME,)])
        self.assertTrue(all(chunk.dtype == np.float32 for chunk in chunks))
        np.testing.assert_allclose(np.concatenate(chunks), 0.25 * _device_gain(2.0), atol=1e-5)
        self.assertEqual(instance.open_kwargs['channels'], 2)
        self.assertEqual(instance.open_kwargs['rate'], native_rate)
        self.assertEqual(instance.open_kwargs['frames_per_buffer'], native_frames)

    def test_debug_wav_contains_processed_pyaudio_samples(self):
        samples = np.full(SAMPLES_PER_FRAME, 0.25, dtype=np.float32)
        blocks = [(samples.tobytes(), 0), (b'', 0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            getter, _, _ = _make_getter(blocks=blocks,
                                         interval=SAMPLES_PER_FRAME / SAMPLE_RATE,
                                         multiplier=2.0,
                                         debug_audio_dir=temp_dir,
                                         debug_clip_duration=SAMPLES_PER_FRAME / SAMPLE_RATE,
                                         debug_max_clips=1)
            output_queue = queue.SimpleQueue()

            getter.loop(output_queue)

            wav_paths = list(Path(temp_dir).glob('*.wav'))
            self.assertEqual(len(wav_paths), 1)
            with wave.open(str(wav_paths[0]), 'rb') as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getframerate(), SAMPLE_RATE)
                self.assertEqual(wav_file.getnframes(), SAMPLES_PER_FRAME)
                pcm = np.frombuffer(wav_file.readframes(SAMPLES_PER_FRAME), dtype='<i2')
            expected = min(32767, round(0.25 * _device_gain(2.0) * 32768))
            np.testing.assert_array_equal(pcm, np.full(SAMPLES_PER_FRAME, expected, dtype=np.int16))
            self.assertFalse(list(Path(temp_dir).glob('*.part')))

    def test_inactive_stream_without_sentinel_exits_after_draining_callback_data(self):
        samples = np.full(SAMPLES_PER_FRAME, 0.125, dtype=np.float32)
        getter, instance, _ = _make_getter(blocks=[(samples.tobytes(), 0)],
                                           interval=SAMPLES_PER_FRAME / SAMPLE_RATE)
        output_queue = queue.SimpleQueue()

        with mock.patch('stream_translator_gpt.audio_getter.logger.warning') as warning:
            getter.loop(output_queue)

        output = _drain(output_queue)
        np.testing.assert_array_equal(output[0], samples)
        self.assertIsNone(output[1])
        warning.assert_any_call('PyAudio stream stopped without an end-of-stream callback')
        self.assertEqual(instance.terminated, 1)

    def test_signal_during_open_closes_the_late_stream_and_terminates_once(self):
        getter, instance, _ = _make_getter()
        open_entered = threading.Event()
        allow_open_to_return = threading.Event()
        stream = _FakeStream(getter._recording_callback, [])

        def blocking_open(**kwargs):
            instance.open_kwargs = kwargs
            instance.stream = stream
            open_entered.set()
            self.assertTrue(allow_open_to_return.wait(timeout=2))
            return stream

        instance.open = blocking_open
        output_queue = queue.SimpleQueue()
        loop_thread = threading.Thread(target=getter.loop, args=(output_queue,))
        handler_exited = threading.Event()

        def invoke_handler():
            try:
                getter._exit_handler(2, None)
            except SystemExit as exc:
                self.assertEqual(exc.code, 0)
                handler_exited.set()

        loop_thread.start()
        self.assertTrue(open_entered.wait(timeout=2))
        handler_thread = threading.Thread(target=invoke_handler)
        handler_thread.start()
        self.assertTrue(getter.stopping.wait(timeout=2))
        allow_open_to_return.set()
        loop_thread.join(timeout=2)
        handler_thread.join(timeout=2)

        self.assertFalse(loop_thread.is_alive())
        self.assertFalse(handler_thread.is_alive())
        self.assertTrue(handler_exited.is_set())
        self.assertEqual(stream.started, 0)
        self.assertEqual(stream.closed, 1)
        self.assertEqual(instance.terminated, 1)
        self.assertIsNone(getter.stream)
        self.assertEqual(_drain(output_queue), [None])

    def test_repeated_signal_exits_without_reentering_cleanup(self):
        getter, _, _ = _make_getter()
        getter.stopping.set()
        getter.signal_received.set()

        with mock.patch.object(getter, '_cleanup') as cleanup:
            with self.assertRaises(SystemExit) as exit_context:
                getter._exit_handler(2, None)

        self.assertEqual(exit_context.exception.code, 0)
        cleanup.assert_not_called()

    def test_signal_waits_for_partial_debug_wav_to_be_flushed(self):
        samples = np.full(SAMPLES_PER_FRAME, 0.25, dtype=np.float32)
        stream_started = threading.Event()

        class LiveStream(_FakeStream):

            def start_stream(self):
                self.started += 1
                self.active = True
                self.callback(samples.tobytes(), SAMPLES_PER_FRAME, object(), 0)
                stream_started.set()

        with tempfile.TemporaryDirectory() as temp_dir:
            getter, instance, _ = _make_getter(debug_audio_dir=temp_dir,
                                               debug_clip_duration=1.0,
                                               debug_max_clips=1)
            stream = LiveStream(getter._recording_callback, [])

            def open_live_stream(**kwargs):
                instance.open_kwargs = kwargs
                instance.stream = stream
                return stream

            instance.open = open_live_stream
            output_queue = queue.SimpleQueue()
            loop_thread = threading.Thread(target=getter.loop, args=(output_queue,))
            loop_thread.start()
            self.assertTrue(stream_started.wait(timeout=2))

            with self.assertRaises(SystemExit) as exit_context:
                getter._exit_handler(2, None)
            loop_thread.join(timeout=2)

            self.assertEqual(exit_context.exception.code, 0)
            self.assertFalse(loop_thread.is_alive())
            self.assertTrue(getter.finished.is_set())
            wav_paths = list(Path(temp_dir).glob('*.wav'))
            self.assertEqual(len(wav_paths), 1)
            with wave.open(str(wav_paths[0]), 'rb') as wav_file:
                self.assertEqual(wav_file.getnframes(), SAMPLES_PER_FRAME)
            self.assertFalse(list(Path(temp_dir).glob('*.part')))


if __name__ == '__main__':
    unittest.main()
