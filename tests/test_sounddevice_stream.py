import queue
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from stream_translator_gpt.audio_getter import SoundDeviceAudioGetter, _DebugAudioDumper
from stream_translator_gpt.common import SAMPLE_RATE, SAMPLES_PER_FRAME


class _FakeInputStream:

    def __init__(self, factory, callback):
        self.factory = factory
        self.callback = callback

    def __enter__(self):
        self.factory.entered = True
        for audio, status, mutate_after_callback in self.factory.blocks:
            self.callback(audio, len(audio), object(), status)
            if mutate_after_callback:
                audio.fill(-0.75)
        self.callback.__self__.recording_queue.put_nowait(None)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.factory.exited = True
        return False


class _FakeInputStreamFactory:

    def __init__(self, blocks):
        self.blocks = blocks
        self.kwargs = None
        self.entered = False
        self.exited = False

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return _FakeInputStream(self, kwargs['callback'])


def _make_getter(input_stream, *, native_rate=SAMPLE_RATE, channels=1, interval=0.5):
    sounddevice = types.ModuleType('sounddevice')
    sounddevice.default = types.SimpleNamespace(device=(3, 4))
    sounddevice.query_devices = mock.Mock(return_value={
        'name': 'Fake input',
        'max_input_channels': channels,
        'default_samplerate': native_rate,
    })
    sounddevice.InputStream = input_stream

    with mock.patch.dict(sys.modules, {'sounddevice': sounddevice}):
        getter = SoundDeviceAudioGetter(device_index=0, interval=interval)
    return getter


def _drain(audio_queue):
    values = []
    while not audio_queue.empty():
        values.append(audio_queue.get_nowait())
    return values


class SoundDeviceStreamTest(unittest.TestCase):

    def test_callback_copies_status_block_and_still_processes_it(self):
        callback_buffer = np.full((SAMPLES_PER_FRAME, 1), 0.25, dtype=np.float32)
        input_stream = _FakeInputStreamFactory([(callback_buffer, 'input overflow', True)])
        getter = _make_getter(input_stream, interval=SAMPLES_PER_FRAME / SAMPLE_RATE)
        output_queue = queue.SimpleQueue()

        with mock.patch('stream_translator_gpt.audio_getter.logger.warning') as warning:
            getter.loop(output_queue)

        output = _drain(output_queue)
        self.assertEqual(len(output), 2)
        np.testing.assert_array_equal(output[0], np.full(SAMPLES_PER_FRAME, 0.25, dtype=np.float32))
        self.assertIsNone(output[1])
        warning.assert_any_call('Sounddevice recorder returned status=%s', 'input overflow')
        self.assertTrue(input_stream.entered)
        self.assertTrue(input_stream.exited)
        self.assertEqual(input_stream.kwargs['device'], 0)
        self.assertEqual(input_stream.kwargs['samplerate'], SAMPLE_RATE)
        self.assertEqual(input_stream.kwargs['blocksize'], SAMPLES_PER_FRAME)

    def test_native_stereo_stream_is_downmixed_resampled_and_split_into_512_samples(self):
        native_rate = 48000
        interval = 0.064
        native_frames = round(native_rate * interval)
        stereo = np.column_stack((
            np.full(native_frames, -0.25, dtype=np.float32),
            np.full(native_frames, 0.75, dtype=np.float32),
        ))
        input_stream = _FakeInputStreamFactory([(stereo, None, False)])
        getter = _make_getter(input_stream, native_rate=native_rate, channels=2, interval=interval)
        output_queue = queue.SimpleQueue()

        getter.loop(output_queue)

        output = _drain(output_queue)
        self.assertIsNone(output[-1])
        chunks = output[:-1]
        self.assertEqual([chunk.shape for chunk in chunks], [(SAMPLES_PER_FRAME,), (SAMPLES_PER_FRAME,)])
        self.assertTrue(all(chunk.dtype == np.float32 for chunk in chunks))
        np.testing.assert_allclose(np.concatenate(chunks), 0.25, atol=1e-6)
        self.assertEqual(input_stream.kwargs['samplerate'], native_rate)
        self.assertEqual(input_stream.kwargs['channels'], 2)
        self.assertEqual(input_stream.kwargs['blocksize'], native_frames)
        self.assertIs(input_stream.kwargs['dtype'], np.float32)


class DebugAudioDumperTest(unittest.TestCase):

    def test_wav_clips_are_clipped_duration_bounded_and_limited(self):
        clip_samples = 4
        samples = np.array([
            -2.0,
            -1.0,
            -0.5,
            2.0,
            0.0,
            0.25,
            0.5,
            0.75,
            0.1,
            0.2,
            0.3,
            0.4,
        ],
                           dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'nested' / 'clips'
            dumper = _DebugAudioDumper(str(output_dir), clip_samples / SAMPLE_RATE, max_clips=2)
            dumper.add(samples[:3])
            dumper.add(samples[3:])
            dumper.close()
            dumper.close()

            wav_paths = sorted(output_dir.glob('*.wav'))
            self.assertEqual(len(wav_paths), 2)
            pcm_clips = []
            for wav_path in wav_paths:
                with wave.open(str(wav_path), 'rb') as wav_file:
                    self.assertEqual(wav_file.getnchannels(), 1)
                    self.assertEqual(wav_file.getsampwidth(), 2)
                    self.assertEqual(wav_file.getframerate(), SAMPLE_RATE)
                    self.assertEqual(wav_file.getnframes(), clip_samples)
                    pcm_clips.append(np.frombuffer(wav_file.readframes(clip_samples), dtype='<i2'))

            expected = np.array([-32768, -32768, -16384, 32767, 0, 8192, 16384, 24576], dtype=np.int16)
            np.testing.assert_array_equal(np.concatenate(pcm_clips), expected)
            self.assertEqual(dumper.clip_count, 2)
            self.assertFalse(list(output_dir.rglob('*.part')))

    def test_failed_atomic_replace_removes_part_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dumper = _DebugAudioDumper(str(output_dir), 4 / SAMPLE_RATE, max_clips=1)

            with mock.patch('stream_translator_gpt.audio_getter.os.replace', side_effect=OSError('disk error')):
                with mock.patch('stream_translator_gpt.audio_getter.logger.exception') as logger_exception:
                    dumper.add(np.ones(4, dtype=np.float32))

            logger_exception.assert_called_once_with('Failed to write device debug audio clip')
            self.assertFalse(list(output_dir.iterdir()))
            self.assertTrue(dumper.closed)

    def test_close_writes_one_partial_clip_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dumper = _DebugAudioDumper(str(output_dir), 8 / SAMPLE_RATE, max_clips=2)
            dumper.add(np.array([0.1, 0.2, 0.3], dtype=np.float32))

            dumper.close()
            dumper.close()

            wav_paths = list(output_dir.glob('*.wav'))
            self.assertEqual(len(wav_paths), 1)
            with wave.open(str(wav_paths[0]), 'rb') as wav_file:
                self.assertEqual(wav_file.getnframes(), 3)
            self.assertFalse(list(output_dir.rglob('*.part')))


if __name__ == '__main__':
    unittest.main()
