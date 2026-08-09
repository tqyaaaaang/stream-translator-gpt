import queue
import unittest
from unittest import mock

import numpy as np
import tyro

from stream_translator_gpt.audio_getter import LocalFileAudioGetter, _RealtimeAudioPacer, _read_ffmpeg_loop
from stream_translator_gpt.common import SAMPLE_RATE, SAMPLES_PER_FRAME
from stream_translator_gpt.main import Config, _create_audio_getter


class _FakeClock:

    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class _FakeStopEvent:

    def __init__(self, clock, overshoot=0.0, stop_on_wait=False, progress_ratio=1.0):
        self.clock = clock
        self.overshoot = overshoot
        self.stop_on_wait = stop_on_wait
        self.progress_ratio = progress_ratio
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def wait(self, delay):
        self.waits.append(delay)
        self.clock.now += delay * self.progress_ratio + self.overshoot
        if self.stop_on_wait:
            self.stopped = True
        return self.stopped


class _SequenceReader:

    def __init__(self, values, clock=None, read_time=0.0):
        self.values = iter(values)
        self.clock = clock
        self.read_time = read_time

    def read(self, size):
        if self.clock is not None:
            self.clock.now += self.read_time
        return next(self.values)


class _TimestampQueue:

    def __init__(self, clock):
        self.clock = clock
        self.items = []

    def put(self, item):
        self.items.append((self.clock.now, item))


def _process(values, clock, event, *, read_time=0.0, pad_final_block=False):
    process = mock.Mock(stdout=_SequenceReader(values, clock=clock, read_time=read_time))
    output_queue = _TimestampQueue(clock)
    pacer = _RealtimeAudioPacer(event, clock=clock)
    block_count = _read_ffmpeg_loop(process,
                                    SAMPLES_PER_FRAME * np.dtype(np.float32).itemsize,
                                    output_queue,
                                    source_name='test file',
                                    pacer=pacer,
                                    pad_final_block=pad_final_block)
    return pacer, output_queue, block_count


class RealtimeFilePacingTest(unittest.TestCase):

    def test_absolute_deadlines_prevent_sleep_overshoot_from_accumulating(self):
        clock = _FakeClock()
        event = _FakeStopEvent(clock, overshoot=0.002)
        block = np.ones(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()

        pacer, output, count = _process([block, block, block, b''], clock, event)

        self.assertEqual(count, 3)
        self.assertEqual(pacer.samples_emitted, SAMPLES_PER_FRAME * 3)
        expected_deadlines = [0.032, 0.064, 0.096]
        for (timestamp, _), deadline in zip(output.items, expected_deadlines):
            self.assertAlmostEqual(timestamp - deadline, 0.002)
        np.testing.assert_allclose(event.waits, [0.032, 0.030, 0.030], atol=1e-12)

    def test_decode_time_is_counted_without_repeated_fixed_sleeps(self):
        clock = _FakeClock()
        event = _FakeStopEvent(clock)
        block = np.ones(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()

        _, output, count = _process([block, block, b''], clock, event, read_time=0.005)

        self.assertEqual(count, 2)
        np.testing.assert_allclose([timestamp for timestamp, _ in output.items], [0.032, 0.064], atol=1e-12)
        np.testing.assert_allclose(event.waits, [0.027, 0.027], atol=1e-12)

    def test_early_timeout_return_is_rechecked_until_the_deadline(self):
        clock = _FakeClock()
        event = _FakeStopEvent(clock, progress_ratio=0.5)
        pacer = _RealtimeAudioPacer(event, clock=clock)

        self.assertTrue(pacer.wait_for_block(SAMPLES_PER_FRAME))

        self.assertGreater(len(event.waits), 1)
        self.assertAlmostEqual(clock.now, SAMPLES_PER_FRAME / SAMPLE_RATE)

    def test_short_final_block_is_zero_padded_but_paced_by_valid_samples(self):
        clock = _FakeClock()
        event = _FakeStopEvent(clock)
        full = np.arange(SAMPLES_PER_FRAME, dtype=np.float32)
        tail = np.arange(17, dtype=np.float32) + 1000

        pacer, output, count = _process([full.tobytes(), tail.tobytes(), b''],
                                        clock,
                                        event,
                                        pad_final_block=True)

        self.assertEqual(count, 2)
        self.assertEqual(pacer.samples_emitted, SAMPLES_PER_FRAME + len(tail))
        self.assertAlmostEqual(output.items[0][0], SAMPLES_PER_FRAME / SAMPLE_RATE)
        self.assertAlmostEqual(output.items[1][0], (SAMPLES_PER_FRAME + len(tail)) / SAMPLE_RATE)
        padded_tail = output.items[1][1]
        self.assertEqual(padded_tail.shape, (SAMPLES_PER_FRAME,))
        np.testing.assert_array_equal(padded_tail[:len(tail)], tail)
        np.testing.assert_array_equal(padded_tail[len(tail):], 0)

    def test_fragmented_pipe_reads_are_combined_before_emitting(self):
        clock = _FakeClock()
        event = _FakeStopEvent(clock)
        first = np.arange(100, dtype=np.float32)
        second = np.arange(100, SAMPLES_PER_FRAME, dtype=np.float32)

        pacer, output, count = _process([first.tobytes(), second.tobytes(), b''],
                                        clock,
                                        event,
                                        pad_final_block=True)

        self.assertEqual(count, 1)
        self.assertEqual(pacer.samples_emitted, SAMPLES_PER_FRAME)
        np.testing.assert_array_equal(output.items[0][1], np.arange(SAMPLES_PER_FRAME, dtype=np.float32))

    def test_offline_reader_keeps_legacy_short_block_discard_behavior(self):
        full = np.ones(SAMPLES_PER_FRAME, dtype=np.float32)
        tail = np.ones(17, dtype=np.float32)
        process = mock.Mock(stdout=_SequenceReader([full.tobytes(), tail.tobytes(), b'']))
        output_queue = queue.SimpleQueue()

        with mock.patch('stream_translator_gpt.audio_getter.logger.warning') as warning:
            count = _read_ffmpeg_loop(process, full.nbytes, output_queue, source_name='test file')

        self.assertEqual(count, 1)
        np.testing.assert_array_equal(output_queue.get_nowait(), full)
        self.assertTrue(output_queue.empty())
        warning.assert_called_once()

    def test_large_lag_rebases_timeline_instead_of_bursting(self):
        clock = _FakeClock()
        event = _FakeStopEvent(clock)
        pacer = _RealtimeAudioPacer(event, clock=clock, max_lag=0.1)
        clock.now = 1.0

        with mock.patch('stream_translator_gpt.audio_getter.logger.info') as info:
            self.assertTrue(pacer.wait_for_block(SAMPLES_PER_FRAME))
            self.assertTrue(pacer.wait_for_block(SAMPLES_PER_FRAME))

        self.assertEqual(len(event.waits), 1)
        self.assertAlmostEqual(event.waits[0], SAMPLES_PER_FRAME / SAMPLE_RATE)
        info.assert_called_once_with(
            'Realtime %s fell %.3fs behind; resuming without a catch-up burst',
            'local file playback',
            1.0 - SAMPLES_PER_FRAME / SAMPLE_RATE)

    def test_stop_event_interrupts_wait_without_emitting_block(self):
        clock = _FakeClock()
        event = _FakeStopEvent(clock, stop_on_wait=True)
        block = np.ones(SAMPLES_PER_FRAME, dtype=np.float32).tobytes()

        pacer, output, count = _process([block, b''], clock, event)

        self.assertEqual(count, 0)
        self.assertEqual(pacer.samples_emitted, 0)
        self.assertEqual(output.items, [])


class RealtimeFileConfigTest(unittest.TestCase):

    def test_cli_defaults_to_offline_and_accepts_realtime_mode(self):
        offline = tyro.cli(Config, args=['clip.mp3'])
        realtime = tyro.cli(Config, args=['clip.mp3', '--local-file-mode', 'realtime'])

        self.assertEqual(offline.local_file_mode, 'offline')
        self.assertEqual(realtime.local_file_mode, 'realtime')

    def test_factory_passes_selected_mode_to_local_file_getter(self):
        sentinel = object()
        with mock.patch('stream_translator_gpt.main.LocalFileAudioGetter', return_value=sentinel) as getter_cls:
            result = _create_audio_getter(Config(url='clip.mp3', local_file_mode='realtime'))

        self.assertIs(result, sentinel)
        getter_cls.assert_called_once_with(file_path='clip.mp3', realtime=True)


class LocalFileProcessLifecycleTest(unittest.TestCase):

    @staticmethod
    def _patch_ffmpeg(process):
        input_builder = mock.Mock()
        output_builder = mock.Mock()
        input_builder.output.return_value = output_builder
        output_builder.run_async.return_value = process
        return mock.patch('stream_translator_gpt.audio_getter.ffmpeg.input', return_value=input_builder)

    def test_normal_eof_waits_for_ffmpeg_without_killing_it(self):
        block = np.ones(SAMPLES_PER_FRAME, dtype=np.float32)
        process = mock.Mock(stdout=_SequenceReader([block.tobytes(), b'']))
        process.wait.return_value = 0
        output_queue = queue.SimpleQueue()

        with self._patch_ffmpeg(process):
            LocalFileAudioGetter('clip.mp3').loop(output_queue)

        process.wait.assert_called_once_with(timeout=5.0)
        process.kill.assert_not_called()
        np.testing.assert_array_equal(output_queue.get_nowait(), block)
        self.assertIsNone(output_queue.get_nowait())
        self.assertTrue(output_queue.empty())

    def test_read_failure_kills_and_reaps_ffmpeg(self):

        class BrokenReader:

            def read(self, size):
                raise OSError('pipe failed')

        process = mock.Mock(stdout=BrokenReader())
        process.poll.return_value = None
        process.wait.return_value = -9
        output_queue = queue.SimpleQueue()

        with self._patch_ffmpeg(process):
            with self.assertRaisesRegex(OSError, 'pipe failed'):
                LocalFileAudioGetter('clip.mp3').loop(output_queue)

        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5.0)
        self.assertTrue(output_queue.empty())

    def test_nonzero_ffmpeg_exit_is_reported_after_reaping(self):
        process = mock.Mock(stdout=_SequenceReader([b'']))
        process.wait.return_value = 23
        process.poll.return_value = 23
        output_queue = queue.SimpleQueue()

        with self._patch_ffmpeg(process):
            with self.assertRaisesRegex(RuntimeError, 'exited with status 23'):
                LocalFileAudioGetter('clip.mp3').loop(output_queue)

        process.wait.assert_called_once_with(timeout=5.0)
        process.kill.assert_not_called()
        self.assertTrue(output_queue.empty())


if __name__ == '__main__':
    unittest.main()
