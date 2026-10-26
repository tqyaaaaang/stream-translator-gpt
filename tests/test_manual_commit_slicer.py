import math
import unittest
from unittest import mock

import numpy as np

from stream_translator_gpt.slicer.manual_commit_slicer import ManualCommitBoundaryDetector
from stream_translator_gpt.utils.common import FRAME_DURATION, SAMPLE_RATE, SAMPLES_PER_FRAME


class _FakeVAD:

    def __init__(self, probability=0.9):
        self.probability = probability
        self.last_audio = None
        self.calls = 0
        self.reset_count = 0

    def get_speech_prob(self, audio):
        self.last_audio = audio.copy()
        self.calls += 1
        return self.probability

    def reset_states(self):
        self.reset_count += 1


def _detector(**overrides):
    options = dict(min_audio_length=0.5, max_audio_length=30.0, target_audio_length=5.0,
                   continuous_no_speech_threshold=1.0, dynamic_no_speech_threshold=True,
                   prefix_retention_length=0.5, vad_threshold=0.35, dynamic_vad_threshold=False,
                   vad=_FakeVAD())
    options.update(overrides)
    return ManualCommitBoundaryDetector(**options)


class ManualCommitBoundaryDetectorTest(unittest.TestCase):

    def setUp(self):
        self.frame = np.ones(SAMPLES_PER_FRAME, dtype=np.float32)

    def _feed_frames(self, detector, count, probability=None):
        if probability is not None:
            detector.vad.probability = probability
        return [detector.feed(self.frame) for _ in range(count)]

    def test_continuous_speech_commits_at_maximum_frame_boundary(self):
        detector = _detector(max_audio_length=1.0)
        self.assertFalse(any(self._feed_frames(detector, 31)))
        self.assertTrue(detector.feed(self.frame))
        self.assertEqual(detector.total_samples, 32 * SAMPLES_PER_FRAME)
        self.assertEqual(detector.last_boundary_sample, detector.total_samples)
        self.assertEqual(detector.segment_samples, 0)
        self.assertEqual(detector.utterance_samples, 0)
        self.assertEqual(detector.boundary_count, 1)
        self.assertFalse(detector.feed(self.frame))
        self.assertEqual(detector.segment_samples, SAMPLES_PER_FRAME)
        self.assertEqual(detector.total_samples, 33 * SAMPLES_PER_FRAME)

    def test_maximum_is_capped_at_thirty_seconds_with_one_frame_quantization(self):
        detector = _detector(max_audio_length=90)
        self.assertEqual(detector.max_audio_length, 30)
        self.assertFalse(any(self._feed_frames(detector, 937)))
        self.assertTrue(detector.feed(self.frame))
        self.assertAlmostEqual(detector.total_samples / SAMPLE_RATE, 30.016)
        self.assertLess(detector.total_samples / SAMPLE_RATE - 30, FRAME_DURATION)

    def test_static_pause_must_be_continuous_and_meet_minimum_length(self):
        detector = _detector(min_audio_length=1.0, dynamic_no_speech_threshold=False,
                             continuous_no_speech_threshold=0.064)
        self._feed_frames(detector, 1)
        self.assertFalse(any(self._feed_frames(detector, 2, probability=0)))
        self._feed_frames(detector, 27, probability=0.9)
        self.assertFalse(detector.feed(self.frame))
        detector.vad.probability = 0
        self.assertFalse(detector.feed(self.frame))
        self.assertTrue(detector.feed(self.frame))

    def test_speech_resets_consecutive_pause(self):
        detector = _detector(min_audio_length=0.032, dynamic_no_speech_threshold=False,
                             continuous_no_speech_threshold=0.096)
        detector.feed(self.frame)
        self.assertFalse(any(self._feed_frames(detector, 2, probability=0)))
        self.assertFalse(any(self._feed_frames(detector, 1, probability=0.9)))
        self.assertFalse(any(self._feed_frames(detector, 2, probability=0)))
        self.assertTrue(detector.feed(self.frame))

    def test_dynamic_threshold_needs_less_pause_after_long_speech(self):
        short = _detector()
        long = _detector()
        self._feed_frames(short, 31)
        self._feed_frames(long, 250)
        self.assertFalse(any(self._feed_frames(short, 4, probability=0)))
        self.assertEqual(self._feed_frames(long, 3, probability=0), [False, False, True])

    def test_leading_silence_does_not_count_toward_minimum_or_dynamic_target(self):
        detector = _detector()
        self.assertFalse(any(self._feed_frames(detector, 500, probability=0)))
        self.assertEqual(detector.utterance_samples, 0)
        self._feed_frames(detector, 1, probability=0.9)
        self.assertFalse(any(self._feed_frames(detector, 16, probability=0)))
        self.assertEqual(detector.segment_samples, 517 * SAMPLES_PER_FRAME)
        self.assertEqual(detector.utterance_samples, 17 * SAMPLES_PER_FRAME)
        self.assertEqual(detector.boundary_count, 0)

    def test_pure_silence_has_only_maximum_length_maintenance_commits(self):
        detector = _detector(max_audio_length=1.0, min_audio_length=0.032,
                             continuous_no_speech_threshold=0)
        results = self._feed_frames(detector, 65, probability=0)
        self.assertEqual([index + 1 for index, boundary in enumerate(results) if boundary], [32, 64])
        self.assertEqual(detector.total_samples, 65 * SAMPLES_PER_FRAME)
        self.assertEqual(detector.last_boundary_sample, 64 * SAMPLES_PER_FRAME)
        self.assertEqual(detector.segment_samples, SAMPLES_PER_FRAME)

    def test_prefix_length_does_not_retain_repeat_or_modify_audio(self):
        outputs = []
        for prefix in (0, 0.5, 20):
            detector = _detector(prefix_retention_length=prefix, max_audio_length=0.512)
            results = self._feed_frames(detector, 33)
            outputs.append(results)
            self.assertEqual(detector.total_samples, 33 * SAMPLES_PER_FRAME)
            self.assertEqual(detector.vad.calls, 33)
            self.assertEqual(detector.segment_samples, SAMPLES_PER_FRAME)
            self.assertFalse(hasattr(detector, 'audio_buffer'))
            self.assertFalse(hasattr(detector, 'prefix_audio_buffer'))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        np.testing.assert_array_equal(self.frame, np.ones(SAMPLES_PER_FRAME, dtype=np.float32))

    def test_short_tail_is_padded_only_for_vad_and_counts_actual_source_samples(self):
        detector = _detector(min_audio_length=0.001, max_audio_length=0.04)
        self.assertFalse(detector.feed(self.frame))
        tail = np.arange(128, dtype=np.float32)
        self.assertTrue(detector.feed(tail))
        self.assertEqual(detector.total_samples, 640)
        self.assertEqual(detector.last_boundary_sample, 640)
        self.assertEqual(detector.vad.calls, 2)
        np.testing.assert_array_equal(detector.vad.last_audio[:128], tail)
        np.testing.assert_array_equal(detector.vad.last_audio[128:], np.zeros(384, dtype=np.float32))
        np.testing.assert_array_equal(tail, np.arange(128, dtype=np.float32))

    def test_empty_tail_does_not_advance_time_or_trigger_vad(self):
        detector = _detector()
        detector.feed(self.frame)
        self.assertFalse(detector.feed(np.empty(0, dtype=np.float32)))
        self.assertEqual(detector.total_samples, SAMPLES_PER_FRAME)
        self.assertEqual(detector.vad.calls, 1)

    def test_forced_idle_boundary_resets_segment_without_resetting_source_clock(self):
        detector = _detector(max_audio_length=1.0)
        self._feed_frames(detector, 20)
        self.assertTrue(detector.mark_boundary())
        self.assertEqual(detector.total_samples, 20 * SAMPLES_PER_FRAME)
        self.assertEqual(detector.last_boundary_sample, detector.total_samples)
        self.assertEqual(detector.segment_samples, 0)
        self.assertEqual(detector.utterance_samples, 0)
        self.assertFalse(any(self._feed_frames(detector, 31, probability=0)))
        self.assertEqual(detector.utterance_samples, 0)
        self.assertTrue(detector.feed(self.frame))
        self.assertEqual(detector.last_boundary_sample, 52 * SAMPLES_PER_FRAME)
        self.assertEqual(detector.boundary_count, 2)

    def test_marking_empty_or_already_reported_boundary_is_idempotent(self):
        detector = _detector(max_audio_length=0.512)
        self.assertFalse(detector.mark_boundary())
        self.assertEqual(detector.boundary_count, 0)
        self._feed_frames(detector, 16)
        self.assertEqual(detector.boundary_count, 1)
        self.assertFalse(detector.mark_boundary())
        self.assertFalse(detector.mark_boundary())
        self.assertEqual(detector.boundary_count, 1)
        detector.feed(self.frame)
        self.assertTrue(detector.mark_boundary())
        self.assertFalse(detector.mark_boundary())
        self.assertEqual(detector.boundary_count, 2)

    def test_hysteresis_and_segment_reset_match_slicer_threshold_rules(self):
        detector = _detector(max_audio_length=0.128, min_audio_length=0.032,
                             dynamic_no_speech_threshold=False, continuous_no_speech_threshold=0.032)
        detector.feed(self.frame)
        self.assertFalse(any(self._feed_frames(detector, 2, probability=0.3)))
        self.assertTrue(detector.feed(self.frame))
        self.assertFalse(detector.feed(self.frame))
        self.assertEqual(detector.utterance_samples, 0)
        self.assertFalse(any(self._feed_frames(detector, 1, probability=0.9)))
        self.assertTrue(any(self._feed_frames(detector, 1, probability=0.2)))

    def test_dynamic_vad_adapts_on_global_source_frames_across_commits(self):
        detector = _detector(dynamic_vad_threshold=True, max_audio_length=1.0)
        self._feed_frames(detector, 1091, probability=0.1)
        self.assertEqual(detector.vad_threshold, 0.35)
        detector.feed(self.frame)
        self.assertAlmostEqual(detector.vad_threshold, 0.35 * 0.25 + 0.1 * 0.75)
        self.assertAlmostEqual(detector.vad_neg_threshold, detector.vad_threshold * 0.7)
        self.assertGreater(detector.boundary_count, 30)

    def test_dynamic_vad_threshold_stays_within_existing_limits(self):
        for probability, expected in ((0, 0.0001), (1, 0.6)):
            with self.subTest(probability=probability):
                detector = _detector(dynamic_vad_threshold=True)
                self._feed_frames(detector, 3000, probability=probability)
                self.assertAlmostEqual(detector.vad_threshold, expected)

    def test_periodic_vad_reset_uses_source_samples_including_short_tail(self):
        detector = _detector()
        self._feed_frames(detector, 9374)
        self.assertEqual(detector.vad.reset_count, 0)
        detector.feed(self.frame[:128])
        self.assertEqual(detector.vad.reset_count, 0)
        detector.feed(self.frame[:384])
        self.assertEqual(detector.vad.reset_count, 1)
        self.assertEqual(detector.total_samples, 300 * SAMPLE_RATE)

    def test_invalid_configuration_fails_before_loading_vad(self):
        invalid = [
            dict(min_audio_length=value) for value in (0, -1, math.nan, math.inf)
        ] + [
            dict(max_audio_length=value) for value in (0, -1, math.nan, math.inf)
        ] + [
            dict(target_audio_length=value) for value in (0, -1, math.nan, math.inf)
        ] + [
            {name: value} for name in ('continuous_no_speech_threshold', 'prefix_retention_length')
            for value in (-1, math.nan, math.inf)
        ] + [
            dict(vad_threshold=value) for value in (-1, 1.1, math.nan, math.inf)
        ] + [dict(min_audio_length=2, max_audio_length=1), dict(min_audio_length=31, max_audio_length=90)]
        for options in invalid:
            with self.subTest(options=options), mock.patch(
                    'stream_translator_gpt.slicer.manual_commit_slicer.VAD') as vad:
                with self.assertRaises(ValueError):
                    _detector(**options, vad=None)
                vad.assert_not_called()

    def test_invalid_frame_shape_does_not_advance_source_clock(self):
        detector = _detector()
        for frame in (np.ones((2, 256)), np.ones(513), np.array(1)):
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                detector.feed(frame)
        self.assertEqual(detector.total_samples, 0)
        self.assertEqual(detector.vad.calls, 0)


if __name__ == '__main__':
    unittest.main()
