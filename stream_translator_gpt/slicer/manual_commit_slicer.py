"""Local commit boundaries for a continuous audio stream, without slicing audio."""

import collections
import logging
import math

import numpy as np

from .audio_slicer import VAD, _get_dynamic_no_speech_threshold, _get_neg_threshold
from ..utils.common import FRAME_DURATION, SAMPLE_RATE, SAMPLES_PER_FRAME


logger = logging.getLogger('main')

MAX_MANUAL_COMMIT_AUDIO_LENGTH = 30.0


def _finite_number(name: str, value: float, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        bound = 'zero or greater' if allow_zero else 'greater than zero'
        raise ValueError(f'{name} must be a finite value of {bound}')
    return number


class ManualCommitBoundaryDetector:
    """Identify commit points while the caller sends every source sample once.

    ``feed`` accepts one frame of at most 512 samples and returns whether to
    commit after that frame. The maximum is consequently quantized to a frame
    boundary: a 30-second limit can produce a 30.016-second segment. A shorter
    final frame is padded only for VAD, never for source-time accounting.

    The maximum length counts all audio since the previous boundary. Minimum
    length and the dynamic pause threshold count from the first speech frame,
    so leading silence does not cause an immediate cut after speech begins.
    Silence-only segments are committed only at the maximum length. EOF and
    the service's minimum commit length remain the caller's responsibility.

    ``prefix_retention_length`` is accepted for configuration compatibility;
    continuous streams must not retain or repeat audio across commits.
    """

    def __init__(self, min_audio_length: float, max_audio_length: float, target_audio_length: float,
                 continuous_no_speech_threshold: float, dynamic_no_speech_threshold: bool,
                 prefix_retention_length: float, vad_threshold: float, dynamic_vad_threshold: bool,
                 *, vad=None):
        self.min_audio_length = _finite_number('min_audio_length', min_audio_length)
        self.max_audio_length = min(_finite_number('max_audio_length', max_audio_length),
                                    MAX_MANUAL_COMMIT_AUDIO_LENGTH)
        if self.min_audio_length > self.max_audio_length:
            raise ValueError('min_audio_length must not exceed the effective max_audio_length '
                             f'({self.max_audio_length:g} seconds)')
        self.target_audio_length = _finite_number('target_audio_length', target_audio_length)
        self.continuous_no_speech_threshold = _finite_number('continuous_no_speech_threshold',
                                                             continuous_no_speech_threshold,
                                                             allow_zero=True)
        _finite_number('prefix_retention_length', prefix_retention_length, allow_zero=True)
        self.vad_threshold = _finite_number('vad_threshold', vad_threshold, allow_zero=True)
        if self.vad_threshold > 1:
            raise ValueError('vad_threshold must be between zero and one')
        self.vad_neg_threshold = _get_neg_threshold(self.vad_threshold)
        self.dynamic_no_speech_threshold = dynamic_no_speech_threshold
        self.dynamic_vad_threshold = dynamic_vad_threshold
        self.vad = vad if vad is not None else VAD()

        self._total_samples = 0
        self._last_boundary_sample = 0
        self._boundary_count = 0
        self._segment_samples = 0
        self._utterance_samples = 0
        self._speech_samples = 0
        self._continuous_no_speech_samples = 0

        # Match the existing slicer's rounded source-frame intervals. Wall time
        # and network pacing must not change VAD adaptation or commit points.
        self._vad_lookback_frames = round(30 / FRAME_DURATION)
        self._vad_prob_buffer = collections.deque(maxlen=self._vad_lookback_frames)
        self._vad_recalc_interval_samples = round(5 / FRAME_DURATION) * SAMPLES_PER_FRAME
        self._next_vad_recalc_sample = self._vad_recalc_interval_samples
        self._vad_reset_interval_samples = round(300 / FRAME_DURATION) * SAMPLES_PER_FRAME
        self._next_vad_reset_sample = self._vad_reset_interval_samples

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def segment_samples(self) -> int:
        return self._segment_samples

    @property
    def utterance_samples(self) -> int:
        return self._utterance_samples

    @property
    def last_boundary_sample(self) -> int:
        return self._last_boundary_sample

    @property
    def boundary_count(self) -> int:
        return self._boundary_count

    def mark_boundary(self) -> bool:
        """Reset segment state for a caller-forced commit, preserving source time.

        Calling this after ``feed`` already returned True is harmless: a source
        position is counted as a boundary only once. Idle and EOF commits must
        use this method so the next utterance never inherits committed audio.
        """
        if self._last_boundary_sample == self._total_samples:
            return False
        self._last_boundary_sample = self._total_samples
        self._boundary_count += 1
        self._segment_samples = 0
        self._utterance_samples = 0
        self._speech_samples = 0
        self._continuous_no_speech_samples = 0
        return True

    def _update_vad_threshold(self, speech_probability: float) -> None:
        if not self.dynamic_vad_threshold:
            return
        self._vad_prob_buffer.append(speech_probability)
        if self._total_samples < self._next_vad_recalc_sample:
            return
        self._next_vad_recalc_sample += self._vad_recalc_interval_samples
        if self._total_samples < self._vad_lookback_frames * SAMPLES_PER_FRAME:
            return
        old_threshold = self.vad_threshold
        recent_threshold = np.quantile(self._vad_prob_buffer, 0.5, method='linear')
        self.vad_threshold = min(0.6, max(0.0001, old_threshold * 0.25 + recent_threshold * 0.75))
        self.vad_neg_threshold = _get_neg_threshold(self.vad_threshold)
        logger.debug('Updated manual-commit VAD threshold from %.4f to %.4f (negative threshold %.4f)',
                     old_threshold, self.vad_threshold, self.vad_neg_threshold)

    def _should_commit(self) -> bool:
        if self._segment_samples / SAMPLE_RATE >= self.max_audio_length:
            return True
        if not self._speech_samples or not self._continuous_no_speech_samples:
            return False
        utterance_length = self._utterance_samples / SAMPLE_RATE
        if utterance_length < self.min_audio_length:
            return False
        pause_threshold = self.continuous_no_speech_threshold
        if self.dynamic_no_speech_threshold:
            pause_threshold = _get_dynamic_no_speech_threshold(utterance_length,
                                                               pause_threshold * 2,
                                                               self.target_audio_length)
        return self._continuous_no_speech_samples / SAMPLE_RATE >= pause_threshold

    def feed(self, audio: np.ndarray) -> bool:
        """Consume source-frame metadata and return a boundary at its end."""
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or len(samples) > SAMPLES_PER_FRAME:
            raise ValueError(f'audio must be one mono frame of at most {SAMPLES_PER_FRAME} samples')
        sample_count = len(samples)
        if not sample_count:
            return False
        vad_audio = (np.pad(samples, (0, SAMPLES_PER_FRAME - sample_count))
                     if sample_count < SAMPLES_PER_FRAME else samples)
        speech_probability = self.vad.get_speech_prob(vad_audio)
        is_speech = speech_probability > (self.vad_neg_threshold if self._speech_samples else self.vad_threshold)

        self._total_samples += sample_count
        self._segment_samples += sample_count
        if is_speech:
            self._speech_samples += sample_count
            self._continuous_no_speech_samples = 0
        else:
            self._continuous_no_speech_samples += sample_count
        if self._speech_samples:
            self._utterance_samples += sample_count
        self._update_vad_threshold(speech_probability)

        should_commit = self._should_commit()
        if should_commit:
            self.mark_boundary()

        if self._total_samples >= self._next_vad_reset_sample:
            self.vad.reset_states()
            self._next_vad_reset_sample += self._vad_reset_interval_samples
            logger.debug('Resetting manual-commit VAD state after %d source samples', self._total_samples)
        return should_commit
