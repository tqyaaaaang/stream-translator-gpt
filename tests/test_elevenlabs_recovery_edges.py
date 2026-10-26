"""Recovery when audio continues beyond a commit whose finals are still pending."""

import base64
import json
import queue
import unittest
from unittest import mock

import numpy as np

from stream_translator_gpt.transcriber.elevenlabs_transcriber import (
    ElevenLabsScribeTranscriber, MIN_COMMIT_SAMPLES,
)
from stream_translator_gpt.utils.common import SAMPLE_RATE, SAMPLES_PER_FRAME


class _Detector:
    def __init__(self):
        self.frames = 0
        self.calls = 0

    def feed(self, audio):
        self.frames += 1
        self.calls += 1
        return self.frames >= 16

    def mark_boundary(self):
        self.frames = 0


class _Socket:
    def __init__(self, *, held_final=None, fail_speculative_send=False):
        self.events = queue.Queue()
        self.events.put({'message_type': 'session_started'})
        self.held_final = held_final
        self.fail_speculative_send = fail_speculative_send
        self.sent = []
        self.commit_count = 0
        self.failed_wait = False

    def send(self, message):
        payload = json.loads(message)
        if self.fail_speculative_send and self.commit_count and not payload['commit']:
            raise ConnectionError('lost while sending following audio')
        self.sent.append(payload)
        if payload['commit']:
            self.commit_count += 1
            if self.held_final is not None:
                if self.held_final != 'none':
                    self.events.put(dict(message_type=self.held_final, text='same', words=None))
            else:
                self.events.put(dict(message_type='committed_transcript', text='same'))
                self.events.put(dict(message_type='committed_transcript_with_timestamps',
                                     text='same', words=None))

    def recv(self, timeout):
        try:
            return json.dumps(self.events.get_nowait())
        except queue.Empty:
            if timeout > 0 and self.commit_count and self.held_final is not None:
                self.failed_wait = True
                raise ConnectionError('lost while waiting for the preceding final')
            raise TimeoutError from None

    def close(self):
        pass

    def pcm(self):
        return b''.join(base64.b64decode(payload['audio_base_64']) for payload in self.sent)


class ElevenLabsRecoveryEdgesTest(unittest.TestCase):
    def _run(self, frame_count, first, second):
        source = queue.Queue()
        data = np.linspace(-0.5, 0.5, frame_count * SAMPLES_PER_FRAME, dtype=np.float32)
        for offset in range(0, len(data), SAMPLES_PER_FRAME):
            source.put(data[offset:offset + SAMPLES_PER_FRAME])
        source.put(None)
        detector = _Detector()
        connector = mock.Mock(side_effect=[first, second])
        transcriber = ElevenLabsScribeTranscriber(
            api_key='test-key', language=None, transcription_filters='', language_filters=None,
            print_result=False, output_timestamps=False, use_history_context=False,
            transcription_keywords=None, boundary_detector=detector, connector=connector,
            reconnect_backoff_seconds=0, final_timeout=1,
        )
        transcriber._pacer = mock.Mock(wait=mock.Mock(return_value=True))
        output = queue.Queue()
        transcriber.loop(source, output)
        results = []
        while not output.empty():
            results.append(output.get_nowait())
        self.assertIsNone(results.pop())
        self.assertEqual(connector.call_count, 2)
        self.assertEqual(detector.calls, frame_count)
        self.assertEqual(transcriber._ack_sample, len(data))
        self.assertFalse(transcriber._journal)
        self.assertEqual([task.transcript for task in results], ['same', 'same'])
        self.assertEqual([task.id for task in results], [1, 2])
        self.assertEqual(results[0].time_range, (0.0, 16 * SAMPLES_PER_FRAME / SAMPLE_RATE))
        self.assertEqual(results[1].time_range,
                         (16 * SAMPLES_PER_FRAME / SAMPLE_RATE, len(data) / SAMPLE_RATE))
        return (data * 32768).astype('<i2').tobytes()

    def test_unsent_next_commit_packet_survives_wait_failure_without_previous_final(self):
        first, second = _Socket(held_final='none'), _Socket()
        expected = self._run(32, first, second)
        self.assertTrue(first.failed_wait)
        self.assertEqual(first.commit_count, 1)
        # The second boundary packet was journalled, but not yet sent, when
        # waiting for the first commit failed. Recovery must include it.
        self.assertEqual(first.pcm(), expected[:28 * SAMPLES_PER_FRAME * 2])
        self.assertEqual(second.pcm(), expected)
        self.assertEqual(second.commit_count, 2)

    def test_unsent_next_commit_survives_while_previous_stable_final_is_not_replayed(self):
        for kind in ('committed_transcript', 'committed_transcript_with_timestamps'):
            with self.subTest(kind=kind):
                first, second = _Socket(held_final=kind), _Socket()
                expected = self._run(32, first, second)
                self.assertTrue(first.failed_wait)
                self.assertEqual(second.pcm(), expected[16 * SAMPLES_PER_FRAME * 2:])
                self.assertEqual(second.commit_count, 1)

    def test_empty_eof_commit_and_short_tail_survive_previous_final_wait_failure(self):
        first, second = _Socket(held_final='committed_transcript'), _Socket()
        expected = self._run(20, first, second)
        self.assertTrue(first.failed_wait)
        tail = expected[16 * SAMPLES_PER_FRAME * 2:]
        self.assertEqual(second.pcm()[:len(tail)], tail)
        self.assertEqual(second.pcm()[len(tail):], bytes(MIN_COMMIT_SAMPLES * 2 - len(tail)))
        self.assertEqual(second.commit_count, 1)

    def test_failed_speculative_audio_send_preserves_pending_final_once_and_replays_new_audio(self):
        first = _Socket(held_final='committed_transcript', fail_speculative_send=True)
        second = _Socket()
        expected = self._run(32, first, second)
        self.assertEqual(first.pcm(), expected[:16 * SAMPLES_PER_FRAME * 2])
        self.assertEqual(second.pcm(), expected[16 * SAMPLES_PER_FRAME * 2:])
        self.assertEqual(second.commit_count, 1)


if __name__ == '__main__':
    unittest.main()
