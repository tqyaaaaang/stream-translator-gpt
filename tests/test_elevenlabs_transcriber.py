"""Protocol and lifecycle contracts; fake sockets never contact ElevenLabs."""

import base64
import json
import queue
import threading
import time
import unittest
from unittest import mock

import numpy as np

from stream_translator_gpt.transcriber.elevenlabs_transcriber import (
    ElevenLabsScribeError, ElevenLabsScribeTranscriber, MIN_COMMIT_SAMPLES,
)
from stream_translator_gpt.utils.common import SAMPLE_RATE


class Detector:
    def __init__(self, interval=16):
        self.interval = interval
        self.frames = 0
        self.calls = 0

    def feed(self, audio):
        self.calls += 1
        self.frames += 1
        return self.frames >= self.interval

    def mark_boundary(self):
        self.frames = 0


class Socket:
    def __init__(self, texts=('hello',), *, reverse=False, words=None, respond=True):
        self.events = queue.Queue()
        self.events.put({'message_type': 'session_started'})
        self.sent = []
        self.texts = iter(texts)
        self.reverse = reverse
        self.words = words
        self.respond = respond
        self.closed = threading.Event()
        self.committed = threading.Event()
        self.commit_count = 0

    def send(self, message):
        payload = json.loads(message)
        self.sent.append(payload)
        if payload['commit']:
            self.commit_count += 1
            self.committed.set()
            if self.respond:
                text = next(self.texts, 'hello')
                events = [dict(message_type='committed_transcript', text=text),
                          dict(message_type='committed_transcript_with_timestamps', text=text, words=self.words)]
                for event in reversed(events) if self.reverse else events:
                    self.events.put(event)

    def recv(self, timeout):
        if self.closed.is_set():
            raise ConnectionError('closed')
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if isinstance(event, Exception):
            raise event
        return json.dumps(event)

    def close(self):
        self.closed.set()

    def pcm(self):
        return b''.join(base64.b64decode(payload['audio_base_64']) for payload in self.sent)


def make_transcriber(socket=None, detector=None, **kwargs):
    values = dict(api_key='test-key', language=None, transcription_filters='', language_filters=None,
                  print_result=False, output_timestamps=False, use_history_context=False,
                  transcription_keywords=None, boundary_detector=detector or Detector(),
                  connector=lambda *args, **kwargs: socket, reconnect_backoff_seconds=0,
                  final_timeout=0.5)
    values.update(kwargs)
    transcriber = ElevenLabsScribeTranscriber(**values)
    transcriber._pacer = mock.Mock(wait=mock.Mock(return_value=True))
    return transcriber


def input_audio(frames=16, *, tail=0, eof=True):
    data = (np.arange(frames * 512 + tail, dtype=np.float32) % 1000 - 500) / 1000
    source = queue.Queue()
    for offset in range(0, len(data), 512):
        source.put(data[offset:offset + 512])
    if eof:
        source.put(None)
    return source, data


def tasks(output):
    items = []
    while not output.empty():
        items.append(output.get_nowait())
    return items


class ScribeProtocolTest(unittest.TestCase):
    def test_manual_commits_send_each_source_sample_once_and_no_redundant_eof_commit(self):
        ws = Socket(texts=('one', 'two'))
        detector = Detector()
        transcriber = make_transcriber(ws, detector)
        source, data = input_audio(32)
        output = queue.Queue()

        transcriber.loop(source, output)

        expected = (data * 32768).astype('<i2').tobytes()
        self.assertEqual(ws.pcm(), expected)
        self.assertEqual(ws.commit_count, 2)
        self.assertEqual(detector.calls, 32)
        results = tasks(output)
        self.assertEqual([item.transcript for item in results[:-1]], ['one', 'two'])
        self.assertEqual([item.id for item in results[:-1]], [1, 2])
        self.assertEqual(results[0].time_range, (0, 0.512))
        self.assertEqual(results[1].time_range, (0.512, 1.024))
        self.assertIsNone(results[-1])
        self.assertTrue(ws.closed.is_set())

    def test_reverse_event_order_and_repeated_sentences_are_not_deduplicated(self):
        ws = Socket(texts=('same', 'same'), reverse=True)
        transcriber = make_transcriber(ws)
        source, _ = input_audio(32)
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual([item.transcript for item in tasks(output)[:-1]], ['same', 'same'])

    def test_empty_finals_acknowledge_audio_without_translation(self):
        ws = Socket(texts=('', 'text'), reverse=True)
        transcriber = make_transcriber(ws)
        source, _ = input_audio(32)
        output = queue.Queue()
        transcriber.loop(source, output)
        result = tasks(output)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].transcript, 'text')
        self.assertEqual(transcriber._ack_sample, 32 * 512)
        self.assertFalse(transcriber._journal)

    def test_word_timestamps_are_mapped_to_media_and_clamped(self):
        ws = Socket(words=[dict(start=0.1, end=0.4), dict(start=float('nan'), end=1)])
        transcriber = make_transcriber(ws)
        source, _ = input_audio()
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual(tasks(output)[0].time_range, (0.1, 0.4))

    def test_short_eof_pads_only_to_minimum_and_preserves_source_duration(self):
        ws = Socket()
        transcriber = make_transcriber(ws)
        source, data = input_audio(0, tail=19)
        output = queue.Queue()
        transcriber.loop(source, output)
        pcm = ws.pcm()
        self.assertEqual(len(pcm), MIN_COMMIT_SAMPLES * 2)
        self.assertEqual(pcm[:len(data) * 2], (data * 32768).astype('<i2').tobytes())
        self.assertEqual(set(pcm[len(data) * 2:]), {0})
        self.assertEqual(tasks(output)[0].time_range[1], len(data) / SAMPLE_RATE)
        self.assertEqual(ws.commit_count, 1)

    def test_eof_flushes_already_sent_audio_with_one_empty_commit(self):
        ws = Socket()
        transcriber = make_transcriber(ws, Detector(1000))
        source, data = input_audio(20)
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual(ws.pcm(), (data * 32768).astype('<i2').tobytes())
        self.assertEqual(ws.sent[-1]['audio_base_64'], '')
        self.assertTrue(ws.sent[-1]['commit'])
        self.assertEqual(ws.commit_count, 1)

    def test_short_local_boundaries_are_coalesced_before_service_minimum(self):
        ws = Socket()
        transcriber = make_transcriber(ws, Detector(1))
        source, _ = input_audio(10)
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual(ws.commit_count, 1)
        self.assertEqual(len(ws.pcm()), MIN_COMMIT_SAMPLES * 2)

    def test_empty_input_never_opens_connection(self):
        connector = mock.Mock()
        transcriber = make_transcriber(connector=connector)
        source, _ = input_audio(0)
        output = queue.Queue()
        transcriber.loop(source, output)
        connector.assert_not_called()
        self.assertEqual(tasks(output), [None])

    def test_filtered_text_does_not_enter_translation_queue(self):
        ws = Socket(texts=('😀',))
        transcriber = make_transcriber(ws, transcription_filters='emoji_filter')
        source, _ = input_audio()
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual(tasks(output), [None])
        self.assertGreater(transcriber._ack_sample, 0)

    def test_mismatched_final_pair_is_error_not_published_fallback(self):
        class Disagreeing(Socket):
            def send(self, message):
                super().send(message)
                if json.loads(message)['commit']:
                    self.events.put(dict(message_type='committed_transcript', text='a'))
                    self.events.put(dict(message_type='committed_transcript_with_timestamps', text='b', words=None))
        ws = Disagreeing(respond=False)
        transcriber = make_transcriber(ws)
        source, _ = input_audio()
        output = queue.Queue()
        with self.assertRaisesRegex(ElevenLabsScribeError, 'disagree'):
            transcriber.loop(source, output)
        self.assertTrue(output.empty())  # worker wrapper supplies error EOF
        self.assertEqual(transcriber._ack_sample, 0)

    def test_late_partial_never_becomes_a_new_task(self):
        class LatePartial(Socket):
            def send(self, message):
                super().send(message)
                if json.loads(message)['commit']:
                    self.events.put(dict(message_type='partial_transcript', text='obsolete'))
        ws = LatePartial(texts=('one', 'two'))
        transcriber = make_transcriber(ws)
        source, _ = input_audio(32)
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual([item.transcript for item in tasks(output)[:-1]], ['one', 'two'])

    def test_audio_continues_while_previous_commit_is_pending(self):
        class DelayedFinal(Socket):
            def __init__(self):
                super().__init__(respond=False)
                self.sent_while_pending = False
                self.pending_text = None

            def send(self, message):
                payload = json.loads(message)
                super().send(message)
                if payload['commit']:
                    self.pending_text = 'first' if self.commit_count == 1 else 'second'
                    if self.commit_count == 2:
                        self.finish()
                elif self.pending_text is not None:
                    self.sent_while_pending = True
                    self.finish()

            def finish(self):
                self.events.put(dict(message_type='committed_transcript', text=self.pending_text))
                self.events.put(dict(message_type='committed_transcript_with_timestamps',
                                     text=self.pending_text, words=None))
                self.pending_text = None

        ws = DelayedFinal()
        transcriber = make_transcriber(ws)
        source, _ = input_audio(32)
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertTrue(ws.sent_while_pending)
        self.assertEqual([item.transcript for item in tasks(output)[:-1]], ['first', 'second'])


class ScribeRecoveryTest(unittest.TestCase):
    def test_idle_flush_resumes_without_repeating_audio_or_detector_state(self):
        first, second = Socket(texts=('one',)), Socket(texts=('two',))
        detector = Detector(1000)
        transcriber = make_transcriber(detector=detector, connector=mock.Mock(side_effect=[first, second]),
                                       idle_timeout=0.06)
        source, data = input_audio(16, eof=False)
        output = queue.Queue()
        errors = []
        def run():
            try:
                transcriber.loop(source, output)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.addCleanup(transcriber.stop)
        self.assertTrue(first.closed.wait(2))
        self.assertEqual(detector.frames, 0)
        for offset in range(0, len(data), 512):
            source.put(data[offset:offset + 512])
        source.put(None)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        expected = (data * 32768).astype('<i2').tobytes()
        self.assertEqual(first.pcm(), expected)
        self.assertEqual(second.pcm(), expected)
        self.assertEqual([task.transcript for task in tasks(output)[:-1]], ['one', 'two'])

    def test_send_failure_replays_unacknowledged_audio_without_rerunning_vad(self):
        class FailSecondCommit(Socket):
            def send(self, message):
                if json.loads(message)['commit'] and self.commit_count == 1:
                    raise ConnectionError('dropped')
                super().send(message)
        first = FailSecondCommit(texts=('one',))
        second = Socket(texts=('two',))
        connector = mock.Mock(side_effect=[first, second])
        detector = Detector()
        transcriber = make_transcriber(detector=detector, connector=connector)
        source, data = input_audio(32)
        output = queue.Queue()

        transcriber.loop(source, output)

        self.assertEqual(second.pcm(), (data[16 * 512:] * 32768).astype('<i2').tobytes())
        result = tasks(output)
        self.assertEqual([task.transcript for task in result[:-1]], ['one', 'two'])
        self.assertEqual(result[1].time_range, (0.512, 1.024))
        self.assertEqual(detector.calls, 32)

    def test_final_received_before_disconnect_is_not_replayed(self):
        class DropCompanion(Socket):
            def send(self, message):
                super().send(message)
                if json.loads(message)['commit']:
                    self.events.put(dict(message_type='committed_transcript', text='one'))
                    self.events.put(ConnectionError('no timestamps'))
        first = DropCompanion(respond=False)
        second = Socket(texts=('two',))
        transcriber = make_transcriber(connector=mock.Mock(side_effect=[first, second]))
        source, data = input_audio(32)
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual(second.pcm(), (data[16 * 512:] * 32768).astype('<i2').tobytes())
        self.assertEqual([task.transcript for task in tasks(output)[:-1]], ['one', 'two'])

    def test_timestamps_received_before_disconnect_are_stable(self):
        class DropText(Socket):
            def send(self, message):
                super().send(message)
                if json.loads(message)['commit']:
                    self.events.put(dict(message_type='committed_transcript_with_timestamps', text='one', words=None))
                    self.events.put(ConnectionError('no plain event'))
        first = DropText(respond=False)
        transcriber = make_transcriber(first)
        source, _ = input_audio()
        output = queue.Queue()
        transcriber.loop(source, output)
        self.assertEqual([task.transcript for task in tasks(output)[:-1]], ['one'])

    def test_auth_error_never_retries_and_redacts_key(self):
        ws = Socket()
        ws.events = queue.Queue()
        ws.events.put(dict(message_type='auth_error', error='invalid test-key'))
        connector = mock.Mock(return_value=ws)
        transcriber = make_transcriber(connector=connector)
        source, _ = input_audio()
        with self.assertRaises(ElevenLabsScribeError) as caught:
            transcriber.loop(source, queue.Queue())
        self.assertNotIn('test-key', str(caught.exception))
        self.assertIn('<redacted>', str(caught.exception))
        self.assertEqual(connector.call_count, 1)

    def test_repeated_partial_is_not_retry_progress(self):
        class Failed(Socket):
            def send(self, message):
                super().send(message)
                if json.loads(message)['commit']:
                    self.events.put(dict(message_type='partial_transcript', text='same partial'))
                    self.events.put(ConnectionError('dropped'))
        connector = mock.Mock(side_effect=lambda *a, **k: Failed(respond=False))
        transcriber = make_transcriber(connector=connector, reconnect_attempts=2)
        source, _ = input_audio()
        with self.assertRaisesRegex(ElevenLabsScribeError, 'exhausted'):
            transcriber.loop(source, queue.Queue())
        self.assertEqual(connector.call_count, 3)


class ScribeCancellationTest(unittest.TestCase):
    def start(self, transcriber, source):
        output = queue.Queue()
        errors = []
        def run():
            try:
                transcriber.loop(source, output)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.addCleanup(transcriber.stop)
        return worker, output, errors

    def finish(self, transcriber, worker, output, errors):
        transcriber.stop()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(tasks(output), [None])

    def test_cancel_waiting_for_final(self):
        ws = Socket(respond=False)
        transcriber = make_transcriber(ws, final_timeout=30)
        source, _ = input_audio()
        worker, output, errors = self.start(transcriber, source)
        self.assertTrue(ws.committed.wait(1))
        self.finish(transcriber, worker, output, errors)

    def test_cancel_waiting_for_input(self):
        transcriber = make_transcriber()
        worker, output, errors = self.start(transcriber, queue.Queue())
        self.finish(transcriber, worker, output, errors)

    def test_cancel_handshake_closes_late_connection(self):
        opening, release = threading.Event(), threading.Event()
        ws = Socket()
        def connect(*args, **kwargs):
            opening.set()
            release.wait(2)
            return ws
        transcriber = make_transcriber(connector=connect)
        source, _ = input_audio()
        worker, output, errors = self.start(transcriber, source)
        self.assertTrue(opening.wait(1))
        self.finish(transcriber, worker, output, errors)
        release.set()
        self.assertTrue(ws.closed.wait(1))
        self.assertFalse(ws.sent)

    def test_cancel_interrupts_close_after_eof(self):
        closing = threading.Event()
        class BlockingClose(Socket):
            def __init__(self):
                super().__init__(texts=('',))
                self.socket = mock.Mock()
                self.socket.shutdown.side_effect = lambda *a: self.closed.set()
                self.socket.close.side_effect = self.closed.set
            def close(self):
                closing.set()
                self.closed.wait(2)
        ws = BlockingClose()
        transcriber = make_transcriber(ws)
        source, _ = input_audio()
        worker, output, errors = self.start(transcriber, source)
        self.assertTrue(closing.wait(1))
        self.finish(transcriber, worker, output, errors)

    def test_cancel_during_send(self):
        sending = threading.Event()
        class BlockingSend(Socket):
            def send(self, message):
                sending.set()
                self.closed.wait(2)
                raise ConnectionError('cancelled')
        ws = BlockingSend()
        transcriber = make_transcriber(ws)
        source, _ = input_audio()
        worker, output, errors = self.start(transcriber, source)
        self.assertTrue(sending.wait(1))
        self.finish(transcriber, worker, output, errors)

    def test_cancel_during_backoff(self):
        attempted = threading.Event()
        def connect(*args, **kwargs):
            attempted.set()
            raise ConnectionError('offline')
        transcriber = make_transcriber(connector=connect, reconnect_backoff_seconds=30)
        source, _ = input_audio()
        worker, output, errors = self.start(transcriber, source)
        self.assertTrue(attempted.wait(1))
        self.finish(transcriber, worker, output, errors)


if __name__ == '__main__':
    unittest.main()
