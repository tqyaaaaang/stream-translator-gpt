import queue
import signal
import unittest
from unittest import mock

from stream_translator_gpt.audio_getter import StreamAudioGetter
from stream_translator_gpt.main import Config, _create_audio_getter


class StreamRealtimeConfigTest(unittest.TestCase):

    def test_modulate_url_enables_realtime_stream_pacing(self):
        sentinel = object()
        config = Config(url='https://example.test/live',
                        use_modulate_transcription_api=True,
                        modulate_api_key='test-key')

        with mock.patch('stream_translator_gpt.main.StreamAudioGetter', return_value=sentinel) as getter_cls:
            result = _create_audio_getter(config)

        self.assertIs(result, sentinel)
        getter_cls.assert_called_once_with(url=config.url,
                                          format=config.format,
                                          cookies=config.cookies,
                                          proxy=config.input_proxy,
                                          realtime=True,
                                          wait_for_video=None)

    def test_non_modulate_url_keeps_legacy_unpaced_stream_behavior(self):
        sentinel = object()
        config = Config(url='https://example.test/live')

        with mock.patch('stream_translator_gpt.main.StreamAudioGetter', return_value=sentinel) as getter_cls:
            result = _create_audio_getter(config)

        self.assertIs(result, sentinel)
        getter_cls.assert_called_once_with(url=config.url,
                                          format=config.format,
                                          cookies=config.cookies,
                                          proxy=config.input_proxy,
                                          realtime=False,
                                          wait_for_video=None)


class StreamRealtimeLoopTest(unittest.TestCase):

    @staticmethod
    def _run_loop(realtime):
        getter = StreamAudioGetter(url='https://example.test/live',
                                   format='bestaudio',
                                   cookies='',
                                   proxy='',
                                   realtime=realtime)
        ffmpeg_process = mock.Mock()
        ytdlp_process = mock.Mock()
        transport_thread = mock.Mock()
        transport_thread.is_alive.return_value = False
        output_queue = queue.SimpleQueue()
        pacer = object()

        patches = (
            mock.patch('stream_translator_gpt.audio_getter._open_stream',
                       return_value=(ffmpeg_process, ytdlp_process, transport_thread)),
            mock.patch('stream_translator_gpt.audio_getter._read_ffmpeg_loop'),
            mock.patch('stream_translator_gpt.audio_getter._wait_for_process', return_value=0),
            mock.patch('stream_translator_gpt.audio_getter._RealtimeAudioPacer', return_value=pacer),
        )
        with patches[0] as open_stream, patches[1] as read_loop, patches[2], patches[3] as pacer_cls:
            getter.loop(output_queue)

        return getter, output_queue, pacer, open_stream, read_loop, pacer_cls

    def test_realtime_stream_paces_decoded_audio_before_enqueueing(self):
        getter, output, pacer, _, read_loop, pacer_cls = self._run_loop(realtime=True)

        pacer_cls.assert_called_once_with(getter.stop_event, source_description='stream playback')
        self.assertIs(read_loop.call_args.kwargs.get('pacer'), pacer)
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())

    def test_non_realtime_stream_does_not_construct_or_use_a_pacer(self):
        _, output, _, _, read_loop, pacer_cls = self._run_loop(realtime=False)

        pacer_cls.assert_not_called()
        self.assertIsNone(read_loop.call_args.kwargs.get('pacer'))
        self.assertFalse(read_loop.call_args.kwargs.get('pad_final_block', False))
        self.assertIsNone(output.get_nowait())
        self.assertTrue(output.empty())

    def test_exit_signal_interrupts_realtime_pacer_wait(self):
        getter = StreamAudioGetter(url='https://example.test/live',
                                   format='bestaudio',
                                   cookies='',
                                   proxy='',
                                   realtime=True)

        with mock.patch.object(getter.stop_event, 'set') as set_stop, \
                mock.patch('stream_translator_gpt.audio_getter.sys.exit') as exit_process:
            getter._exit_handler(signal.SIGINT, None)

        set_stop.assert_called_once_with()
        exit_process.assert_called_once_with(0)


if __name__ == '__main__':
    unittest.main()
