import argparse
import os
import queue
import sys
import threading
import time
import logging
from logging.handlers import QueueHandler, QueueListener

import google.generativeai as genai
from google.api_core.client_options import ClientOptions

from .audio_getter import StreamAudioGetter, LocalFileAudioGetter, DeviceAudioGetter
from .audio_slicer import AudioSlicer
from .audio_transcriber import OpenaiWhisper, FasterWhisper, RemoteOpenaiWhisper
from .llm_translator import LLMClint, ParallelTranslator, SerialTranslator
from .result_exporter import ResultExporter


logger = logging.getLogger('main')


def _start_daemon_thread(func, name, *args, **kwargs):
    thread = threading.Thread(target=func, name=name, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()


def main(url, format, cookies, input_proxy, is_file, device_index, device_recording_interval, frame_duration,
         continuous_no_speech_threshold, min_audio_length, max_audio_length, prefix_retention_length, vad_threshold,
         model, language, use_faster_whisper, use_whisper_api, whisper_base_url, whisper_api_key, whisper_filters, openai_api_key, google_api_key,
         gpt_translation_prompt, gpt_translation_history_size, gpt_model, gemini_model, gpt_translation_timeout,
         gpt_base_url, gemini_base_url, processing_proxy, use_json_result, retry_if_translation_fails,
         output_timestamps, hide_transcribe_result, output_proxy, output_file_path, cqhttp_url, cqhttp_token,
         discord_webhook_url, telegram_token, telegram_chat_id, **transcribe_options):
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
    if gpt_base_url:
        os.environ['OPENAI_BASE_URL'] = gpt_base_url
    if google_api_key:
        gemini_client_options = ClientOptions(api_endpoint=gemini_base_url)
        genai.configure(api_key=google_api_key, client_options=gemini_client_options, transport='rest')

    getter_to_slicer_queue = queue.SimpleQueue()
    slicer_to_transcriber_queue = queue.SimpleQueue()
    transcriber_to_translator_queue = queue.SimpleQueue()
    translator_to_exporter_queue = queue.SimpleQueue() if gpt_translation_prompt else transcriber_to_translator_queue

    _start_daemon_thread(
        ResultExporter.work,
        'exporter',
        output_whisper_result=not hide_transcribe_result,
        output_timestamps=output_timestamps,
        proxy=output_proxy,
        output_file_path=output_file_path,
        cqhttp_url=cqhttp_url,
        cqhttp_token=cqhttp_token,
        discord_webhook_url=discord_webhook_url,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        input_queue=translator_to_exporter_queue,
    )
    if gpt_translation_prompt:
        if google_api_key:
            llm_client = LLMClint(
                llm_type=LLMClint.LLM_TYPE.GEMINI,
                model=gemini_model,
                prompt=gpt_translation_prompt,
                history_size=gpt_translation_history_size,
                proxy=processing_proxy,
                use_json_result=use_json_result,
            )
        else:
            llm_client = LLMClint(
                llm_type=LLMClint.LLM_TYPE.GPT,
                model=gpt_model,
                prompt=gpt_translation_prompt,
                history_size=gpt_translation_history_size,
                proxy=processing_proxy,
                use_json_result=use_json_result,
            )
        if gpt_translation_history_size == 0:
            _start_daemon_thread(
                ParallelTranslator.work,
                'translator',
                llm_client=llm_client,
                timeout=gpt_translation_timeout,
                retry_if_translation_fails=retry_if_translation_fails,
                input_queue=transcriber_to_translator_queue,
                output_queue=translator_to_exporter_queue,
            )
        else:
            _start_daemon_thread(
                SerialTranslator.work,
                'translator',
                llm_client=llm_client,
                timeout=gpt_translation_timeout,
                retry_if_translation_fails=retry_if_translation_fails,
                input_queue=transcriber_to_translator_queue,
                output_queue=translator_to_exporter_queue,
            )
    if use_faster_whisper:
        _start_daemon_thread(FasterWhisper.work,
                             'transcriber',
                             model=model,
                             language=language,
                             print_result=not hide_transcribe_result,
                             output_timestamps=output_timestamps,
                             input_queue=slicer_to_transcriber_queue,
                             output_queue=transcriber_to_translator_queue,
                             whisper_filters=whisper_filters,
                             **transcribe_options)
    elif use_whisper_api:
        _start_daemon_thread(RemoteOpenaiWhisper.work,
                             'transcriber',
                             model=model,
                             base_url=whisper_base_url,
                             api_key=whisper_api_key,
                             language=language,
                             proxy=processing_proxy,
                             print_result=not hide_transcribe_result,
                             output_timestamps=output_timestamps,
                             input_queue=slicer_to_transcriber_queue,
                             output_queue=transcriber_to_translator_queue,
                             whisper_filters=whisper_filters,
                             **transcribe_options)
    else:
        _start_daemon_thread(OpenaiWhisper.work,
                             'transcriber',
                             model=model,
                             language=language,
                             print_result=not hide_transcribe_result,
                             output_timestamps=output_timestamps,
                             input_queue=slicer_to_transcriber_queue,
                             output_queue=transcriber_to_translator_queue,
                             whisper_filters=whisper_filters,
                             **transcribe_options)
    _start_daemon_thread(
        AudioSlicer.work,
        'slicer',
        frame_duration=frame_duration,
        continuous_no_speech_threshold=continuous_no_speech_threshold,
        min_audio_length=min_audio_length,
        max_audio_length=max_audio_length,
        prefix_retention_length=prefix_retention_length,
        vad_threshold=vad_threshold,
        input_queue=getter_to_slicer_queue,
        output_queue=slicer_to_transcriber_queue,
    )
    if url.lower() == 'device':
        DeviceAudioGetter.work(
            device_index=device_index,
            frame_duration=frame_duration,
            recording_interval=device_recording_interval,
            output_queue=getter_to_slicer_queue,
        )
    elif is_file or os.path.isabs(url):
        LocalFileAudioGetter.work(
            file_path=url,
            frame_duration=frame_duration,
            output_queue=getter_to_slicer_queue,
        )
    else:
        StreamAudioGetter.work(
            url=url,
            format=format,
            cookies=cookies,
            proxy=input_proxy,
            frame_duration=frame_duration,
            output_queue=getter_to_slicer_queue,
        )

    # Wait for others process finish.
    while (not getter_to_slicer_queue.empty() or not slicer_to_transcriber_queue.empty() or
           not transcriber_to_translator_queue.empty() or not translator_to_exporter_queue.empty()):
        time.sleep(5)
    logger.error('Stream ended')


def cli():
    parser = argparse.ArgumentParser(description='Parameters for translator.py')
    parser.add_argument('URL',
                        type=str,
                        help='The URL of the stream. '
                        'If a local file path is filled in, it will be used as input. '
                        'If fill in "device", the input will be obtained from your PC device.')
    parser.add_argument('--format',
                        type=str,
                        default='wa*',
                        help='Stream format code, '
                        'this parameter will be passed directly to yt-dlp.')
    parser.add_argument('--cookies',
                        type=str,
                        default=None,
                        help='Used to open member-only stream, '
                        'this parameter will be passed directly to yt-dlp.')
    parser.add_argument('--input_proxy',
                        type=str,
                        default=None,
                        help='Use the specified HTTP/HTTPS/SOCKS proxy for yt-dlp, '
                        'e.g. http://127.0.0.1:7890.')
    parser.add_argument('--is_file',
                        dest='is_file',
                        action='store_true',
                        help='Parse the input URL as a local file path')
    parser.add_argument('--device_index',
                        type=int,
                        default=None,
                        help='The index of the device that needs to be recorded. '
                        'If not set, the system default recording device will be used.')
    parser.add_argument('--print_all_devices', action='store_true', help='Print all audio devices info then exit.')
    parser.add_argument('--device_recording_interval',
                        type=float,
                        default=0.5,
                        help='The shorter the recording interval, the lower the latency,'
                        'but it will increase CPU usage.'
                        'It is recommended to set it between 0.1 and 1.0.')
    parser.add_argument('--frame_duration',
                        type=float,
                        default=0.1,
                        help='The unit that processes live streaming data in seconds, '
                        'should be >= 0.03')
    parser.add_argument('--continuous_no_speech_threshold',
                        type=float,
                        default=0.5,
                        help='Slice if there is no speech for a continuous period in second.')
    parser.add_argument('--min_audio_length', type=float, default=1.5, help='Minimum slice audio length in seconds.')
    parser.add_argument('--max_audio_length', type=float, default=15.0, help='Maximum slice audio length in seconds.')
    parser.add_argument('--prefix_retention_length',
                        type=float,
                        default=0.5,
                        help='The length of the retention prefix audio during slicing.')
    parser.add_argument('--vad_threshold',
                        type=float,
                        default=0.35,
                        help='The threshold of Voice activity detection.'
                        'if the speech probability of a frame is higher than this value, '
                        'then this frame is speech.')
    parser.add_argument('--model',
                        type=str,
                        default='small',
                        help='Select Whisper/Faster-Whisper model size. '
                        'See https://github.com/openai/whisper#available-models-and-languages for available models.')
    parser.add_argument('--language',
                        type=str,
                        default='auto',
                        help='Language spoken in the stream. '
                        'Default option is to auto detect the spoken language. '
                        'See https://github.com/openai/whisper#available-models-and-languages for available languages.')
    parser.add_argument('--beam_size',
                        type=int,
                        default=5,
                        help='Number of beams in beam search. '
                        'Set to 0 to use greedy algorithm instead.')
    parser.add_argument('--best_of',
                        type=int,
                        default=5,
                        help='Number of candidates when sampling with non-zero temperature.')
    parser.add_argument('--use_faster_whisper',
                        action='store_true',
                        help='Set this flag to use faster-whisper implementation instead of '
                        'the original OpenAI implementation.')
    parser.add_argument('--use_whisper_api',
                        action='store_true',
                        help='Set this flag to use OpenAI Whisper API instead of '
                        'the original local Whipser.')
    parser.add_argument('--whisper_base_url', type=str, default=None, help='Customize the API endpoint of Whisper.')
    parser.add_argument('--whisper_api_key',
                        type=str,
                        default=None,
                        help='API key for Whisper if uses customized API endpoint.')
    parser.add_argument('--whisper_filters',
                        type=str,
                        default='emoji_filter',
                        help='Filters apply to whisper results, separated by ",". '
                        'We provide emoji_filter and japanese_stream_filter.')
    parser.add_argument('--openai_api_key',
                        type=str,
                        default=None,
                        help='OpenAI API key if using GPT translation / Whisper API.')
    parser.add_argument('--google_api_key', type=str, default=None, help='Google API key if using Gemini translation.')
    parser.add_argument('--gpt_model',
                        type=str,
                        default='gpt-4o-mini',
                        help='OpenAI\'s GPT model name, gpt-3.5-turbo / gpt-4o / gpt-4o-mini.')
    parser.add_argument('--gemini_model',
                        type=str,
                        default='gemini-1.5-flash',
                        help='Google\'s Gemini model name, gemini-1.5-flash / gemini-1.5-pro')
    parser.add_argument('--gpt_translation_prompt',
                        type=str,
                        default=None,
                        help='If set, will translate result text to target language via GPT / Gemini API. '
                        'Example: \"Translate from Japanese to Chinese\"')
    parser.add_argument('--gpt_translation_history_size',
                        type=int,
                        default=0,
                        help='The number of previous messages sent when calling the GPT / Gemini API. '
                        'If the history size is 0, the translation will be run parallelly. '
                        'If the history size > 0, the translation will be run serially.')
    parser.add_argument('--gpt_translation_timeout',
                        type=int,
                        default=10,
                        help='If the GPT / Gemini translation exceeds this number of seconds, '
                        'the translation will be discarded.')
    parser.add_argument('--gpt_base_url', type=str, default=None, help='Customize the API endpoint of GPT.')
    parser.add_argument('--gemini_base_url', type=str, default=None, help='Customize the API endpoint of Gemini.')
    parser.add_argument('--processing_proxy',
                        type=str,
                        default=None,
                        help='Use the specified HTTP/HTTPS/SOCKS proxy for Whisper/GPT API '
                        '(Gemini currently doesn\'t support specifying a proxy within the program), '
                        'e.g. http://127.0.0.1:7890.')
    parser.add_argument('--use_json_result',
                        action='store_true',
                        help='Using JSON result in LLM translation for some locally deployed models.')
    parser.add_argument('--retry_if_translation_fails',
                        action='store_true',
                        help='Retry when translation times out/fails. Used to generate subtitles offline.')
    parser.add_argument('--output_timestamps',
                        action='store_true',
                        help='Output the timestamp of the text when outputting the text.')
    parser.add_argument('--hide_transcribe_result', action='store_true', help='Hide the result of Whisper transcribe.')
    parser.add_argument('--output_proxy',
                        type=str,
                        default=None,
                        help='Use the specified HTTP/HTTPS/SOCKS proxy for Cqhttp/Discord/Telegram, '
                        'e.g. http://127.0.0.1:7890.')
    parser.add_argument('--output_file_path',
                        type=str,
                        default=None,
                        help='If set, will save the result text to this path.')
    parser.add_argument('--cqhttp_url',
                        type=str,
                        default=None,
                        help='If set, will send the result text to this Cqhttp server.')
    parser.add_argument('--cqhttp_token',
                        type=str,
                        default=None,
                        help='Token of cqhttp, if it is not set on the server side, '
                        'it does not need to fill in.')
    parser.add_argument('--discord_webhook_url',
                        type=str,
                        default=None,
                        help='If set, will send the result text to this Discord channel.')
    parser.add_argument('--telegram_token', type=str, default=None, help='Token of Telegram bot.')
    parser.add_argument('--telegram_chat_id',
                        type=int,
                        default=None,
                        help='If set, will send the result text to this Telegram chat. '
                        'Needs to be used with \"--telegram_token\".')
    parser.add_argument('--verbose',
                        dest='verbose',
                        type=int,
                        nargs='?',
                        default=0,
                        const=1,
                        help='Verbose mode. Use --verbose 2 to output more detailed debugging messages.')

    args = parser.parse_args().__dict__
    url = args.pop('URL')

    if args['print_all_devices']:
        import sounddevice as sd
        print(sd.query_devices())
        exit(0)

    if args['model'].endswith('.en'):
        if args['model'] == 'large.en':
            print('English model does not have large model, please choose from {tiny.en, small.en, medium.en}')
            sys.exit(0)
        if args['language'] != 'English' and args['language'] != 'en':
            if args['language'] == 'auto':
                print('Using .en model, setting language from auto to English')
                args['language'] = 'en'
            else:
                print('English model cannot be used to detect non english language, please choose a non .en model')
                sys.exit(0)

    if args['use_faster_whisper'] and args['use_whisper_api']:
        print('Cannot use Faster Whisper and Whisper API at the same time')
        sys.exit(0)

    if args['use_whisper_api'] and not args['openai_api_key']:
        print('Please fill in the OpenAI API key when enabling Whisper API')
        sys.exit(0)

    if args['gpt_translation_prompt'] and not (args['openai_api_key'] or args['google_api_key']):
        print('Please fill in the OpenAI / Google API key when enabling LLM translation')
        sys.exit(0)

    if args['language'] == 'auto':
        args['language'] = None

    if args['beam_size'] == 0:
        args['beam_size'] = None

    if args['verbose']:
        logger.setLevel(logging.INFO)

        log_queue = queue.Queue()
        handler = QueueHandler(log_queue)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='{asctime}.{msecs:03.0f}:{levelname}:{threadName}: {message}', datefmt='%H:%M:%S', style='{')
        handler.setFormatter(formatter)

        if args['verbose'] >= 2:
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)

        logger.addHandler(handler)

        write_handler = logging.StreamHandler()
        listener = QueueListener(log_queue, write_handler)
        listener.start()
        
        main_thread = threading.current_thread()
        main_thread.name = 'main'
    else:
        logger.setLevel(logging.WARNING)

        log_queue = queue.Queue()
        handler = QueueHandler(log_queue)
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter(fmt='%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        write_handler = logging.StreamHandler()
        listener = QueueListener(log_queue, write_handler)
        listener.start()

    main(url, **args)
