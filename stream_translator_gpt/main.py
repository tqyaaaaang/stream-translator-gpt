import dataclasses
import logging
import math
import os
import platform
import queue
import shutil
import signal
import sys
import time
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal

import tyro

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "stream_translator_gpt"

from .utils.common import ClientPool, SAMPLE_RATE, SAMPLES_PER_FRAME, start_daemon_thread, is_url
from .getter.audio_getter import DeviceAudioGetter, LocalFileAudioGetter, SoundDeviceAudioGetter, StreamAudioGetter
from .getter.time_range import normalize_time_range, normalize_time_range_strict
from .transcriber.audio_transcriber import OpenaiWhisper, FasterWhisper, SimulStreaming, RemoteOpenaiTranscriber, HFTranscriber
from .transcriber.modulate_transcriber import (MODULATE_INPUT_QUEUE_SECONDS, MODULATE_STREAM_MAX_SEND_RATE,
                                   ModulateStreamingTranscriber)
from .transcriber.elevenlabs_transcriber import ElevenLabsScribeTranscriber
from .exporter.result_exporter import ResultExporter
from .utils.logging_utils import configure_logging, normalize_verbosity_args, shutdown_logging, verbosity_from_argv
from . import __version__


logger = logging.getLogger('main')

DEFAULT_WAIT_FOR_VIDEO_SECONDS = 60
ELEVENLABS_INPUT_QUEUE_SECONDS = 600

_LEGACY_LANGUAGE_FILTER_ALIASES = {
    'en': 'en',
    'english': 'en',
    'ja': 'ja',
    'japanese': 'ja',
    'zh': 'zh',
    'chinese': 'zh',
    'zh-cn': 'zh',
    'zh-tw': 'zh',
}


@dataclasses.dataclass
class Config:
    url: Annotated[tyro.conf.Positional[str], tyro.conf.arg(metavar='URL')]
    """The URL of the stream. If a local file path is filled in, it will be used as input. If fill in "device", the
    input will be obtained from your PC device."""

    openai_api_key: str | None = None
    """OpenAI API key for GPT translation. It is also used as the fallback key for the Whisper API when
    --whisper-api-key is not set. Separate multiple keys with "," to rotate between them."""

    whisper_api_key: str | None = None
    """API key for a Whisper-compatible transcription endpoint. Falls back to --openai-api-key when unset. Separate
    multiple keys with "," to rotate between them."""

    modulate_api_key: str | None = None
    """API key for Modulate multilingual streaming transcription."""

    elevenlabs_api_key: str | None = None
    """API key for ElevenLabs Scribe v2 realtime transcription."""

    google_api_key: str | None = None
    """Google API key if using Gemini translation. If you have multiple keys, you can separate them with "," and each
    key will be used in turn."""

    openai_base_url: str | None = None
    """Customize the OpenAI endpoint used for GPT translation. It is also the fallback endpoint for transcription
    when --whisper-base-url is not set."""

    whisper_base_url: str | None = None
    """Customize the endpoint used by the Whisper-compatible transcription API."""

    google_base_url: str | None = None
    """Customize the API endpoint of Google (Affects Gemini translation)."""

    verify_ssl: bool = True
    """TLS certificate verification for OpenAI / Google API and HuggingFace downloads. Use --no-verify-ssl when your
    API endpoint or proxy has a self-signed or invalid certificate. If the base URL host is a bare IP, verification is
    disabled automatically."""

    proxy: str | None = None
    """Used to set the proxy for all --*-proxy flags if they are not specifically set. Also sets http_proxy environment
    variables."""

    format: str = 'ba/wa*'
    """Stream format code, this parameter will be passed directly to yt-dlp. You can get the list of available format
    codes by "yt-dlp {url} -F"."""

    list_format: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Print all available formats then exit."""

    stream_format_diagnostic_file: str | None = None
    """Write allowlisted yt-dlp format metadata, replacing this file for each stream. Disabled by default."""

    cookies: str | None = None
    """Used to open member-only stream, this parameter will be passed directly to yt-dlp."""

    input_proxy: str | None = None
    """Use the specified HTTP/HTTPS/SOCKS proxy for yt-dlp, e.g. http://127.0.0.1:7890."""

    wait_for_video: Annotated[int | None, tyro.conf.arg(metavar='[SECONDS]')] = None
    """Wait for a scheduled stream to become available. The value is yt-dlp's minimum number of seconds between
    availability checks. Omit the option to disable waiting; using it without a value selects 60 seconds."""

    live_from_start: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Ask yt-dlp to download a supported livestream from its beginning instead of the current live edge."""

    ss: Annotated[str | None, tyro.conf.arg(metavar='TIME')] = None
    """Start at this source position for local files and URL streams (FFmpeg-style non-negative seconds or
    [HH:]MM:SS[.fraction]). Seek/download only the selected range where supported. Ignored with a warning for device
    input or --live-from-start."""

    to: Annotated[str | None, tyro.conf.arg(metavar='TIME')] = None
    """Stop at this absolute source position, not a duration; accepts the same time formats as --ss. Must be later
    than --ss (or zero if --ss is omitted). Ignored with a warning for device input or --live-from-start."""

    time_range_strict: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Use conservative precise trimming for HLS/TS instead of the default fast seek, which may be off by a segment.
    Strict mode reads preceding audio and may start more slowly. Requires --ss or --to; otherwise warns and is ignored.
    Also ignored with a warning for device input or --live-from-start."""

    pacer: Literal['unlimited', 'realtime'] = 'unlimited'
    """Control decoded audio pacing for URL streams and local files. Unlimited mode reads as fast as downstream
    capacity permits; realtime emits one second of decoded audio per second of wall-clock time. Device input ignores this
    option and logs a warning because capture is already paced by the audio device. ElevenLabs Scribe additionally caps
    API sending (including replay) at 1x in realtime mode and 5x otherwise."""

    device_index: int | None = None
    """The index of the device that needs to be recorded. If not set, the system default recording device will be
    used."""

    device_recording_interval: float = 0.5
    """The shorter the recording interval, the lower the latency, but it will increase CPU usage. It is recommended to
    set it between 0.1 and 1.0."""

    device_audio_getter: Literal['pyaudio', 'sounddevice'] = 'pyaudio'
    """Select the device capture implementation. PyAudio remains the default; use sounddevice as a compatibility
    fallback for input-capable devices that PyAudio cannot capture."""

    device_multiplier: float = 1.0
    """Legacy perceptual volume factor for device capture. 1 keeps the original amplitude. Supported by both the
    PyAudio and sounddevice getters."""

    device_audio_debug_dir: str | None = None
    """Save processed device input as bounded PCM WAV clips in this directory. Disabled by default."""

    device_audio_debug_clip_duration: float = 10.0
    """Length in seconds of each device debug WAV clip."""

    device_audio_debug_max_clips: int = 10
    """Maximum number of device debug WAV clips written during one run."""

    list_devices: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Print all audio devices info then exit."""

    mic: bool = False
    """Use microphone instead of system audio (loopback)."""

    min_audio_length: float = 0.5
    """Minimum slice audio length in seconds."""

    max_audio_length: float = 30.0
    """Maximum slice audio length in seconds. ElevenLabs Scribe warns and caps values above 30 seconds."""

    target_audio_length: float = 5.0
    """When dynamic no speech threshold is enabled (enabled by default), the program will slice the audio as close to
    this length as possible."""

    continuous_no_speech_threshold: float = 1.0
    """Slice if there is no speech during this number of seconds. If the dynamic no speech threshold is enabled
    (enabled by default), the actual threshold will be dynamically adjusted based on this value."""

    dynamic_no_speech_threshold: bool = True
    """Dynamically adjust the no speech threshold based on --continuous-no-speech-threshold. Disable with
    --no-dynamic-no-speech-threshold."""

    prefix_retention_length: float = 0.5
    """The length of the retention prefix audio during slicing. Not used by ElevenLabs' continuous audio stream."""

    vad_threshold: float = 0.35
    """Range 0~1. The higher this value, the stricter the speech judgment. If dynamic VAD threshold is enabled (enabled
    by default), this threshold will be adjusted dynamically based on this value."""

    dynamic_vad_threshold: bool = True
    """Dynamically adjust the VAD threshold based on --vad-threshold. Disable with --no-dynamic-vad-threshold."""

    model: str = 'turbo'
    """Select Whisper/Faster-Whisper/SimulStreaming model size. See
    https://github.com/openai/whisper#available-models-and-languages for available models."""

    language: str | None = 'auto'
    """Language spoken in the stream. Default option is to auto detect the spoken language. See
    https://github.com/openai/whisper#available-models-and-languages for available languages."""

    use_faster_whisper: bool = False
    """Use Faster-Whisper instead of Whisper. If used with --use-simul-streaming, SimulStreaming with Faster-Whisper
    as the encoder will be used."""

    use_simul_streaming: bool = False
    """Use SimulStreaming instead of Whisper. If used with --use-faster-whisper, SimulStreaming with Faster-Whisper as
    the encoder will be used."""

    use_openai_transcription_api: bool = False
    """Use OpenAI Transcription API instead of the original local Whisper."""

    use_modulate_transcription_api: bool = False
    """Use Modulate multilingual streaming transcription. Audio is sent directly without VAD slicing."""

    use_elevenlabs_scribe_api: bool = False
    """Use ElevenLabs Scribe v2 realtime transcription with local VAD deciding manual commits."""

    modulate_partial_cutoff_seconds: float = 10.0
    """After a Modulate partial utterance has remained open for longer than this many seconds, force a new output
    segment at the next newly observed full-width or half-width punctuation mark."""

    transcribe_reconnect_attempts: Annotated[int, tyro.conf.arg(aliases=('--modulate-reconnect-attempts',))] = 3
    """Maximum automatic reconnect attempts for Modulate and ElevenLabs Scribe. Zero disables retries.
    --modulate-reconnect-attempts is a deprecated alias."""

    transcribe_reconnect_backoff_seconds: Annotated[
        float, tyro.conf.arg(aliases=('--modulate-reconnect-backoff-seconds',))] = 1.0
    """Initial reconnect delay for Modulate and ElevenLabs Scribe. Must be finite and non-negative; zero skips the
    delay. Repeated failures use exponential backoff up to 30 seconds.
    --modulate-reconnect-backoff-seconds is a deprecated alias."""

    modulate_websocket_compression: bool = False
    """Enable per-message deflate compression for the Modulate WebSocket. Disabled by default."""

    openai_transcription_model: str = 'gpt-transcribe'
    """OpenAI's transcription model name, gpt-transcribe / whisper-1 / gpt-4o-mini-transcribe /
    gpt-4o-transcribe."""

    use_hf_asr: bool = False
    """Use a HuggingFace ASR model (via transformers pipeline) specified by --model."""

    transcription_filters: str = 'emoji_filter,repetition_filter'
    """Filters apply to transcription results, separated by ",". We provide emoji_filter and repetition_filter."""

    language_filters: str | None = None
    """Comma-separated languages whose language-specific transcription filters should be enabled. Supported values
    include en/english, ja/japanese, and zh/chinese/zh-cn/zh-tw. Disabled when omitted and independent of --language."""

    transcription_keywords: str | None = None
    """Comma-separated transcription keywords. OpenAI gpt-transcribe sends them as keywords; other Whisper backends
    use them as an initial prompt. SimulStreaming uses them only during initialization; HuggingFace ASR ignores them.
    ElevenLabs Scribe sends non-empty lists as paid keyterms (first 50, at most 20 characters each); empty lists omit
    the keyterms parameter entirely."""

    transcription_context: bool = False
    """Pass the previous transcription result as context. Only one result is retained. SimulStreaming and HuggingFace
    ASR do not support text context propagation. ElevenLabs manages its own session context and does not use this flag
    to inject previous text. Enable with --transcription-context, disabled by default."""

    gpt_model: str = 'gpt-5.6-luna'
    """OpenAI's GPT model name, gpt-5.4-nano / gpt-5.4-mini / gpt-5.6-luna / gpt-5.6-terra."""

    openai_translation_api: Literal['auto', 'responses', 'chat-completions'] = 'auto'
    """OpenAI translation generation API. Auto uses Responses for OpenAI and Chat Completions for a custom
    --openai-base-url, preserving compatibility with older OpenAI-compatible endpoints. Select responses explicitly
    when a custom endpoint implements /v1/responses."""

    gemini_model: str = 'gemini-3.5-flash-lite'
    """Google's Gemini model name, gemini-3-flash-preview / gemini-3.1-flash-lite / gemini-3.5-flash /
    gemini-3.5-flash-lite / gemini-3.6-flash."""

    translation_prompt: str | None = None
    """If set, will translate result text to target language via GPT / Gemini API. Example: "Translate from Japanese
    to Chinese". Adding context (who the streamer is, what the stream is about) improves translation quality. Output
    formatting is controlled internally; use this prompt for translation content, terminology, background and style."""

    translation_history_size: int = 3
    """Number of preceding raw transcripts included in each request's Reference. Must be zero or greater. With
    rolling context enabled, a zero-sized Reference normally serializes requests."""

    translation_threads: int = 3
    """Maximum concurrent translation requests in the new scheduler. Must be at least 1. Results are exported in
    source order. Ignored by --use-old-translator."""

    use_old_translator: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Use the unchanged historical translator as an emergency fallback. Its rolling-context mode is sequential;
    --translation-threads does not change the legacy implementation."""

    use_translation_context: bool = False
    """Maintain an internal rolling LLM-generated summary. Concurrent requests use the latest available context in
    source order and wait when the Reference cannot bridge its gap. If failures make the gap impossible to bridge,
    translation continues with the latest available context and logs INFO. The context itself is never exported."""

    translation_timeout: int = 10
    """Total seconds allowed for a translation after its first dispatch, including retries and retry backoff.
    Expired results are discarded. Zero disables this deadline; individual network calls still have timeouts."""

    use_json_result: bool = False
    """Using JSON result in LLM translation for some locally deployed models."""

    retry_if_translation_fails: bool = False
    """Retry failed translation requests within --translation-timeout, preserving their original Reference and
    context. With a timeout of 0, retries may continue until stopped. Useful for generating subtitles offline."""

    temperature: float | None = None
    """GPT/Gemini parameter. Controls output randomness, higher values produce more diverse results."""

    top_p: float | None = None
    """GPT/Gemini parameter. Nucleus sampling threshold, only tokens with cumulative probability above this value are
    considered."""

    top_k: int | None = None
    """Gemini parameter. Limits token selection to the top K most probable candidates."""

    prompt_cache_key: str | None = None
    """Optional stable OpenAI prompt-cache routing key. It can improve implicit cache routing on its own; when
    --openai-prompt-cache is enabled it overrides the key derived from the model and translation instructions."""

    openai_prompt_cache: bool = False
    """Enable GPT-5.6 explicit prompt caching for the complete, stable translation instructions. Transcript text and
    history remain outside the cached prefix. OpenAI only caches eligible prefixes of at least 1024 tokens."""

    reasoning_effort: str | None = None
    """GPT parameter. Controls reasoning depth for reasoning models. Options: none / minimal / low / medium / high /
    xhigh / max."""

    verbosity: str | None = None
    """GPT parameter. Controls the verbosity of the response. Options: low / medium / high."""

    service_tier: str | None = None
    """GPT parameter. Specifies processing priority tier. Options: auto / default / flex / priority."""

    debug_mode: bool = False
    """Enable debug mode and log usage info after each translation call. Full LLM prompts are logged only when
    -vvv is also enabled."""

    verbose: Annotated[tyro.conf.UseCounterAction[int], tyro.conf.arg(aliases=('-v',))] = 0
    """Increase log verbosity. Use -v for INFO, -vv for DEBUG and -vvv for very detailed debug logging."""

    processing_proxy: str | None = None
    """Use the specified HTTP/HTTPS/SOCKS proxy for Whisper/GPT API (Gemini currently doesn't support specifying a
    proxy within the program), e.g. http://127.0.0.1:7890."""

    output_timestamps: bool = False
    """Output timestamps in the console and non-file export channels. Does not affect file output."""

    output_file_timestamps: bool = False
    """Prefix each transcription and translation line written to a file with [HH:MM:SS,d]."""

    show_transcribe_result: bool = True
    """Print / export the transcription result. Disable with --no-show-transcribe-result to only output the
    translation."""

    output_file_path: str | None = None
    """Append result text to this UTF-8 file. Existing content is preserved."""

    cqhttp_url: str | None = None
    """If set, will send the result text to this Cqhttp server."""

    cqhttp_token: str | None = None
    """Token of cqhttp, if it is not set on the server side, it does not need to fill in."""

    discord_webhook_url: str | None = None
    """If set, will send the result text to this Discord channel."""

    telegram_token: str | None = None
    """Token of Telegram bot."""

    telegram_chat_id: int | None = None
    """If set, will send the result text to this Telegram chat. Needs to be used with --telegram-token."""

    output_proxy: str | None = None
    """Use the specified HTTP/HTTPS/SOCKS proxy for Cqhttp/Discord/Telegram, e.g. http://127.0.0.1:7890."""


def _normalize_input_time_range(config: Config):
    """Ignore incompatible ranges once, or validate before starting any workers."""
    config.time_range_strict = normalize_time_range_strict(config.ss, config.to, config.time_range_strict)
    if config.ss is None and config.to is None:
        return
    reasons = []
    if config.url.lower() == 'device':
        reasons.append('device input')
    if config.live_from_start:
        reasons.append('--live-from-start')
    if reasons:
        options = '--ss, --to and --time-range-strict' if config.time_range_strict else '--ss and --to'
        logger.warning('Ignoring %s with %s', options, ' and '.join(reasons))
        config.ss = config.to = None
        config.time_range_strict = False
        return
    normalize_time_range(config.ss, config.to)


def _create_audio_getter(config: Config):
    _normalize_input_time_range(config)
    time_range = {name: value for name in ('ss', 'to')
                  if (value := getattr(config, name)) is not None}
    if config.time_range_strict:
        time_range['time_range_strict'] = True
    realtime = config.pacer == 'realtime'
    if config.url.lower() == 'device':
        if realtime:
            logger.warning('Ignoring --pacer realtime for device input because audio capture is already real-time')
        logger.debug('Selected %s device audio input', config.device_audio_getter)
        if config.device_audio_getter == 'sounddevice':
            return SoundDeviceAudioGetter(device_index=config.device_index,
                                          interval=config.device_recording_interval,
                                          multiplier=config.device_multiplier,
                                          debug_audio_dir=config.device_audio_debug_dir,
                                          debug_clip_duration=config.device_audio_debug_clip_duration,
                                          debug_max_clips=config.device_audio_debug_max_clips)
        return DeviceAudioGetter(
            device_index=config.device_index,
            use_mic=config.mic,
            interval=config.device_recording_interval,
            multiplier=config.device_multiplier,
            debug_audio_dir=config.device_audio_debug_dir,
            debug_clip_duration=config.device_audio_debug_clip_duration,
            debug_max_clips=config.device_audio_debug_max_clips,
        )
    if config.use_elevenlabs_scribe_api:
        time_range['preserve_final_block'] = True
    if is_url(config.url):
        logger.debug('Selected stream audio input')
        return StreamAudioGetter(
            url=config.url,
            format=config.format,
            cookies=config.cookies,
            proxy=config.input_proxy,
            realtime=realtime,
            wait_for_video=config.wait_for_video,
            live_from_start=config.live_from_start,
            stream_format_diagnostic_file=config.stream_format_diagnostic_file,
            **time_range,
        )
    logger.debug('Selected local file audio input with %s pacing', config.pacer)
    return LocalFileAudioGetter(file_path=config.url, realtime=realtime, **time_range)


def _transcription_engine_label(config: Config) -> str:
    """Return the concise user-facing name of the selected transcription backend."""
    if config.use_elevenlabs_scribe_api:
        return 'ElevenLabs Scribe v2 Realtime'
    if config.use_modulate_transcription_api:
        return 'Modulate'
    if config.use_simul_streaming:
        return 'SimulStreaming'
    if config.use_faster_whisper:
        return 'Faster Whisper'
    if config.use_openai_transcription_api:
        return 'OpenAI'
    if config.use_hf_asr:
        return 'Hugging Face ASR'
    return 'Whisper'


def _create_translator(config: Config):
    """Select one complete implementation; legacy never imports the new translation stack."""
    if not config.translation_prompt:
        return None

    provider_options = ({
        'model': config.gemini_model,
        'temperature': config.temperature,
        'top_p': config.top_p,
        'top_k': config.top_k,
    } if config.google_api_key else {
        'model': config.gpt_model,
        'prompt_cache_key': config.prompt_cache_key,
        'prompt_cache_enabled': config.openai_prompt_cache,
        'api_mode': config.openai_translation_api,
        'temperature': config.temperature,
        'top_p': config.top_p,
        'reasoning_effort': config.reasoning_effort,
        'verbosity': config.verbosity,
        'service_tier': config.service_tier,
    })
    logger.debug('Selected %s translation backend', 'Gemini' if config.google_api_key else 'OpenAI')

    if config.use_old_translator:
        from .translator.legacy.legacy_translator import GPTTranslator, GeminiTranslator

        logger.warning('--use-old-translator is deprecated; using the unchanged historical translator. '
                       '--translation-threads is ignored.')
        translator_type = GeminiTranslator if config.google_api_key else GPTTranslator
        return translator_type(
            prompt=config.translation_prompt,
            history_size=config.translation_history_size,
            use_json_result=config.use_json_result,
            use_translation_context=config.use_translation_context,
            timeout=config.translation_timeout,
            retry_if_translation_fails=config.retry_if_translation_fails,
            debug_mode=config.debug_mode,
            **provider_options,
        )

    from .translator.backends import OpenAITranslationBackend, GeminiTranslationBackend
    from .translator.clients import TranslationClientPool
    from .translator.pipeline import TranslationPipeline
    from .translator.protocol import TranslationProtocol
    from .translator.types import TranslationSettings

    client_pool = TranslationClientPool(
        provider='google' if config.google_api_key else 'openai',
        api_keys=config.google_api_key or config.openai_api_key,
        base_url=config.google_base_url if config.google_api_key else config.openai_base_url,
        proxy=config.processing_proxy,
        verify_ssl=config.verify_ssl,
    )
    protocol = TranslationProtocol(
        prompt=config.translation_prompt,
        history_size=config.translation_history_size,
        use_json_result=config.use_json_result,
        use_translation_context=config.use_translation_context,
    )
    backend_type = GeminiTranslationBackend if config.google_api_key else OpenAITranslationBackend
    backend = backend_type(client_pool=client_pool, protocol=protocol,
                           debug_mode=config.debug_mode, **provider_options)
    settings = TranslationSettings(
        workers=config.translation_threads,
        history_size=config.translation_history_size,
        use_translation_context=config.use_translation_context,
        timeout=config.translation_timeout,
        retry_if_translation_fails=config.retry_if_translation_fails,
    )
    return TranslationPipeline(backend, settings)


def run(config: Config):
    """Run the transcription / translation pipeline until the input is exhausted."""
    logger.info('Initializing transcription and translation pipeline')
    legacy_translation = config.use_old_translator and bool(config.translation_prompt)
    ClientPool.init(openai_api_key=config.openai_api_key if legacy_translation else None,
                    google_api_key=config.google_api_key if legacy_translation else None,
                    proxy=config.processing_proxy,
                    openai_base_url=config.openai_base_url,
                    whisper_api_key=(config.whisper_api_key if legacy_translation else
                                     config.whisper_api_key or config.openai_api_key),
                    whisper_base_url=(config.whisper_base_url if legacy_translation else
                                      config.whisper_base_url or config.openai_base_url),
                    google_base_url=config.google_base_url,
                    verify_ssl=config.verify_ssl)

    # Init queues
    modulate_buffer_chunks = math.ceil(MODULATE_INPUT_QUEUE_SECONDS * SAMPLE_RATE / SAMPLES_PER_FRAME)
    if config.use_elevenlabs_scribe_api:
        scribe_buffer_chunks = math.ceil(ELEVENLABS_INPUT_QUEUE_SECONDS * SAMPLE_RATE / SAMPLES_PER_FRAME)
        getter_to_slicer_queue = queue.Queue(maxsize=scribe_buffer_chunks)
    else:
        getter_to_slicer_queue = (queue.Queue(maxsize=modulate_buffer_chunks)
                                  if config.use_modulate_transcription_api else queue.SimpleQueue())
    slicer_to_transcriber_queue = queue.SimpleQueue()
    transcriber_to_translator_queue = queue.SimpleQueue()
    translator_to_exporter_queue = queue.SimpleQueue() if config.translation_prompt else transcriber_to_translator_queue

    # Init workers
    with ThreadPoolExecutor(thread_name_prefix='initializer') as executor:

        audio_getter_future = executor.submit(_create_audio_getter, config)
        if config.use_modulate_transcription_api or config.use_elevenlabs_scribe_api:
            slicer_future = None
        else:
            # Keep the VAD dependency out of the Modulate path entirely.
            from .slicer.audio_slicer import AudioSlicer
            slicer_future = executor.submit(
                AudioSlicer,
                min_audio_length=config.min_audio_length,
                max_audio_length=config.max_audio_length,
                target_audio_length=config.target_audio_length,
                continuous_no_speech_threshold=config.continuous_no_speech_threshold,
                dynamic_no_speech_threshold=config.dynamic_no_speech_threshold,
                prefix_retention_length=config.prefix_retention_length,
                vad_threshold=config.vad_threshold,
                dynamic_vad_threshold=config.dynamic_vad_threshold,
            )

        def init_transcriber():
            common_args = {
                'transcription_filters': config.transcription_filters,
                'language_filters': config.language_filters,
                'print_result': config.show_transcribe_result,
                'output_timestamps': config.output_timestamps,
                'use_history_context': config.transcription_context,
                'transcription_keywords': config.transcription_keywords,
            }
            if config.use_elevenlabs_scribe_api:
                logger.debug('Selected ElevenLabs Scribe v2 realtime transcription backend')
                return ElevenLabsScribeTranscriber(
                    api_key=config.elevenlabs_api_key,
                    reconnect_attempts=config.transcribe_reconnect_attempts,
                    reconnect_backoff_seconds=config.transcribe_reconnect_backoff_seconds,
                    language=config.language,
                    proxy=config.processing_proxy,
                    verify_ssl=config.verify_ssl,
                    max_audio_send_rate=1.0 if config.pacer == 'realtime' else 5.0,
                    min_audio_length=config.min_audio_length,
                    max_audio_length=config.max_audio_length,
                    target_audio_length=config.target_audio_length,
                    continuous_no_speech_threshold=config.continuous_no_speech_threshold,
                    dynamic_no_speech_threshold=config.dynamic_no_speech_threshold,
                    prefix_retention_length=config.prefix_retention_length,
                    vad_threshold=config.vad_threshold,
                    dynamic_vad_threshold=config.dynamic_vad_threshold,
                    **common_args,
                )
            elif config.use_modulate_transcription_api:
                logger.debug('Selected Modulate streaming transcription backend')
                return ModulateStreamingTranscriber(
                    api_key=config.modulate_api_key,
                    partial_cutoff_seconds=config.modulate_partial_cutoff_seconds,
                    reconnect_attempts=config.transcribe_reconnect_attempts,
                    reconnect_backoff_seconds=config.transcribe_reconnect_backoff_seconds,
                    proxy=config.processing_proxy,
                    verify_ssl=config.verify_ssl,
                    websocket_compression=config.modulate_websocket_compression,
                    max_audio_send_rate=(MODULATE_STREAM_MAX_SEND_RATE if is_url(config.url) else None),
                    language=config.language,
                    **common_args,
                )
            elif config.use_simul_streaming:
                logger.debug('Selected SimulStreaming transcription backend')
                return SimulStreaming(model=config.model,
                                      language=config.language,
                                      use_faster_whisper=config.use_faster_whisper,
                                      proxy=config.processing_proxy,
                                      **common_args)
            elif config.use_faster_whisper:
                logger.debug('Selected Faster-Whisper transcription backend')
                return FasterWhisper(model=config.model,
                                     language=config.language,
                                     proxy=config.processing_proxy,
                                     **common_args)
            elif config.use_openai_transcription_api:
                logger.debug('Selected OpenAI transcription API backend')
                return RemoteOpenaiTranscriber(model=config.openai_transcription_model,
                                               language=config.language,
                                               proxy=config.processing_proxy,
                                               **common_args)
            elif config.use_hf_asr:
                logger.debug('Selected Hugging Face ASR backend')
                return HFTranscriber(model=config.model,
                                     language=config.language,
                                     proxy=config.processing_proxy,
                                     **common_args)
            else:
                logger.debug('Selected Whisper transcription backend')
                return OpenaiWhisper(model=config.model, language=config.language, **common_args)

        logger.warning('Using %s as transcription engine.', _transcription_engine_label(config))
        transcriber_future = executor.submit(init_transcriber)

        translator_future = executor.submit(_create_translator, config)
        exporter_future = executor.submit(
            ResultExporter,
            cqhttp_url=config.cqhttp_url,
            cqhttp_token=config.cqhttp_token,
            discord_webhook_url=config.discord_webhook_url,
            telegram_token=config.telegram_token,
            telegram_chat_id=config.telegram_chat_id,
            output_file_path=config.output_file_path,
            proxy=config.output_proxy,
            output_whisper_result=config.show_transcribe_result,
            output_timestamps=config.output_timestamps,
            output_file_timestamps=config.output_file_timestamps,
        )

        audio_getter = audio_getter_future.result()
        slicer = slicer_future.result() if slicer_future else None
        transcriber = transcriber_future.result()
        translator = translator_future.result()
        exporter = exporter_future.result()
        logger.debug('All pipeline workers initialized')

    logger.warning('Initialization complete, starting up...')

    # Start working
    pipeline_threads = []
    pipeline_failed = threading.Event()

    def stop_audio_getter():
        stop = getattr(audio_getter, 'stop', None)
        if callable(stop):
            stop()

    def stop_translation():
        stop = getattr(translator, 'stop', None)
        if callable(stop):
            stop()

    def stop_transcription():
        stop = getattr(transcriber, 'stop', None)
        if callable(stop):
            stop()

    def stop_pipeline():
        # Cancel network/context waits before the getter publishes EOF. Otherwise
        # a cancelled, still-priming transcriber can mistake EOF for a bad source.
        try:
            stop_translation()
        finally:
            try:
                stop_transcription()
            finally:
                stop_audio_getter()

    stopping_from_signal = False

    def handle_stop_signal(signum, frame):
        nonlocal stopping_from_signal
        if stopping_from_signal:
            # Don't re-enter stop() locks if a second signal interrupts cleanup.
            raise SystemExit(0 if signum == signal.SIGINT else 128 + int(signum))
        stopping_from_signal = True
        # Preserve each getter's platform-specific cleanup and exit status, but
        # wake downstream network waits before its potentially blocking cleanup.
        try:
            stop_translation()
        finally:
            try:
                stop_transcription()
            finally:
                audio_getter._exit_handler(signum, frame)

    def handle_worker_error():
        pipeline_failed.set()
        stop_pipeline()

    if hasattr(audio_getter, '_exit_handler'):
        signal.signal(signal.SIGINT, handle_stop_signal)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, handle_stop_signal)

    try:
        getter_thread = start_daemon_thread(
            audio_getter.loop,
            name='getter',
            on_error=handle_worker_error,
            # Getters own their terminal sentinel, including exceptional exits.
            output_eof_on_error=False,
            output_queue=getter_to_slicer_queue,
        )
        pipeline_threads.append(getter_thread)
        if slicer:
            pipeline_threads.append(start_daemon_thread(
                slicer.loop,
                name='slicer',
                on_error=handle_worker_error,
                input_queue=getter_to_slicer_queue,
                output_queue=slicer_to_transcriber_queue,
            ))
        transcriber_thread = start_daemon_thread(
            transcriber.loop,
            name='transcriber',
            on_error=handle_worker_error,
            input_queue=(getter_to_slicer_queue
                         if config.use_modulate_transcription_api or config.use_elevenlabs_scribe_api
                         else slicer_to_transcriber_queue),
            output_queue=transcriber_to_translator_queue,
        )
        pipeline_threads.append(transcriber_thread)
        translator_thread = None
        if translator:
            translator_thread = start_daemon_thread(
                translator.loop,
                name='translator',
                on_error=handle_worker_error,
                # The new pipeline owns EOF on every exit. The historical
                # translator relies on the worker wrapper after an exception.
                output_eof_on_error=legacy_translation,
                input_queue=transcriber_to_translator_queue,
                output_queue=translator_to_exporter_queue,
            )
            pipeline_threads.append(translator_thread)
        exporter_thread = start_daemon_thread(
            exporter.loop,
            name='exporter',
            on_error=handle_worker_error,
            input_queue=translator_to_exporter_queue,
        )
        pipeline_threads.append(exporter_thread)

        while exporter_thread.is_alive():
            time.sleep(1)
        input_running = getter_thread.is_alive()
        transcription_running = transcriber_thread.is_alive()
        translation_running = translator_thread is not None and translator_thread.is_alive()
        output_finished = getattr(translator, 'output_finished', None)
        translation_output_finished = output_finished is not None and output_finished.is_set()
        if input_running or transcription_running or (translation_running and not translation_output_finished):
            logger.debug('Exporter stopped before its upstream workers; requesting pipeline shutdown')
            if translation_running and not translation_output_finished:
                stop_translation()
            if transcription_running:
                stop_transcription()
            if input_running:
                stop_audio_getter()
    except BaseException:
        # Includes KeyboardInterrupt and SystemExit raised by the getters' signal
        # handlers. A normal input EOF does not take this cancellation path.
        stop_pipeline()
        raise
    finally:
        shutdown_deadline = time.monotonic() + 10
        for worker_thread in pipeline_threads:
            worker_thread.join(timeout=max(0, shutdown_deadline - time.monotonic()))
        remaining_workers = [worker_thread.name for worker_thread in pipeline_threads if worker_thread.is_alive()]
        if remaining_workers:
            pipeline_failed.set()
            logger.error('Pipeline worker(s) did not stop within 10 seconds: %s', ', '.join(remaining_workers))
    if pipeline_failed.is_set():
        raise RuntimeError('Pipeline stopped because a worker failed')
    logger.warning('All processing completed, program exits.')


def _legacy_language_filters_from_argv(argv) -> str | None:
    """Reproduce the former boolean filter selection for an explicitly supplied legacy flag."""
    language = 'auto'
    index = 0
    while index < len(argv):
        normalized_arg = argv[index].replace('_', '-')
        if normalized_arg == '--language' and index + 1 < len(argv):
            language = argv[index + 1]
            index += 2
            continue
        if normalized_arg.startswith('--language='):
            language = normalized_arg.split('=', 1)[1]
        index += 1

    language_key = str(language or 'auto').strip().casefold().replace('_', '-')
    if language_key in ('', 'auto', 'none'):
        return 'en,ja,zh'
    return _LEGACY_LANGUAGE_FILTER_ALIASES.get(language_key)


def _preprocess_deprecated_flags(argv):
    """Rewrite deprecated flags, optional values, and legacy verbosity arguments."""
    processed = []
    args = normalize_verbosity_args(argv)
    legacy_language_filters = _legacy_language_filters_from_argv(args)
    reconnect_aliases = {
        '--modulate-reconnect-attempts': '--transcribe-reconnect-attempts',
        '--modulate-reconnect-backoff-seconds': '--transcribe-reconnect-backoff-seconds',
    }
    positional_only = False
    index = 0
    while index < len(args):
        arg = args[index]
        normalized_arg = arg.replace('_', '-')
        if arg == '--':
            positional_only = True
        option, separator, value = arg.partition('=')
        replacement = reconnect_aliases.get(option.replace('_', '-'))
        if replacement and not positional_only:
            logger.warning('%s is deprecated. Please use %s instead.', option, replacement)
            processed.append(replacement + separator + value)
            index += 1
            continue
        if normalized_arg == '--wait-for-video':
            processed.append('--wait-for-video')
            if index + 1 < len(args):
                try:
                    int(args[index + 1])
                except ValueError:
                    pass
                else:
                    processed.append(args[index + 1])
                    index += 2
                    continue
            processed.append(str(DEFAULT_WAIT_FOR_VIDEO_SECONDS))
            index += 1
            continue
        if normalized_arg.startswith('--wait-for-video='):
            processed.append(normalized_arg)
            index += 1
            continue
        if normalized_arg == '--no-language-based-filter':
            logger.warning('--no-language-based-filter is deprecated and no longer needed because language filters '
                           'are disabled by default. Use --language-filters with an explicit language list to enable '
                           'them.')
            index += 1
            continue
        if normalized_arg == '--language-based-filter':
            logger.warning('--language-based-filter is deprecated. Use --language-filters with an explicit language '
                           'list instead.')
            if legacy_language_filters:
                processed.extend(('--language-filters', legacy_language_filters))
            index += 1
            continue
        if arg.replace('_', '-') == '--hide-transcribe-result':
            logger.warning('--hide_transcribe_result is deprecated and will be removed in future versions. '
                           'Please use --no-show-transcribe-result instead.')
            processed.append('--no-show-transcribe-result')
        elif arg.replace('_', '-') == '--transcription-initial-prompt':
            logger.warning('--transcription-initial-prompt is deprecated and will be removed in future versions. '
                           'Please use --transcription-keywords instead.')
            processed.append('--transcription-keywords')
        else:
            processed.append(arg)
        index += 1
    return processed


def _apply_overall_proxy(config: Config):
    """Propagate --proxy to the environment and to all unset --*-proxy options."""
    if not config.proxy:
        return
    os.environ['http_proxy'] = config.proxy
    os.environ['https_proxy'] = config.proxy
    os.environ['HTTP_PROXY'] = config.proxy
    os.environ['HTTPS_PROXY'] = config.proxy
    if config.input_proxy is None:
        config.input_proxy = config.proxy
    if config.processing_proxy is None:
        config.processing_proxy = config.proxy
    if config.output_proxy is None:
        config.output_proxy = config.proxy
    logger.info('Applied the overall proxy to environment and unset proxy options')


def _print_audio_devices(device_audio_getter: str = 'pyaudio'):
    if device_audio_getter == 'sounddevice':
        try:
            import sounddevice as sd
        except ImportError:
            logger.error('sounddevice is not installed. Unable to list sounddevice devices.')
            sys.exit(1)

        print('Available sounddevice input devices:')
        for index, device in enumerate(sd.query_devices()):
            if int(device.get('max_input_channels', 0)) > 0:
                print(f"{index}: {device['name']}")
        return

    if platform.system() == 'Windows':
        import pyaudiowpatch as pa
    else:
        try:
            import pyaudio as pa
        except ImportError:
            logger.error('PyAudio is not installed. Unable to list devices.')
            logger.error('Debian/Ubuntu/Colab: apt install portaudio19-dev && pip install pyaudio')
            sys.exit(1)

    pyaudio = pa.PyAudio()

    print("Available audio devices:")
    for i in range(pyaudio.get_device_count()):
        dev = pyaudio.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            print(f"{dev['index']}: {dev['name']}")

    if platform.system() == 'Windows':
        print("\nLoopback devices (for system audio):")
        for loopback in pyaudio.get_loopback_device_info_generator():
            print(f"{loopback['index']}: {loopback['name']}")
    pyaudio.terminate()


def _print_stream_formats(config: Config):
    cmd = [sys.executable, '-m', 'yt_dlp', config.url, '-F']
    if config.cookies:
        cmd.extend(['--cookies', config.cookies])
    if config.input_proxy:
        cmd.extend(['--proxy', config.input_proxy])
    logger.info('Listing available stream formats')
    result = subprocess.run(cmd)
    if result.returncode:
        logger.error('yt-dlp format listing exited with status %d', result.returncode)


def _check_ffmpeg(config: Config):
    if config.url.lower() != 'device' and not shutil.which('ffmpeg'):
        if platform.system() == 'Windows':
            logger.error('ffmpeg not found. Please install it with: winget install ffmpeg')
        else:
            logger.error('ffmpeg not found. Please install it with: sudo apt install ffmpeg')
        sys.exit(1)


def _validate_and_normalize(config: Config):
    """Check option combinations and normalize values, exiting with an error message on invalid input."""
    try:
        _normalize_input_time_range(config)
    except ValueError as exc:
        logger.error('%s', exc)
        sys.exit(1)

    if config.openai_translation_api == 'auto':
        config.openai_translation_api = 'chat-completions' if config.openai_base_url else 'responses'
        logger.info('Resolved OpenAI translation API mode to %s', config.openai_translation_api)

    for option, value, minimum in (
            ('--translation-threads', config.translation_threads, 1),
            ('--translation-history-size', config.translation_history_size, 0),
            ('--translation-timeout', config.translation_timeout, 0)):
        if value < minimum:
            logger.error('%s must be %d or greater', option, minimum)
            sys.exit(1)

    if not config.use_elevenlabs_scribe_api and config.model.endswith('.en'):
        if config.model == 'large.en':
            logger.error('English model does not have large model, please choose from '
                         '{tiny.en, small.en, medium.en}')
            sys.exit(1)
        if config.language != 'English' and config.language != 'en':
            if config.language == 'auto':
                logger.warning('Using .en model, setting language from auto to English')
                config.language = 'en'
            else:
                logger.error('English model cannot be used to detect non english language, '
                             'please choose a non .en model')
                sys.exit(1)

    if config.url.lower() == 'device':
        if config.device_recording_interval <= 0:
            logger.error('Device recording interval must be greater than zero')
            sys.exit(1)
        try:
            from .getter.audio_getter import _device_gain
            _device_gain(config.device_multiplier)
        except ValueError as exc:
            logger.error('%s', exc)
            sys.exit(1)
        if config.device_audio_debug_dir and config.device_audio_debug_clip_duration <= 0:
            logger.error('Device audio debug clip duration must be greater than zero')
            sys.exit(1)
        if config.device_audio_debug_dir and config.device_audio_debug_max_clips <= 0:
            logger.error('Device audio debug max clips must be greater than zero')
            sys.exit(1)

    if config.wait_for_video is not None and config.wait_for_video <= 0:
        logger.error('Wait-for-video interval must be greater than zero')
        sys.exit(1)

    transcription_encoder_flag_num = 0
    transcription_decoder_flag_num = 0
    if config.use_faster_whisper:
        transcription_encoder_flag_num += 1
    if config.use_simul_streaming:
        transcription_decoder_flag_num += 1
    if config.use_openai_transcription_api:
        transcription_encoder_flag_num += 1
        transcription_decoder_flag_num += 1
    if config.use_modulate_transcription_api:
        transcription_encoder_flag_num += 1
        transcription_decoder_flag_num += 1
    if config.use_elevenlabs_scribe_api:
        transcription_encoder_flag_num += 1
        transcription_decoder_flag_num += 1
    if config.use_hf_asr:
        transcription_encoder_flag_num += 1
        transcription_decoder_flag_num += 1
    if transcription_encoder_flag_num > 1:
        logger.error('Cannot combine Faster Whisper, OpenAI Transcription API, Modulate, ElevenLabs Scribe or '
                     'HuggingFace ASR backends')
        sys.exit(1)
    if transcription_decoder_flag_num > 1:
        logger.error('Cannot combine SimulStreaming, OpenAI Transcription API, Modulate, ElevenLabs Scribe or '
                     'HuggingFace ASR backends')
        sys.exit(1)

    if config.use_openai_transcription_api and not (config.whisper_api_key or config.openai_api_key):
        logger.error('Please fill in the Whisper API key or OpenAI API key when enabling remote transcription')
        sys.exit(1)

    if config.use_elevenlabs_scribe_api:
        if not (config.elevenlabs_api_key or '').strip():
            logger.error('Please fill in the ElevenLabs API key when enabling Scribe realtime transcription')
            sys.exit(1)
        if (not math.isfinite(config.min_audio_length) or config.min_audio_length <= 0
                or not math.isfinite(config.max_audio_length) or config.max_audio_length <= 0):
            logger.error('ElevenLabs Scribe minimum and maximum audio lengths must be finite and greater than zero')
            sys.exit(1)
        if config.max_audio_length > 30:
            logger.warning('ElevenLabs Scribe limits local audio commits to 30 seconds; reducing '
                           '--max-audio-length from %.3f to 30', config.max_audio_length)
            config.max_audio_length = 30.0
        if config.min_audio_length > config.max_audio_length:
            logger.error('ElevenLabs Scribe minimum audio length must not exceed the maximum audio length')
            sys.exit(1)

    if config.use_modulate_transcription_api:
        if not (config.modulate_api_key or '').strip():
            logger.error('Please fill in the Modulate API key when enabling Modulate streaming transcription')
            sys.exit(1)
        if (not math.isfinite(config.modulate_partial_cutoff_seconds)
                or config.modulate_partial_cutoff_seconds <= 0):
            logger.error('Modulate partial cutoff seconds must be a finite value greater than zero')
            sys.exit(1)
        if config.url.lower() != 'device' and not is_url(config.url) and config.pacer != 'realtime':
            logger.warning('Modulate local-file transcription works best with --pacer realtime so partial '
                           'cutoff timing follows the media timeline')

    if config.use_modulate_transcription_api or config.use_elevenlabs_scribe_api:
        if config.transcribe_reconnect_attempts < 0:
            logger.error('--transcribe-reconnect-attempts must be zero or greater')
            sys.exit(1)
        if (not math.isfinite(config.transcribe_reconnect_backoff_seconds)
                or config.transcribe_reconnect_backoff_seconds < 0):
            logger.error('--transcribe-reconnect-backoff-seconds must be a finite value of zero or greater')
            sys.exit(1)

    if config.translation_prompt and not (config.openai_api_key or config.google_api_key):
        logger.error('Please fill in the OpenAI / Google API key when enabling LLM translation')
        sys.exit(1)
    if config.use_translation_context and not config.translation_prompt:
        logger.error('--use-translation-context requires --translation-prompt')
        sys.exit(1)
    if config.translation_prompt and not config.google_api_key and config.openai_prompt_cache:
        if config.openai_translation_api != 'responses':
            logger.error('OpenAI explicit prompt caching requires --openai-translation-api responses')
            sys.exit(1)
        if config.use_old_translator:
            from .translator.legacy.legacy_translator import _gpt_model_version
        else:
            from .translator.backends import _gpt_model_version
        model_version = _gpt_model_version(config.gpt_model)
        if model_version is not None and model_version < (5, 6):
            logger.error('OpenAI explicit prompt caching requires GPT-5.6 or newer; selected model is %s',
                         config.gpt_model)
            sys.exit(1)

    if config.language == 'auto':
        config.language = None

    if config.output_file_path:
        output_dir = os.path.dirname(os.path.abspath(config.output_file_path))
        if not os.path.isdir(output_dir):
            logger.error('Output directory does not exist: %s', output_dir)
            sys.exit(1)


def cli():
    argv = sys.argv[1:]
    bootstrap_verbosity = verbosity_from_argv(argv)
    if any(arg.replace('_', '-') == '--debug-mode' for arg in argv):
        bootstrap_verbosity = max(bootstrap_verbosity, 2)
    configure_logging(bootstrap_verbosity)

    try:
        logger.warning('Version: %s', __version__)
        config = tyro.cli(Config, args=_preprocess_deprecated_flags(argv), prog='stream-translator-gpt')
        configure_logging(max(config.verbose, 2 if config.debug_mode else 0))

        _apply_overall_proxy(config)

        if config.list_devices:
            _print_audio_devices(config.device_audio_getter)
            return
        if config.list_format:
            _print_stream_formats(config)
            return

        _check_ffmpeg(config)
        _validate_and_normalize(config)
        run(config)
    finally:
        shutdown_logging()


if __name__ == '__main__':
    cli()
