# stream-translator-gpt

[![PyPI version](https://badge.fury.io/py/stream-translator-gpt.svg)](https://badge.fury.io/py/stream-translator-gpt) [![Python Versions](https://img.shields.io/pypi/pyversions/stream-translator-gpt.svg)](https://pypi.org/project/stream-translator-gpt/) [![Downloads](https://static.pepy.tech/badge/stream-translator-gpt)](https://pepy.tech/project/stream-translator-gpt) [![License](https://img.shields.io/github/license/ionic-bond/stream-translator-gpt.svg)](https://github.com/ionic-bond/stream-translator-gpt/blob/main/LICENSE) [![Gradio](https://img.shields.io/badge/WebUI-Gradio-orange)](https://gradio.app)

**English** | [**中文**](./README_CN.md) | [**日本語**](./README_JP.md)

Real-time transcription and translation for live streams, local media files, and device audio. Available as a command-line tool and a Gradio WebUI.

## Quick Start on Colab (Recommended)

The easiest way to use this tool, no local environment to set up, and Colab's performance is more than enough for stable everyday use. All you need is your own API key, depending on which services you use:

- [Create a **Google API key**](https://aistudio.google.com/app/apikey) for **Gemini API** translation (recommended, since the **Gemini Flash-Lite** model has a free quota of 15 requests per minute / 500 per day)
- [Create an **OpenAI API key**](https://platform.openai.com/api-keys) for **OpenAI Transcription API** transcription or **GPT API** translation (any **OpenAI-compatible API** can also be used)
- A [**Modulate API key**](https://docs.modulate.ai/guides/authentication) is required to use **Modulate Multilingual Transcription — Streaming**.

|                                                                                        Command Line                                                                                        |                                                                                        WebUI                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/command_line.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/webui.ipynb) |

> [!NOTE]
> Due to frequent scraping and theft of API keys, we are unable to provide a trial API key. You need to fill in your own API key.

## How It Works

```mermaid
flowchart LR
    subgraph ga["`**Input**`"]
        direction LR
        aa("`**FFmpeg**`")
        ab("`**Device audio**`")
        ac("`**yt-dlp**`")
        ad("`**Local media file**`")
        ae("`**Live streaming**`")
        ac --> aa
        ad --> aa
        ae --> ac
    end
    subgraph gb["`**Audio Slicing**`"]
        direction LR
        ba("`**Silero VAD**`")
    end
    subgraph gc["`**Transcription**`"]
        direction LR
        ca("`**Whisper**`")
        cb("`**Faster-Whisper**`")
        cc("`**SimulStreaming**`")
        cd("`**OpenAI Transcription API**`")
        ce("`**HuggingFace ASR**`")
        cf("`**Modulate Streaming**`")
    end
    subgraph gd["`**Translation**`"]
        direction LR
        da("`**GPT API**`")
        db("`**Gemini API**`")
    end
    subgraph ge["`**Output**`"]
        direction LR
        ea("`**Print to terminal**`")
        ee("`**Save to file**`")
        ec("`**Discord**`")
        ed("`**Telegram**`")
        eb("`**Cqhttp**`")
    end
    aa --> gb
    ab --> gb
    gb ==> gc
    aa -.-> cf
    ab -.-> cf
    gc ==> gd
    gd ==> ge
```

- **Input**: [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) extracts audio from live streams; local media files and PC device audio are also supported.
- **Audio slicing**: dynamic-threshold slicing based on [**Silero-VAD**](https://github.com/snakers4/silero-vad). The Modulate streaming backend consumes the 16 kHz audio stream directly and bypasses VAD.
- **Transcription**: locally with [**Whisper**](https://github.com/openai/whisper) / [**Faster-Whisper**](https://github.com/SYSTRAN/faster-whisper) / [**SimulStreaming**](https://github.com/ufal/SimulStreaming) / [**HuggingFace ASR**](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition), or remotely with the [**OpenAI Transcription API**](https://platform.openai.com/docs/guides/speech-to-text) / [**Modulate Multilingual Transcription — Streaming**](https://docs.modulate.ai/api-reference/stt/streaming).
- **Translation** (optional): OpenAI's [**GPT API**](https://platform.openai.com/docs/overview) or Google's [**Gemini API**](https://ai.google.dev/gemini-api/docs).
- **Output**: print to **terminal**, append to a UTF-8 **text file**, or send to **Discord** / **Telegram** / **Cqhttp**.

## Local Installation (Advanced)

Running locally requires some experience with Python environments (especially on Windows) — you are essentially rebuilding an environment similar to Colab's. If in doubt, use Colab instead.

1. **Python** >= 3.10
2. **FFmpeg** (skip if already installed):
   - Windows: `winget install ffmpeg`
   - Linux (Debian/Ubuntu): `sudo apt install ffmpeg`
3. For **local transcription** (Whisper / Faster-Whisper / SimulStreaming / HuggingFace ASR) — not needed if you only use the OpenAI Transcription API or Modulate Streaming:
   - [Install **CUDA** on your system](https://developer.nvidia.com/cuda-downloads)
   - [Install **PyTorch** (with CUDA) to your Python](https://pytorch.org/get-started/locally/)
   - [Install **cuDNN** to your CUDA dir](https://developer.nvidia.com/cudnn-downloads) if you want to use **Faster-Whisper**

Then install the package:

```
pip install stream-translator-gpt -U
```

Or with the WebUI included:

```
pip install stream-translator-gpt[webui] -U
```

## Usage

### WebUI

```
stream-translator-gpt-webui
```

To select the WebUI interface language at startup, use `--ui-language` with `en`, `zh`, or `ja`.

Then open the printed local URL in your browser. All CLI features are available through the interface, and your settings can be saved as presets.

### Command line

```
stream-translator-gpt URL [OPTIONS]
```

Use `--ss TIME` / `--to TIME` to process a source interval, for example
`stream-translator-gpt recording.mp3 --ss 00:10:00 --to 00:12:00` (two minutes).
These are FFmpeg-style positions: `--to` is the absolute end, not a duration. Both options are optional;
times must be non-negative seconds or `[HH:]MM:SS[.fraction]` (seconds also accept `s`, `ms`, `us`).
URL inputs use yt-dlp's FFmpeg downloader. HLS/TS inputs default to fast seeking directly to the target;
the result can be offset by a segment, with no guaranteed 10-second error limit.
Add `--time-range-strict` for reliable HLS/TS trimming: decode from the beginning, then discard the preceding
audio. This reads the prefix and can delay startup, but does not enforce 1× speed or send discarded audio to an API.
Other seekable files and media URLs retain efficient indexed seeking with precise trimming in either mode.
For other formats, servers, media indexing, and live DVR availability determine how much data can be skipped.
Output timestamps start at zero for the selected interval. Device input and `--live-from-start` warn and ignore
range options. `--time-range-strict` warns and is ignored without `--ss` or `--to`; `--pacer` works independently.

**Transcription backends** (default is local **Whisper**):

- ```stream-translator-gpt {URL} --language {input_language}```
- **Faster-Whisper**: ```stream-translator-gpt {URL} --language {input_language} --use-faster-whisper```
- **SimulStreaming**: ```stream-translator-gpt {URL} --language {input_language} --use-simul-streaming```
- **SimulStreaming** with **Faster-Whisper** as the encoder: ```stream-translator-gpt {URL} --language {input_language} --use-simul-streaming --use-faster-whisper```
- **OpenAI Transcription API**: ```stream-translator-gpt {URL} --language {input_language} --use-openai-transcription-api --openai-api-key {your_openai_key}```
- **Modulate Multilingual Transcription — Streaming**: ```stream-translator-gpt {URL} --use-modulate-transcription-api --modulate-api-key {your_modulate_key}```
- **HuggingFace ASR** model (requires `pip install stream-translator-gpt[hf_asr]`; only models with `pipeline_tag: automatic-speech-recognition` on Hugging Face Hub are supported): ```stream-translator-gpt {URL} --model {hf_model_name} --use-hf-asr```

The Modulate backend sends 16 kHz audio directly to the model's streaming mode without Silero VAD. Every completed model utterance becomes a new sentence. Partial results are monitored for long utterances: once `--modulate-partial-cutoff-seconds` has elapsed from the first uncommitted text (or a forced split that already has a remainder), the next full-width or half-width punctuation mark commits the current segment. Silence between sentences doesn't consume the next sentence's timeout. For URL streams, Modulate normally follows the source rate at the live edge; while reconnect replay or the live input queue remains buffered, it sends audio at up to 1.5x the source rate to catch up. Unexpected transport interruptions are retried with exponential backoff; after reconnecting, the program replays up to 10 minutes of audio received since the most recent final result. Explicit authentication, quota, and API errors are not retried. For a local file, use real-time playback so the timeout follows the media timeline:

```bash
stream-translator-gpt /path/to/file --pacer realtime --use-modulate-transcription-api --modulate-api-key {your_modulate_key}
```

**Translation** (enabled by setting `--translation-prompt`; the provider is chosen by which API key you fill in):

- By **Gemini**: ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key}```
- By **GPT**: ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --openai-api-key {your_openai_key}```
- **OpenAI Transcription API** and **Gemini** at the same time: ```stream-translator-gpt {URL} --language {input_language} --use-openai-transcription-api --openai-api-key {your_openai_key} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key}```

> [!TIP]
> The translation prompt is passed to the LLM as-is, so it can carry more than just the language pair. If your API quota allows, give the model some context — who the streamer is, what the stream is about, preferred terminology — and it will translate more accurately and fix ASR errors more reliably.

**Input sources** (besides stream URLs):

- **Local video/audio file**: ```stream-translator-gpt /path/to/file --language {input_language}```
- **System audio** (loopback): ```stream-translator-gpt device --language {input_language}```
- **Microphone**: ```stream-translator-gpt device --mic --language {input_language}```

**Output destinations** (besides the terminal):

- **Discord**: ```stream-translator-gpt {URL} --language {input_language} --discord-webhook-url {your_discord_webhook_url}```
- **Telegram**: ```stream-translator-gpt {URL} --language {input_language} --telegram-token {your_telegram_token} --telegram-chat-id {your_telegram_chat_id}```
- **Cqhttp**: ```stream-translator-gpt {URL} --language {input_language} --cqhttp-url {your_cqhttp_url} --cqhttp-token {your_cqhttp_token}```
- **Timestamped text file**: ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key} --no-show-transcribe-result --retry-if-translation-fails --output-file-path ./result.txt --output-file-timestamps```

### Package layout

`stream_translator_gpt/` is organized by pipeline stage: `getter/` captures audio, `slicer/` handles VAD and
segmentation, `transcriber/` transcribes, `translator/` translates, `exporter/` publishes results, and `utils/`
holds shared helpers. `main.py` retains the CLI and pipeline assembly. Transcription filters and SimulStreaming
live under `transcriber/`; the VAD model is bundled with `slicer/`. The independent top-level `utils/term_utils.py`
and its `stream-translator-build-terms` command are unchanged.

### Translation architecture

The default translation stack separates lifecycle (`translator/pipeline.py`), scheduling and ordered publication
(`translator/scheduler.py`), request-worker loops (`translator/worker.py`), provider API calls
(`translator/backends.py`), worker-owned clients (`translator/clients.py`), and prompt/result handling
(`translator/protocol.py`). Components exchange immutable requests and results defined in `translator/types.py`;
only the scheduler updates pipeline tasks. A worker never waits for context, output order, or retry backoff.

`--use-old-translator` selects the complete, unchanged implementation from commit `ab314cf`, archived in
`translator/legacy/legacy_translator.py`. It does not import or use the new translation stack. Its original client pool, retry/timeout
behavior and shutdown limitations remain unchanged: context mode is sequential, otherwise the limit is 10 concurrent
requests. `--translation-threads` applies only to the new implementation.

## All Options

> [!NOTE]
> All options accept both hyphens and underscores: `--openai-api-key` and `--openai_api_key` are equivalent.
> Every boolean option has an inverse `--no-*` form that turns it off (e.g. `--no-dynamic-vad-threshold` disables `--dynamic-vad-threshold`, which is on by default).

| Option                             | Default Value                  | Description                                                                                                                                                                                                                                                           |
| :--------------------------------- | :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `URL`                              |                                | The URL of the stream. If a local file path is filled in, it will be used as input. If fill in "device", the input will be obtained from your PC device.                                                                                                              |
| **Overall Options**                |
| `--openai-api-key`                 |                                | OpenAI API key for GPT translation. It is also the fallback key for remote transcription when `--whisper-api-key` is not set. Multiple comma-separated keys are rotated.                                                                                           |
| `--whisper-api-key`                |                                | API key for a Whisper-compatible transcription endpoint. Overrides `--openai-api-key` for remote transcription. Multiple comma-separated keys are rotated.                                                                                                       |
| `--modulate-api-key`               |                                | API key for Modulate Multilingual Transcription — Streaming. Required when `--use-modulate-transcription-api` is enabled.                                                                                                                                            |
| `--google-api-key`                 |                                | Google API key if using Gemini translation. If you have multiple keys, you can separate them with "," and each key will be used in turn.                                                                                                                              |
| `--openai-base-url`                |                                | Customize the OpenAI endpoint for GPT translation. It is also the fallback transcription endpoint when `--whisper-base-url` is not set.                                                                                                                            |
| `--whisper-base-url`               |                                | Customize the endpoint used by the Whisper-compatible transcription API.                                                                                                                                                                                           |
| `--google-base-url`                |                                | Customize the API endpoint of Google (Affects Gemini translation).                                                                                                                                                                                                    |
| `--no-verify-ssl`                  |                                | Disable TLS certificate verification for OpenAI / Google API and HuggingFace downloads. Use this when your API endpoint or proxy has a self-signed or invalid certificate. If the base URL host is a bare IP, verification is disabled automatically.                 |
| `--proxy`                          |                                | Used to set the proxy for all --*-proxy options if they are not specifically set. Also sets http_proxy environment variables.                                                                                                                                         |
| **Input Options**                  |
| `--format`                         | ba/wa*                         | Stream format code, this parameter will be passed directly to yt-dlp. You can get the list of available format codes by `yt-dlp {url} -F`                                                                                                                             |
| `--list-format`                    |                                | Print all available formats then exit.                                                                                                                                                                                                                                |
| `--cookies`                        |                                | Used to open member-only stream, this parameter will be passed directly to yt-dlp.                                                                                                                                                                                    |
| `--input-proxy`                    |                                | Use the specified HTTP/HTTPS/SOCKS proxy for yt-dlp, e.g. http://127.0.0.1:7890.                                                                                                                                                                                      |
| `--wait-for-video [SECONDS]`       | disabled; bare flag: 60        | Wait for a scheduled stream to become available. This is yt-dlp's minimum interval between availability checks; a known schedule may sleep until its start time. Omit the option to disable waiting; pass it without a value to use 60 seconds.                    |
| `--live-from-start`                | False                          | Ask yt-dlp to download a supported livestream from its beginning instead of the current live edge. This yt-dlp feature is experimental and extractor-dependent.                                                                                                     |
| `--ss TIME`                       | disabled                       | Start at this source position for local files or URLs; non-negative seconds or `[HH:]MM:SS[.fraction]`. Ignored with a warning for device input or `--live-from-start`. |
| `--to TIME`                       | disabled                       | Stop at this absolute source position, not a duration; must be greater than `--ss` (or zero). Ignored with a warning for device input or `--live-from-start`. |
| `--time-range-strict`             | disabled                       | Use reliable linear trimming for HLS/TS instead of fast, potentially segment-offset seeking. Reads preceding audio. Ignored with a warning without `--ss` / `--to`, for device input, or with `--live-from-start`. |
| `--stream-format-diagnostic-file PATH` | disabled                  | Have yt-dlp write allowlisted metadata about the selected stream format to the user-specified path, replacing that file for each stream. This is disabled unless explicitly supplied; no implicit diagnostic file is created.                                      |
| `--pacer`                          | unlimited                      | Input pacing for URL streams and local files: `unlimited` reads as fast as downstream capacity permits; `realtime` emits one second of decoded audio per second of wall-clock time. Device input is already real-time, so a non-default value is ignored with a warning. |
| `--device-index`                   |                                | The index of the device that needs to be recorded. If not set, the system default recording device will be used.                                                                                                                                                      |
| `--list-devices`                   |                                | Print all audio devices info then exit.                                                                                                                                                                                                                               |
| `--device-recording-interval`      | 0.5                            | The shorter the recording interval, the lower the latency, but it will increase CPU usage. It is recommended to set it between 0.1 and 1.0.                                                                                                                           |
| `--device-audio-getter`            | pyaudio                        | Select `pyaudio` or the manual `sounddevice` fallback. SoundDevice captures input-capable devices; choose a monitor, stereo-mix, or virtual input with `--device-index` for system audio.                                                                            |
| `--device-multiplier`              | 1.0                            | Legacy perceptual volume factor used by both device getters. 1 keeps the original amplitude.                                                                                                                                                                       |
| `--device-audio-debug-dir`          |                                | Save processed device input as bounded mono 16 kHz PCM WAV clips in this directory. Disabled by default.                                                                                                                                                            |
| `--device-audio-debug-clip-duration` | 10.0                           | Duration in seconds of each device debug WAV clip.                                                                                                                                                                                                                  |
| `--device-audio-debug-max-clips`    | 10                             | Maximum number of device debug WAV clips written during one run.                                                                                                                                                                                                    |
| `--mic`                            |                                | Use microphone instead of system audio (loopback).                                                                                                                                                                                                                    |
| **Audio Slicing Options**          |
| `--min-audio-length`               | 0.5                            | Minimum slice audio length in seconds.                                                                                                                                                                                                                                |
| `--max-audio-length`               | 30.0                           | Maximum slice audio length in seconds.                                                                                                                                                                                                                                |
| `--target-audio-length`            | 5.0                            | When dynamic no speech threshold is enabled (enabled by default), the program will slice the audio as close to this length as possible.                                                                                                                               |
| `--continuous-no-speech-threshold` | 1.0                            | Slice if there is no speech during this number of seconds. If the dynamic no speech threshold is enabled (enabled by default), the actual threshold will be dynamically adjusted based on this value.                                                                 |
| `--no-dynamic-no-speech-threshold` |                                | Disable dynamic no speech threshold (enabled by default).                                                                                                                                                                                                             |
| `--prefix-retention-length`        | 0.5                            | The length of the retention prefix audio during slicing.                                                                                                                                                                                                              |
| `--vad-threshold`                  | 0.35                           | Range 0~1. The higher this value, the stricter the speech judgment. If dynamic VAD threshold is enabled (enabled by default), this threshold will be adjusted dynamically based on this value.                                                                        |
| `--no-dynamic-vad-threshold`       |                                | Disable dynamic VAD threshold (enabled by default).                                                                                                                                                                                                                   |
| **Transcription Options**          |
| `--model`                          | turbo                          | Select Whisper/Faster-Whisper/SimulStreaming model size. See [here](https://github.com/openai/whisper#available-models-and-languages) for available models.                                                                                                           |
| `--language`                       | auto                           | Language spoken in the stream. Default option is to auto detect the spoken language. See [here](https://github.com/openai/whisper#available-models-and-languages) for available languages.                                                                            |
| `--use-faster-whisper`             |                                | Use Faster-Whisper instead of Whisper. If used with --use-simul-streaming, SimulStreaming with Faster-Whisper as the encoder will be used.                                                                                                                            |
| `--use-simul-streaming`            |                                | Use SimulStreaming instead of Whisper. If used with --use-faster-whisper, SimulStreaming with Faster-Whisper as the encoder will be used.                                                                                                                             |
| `--use-openai-transcription-api`   |                                | Use OpenAI Transcription API instead of the original local Whisper.                                                                                                                                                                                                   |
| `--use-modulate-transcription-api` |                                | Use Modulate Multilingual Transcription — Streaming. It streams 16 kHz audio directly and bypasses Silero VAD.                                                                                                                                                        |
| `--modulate-partial-cutoff-seconds` | 10.0                           | After this many seconds from the first uncommitted text, commit at the next full-width or half-width punctuation mark. Silence before that text does not count.                                                                                                      |
| `--modulate-reconnect-attempts`    | 3                              | Maximum number of reconnect attempts after an unexpected Modulate transport interruption. Explicit authentication, quota, and API errors are not retried.                                                                                                            |
| `--modulate-reconnect-backoff-seconds` | 1.0                        | Initial reconnect delay in seconds. Each subsequent retry uses exponential backoff; a successful reconnect replays up to 10 minutes of audio since the most recent final result.                                                                                      |
| `--modulate-websocket-compression` | False                          | Enable WebSocket per-message deflate for Modulate. Compression can reduce network bandwidth at the cost of additional CPU work; the default uncompressed connection favors CPU-constrained deployments.                                                               |
| `--openai-transcription-model`     | gpt-transcribe                 | OpenAI transcription model. `gpt-transcribe` is recommended for ordinary transcription; `whisper-1`, `gpt-4o-mini-transcribe`, and `gpt-4o-transcribe` are also supported.                                                                                            |
| `--use-hf-asr`                     |                                | Use a HuggingFace ASR model. Use `--model` to specify the model ID. Requires `pip install stream-translator-gpt[hf_asr]`.                                                                                                                                             |
| `--transcription-filters`          | emoji_filter,repetition_filter | Filters apply to transcription results, separated by ",". We provide emoji_filter and repetition_filter.                                                                                                                                                              |
| `--language-filters`               |                                | Comma-separated language-specific transcription filters to enable. Supported aliases: `en`/`english`, `ja`/`japanese`, and `zh`/`chinese` (including `zh-cn` and `zh-tw`). Disabled when omitted.                                                                        |
| `--transcription-keywords`         |                                | Comma-separated transcription keywords. `gpt-transcribe` sends them as `keywords`; other Whisper backends use them as an initial prompt. The deprecated `--transcription-initial-prompt` alias is still accepted.                                                     |
| `--transcription-context`          |                                | Pass the previous transcription result as context; only one result is retained. SimulStreaming and HuggingFace ASR do not support text context (disabled by default).                                                                                                 |
| **Translation Options**            |
| `--gpt-model`                      | gpt-5.6-luna                   | OpenAI's GPT model name, gpt-5.4-nano / gpt-5.4-mini / gpt-5.6-luna / gpt-5.6-terra                                                                                                                                                                                   |
| `--openai-translation-api`         | auto                           | OpenAI translation API mode: auto / responses / chat-completions. `auto` uses Responses for OpenAI's endpoint and Chat Completions when `--openai-base-url` is set; explicitly select `responses` for a custom endpoint that implements `/v1/responses`.                |
| `--gemini-model`                   | gemini-3.5-flash-lite          | Google's Gemini model name, gemini-3-flash-preview / gemini-3.1-flash-lite / gemini-3.5-flash / gemini-3.5-flash-lite / gemini-3.6-flash                                                                                                                              |
| `--translation-prompt`             |                                | If set, will translate the result text to target language via GPT / Gemini API (According to which API key is filled in). Example: "Translate from Japanese to Chinese". Adding context (who the streamer is, what the stream is about) improves translation quality. Output formatting is controlled internally; use this prompt only for translation content, terminology, background and style. |
| `--translation-history-size`       | 3                              | Number of preceding raw transcripts included as `Reference` (at least 0). With rolling context enabled, a value of 0 normally serializes requests. |
| `--translation-threads`            | 3                              | Maximum concurrent translation requests (at least 1). Work is assigned to idle threads; output remains in source order. With rolling context, values greater than the Reference window + 1 produce a warning but are accepted. |
| `--use-old-translator`             |                                | Emergency fallback to the unchanged historical translator, with a deprecation warning. It ignores `--translation-threads` and processes rolling-context requests sequentially. |
| `--use-translation-context`        |                                | Maintain an internal summary of at most 100 Unicode characters for longer-horizon topics, activities, entities, and unresolved state. Concurrent requests use the latest available `Context` in source order plus recent raw `Reference` transcripts. Requests wait when Reference cannot bridge the gap; if no pending request or retry can repair it, continue with the latest available context and log INFO. Context is never exported. |
| `--translation-timeout`            | 10                             | Total seconds from a translation's first dispatch, including retries and retry backoff. Expired results are discarded. 0 disables the overall deadline, but individual network calls still time out. |
| `--use-json-result`                |                                | Using JSON result in LLM translation for some locally deployed models.                                                                                                                                                                                                |
| `--retry-if-translation-fails`     |                                | Retry failed requests within the overall translation deadline, reusing the original Reference/context snapshot. With `--translation-timeout 0`, retries may continue until stopped. |
| `--temperature`                    |                                | GPT/Gemini parameter. Controls output randomness, higher values produce more diverse results.                                                                                                                                                                         |
| `--top-p`                          |                                | GPT/Gemini parameter. Nucleus sampling threshold, only tokens with cumulative probability above this value are considered.                                                                                                                                            |
| `--top-k`                          |                                | Gemini parameter. Limits token selection to the top K most probable candidates.                                                                                                                                                                                       |
| `--openai-prompt-cache`            |                                | Opt in to GPT-5.6 explicit prompt caching in Responses mode. The breakpoint follows the complete stable system instructions; dynamic history and the current transcript stay after it. Prefixes shorter than 1,024 tokens are not cached and produce a warning.       |
| `--prompt-cache-key`               |                                | Optional prompt-cache routing key. Without `--openai-prompt-cache` it is still sent for implicit cache routing; with explicit caching it overrides the stable key derived from the model and complete system instructions.                                               |
| `--reasoning-effort`               | auto (lowest supported)        | GPT parameter. Controls reasoning depth for reasoning models. Options: none / minimal / low / medium / high / xhigh / max.                                                                                                                                            |
| `--verbosity`                      |                                | GPT parameter. Controls the verbosity of the response. Options: low / medium / high.                                                                                                                                                                                  |
| `--service-tier`                   |                                | GPT parameter. Specifies processing priority tier. Options: auto / default / flex / priority.                                                                                                                                                                         |
| `--debug-mode`                     |                                | Enable debug mode and log usage info after each translation call. Full LLM prompts are logged only when `-vvv` is also enabled.                                                                                                                                       |
| `--processing-proxy`               |                                | Use the specified HTTP/HTTPS/SOCKS proxy for Whisper/GPT API (Gemini currently doesn't support specifying a proxy within the program), e.g. http://127.0.0.1:7890.                                                                                                    |
| **Output Options**                 |
| `--output-timestamps`              |                                | Output timestamps to the console and non-file export channels. This does not affect output files.                                                                                                                                                                    |
| `--no-show-transcribe-result`      |                                | Hide the transcription result (shown by default).                                                                                                                                                                                                                     |
| `--output-file-path`               |                                | Append result text to this UTF-8 file. Existing content is preserved.                                                                                                                                                                                                 |
| `--output-file-timestamps`         |                                | Prefix every transcription and translation line in the file with its start timestamp: `[start_timestamp] text`. The end timestamp is omitted.                                                                                                                       |
| `--cqhttp-url`                     |                                | If set, will send the result text to this Cqhttp server.                                                                                                                                                                                                              |
| `--cqhttp-token`                   |                                | Token of Cqhttp, if it is not set on the server side, it does not need to fill in.                                                                                                                                                                                    |
| `--discord-webhook-url`            |                                | If set, will send the result text to this Discord channel.                                                                                                                                                                                                            |
| `--telegram-token`                 |                                | Token of Telegram bot.                                                                                                                                                                                                                                                |
| `--telegram-chat-id`               |                                | If set, will send the result text to this Telegram chat. Needs to be used with --telegram-token.                                                                                                                                                                      |
| `--output-proxy`                   |                                | Use the specified HTTP/HTTPS/SOCKS proxy for Cqhttp/Discord/Telegram, e.g. http://127.0.0.1:7890.                                                                                                                                                                     |

## Contact Me

Telegram: [@ionic_bond](https://t.me/ionic_bond)

## Donate

[PayPal Donate](https://www.paypal.com/donate/?hosted_button_id=D5DRBK9BL6DUA) or [PayPal](https://paypal.me/ionicbond3)
