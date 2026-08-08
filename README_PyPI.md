# stream-translator-gpt

Real-time transcription and translation for live streams, local media files, and device audio. Available as a command-line tool and a Gradio WebUI.

## Quick Start on Colab (Recommended)

The easiest way to use this tool, no local environment to set up, and Colab's performance is more than enough for stable everyday use. All you need is your own API key, depending on which services you use:

- [Create a **Google API key**](https://aistudio.google.com/app/apikey) for **Gemini API** translation (recommended, since the **Gemini Flash-Lite** model has a free quota of 15 requests per minute / 500 per day)
- [Create an **OpenAI API key**](https://platform.openai.com/api-keys) for **OpenAI Transcription API** transcription or **GPT API** translation (any **OpenAI-compatible API** can also be used)
- A [**Modulate API key**](https://docs.modulate.ai/guides/authentication) is required to use **Modulate Multilingual Transcription — Streaming**.

|                                                                                        Command Line                                                                                        |                                                                                        WebUI                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/command_line.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/webui.ipynb) |

(Due to frequent scraping and theft of API keys, we are unable to provide a trial API key. You need to fill in your own API key.)

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

**Transcription backends** (default is local **Whisper**):

- ```stream-translator-gpt {URL} --language {input_language}```
- **Faster-Whisper**: ```stream-translator-gpt {URL} --language {input_language} --use-faster-whisper```
- **SimulStreaming**: ```stream-translator-gpt {URL} --language {input_language} --use-simul-streaming```
- **SimulStreaming** with **Faster-Whisper** as the encoder: ```stream-translator-gpt {URL} --language {input_language} --use-simul-streaming --use-faster-whisper```
- **OpenAI Transcription API**: ```stream-translator-gpt {URL} --language {input_language} --use-openai-transcription-api --openai-api-key {your_openai_key}```
- **Modulate Multilingual Transcription — Streaming**: ```stream-translator-gpt {URL} --use-modulate-transcription-api --modulate-api-key {your_modulate_key}```
- **HuggingFace ASR** model (requires `pip install stream-translator-gpt[hf_asr]`; only models with `pipeline_tag: automatic-speech-recognition` on Hugging Face Hub are supported): ```stream-translator-gpt {URL} --model {hf_model_name} --use-hf-asr```

The Modulate backend sends 16 kHz audio directly to the model's streaming mode without Silero VAD. Every completed model utterance becomes a new sentence. Partial results are monitored for long utterances: once `--modulate-partial-cutoff-seconds` has elapsed from the first uncommitted text (or a forced split that already has a remainder), the next full-width or half-width punctuation mark commits the current segment. Silence between sentences doesn't consume the next sentence's timeout. Unexpected transport interruptions are retried with exponential backoff; after reconnecting, the program replays up to 120 seconds of audio received since the most recent final result. Explicit authentication, quota, and API errors are not retried. For a local file, use real-time playback so the timeout follows the media timeline:

```bash
stream-translator-gpt /path/to/file --local-file-mode realtime --use-modulate-transcription-api --modulate-api-key {your_modulate_key}
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
- **.srt subtitle file** (offline generation): ```stream-translator-gpt {URL} --language {input_language} --translation-prompt "Translate from {input_language} to {output_language}" --google-api-key {your_google_key} --no-show-transcribe-result --retry-if-translation-fails --output-timestamps --output-file-path ./result.srt```

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
| `--local-file-mode`                | offline                        | Local file playback mode: `offline` processes as fast as possible; `realtime` emits one second of decoded audio per second of wall-clock time.                                                                                                                         |
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
| `--modulate-reconnect-backoff-seconds` | 1.0                        | Initial reconnect delay in seconds. Each subsequent retry uses exponential backoff; a successful reconnect replays up to 120 seconds of audio since the most recent final result.                                                                                      |
| `--openai-transcription-model`     | gpt-transcribe                 | OpenAI transcription model. `gpt-transcribe` is recommended for ordinary transcription; `whisper-1`, `gpt-4o-mini-transcribe`, and `gpt-4o-transcribe` are also supported.                                                                                            |
| `--use-hf-asr`                     |                                | Use a HuggingFace ASR model. Use `--model` to specify the model ID. Requires `pip install stream-translator-gpt[hf_asr]`.                                                                                                                                             |
| `--transcription-filters`          | emoji_filter,repetition_filter | Filters apply to transcription results, separated by ",". We provide emoji_filter and repetition_filter.                                                                                                                                                              |
| `--no-language-based-filter`       |                                | Disable the currently provided English, Chinese, and Japanese language filters based on ASR language (enabled by default).                                                                                                                                            |
| `--transcription-keywords`         |                                | Comma-separated transcription keywords. `gpt-transcribe` sends them as `keywords`; other Whisper backends use them as an initial prompt. The deprecated `--transcription-initial-prompt` alias is still accepted.                                                     |
| `--transcription-context`          |                                | Pass the previous transcription result as context; only one result is retained. SimulStreaming and HuggingFace ASR do not support text context (disabled by default).                                                                                                 |
| **Translation Options**            |
| `--gpt-model`                      | gpt-5.6-luna                   | OpenAI's GPT model name, gpt-5.4-nano / gpt-5.4-mini / gpt-5.6-luna / gpt-5.6-terra                                                                                                                                                                                   |
| `--openai-translation-api`         | auto                           | OpenAI translation API mode: auto / responses / chat-completions. `auto` uses Responses for OpenAI's endpoint and Chat Completions when `--openai-base-url` is set; explicitly select `responses` for a custom endpoint that implements `/v1/responses`.                |
| `--gemini-model`                   | gemini-3.5-flash-lite          | Google's Gemini model name, gemini-3-flash-preview / gemini-3.1-flash-lite / gemini-3.5-flash / gemini-3.5-flash-lite / gemini-3.6-flash                                                                                                                              |
| `--translation-prompt`             |                                | If set, will translate the result text to target language via GPT / Gemini API (According to which API key is filled in). Example: "Translate from Japanese to Chinese". Adding context (who the streamer is, what the stream is about) improves translation quality. |
| `--translation-history-size`       | 3                              | The number of previous transcripts sent as context when calling the LLM API. It is recommended to disable context (set to 0) for weaker models.                                                                                                                       |
| `--translation-timeout`            | 10                             | If the GPT / Gemini translation exceeds this number of seconds, the translation will be discarded.                                                                                                                                                                    |
| `--use-json-result`                |                                | Using JSON result in LLM translation for some locally deployed models.                                                                                                                                                                                                |
| `--retry-if-translation-fails`     |                                | Retry when translation times out/fails. Used to generate subtitles offline.                                                                                                                                                                                           |
| `--temperature`                    |                                | GPT/Gemini parameter. Controls output randomness, higher values produce more diverse results.                                                                                                                                                                         |
| `--top-p`                          |                                | GPT/Gemini parameter. Nucleus sampling threshold, only tokens with cumulative probability above this value are considered.                                                                                                                                            |
| `--top-k`                          |                                | Gemini parameter. Limits token selection to the top K most probable candidates.                                                                                                                                                                                       |
| `--openai-prompt-cache`            |                                | Opt in to GPT-5.6 explicit prompt caching in Responses mode. The breakpoint follows the complete stable system instructions; dynamic history and the current transcript stay after it. Prefixes shorter than 1,024 tokens are not cached and produce a warning.       |
| `--prompt-cache-key`               |                                | Optional prompt-cache routing key. Without `--openai-prompt-cache` it is still sent for implicit cache routing; with explicit caching it overrides the stable key derived from the model and complete system instructions.                                               |
| `--reasoning-effort`               | auto (lowest supported)        | GPT parameter. Controls reasoning depth for reasoning models. Options: none / minimal / low / medium / high / xhigh / max.                                                                                                                                            |
| `--verbosity`                      |                                | GPT parameter. Controls the verbosity of the response. Options: low / medium / high.                                                                                                                                                                                  |
| `--service-tier`                   |                                | GPT parameter. Specifies processing priority tier. Options: auto / default / flex / priority.                                                                                                                                                                         |
| `--debug-mode`                     |                                | Enable debug mode. Print messages sent to LLM and usage info after each translation call.                                                                                                                                                                             |
| `--processing-proxy`               |                                | Use the specified HTTP/HTTPS/SOCKS proxy for Whisper/GPT API (Gemini currently doesn't support specifying a proxy within the program), e.g. http://127.0.0.1:7890.                                                                                                    |
| **Output Options**                 |
| `--output-timestamps`              |                                | Output the timestamp of the text when outputting the text.                                                                                                                                                                                                            |
| `--no-show-transcribe-result`      |                                | Hide the transcription result (shown by default).                                                                                                                                                                                                                     |
| `--output-file-path`               |                                | If set, will save the result text to this path.                                                                                                                                                                                                                       |
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
