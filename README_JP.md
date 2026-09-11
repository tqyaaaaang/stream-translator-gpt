# stream-translator-gpt

[![PyPI version](https://badge.fury.io/py/stream-translator-gpt.svg)](https://badge.fury.io/py/stream-translator-gpt) [![Python Versions](https://img.shields.io/pypi/pyversions/stream-translator-gpt.svg)](https://pypi.org/project/stream-translator-gpt/) [![Downloads](https://static.pepy.tech/badge/stream-translator-gpt)](https://pepy.tech/project/stream-translator-gpt) [![License](https://img.shields.io/github/license/ionic-bond/stream-translator-gpt.svg)](https://github.com/ionic-bond/stream-translator-gpt/blob/main/LICENSE) [![Gradio](https://img.shields.io/badge/WebUI-Gradio-orange)](https://gradio.app)

[**English**](./README.md) | [**中文**](./README_CN.md) | **日本語**

ライブストリーム・ローカルメディアファイル・デバイス音声のリアルタイム文字起こしと翻訳。コマンドラインツールと Gradio WebUI の 2 つの形態で利用できます。

## Colab でクイックスタート（推奨）

最も簡単な使い方です。ローカル環境の構築は不要で、Colab の性能で日常的に安定して使えます。必要なのは、使うサービスに応じた自分の API キーだけです：

- **Gemini API** で翻訳する場合：[**Google API** キーを作成](https://aistudio.google.com/app/apikey)（おすすめ。Gemini の **Flash-Lite** モデルには毎分 15 回・毎日 500 回の無料枠があります）
- **OpenAI Transcription API** で文字起こし、または **GPT API** で翻訳する場合：[**OpenAI API** キーを作成](https://platform.openai.com/api-keys)（**OpenAI 互換の API** も利用できます）
- **Modulate Multilingual Transcription — Streaming** を使うには [**Modulate API キー**](https://docs.modulate.ai/guides/authentication)が必要です。

|                                                                                        コマンドライン                                                                                         |                                                                                         WebUI                                                                                          |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/command_line_JP.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ionic-bond/stream-translator-gpt/blob/main/colab/webui_JP.ipynb) |

> [!NOTE]
> API キーのスクレイピングと盗用が頻発しているため、試用 API キーは提供できません。ご自身の API キーをご記入ください。

## 仕組み

```mermaid
flowchart LR
    subgraph ga["`**入力**`"]
        direction LR
        aa("`**FFmpeg**`")
        ab("`**デバイス音声**`")
        ac("`**yt-dlp**`")
        ad("`**ローカルメディアファイル**`")
        ae("`**ライブストリーム**`")
        ac --> aa
        ad --> aa
        ae --> ac
    end
    subgraph gb["`**音声スライシング**`"]
        direction LR
        ba("`**Silero VAD**`")
    end
    subgraph gc["`**文字起こし**`"]
        direction LR
        ca("`**Whisper**`")
        cb("`**Faster-Whisper**`")
        cc("`**SimulStreaming**`")
        cd("`**OpenAI Transcription API**`")
        ce("`**HuggingFace ASR**`")
        cf("`**Modulate Streaming**`")
    end
    subgraph gd["`**翻訳**`"]
        direction LR
        da("`**GPT API**`")
        db("`**Gemini API**`")
    end
    subgraph ge["`**出力**`"]
        direction LR
        ea("`**ターミナルに出力**`")
        ee("`**ファイルに保存**`")
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

- **入力**：[**yt-dlp**](https://github.com/yt-dlp/yt-dlp) がライブストリームから音声を抽出。ローカルメディアファイルや PC のデバイス音声にも対応。
- **音声スライシング**：[**Silero-VAD**](https://github.com/snakers4/silero-vad) に基づく動的しきい値スライシング。Modulate ストリーミングバックエンドは 16 kHz 音声ストリームを直接処理し、VAD をバイパスします。
- **文字起こし**：ローカルで [**Whisper**](https://github.com/openai/whisper) / [**Faster-Whisper**](https://github.com/SYSTRAN/faster-whisper) / [**SimulStreaming**](https://github.com/ufal/SimulStreaming) / [**HuggingFace ASR**](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)、またはリモートで [**OpenAI Transcription API**](https://platform.openai.com/docs/guides/speech-to-text) / [**Modulate Multilingual Transcription — Streaming**](https://docs.modulate.ai/api-reference/stt/streaming) を利用。
- **翻訳**（オプション）：OpenAI の [**GPT API**](https://platform.openai.com/docs/overview) または Google の [**Gemini API**](https://ai.google.dev/gemini-api/docs)。
- **出力**：**ターミナル**に出力、UTF-8 **テキストファイル**に追記、または **Discord** / **Telegram** / **Cqhttp** に送信。

## ローカルインストール（上級者向け）

ローカルでの実行には Python 環境の経験がある程度必要です（特に Windows）。実質的に Colab と同様の環境を自分で構築することになります。迷ったら Colab を使ってください。

1. **Python** >= 3.10
2. **FFmpeg**（インストール済みならスキップ）：
   - Windows: `winget install ffmpeg`
   - Linux (Debian/Ubuntu): `sudo apt install ffmpeg`
3. **ローカル文字起こし**（Whisper / Faster-Whisper / SimulStreaming / HuggingFace ASR）を使う場合のみ — OpenAI Transcription API または Modulate Streaming だけを使うなら不要：
   - [システムに **CUDA** をインストール](https://developer.nvidia.com/cuda-downloads)
   - [Python に **PyTorch**（CUDA 版）をインストール](https://pytorch.org/get-started/locally/)
   - **Faster-Whisper** を使う場合は [**cuDNN** を CUDA ディレクトリにインストール](https://developer.nvidia.com/cudnn-downloads)

その後、パッケージをインストールします：

```
pip install stream-translator-gpt -U
```

WebUI も含める場合：

```
pip install stream-translator-gpt[webui] -U
```

## 使い方

### WebUI

```
stream-translator-gpt-webui
```

起動時に `--ui-language` を使って WebUI の表示言語を `en`、`zh`、`ja` から指定できます。

表示されたローカル URL をブラウザで開いてください。CLI の全機能を画面上で利用でき、設定はプリセットとして保存できます。

### コマンドライン

```
stream-translator-gpt URL [OPTIONS]
```

**文字起こしバックエンド**（デフォルトはローカル **Whisper**）：

- ```stream-translator-gpt {URL} --language {入力言語}```
- **Faster-Whisper**: ```stream-translator-gpt {URL} --language {入力言語} --use-faster-whisper```
- **SimulStreaming**: ```stream-translator-gpt {URL} --language {入力言語} --use-simul-streaming```
- **Faster-Whisper** をエンコーダーとする **SimulStreaming**: ```stream-translator-gpt {URL} --language {入力言語} --use-simul-streaming --use-faster-whisper```
- **OpenAI Transcription API**: ```stream-translator-gpt {URL} --language {入力言語} --use-openai-transcription-api --openai-api-key {OpenAI キー}```
- **Modulate Multilingual Transcription — Streaming**: ```stream-translator-gpt {URL} --use-modulate-transcription-api --modulate-api-key {Modulate キー}```
- **HuggingFace ASR** モデル（`pip install stream-translator-gpt[hf_asr]` が必要。Hugging Face Hub で `pipeline_tag` が `automatic-speech-recognition` のモデルのみ対応）: ```stream-translator-gpt {URL} --model {HF モデル名} --use-hf-asr```

Modulate バックエンドは Silero VAD を使わず、16 kHz 音声をモデルのストリーミングモードへ直接送信します。モデルが完了した各 utterance は新しい文になります。partial result を継続的に監視し、現在の未確定文の最初の文字（または分割時に既に存在する後続テキスト）から `--modulate-partial-cutoff-seconds` を超えると、その後に現れる最初の全角または半角の句読点で現在の区間を確定します。文と文の間の無音時間は、次の文の待機時間には含まれません。URL のライブストリームでは、ライブエッジにいる間は通常ソースの速度で Modulate へ送信し、再接続時のリプレイデータまたはライブ入力キューに滞留がある間は、追いつくまで最大 1.5 倍速で送信します。予期しない通信切断時は指数バックオフで再接続し、成功後に直近の final 結果以降に受信した音声を最大 10 分間リプレイします。明示的な認証、クォータ、および API エラーは再試行しません。ローカルファイルでは、タイムアウト判定をメディアの時間軸に合わせるため、リアルタイム再生を使用してください：

```bash
stream-translator-gpt {ファイルパス} --pacer realtime --use-modulate-transcription-api --modulate-api-key {Modulate キー}
```

**翻訳**（`--translation-prompt` を設定すると有効になり、記入した API キーに応じてプロバイダが選ばれます）：

- **Gemini** で翻訳: ```stream-translator-gpt {URL} --language {入力言語} --translation-prompt "{入力言語}から{出力言語}に翻訳" --google-api-key {Google キー}```
- **GPT** で翻訳: ```stream-translator-gpt {URL} --language {入力言語} --translation-prompt "{入力言語}から{出力言語}に翻訳" --openai-api-key {OpenAI キー}```
- **OpenAI Transcription API** と **Gemini** を同時に使用: ```stream-translator-gpt {URL} --language {入力言語} --use-openai-transcription-api --openai-api-key {OpenAI キー} --translation-prompt "{入力言語}から{出力言語}に翻訳" --google-api-key {Google キー}```

> [!TIP]
> 翻訳プロンプトはそのまま LLM に渡されるため、言語ペア以外の指示も書けます。API の利用枠に余裕があれば、配信者が誰か・何の配信かといった背景情報や用語の指定を加えると、翻訳の精度が上がり、文脈による音声認識の誤字修正もより確実になります。

**入力ソース**（ストリーム URL 以外）：

- **ローカル動画/音声ファイル**: ```stream-translator-gpt {ファイルパス} --language {入力言語}```
- **システム音声**（ループバック）: ```stream-translator-gpt device --language {入力言語}```
- **マイク**: ```stream-translator-gpt device --mic --language {入力言語}```

**出力先**（ターミナル以外）：

- **Discord**: ```stream-translator-gpt {URL} --language {入力言語} --discord-webhook-url {Discord webhook URL}```
- **Telegram**: ```stream-translator-gpt {URL} --language {入力言語} --telegram-token {Telegram トークン} --telegram-chat-id {Telegram チャット ID}```
- **Cqhttp**: ```stream-translator-gpt {URL} --language {入力言語} --cqhttp-url {Cqhttp URL} --cqhttp-token {Cqhttp トークン}```
- **タイムスタンプ付きテキストファイル**: ```stream-translator-gpt {URL} --language {入力言語} --translation-prompt "{入力言語}から{出力言語}に翻訳" --google-api-key {Google キー} --no-show-transcribe-result --retry-if-translation-fails --output-file-path ./result.txt --output-file-timestamps```

### 翻訳モジュールの責務

デフォルト実装は、ライフサイクル（`translation_pipeline.py`）、スケジューリングと順序付き出力
（`translation_scheduler.py`）、ワーカーループ（`translation_worker.py`）、API 呼び出し
（`translation_backends.py`）、ワーカー専用クライアント（`translation_clients.py`）、プロンプトと結果解析
（`translation_protocol.py`）を分離しています。`translation_types.py` の不変リクエスト・結果で通信し、
パイプラインタスクを更新するのはスケジューラだけです。ワーカーは context や出力順、再試行の待機を行いません。

`--use-old-translator` は `ab314cf` の実装を無変更で保存した `legacy_translator.py` を直接使用し、
新しい翻訳モジュールを読み込みません。従来のクライアントプール、再試行、タイムアウト、終了時の制約も維持します。
context モードは逐次処理、それ以外は最大 10 リクエストです。`--translation-threads` は新実装だけに適用されます。

## すべてのオプション

> [!NOTE]
> すべてのオプションはハイフンとアンダースコアのどちらでも指定できます（`--openai-api-key` = `--openai_api_key`）。
> すべてのブールオプションには、無効化用の `--no-*` 形式があります（例：デフォルトで有効な `--dynamic-vad-threshold` は `--no-dynamic-vad-threshold` で無効化できます）。

| オプション                         | デフォルト値                   | 説明                                                                                                                                                                                                                       |
| :--------------------------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `URL`                              |                                | ストリームの URL。ローカルファイルパスを入力すると、そのファイルが入力として使用されます。「device」と入力すると、PC デバイスから入力を取得します。                                                                        |
| **全般オプション**                 |
| `--openai-api-key`                 |                                | GPT 翻訳用の OpenAI API キー。`--whisper-api-key` 未指定時はリモート文字起こしにも使用されます。複数のキーは "," 区切りで順番に使用されます。                                                                           |
| `--whisper-api-key`                |                                | Whisper 互換文字起こしエンドポイント用の API キー。リモート文字起こしでは `--openai-api-key` より優先されます。                                                                                                         |
| `--modulate-api-key`               |                                | Modulate Multilingual Transcription — Streaming 用の API キー。`--use-modulate-transcription-api` を有効にする場合は必須です。                                                                                          |
| `--google-api-key`                 |                                | Gemini 翻訳を使う場合の Google API キー。複数ある場合は "," で区切ると順番に使用されます。                                                                                                                                 |
| `--openai-base-url`                |                                | GPT 翻訳用の OpenAI API エンドポイント。`--whisper-base-url` 未指定時はリモート文字起こしにも使用されます。                                                                                                                               |
| `--whisper-base-url`               |                                | Whisper 互換文字起こし API が使用するエンドポイントをカスタマイズ。                                                                                                                                                         |
| `--google-base-url`                |                                | Google の API エンドポイントをカスタマイズ（Gemini 翻訳に影響）。                                                                                                                                                          |
| `--no-verify-ssl`                  |                                | OpenAI / Google API と HuggingFace ダウンロードの TLS 証明書検証を無効化。API エンドポイントやプロキシが自己署名・無効な証明書を使う場合に。base URL のホストが IP アドレスの場合は自動的に無効化されます。                |
| `--proxy`                          |                                | 個別に設定されていないすべての --*-proxy オプションにプロキシを設定。http_proxy 環境変数も設定されます。                                                                                                                   |
| **入力オプション**                 |
| `--format`                         | ba/wa*                         | ストリームのフォーマットコード。yt-dlp にそのまま渡されます。`yt-dlp {url} -F` で利用可能なフォーマットの一覧を取得できます。                                                                                              |
| `--list-format`                    |                                | 利用可能なフォーマットを一覧表示して終了。                                                                                                                                                                                 |
| `--cookies`                        |                                | メンバー限定配信を開くために使用。yt-dlp にそのまま渡されます。                                                                                                                                                            |
| `--input-proxy`                    |                                | yt-dlp に HTTP/HTTPS/SOCKS プロキシを指定。例：http://127.0.0.1:7890。                                                                                                                                                     |
| `--wait-for-video [秒]`            | 無効；値なしでは 60            | 予約配信が利用可能になるまで待機します。この値は yt-dlp の利用可能性チェック間の最小間隔で、開始時刻が既知ならその時刻まで直接待機する場合があります。未指定では待機せず、値なしでは 60 秒です。                         |
| `--live-from-start`                | False                          | 対応するライブ配信を現在位置ではなく先頭からダウンロードするよう yt-dlp に要求します。この yt-dlp 機能は実験的で、対応する extractor でのみ利用できます。                                      |
| `--stream-format-diagnostic-file PATH` | 無効                      | 選択したストリーム形式の許可リスト方式メタデータをユーザー指定パスへ書き込み、ストリームごとにそのファイルを置き換えます。明示指定時のみ有効で、未指定時に暗黙の診断ファイルは作成されません。          |
| `--pacer`                          | unlimited                      | URL ストリームとローカルファイルの入力ペーシング。`unlimited` は下流の処理能力に応じて可能な限り高速に読み込み、`realtime` は実時間1秒ごとにデコード音声1秒分を出力します。デバイス入力は元からリアルタイムなので、非デフォルト値は警告後に無視されます。 |
| `--device-index`                   |                                | 録音するデバイスの番号。未設定の場合はシステムデフォルトの録音デバイスを使用。                                                                                                                                             |
| `--list-devices`                   |                                | すべてのオーディオデバイス情報を表示して終了。                                                                                                                                                                             |
| `--device-recording-interval`      | 0.5                            | 録音間隔が短いほど遅延は低くなりますが、CPU 使用率が上がります。0.1〜1.0 の間を推奨。                                                                                                                                      |
| `--device-audio-getter`            | pyaudio                        | `pyaudio` または手動の `sounddevice` フォールバックを選択。SoundDevice は入力可能なデバイスを取得するため、システム音声には `--device-index` で monitor / stereo-mix / 仮想入力を指定します。                 |
| `--device-multiplier`              | 1.0                            | 両方のデバイス音声ゲッターで使える従来の知覚音量係数です。1 で元の振幅を維持します。                                                                                                                                    |
| `--device-audio-debug-dir`          |                                | 処理後のデバイス入力を、数を制限したモノラル 16 kHz PCM WAV クリップとしてこのディレクトリに保存します。デフォルトは無効です。                                                                                          |
| `--device-audio-debug-clip-duration` | 10.0                           | 各デバイスデバッグ WAV クリップの長さ（秒）。                                                                                                                                                                          |
| `--device-audio-debug-max-clips`    | 10                             | 1 回の実行で保存するデバイスデバッグ WAV クリップの最大数。                                                                                                                                                             |
| `--mic`                            |                                | システム音声（ループバック）の代わりにマイクを使用。                                                                                                                                                                       |
| **音声スライシングオプション**     |
| `--min-audio-length`               | 0.5                            | 音声スライスの最小長（秒）。                                                                                                                                                                                               |
| `--max-audio-length`               | 30.0                           | 音声スライスの最大長（秒）。                                                                                                                                                                                               |
| `--target-audio-length`            | 5.0                            | 動的無音しきい値が有効な場合（デフォルトで有効）、この長さに近くなるように音声をスライスします。                                                                                                                           |
| `--continuous-no-speech-threshold` | 1.0                            | この秒数の間無音が続くとスライスします。動的無音しきい値が有効な場合（デフォルトで有効）、実際のしきい値はこの値を基に動的に調整されます。                                                                                 |
| `--no-dynamic-no-speech-threshold` |                                | 動的無音しきい値を無効化（デフォルトで有効）。                                                                                                                                                                             |
| `--prefix-retention-length`        | 0.5                            | スライス時に保持するプレフィックス音声の長さ。                                                                                                                                                                             |
| `--vad-threshold`                  | 0.35                           | 範囲 0〜1。値が高いほど音声判定が厳しくなります。動的 VAD しきい値が有効な場合（デフォルトで有効）、しきい値はこの値を基に動的に調整されます。                                                                             |
| `--no-dynamic-vad-threshold`       |                                | 動的 VAD しきい値を無効化（デフォルトで有効）。                                                                                                                                                                            |
| **文字起こしオプション**           |
| `--model`                          | turbo                          | Whisper/Faster-Whisper/SimulStreaming のモデルサイズを選択。利用可能なモデルは[こちら](https://github.com/openai/whisper#available-models-and-languages)を参照。                                                           |
| `--language`                       | auto                           | ストリームで話されている言語。デフォルトは自動検出。利用可能な言語は[こちら](https://github.com/openai/whisper#available-models-and-languages)を参照。                                                                     |
| `--use-faster-whisper`             |                                | Whisper の代わりに Faster-Whisper を使用。--use-simul-streaming と併用すると、Faster-Whisper をエンコーダーとした SimulStreaming が使用されます。                                                                          |
| `--use-simul-streaming`            |                                | Whisper の代わりに SimulStreaming を使用。--use-faster-whisper と併用すると、Faster-Whisper をエンコーダーとした SimulStreaming が使用されます。                                                                           |
| `--use-openai-transcription-api`   |                                | ローカル Whisper の代わりに OpenAI Transcription API を使用。                                                                                                                                                              |
| `--use-modulate-transcription-api` |                                | Modulate Multilingual Transcription — Streaming を使用し、16 kHz 音声を直接ストリーミングして Silero VAD をバイパスします。                                                                                               |
| `--modulate-partial-cutoff-seconds` | 10.0                           | 現在の未確定文の最初の文字からこの秒数を超えると、その後の最初の全角または半角の句読点で確定します。それ以前の無音時間は含まれません。                                                                      |
| `--modulate-reconnect-attempts`    | 3                              | Modulate の予期しない通信切断後に行う再接続の最大試行回数です。明示的な認証、クォータ、および API エラーは再試行しません。                                                                                                 |
| `--modulate-reconnect-backoff-seconds` | 1.0                        | 最初の再接続までの待機秒数です。以降は指数バックオフを使用し、再接続後は直近の final 結果以降の音声を最大 10 分間リプレイします。                                                                                           |
| `--modulate-websocket-compression` | False                          | Modulate の WebSocket per-message deflate を有効にします。圧縮はネットワーク帯域を削減できますが CPU 負荷が増えます。デフォルトは非圧縮で、CPU リソースが限られネットワーク条件が良い環境を優先します。                        |
| `--openai-transcription-model`     | gpt-transcribe                 | OpenAI 文字起こしモデル。通常の文字起こしには `gpt-transcribe` を推奨。`whisper-1`、`gpt-4o-mini-transcribe`、`gpt-4o-transcribe` も使用できます。                                                                         |
| `--use-hf-asr`                     |                                | HuggingFace ASR モデルを使用。`--model` でモデル ID を指定。`pip install stream-translator-gpt[hf_asr]` が必要。                                                                                                           |
| `--transcription-filters`          | emoji_filter,repetition_filter | 文字起こし結果に適用するフィルター（"," 区切り）。emoji_filter と repetition_filter を提供しています。                                                                                                                     |
| `--language-filters`               |                                | 有効にする言語別文字起こしフィルターをカンマ区切りで指定します。`en`/`english`、`ja`/`japanese`、`zh`/`chinese`（`zh-cn`、`zh-tw` を含む）に対応し、省略時は無効です。                                                       |
| `--transcription-keywords`         |                                | カンマ区切りの文字起こしキーワード。`gpt-transcribe` では `keywords`、その他の Whisper バックエンドでは初期プロンプトとして使用。旧 `--transcription-initial-prompt` は非推奨エイリアスとして利用可能。                    |
| `--transcription-context`          |                                | 直前の文字起こし結果をコンテキストとして渡します。保持するのは 1 件のみで、SimulStreaming と HuggingFace ASR はテキストコンテキストに非対応です（デフォルト無効）。                                                        |
| **翻訳オプション**                 |
| `--gpt-model`                      | gpt-5.6-luna                   | OpenAI の GPT モデル名。gpt-5.4-nano / gpt-5.4-mini / gpt-5.6-luna / gpt-5.6-terra                                                                                                                                         |
| `--openai-translation-api`         | auto                           | OpenAI 翻訳 API モード：auto / responses / chat-completions。`auto` は OpenAI 公式エンドポイントでは Responses、`--openai-base-url` 設定時は Chat Completions を使用します。`/v1/responses` を実装するカスタムエンドポイントでは `responses` を明示指定してください。 |
| `--gemini-model`                   | gemini-3.5-flash-lite          | Google の Gemini モデル名。gemini-3-flash-preview / gemini-3.1-flash-lite / gemini-3.5-flash / gemini-3.5-flash-lite / gemini-3.6-flash                                                                                    |
| `--translation-prompt`             |                                | 設定すると、GPT / Gemini API で結果テキストをターゲット言語に翻訳します（記入した API キーに応じて選択）。例：「日本語から中国語に翻訳」。プロンプトに背景情報（配信者が誰か、何の配信か）を加えると翻訳品質が向上します。出力形式はプログラム内部で制御されるため、このプロンプトには翻訳内容、用語、背景、スタイルのみを指定してください。 |
| `--translation-history-size`       | 3                              | 各リクエストの `Reference` に含める直前の生の文字起こしの数（0 以上）。ローリングコンテキスト有効時に 0 を指定すると、通常は逐次処理になります。 |
| `--translation-threads`            | 3                              | 翻訳リクエストの最大同時実行数（1 以上）。空いているスレッドへ動的に割り当て、出力は原文順を維持します。コンテキスト有効時に Reference の窓幅 + 1 を超えると警告しますが、設定は有効です。 |
| `--use-old-translator`             |                                | 緊急時に無変更で保存された旧翻訳実装へ戻し、非推奨警告を出します。旧実装は `--translation-threads` を無視し、コンテキスト有効時には逐次処理します。 |
| `--use-translation-context`        |                                | 長期的な話題・活動・登場主体・未解決の状態を 100 Unicode 文字以内の内部要約として保持します。並列リクエストは原文順で最新の利用可能な `Context` と直近の生の `Reference` を使います。Reference が欠落区間を覆えなければ待機し、処理中または再試行待ちのタスクでも補えない場合は最新の Context で続行して INFO を記録します。Context 自体は出力されません。 |
| `--translation-timeout`            | 10                             | 各文の最初の実行開始からの総制限時間（秒）。再試行と待機を含み、期限切れの結果は破棄します。0 は総期限を無効にしますが、個々のネットワーク呼び出しにはタイムアウトがあります。 |
| `--use-json-result`                |                                | LLM 翻訳で JSON 結果を使用。一部のローカルデプロイモデル向け。                                                                                                                                                             |
| `--retry-if-translation-fails`     |                                | 元の Reference/Context を保持し、総期限内で失敗したリクエストを再試行します。`--translation-timeout 0` と併用すると、停止するまで再試行し続ける場合があります。 |
| `--temperature`                    |                                | GPT/Gemini パラメータ。出力のランダム性を制御し、高いほど多様な結果になります。                                                                                                                                            |
| `--top-p`                          |                                | GPT/Gemini パラメータ。Nucleus サンプリングのしきい値。累積確率がこの値を超えるトークンのみが考慮されます。                                                                                                                |
| `--top-k`                          |                                | Gemini パラメータ。トークン選択を確率上位 K 個の候補に制限します。                                                                                                                                                         |
| `--openai-prompt-cache`            |                                | Responses モードで GPT-5.6 の明示的プロンプトキャッシュを有効にします。完全で安定したシステム指示の直後にブレークポイントを置き、動的な履歴と現在の文字起こしはその後に配置します。1,024 tokens 未満のプレフィックスはキャッシュされず、warning が出力されます。 |
| `--prompt-cache-key`               |                                | 任意のプロンプトキャッシュ用ルーティングキー。`--openai-prompt-cache` が無効でも暗黙キャッシュのルーティング用に送信され、有効時はモデルと完全なシステム指示から生成される安定キーを上書きします。        |
| `--reasoning-effort`               | 自動（最低対応レベル）         | GPT パラメータ。推論モデルの推論深度を制御。オプション：none / minimal / low / medium / high / xhigh / max。                                                                                                               |
| `--verbosity`                      |                                | GPT パラメータ。レスポンスの詳細度を制御。オプション：low / medium / high。                                                                                                                                                |
| `--service-tier`                   |                                | GPT パラメータ。処理の優先度を指定。オプション：auto / default / flex / priority。                                                                                                                                         |
| `--debug-mode`                     |                                | デバッグモードを有効化し、翻訳呼び出しごとに使用量情報を記録。完全な LLM プロンプトは `-vvv` も同時に有効な場合のみ記録。                                                                                                                    |
| `--processing-proxy`               |                                | Whisper/GPT API に HTTP/HTTPS/SOCKS プロキシを指定（Gemini は現在プログラム内でのプロキシ指定に非対応）。例：http://127.0.0.1:7890。                                                                                       |
| **出力オプション**                 |
| `--output-timestamps`              |                                | コンソールおよびファイル以外の出力先にタイムスタンプを付加します。出力ファイルには影響しません。                                                                                                                                                   |
| `--no-show-transcribe-result`      |                                | 文字起こし結果を非表示にする（デフォルトで表示）。                                                                                                                                                                         |
| `--output-file-path`               |                                | 結果テキストを UTF-8 でこのファイルに追記し、既存の内容を保持します。                                                                                                                                                             |
| `--output-file-timestamps`         |                                | ファイル内の文字起こし行と翻訳行のそれぞれに、`[開始タイムスタンプ] テキスト` の形式で開始時刻を付けます。終了時刻は出力しません。                                                                                                           |
| `--cqhttp-url`                     |                                | 設定すると、結果テキストをこの Cqhttp サーバーに送信します。                                                                                                                                                               |
| `--cqhttp-token`                   |                                | Cqhttp のトークン。サーバー側で設定されていなければ記入不要。                                                                                                                                                              |
| `--discord-webhook-url`            |                                | 設定すると、結果テキストをこの Discord チャンネルに送信します。                                                                                                                                                            |
| `--telegram-token`                 |                                | Telegram ボットのトークン。                                                                                                                                                                                                |
| `--telegram-chat-id`               |                                | 設定すると、結果テキストをこの Telegram チャットに送信します。--telegram-token と併用してください。                                                                                                                        |
| `--output-proxy`                   |                                | Cqhttp/Discord/Telegram に HTTP/HTTPS/SOCKS プロキシを指定。例：http://127.0.0.1:7890。                                                                                                                                    |

## 連絡先

Telegram: [@ionic_bond](https://t.me/ionic_bond)

## 寄付

[PayPal で寄付](https://www.paypal.com/donate/?hosted_button_id=D5DRBK9BL6DUA) または [PayPal](https://paypal.me/ionicbond3)
