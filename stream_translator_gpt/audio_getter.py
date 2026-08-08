import logging
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading

import ffmpeg
import numpy as np
from scipy.signal import resample

from .common import SAMPLE_RATE, SAMPLES_PER_FRAME, LoopWorkerBase
from .logging_utils import redact_url


logger = logging.getLogger('main')


def _read_ffmpeg_loop(ffmpeg_process, byte_size: int, output_queue, source_name: str = 'ffmpeg'):
    complete_blocks = 0
    while True:
        in_bytes = ffmpeg_process.stdout.read(byte_size)
        if not in_bytes:
            logger.debug('%s audio output reached EOF after %d complete block(s)', source_name, complete_blocks)
            break
        if len(in_bytes) != byte_size:
            logger.warning('%s returned a short audio block; expected %d bytes, received %d bytes; discarding it',
                           source_name, byte_size, len(in_bytes))
            continue
        audio = np.frombuffer(in_bytes, np.float32).flatten()
        output_queue.put(audio)
        complete_blocks += 1
    return complete_blocks


def _wait_for_process(process, process_name: str, timeout: float = 5.0):
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error('%s did not exit within %.1fs; killing it', process_name, timeout)
        process.kill()
        return_code = process.wait()

    if return_code == 0:
        logger.debug('%s exited successfully', process_name)
    else:
        logger.error('%s exited with status %s', process_name, return_code)
    return return_code


def _transport(ytdlp_proc, ffmpeg_proc):
    ffmpeg_closed_early = False
    try:
        while True:
            chunk = ytdlp_proc.stdout.read(1024)
            if not chunk:
                logger.debug('yt-dlp output reached EOF')
                break
            try:
                ffmpeg_proc.stdin.write(chunk)
            except (BrokenPipeError, OSError):
                ffmpeg_closed_early = True
                logger.error('ffmpeg closed its input pipe before yt-dlp finished')
                break
    except Exception:
        ffmpeg_closed_early = True
        logger.exception('Unexpected failure while transporting yt-dlp audio to ffmpeg')
    finally:
        try:
            ffmpeg_proc.stdin.close()
        except (BrokenPipeError, OSError):
            logger.debug('ffmpeg input pipe was already closed')

        if ffmpeg_closed_early and ytdlp_proc.poll() is None:
            ytdlp_proc.terminate()
        _wait_for_process(ytdlp_proc, 'yt-dlp')


def _open_stream(url: str, format: str, cookies: str, proxy: str, cwd: str):
    cmd = [sys.executable, '-m', 'yt_dlp', url, '-f', format, '-o', '-', '-q']
    if cookies:
        cmd.extend(['--cookies', cookies])
    if proxy:
        cmd.extend(['--proxy', proxy])
    ytdlp_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=cwd)

    try:
        ffmpeg_process = (ffmpeg.input('pipe:', loglevel='panic').output('pipe:',
                                                                         format='f32le',
                                                                         acodec='pcm_f32le',
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(pipe_stdin=True,
                                                                                                   pipe_stdout=True))
    except ffmpeg.Error as e:
        raise RuntimeError(f'Failed to load audio: {e.stderr.decode()}') from e

    logger.debug('Started yt-dlp process %d and ffmpeg process %d', ytdlp_process.pid, ffmpeg_process.pid)
    thread = threading.Thread(target=_transport,
                              name='stream-transport',
                              args=(ytdlp_process, ffmpeg_process),
                              daemon=True)
    thread.start()
    return ffmpeg_process, ytdlp_process, thread


class StreamAudioGetter(LoopWorkerBase):

    def __init__(self, url: str, format: str, cookies: str, proxy: str) -> None:
        self.url = url
        self.format = format
        self.cookies = cookies
        self.proxy = proxy
        self.temp_dir = tempfile.mkdtemp()
        self.ffmpeg_process = None
        self.ytdlp_process = None
        self.transport_thread = None
        self.byte_size = round(SAMPLES_PER_FRAME * 4)  # Factor 4 comes from float32 (4 bytes per sample)

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _exit_handler(self, signum, frame):
        logger.info('Stopping stream audio getter after signal %s', signum)
        if self.ffmpeg_process:
            self.ffmpeg_process.kill()
        if self.ytdlp_process:
            self.ytdlp_process.kill()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        logger.warning('Opening stream: %s', redact_url(self.url))
        self.ffmpeg_process, self.ytdlp_process, self.transport_thread = _open_stream(
            self.url, self.format, self.cookies, self.proxy, self.temp_dir)
        _read_ffmpeg_loop(self.ffmpeg_process, self.byte_size, output_queue, source_name='stream ffmpeg')
        _wait_for_process(self.ffmpeg_process, 'ffmpeg')
        self.transport_thread.join(timeout=5)
        if self.transport_thread.is_alive():
            logger.error('Stream transport thread did not exit within 5.0s; killing yt-dlp')
            if self.ytdlp_process.poll() is None:
                self.ytdlp_process.kill()
            self.transport_thread.join(timeout=1)
        logger.info('Stream audio input finished')

        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        output_queue.put(None)


class LocalFileAudioGetter(LoopWorkerBase):

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.ffmpeg_process = None
        self.byte_size = round(SAMPLES_PER_FRAME * 4)  # Factor 4 comes from float32 (4 bytes per sample)

    def _exit_handler(self, signum, frame):
        logger.info('Stopping local file audio getter after signal %s', signum)
        if self.ffmpeg_process:
            self.ffmpeg_process.kill()
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        logger.warning('Opening local file: %s', self.file_path)
        try:
            self.ffmpeg_process = (ffmpeg.input(self.file_path,
                                                loglevel='panic').output('pipe:',
                                                                         format='f32le',
                                                                         acodec='pcm_f32le',
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(pipe_stdin=True,
                                                                                                   pipe_stdout=True))
        except ffmpeg.Error as e:
            raise RuntimeError(f'Failed to load audio: {e.stderr.decode()}') from e

        _read_ffmpeg_loop(self.ffmpeg_process, self.byte_size, output_queue)
        logger.info('Local file audio input finished')

        self.ffmpeg_process.kill()
        output_queue.put(None)


def _sounddevice_gain(volume_factor: float) -> float:
    """Convert the legacy perceptual volume factor to an amplitude gain."""
    if not math.isfinite(volume_factor) or volume_factor < 0:
        raise ValueError('device multiplier must be a finite, non-negative number')
    try:
        gain = 2**((volume_factor**(1 / 8) * 192 - 192) / 6)
    except OverflowError as exc:
        raise ValueError('device multiplier is too large') from exc
    if not math.isfinite(gain):
        raise ValueError('device multiplier produces a non-finite gain')
    return gain


class SoundDeviceAudioGetter(LoopWorkerBase):
    """Compatibility device getter backed by sounddevice's blocking recorder."""

    def __init__(self, device_index: int | None, interval: float = 0.5, multiplier: float = 1.0) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError('sounddevice is not installed. Install it to use the sounddevice audio getter.') from exc

        if interval <= 0:
            raise ValueError('device recording interval must be greater than zero')
        self.sounddevice = sd
        self.device_index = device_index if device_index is not None else sd.default.device[0]
        if self.device_index is None or self.device_index < 0:
            raise RuntimeError('No default sounddevice input device is available.')
        device_info = sd.query_devices(self.device_index)
        if int(device_info.get('max_input_channels', 0)) < 1:
            raise RuntimeError(f'Sounddevice device {self.device_index} does not support audio input.')

        self.device_name = device_info['name']
        self.interval = interval
        self.volume_factor = multiplier
        self.gain = _sounddevice_gain(multiplier)

    def _exit_handler(self, signum, frame):
        logger.info('Stopping sounddevice audio getter after signal %s', signum)
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        logger.warning('Recording sounddevice input: %s (index %s)', self.device_name, self.device_index)
        logger.warning('Recording multiplier: %.6f (volume factor %.3f)', self.gain, self.volume_factor)
        read_size = max(1, round(SAMPLE_RATE * self.interval))
        buffer = np.array([], dtype=np.float32)

        while True:
            audio = self.sounddevice.rec(frames=read_size,
                                         samplerate=SAMPLE_RATE,
                                         channels=1,
                                         dtype=np.float32,
                                         device=self.device_index,
                                         blocking=True)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1) * self.gain
            buffer = np.concatenate((buffer, audio))
            while len(buffer) >= SAMPLES_PER_FRAME:
                output_queue.put(buffer[:SAMPLES_PER_FRAME])
                buffer = buffer[SAMPLES_PER_FRAME:]


class DeviceAudioGetter(LoopWorkerBase):

    def __init__(self, device_index: int, use_mic: bool, interval: float = 0.5) -> None:
        if platform.system() == 'Windows':
            import pyaudiowpatch as pyaudio
        else:
            try:
                import pyaudio
            except ImportError as e:
                raise RuntimeError("PyAudio is not installed. Please install it to use device capture.\n"
                                   "Debian/Ubuntu/Colab: apt install portaudio19-dev && pip install pyaudio") from e

        self.pyaudio = pyaudio.PyAudio()
        self.pyaudio_module = pyaudio
        self.device_index = device_index
        self.use_mic = use_mic
        self.interval = interval
        self.stream = None

        if use_mic:
            if self.device_index is None:
                default_device = self.pyaudio.get_default_input_device_info()
                self.device_index = default_device['index']
        else:
            if platform.system() == 'Windows':
                if self.device_index is None:
                    try:
                        wasapi_info = self.pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
                        default_speakers = self.pyaudio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                        if not default_speakers["isLoopbackDevice"]:
                            for loopback in self.pyaudio.get_loopback_device_info_generator():
                                if default_speakers["name"] in loopback["name"]:
                                    self.device_index = loopback["index"]
                                    break
                            else:
                                self.device_index = self.pyaudio.get_default_wasapi_loopback()['index']
                        else:
                            self.device_index = default_speakers["index"]
                    except (OSError, ValueError):
                        logger.debug('Unable to select the default WASAPI loopback device; falling back to the first '
                                     'available loopback device',
                                     exc_info=True)
                        try:
                            loopback = next(self.pyaudio.get_loopback_device_info_generator())
                            self.device_index = loopback['index']
                        except StopIteration:
                            raise RuntimeError("No loopback device found.")
            else:
                if self.device_index is None:
                    for i in range(self.pyaudio.get_device_count()):
                        info = self.pyaudio.get_device_info_by_index(i)
                        if 'monitor' in info['name'].lower() and info['maxInputChannels'] > 0:
                            self.device_index = info['index']
                            break
                    else:
                        raise RuntimeError("No monitor device found for loopback capture.")

        self.device_name = self.pyaudio.get_device_info_by_index(self.device_index)['name']

    def _exit_handler(self, signum, frame):
        logger.info('Stopping device audio getter after signal %s', signum)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        logger.warning('Recording device: %s (%s)', self.device_name, 'Input' if self.use_mic else 'Output')

        try:
            device_info = self.pyaudio.get_device_info_by_index(self.device_index)
            native_rate = int(device_info['defaultSampleRate'])
            try:
                native_channels = int(device_info['maxInputChannels'])
            except Exception:
                logger.debug('Unable to determine the device channel count; defaulting to one channel', exc_info=True)
                native_channels = 1
            if native_channels < 1:
                native_channels = 2

            read_size = int(native_rate * self.interval)
            logger.debug('Opening audio device index=%s, native_rate=%d, channels=%d, read_size=%d, resample=%s',
                         self.device_index, native_rate, native_channels, read_size, native_rate != SAMPLE_RATE)
            self.stream = self.pyaudio.open(format=self.pyaudio_module.paFloat32,
                                            channels=native_channels,
                                            rate=native_rate,
                                            input=True,
                                            input_device_index=self.device_index,
                                            frames_per_buffer=read_size)
            self.stream.start_stream()
            buffer = np.array([], dtype=np.float32)

            while self.stream.is_active():
                try:
                    in_data = self.stream.read(read_size, exception_on_overflow=False)
                    audio = np.frombuffer(in_data, dtype=np.float32)
                    if native_channels > 1:
                        audio = audio.reshape(-1, native_channels).mean(axis=1)
                    if native_rate != SAMPLE_RATE:
                        target_len = int(len(audio) * SAMPLE_RATE / native_rate)
                        audio = resample(audio, target_len)
                    buffer = np.concatenate((buffer, audio))
                    while len(buffer) >= SAMPLES_PER_FRAME:
                        chunk = buffer[:SAMPLES_PER_FRAME]
                        buffer = buffer[SAMPLES_PER_FRAME:]
                        output_queue.put(chunk)
                except OSError as e:
                    logger.warning('Audio read error: %s', e)
                    continue
        except Exception:
            logger.exception('Audio recording error')
        finally:
            logger.debug('Closing audio device')
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.pyaudio.terminate()
        output_queue.put(None)
