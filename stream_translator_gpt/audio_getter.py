import os
import queue
import signal
import subprocess
import sys
import threading
import logging
import typing
import math
import pydub
import time

import ffmpeg
import numpy as np

from .common import SAMPLE_RATE, LoopWorkerBase


logger = logging.getLogger('main')


def _transport(ytdlp_proc: subprocess.Popen, ffmpeg_proc: subprocess.Popen):
    while (ytdlp_proc.poll() is None) and (ffmpeg_proc.poll() is None):
        try:
            chunk = ytdlp_proc.stdout.read(1024)
            ffmpeg_proc.stdin.write(chunk)
        except (BrokenPipeError, OSError):
            pass
    if ytdlp_proc.poll():
        logger.error('yt-dlp exit with %s', str(ytdlp_proc.poll()))
    if ffmpeg_proc.poll():
        logger.error('ffmpeg exit with %s', str(ffmpeg_proc.poll()))
    ytdlp_proc.kill()
    ffmpeg_proc.kill()


def _open_stream(url: str, format: str, cookies: str, proxy: str) -> typing.Tuple[subprocess.Popen, subprocess.Popen]:
    cmd = ['yt-dlp', url, '-f', format, '-o', '-', '-q']
    if cookies:
        cmd.extend(['--cookies', cookies])
    if proxy:
        cmd.extend(['--proxy', proxy])
    ytdlp_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        ffmpeg_process = (ffmpeg.input('pipe:', loglevel='panic').output('pipe:',
                                                                         format='s16le',
                                                                         acodec='pcm_s16le',
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(pipe_stdin=True,
                                                                                                   pipe_stdout=True))
    except ffmpeg.Error as e:
        raise RuntimeError(f'Failed to load audio: {e.stderr.decode()}') from e
    
    logger.debug('started yt-dlp process %d and ffmpeg process %d', ytdlp_process.pid, ffmpeg_process.pid)

    thread = threading.Thread(target=_transport, name='yt-dlp', args=(ytdlp_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, ytdlp_process


class StreamAudioGetter(LoopWorkerBase):

    def __init__(self, url: str, format: str, cookies: str, proxy: str, frame_duration: float) -> None:
        self._cleanup_ytdlp_cache()

        logger.warning('Opening stream: %s', url)
        self.ffmpeg_process, self.ytdlp_process = _open_stream(url, format, cookies, proxy)
        self.byte_size = round(frame_duration * SAMPLE_RATE *
                               2)  # Factor 2 comes from reading the int16 stream as bytes
        signal.signal(signal.SIGINT, self._exit_handler)

    def __del__(self):
        self._cleanup_ytdlp_cache()

    def _exit_handler(self, signum, frame):
        self.ffmpeg_process.kill()
        if self.ytdlp_process:
            self.ytdlp_process.kill()
        sys.exit(0)

    def _cleanup_ytdlp_cache(self):
        for file in os.listdir('./'):
            if file.startswith('--Frag'):
                os.remove(file)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        while self.ffmpeg_process.poll() is None:
            in_bytes = self.ffmpeg_process.stdout.read(self.byte_size)
            if not in_bytes:
                logger.error('ffmpeg process closed the output pipe')
                break
            if len(in_bytes) != self.byte_size:
                logger.error(f'audio getter received a chunk of wrong size ({self.byte_size} expected, {len(in_bytes)} received): {in_bytes}')
                continue
            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            output_queue.put(audio)

        self.ffmpeg_process.kill()
        if self.ytdlp_process:
            self.ytdlp_process.kill()


class LocalFileAudioGetter(LoopWorkerBase):

    def __init__(self, file_path: str, frame_duration: float) -> None:
        logger.warning('Opening local file: %s', file_path)
        try:
            self.ffmpeg_process = (ffmpeg.input(file_path,
                                                loglevel='panic').output('pipe:',
                                                                         format='s16le',
                                                                         acodec='pcm_s16le',
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(pipe_stdin=True,
                                                                                                   pipe_stdout=True))
        except ffmpeg.Error as e:
            raise RuntimeError(f'Failed to load audio: {e.stderr.decode()}') from e
        self.byte_size = round(frame_duration * SAMPLE_RATE *
                               2)  # Factor 2 comes from reading the int16 stream as bytes
        signal.signal(signal.SIGINT, self._exit_handler)

    def _exit_handler(self, signum, frame):
        self.ffmpeg_process.kill()
        sys.exit(0)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        while self.ffmpeg_process.poll() is None:
            in_bytes = self.ffmpeg_process.stdout.read(self.byte_size)
            if not in_bytes:
                break
            if len(in_bytes) != self.byte_size:
                continue
            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            output_queue.put(audio)

        self.ffmpeg_process.kill()


class DeviceAudioGetter(LoopWorkerBase):

    def __init__(self, device_index: int, frame_duration: float, recording_interval: float, multiplier: float) -> None:
        import sounddevice as sd
        if device_index:
            sd.default.device[0] = device_index
        sd.default.dtype[0] = np.float32
        self.frame_duration = frame_duration
        self.recording_frame_num = max(1, round(recording_interval / frame_duration))
        logger.warning('Recording device: %s', sd.query_devices(sd.default.device[0])['name'])
        self.channels = 1
        self.debug_count = -1
        self.count = 0
        self.audio_clip = np.array([])
        volumeFactor = multiplier
        self.multiplier = pow(2, (math.sqrt(math.sqrt(math.sqrt(volumeFactor))) * 192 - 192)/6)
        logger.warning('Recording multiplier: %f', self.multiplier)


    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        import sounddevice as sd
        while True:
            audio = sd.rec(frames=round(SAMPLE_RATE * self.frame_duration * self.recording_frame_num),
                           samplerate=SAMPLE_RATE,
                           channels=self.channels,
                           blocking=True)
            logger.debug('record audio clip of shape %s, preview %s', str(audio.shape), str(audio))
            audio *= self.multiplier
            flat_audio = audio.flatten()
            if self.debug_count != -1:
                self.count += 1
                self.audio_clip = np.append(self.audio_clip, audio)
                if self.count == self.debug_count:
                    normalized_clip = np.int16(self.audio_clip * 2 ** 15)
                    clip = pydub.AudioSegment(normalized_clip.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=self.channels)
                    clip.export(str(time.time()) + '.mp3', format='mp3', bitrate='320k')
                    logger.warning('export clip.')
                    self.count = 0
                    self.audio_clip = np.array([])
            split_audios = np.array_split(flat_audio, self.recording_frame_num)
            for split_audio in split_audios:
                output_queue.put(split_audio)
