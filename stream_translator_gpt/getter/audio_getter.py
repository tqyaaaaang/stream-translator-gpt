import logging
import math
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import ffmpeg
import numpy as np
from scipy.signal import resample

from .time_range import ffmpeg_time_range_options, format_seconds, is_local_hls, normalize_time_range
from ..utils.common import SAMPLE_RATE, SAMPLES_PER_FRAME, LoopWorkerBase
from ..utils.logging_utils import redact_url


logger = logging.getLogger('main')
_stream_versions_lock = threading.Lock()
_stream_versions_logged = False
_STREAM_FORMAT_TEMPLATE = ('format_id=%(format_id)s protocol=%(protocol)s ext=%(ext)s acodec=%(acodec)s '
                           'vcodec=%(vcodec)s live_status=%(live_status)s')


def _signal_exit_code(signum: int) -> int:
    """Keep interactive Ctrl+C compatible while preserving SIGTERM failure semantics."""
    return 0 if signum == signal.SIGINT else 128 + int(signum)


def _signal_cleanup_timeout(signum: int) -> float:
    """Allow service-manager termination more cleanup time than interactive Ctrl+C."""
    return 10.0 if hasattr(signal, 'SIGTERM') and signum == signal.SIGTERM else 2.0


@dataclass
class _StreamTransportStats:
    """Small, content-free diagnostics shared with the transport thread."""

    started_at: float
    first_media_elapsed: float | None = None
    media_bytes: int = 0
    media_chunks: int = 0
    ytdlp_return_code: int | None = None
    transport_error: str | None = None
    format_diagnostic_path: Path | None = None
    format_diagnostic_logged: bool = False


def _log_stream_tool_versions():
    """Log local tool versions once without adding network activity."""
    global _stream_versions_logged
    if not logger.hasHandlers() or not logger.isEnabledFor(logging.DEBUG) or _stream_versions_logged:
        return
    with _stream_versions_lock:
        if _stream_versions_logged:
            return
        _stream_versions_logged = True

        try:
            ytdlp_version = version('yt-dlp')
        except Exception as exc:
            # Broken package metadata must never prevent stream startup.
            ytdlp_version = ('unknown' if isinstance(exc, PackageNotFoundError)
                              else f'unavailable ({type(exc).__name__})')

        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True,
                                    text=True,
                                    encoding='utf-8',
                                    errors='replace',
                                    timeout=5,
                                    check=False)
            ffmpeg_version = next((line.strip() for line in result.stdout.splitlines() if line.strip()), 'unknown')
        except (OSError, subprocess.SubprocessError) as exc:
            ffmpeg_version = f'unavailable ({type(exc).__name__})'
        logger.debug('Stream tools: yt-dlp %s; %s', ytdlp_version, ffmpeg_version)


def _log_stream_format_diagnostic(stats: _StreamTransportStats):
    """Log yt-dlp's selected non-sensitive format fields at most once."""
    if (not logger.isEnabledFor(logging.DEBUG) or stats.format_diagnostic_logged
            or stats.format_diagnostic_path is None):
        return
    try:
        lines = stats.format_diagnostic_path.read_text(encoding='utf-8', errors='replace').splitlines()
    except (OSError, ValueError):
        return
    selected_format = next((line.strip() for line in reversed(lines) if line.strip()), '')
    if not selected_format:
        return
    stats.format_diagnostic_logged = True
    logger.debug('yt-dlp selected stream format: %s', selected_format[:500])


class _RealtimeAudioPacer:
    """Pace decoded samples against an absolute monotonic media timeline."""

    def __init__(self,
                 stop_event: threading.Event,
                 clock=None,
                 max_lag: float = 0.1,
                 source_description: str = 'local file playback') -> None:
        self.stop_event = stop_event
        self.clock = clock or time.monotonic
        self.max_lag = max_lag
        self.source_description = source_description
        self.started_at = self.clock()
        self.samples_emitted = 0

    def wait_for_block(self, sample_count: int) -> bool:
        if self.stop_event.is_set():
            return False
        now = self.clock()
        next_sample_count = self.samples_emitted + sample_count
        deadline = self.started_at + next_sample_count / SAMPLE_RATE
        lag = now - deadline
        if lag > self.max_lag:
            self.started_at += lag
            deadline = now
            logger.info('Realtime %s fell %.3fs behind; resuming without a catch-up burst',
                           self.source_description, lag)
        while (delay := deadline - self.clock()) > 0:
            if self.stop_event.wait(delay):
                return False
        if self.stop_event.is_set():
            return False
        self.samples_emitted = next_sample_count
        return True


def _put_audio_block(output_queue, audio: np.ndarray, stop_event: threading.Event | None = None) -> bool:
    """Put one audio block while allowing cancellation to break bounded-queue backpressure."""
    if stop_event is None:
        output_queue.put(audio)
        return True
    while not stop_event.is_set():
        try:
            output_queue.put(audio, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _signal_end_of_stream(output_queue,
                          *,
                          discard_buffered: bool = False,
                          stop_event: threading.Event | None = None) -> None:
    """Publish one terminal sentinel without blocking a cancelled bounded producer."""
    while True:
        if discard_buffered or (stop_event is not None and stop_event.is_set()):
            # Recheck cancellation even if natural EOF was already waiting for
            # queue space when the downstream consumer stopped. A one-time
            # check followed by blocking put(None) can deadlock on shutdown.
            while True:
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                output_queue.put_nowait(None)
                return
            except queue.Full:
                continue
        try:
            if stop_event is None:
                output_queue.put(None)
            else:
                output_queue.put(None, timeout=0.1)
            return
        except queue.Full:
            continue


def _wake_recording_queue(recording_queue, *, discard_buffered: bool) -> None:
    """Wake a callback consumer without blocking the callback or cancellation caller."""
    if discard_buffered:
        while True:
            try:
                recording_queue.get_nowait()
            except queue.Empty:
                break
    while True:
        try:
            recording_queue.put_nowait(None)
            return
        except queue.Full:
            # A natural callback EOF preserves all queued audio when capacity
            # permits. If the bounded queue is full, match the legacy behavior
            # and sacrifice only the oldest block to guarantee room for EOF.
            try:
                recording_queue.get_nowait()
            except queue.Empty:
                continue


def _read_ffmpeg_loop(ffmpeg_process,
                      byte_size: int,
                      output_queue,
                      source_name: str = 'ffmpeg',
                      pacer: _RealtimeAudioPacer | None = None,
                      pad_final_block: bool = False,
                      started_at: float | None = None,
                      stop_event: threading.Event | None = None):
    complete_blocks = 0
    while True:
        chunks = []
        remaining = byte_size
        reached_eof = False
        while remaining:
            chunk = ffmpeg_process.stdout.read(remaining)
            if not chunk:
                reached_eof = True
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        in_bytes = b''.join(chunks)
        if not in_bytes:
            logger.debug('%s audio output reached EOF after %d complete block(s)', source_name, complete_blocks)
            break
        valid_sample_count = len(in_bytes) // np.dtype(np.float32).itemsize
        if len(in_bytes) != byte_size and not (pad_final_block and len(in_bytes) % 4 == 0):
            logger.warning('%s returned a short audio block; expected %d bytes, received %d bytes; discarding it',
                           source_name, byte_size, len(in_bytes))
            if reached_eof:
                logger.debug('%s audio output reached EOF after %d complete block(s)', source_name, complete_blocks)
                break
            continue
        audio = np.frombuffer(in_bytes, np.float32).flatten()
        if len(in_bytes) != byte_size:
            expected_samples = byte_size // np.dtype(np.float32).itemsize
            audio = np.pad(audio, (0, expected_samples - len(audio)))
            logger.debug('%s padded its final audio block from %d to %d sample(s)', source_name, valid_sample_count,
                         expected_samples)
        if complete_blocks == 0 and logger.isEnabledFor(logging.DEBUG):
            finite_audio = audio[np.isfinite(audio)]
            elapsed = time.monotonic() - started_at if started_at is not None else None
            if finite_audio.size:
                minimum = float(np.min(finite_audio))
                maximum = float(np.max(finite_audio))
                rms = float(np.sqrt(np.mean(np.square(finite_audio, dtype=np.float64))))
                logger.debug('%s produced its first PCM block%s; samples=%d, finite=%d, min=%.6f, max=%.6f, '
                             'rms=%.6f', source_name, f' after {elapsed:.3f}s' if elapsed is not None else '',
                             audio.size, finite_audio.size, minimum, maximum, rms)
            else:
                logger.debug('%s produced its first PCM block%s; samples=%d, finite=0', source_name,
                             f' after {elapsed:.3f}s' if elapsed is not None else '', audio.size)
        if pacer is not None and not pacer.wait_for_block(valid_sample_count):
            logger.debug('%s realtime playback stopped after %d emitted block(s)', source_name, complete_blocks)
            break
        if not _put_audio_block(output_queue, audio, stop_event):
            logger.debug('%s audio output stopped while waiting for downstream capacity after %d emitted block(s)',
                         source_name, complete_blocks)
            break
        complete_blocks += 1
        if reached_eof:
            logger.debug('%s audio output reached EOF after %d emitted block(s)', source_name, complete_blocks)
            break
    return complete_blocks


def _kill_process(process, process_name: str, process_tree: bool = False):
    """Kill a child, including its private POSIX process group when requested."""
    process_id = getattr(process, 'pid', None)
    if (process_tree and os.name != 'nt' and isinstance(process_id, int)
            and not isinstance(process_id, bool)):
        try:
            # _open_stream starts yt-dlp as the leader of a private session,
            # therefore its PID is also the process-group ID. Using the PID
            # directly avoids ever resolving and signalling a shared host
            # process group if a foreign Popen is passed accidentally.
            os.killpg(process_id, signal.SIGKILL)
            return
        except OSError:
            logger.debug('Could not kill the %s process group; falling back to its parent process',
                         process_name, exc_info=True)
    process.kill()


def _wait_for_process(process,
                      process_name: str,
                      timeout: float = 5.0,
                      process_tree: bool = False):
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error('%s did not exit within %.1fs; killing it', process_name, timeout)
        _kill_process(process, process_name, process_tree=process_tree)
        return_code = process.wait()

    if return_code == 0:
        logger.debug('%s exited successfully', process_name)
    else:
        logger.error('%s exited with status %s', process_name, return_code)
    return return_code


def _cleanup_stream_children(ffmpeg_process, ytdlp_process, transport_thread=None):
    """Kill and reap both sides of a failed stream pipeline."""
    for process_name, process in (('ffmpeg', ffmpeg_process), ('yt-dlp', ytdlp_process)):
        if process is None:
            continue
        try:
            # The yt-dlp leader may already have exited while its internal HLS
            # ffmpeg still owns the stdout pipe, so always signal its private
            # POSIX group during failure cleanup.
            if (process_name == 'yt-dlp' and os.name != 'nt') or process.poll() is None:
                _kill_process(process, process_name, process_tree=process_name == 'yt-dlp')
                logger.debug('Killed %s during stream failure cleanup', process_name)
        except OSError:
            logger.debug('%s exited before stream failure cleanup could kill it', process_name, exc_info=True)

    if transport_thread is not None and transport_thread.is_alive():
        transport_thread.join(timeout=5)

    for process_name, process in (('ffmpeg', ffmpeg_process), ('yt-dlp', ytdlp_process)):
        if process is None:
            continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error('%s did not exit during stream failure cleanup; killing it again', process_name)
            try:
                _kill_process(process, process_name, process_tree=process_name == 'yt-dlp')
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                logger.exception('Could not reap %s during stream failure cleanup', process_name)
        except OSError:
            logger.debug('%s was already reaped during stream failure cleanup', process_name, exc_info=True)

    if transport_thread is not None and transport_thread.is_alive():
        logger.error('Stream transport thread remained alive after child-process cleanup')


def _transport(ytdlp_proc, ffmpeg_proc, stats: _StreamTransportStats | None = None):
    stats = stats or _StreamTransportStats(started_at=time.monotonic())
    ffmpeg_closed_early = False
    try:
        while True:
            chunk = ytdlp_proc.stdout.read(1024)
            if not chunk:
                logger.debug('yt-dlp output reached EOF')
                break
            stats.media_bytes += len(chunk)
            stats.media_chunks += 1
            if stats.first_media_elapsed is None:
                stats.first_media_elapsed = time.monotonic() - stats.started_at
                logger.debug('yt-dlp produced its first media bytes after %.3fs', stats.first_media_elapsed)
                _log_stream_format_diagnostic(stats)
            try:
                ffmpeg_proc.stdin.write(chunk)
                ffmpeg_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                ffmpeg_closed_early = True
                stats.transport_error = 'ffmpeg input pipe closed before yt-dlp finished'
                logger.error('ffmpeg closed its input pipe before yt-dlp finished')
                break
    except Exception as exc:
        ffmpeg_closed_early = True
        stats.transport_error = f'unexpected {type(exc).__name__}'
        logger.exception('Unexpected failure while transporting yt-dlp audio to ffmpeg')
    finally:
        try:
            ffmpeg_proc.stdin.close()
        except (BrokenPipeError, OSError):
            logger.debug('ffmpeg input pipe was already closed')

        if ffmpeg_closed_early and ytdlp_proc.poll() is None:
            _kill_process(ytdlp_proc, 'yt-dlp', process_tree=True)
        stats.ytdlp_return_code = _wait_for_process(ytdlp_proc, 'yt-dlp', process_tree=True)
        close_stdout = getattr(ytdlp_proc.stdout, 'close', None)
        if callable(close_stdout):
            try:
                close_stdout()
            except OSError:
                logger.debug('yt-dlp stdout was already closed', exc_info=True)
        _log_stream_format_diagnostic(stats)
        logger.debug('yt-dlp media transport finished after %.3fs; bytes=%d, chunks=%d, return_code=%s',
                     time.monotonic() - stats.started_at, stats.media_bytes, stats.media_chunks,
                     stats.ytdlp_return_code)
    return stats


def _open_stream(url: str,
                 format: str,
                 cookies: str,
                 proxy: str,
                 cwd: str,
                 wait_for_video: int | None = None,
                 live_from_start: bool = False,
                 stream_format_diagnostic_file: str | None = None,
                 ss: str | float | None = None,
                 to: str | float | None = None):
    if live_from_start and (ss is not None or to is not None):
        logger.warning('--ss and --to are ignored when --live-from-start is enabled')
        ss = to = None
    ss, to = normalize_time_range(ss, to)
    _log_stream_tool_versions()
    # stdout is reserved exclusively for media bytes. Quiet mode suppresses
    # progress/status text, while warnings and errors remain on inherited stderr.
    cmd = [sys.executable, '-m', 'yt_dlp', url, '-f', format, '-o', '-', '--quiet']
    format_diagnostic_path = None
    if stream_format_diagnostic_file and stream_format_diagnostic_file.strip():
        # yt-dlp runs in a private temporary working directory. Resolve the
        # user-facing path here so a relative path remains relative to the
        # caller and the diagnostic survives temporary-directory cleanup.
        format_diagnostic_path = Path(stream_format_diagnostic_file).expanduser().resolve()
        # yt-dlp's --print-to-file appends. Treat the explicitly supplied path
        # as one diagnostic artifact per stream instead of accumulating stale
        # records across repeated runs.
        format_diagnostic_path.write_text('', encoding='utf-8')
        cmd.extend(['--print-to-file', f'before_dl:{_STREAM_FORMAT_TEMPLATE}', str(format_diagnostic_path)])
    if wait_for_video is not None:
        cmd.extend(['--wait-for-video', str(wait_for_video)])
    if live_from_start:
        cmd.append('--live-from-start')
    if cookies:
        cmd.extend(['--cookies', cookies])
    if proxy:
        cmd.extend(['--proxy', proxy])
    process_options = {}
    if ss is not None or to is not None:
        # Select seek options after extraction, using the chosen media protocol
        # rather than guessing from the page URL. The wrapper owns both range
        # boundaries, so the downstream decoder still sees a normal EOF.
        cmd = [sys.executable, '-m', 'stream_translator_gpt.getter.range_downloader',
               format_seconds(ss) if ss is not None else '',
               format_seconds(to) if to is not None else '', '--', *cmd[3:]]
        # The downloader runs in a private temp directory. Keep source-checkout
        # launches working even when this package is not installed into Python.
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[2])
        existing_path = environment.get('PYTHONPATH')
        environment['PYTHONPATH'] = package_root + (os.pathsep + existing_path if existing_path else '')
        process_options['env'] = environment
    started_at = time.monotonic()
    stats = _StreamTransportStats(started_at=started_at, format_diagnostic_path=format_diagnostic_path)
    # Live HLS extractors can spawn an internal ffmpeg downloader. On POSIX,
    # place yt-dlp and its descendants in a private session so notebook
    # shutdown can terminate the complete process tree deterministically.
    ytdlp_process = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=None,
                                     cwd=cwd,
                                     start_new_session=os.name != 'nt',
                                     **process_options)

    try:
        # Keep stderr inherited: media remains isolated on stdout, while ffmpeg
        # errors stay visible without creating a pipe that nobody drains.
        ffmpeg_process = (ffmpeg.input('pipe:0', loglevel='error').output('pipe:1',
                                                                         format='f32le',
                                                                         acodec='pcm_f32le',
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(pipe_stdin=True,
                                                                                                   pipe_stdout=True))
    except Exception as exc:
        if ytdlp_process.poll() is None:
            try:
                _kill_process(ytdlp_process, 'yt-dlp', process_tree=True)
            except OSError:
                logger.debug('yt-dlp exited before it could be terminated after ffmpeg startup failure', exc_info=True)
        _wait_for_process(ytdlp_process, 'yt-dlp', process_tree=True)
        if isinstance(exc, ffmpeg.Error):
            stderr = exc.stderr.decode(errors='replace') if exc.stderr else str(exc)
            raise RuntimeError(f'Failed to load audio: {stderr}') from exc
        raise

    logger.debug('Started yt-dlp process %d and ffmpeg process %d', ytdlp_process.pid, ffmpeg_process.pid)
    try:
        thread = threading.Thread(target=_transport,
                                  name='stream-transport',
                                  args=(ytdlp_process, ffmpeg_process, stats),
                                  daemon=True)
        thread.start()
    except Exception:
        _cleanup_stream_children(ffmpeg_process, ytdlp_process)
        raise
    return ffmpeg_process, ytdlp_process, thread, stats


class StreamAudioGetter(LoopWorkerBase):

    def __init__(self,
                 url: str,
                 format: str,
                 cookies: str,
                 proxy: str,
                 realtime: bool = False,
                 wait_for_video: int | None = None,
                 live_from_start: bool = False,
                 stream_format_diagnostic_file: str | None = None,
                 ss: str | float | None = None,
                 to: str | float | None = None) -> None:
        if live_from_start and (ss is not None or to is not None):
            logger.warning('--ss and --to are ignored when --live-from-start is enabled')
            ss = to = None
        self.ss, self.to = normalize_time_range(ss, to)
        self.url = url
        self.format = format
        self.cookies = cookies
        self.proxy = proxy
        self.realtime = realtime
        self.wait_for_video = wait_for_video
        self.live_from_start = live_from_start
        self.stream_format_diagnostic_file = stream_format_diagnostic_file
        self.temp_dir = tempfile.mkdtemp()
        self.ffmpeg_process = None
        self.ytdlp_process = None
        self.transport_thread = None
        self.stop_event = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped_ffmpeg_process = None
        self._stopped_ytdlp_process = None
        self.pacer = None
        self.byte_size = round(SAMPLES_PER_FRAME * 4)  # Factor 4 comes from float32 (4 bytes per sample)

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def stop(self) -> None:
        """Request shutdown and interrupt any blocking child-process reads."""
        processes_to_stop = []
        with self._stop_lock:
            self.stop_event.set()
            for process_name, process, stopped_attr in (
                    ('ffmpeg', self.ffmpeg_process, '_stopped_ffmpeg_process'),
                    ('yt-dlp', self.ytdlp_process, '_stopped_ytdlp_process')):
                if process is None or process is getattr(self, stopped_attr):
                    continue
                setattr(self, stopped_attr, process)
                processes_to_stop.append((process_name, process))
        for process_name, process in processes_to_stop:
            try:
                _kill_process(process, process_name, process_tree=process_name == 'yt-dlp')
            except OSError:
                logger.debug('%s exited before the stream getter stop request', process_name, exc_info=True)

    def _exit_handler(self, signum, frame):
        logger.info('Stopping stream audio getter after signal %s', signum)
        self.stop()
        timeout = _signal_cleanup_timeout(signum)
        if self.started.is_set() and not self.finished.wait(timeout=timeout):
            logger.warning('Timed out waiting for stream audio getter to stop cleanly')
        sys.exit(_signal_exit_code(signum))

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        self.started.set()
        logger.warning('Opening stream: %s', redact_url(self.url))
        if self.wait_for_video is not None:
            logger.warning('Waiting for scheduled stream availability; yt-dlp minimum retry interval is %d second(s)',
                           self.wait_for_video)
        completed = False
        try:
            range_options = {}
            if self.ss is not None or self.to is not None:
                range_options = {'ss': self.ss, 'to': self.to}
            self.ffmpeg_process, self.ytdlp_process, self.transport_thread, transport_stats = _open_stream(
                self.url,
                self.format,
                self.cookies,
                self.proxy,
                self.temp_dir,
                wait_for_video=self.wait_for_video,
                live_from_start=self.live_from_start,
                stream_format_diagnostic_file=self.stream_format_diagnostic_file,
                **range_options)
            if self.stop_event.is_set():
                logger.debug('Stream audio getter was stopped while opening its child processes')
                return
            if self.realtime:
                self.pacer = _RealtimeAudioPacer(self.stop_event, source_description='stream playback')
            pcm_blocks = _read_ffmpeg_loop(self.ffmpeg_process,
                                           self.byte_size,
                                           output_queue,
                                           source_name='stream ffmpeg',
                                           pacer=self.pacer,
                                           pad_final_block=bool(range_options),
                                           started_at=transport_stats.started_at,
                                           stop_event=self.stop_event)
            if self.stop_event.is_set():
                logger.debug('Stream audio getter stopped after %d PCM block(s)', pcm_blocks)
                return
            ffmpeg_return_code = _wait_for_process(self.ffmpeg_process, 'ffmpeg')
            self.transport_thread.join(timeout=5)
            transport_stuck = self.transport_thread.is_alive()
            if transport_stuck:
                logger.error('Stream transport thread did not exit within 5.0s; killing yt-dlp')
                _kill_process(self.ytdlp_process, 'yt-dlp', process_tree=True)
                self.transport_thread.join(timeout=1)
                transport_stuck = self.transport_thread.is_alive()

            ytdlp_return_code = transport_stats.ytdlp_return_code
            logger.debug('Stream process summary: PCM blocks=%d, media bytes=%d, yt-dlp return_code=%s, '
                         'ffmpeg return_code=%s, transport_error=%s', pcm_blocks, transport_stats.media_bytes,
                         ytdlp_return_code, ffmpeg_return_code, transport_stats.transport_error)
            failures = []
            if transport_stuck:
                failures.append('transport thread did not stop')
            if transport_stats.transport_error:
                failures.append(f'transport failed ({transport_stats.transport_error})')
            if ytdlp_return_code is None:
                failures.append('yt-dlp exit status was unavailable')
            elif ytdlp_return_code != 0:
                failures.append(f'yt-dlp exited with status {ytdlp_return_code}')
            if ffmpeg_return_code != 0:
                failures.append(f'ffmpeg exited with status {ffmpeg_return_code}')
            if pcm_blocks == 0:
                failures.append('no decoded PCM audio was produced')
            if failures:
                raise RuntimeError('Stream audio pipeline failed: ' + '; '.join(failures))

            logger.info('Stream audio input finished')
            completed = True
        finally:
            try:
                _signal_end_of_stream(output_queue,
                                      discard_buffered=self.stop_event.is_set() or not completed,
                                      stop_event=self.stop_event)
            finally:
                try:
                    if not completed:
                        _cleanup_stream_children(self.ffmpeg_process, self.ytdlp_process, self.transport_thread)
                    if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                        shutil.rmtree(self.temp_dir, ignore_errors=True)
                finally:
                    self.finished.set()


class LocalFileAudioGetter(LoopWorkerBase):

    def __init__(self,
                 file_path: str,
                 realtime: bool = False,
                 ss: str | float | None = None,
                 to: str | float | None = None) -> None:
        self.ss, self.to = normalize_time_range(ss, to)
        self.file_path = file_path
        self.realtime = realtime
        self.ffmpeg_process = None
        self.stop_event = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped_ffmpeg_process = None
        self.pacer = None
        self.byte_size = round(SAMPLES_PER_FRAME * 4)  # Factor 4 comes from float32 (4 bytes per sample)

    def stop(self) -> None:
        """Request cancellation and interrupt the decoder without waiting from a foreign thread."""
        process = None
        with self._stop_lock:
            self.stop_event.set()
            if (self.ffmpeg_process is not None
                    and self.ffmpeg_process is not self._stopped_ffmpeg_process):
                process = self.ffmpeg_process
                self._stopped_ffmpeg_process = process
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                logger.debug('Local file ffmpeg exited before the stop request', exc_info=True)

    def _exit_handler(self, signum, frame):
        logger.info('Stopping local file audio getter after signal %s', signum)
        self.stop()
        timeout = _signal_cleanup_timeout(signum)
        if self.started.is_set() and not self.finished.wait(timeout=timeout):
            logger.warning('Timed out waiting for local file audio getter to stop cleanly')
        sys.exit(_signal_exit_code(signum))

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        self.started.set()
        pacing = 'realtime' if self.realtime else 'unlimited'
        logger.warning('Opening local file with %s pacing: %s', pacing, self.file_path)
        failed = False
        try:
            try:
                has_range = self.ss is not None or self.to is not None
                linear_seek = has_range and (Path(self.file_path).suffix.lower() in ('.ts', '.mpegts')
                                             or is_local_hls(self.file_path))
                input_options, output_options = ffmpeg_time_range_options(
                    self.ss, self.to, linear=linear_seek)
                input_options['loglevel'] = 'panic'
                self.ffmpeg_process = (ffmpeg.input(self.file_path,
                                                    **input_options).output('pipe:',
                                                                             format='f32le',
                                                                             acodec='pcm_f32le',
                                                                             ac=1,
                                                                             ar=SAMPLE_RATE,
                                                                             **output_options).run_async(
                                                                                  pipe_stdin=True,
                                                                                  pipe_stdout=True))
            except ffmpeg.Error as e:
                stderr = e.stderr.decode(errors='replace') if e.stderr else str(e)
                raise RuntimeError(f'Failed to load audio: {stderr}') from e

            if self.stop_event.is_set():
                self.stop()
            if self.realtime:
                self.pacer = _RealtimeAudioPacer(self.stop_event)
            _read_ffmpeg_loop(self.ffmpeg_process,
                              self.byte_size,
                              output_queue,
                              source_name='local file ffmpeg',
                              pacer=self.pacer,
                              pad_final_block=self.realtime or self.ss is not None or self.to is not None,
                              stop_event=self.stop_event)
            if self.stop_event.is_set():
                self.stop()
            return_code = _wait_for_process(self.ffmpeg_process, 'local file ffmpeg')
            if not self.stop_event.is_set() and return_code != 0:
                raise RuntimeError(f'Local file ffmpeg exited with status {return_code}')
            logger.info('Local file audio input finished')
        except Exception:
            failed = True
            if self.ffmpeg_process is not None:
                if self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.kill()
                    try:
                        self.ffmpeg_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        logger.error('Local file ffmpeg did not exit within 5.0s during failure cleanup')
            raise
        finally:
            try:
                _signal_end_of_stream(output_queue, discard_buffered=self.stop_event.is_set() or failed,
                                      stop_event=self.stop_event)
            finally:
                self.finished.set()


def _device_gain(volume_factor: float) -> float:
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


# Kept for callers that imported the helper while the option was SoundDevice-only.
_sounddevice_gain = _device_gain


def _prepare_device_audio(audio: np.ndarray, native_rate: int, channels: int, gain: float) -> np.ndarray:
    """Downmix, resample and apply gain outside an audio backend's callback thread."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio.reshape(-1)
        if channels > 1:
            complete_samples = len(audio) - len(audio) % channels
            if complete_samples != len(audio):
                logger.warning('Discarding %d incomplete interleaved device sample(s)', len(audio) - complete_samples)
                audio = audio[:complete_samples]
            if complete_samples:
                audio = audio.reshape(-1, channels).mean(axis=1)
    if native_rate != SAMPLE_RATE and len(audio):
        target_len = max(1, round(len(audio) * SAMPLE_RATE / native_rate))
        audio = resample(audio, target_len)
    return np.asarray(audio, dtype=np.float32) * gain


class SoundDeviceAudioGetter(LoopWorkerBase):
    """Compatibility device getter backed by a continuous sounddevice stream."""

    def __init__(self,
                 device_index: int | None,
                 interval: float = 0.5,
                 multiplier: float = 1.0,
                 debug_audio_dir: str | None = None,
                 debug_clip_duration: float = 10.0,
                 debug_max_clips: int = 10) -> None:
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
        self.native_rate = int(float(device_info.get('default_samplerate', SAMPLE_RATE)))
        if self.native_rate <= 0:
            self.native_rate = SAMPLE_RATE
        self.channels = min(int(device_info.get('max_input_channels', 1)), 2)
        self.interval = interval
        self.volume_factor = multiplier
        self.gain = _device_gain(multiplier)
        self.recording_queue = queue.Queue(maxsize=8)
        self.dropped_blocks = 0
        self.stream = None
        self.stopping = threading.Event()
        self.cancel_requested = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self._stop_lock = threading.Lock()
        self._aborted_stream = None
        self.debug_dumper = (_DebugAudioDumper(debug_audio_dir, debug_clip_duration, debug_max_clips)
                             if debug_audio_dir else None)

    def stop(self) -> None:
        """Request capture shutdown and wake the owner thread exactly once."""
        queue_stop = False
        stream_to_abort = None
        with self._stop_lock:
            self.cancel_requested.set()
            if not self.stopping.is_set():
                self.stopping.set()
                queue_stop = True
            if self.stream is not None and self.stream is not self._aborted_stream:
                stream_to_abort = self.stream
                self._aborted_stream = self.stream
        if queue_stop:
            _wake_recording_queue(self.recording_queue, discard_buffered=True)
        if stream_to_abort is not None:
            try:
                stream_to_abort.abort()
            except Exception:
                logger.debug('Unable to abort sounddevice stream cleanly', exc_info=True)

    def _exit_handler(self, signum, frame):
        logger.info('Stopping sounddevice audio getter after signal %s', signum)
        self.stop()
        timeout = _signal_cleanup_timeout(signum)
        if self.started.is_set() and not self.finished.wait(timeout=timeout):
            logger.warning('Timed out waiting for sounddevice audio getter to stop cleanly')
        sys.exit(_signal_exit_code(signum))

    def _recording_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning('Sounddevice recorder returned status=%s', status)
        if self.stopping.is_set():
            return
        audio = indata.copy()
        try:
            self.recording_queue.put_nowait(audio)
        except queue.Full:
            try:
                self.recording_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.recording_queue.put_nowait(audio)
            except queue.Full:
                return
            self.dropped_blocks += 1
            if self.dropped_blocks == 1 or self.dropped_blocks % 100 == 0:
                logger.warning('Sounddevice processing is behind; dropped %d queued audio block(s)',
                               self.dropped_blocks)

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        return _prepare_device_audio(audio, self.native_rate, self.channels, self.gain)

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        self.started.set()
        logger.warning('Recording sounddevice input: %s (index %s)', self.device_name, self.device_index)
        logger.warning('Recording multiplier: %.6f (volume factor %.3f)', self.gain, self.volume_factor)
        read_size = max(1, round(self.native_rate * self.interval))
        buffer = np.array([], dtype=np.float32)
        failed = False

        try:
            if self.stopping.is_set():
                return
            with self.sounddevice.InputStream(blocksize=read_size,
                                              samplerate=self.native_rate,
                                              channels=self.channels,
                                              dtype=np.float32,
                                              device=self.device_index,
                                              callback=self._recording_callback) as stream:
                self.stream = stream
                while True:
                    audio = self.recording_queue.get()
                    if audio is None:
                        break
                    audio = self._prepare_audio(audio)
                    if self.debug_dumper is not None:
                        self.debug_dumper.add(audio)
                    buffer = np.concatenate((buffer, audio))
                    while len(buffer) >= SAMPLES_PER_FRAME:
                        if not _put_audio_block(output_queue, buffer[:SAMPLES_PER_FRAME], self.cancel_requested):
                            break
                        buffer = buffer[SAMPLES_PER_FRAME:]
                    if self.cancel_requested.is_set():
                        break
        except Exception:
            failed = True
            raise
        finally:
            self.stopping.set()
            self.stream = None
            try:
                _signal_end_of_stream(output_queue,
                                      discard_buffered=self.cancel_requested.is_set() or failed,
                                      stop_event=self.cancel_requested)
            finally:
                try:
                    if self.debug_dumper is not None:
                        self.debug_dumper.close()
                finally:
                    self.finished.set()


class _DebugAudioDumper:
    """Write bounded, atomic PCM16 WAV clips from processed device audio."""

    def __init__(self, output_dir: str, clip_duration: float, max_clips: int) -> None:
        if clip_duration <= 0:
            raise ValueError('device debug clip duration must be greater than zero')
        if max_clips <= 0:
            raise ValueError('device debug max clips must be greater than zero')
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clip_samples = max(1, round(clip_duration * SAMPLE_RATE))
        self.max_clips = max_clips
        self.clip_count = 0
        self.chunks = []
        self.sample_count = 0
        self.closed = False
        self.limit_logged = False

    def add(self, audio: np.ndarray) -> None:
        if self.closed or self.clip_count >= self.max_clips:
            self._log_limit_once()
            return
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(audio):
            return
        self.chunks.append(audio.copy())
        self.sample_count += len(audio)
        while self.sample_count >= self.clip_samples and self.clip_count < self.max_clips:
            combined = np.concatenate(self.chunks)
            if not self._write_clip(combined[:self.clip_samples]):
                self.chunks = []
                self.sample_count = 0
                return
            remainder = combined[self.clip_samples:]
            self.chunks = [remainder] if len(remainder) else []
            self.sample_count = len(remainder)
        if self.clip_count >= self.max_clips:
            self.chunks = []
            self.sample_count = 0
            self._log_limit_once()

    def close(self) -> None:
        if self.closed:
            return
        if self.sample_count and self.clip_count < self.max_clips:
            self._write_clip(np.concatenate(self.chunks))
        self.chunks = []
        self.sample_count = 0
        self.closed = True

    def _log_limit_once(self) -> None:
        if not self.limit_logged and self.clip_count >= self.max_clips:
            logger.warning('Device debug audio reached the maximum of %d clip(s); capture disabled', self.max_clips)
            self.limit_logged = True

    def _write_clip(self, audio: np.ndarray) -> bool:
        sequence = self.clip_count + 1
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        final_path = self.output_dir / f'device-audio-{timestamp}-{os.getpid()}-{sequence:04d}.wav'
        while final_path.exists():
            sequence += 1
            final_path = self.output_dir / f'device-audio-{timestamp}-{os.getpid()}-{sequence:04d}.wav'
        part_path = final_path.with_suffix(final_path.suffix + '.part')
        clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm = np.clip(np.rint(clipped * 32768.0), -32768, 32767).astype('<i2')
        try:
            with wave.open(str(part_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(pcm.tobytes())
            os.replace(part_path, final_path)
        except Exception:
            try:
                part_path.unlink(missing_ok=True)
            finally:
                self.closed = True
            logger.exception('Failed to write device debug audio clip')
            return False
        self.clip_count += 1
        logger.info('Saved device debug audio clip %d/%d: %s', self.clip_count, self.max_clips, final_path)
        return True


class DeviceAudioGetter(LoopWorkerBase):
    """PyAudio device getter backed by a non-blocking callback stream."""

    def __init__(self,
                 device_index: int | None,
                 use_mic: bool,
                 interval: float = 0.5,
                 multiplier: float = 1.0,
                 debug_audio_dir: str | None = None,
                 debug_clip_duration: float = 10.0,
                 debug_max_clips: int = 10) -> None:
        if interval <= 0:
            raise ValueError('device recording interval must be greater than zero')
        gain = _device_gain(multiplier)
        if debug_audio_dir and debug_clip_duration <= 0:
            raise ValueError('device debug clip duration must be greater than zero')
        if debug_audio_dir and debug_max_clips <= 0:
            raise ValueError('device debug max clips must be greater than zero')

        if platform.system() == 'Windows':
            import pyaudiowpatch as pyaudio
        else:
            try:
                import pyaudio
            except ImportError as e:
                raise RuntimeError("PyAudio is not installed. Please install it to use device capture.\n"
                                   "Debian/Ubuntu/Colab: apt install portaudio19-dev && pip install pyaudio") from e

        self.pyaudio = None
        self.pyaudio_module = pyaudio
        self.requested_device_index = device_index
        self.device_index = device_index
        self.use_mic = use_mic
        self.interval = interval
        self.stream = None
        self.device_name = None
        self.native_rate = None
        self.channels = None
        self.read_size = None
        self.volume_factor = multiplier
        self.gain = gain
        self.recording_queue = queue.Queue(maxsize=8)
        self.dropped_blocks = 0
        self.logged_dropped_blocks = 0
        self.callback_error = None
        self.stopping = threading.Event()
        self.cancel_requested = threading.Event()
        self.signal_received = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self._stop_lock = threading.Lock()
        self.cleanup_lock = threading.Lock()
        self.pyaudio_terminated = True
        self.debug_dumper = (_DebugAudioDumper(debug_audio_dir, debug_clip_duration, debug_max_clips)
                             if debug_audio_dir else None)

    def _initialize_pyaudio(self) -> None:
        """Initialize PortAudio and resolve the device in the capture thread.

        WASAPI initialization is thread-sensitive for some Windows endpoints. The
        getter itself is constructed by the initializer pool, while ``loop`` runs
        in a dedicated worker, so creating ``PyAudio`` in ``__init__`` can make a
        later ``start_stream`` fail with ``paUnanticipatedHostError``.
        """
        if self.pyaudio is not None:
            return

        self.device_index = self.requested_device_index
        self.pyaudio = self.pyaudio_module.PyAudio()
        self.pyaudio_terminated = False
        try:
            device_info = self._select_device_info()
            self.device_name = device_info['name']
            self.native_rate = int(float(device_info['defaultSampleRate']))
            if self.native_rate <= 0:
                raise RuntimeError(f'PyAudio device {self.device_index} reported an invalid sample rate.')
            try:
                self.channels = int(device_info['maxInputChannels'])
            except (KeyError, TypeError, ValueError):
                logger.debug('Unable to determine the PyAudio device channel count; defaulting to one channel',
                             exc_info=True)
                self.channels = 1
            if self.channels < 1:
                # Preserve the upstream getter's fallback for output/loopback devices whose metadata reports zero.
                logger.warning('PyAudio device %s reported no input channels; trying two channels', self.device_index)
                self.channels = 2
            self.read_size = max(1, round(self.native_rate * self.interval))
        except Exception:
            try:
                self.pyaudio.terminate()
            except Exception:
                logger.debug('Unable to terminate PyAudio after initialization failure', exc_info=True)
            finally:
                self.pyaudio = None
                self.pyaudio_terminated = True
            raise

    def _select_device_info(self):
        if self.use_mic:
            if self.device_index is None:
                default_device = self.pyaudio.get_default_input_device_info()
                self.device_index = default_device['index']
        elif platform.system() == 'Windows':
            if self.device_index is None:
                try:
                    wasapi_info = self.pyaudio.get_host_api_info_by_type(self.pyaudio_module.paWASAPI)
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
        elif self.device_index is None:
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                if 'monitor' in info['name'].lower() and info['maxInputChannels'] > 0:
                    self.device_index = info['index']
                    break
            else:
                raise RuntimeError("No monitor device found for loopback capture.")

        return self.pyaudio.get_device_info_by_index(self.device_index)

    def _queue_stop(self, *, cancelled: bool = False) -> None:
        queue_stop = False
        with self._stop_lock:
            if cancelled:
                self.cancel_requested.set()
            if not self.stopping.is_set():
                self.stopping.set()
                queue_stop = True
        if queue_stop:
            _wake_recording_queue(self.recording_queue, discard_buffered=cancelled)

    def stop(self) -> None:
        """Wake the capture owner thread; PortAudio resources remain owner-thread managed."""
        self._queue_stop(cancelled=True)

    def _recording_callback(self, in_data, frame_count, time_info, status_flags):
        if self.stopping.is_set():
            return None, self.pyaudio_module.paComplete
        try:
            if not in_data:
                self._queue_stop()
                return None, self.pyaudio_module.paComplete
            block = (bytes(in_data), status_flags)
            try:
                self.recording_queue.put_nowait(block)
            except queue.Full:
                try:
                    oldest = self.recording_queue.get_nowait()
                except queue.Empty:
                    oldest = None
                if oldest is None or self.stopping.is_set():
                    self._queue_stop()
                    return None, self.pyaudio_module.paComplete
                try:
                    self.recording_queue.put_nowait(block)
                except queue.Full:
                    return None, self.pyaudio_module.paContinue
                self.dropped_blocks += 1
            return None, self.pyaudio_module.paContinue
        except Exception:
            self.callback_error = sys.exc_info()
            self._queue_stop()
            return None, getattr(self.pyaudio_module, 'paAbort', self.pyaudio_module.paComplete)

    def _log_dropped_blocks(self) -> None:
        if self.dropped_blocks == self.logged_dropped_blocks:
            return
        if self.logged_dropped_blocks == 0 or self.dropped_blocks // 100 > self.logged_dropped_blocks // 100:
            logger.warning('PyAudio processing is behind; dropped %d queued audio block(s)', self.dropped_blocks)
        self.logged_dropped_blocks = self.dropped_blocks

    def _cleanup(self) -> None:
        with self.cleanup_lock:
            stream = self.stream
            self.stream = None
            if stream is not None:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                except Exception:
                    logger.debug('Unable to stop PyAudio stream cleanly', exc_info=True)
                try:
                    stream.close()
                except Exception:
                    logger.debug('Unable to close PyAudio stream cleanly', exc_info=True)
            pyaudio = self.pyaudio
            self.pyaudio = None
            if pyaudio is not None and not self.pyaudio_terminated:
                try:
                    pyaudio.terminate()
                except Exception:
                    logger.debug('Unable to terminate PyAudio cleanly', exc_info=True)
                finally:
                    self.pyaudio_terminated = True

    def _exit_handler(self, signum, frame):
        logger.info('Stopping PyAudio device getter after signal %s', signum)
        if self.signal_received.is_set():
            raise SystemExit(_signal_exit_code(signum))
        self.signal_received.set()
        self.stop()
        timeout = _signal_cleanup_timeout(signum)
        if self.started.is_set() and not self.finished.wait(timeout=timeout):
            logger.warning('Timed out waiting for PyAudio device getter to stop cleanly')
        sys.exit(_signal_exit_code(signum))

    def loop(self, output_queue: queue.SimpleQueue[np.array]):
        self.started.set()
        buffer = np.array([], dtype=np.float32)
        failed = False

        try:
            with self.cleanup_lock:
                if self.stopping.is_set():
                    return
                self._initialize_pyaudio()
                logger.warning('Recording PyAudio device: %s (%s)', self.device_name,
                               'Input' if self.use_mic else 'Output')
                logger.warning('Recording multiplier: %.6f (volume factor %.3f)', self.gain, self.volume_factor)
                logger.debug('Opening audio device index=%s, native_rate=%d, channels=%d, read_size=%d, resample=%s',
                             self.device_index, self.native_rate, self.channels, self.read_size,
                             self.native_rate != SAMPLE_RATE)
                stream = self.pyaudio.open(format=self.pyaudio_module.paFloat32,
                                           channels=self.channels,
                                           rate=self.native_rate,
                                           input=True,
                                           input_device_index=self.device_index,
                                           frames_per_buffer=self.read_size,
                                           stream_callback=self._recording_callback,
                                           start=False)
                self.stream = stream
                if not self.stopping.is_set():
                    stream.start_stream()

            while True:
                try:
                    item = self.recording_queue.get(timeout=0.25)
                except queue.Empty:
                    self._log_dropped_blocks()
                    if self.callback_error is not None:
                        _, error, _ = self.callback_error
                        raise RuntimeError(f'PyAudio callback failed: {error}') from error
                    if self.stopping.is_set():
                        break
                    try:
                        active = stream.is_active()
                    except Exception as exc:
                        raise RuntimeError('Unable to query PyAudio stream state') from exc
                    if not active:
                        logger.warning('PyAudio stream stopped without an end-of-stream callback')
                        break
                    continue

                self._log_dropped_blocks()
                if item is None:
                    if self.callback_error is not None:
                        _, error, _ = self.callback_error
                        raise RuntimeError(f'PyAudio callback failed: {error}') from error
                    break
                in_data, status_flags = item
                if status_flags:
                    logger.warning('PyAudio recorder returned status=%s', status_flags)
                audio = np.frombuffer(in_data, dtype=np.float32)
                audio = _prepare_device_audio(audio, self.native_rate, self.channels, self.gain)
                if self.debug_dumper is not None:
                    self.debug_dumper.add(audio)
                buffer = np.concatenate((buffer, audio))
                while len(buffer) >= SAMPLES_PER_FRAME:
                    if not _put_audio_block(output_queue, buffer[:SAMPLES_PER_FRAME], self.cancel_requested):
                        break
                    buffer = buffer[SAMPLES_PER_FRAME:]
                if self.cancel_requested.is_set():
                    break
        except Exception:
            failed = True
            raise
        finally:
            logger.debug('Closing PyAudio device')
            self.stopping.set()
            try:
                _signal_end_of_stream(output_queue,
                                      discard_buffered=self.cancel_requested.is_set() or failed,
                                      stop_event=self.cancel_requested)
            finally:
                try:
                    self._cleanup()
                    if self.debug_dumper is not None:
                        self.debug_dumper.close()
                finally:
                    self.finished.set()
