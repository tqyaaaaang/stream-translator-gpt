"""Range-only yt-dlp entry point; stdout remains exclusively media bytes.

Choose FFmpeg's seek strategy after format selection, not from the webpage URL.
This keeps yt-dlp responsible for extraction, authentication and transport and
does not require a second metadata request from the parent process.
"""

import re
import sys
from urllib.parse import urlsplit

from yt_dlp import YoutubeDL, parse_options
from yt_dlp.downloader import get_suitable_downloader
from yt_dlp.downloader.external import FFmpegFD
from yt_dlp.postprocessor import PostProcessor
from yt_dlp.utils import DownloadError

from .time_range import ffmpeg_time_range_options, normalize_time_range
from ..utils.common import SAMPLE_RATE


def _requires_linear_seek(info: dict) -> bool:
    # With copyts, FFmpeg cannot normalize timestamp resets across HLS
    # discontinuities. Even a nonempty result can then contain the wrong audio.
    # Do not guess safety from a manifest snapshot: a live playlist can change.
    for selected in [info, *(info.get('requested_formats') or [])]:
        protocol = str(selected.get('protocol') or '').lower()
        extension = str(selected.get('ext') or '').lower()
        path = urlsplit(selected.get('url') or '').path.lower()
        if ('m3u8' in protocol or extension in ('m3u8', 'ts', 'mpegts')
                or path.endswith(('.m3u8', '.ts'))):
            return True
    return False


def _option_args(options: dict) -> list[str]:
    result = []
    for name, value in options.items():
        result.append(f'-{name}')
        if value is not None:
            result.append(str(value))
    return result


def configure_range_downloader(params: dict, info: dict, ss: float | None, to: float | None) -> bool:
    """Configure one resolved download and return whether linear seeking is used."""
    linear = _requires_linear_seek(info)
    input_options, output_options = ffmpeg_time_range_options(ss, to, linear=linear)
    args = dict(params.get('external_downloader_args') or {})
    # Indexed keys take precedence over the generic keys below in yt-dlp.
    # Range mode owns FFmpeg input/output options, including inherited config.
    args = {key: value for key, value in args.items() if not re.fullmatch(r'ffmpeg_[io]\d+', key)}
    args['ffmpeg_i'] = _option_args(input_options)
    # Decode upstream and send a pipe-safe container. A downstream -t would
    # close the pipe early; an input-side -t would count seek preroll as output.
    args['ffmpeg_o'] = ['-vn', '-c:a', 'pcm_f32le', '-ar', str(SAMPLE_RATE), '-ac', '1',
                        *_option_args(output_options), '-f', 'matroska']
    params['external_downloader_args'] = args
    params['external_downloader'] = {'default': 'ffmpeg'}
    return linear


class RangeDownloadPP(PostProcessor):
    """A before_dl hook: selected formats are known, downloading has not begun."""

    def __init__(self, downloader, ss: float | None, to: float | None):
        super().__init__(downloader)
        self.ss = ss
        self.to = to

    def run(self, info):
        linear = configure_range_downloader(self._downloader.params, info, self.ss, self.to)
        if get_suitable_downloader(info, self._downloader.params, to_stdout=True) != FFmpegFD:
            # yt-dlp can silently fall back to a native downloader for protocols
            # FFmpeg cannot handle. That would silently ignore our range.
            message = 'The selected format cannot be downloaded as an audio time range with FFmpeg'
            self._downloader.to_stderr(f'ERROR: {message}')
            # PostProcessingError is deferred by before_dl; it can let the
            # native downloader run anyway. DownloadError stops it immediately.
            raise DownloadError(message)
        self.write_debug('Audio range seek: ' + ('linear (discontinuous timestamps possible)' if linear
                                               else 'indexed seek with timestamp trimming'))
        return [], info


def parse_range_options(argv: list[str]):
    if len(argv) < 3 or argv[2] != '--':
        raise ValueError('Expected START END -- followed by yt-dlp options')
    ss, to = normalize_time_range(argv[0] or None, argv[1] or None)
    if ss is None and to is None:
        raise ValueError('At least one audio range boundary is required')
    parsed = parse_options(argv[3:])
    # This must already be set during format selection (e.g. separate video
    # and audio). The before_dl hook refines arguments for the selected format.
    parsed.ydl_opts['external_downloader'] = {'default': 'ffmpeg'}
    # Our two boundaries replace any download-sections inherited from yt-dlp's
    # user configuration; input-side section durations would count preroll.
    parsed.ydl_opts.pop('download_ranges', None)
    return ss, to, parsed


def main(argv=None) -> int:
    ss, to, parsed = parse_range_options(sys.argv[1:] if argv is None else argv)
    with YoutubeDL(parsed.ydl_opts) as downloader:
        downloader.add_post_processor(RangeDownloadPP(downloader, ss, to), when='before_dl')
        if parsed.options.load_info_filename is not None:
            return downloader.download_with_info_file(parsed.options.load_info_filename)
        return downloader.download(parsed.urls)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except DownloadError:
        # yt-dlp has already reported the useful error on stderr.
        raise SystemExit(1)
    except (KeyboardInterrupt, BrokenPipeError):
        raise SystemExit(1)
