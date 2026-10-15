"""Parse source positions shared by the local-file and stream audio getters."""

import re
from decimal import Decimal, localcontext


_SECONDS = re.compile(r'([0-9]+(?:\.[0-9]*)?)(s|ms|us)?', re.ASCII)
_CLOCK = re.compile(r'(?:(?P<hours>[0-9]+):)?(?P<minutes>[0-5]?[0-9]):'
                    r'(?P<seconds>[0-5]?[0-9](?:\.[0-9]*)?)', re.ASCII)
# FFmpeg represents durations as signed 64-bit microseconds. Leave a little
# headroom so conversion to float cannot round an accepted value out of range.
_MAX_SECONDS = (2 ** 63 - 1) // 1_000_000


def _parse_position(value: str | float, option: str) -> float:
    error = (f'{option} must be a non-negative finite time in seconds or '
             '[HH:]MM:SS[.fraction] (seconds also accept s, ms and us suffixes)')
    if isinstance(value, bool):
        raise ValueError(error)
    if not isinstance(value, (int, float, str)):
        raise ValueError(error)
    with localcontext() as context:
        # Keep even long fractional inputs exact until microsecond truncation.
        context.prec = max(28, len(str(value)) + 12)
        if isinstance(value, (int, float)):
            seconds = Decimal(str(value))
        else:
            clock = _CLOCK.fullmatch(value)
            plain = _SECONDS.fullmatch(value.lstrip(' \t\r\n\v\f').removeprefix('+'))
            if clock:
                seconds = (Decimal(clock['hours'] or '0') * 3600
                           + Decimal(clock['minutes']) * 60 + Decimal(clock['seconds']))
            elif plain:
                scale = {'s': 1, 'ms': 1000, 'us': 1_000_000}
                seconds = Decimal(plain[1]) / scale[plain[2] or 's']
            else:
                raise ValueError(error)
        if not seconds.is_finite() or seconds < 0 or seconds > _MAX_SECONDS:
            raise ValueError(error)
        # FFmpeg truncates sub-microsecond precision. Validate the final interval
        # after this conversion so distinct strings cannot silently become empty.
        return int(seconds * 1_000_000) / 1_000_000


def normalize_time_range(ss: str | float | None, to: str | float | None
                         ) -> tuple[float | None, float | None]:
    """Return source start/end seconds; ``to`` is an absolute end, not a length.

    Only non-negative source positions are accepted. In particular, a negative
    value must never reach yt-dlp, where it would mean an offset from the end.
    """
    start = _parse_position(ss, '--ss') if ss is not None else None
    end = _parse_position(to, '--to') if to is not None else None
    if end is not None and end <= (start or 0):
        raise ValueError('--to must be greater than --ss (or zero when --ss is omitted)')
    return start, end


def format_seconds(seconds: float) -> str:
    """Use FFmpeg's microsecond precision without scientific notation."""
    return f'{seconds:.6f}'.rstrip('0').rstrip('.') or '0'
