import atexit
import copy
import logging
import queue
import re
import sys
import threading
from logging.handlers import QueueHandler, QueueListener
from urllib.parse import urlsplit, urlunsplit


LOGGER_NAME = 'main'
DDEBUG = logging.DEBUG - 1
logging.addLevelName(DDEBUG, 'DDEBUG')

logger = logging.getLogger(LOGGER_NAME)

_VERBOSITY_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG, DDEBUG)
_state_lock = threading.RLock()
_listener = None
_queue_handler = None
_stream_handler = None
_atexit_registered = False

_SENSITIVE_OPTION_RE = re.compile(r'(?:^|[-_])(key|token|secret|password|webhook)(?:$|[-_])', re.IGNORECASE)


class _LocalQueueHandler(QueueHandler):
    """Queue records without formatting them in latency-sensitive workers."""

    def prepare(self, record):
        return copy.copy(record)


def _normalize_verbosity(verbosity: int) -> int:
    try:
        verbosity = int(verbosity)
    except (TypeError, ValueError):
        verbosity = 0
    return max(0, min(3, verbosity))


def _shutdown_logging_locked():
    global _listener, _queue_handler, _stream_handler

    listener = _listener
    queue_handler = _queue_handler
    stream_handler = _stream_handler
    _listener = None
    _queue_handler = None
    _stream_handler = None

    if queue_handler is not None:
        logger.removeHandler(queue_handler)
    if listener is not None:
        listener.stop()
    if queue_handler is not None:
        queue_handler.close()
    if stream_handler is not None:
        stream_handler.close()


def configure_logging(verbosity: int = 0, stream=None):
    """Configure the application's non-blocking logger.

    Verbosity 0 shows warnings with a message-only format. Levels 1, 2 and 3
    enable INFO, DEBUG and DDEBUG respectively and include timestamp, level and
    thread name. Reconfiguration is idempotent and drains the previous queue.
    """
    global _listener, _queue_handler, _stream_handler, _atexit_registered

    verbosity = _normalize_verbosity(verbosity)
    level = _VERBOSITY_LEVELS[verbosity]

    with _state_lock:
        if verbosity:
            formatter = logging.Formatter(fmt='{asctime}.{msecs:03.0f}:{levelname}:{threadName}: {message}',
                                          datefmt='%H:%M:%S',
                                          style='{')
        else:
            formatter = logging.Formatter(fmt='%(message)s')

        if _listener is not None:
            # Keep the queue handler attached while the old listener drains. Records
            # produced during reconfiguration stay in the same queue and are picked
            # up by the replacement listener after its sentinel.
            _listener.stop()
            _queue_handler.setLevel(level)
            _stream_handler.setLevel(level)
            _stream_handler.setFormatter(formatter)
            if stream is not None:
                _stream_handler.setStream(stream)
            listener = QueueListener(_queue_handler.queue, _stream_handler, respect_handler_level=True)
            _listener = listener
            logger.setLevel(level)
            listener.start()
            threading.current_thread().name = 'main'
            return logger

        log_queue = queue.Queue()
        queue_handler = _LocalQueueHandler(log_queue)
        queue_handler.setLevel(level)

        stream_handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)

        listener = QueueListener(log_queue, stream_handler, respect_handler_level=True)
        logger.setLevel(level)
        logger.propagate = False
        logger.addHandler(queue_handler)

        _queue_handler = queue_handler
        _stream_handler = stream_handler
        _listener = listener
        listener.start()

        threading.current_thread().name = 'main'

        if not _atexit_registered:
            atexit.register(shutdown_logging)
            _atexit_registered = True

    return logger


def shutdown_logging():
    """Flush pending records and stop the application's listener, if any."""
    with _state_lock:
        _shutdown_logging_locked()


def verbosity_from_argv(argv) -> int:
    """Read ``-v``/``--verbose`` verbosity without consuming other options."""
    verbosity = 0
    index = 0
    argv = list(argv)
    while index < len(argv):
        arg = argv[index]
        if arg in ('-v', '--verbose'):
            if index + 1 < len(argv) and re.fullmatch(r'\d+', argv[index + 1]):
                verbosity = max(verbosity, int(argv[index + 1]))
                index += 2
                continue
            verbosity += 1
        elif re.fullmatch(r'-v{2,}', arg):
            verbosity += len(arg) - 1
        elif arg.startswith('--verbose=') and re.fullmatch(r'\d+', arg.partition('=')[2]):
            verbosity = max(verbosity, int(arg.partition('=')[2]))
        index += 1
    return _normalize_verbosity(verbosity)


def normalize_verbosity_args(argv):
    """Convert legacy numeric verbosity forms to counter-style arguments."""
    normalized = []
    index = 0
    argv = list(argv)
    while index < len(argv):
        arg = argv[index]
        if arg in ('-v', '--verbose') and index + 1 < len(argv) and re.fullmatch(r'\d+', argv[index + 1]):
            normalized.extend(['--verbose'] * _normalize_verbosity(int(argv[index + 1])))
            index += 2
            continue
        if arg.startswith('--verbose=') and re.fullmatch(r'\d+', arg.partition('=')[2]):
            normalized.extend(['--verbose'] * _normalize_verbosity(int(arg.partition('=')[2])))
            index += 1
            continue
        normalized.append(arg)
        index += 1
    return normalized


def redact_url(value: str) -> str:
    """Remove credentials and signed query data from a URL used in diagnostics."""
    try:
        parsed = urlsplit(str(value))
    except ValueError:
        return '<redacted-url>'
    if not parsed.scheme:
        return str(value)
    if not parsed.netloc:
        if parsed.scheme.lower() in {'data', 'mailto', 'tel'}:
            return f'{parsed.scheme}:<redacted>'
        return str(value)

    hostname = parsed.hostname
    if not hostname:
        return '<redacted-url>'
    if ':' in hostname and not hostname.startswith('['):
        hostname = f'[{hostname}]'
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f'{hostname}:{port}' if port is not None else hostname
    if parsed.username is not None or parsed.password is not None:
        netloc = f'<redacted>@{netloc}'

    query = '<redacted>' if parsed.query else ''
    fragment = '<redacted>' if parsed.fragment else ''
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))


def redact_command(argv):
    """Return a display-safe command without changing the command that will run."""
    redacted = []
    redact_next = False
    for arg in map(str, argv):
        if redact_next:
            redacted.append('<redacted>')
            redact_next = False
            continue

        option, separator, value = arg.partition('=')
        if option.startswith('-') and _SENSITIVE_OPTION_RE.search(option):
            if separator:
                redacted.append(f'{option}=<redacted>')
            else:
                redacted.append(option)
                redact_next = True
            continue

        redacted.append(redact_url(arg))
    return redacted
