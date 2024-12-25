from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import logging
import time

import numpy as np
from whisper.audio import SAMPLE_RATE


logger = logging.getLogger('main')


DDEBUG = logging.DEBUG - 1


class TranslationTask:
    class Phase(Enum):
        SLICED = 1
        TRANSCRIBED = 2
        TRANSLATED = 3

    def __init__(self, line_id: int, audio: np.array, time_range: tuple[float, float]):
        self.id = line_id
        self.audio = audio
        self.transcribed_text = None
        self.translated_text = None
        self.time_range = time_range
        self.start_time = None
        self.phase = TranslationTask.Phase.SLICED


def _auto_args(func, kwargs):
    names = func.__code__.co_varnames
    return {k: v for k, v in kwargs.items() if k in names}


class LoopWorkerBase(ABC):

    @abstractmethod
    def loop(self):
        pass

    @classmethod
    def work(cls, **kwargs):
        obj = cls(**_auto_args(cls.__init__, kwargs))
        obj.loop(**_auto_args(obj.loop, kwargs))


def sec2str(second: float):
    dt = datetime.utcfromtimestamp(second)
    result = dt.strftime('%H:%M:%S')
    result += ',' + str(int(second * 10 % 10))
    return result


class LogTime:
    def __init__(self, text, *args, level = logging.INFO):
        self.text = text
        self.args = args
        self.level = level

    def set_log (self, text, *args) -> None:
        self.text = text
        self.args = args

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000
        if self.text:
            logger.log(self.level, self.text + ', run in time %.2fms', *self.args, duration_ms)
        else:
            logger.log(self.level, self.text, *self.args)
