"""Translation pipeline entry point, separate from scheduling and API providers."""

import logging
import threading

from .common import LoopWorkerBase
from .translation_scheduler import ParallelTranslationScheduler
from .translation_types import TranslationBackend, TranslationSettings


logger = logging.getLogger('main')


class TranslationPipeline(LoopWorkerBase):
    """Connect pipeline queues to a scheduler and an independent provider backend.

    The pipeline exposes only lifecycle operations. The scheduler owns logical
    state and ordering; each request worker owns one backend session. No object
    doubles as both the pipeline entry point and a provider request handler.
    """

    def __init__(self, backend: TranslationBackend, settings: TranslationSettings):
        self.output_finished = threading.Event()
        self.scheduler = ParallelTranslationScheduler(backend, settings,
                                                       output_finished=self.output_finished)
        if settings.use_translation_context and settings.workers > settings.history_size + 1:
            logger.warning(
                'Translation threads (%d) exceed reference context window + 1 (%d); '
                'some workers may wait for a newer translation context',
                settings.workers, settings.history_size + 1,
            )

    def loop(self, input_queue, output_queue):
        """Run the dispatcher on the caller's pipeline thread exactly once."""
        self.scheduler.loop(input_queue, output_queue)

    def stop(self):
        """Stop accepting/publishing work and wake the dispatcher immediately."""
        self.scheduler.stop()
