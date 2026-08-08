import logging

from .common import emoji_filter, symbol_filter, repetition_filter
from .chinese import chinese_filter
from .english import english_filter
from .japanese import japanese_filter


logger = logging.getLogger('main')


LANGUAGE_FILTERS = {
    'en': [english_filter],
    'english': [english_filter],
    'ja': [japanese_filter],
    'japanese': [japanese_filter],
    'zh': [chinese_filter],
    'chinese': [chinese_filter],
    'zh-cn': [chinese_filter],
    'zh-tw': [chinese_filter],
}


def get_language_filters(language: str = None) -> list:
    if not language or language.lower() in ('auto', 'none'):
        all_filters = []
        for flist in LANGUAGE_FILTERS.values():
            for f in flist:
                if f not in all_filters:
                    all_filters.append(f)
        logger.debug('Selected language-based transcription filters: %s',
                     ', '.join(f.__name__ for f in all_filters) or 'none')
        return all_filters

    lang_key = language.lower()
    selected_filters = LANGUAGE_FILTERS.get(lang_key, [])
    logger.debug('Selected language-based transcription filters for %s: %s', language,
                 ', '.join(f.__name__ for f in selected_filters) or 'none')
    return selected_filters
