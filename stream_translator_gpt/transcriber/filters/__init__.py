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


def get_language_filters(languages: str | None = None) -> list:
    """Resolve an explicit comma-separated language list to transcription filters."""
    selected_filters = []
    unknown_languages = set()

    for language in (languages or '').split(','):
        language = language.strip()
        if not language:
            continue

        lang_key = language.casefold().replace('_', '-')
        language_filter_functions = LANGUAGE_FILTERS.get(lang_key)
        if language_filter_functions is None:
            if lang_key not in unknown_languages:
                logger.warning('Unknown language filter %r, skipping.', language)
                unknown_languages.add(lang_key)
            continue

        for filter_function in language_filter_functions:
            if filter_function not in selected_filters:
                selected_filters.append(filter_function)

    logger.debug('Selected language-based transcription filters: %s',
                 ', '.join(filter_function.__name__ for filter_function in selected_filters) or 'none')
    return selected_filters
