import logging
import re


logger = logging.getLogger('main')

# Common Whisper silence/noise hallucinations, mostly derived from video outro
# subtitles present in the training data.
HALLUCINATION_KEYWORDS = [
    'amara.org',
    'subtitles by the',
    'subtitles made by the',
    'subtitles created by the',
    'subtitles by',
    'please subscribe',
    'subscribe to my channel',
    'like and subscribe',
]

HALLUCINATION_SENTENCES = {
    'thanks for watching',
    'thanks for watching the video',
    'thank you for watching',
    'thank you for watching the video',
    'thank you for watching my video',
    'thank you for watching until the end',
    'thank you very much for watching',
    'see you next time',
    'see you in the next video',
    'see you in the next one',
    'the end',
}


def english_filter(text: str):
    text = re.sub(r'\[.+?\]', '', text)
    clean_text = text.strip(" .,!?!~\t\n\r").lower()

    if any(keyword in clean_text for keyword in HALLUCINATION_KEYWORDS):
        logger.warning('filter %s', text)
        return ''

    if clean_text in HALLUCINATION_SENTENCES or len(clean_text) < 3:
        logger.warning('filter %s', text)
        return ''

    return text
