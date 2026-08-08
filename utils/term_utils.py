"""Utilities for combining user-maintained terminology lists."""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence


# Plain spaces are intentionally not separators because multi-word terms are valid.
# The Japanese middle dot (・) and slash are also intentionally excluded because they
# commonly occur inside names and titles.
_ITEM_SEPARATORS = frozenset(',，、;；|｜\n\r\t\u2028\u2029•‣⁃◦')
_ANNOTATION_BRACKET_PAIRS = {
    '(': ')',
    '（': '）',
    '[': ']',
    '［': '］',
    '{': '}',
    '｛': '｝',
    '【': '】',
    '〔': '〕',
}
_GROUPING_BRACKET_PAIRS = {
    **_ANNOTATION_BRACKET_PAIRS,
    '<': '>',
    '＜': '＞',
    '〈': '〉',
    '《': '》',
    '「': '」',
    '『': '』',
}
_ANNOTATION_OPEN_BY_CLOSE = {close: opening for opening, close in _ANNOTATION_BRACKET_PAIRS.items()}
_STRONG_MAPPING_SEPARATOR_RE = re.compile(r'\s*(?:-{1,2}>|={1,2}>|→|⇒|➜|➝|➞|➡|⟶|⟹|↦)\s*')
_WEAK_MAPPING_SEPARATOR_RE = re.compile(r'\s*(?:=|＝|:|：)\s*')
_WHITESPACE_RE = re.compile(r'\s+')
_BULLET_PREFIX_RE = re.compile(r'^(?:[-*]\s+)+')


def _clean_term(value: str) -> str:
    value = _BULLET_PREFIX_RE.sub('', value.strip())
    return _WHITESPACE_RE.sub(' ', value).strip()


def _dedupe_key(value: str) -> str:
    # Canonically equivalent Unicode spellings are duplicates, but compatibility
    # forms and letter case are retained because they can request different ASR
    # output spellings (for example, ACME and ＡＣＭＥ).
    return unicodedata.normalize('NFC', _clean_term(value))


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = _clean_term(value)
        key = _dedupe_key(value)
        if not value or not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _split_top_level_items(value: str | None) -> list[str]:
    """Split common list delimiters while preserving delimiters inside brackets."""
    if not value:
        return []

    items = []
    current = []
    closing_stack = []

    for character in value:
        if character in _GROUPING_BRACKET_PAIRS:
            closing_stack.append(_GROUPING_BRACKET_PAIRS[character])
            current.append(character)
            continue
        if closing_stack and character == closing_stack[-1]:
            closing_stack.pop()
            current.append(character)
            continue
        if character in _ITEM_SEPARATORS and not closing_stack:
            item = _clean_term(''.join(current))
            if item:
                items.append(item)
            current.clear()
            continue
        current.append(character)

    item = _clean_term(''.join(current))
    if item:
        items.append(item)
    return items


def split_term_list(value: str | None) -> list[str]:
    """Parse a terminology list and return unique terms in first-seen order."""
    return _ordered_unique(_split_top_level_items(value))


def _strip_bracketed_text(value: str) -> str:
    """Remove balanced trailing annotation groups without corrupting titles."""
    original = _clean_term(value)
    result = original

    while result and result[-1] in _ANNOTATION_OPEN_BY_CLOSE:
        opening_stack = []
        annotation_start = None
        for index in range(len(result) - 1, -1, -1):
            character = result[index]
            if character in _ANNOTATION_OPEN_BY_CLOSE:
                opening_stack.append(_ANNOTATION_OPEN_BY_CLOSE[character])
                continue
            if opening_stack and character == opening_stack[-1]:
                opening_stack.pop()
                if not opening_stack:
                    annotation_start = index
                    break
        if annotation_start is None:
            break
        prefix = _clean_term(result[:annotation_start])
        if not prefix:
            return original
        result = prefix

    return result


def extract_stream_keyword_terms(value: str | None) -> list[str]:
    """Extract keyword names while dropping their parenthesized descriptions."""
    return _ordered_unique(_strip_bracketed_text(item) for item in _split_top_level_items(value))


def extract_translation_source_terms(value: str | None) -> list[str]:
    """Extract source-side terms from a list of source-to-target mappings."""
    terms = []
    for item in _split_top_level_items(value):
        # Prefer an explicit arrow even if the source itself contains a colon or
        # equals sign, as in "Re:ゼロ -> Re:Zero".
        separator = _STRONG_MAPPING_SEPARATOR_RE.search(item)
        if separator is None:
            separator = _WEAK_MAPPING_SEPARATOR_RE.search(item)
        if separator is None:
            continue
        source = _clean_term(item[:separator.start()])
        target = _clean_term(item[separator.end():])
        if not source or not target:
            continue
        terms.append(source)
    return _ordered_unique(terms)


def build_modulate_custom_terms(base_terms: str | None,
                                term_translations: str | None,
                                stream_keywords: str | None,
                                preserve_terms: str | None) -> str:
    """Combine all Modulate term sources into one comma-delimited unique list."""
    terms = _ordered_unique([
        *split_term_list(base_terms),
        *extract_translation_source_terms(term_translations),
        *extract_stream_keyword_terms(stream_keywords),
        *split_term_list(preserve_terms),
    ])
    return ','.join(terms)


def _run(argv: Sequence[str]) -> int:
    args = list(argv)
    if len(args) != 4:
        raise SystemExit(
            'Usage: stream-translator-build-terms '
            '<base-terms> <term-translations> <stream-keywords> <preserve-terms>'
        )
    output = build_modulate_custom_terms(*args)
    stdout_buffer = getattr(sys.stdout, 'buffer', None)
    if stdout_buffer is not None:
        stdout_buffer.write(output.encode('utf-8'))
    else:
        sys.stdout.write(output)
    return 0


def main() -> int:
    """Console entry point installed by pip."""
    return _run(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
