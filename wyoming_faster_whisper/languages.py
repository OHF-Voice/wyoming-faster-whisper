"""Whisper's language codes, and normalizing a client's code onto them.

Shared by the two backends that run Whisper checkpoints (faster-whisper and
transformers), which both reject an unknown code by raising rather than falling
back. faster-whisper is a hard dependency, so importing it here is always safe --
unlike every other backend, which is an optional extra.
"""

import logging
from typing import Optional

import faster_whisper

from .const import base_language

_LOGGER = logging.getLogger(__name__)

WHISPER_LANGUAGES = frozenset(
    faster_whisper.tokenizer._LANGUAGE_CODES  # pylint: disable=protected-access
)


def whisper_language(language: Optional[str]) -> Optional[str]:
    """Normalize a client's language code to one Whisper knows, else None.

    Both Whisper backends raise for anything outside the code list --
    faster-whisper a ValueError from its Tokenizer, transformers a ValueError
    from set_prefix_tokens, deferred until generate() -- which fails the whole
    transcription.

    Home Assistant normally sends back one of the codes we advertised, but it
    falls through to the raw pipeline language when its matcher finds nothing
    (Irish and Cornish are Home Assistant languages Whisper has no token for) or
    when a pipeline has no stt_language stored, in which case a locale tag like
    "pt-BR" arrives verbatim. Strip the region if that helps; otherwise return
    None and let Whisper auto-detect, which is a far better outcome than an
    error.
    """
    if not language:
        return None

    code = language.lower()
    if code in WHISPER_LANGUAGES:
        return code

    base = base_language(code)
    if base in WHISPER_LANGUAGES:
        _LOGGER.debug("Language '%s' narrowed to '%s' for Whisper", language, base)
        return base

    _LOGGER.warning(
        "Language '%s' is not supported by Whisper; auto-detecting instead",
        language,
    )
    return None
