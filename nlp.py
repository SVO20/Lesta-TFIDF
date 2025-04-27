"""
NLP toolkit.

- Cyrillic tokeniser + stop-list
- Lemmas via pymystem3
- xxhash64 for duplicate detection
- LZMA compression for raw text

"""

import lzma
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import xxhash
from pymystem3 import Mystem

from config import STOP_WORDS_PATH

# ---------------------------------------------------------------------------
# 0. One-time resources
# ---------------------------------------------------------------------------

with open(STOP_WORDS_PATH, encoding="utf-8") as f:
    stop_words_set = {w.strip() for w in f if w.strip()}

re_russian_word = re.compile(r"\b[а-яА-ЯёЁ]+\b")
mystem = Mystem()


# ---------------------------------------------------------------------------
# 1. Pipeline context
# ---------------------------------------------------------------------------


@dataclass
class NlpDocContext:
    """Mutable carrier for every artefact produced during parsing."""

    text_input: str
    xxhash64: Optional[int] = None
    compressed_text: Optional[bytes] = None
    tokens_lemmatized: Optional[list[str]] = None
    lemmas_count_map: Optional[dict[str, int]] = None
    lemmas_tf_map: Optional[dict[str, float]] = None

    def clear(self) -> None:
        """Drop all data to help garbaje collector."""
        for field_name in self.__annotations__:
            object.__setattr__(self, field_name, None)

    def is_full(self) -> bool:
        """Checks that every field is set."""
        for field_name in self.__annotations__:
            if getattr(self, field_name) is None:
                return False
        return True


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------


def hash_original_text(nlp: NlpDocContext) -> Optional[int]:
    """"
    Store 63-bit xxhash64 inside nlp.

    It is clear workaround of SQLite limitation on BigInteger.
    Applicable for text-corpus volume less than 100K documents.
    """
    nlp.xxhash64 = xxhash.xxh64(nlp.text_input.encode()).intdigest() & (2 ** 63 - 1)  # keep only lower 63 bits
    return nlp.xxhash64 or None


def tokenize(nlp: NlpDocContext, text: str = "") -> Optional[list[str]]:
    """Lemmatise text and filter stop-words."""
    if not nlp.text_input and not text:
        raise ValueError("Text must be presented in 'nlp.text_input' or 'text' argument.")
    if not text:
        text = nlp.text_input
    elif not nlp.text_input:
        nlp.text_input = text
    else:
        raise ValueError("The 'text' argument given, but 'nlp.text_input' is not empty.")

    tokens_iterator = re_russian_word.finditer(text)
    tokens = [match.group() for match in tokens_iterator
              if match.group().lower() not in stop_words_set]
    lemmas = mystem.lemmatize(" ".join(tokens))
    # filter lemmatized tokens to ensure 'words contained letters only'
    tokens_lemmatized = [t.strip() for t in lemmas if t.strip().isalpha()]

    nlp.tokens_lemmatized = tokens_lemmatized
    return tokens_lemmatized or None


def compute_count_tf(nlp: NlpDocContext) -> tuple[Optional[dict[str, int]], Optional[dict[str, float]]]:
    """Populate *count* and *TF* maps for the current document."""
    assert nlp.tokens_lemmatized, "Run tokenize() first."

    nlp.lemmas_count_map = Counter(nlp.tokens_lemmatized)  # dict {'token1': <token1_count>, ...}
    total = sum(nlp.lemmas_count_map.values())

    nlp.lemmas_tf_map = {word: count / total for word, count in nlp.lemmas_count_map.items()}
    return nlp.lemmas_count_map or None, nlp.lemmas_tf_map or None


def compress_original_text(nlp: NlpDocContext) -> Optional[bytes]:
    """LZMA-compress raw text."""
    nlp.compressed_text = lzma.compress(nlp.text_input.encode())
    return nlp.compressed_text or None
