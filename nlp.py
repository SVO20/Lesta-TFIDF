import lzma
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import xxhash
from pymystem3 import Mystem

from config import STOP_WORDS_PATH


@dataclass
class NlpDocContext:
    text_input: str
    xxhash64: Optional[int] = None
    compressed_text: Optional[bytes] = None
    tokens_lemmatized: Optional[list] = None
    lemmas_count_map: Optional[dict[str, int]] = None
    lemmas_tf_map: Optional[dict[str, float]] = None

    def clear(self) -> None:
        """
        Forcefully clears all fields, setting them to None,
        even if it contradicts the expected data type of field,
        for the sake of memory efficiency.
        """
        for field_name in self.__annotations__:
            object.__setattr__(self, field_name, None)

    def is_full(self):
        for field_name in self.__annotations__:
            if getattr(self, field_name) is None:
                return False
        return True

# Russian stop-words preloaded from the SpaCy Large model
with open(STOP_WORDS_PATH, encoding="utf-8") as f:
    stop_words_set = set(line.strip() for line in f if line.strip())

mystem = Mystem()
re_russian_word = re.compile(r'\b[а-яА-ЯёЁ]+\b')


def tokenize(nlp: NlpDocContext, text: str = "") -> Optional[list]:
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
    return tokens_lemmatized if tokens_lemmatized else None


def compute_tf(nlp: NlpDocContext) -> Optional[dict[str, float]]:
    """ Compute TF (Term Frequency) for a single document represented as tokens list. """

    assert nlp.tokens_lemmatized, "'nlp.tokens_lemmatized' list should not be empty"

    nlp.lemmas_count_map = Counter(nlp.tokens_lemmatized)  # dict {'token1': <token1_count>, ...}
    total_words = sum(nlp.lemmas_count_map.values())

    nlp.lemmas_tf_map = {word: count / total_words for word, count in nlp.lemmas_count_map.items()}
    return nlp.lemmas_tf_map if nlp.lemmas_tf_map else None


def hash_original_text(nlp: NlpDocContext) -> Optional[int]:
    nlp.xxhash64 = xxhash.xxh64(nlp.text_input.encode('utf-8')).intdigest()
    return nlp.xxhash64 if nlp.xxhash64 else None


def compress_original_text(nlp: NlpDocContext) -> Optional[bytes]:
    nlp.compressed_text = lzma.compress(nlp.text_input.encode())
    return nlp.compressed_text if nlp.compressed_text else None
