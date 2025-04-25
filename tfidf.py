import re
from collections import Counter

import pandas as pd
from pymystem3 import Mystem

from config import STOP_WORDS_PATH

# Russian stop-words preloaded from the SpaCy Large model
with open(STOP_WORDS_PATH, encoding="utf-8") as f:
    stop_words_set = set(line.strip() for line in f if line.strip())

mystem = Mystem()
re_russian_word = re.compile(r'\b[а-яА-ЯёЁ]+\b')


def tokenize(text: str) -> list:
    tokens_iterator = re_russian_word.finditer(text)
    tokens = [match.group() for match in tokens_iterator
              if match.group().lower() not in stop_words_set]
    lemmatized_tokens = mystem.lemmatize(" ".join(tokens))
    # filter lemmatized tokens as 'words contained letters only'
    result_tokens = [t.strip() for t in lemmatized_tokens if t.strip().isalpha()]
    return result_tokens


def compute_tf(tokens: list[str], top_n: int = 50) -> pd.DataFrame:
    """
    Compute TF (Term Frequency) for a single document represented as tokens list.
    Returns a DataFrame with top_n rows sorted by descending TF.

    """
    assert tokens, "tokens list should not be empty"

    terms_counts = Counter(tokens) # dict {'term1': <term1_count>, ...}
    total_words = sum(terms_counts.values())

    # Convert the count dict to DataFrame
    df = pd.DataFrame([(word, count / total_words) for word, count in terms_counts.items()],
                      columns=['word', 'tf'])
    df_sorted = df.sort_values(by="tf", ascending=False).head(top_n).reset_index(drop=True)
    return df_sorted
