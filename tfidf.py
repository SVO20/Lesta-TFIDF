import re
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

