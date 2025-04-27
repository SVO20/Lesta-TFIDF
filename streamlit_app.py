"""
Streamlit application.

Notice repetitive checks around *st.session_state*.
That is on purpose: Streamlit reruns the whole file on every user
interaction, so we safeguard one-off initialisation manually.

Run as `streamlit run streamlit_app.py`
Running at `http://localhost:8501`

"""

import pandas as pd
import streamlit as st

from database import setup_database, Corpus
from logger import info
from nlp import NlpDocContext, hash_original_text, tokenize, compute_count_tf, compress_original_text

# --------------------------------------------------------------------------------------
# 0. Boostrapping: choose or (re)create the database -----------------------------------
# --------------------------------------------------------------------------------------

if 'db_ready' not in st.session_state:
    st.title("TD-IFD Analyser")
    choice = st.radio("База данных:", ("Использовать существующую базу",
                                       "Создать новую базу"),
                      index=1)
    if st.button("Подтвердить выбор базы данных"):
        st.session_state.db_ready = True
        st.session_state.use_existing_db = (choice == "Использовать существующую базу")
        st.session_state.hashmap = {}  # {xxhash64: doc_id}
        st.rerun()
    else:
        st.warning("Надо выбрать пункт")
        st.stop()

# --------------------------------------------------------------------------------------
# 1. One-time init – keep objects alive in session_state -------------------------------
# --------------------------------------------------------------------------------------

if 'engine' not in st.session_state:
    # Create or open SQLite file depending on previous choice
    st.session_state.engine = setup_database(use_existing=st.session_state.use_existing_db)
engine = st.session_state.engine

if 'corpus' not in st.session_state:
    st.session_state.corpus = Corpus(engine)
corpus = st.session_state.corpus

if 'hashmap' not in st.session_state:
    # Speed-up duplicate detection: {xxhash64: doc_id}
    st.session_state.hashmap = corpus.get_hashmap()
hashmap = st.session_state.hashmap

# Streamlit re-uses widget keys between reruns, so we bump a counter to
# force the uploader to be recreated – otherwise stale files linger.
if 'uploader_round' not in st.session_state:
    st.session_state.uploader_round = 0  # int → will be unique key suffix

if 'uploaded_files_info' not in st.session_state:
    # Cache meta about every successfully processed file
    st.session_state.uploaded_files_info = []  # [{filename, doc_id, hash}]
    st.session_state.last_uploaded = None  # (filename, doc_id)

# --------------------------------------------------------------------------------------
# 2. File upload form ------------------------------------------------------------------
# --------------------------------------------------------------------------------------

uploader_key = f"file_uploader_{st.session_state.uploader_round}"  # fresh each round

with st.form(key="upload_form"):
    st.title("Добавление документа для анализа")
    files = st.file_uploader(label="Загрузите один или несколько текстовых файлов (.txt)",
                             type=["txt"],
                             accept_multiple_files=True,
                             key=uploader_key)  # forces clean widget after rerun
    submit = st.form_submit_button("Submit")

# --------------------------------------------------------------------------------------
# 3. Recieve uploaded files ------------------------------------------------------------
# --------------------------------------------------------------------------------------

if submit and files:
    documents_loaded: list[int] = []
    last_filename = ""
    last_doc_id: int | None = None

    for f in files:
        file_contents = f.read().decode("utf-8")
        if not file_contents:
            continue  # empty file – skip silently

        # -------- NLP pipeline (single document) --------------------------------------

        doc_cxt = NlpDocContext(file_contents)
        try:

            xxhash64 = hash_original_text(doc_cxt)
            if xxhash64 in hashmap:
                # Exact duplicate – no need to parse once more
                doc_id = hashmap[xxhash64]
                st.info(f"File {f.name} has same content as existing document doc_id={doc_id}.")
                documents_loaded.append(doc_id)
                last_filename, last_doc_id = f.name, doc_id
                continue

            tokens = tokenize(doc_cxt)
            if not tokens:
                st.warning(f"File {f.name} has no valid tokens. To drop.")
                continue

            lemmas_count_map, lemmas_tf_map = compute_count_tf(doc_cxt)
            if lemmas_count_map is None or lemmas_tf_map is None:
                st.error(f"File {f.name} gives incorrect results during analysis. To drop.")
                continue

            compress_original_text(doc_cxt)

            # -------- Commit document to database ------------------------------------

            doc_id = corpus.add_document(doc_cxt)
            hashmap[xxhash64] = doc_id  # keep hashmap in sync
            info(f"File {f.name} prepared and added to database with doc_id={doc_id}")

        except Exception:
            # Any unforeseen error – let Streamlit show the traceback
            raise
        finally:
            # Explicitly free heavy objects (NLP is memory hungry)
            doc_cxt.clear()

        last_filename, last_doc_id = f.name, doc_id

        # -------- NLP pipeline END ---------------------------------------------------

    if last_doc_id is None:
        st.error("last_doc_id is None ??!")  # should never happen
        st.stop()
    else:
        st.session_state.last_uploaded = (last_filename, last_doc_id)

    # Prepare uploader for next round – new key, clean state
    st.session_state.uploader_round += 1
    st.rerun()  # full-script rerun

# --------------------------------------------------------------------------------------
# 4. Display TF-IDF table for the last processed document ------------------------------
# --------------------------------------------------------------------------------------

if st.session_state.last_uploaded:
    st.info(f"Последний выбранный файл был: **{st.session_state.last_uploaded[0]}**\n\n"
            f"Ниже представлен анализ его содержания.\n\n\n\n"
            f"Для анализа другого файла выберите его в диалоге выше, он будет либо загружен, либо взят из базы данных.")

    last_filename, last_doc_id = st.session_state.last_uploaded

    lemmas_info = corpus.document_lemmas_info(last_doc_id)
    df = pd.DataFrame(lemmas_info)

    # -- interactive helper: sort order ----------------------------------------------
    sort_field = st.selectbox("Выберите поле для сортировки:",
                              options=["tf-idf", "count", "tf", "idf"],
                              index=0)  # default: tf-idf
    ascending = st.checkbox("Сортировать по возрастанию значения?", value=False)
    df = df.sort_values(by=sort_field, ascending=ascending)

    # Friendly 1-based index – looks nicer in a human table
    df.index = df.index + 1
    # Streamlit to render HTML <table>
    st.table(df.head(50))
